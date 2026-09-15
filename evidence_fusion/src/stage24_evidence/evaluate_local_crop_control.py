"""Ordinary local re-extraction pilot with frozen features and limited normal bank."""
import json,time
import numpy as np
import pandas as pd
import torch
from PIL import Image
from anomalib.models.image.patchcore.torch_model import PatchcoreModel
from local_learning import ROOT,RUN
from common import image_transform,sha256
from run_baselines import seed_all
from evaluate_exchange import intervals
from sklearn.metrics import roc_auc_score,average_precision_score

BASE=ROOT/'results/stage25_tiny_defects';OUT=BASE/'crop_control_v1'

@torch.inference_mode()
def main():
    OUT.mkdir(parents=True,exist_ok=False);seed_all(42);torch.set_num_threads(4)
    f=pd.read_csv(BASE/'v1/inference_manifest.csv');evals=f[f.role=='source_fit'];assert len(evals)==1462
    proposals=pd.read_csv(BASE/'candidate_v1/candidates_no_labels.csv')
    assert set(proposals.image_id)==set(evals.image_id)
    protocol=dict(code_sha256=sha256(__file__),source_images=1462,normal_reference_images_per_category=8,bank_size=8192,bank_sampling='Uniform random without replacement seed42; no coreset optimization',reference_crops='Eight fixed grid windows per normal image, each quarter width/height',input_sizes=dict(full=512,crop=256),aggregation='Max nearest-neighbor Euclidean patch distance, then max across 8 crops',methods=['mixed8','top8','grid8','full_limited_bank'],primary='mixed8 versus full_limited_bank and original D512',limitations='Limited-normal ordinary baseline, not novel model. Candidate maps cached; measured time excludes original proposal extraction. Normal full/crop bank extraction budgets differ, same final bank size. No supervised head or target data.')
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
    model=PatchcoreModel(layers=['layer2','layer3'],backbone='wide_resnet50_2',pre_trained=True,num_neighbors=9).cuda().eval()
    transforms={s:image_transform(s) for s in [256,512]}
    def embed(images,size):
        batch=torch.stack([transforms[size](im) for im in images]).cuda()
        ft=model.feature_extractor(batch);ft={k:model.feature_pooler(v) for k,v in ft.items()}
        grid=model.generate_embedding(ft)
        return grid.permute(0,2,3,1).reshape(len(images),-1,grid.shape[1])
    def bank(parts):
        all_=torch.cat(parts);assert len(all_)>=8192
        selected=torch.randperm(len(all_),generator=torch.Generator().manual_seed(42))[:8192]
        return all_[selected].cuda()
    def nearest_max(embedding,memory):
        model.memory_bank=memory
        return torch.stack([model.nearest_neighbors(v,n_neighbors=1)[0].max() for v in embedding]).cpu().numpy()
    records=[]
    for category,g in evals.groupby('category',sort=True):
        ref=f[(f.dataset=='VisA')&(f.category==category)&(f.original_split=='train')].sort_values('image_id').head(8)
        assert len(ref)==8 and not set(ref.image_id)&set(g.image_id)
        dest=OUT/category;dest.mkdir();ref.to_csv(dest/'reference_images.csv',index=False)
        full_parts=[];crop_parts=[]
        for r in ref.itertuples():
            with Image.open(r.path) as handle:im=handle.convert('RGB')
            full_parts.append(embed([im],512).flatten(0,1).cpu())
            # Same fixed grid8 layout as the candidate audit, mapped to native pixels.
            crops=[im.crop((round((x-64)*im.width/512),round((y-64)*im.height/512),round((x+64)*im.width/512),round((y+64)*im.height/512))) for y in [128,384] for x in [64,192,320,448]]
            for start in [0,4]:crop_parts.append(embed(crops[start:start+4],256).flatten(0,1).cpu())
        full_bank=bank(full_parts);crop_bank=bank(crop_parts);del full_parts,crop_parts
        torch.save(full_bank.cpu(),dest/'full_bank.pt');torch.save(crop_bank.cpu(),dest/'crop_bank.pt')
        # Self-neighbor zero-distance within numerical tolerance.
        model.memory_bank=full_bank
        assert float(model.nearest_neighbors(full_bank[:4],n_neighbors=1)[0].max())<.1
        print('BANKS_READY',category,flush=True)
        for r in g.sort_values('image_id').itertuples():
            started=time.perf_counter()
            with Image.open(r.path) as handle:im=handle.convert('RGB')
            full_score=float(nearest_max(embed([im],512),full_bank)[0])
            p=proposals[proposals.image_id==r.image_id]
            bounds=['native_x','native_y','native_right','native_bottom']
            unique=list(dict.fromkeys(tuple(int(v) for v in row) for row in p[bounds].to_numpy()))
            scores={}
            for start in range(0,len(unique),4):
                boxes=unique[start:start+4];values=nearest_max(embed([im.crop(b) for b in boxes],256),crop_bank)
                scores.update(zip(boxes,map(float,values)))
            record=dict(image_id=r.image_id,category=category,full_limited_bank=full_score)
            for method,mp in p.groupby('method'):
                assert len(mp)==8
                record[method]=max(scores[tuple(int(v) for v in row)] for row in mp[bounds].to_numpy())
            torch.cuda.synchronize();record['shared_comparison_seconds']=time.perf_counter()-started;records.append(record)
        pd.DataFrame(records).to_csv(OUT/'predictions.partial.csv',index=False)
        del full_bank,crop_bank;print('COMPLETE',category,flush=True)
    pred=pd.DataFrame(records);pred.to_csv(OUT/'predictions_no_labels.csv',index=False)
    labels=pd.read_csv(BASE/'v1/partition_manifest.csv')[['image_id','label','stratum']]
    previous=pd.read_csv(RUN/'20260911_k/oof_predictions.csv')[['image_id','D_z']]
    scored=pred.merge(labels,on='image_id',validate='one_to_one').merge(previous,on='image_id',validate='one_to_one')
    scored.to_csv(OUT/'evaluation_predictions.csv',index=False);methods=['mixed8','top8','grid8','full_limited_bank','D_z']
    rows=[];cis=[]
    for scope,data in [('all_source_fit',scored),('tiny_only_plus_all_normal',scored[(scored.label==0)|(scored.stratum=='tiny_only')])]:
        for category,g in data.groupby('category'):
            assert g.label.nunique()==2
            for method in methods:rows.append(dict(scope=scope,category=category,method=method,auroc=roc_auc_score(g.label,g[method]),ap=average_precision_score(g.label,g[method])))
        for ci in intervals(data,[('mixed8',b) for b in ['full_limited_bank','D_z','top8','grid8']],replicates=2000):cis.append(dict(scope=scope,**ci))
    table=pd.DataFrame(rows);table.to_csv(OUT/'category_metrics.csv',index=False)
    macro=table.groupby(['scope','method'])[['auroc','ap']].mean();macro.to_csv(OUT/'macro_metrics.csv');pd.DataFrame(cis).to_csv(OUT/'paired_intervals.csv',index=False)
    (OUT/'complete.json').write_text(json.dumps(dict(status='complete',images=len(scored),trained_head=False,limitations=protocol['limitations']),indent=2));print(macro.to_string())

if __name__=='__main__':main()
