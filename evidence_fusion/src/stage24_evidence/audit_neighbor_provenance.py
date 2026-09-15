"""Recover selected fixed-bank neighbor origins; no coreset rebuild or fitting."""
import json
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score
from anomalib.models.image.patchcore.torch_model import PatchcoreModel
from local_learning import RUN
from common import image_transform,sha256
from run_baselines import seed_all

OUT=RUN/'20260912_z'

@torch.inference_mode()
def main():
    OUT.mkdir(parents=True,exist_ok=False);seed_all(42);torch.set_num_threads(4)
    f=pd.read_csv(RUN/'20260911_k/oof_predictions.csv')
    ranks=f.groupby('category').apply(lambda g:roc_auc_score(g.label,g.D_z),include_groups=False).sort_values()
    ranks.to_csv(OUT/'source_category_auroc.csv');chosen=ranks.index[:2].tolist()
    manifest=pd.read_csv(RUN/'20260910_a/manifest/images.csv').set_index('image_id')
    selected=[]
    for category in chosen:
        g=f[f.category==category]
        for label,ascending in [(0,False),(1,True)]:
            row=g[g.label==label].sort_values(['D_z','image_id'],ascending=[ascending,True]).iloc[0]
            selected.append(dict(category=category,image_id=row.image_id,label=label,D_z=row.D_z))
    pd.DataFrame(selected).to_csv(OUT/'selected_cases.csv',index=False)
    (OUT/'protocol.json').write_text(json.dumps(dict(code_sha256=sha256(__file__),categories=chosen,selection='Two lowest source D AUROC categories; highest-scored normal and lowest-scored anomaly in each',diagnostic_only=True,no_training=True,no_target=True,provenance='Scan original used normal train IDs, batch4; approximate distance shortlist then direct Euclidean verification; accept relative embedding error <=1e-5',limitations='Posthoc extreme cases are not representative rates; raw maximum patch need not equal smoothed map peak; no proof of incorrect correspondence from dissimilarity alone.'),indent=2))
    transform=image_transform(512);records=[]
    for category in chosen:
        base=RUN/'20260911_d/fixed_bank/VisA'/category
        model=PatchcoreModel(layers=['layer2','layer3'],backbone='wide_resnet50_2',pre_trained=True,num_neighbors=9).cuda().eval()
        model.memory_bank=torch.load(base/'memory_bank.pt',map_location='cuda',weights_only=True)
        def images(ids):
            output=[]
            for image_id in ids:
                with Image.open(manifest.loc[image_id,'path']) as h:output.append(transform(h.convert('RGB')))
            return torch.stack(output).cuda()
        def embedding(batch):
            ft=model.feature_extractor(batch);ft={k:model.feature_pooler(v) for k,v in ft.items()}
            grid=model.generate_embedding(ft);return grid.permute(0,2,3,1).reshape(-1,grid.shape[1]),grid.shape[-2:]
        targets=[];cases=[]
        previous=pd.read_csv(base/'predictions.csv').set_index('image_id')
        for s in [v for v in selected if v['category']==category]:
            batch=images([s['image_id']]);output=model(batch)
            assert np.isclose(float(output.pred_score),previous.loc[s['image_id'],'D_raw'],rtol=1e-4,atol=1e-4)
            emb,shape=embedding(batch);distance,location=model.nearest_neighbors(emb,n_neighbors=1)
            peak=int(distance.argmax());bank_index=int(location[peak]);targets.append(model.memory_bank[bank_index])
            map_peak=int(output.anomaly_map[0,0].argmax())
            cases.append(dict(**s,query_path=manifest.loc[s['image_id'],'path'],bank_index=bank_index,raw_patch_y=peak//shape[1],raw_patch_x=peak%shape[1],patch_grid=list(shape),nearest_distance=float(distance[peak]),smoothed_peak_y=map_peak//512,smoothed_peak_x=map_peak%512))
        target=torch.stack(targets);best=[float('inf')]*len(target);origins=[None]*len(target)
        train_ids=json.loads((base/'used_image_ids.json').read_text())['train']
        assert all(manifest.loc[i,'split']=='train' and manifest.loc[i,'dataset']=='VisA' for i in train_ids)
        for start in range(0,len(train_ids),4):
            ids=train_ids[start:start+4];emb,shape=embedding(images(ids));npatch=shape[0]*shape[1]
            approx=(target.square().sum(1)[:,None]+emb.square().sum(1)[None,:]-2*target@emb.T)
            shortlist=approx.topk(min(32,len(emb)),largest=False,dim=1).indices
            for j in range(len(target)):
                exact=torch.linalg.vector_norm(emb[shortlist[j]]-target[j],dim=1);arg=int(exact.argmin());error=float(exact[arg]);index=int(shortlist[j,arg])
                if error<best[j]:
                    best[j]=error;image_id=ids[index//npatch];cell=index%npatch
                    origins[j]=dict(reference_image_id=image_id,reference_path=manifest.loc[image_id,'path'],reference_patch_y=cell//shape[1],reference_patch_x=cell%shape[1])
            if start%100==0:print('SCAN',category,start,len(train_ids),best,flush=True)
        for j,case in enumerate(cases):
            relative=best[j]/max(float(torch.linalg.vector_norm(target[j])),1e-12)
            records.append(dict(**case,**origins[j],embedding_error=best[j],relative_embedding_error=relative,provenance_verified=relative<=1e-5))
        (OUT/'provenance.partial.json').write_text(json.dumps(records,indent=2));print('COMPLETE',category,flush=True)
        del model;torch.cuda.empty_cache()
    (OUT/'provenance.json').write_text(json.dumps(records,indent=2))
    (OUT/'complete.json').write_text(json.dumps(dict(status='complete',cases=len(records),all_origins_verified=all(r['provenance_verified'] for r in records),score_parity=True,limitations='Recovered selected raw-peak nearest neighbors only; no universal geometry/correspondence correctness claim. Shortlist verification can fail and must be reported.'),indent=2))
    print(json.dumps(records,indent=2))

if __name__=='__main__':main()
