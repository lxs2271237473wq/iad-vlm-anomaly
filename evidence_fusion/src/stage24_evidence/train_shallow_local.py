"""Actual source-supervised local deep-feature heads, frozen backbone, fixed validation."""
import json,gc
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn.functional as F
from PIL import Image
from anomalib.models.image.patchcore.torch_model import PatchcoreModel
from common import image_transform,sha256
from local_learning import ROOT,RUN
from run_baselines import seed_all
from audit_tiny_candidates import candidates,native
from evaluate_exchange import intervals
from sklearn.metrics import roc_auc_score,average_precision_score

BASE=ROOT/'results/stage25_tiny_defects';OUT=BASE/'shallow_local_v1'

def main():
    OUT.mkdir(parents=True,exist_ok=False);seed_all(42);torch.set_num_threads(4);cv2.setNumThreads(1)
    manifest=pd.read_csv(BASE/'v1/inference_manifest.csv');fit=manifest[manifest.role=='source_fit'];validation=manifest[manifest.role=='source_validation']
    assert len(fit)==1462 and len(validation)==300 and not set(fit.category)&set(validation.category)
    protocol=dict(code_sha256=sha256(__file__),candidate_code_sha256=sha256(__file__.replace('train_shallow_local.py','audit_tiny_candidates.py')),training_categories=sorted(fit.category.unique()),validation_categories=sorted(validation.category.unique()),features='Frozen WRN50_2 layer1/layer2, upsample layer2 to layer1, concatenate 768 channels at 64x64 from 256 crop',candidates='Unchanged mixed8 on 512 teacher map, native quarter-field crop',supervision='Source-fit real masks only; 16 positives,16 teacher-high negatives,16 random negatives per crop; 3x3 exclusion at feature grid',heads='Linear and 768-64-ReLU-1 MLP, same samples/standardizer/BCE budget',epochs=20,lr=.001,weight_decay=.001,seeds=[42,123,2026],primary='MLP versus linear and original D512 on source_validation',aggregation='Max patch logit over eight crops',limitations='Known supervised deep local baseline, not claimed novelty; additional real masks and compute; no AD2 or holdout fitting.')
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
    # Freeze all candidates from label-free manifest/maps before loading any masks.
    lookup={};candidate_records=[]
    for r in pd.concat([fit,validation]).sort_values('image_id').itertuples():
        a=np.load(RUN/'20260911_d/fixed_bank/VisA'/r.category/'maps'/f'{r.image_id}.npz')['anomaly_map'];boxes=candidates(a)['mixed8'];lookup[r.image_id]=boxes
        for i,b in enumerate(boxes):candidate_records.append(dict(image_id=r.image_id,role=r.role,index=i,box=list(b),native_box=list(native(b,r.width,r.height))))
    (OUT/'candidates_no_labels.json').write_text(json.dumps(candidate_records))
    print('CANDIDATES_FROZEN',flush=True)
    backbone=PatchcoreModel(layers=['layer1','layer2'],backbone='wide_resnet50_2',pre_trained=True,num_neighbors=9).cuda().eval();transform=image_transform(256)
    @torch.inference_mode()
    def features(crop):
        ft=backbone.feature_extractor(transform(crop)[None].cuda())
        grid=torch.cat([ft['layer1'],F.interpolate(ft['layer2'],size=ft['layer1'].shape[-2:],mode='bilinear',align_corners=False)],1)
        assert grid.shape==(1,768,64,64)
        return grid[0].permute(1,2,0).reshape(-1,768)
    labels=pd.read_csv(BASE/'v1/partition_manifest.csv').set_index('image_id');rng=np.random.default_rng(42)
    xs=[];ys=[];cs=[];image_ids=[];sample_counts=[]
    for category,g in fit.groupby('category',sort=True):
        for r in g.sort_values('image_id').itertuples():
            with Image.open(r.path) as h:im=h.convert('RGB')
            info=labels.loc[r.image_id]
            if info.label:
                with Image.open(info.mask_path) as h:mask=np.asarray(h.convert('L'))>0
            else:mask=np.zeros((r.height,r.width),bool)
            teacher=np.load(RUN/'20260911_d/fixed_bank/VisA'/category/'maps'/f'{r.image_id}.npz')['anomaly_map']
            for b in lookup[r.image_id]:
                x0,y0,x1,y1=native(b,r.width,r.height);fx=features(im.crop((x0,y0,x1,y1)))
                positive=cv2.resize(mask[y0:y1,x0:x1].astype(np.float32),(64,64),interpolation=cv2.INTER_AREA)>0
                excluded=cv2.dilate(positive.astype(np.uint8),np.ones((3,3),np.uint8))>0
                pos=np.flatnonzero(positive.ravel());neg=np.flatnonzero(~excluded.ravel())
                score=cv2.resize(teacher[b[1]:b[3],b[0]:b[2]],(64,64),interpolation=cv2.INTER_LINEAR).ravel()
                pick=lambda values,n:rng.choice(values,min(n,len(values)),replace=False)
                p=pick(pos,16);hard=neg[np.argsort(score[neg])[-min(128,len(neg)):]] if len(neg) else neg
                n=np.unique(np.r_[pick(hard,16),pick(neg,16)]);idx=np.r_[p,n].astype(int)
                if not len(idx):continue
                xs.append(fx[idx].cpu().numpy().astype(np.float16));ys.append(np.r_[np.ones(len(p)),np.zeros(len(n))].astype(np.float32));cs.extend([category]*len(idx));image_ids.extend([r.image_id]*len(idx))
                sample_counts.append(dict(image_id=r.image_id,positive=len(p),negative=len(n)))
        print('SAMPLED',category,flush=True)
    x=np.concatenate(xs).astype(np.float32);y=np.concatenate(ys);del xs,ys;gc.collect()
    np.savez_compressed(OUT/'training_samples.npz',x=x.astype(np.float16),y=y,category=np.asarray(cs),image_id=np.asarray(image_ids));pd.DataFrame(sample_counts).to_csv(OUT/'sample_counts.csv',index=False)
    mean=x.mean(0);scale=np.maximum(x.std(0),1e-6);np.savez(OUT/'scaler.npz',mean=mean,scale=scale)
    meta=pd.DataFrame(dict(category=cs,image_id=image_ids,label=y));counts=meta.groupby(['category','label','image_id']).image_id.transform('size').to_numpy();images=meta.groupby(['category','label']).image_id.transform('nunique').to_numpy();w=1/(counts*images);w/=w.mean()
    tx=torch.tensor((x-mean)/scale,device='cuda');ty=torch.tensor(y,device='cuda');tw=torch.tensor(w,dtype=torch.float32,device='cuda');del x;models={};reports=[]
    for kind in ['linear','mlp']:
        for seed in [42,123,2026]:
            seed_all(seed);model=(torch.nn.Linear(768,1) if kind=='linear' else torch.nn.Sequential(torch.nn.Linear(768,64),torch.nn.ReLU(),torch.nn.Linear(64,1))).cuda()
            optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001)
            for epoch in range(20):
                for idx in torch.randperm(len(tx),device='cuda').split(4096):
                    loss=(F.binary_cross_entropy_with_logits(model(tx[idx]).flatten(),ty[idx],reduction='none')*tw[idx]).mean();assert torch.isfinite(loss)
                    optimizer.zero_grad();loss.backward();optimizer.step()
            name=f'{kind}_{seed}';models[name]=model.eval();torch.save(model.state_dict(),OUT/f'{name}.pt');reports.append(dict(model=name,final_batch_loss=float(loss),epochs=20));print('TRAINED',name,flush=True)
    del tx,ty,tw;gc.collect();torch.cuda.empty_cache();(OUT/'training_complete.json').write_text(json.dumps(reports,indent=2))
    rows=[]
    with torch.inference_mode():
        for category,g in validation.groupby('category',sort=True):
            for r in g.sort_values('image_id').itertuples():
                with Image.open(r.path) as h:im=h.convert('RGB')
                best={name:-np.inf for name in models}
                for b in lookup[r.image_id]:
                    fx=features(im.crop(native(b,r.width,r.height)));z=(fx-torch.as_tensor(mean,device='cuda'))/torch.as_tensor(scale,device='cuda')
                    for name,model in models.items():best[name]=max(best[name],float(model(z).max()))
                assert np.isfinite(list(best.values())).all();rows.append(dict(image_id=r.image_id,category=category,**best))
            pd.DataFrame(rows).to_csv(OUT/'validation.partial.csv',index=False);print('VALIDATED',category,flush=True)
    f=pd.DataFrame(rows);f.to_csv(OUT/'validation_no_labels.csv',index=False)
    f=f.merge(labels[['label','stratum']],left_on='image_id',right_index=True,validate='one_to_one').merge(pd.read_csv(RUN/'20260911_k/oof_predictions.csv')[['image_id','D_z']],on='image_id',validate='one_to_one');f.to_csv(OUT/'evaluation_predictions.csv',index=False)
    metrics=[];cis=[]
    for scope,data in [('validation_all',f),('validation_tiny',f[(f.label==0)|(f.stratum=='tiny_only')])]:
        for category,g in data.groupby('category'):
            for name in ['D_z']+list(models):metrics.append(dict(scope=scope,category=category,method=name,auroc=roc_auc_score(g.label,g[name]),ap=average_precision_score(g.label,g[name])))
        for ci in intervals(data,[(f'mlp_{s}',b) for s in [42,123,2026] for b in ['D_z',f'linear_{s}']],replicates=2000):cis.append(dict(scope=scope,**ci))
    table=pd.DataFrame(metrics);table.to_csv(OUT/'category_metrics.csv',index=False);macro=table.groupby(['scope','method'])[['auroc','ap']].mean();macro.to_csv(OUT/'macro_metrics.csv');pd.DataFrame(cis).to_csv(OUT/'paired_intervals.csv',index=False)
    (OUT/'complete.json').write_text(json.dumps(dict(status='complete',training_images=1462,validation_images=300,seeds=[42,123,2026],target_used=False),indent=2));print(macro.to_string())

if __name__=='__main__':main()
