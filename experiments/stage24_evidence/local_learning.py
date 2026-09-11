"""Source-mask-supervised local controls; outer category LOCO, no target data.

Frozen local statistics, identical supervision for linear and small MLP.
Masks choose training samples only. Dense inference has no mask argument.
"""
import gc,json
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score,average_precision_score
from common import sha256
from run_baselines import seed_all
from evaluate_exchange import intervals

ROOT=Path('/root/private_data/iad-vlm-anomaly');RUN=ROOT/'results/stage24_evidence'
OUT=RUN/'20260911_j'
NAMES=['response','center_surround','surround','response_std','intensity_mean','intensity_std','image_gradient']

def features(row):
    base=RUN/'20260911_d/fixed_bank/VisA'/row.category
    a=np.load(base/'maps'/f'{row.image_id}.npz')['anomaly_map'].astype(np.float32)
    p=json.loads((base/'normal_calibration.json').read_text())['D_raw']
    scale=p['scale'];center=p['center']
    small=cv2.GaussianBlur(a,(0,0),4);large=cv2.GaussianBlur(a,(0,0),16)
    std=np.sqrt(np.maximum(cv2.GaussianBlur(a*a,(0,0),16)-large*large,0))
    with Image.open(row.path) as handle:gray=np.asarray(handle.convert('L').resize((512,512),Image.Resampling.BILINEAR),dtype=np.float32)/255
    mean=cv2.GaussianBlur(gray,(0,0),16)
    texture=np.sqrt(np.maximum(cv2.GaussianBlur(gray*gray,(0,0),16)-mean*mean,0))
    gx=cv2.Sobel(gray,cv2.CV_32F,1,0);gy=cv2.Sobel(gray,cv2.CV_32F,0,1)
    gradient=cv2.GaussianBlur(np.sqrt(gx*gx+gy*gy),(0,0),4)
    x=np.stack([(a-center)/scale,(small-large)/scale,(large-center)/scale,std/scale,mean,texture,gradient],axis=-1)
    assert x.shape==(512,512,7) and np.isfinite(x).all()
    return x.reshape(-1,7)

def sample_indices(x,mask,rng):
    # Protect a boundary band when sampling non-defect pixels.
    positive=np.flatnonzero(mask.ravel())
    exclusion=cv2.dilate(mask.astype(np.uint8),np.ones((17,17),np.uint8))>0
    negatives=np.flatnonzero(~exclusion.ravel())
    if len(negatives)==0:raise ValueError('No valid negative pixels')
    ranked=negatives[np.argsort(x[negatives,0])]
    hard=ranked[-max(1,len(ranked)//100):]
    pick=lambda values,n:rng.choice(values,size=min(n,len(values)),replace=False)
    pos=pick(positive,32) if len(positive) else np.array([],dtype=int)
    neg=np.unique(np.concatenate([pick(hard,32),pick(negatives,32)]))
    return np.concatenate([pos,neg]),np.concatenate([np.ones(len(pos)),np.zeros(len(neg))])

def fit_model(x,y,w,kind,seed):
    seed_all(seed)
    model=(torch.nn.Linear(7,1) if kind=='linear' else torch.nn.Sequential(torch.nn.Linear(7,32),torch.nn.ReLU(),torch.nn.Linear(32,1))).cuda()
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001)
    x=torch.as_tensor(x,device='cuda');y=torch.as_tensor(y,dtype=torch.float32,device='cuda');w=torch.as_tensor(w,dtype=torch.float32,device='cuda')
    with torch.no_grad():initial=float((torch.nn.functional.binary_cross_entropy_with_logits(model(x).squeeze(1),y,reduction='none')*w).mean())
    for epoch in range(20):
        order=torch.randperm(len(x),device='cuda')
        for idx in order.split(4096):
            loss=(torch.nn.functional.binary_cross_entropy_with_logits(model(x[idx]).squeeze(1),y[idx],reduction='none')*w[idx]).mean()
            assert torch.isfinite(loss)
            optimizer.zero_grad();loss.backward();optimizer.step()
    with torch.no_grad():final=float((torch.nn.functional.binary_cross_entropy_with_logits(model(x).squeeze(1),y,reduction='none')*w).mean())
    model.training_report=dict(initial_loss=initial,final_loss=final,epochs=20,training_samples=len(x))
    return model.eval()

@torch.inference_mode()
def score(model,x,mean,scale):
    values=[]
    for start in range(0,len(x),32768):
        values.append(model(torch.as_tensor((x[start:start+32768]-mean)/scale,device='cuda')).squeeze(1).cpu().numpy())
    logits=np.concatenate(values)
    assert np.isfinite(logits).all()
    return float(logits.max())

def main():
    OUT.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4);cv2.setNumThreads(1)
    manifest=pd.read_csv(RUN/'20260910_a/manifest/images.csv')
    labels=pd.read_csv(RUN/'20260910_a/manifest/evaluation_labels.csv')
    source=manifest[(manifest.dataset=='VisA')&(manifest.split=='evaluation')].merge(labels,on='image_id',validate='one_to_one')
    assert len(source)==2162 and source.category.nunique()==12
    visa=pd.read_csv(ROOT/'datasets/VisA/split_csv/1cls.csv')
    masks={str(ROOT/'datasets/VisA'/r.image):str(ROOT/'datasets/VisA'/r.mask) for r in visa.itertuples() if isinstance(r.mask,str) and r.mask}
    cache=OUT/'training_samples.npz'
    signature=dict(code_sha256=sha256(Path(__file__)),manifest_sha256=sha256(RUN/'20260910_a/manifest/images.csv'),labels_sha256=sha256(RUN/'20260910_a/manifest/evaluation_labels.csv'))
    if cache.exists():assert json.loads((OUT/'cache_signature.json').read_text())==signature
    if not cache.exists():
        xs=[];ys=[];cats=[];ids=[];rng=np.random.default_rng(42)
        for category,g in source.groupby('category',sort=True):
            for r in g.sort_values('image_id').itertuples():
                x=features(r)
                mask=np.zeros((512,512),dtype=bool)
                if r.label==1:
                    with Image.open(masks[r.path]) as handle:mask=np.array(handle.convert('L').resize((512,512),Image.Resampling.NEAREST))>0
                    assert mask.any()
                idx,y=sample_indices(x,mask,rng)
                xs.append(x[idx]);ys.append(y);cats.extend([category]*len(idx));ids.extend([r.image_id]*len(idx))
            print('SAMPLED',category,flush=True)
        np.savez_compressed(cache,x=np.concatenate(xs).astype(np.float32),y=np.concatenate(ys).astype(np.float32),category=np.asarray(cats),image_id=np.asarray(ids))
        (OUT/'cache_signature.json').write_text(json.dumps(signature,indent=2))
    data=np.load(cache);x=data['x'];y=data['y'];cat=data['category'];ids=data['image_id']
    original=pd.read_csv(RUN/'20260911_i/oof_predictions.csv')[['image_id','D_z','linear_contrast']]
    frames=[]
    for heldout,g in source.groupby('category',sort=True):
        dest=OUT/heldout;dest.mkdir(exist_ok=True)
        if (dest/'complete.json').exists():
            assert json.loads((dest/'complete.json').read_text())['code_sha256']==sha256(Path(__file__))
            frames.append(pd.read_csv(dest/'predictions.csv'));continue
        train=cat!=heldout
        assert not set(ids[train])&set(g.image_id)
        mean=x[train].mean(0);scale=np.maximum(x[train].std(0),1e-6)
        tx=(x[train]-mean)/scale;ty=y[train]
        meta=pd.DataFrame(dict(category=cat[train],image_id=ids[train],label=ty))
        # Equal contribution per category and class, then per image with that class.
        pixel_counts=meta.groupby(['category','label','image_id']).image_id.transform('size').to_numpy()
        image_counts=meta.groupby(['category','label']).image_id.transform('nunique').to_numpy()
        weights=1/(pixel_counts*image_counts);weights/=weights.mean()
        models={kind:fit_model(tx,ty,weights,kind,42) for kind in ['linear','mlp']}
        np.savez(dest/'scaler.npz',mean=mean,scale=scale)
        for kind,model in models.items():torch.save(model.state_dict(),dest/f'{kind}.pt')
        records=[]
        # Inference features do not accept labels or masks.
        for r in g.drop(columns=['label','scene_group']).sort_values('image_id').itertuples():
            fx=features(r)
            records.append(dict(image_id=r.image_id,category=heldout,**{kind:score(model,fx,mean,scale) for kind,model in models.items()}))
        pred=pd.DataFrame(records);pred.to_csv(dest/'predictions.csv',index=False);frames.append(pred)
        (dest/'complete.json').write_text(json.dumps(dict(code_sha256=sha256(Path(__file__)),heldout=heldout,training_categories=sorted(set(cat[train])),epochs=20,seed=42,features=NAMES,training_reports={kind:model.training_report for kind,model in models.items()},supervision='source pixel masks; heldout category excluded from model fitting'),indent=2))
        del models;gc.collect();torch.cuda.empty_cache();print('FOLD_COMPLETE',heldout,flush=True)
    f=pd.concat(frames).merge(source[['image_id','label']],on='image_id',validate='one_to_one').merge(original,on='image_id',validate='one_to_one');assert len(f)==2162
    f.to_csv(OUT/'oof_predictions.csv',index=False)
    rows=[]
    for category,g in f.groupby('category'):
        for method in ['D_z','linear_contrast','linear','mlp']:rows.append(dict(category=category,method=method,auroc=roc_auc_score(g.label,g[method]),ap=average_precision_score(g.label,g[method])))
    table=pd.DataFrame(rows);table.to_csv(OUT/'category_metrics.csv',index=False)
    macro=table.groupby('method')[['auroc','ap']].mean();macro.to_csv(OUT/'macro_metrics.csv')
    ci=intervals(f,[('linear','D_z'),('mlp','D_z'),('mlp','linear'),('mlp','linear_contrast')],replicates=2000)
    pd.DataFrame(ci).to_csv(OUT/'paired_intervals.csv',index=False)
    (OUT/'complete.json').write_text(json.dumps(dict(status='complete',source_only=True,supervision='source pixel masks, stronger than prior image-label fusion',
        primary_comparison='MLP versus linear under identical pixel supervision',
        limitation='Fixed one-seed control, fixed max aggregation, conditional bootstrap. No target evaluation or novel-method claim.'),indent=2))
    print(macro.to_string());print(pd.DataFrame(ci).to_string(index=False))

if __name__=='__main__':main()
