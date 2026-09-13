"""Frozen three-seed source-pixel-supervised controls transferred to AD2."""
import json
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score,average_precision_score
from local_learning import RUN,fit_model,features as source_features
from local_learning_robustness import MASKS,SEEDS,convex_linear,dense_scores
from baseline_intervals import group_weights,matrix
from common import sha256

OUT=RUN/'20260911_l'

def extract(row,base):
    a=np.load(base/'maps'/f'{row.image_id}.npz')['anomaly_map'].astype(np.float32)
    p=json.loads((base/'normal_calibration.json').read_text())['D_raw'];scale=p['scale'];center=p['center']
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

def main():
    OUT.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4);cv2.setNumThreads(1)
    manifest=pd.read_csv(RUN/'20260910_a/manifest/images.csv');assert 'label' not in manifest
    # Exact source feature parity before any new model is fitted.
    row=next(manifest[(manifest.dataset=='VisA')&(manifest.split=='evaluation')].itertuples())
    np.testing.assert_array_equal(extract(row,RUN/'20260911_d/fixed_bank/VisA'/row.category),source_features(row))
    data=np.load(RUN/'20260911_j/training_samples.npz');x=data['x'];y=data['y']
    meta=pd.DataFrame(dict(category=data['category'],image_id=data['image_id'],label=y))
    target=manifest[(manifest.dataset=='AD2')&(manifest.split=='evaluation')];assert len(target)==1084
    assert not set(meta.image_id)&set(target.image_id)
    mean=x.mean(0);scale=np.maximum(x.std(0),1e-6);tx=(x-mean)/scale
    counts=meta.groupby(['category','label','image_id']).image_id.transform('size').to_numpy()
    images=meta.groupby(['category','label']).image_id.transform('nunique').to_numpy()
    w=1/(counts*images);w/=w.mean()
    linear,report=convex_linear(tx,y,w)
    models={'linear_convex':(linear,torch.ones(7,device='cuda'))}
    training={}
    for variant,mask in MASKS.items():
        for seed in SEEDS:
            name=f'{variant}_{seed}';model=fit_model(tx*mask,y,w,'mlp',seed)
            models[name]=(model,torch.tensor(mask,device='cuda'));training[name]=model.training_report
            torch.save(model.state_dict(),OUT/f'{name}.pt')
    torch.save(linear.state_dict(),OUT/'linear_convex.pt');np.savez(OUT/'scaler.npz',mean=mean,scale=scale)
    (OUT/'frozen_config.json').write_text(json.dumps(dict(seeds=SEEDS,source_feature_parity=True,source_categories=sorted(meta.category.unique()),
        code_sha256=sha256(Path(__file__)),sample_cache_sha256=sha256(RUN/'20260911_j/training_samples.npz'),
        training=training,linear_convergence=report,target_labels_used_for_fitting=False,
        protocol='All source categories fit final heads; target normal bank/calibration reused; no target fine-tuning or seed selection.'),indent=2))
    print('SOURCE_HEADS_FROZEN',flush=True)
    records=[]
    for category,g in target.groupby('category',sort=True):
        for row in g.sort_values('image_id').itertuples():
            scores=dense_scores(models,extract(row,RUN/'20260911_e/fixed_bank/AD2'/category),mean,scale)
            records.append(dict(image_id=row.image_id,category=category,**scores))
        pd.DataFrame(records).to_csv(OUT/'predictions.partial.csv',index=False)
        print('TARGET_COMPLETE',category,flush=True)
    predictions=pd.DataFrame(records);predictions.to_csv(OUT/'predictions_no_labels.csv',index=False)
    # Only now load target evaluation labels and existing reference scores.
    base=pd.read_csv(RUN/'20260911_e/evaluation/predictions_with_evaluation_labels.csv')
    f=predictions.merge(base[['image_id','label','scene_group','D512']],on='image_id',validate='one_to_one');assert len(f)==1084
    rows=[];methods=['D512']+list(models)
    for category,g in f.groupby('category'):
        for method in methods:rows.append(dict(category=category,method=method,auroc=roc_auc_score(g.label,g[method]),ap=average_precision_score(g.label,g[method])))
    table=pd.DataFrame(rows);table.to_csv(OUT/'category_metrics.csv',index=False)
    macro=table.groupby('method')[['auroc','ap']].mean();macro.to_csv(OUT/'macro_metrics.csv')
    rng=np.random.default_rng(42);cache=[]
    for _,g in f.groupby('category'):
        cache.append((g,group_weights(g.loc[g.label==1,'scene_group'],rng,5000),group_weights(g.loc[g.label==0,'scene_group'],rng,5000)))
    rows=[]
    for seed in SEEDS:
        for baseline in ['D512','linear_convex',f'map_{seed}']:
            method=f'full_{seed}';points=[];draws=[]
            for g,cp,cn in cache:
                delta=matrix(g,method)-matrix(g,baseline);points.append(float(delta.mean()))
                draws.append(np.einsum('bi,ij,bj->b',cp,delta,cn,optimize=True))
            d=np.stack(draws).mean(0)
            rows.append(dict(method=method,baseline=baseline,delta_macro_auroc=float(np.mean(points)),lower95=float(np.quantile(d,.025)),upper95=float(np.quantile(d,.975)),categories_improved=sum(v>0 for v in points)))
    pd.DataFrame(rows).to_csv(OUT/'paired_intervals.csv',index=False)
    summary=[]
    for variant in MASKS:
        values=macro.loc[[f'{variant}_{s}' for s in SEEDS],'auroc'];summary.append(dict(variant=variant,mean=float(values.mean()),std=float(values.std()),minimum=float(values.min()),maximum=float(values.max())))
    pd.DataFrame(summary).to_csv(OUT/'seed_summary.csv',index=False)
    (OUT/'complete.json').write_text(json.dumps(dict(status='complete',target_images=1084,target_categories=8,
        limitations='Source pixel supervision; target public data previously inspected; fixed backbone/sampling, varying heads only; provisional scene bootstrap; no multiple-test correction or seed selection.'),indent=2))
    print(macro.to_string());print(pd.DataFrame(rows).to_string(index=False))

if __name__=='__main__':main()
