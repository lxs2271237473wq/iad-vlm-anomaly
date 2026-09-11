"""Normal-conditioned scale residual: exploratory source-only hypothesis.

Fit log(high-resolution map) conditional on log(low-resolution map), using
normal calibration images only. Separate normal images calibrate image scores.
This is a research probe, not a claim of a novel or validated method.
"""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from common import robust_fit,robust_apply,sha256

def maps(low,high,image_id):
    a=np.load(low/'maps'/f'{image_id}.npz')['anomaly_map']
    b=np.load(high/'maps'/f'{image_id}.npz')['anomaly_map']
    a=F.interpolate(torch.from_numpy(a)[None,None],size=b.shape,mode='bilinear',align_corners=False)[0,0].numpy()
    return np.log(np.maximum(a,1e-6)),np.log(np.maximum(b,1e-6))

def fit_relation(xs,ys,bins=20):
    x=np.concatenate(xs);y=np.concatenate(ys)
    edges=np.unique(np.quantile(x,np.linspace(0,1,bins+1)[1:-1]))
    groups=np.searchsorted(edges,x,side='right')
    global_scale=max(float(np.std(y)),1e-6)
    centers=[];scales=[]
    for i in range(len(edges)+1):
        values=y[groups==i]
        center=float(np.median(values)) if len(values) else float(np.median(y))
        scale=float(1.4826*np.median(np.abs(values-center))) if len(values) else global_scale
        centers.append(center);scales.append(max(scale,.1*global_scale,1e-6))
    return dict(edges=edges.tolist(),centers=centers,scales=scales)

def residual(x,y,model):
    indices=np.searchsorted(model['edges'],x,side='right')
    return (y-np.asarray(model['centers'])[indices])/np.asarray(model['scales'])[indices]

def aggregate(a):
    a=a.ravel();k=max(1,int(np.ceil(len(a)*.01)))
    return float(np.partition(a,-k)[-k:].mean())

def run(low,high,out):
    out.mkdir(parents=True,exist_ok=False)
    lo=pd.read_csv(low/'predictions.csv');hi=pd.read_csv(high/'predictions.csv')
    assert set(lo.image_id)==set(hi.image_id)
    assert (hi.dataset=='VisA').all() and 'label' not in hi
    normals=sorted(hi.loc[hi.split=='calibration','image_id'])
    fit_ids=normals[::2];scale_ids=normals[1::2]
    assert len(fit_ids)>=2 and len(scale_ids)>=2 and not set(fit_ids)&set(scale_ids)
    rng=np.random.default_rng(42);xs=[];ys=[]
    for image_id in fit_ids:
        x,y=maps(low,high,image_id)
        idx=rng.choice(x.size,size=min(x.size,4096),replace=False)
        xs.append(x.ravel()[idx]);ys.append(y.ravel()[idx])
    relation=fit_relation(xs,ys)
    # Unconditional high-scale standardization isolates benefit of conditioning.
    global_model=fit_relation([np.zeros_like(x) for x in xs],ys,bins=1)
    records=[]
    for row in hi.itertuples():
        x,y=maps(low,high,row.image_id)
        r=residual(x,y,relation)
        unconditional=(y-global_model['centers'][0])/global_model['scales'][0]
        records.append(dict(image_id=row.image_id,category=row.category,split=row.split,
            conditional_top1pct=aggregate(r),unconditional_top1pct=aggregate(unconditional),
            high_map_top1pct=aggregate(y)))
    frame=pd.DataFrame(records)
    frame=frame.merge(lo[['image_id','D_z']],on='image_id',validate='one_to_one').rename(columns={'D_z':'D256_z'})
    frame=frame.merge(hi[['image_id','D_z']],on='image_id',validate='one_to_one').rename(columns={'D_z':'D512_z'})
    params={}
    for field in ['conditional_top1pct','unconditional_top1pct','high_map_top1pct']:
        params[field]=robust_fit(frame.loc[frame.image_id.isin(scale_ids),field])
        frame[field+'_z']=robust_apply(frame[field],params[field])
    frame['fixed_multiscale']=.5*(frame.D256_z+frame.D512_z)
    frame.to_csv(out/'scores.csv',index=False)
    (out/'config.json').write_text(json.dumps(dict(relation=relation,normal_score_calibration=params,
        fit_image_ids=fit_ids,score_calibration_image_ids=scale_ids,bins=20,pixels_per_fit_image=4096,
        aggregation='top 1 percent mean',labels_used=False,code_sha256=sha256(Path(__file__)),
        hypothesis='Fine-scale evidence that is unexpected given coarse-scale normal response.'),indent=2))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--low',type=Path,required=True);ap.add_argument('--high',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args()
    expected=sorted(p.name for p in (args.low/'VisA').iterdir() if (p/'complete.json').exists())
    assert len(expected)==12
    for category in expected:
        high=args.high/'VisA'/category
        assert json.loads((high/'complete.json').read_text())['status']=='baseline_complete'
        target=args.out/category
        if not (target/'config.json').exists():run(args.low/'VisA'/category,high,target)
        print('RESIDUAL_COMPLETE',category,flush=True)
    pd.concat([pd.read_csv(args.out/c/'scores.csv') for c in expected]).to_csv(args.out/'all_scores.csv',index=False)
    (args.out/'complete.json').write_text(json.dumps(dict(status='complete',categories=expected,target_evaluated=False)))

if __name__=='__main__':main()
