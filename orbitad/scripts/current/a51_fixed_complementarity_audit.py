"""Label-free fixed fusion audit of OrbitAD and SuperAD-Reg4 public maps.

All formulas are declared here before evaluation. Per-image robust scaling uses no
labels or masks. This is an exploratory public-development audit; a winning rule
must transfer unchanged before it can replace the frozen primary method.
"""
import csv
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import tifffile
from PIL import Image

ROOT = Path('/root/private_data/iad-vlm-anomaly')
DATA = ROOT/'datasets/MVTec_AD_2'
A10 = ROOT/'orbitad/results/a10_multilayer_v1'
A16 = ROOT/'orbitad/results/a16_normal_only_component_soft_router_v1'
A21 = ROOT/'orbitad/results/a21_layerwise_evidence_v1'
A28 = ROOT/'orbitad/results/a28_depth_resolution_factorial_v1'
SUPER = ROOT/'ad2_model_zoo/results/a45_superad_reg4_public_v1'
OUT = ROOT/'orbitad/results/a51_fixed_complementarity_audit_v1'
METHODS = ('orbitad','superad_reg4','orbit_robust','super_robust','mean_robust','max_robust','geomean_robust','min_robust')
BANDS = ('all','tiny_le_0.1pct','small_0.1_to_1pct','large_gt_1pct')
BINS=65536; MAX_FPR=.05

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    rows=list(csv.DictReader(open(A10/'public_meta.csv')))
    a16_scales=json.loads((A16/'frozen_scales.json').read_text())
    thresholds=json.loads((A16/'route_thresholds.json').read_text())
    layer_scales=json.loads((A28/'frozen_layer_scales.json').read_text())
    protocol={
        'role':'exploratory public-development audit; transfer required before promotion',
        'formulas_declared_before_metric_read':list(METHODS),
        'robust_scaling':'per-image clip((x-q01)/(q995-q01),0,1); label-free',
        'fusion':'fixed equal arithmetic/max/geometric/min; no fitted weights',
        'metric':'native category-condition macro AU-PRO@FPR<=0.05',
    }
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2))

    def base(row,branch,depth=None):
        i,cat=int(row['map_index']),row['category']
        if depth is None:
            x=np.load(A10/branch/cat/'public'/f'{i:05d}.npy'); scale=a16_scales[cat][branch]
        else:
            x=np.load(A21/branch/f'layer{depth}'/cat/'public'/f'{i:05d}.npy'); scale=layer_scales[cat][branch][str(depth)]
        return np.asarray(x,np.float32)/scale
    def alpha(x,t):
        binary=(x>t).astype(np.uint8); n,lab,stats,_=cv2.connectedComponentsWithStats(binary,8)
        out=np.zeros_like(x,np.float32)
        for k in range(1,n): out[lab==k]=np.exp(-(stats[k,cv2.CC_STAT_AREA]/binary.size)/.001)
        return out
    def orbit(row,size):
        cat=row['category']; f=base(row,'full512'); p=base(row,'patch640')
        fn=cv2.resize(f,size); pn=cv2.resize(p,size); gate=cv2.resize(alpha(p,thresholds[cat]),size)
        a=.5*(fn+pn); a=a+gate*(pn-a)
        lp=base(row,'patch640',8); lf=base(row,'full512',8)
        l8=.5*(lp+cv2.resize(lf,(lp.shape[1],lp.shape[0]))); l8=cv2.resize(l8,size)
        return .5*(a+l8)
    def supermap(row):
        rel=Path(row['image_path']); return np.asarray(tifffile.imread(SUPER/row['category']/'anomaly_maps/seed=0'/row['category']/'test'/rel.parent.name/f'{rel.stem}.tiff'),np.float32)
    def robust(x):
        lo,hi=np.quantile(x,[.01,.995]); return np.clip((x-lo)/max(hi-lo,1e-12),0,1).astype(np.float32)
    def values(row):
        with Image.open(DATA/row['image_path']) as im: size=im.size
        o=orbit(row,size); s=supermap(row)
        if s.shape != o.shape: s=cv2.resize(s,size)
        on,sn=robust(o),robust(s)
        return {'orbitad':o,'superad_reg4':s,'orbit_robust':on,'super_robust':sn,
                'mean_robust':.5*(on+sn),'max_robust':np.maximum(on,sn),
                'geomean_robust':np.sqrt(np.maximum(on*sn,0)),'min_robust':np.minimum(on,sn)}
    maxima={m:0. for m in METHODS}
    for n,row in enumerate(rows,1):
        for m,x in values(row).items(): maxima[m]=max(maxima[m],float(np.max(x)))
        if n%100==0: print('maxima',n,len(rows),flush=True)
    edges={m:np.linspace(0,max(v*1.001,v+1e-6),BINS+1) for m,v in maxima.items()}
    class Stats:
        def __init__(self): self.neg=np.zeros(BINS,np.int64); self.region=np.zeros(BINS,np.float64); self.regions=0
    stats={}
    def get(m,c,d,b):
        k=m,c,d,b
        if k not in stats: stats[k]=Stats()
        return stats[k]
    for n,row in enumerate(rows,1):
        vals=values(row)
        if int(row['label'])==0:
            first=next(iter(vals.values())); mask=np.zeros(first.shape,np.uint8)
        else: mask=(cv2.imread(str(DATA/row['mask_path']),0)>0).astype(np.uint8)
        for m,x in vals.items():
            if x.shape!=mask.shape: x=cv2.resize(x,(mask.shape[1],mask.shape[0]))
            nh=np.histogram(x[mask==0],bins=edges[m])[0]
            for b in BANDS: get(m,row['category'],row['condition'],b).neg += nh
            count,lab=cv2.connectedComponents(mask,8)
            for rid in range(1,count):
                z=x[lab==rid]; ratio=len(z)/mask.size
                band='tiny_le_0.1pct' if ratio<=.001 else ('small_0.1_to_1pct' if ratio<=.01 else 'large_gt_1pct')
                h=np.histogram(z,bins=edges[m])[0]/len(z)
                for b in ('all',band): q=get(m,row['category'],row['condition'],b); q.region+=h; q.regions+=1
        if n%50==0: print('score',n,len(rows),flush=True)
    def aupro(q):
        if q.neg.sum()==0 or q.regions==0:return float('nan')
        x=np.r_[0.,np.cumsum(q.neg[::-1],dtype=np.float64)]/q.neg.sum(); y=np.r_[0.,np.cumsum(q.region[::-1])/q.regions]
        keep=x<=MAX_FPR; xx,yy=x[keep],y[keep]
        if xx[-1]<MAX_FPR:
            i=np.searchsorted(x,MAX_FPR,side='right'); w=(MAX_FPR-x[i-1])/max(x[i]-x[i-1],1e-12)
            xx=np.r_[xx,MAX_FPR]; yy=np.r_[yy,y[i-1]+w*(y[i]-y[i-1])]
        return float(np.trapezoid(yy,xx)/MAX_FPR)
    detail=[]
    for (m,c,d,b),q in sorted(stats.items()): detail.append({'method':m,'category':c,'condition':d,'size_band':b,'regions':q.regions,'aupro_0_05':aupro(q)})
    with (OUT/'native_category_condition_size.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(detail[0]));w.writeheader();w.writerows(detail)
    summary={}
    for m in METHODS:
        summary[m]={}
        for b in BANDS:
            z=[r for r in detail if r['method']==m and r['size_band']==b and np.isfinite(r['aupro_0_05'])]
            g=defaultdict(list)
            for r in z:g[r['category']].append(r['aupro_0_05'])
            summary[m][b]={'category_macro':float(np.mean([np.mean(v) for v in g.values()])), 'categories':len(g)}
    # Paired category bootstrap against the strongest constituent.
    comparisons={}
    by={(r['method'],r['category']):[] for r in detail if r['size_band']=='all'}
    for r in detail:
        if r['size_band']=='all':by[r['method'],r['category']].append(r['aupro_0_05'])
    rng=np.random.default_rng(20260917); cats=sorted({r['category'] for r in detail})
    for m in METHODS:
        if m=='superad_reg4':continue
        ds=np.array([np.mean(by[m,c])-np.mean(by['superad_reg4',c]) for c in cats])
        boot=np.mean(rng.choice(ds,(100000,len(ds)),replace=True),axis=1)
        comparisons[f'{m}_minus_superad_reg4']={'mean_delta':float(ds.mean()),'wins':int((ds>0).sum()),'bootstrap_95ci':[float(x) for x in np.quantile(boot,[.025,.975])], 'by_category':dict(zip(cats,map(float,ds)))}
    payload={'summary':summary,'comparisons':comparisons}
    (OUT/'summary.json').write_text(json.dumps(payload,indent=2));(OUT/'A51_COMPLETE.json').write_text(json.dumps({'status':'complete'},indent=2))
    print(json.dumps(payload,indent=2),flush=True)

if __name__=='__main__':main()
