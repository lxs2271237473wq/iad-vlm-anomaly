"""Predeclared label-free component-scale routing between SuperAD-Reg4 and OrbitAD."""
import csv,json
from collections import defaultdict
from pathlib import Path
import cv2,numpy as np,tifffile
from PIL import Image

R=Path('/root/private_data/iad-vlm-anomaly'); D=R/'datasets/MVTec_AD_2'
A10=R/'orbitad/results/a10_multilayer_v1'; A16=R/'orbitad/results/a16_normal_only_component_soft_router_v1'
A21=R/'orbitad/results/a21_layerwise_evidence_v1'; A28=R/'orbitad/results/a28_depth_resolution_factorial_v1'
SUPER=R/'ad2_model_zoo/results/a45_superad_reg4_public_v1'; BASE=R/'orbitad/results/a47_superad_reg4_unified_eval_v1'
OUT=R/'orbitad/results/a53_scale_routed_complementarity_v1'
METHODS=('scale_route_orbit','scale_route_mean','scale_route_max'); BANDS=('all','tiny_le_0.1pct','small_0.1_to_1pct','large_gt_1pct'); N=65536

def main():
 OUT.mkdir(parents=True,exist_ok=True); rows=list(csv.DictReader(open(A10/'public_meta.csv')))
 scales=json.loads((A16/'frozen_scales.json').read_text()); th=json.loads((A16/'route_thresholds.json').read_text()); ls=json.loads((A28/'frozen_layer_scales.json').read_text())
 (OUT/'protocol.json').write_text(json.dumps({'status':'predeclared before A51 results','normalization':'per-image q01/q995, no label','candidate_mask':'union normalized score > 0.5','area_boundary':0.001,'orbit_weight':'1-exp(-component_area_ratio/0.001)','outside':'SuperAD-Reg4','metric':'native category-condition macro AU-PRO@0.05','promotion':'requires frozen transfer'},indent=2))
 def load(row,b,depth=None):
  i,c=int(row['map_index']),row['category']
  if depth is None:p=A10/b/c/'public'/f'{i:05d}.npy';s=scales[c][b]
  else:p=A21/b/f'layer{depth}'/c/'public'/f'{i:05d}.npy';s=ls[c][b][str(depth)]
  return np.asarray(np.load(p),np.float32)/s
 def gate(x,t):
  q=(x>t).astype(np.uint8);n,l,st,_=cv2.connectedComponentsWithStats(q,8);a=np.zeros_like(x,np.float32)
  for k in range(1,n):a[l==k]=np.exp(-(st[k,cv2.CC_STAT_AREA]/q.size)/.001)
  return a
 def qnorm(x):
  lo,hi=np.quantile(x,[.01,.995]);return np.clip((x-lo)/max(hi-lo,1e-12),0,1).astype(np.float32)
 def maps(row):
  with Image.open(D/row['image_path']) as im:size=im.size
  f,p=load(row,'full512'),load(row,'patch640');fn,pn=cv2.resize(f,size),cv2.resize(p,size)
  a=.5*(fn+pn);a+=cv2.resize(gate(p,th[row['category']]),size)*(pn-a)
  lp,lf=load(row,'patch640',8),load(row,'full512',8);mid=.5*(lp+cv2.resize(lf,(lp.shape[1],lp.shape[0])))
  on=qnorm(.5*(a+cv2.resize(mid,size)))
  rel=Path(row['image_path']);sn=qnorm(np.asarray(tifffile.imread(SUPER/row['category']/'anomaly_maps/seed=0'/row['category']/'test'/rel.parent.name/f'{rel.stem}.tiff'),np.float32))
  if sn.shape!=on.shape:sn=cv2.resize(sn,size)
  binary=(np.maximum(sn,on)>.5).astype(np.uint8);n,lab,st,_=cv2.connectedComponentsWithStats(binary,8)
  out={m:sn.copy() for m in METHODS}
  for k in range(1,n):
   pix=lab==k;w=1-np.exp(-(st[k,cv2.CC_STAT_AREA]/binary.size)/.001)
   out['scale_route_orbit'][pix]=(1-w)*sn[pix]+w*on[pix]
   out['scale_route_mean'][pix]=(1-w)*sn[pix]+w*.5*(sn[pix]+on[pix])
   out['scale_route_max'][pix]=(1-w)*sn[pix]+w*np.maximum(sn[pix],on[pix])
  return out
 edges=np.linspace(0,1.000001,N+1)
 class S:
  def __init__(self):self.neg=np.zeros(N,np.int64);self.reg=np.zeros(N,np.float64);self.n=0
 stats={}
 def get(m,c,d,b):
  k=m,c,d,b
  if k not in stats:stats[k]=S()
  return stats[k]
 for ix,row in enumerate(rows,1):
  vv=maps(row);first=next(iter(vv.values()))
  mask=np.zeros(first.shape,np.uint8) if int(row['label'])==0 else (cv2.imread(str(D/row['mask_path']),0)>0).astype(np.uint8)
  cc,labels=cv2.connectedComponents(mask,8)
  for m,x in vv.items():
   if x.shape!=mask.shape:x=cv2.resize(x,(mask.shape[1],mask.shape[0]))
   h=np.histogram(x[mask==0],bins=edges)[0]
   for b in BANDS:get(m,row['category'],row['condition'],b).neg+=h
   for rid in range(1,cc):
    z=x[labels==rid];r=len(z)/mask.size;b='tiny_le_0.1pct' if r<=.001 else ('small_0.1_to_1pct' if r<=.01 else 'large_gt_1pct');rh=np.histogram(z,bins=edges)[0]/len(z)
    for target in ('all',b):q=get(m,row['category'],row['condition'],target);q.reg+=rh;q.n+=1
  if ix%50==0:print('evaluated',ix,len(rows),flush=True)
 def pro(q):
  if not q.n:return float('nan')
  x=np.r_[0.,np.cumsum(q.neg[::-1],dtype=float)]/q.neg.sum();y=np.r_[0.,np.cumsum(q.reg[::-1])/q.n];k=x<=.05;xx,yy=x[k],y[k]
  if xx[-1]<.05:
   i=np.searchsorted(x,.05,side='right');w=(.05-x[i-1])/max(x[i]-x[i-1],1e-12);xx=np.r_[xx,.05];yy=np.r_[yy,y[i-1]+w*(y[i]-y[i-1])]
  return float(np.trapezoid(yy,xx)/.05)
 detail=[{'method':m,'category':c,'condition':d,'size_band':b,'regions':q.n,'aupro_0_05':pro(q)} for (m,c,d,b),q in sorted(stats.items())]
 with (OUT/'native_category_condition_size.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(detail[0]));w.writeheader();w.writerows(detail)
 base=list(csv.DictReader(open(BASE/'native_category_condition_size.csv')));cats=sorted({r['category'] for r in rows});rng=np.random.default_rng(20260917);payload={'summary':{},'comparisons':{}}
 for m in METHODS:
  payload['summary'][m]={}
  for b in BANDS:
   z=[r for r in detail if r['method']==m and r['size_band']==b and np.isfinite(r['aupro_0_05'])];g=defaultdict(list)
   for r in z:g[r['category']].append(r['aupro_0_05'])
   payload['summary'][m][b]={'category_macro':float(np.mean([np.mean(v) for v in g.values()]))}
  ds=[]
  for c in cats:
   x=np.mean([r['aupro_0_05'] for r in detail if r['method']==m and r['category']==c and r['size_band']=='all']);y=np.mean([float(r['aupro_0_05']) for r in base if r['category']==c and r['size_band']=='all']);ds.append(x-y)
  ds=np.array(ds);boot=np.mean(rng.choice(ds,(100000,len(ds)),replace=True),1);payload['comparisons'][m]={'mean_delta_vs_superad':float(ds.mean()),'wins':int((ds>0).sum()),'bootstrap_95ci':[float(x) for x in np.quantile(boot,[.025,.975])],'by_category':dict(zip(cats,map(float,ds)))}
 (OUT/'summary.json').write_text(json.dumps(payload,indent=2));(OUT/'A53_COMPLETE.json').write_text(json.dumps({'status':'complete'},indent=2));print(json.dumps(payload,indent=2),flush=True)
if __name__=='__main__':main()
