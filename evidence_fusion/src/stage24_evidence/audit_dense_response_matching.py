"""Dense negative candidates, fixed cached positive samples; source diagnostic only."""
import json
import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from local_learning import RUN,ROOT,features
from common import sha256

OUT=RUN/'20260912_u'

def nearest_indices(values,queries):
    order=np.argsort(values,kind='stable');sorted_values=values[order]
    right=np.clip(np.searchsorted(sorted_values,queries),0,len(order)-1)
    left=np.maximum(right-1,0)
    pick=np.where(abs(sorted_values[left]-queries)<=abs(sorted_values[right]-queries),left,right)
    return order[pick]

def main():
    # Test boundary and nearest-neighbor correctness against exhaustive distances.
    rng=np.random.default_rng(42);v=rng.normal(size=100);q=np.r_[rng.normal(size=50),-100,100]
    selected=nearest_indices(v,q)
    assert np.allclose(abs(v[selected]-q),abs(v[:,None]-q).min(0))
    OUT.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4);cv2.setNumThreads(1)
    d=np.load(RUN/'20260911_j/training_samples.npz')
    manifest=pd.read_csv(RUN/'20260910_a/manifest/images.csv')
    source=manifest[(manifest.dataset=='VisA')&(manifest.split=='evaluation')].set_index('image_id')
    official=pd.read_csv(ROOT/'datasets/VisA/split_csv/1cls.csv')
    masks={str(ROOT/'datasets/VisA'/r.image):str(ROOT/'datasets/VisA'/r.mask) for r in official.itertuples() if isinstance(r.mask,str) and r.mask}
    protocol=dict(code_sha256=sha256(__file__),calipers=[.01,.05,.1],exclusion='17x17 GT dilation; matching diagnostic only',positives='Unchanged cached source positive features',negatives='All pixels outside exclusion in same image',head_seed=42,target_used=False,nearest_search_test=True)
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
    rows=[];pairs=[]
    for category in sorted(set(d['category'])):
        ix=(d['category']==category)&(d['y']==1);x=d['x'][ix];ids=d['image_id'][ix]
        scaler=np.load(RUN/'20260911_j'/category/'scaler.npz')
        model=torch.nn.Sequential(torch.nn.Linear(7,32),torch.nn.ReLU(),torch.nn.Linear(32,1))
        model.load_state_dict(torch.load(RUN/'20260911_j'/category/'mlp.pt',map_location='cpu',weights_only=True));model.eval()
        def score(z):
            with torch.no_grad():return model(torch.tensor((z-scaler['mean'])/scaler['scale'])).flatten().numpy()
        for image_id in sorted(set(ids)):
            row=source.loc[image_id];assert row.category==category
            from types import SimpleNamespace
            fx=features(SimpleNamespace(image_id=image_id,category=category,path=row.path))
            with Image.open(masks[row.path]) as handle:mask=np.asarray(handle.convert('L').resize((512,512),Image.Resampling.NEAREST))>0
            candidates=np.flatnonzero(~(cv2.dilate(mask.astype(np.uint8),np.ones((17,17),np.uint8))>0).ravel())
            assert len(candidates)>0
            px=x[ids==image_id];selected=candidates[nearest_indices(fx[candidates,0],px[:,0])]
            nx=fx[selected];distance=abs(px[:,0]-nx[:,0]);ps=score(px);ns=score(nx)
            assert np.isfinite(distance).all() and not mask.ravel()[selected].any()
            for i,(flat,dist) in enumerate(zip(selected,distance)):
                pairs.append(dict(category=category,image_id=image_id,positive_cache_ordinal=i,negative_y=int(flat//512),negative_x=int(flat%512),distance=float(dist),positive_response=float(px[i,0]),negative_response=float(nx[i,0])))
            for caliper in [.01,.05,.1]:
                keep=distance<=caliper;n=int(keep.sum())
                rows.append(dict(category=category,image_id=image_id,caliper=caliper,positive_count=len(px),matched_count=n,coverage=n/len(px),median_nearest_distance=float(np.median(distance)),reuse_fraction=1-len(set(selected[keep]))/n if n else np.nan,response_pair_accuracy=float(np.mean((np.sign(px[keep,0]-nx[keep,0])+1)/2)) if n else np.nan,MLP_pair_accuracy=float(np.mean((np.sign(ps[keep]-ns[keep])+1)/2)) if n else np.nan))
        pd.DataFrame(rows).to_csv(OUT/'per_image.partial.csv',index=False)
        pd.DataFrame(pairs).to_csv(OUT/'pairs.partial.csv',index=False)
        print('COMPLETE',category,flush=True)
    f=pd.DataFrame(rows);f.to_csv(OUT/'per_image.csv',index=False);pd.DataFrame(pairs).to_csv(OUT/'pairs.csv',index=False)
    summary=f.groupby(['category','caliper']).agg(images=('image_id','size'),images_with_matches=('matched_count',lambda s:int((s>0).sum())),coverage=('coverage','mean'),median_nearest_distance=('median_nearest_distance','median'),reuse_fraction=('reuse_fraction','mean'),response_pair_accuracy=('response_pair_accuracy','mean'),MLP_pair_accuracy=('MLP_pair_accuracy','mean'))
    summary.to_csv(OUT/'category_summary.csv');macro=summary.groupby('caliper').mean();macro.to_csv(OUT/'macro_summary.csv')
    (OUT/'complete.json').write_text(json.dumps(dict(status='complete',images=f.image_id.nunique(),limitations='GT-assisted diagnostic, unchanged sampled positives, conditional matched accuracy, replacement allowed; no independent-pair inference or target evaluation.',**protocol),indent=2))
    print(macro.to_string())

if __name__=='__main__':main()
