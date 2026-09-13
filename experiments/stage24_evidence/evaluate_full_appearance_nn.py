"""Frozen normal-only appearance nearest-neighbor, dense max image scoring."""
import json,time
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from scipy.spatial import cKDTree
from sklearn.metrics import roc_auc_score,average_precision_score
from local_learning import RUN,ROOT,features
from common import sha256
from evaluate_exchange import intervals

OUT=RUN/'20260912_x'

def main():
    OUT.mkdir(parents=True,exist_ok=False);cv2.setNumThreads(1)
    cache=np.load(RUN/'20260911_m/sampled_normal_features.npz',allow_pickle=True)
    use=(cache['dataset']=='VisA')&(cache['role']=='fit')
    normal=cache['features'][use];cats=cache['category'][use];ids=cache['image_id'][use]
    manifest=pd.read_csv(RUN/'20260910_a/manifest/images.csv')
    source=manifest[(manifest.dataset=='VisA')&(manifest.split=='evaluation')]
    assert len(source)==2162 and not set(source.image_id)&set(ids)
    protocol=dict(code_sha256=sha256(__file__),reference='Same source category 8 normal fit images; uniform plus high-response pixels; unique 7D vectors as previous reference probe',features='Local intensity mean/std and gradient; fixed normal robust scaling',score='Maximum 3D appearance nearest-neighbor distance over all 512x512 pixels',selection='All 2162 source evaluation images; no masks/labels used to score',no_training=True,no_fusion=True,limitations='High-response reference sampling depends on fixed detector; existing cached detector maps loaded by shared feature extractor. Runtime includes shared extraction, not a standalone optimized deployment benchmark.')
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
    rows=[];references=[]
    for category,g in source.groupby('category',sort=True):
        ref=np.unique(normal[cats==category],axis=0);assert len(set(ids[cats==category]))==8
        center=np.median(ref,axis=0);scale=1.4826*np.median(abs(ref-center),axis=0)
        scale=np.where(scale>1e-6,scale,np.maximum(ref.std(0),1e-6))
        z=(ref[:,4:]-center[4:])/scale[4:];tree=cKDTree(z)
        # Verify exact query implementation against brute-force distances.
        query=z[:8]+.01
        expected=np.sqrt(((query[:,None,:]-z[None,:,:])**2).sum(2)).min(1)
        assert np.allclose(tree.query(query)[0],expected)
        references.append(dict(category=category,points=len(ref),center=center.tolist(),scale=scale.tolist()))
        for r in g.sort_values('image_id').itertuples():
            start=time.perf_counter();x=features(r)[:,4:];dist=tree.query((x-center[4:])/scale[4:],workers=4)[0]
            assert np.isfinite(dist).all();peak=int(dist.argmax())
            rows.append(dict(image_id=r.image_id,category=category,appearance_nn_max=float(dist[peak]),peak_y=peak//512,peak_x=peak%512,seconds=time.perf_counter()-start))
        pd.DataFrame(rows).to_csv(OUT/'predictions.partial.csv',index=False);print('COMPLETE',category,flush=True)
    f=pd.DataFrame(rows);f.to_csv(OUT/'predictions_no_labels.csv',index=False)
    (OUT/'references.json').write_text(json.dumps(references,indent=2))
    # Evaluation starts after all dense predictions are saved.
    previous=pd.read_csv(RUN/'20260911_k/oof_predictions.csv')
    f=f.merge(previous[['image_id','label','D_z','map_42','full_42']],on='image_id',validate='one_to_one');assert len(f)==2162
    f.to_csv(OUT/'evaluation_predictions.csv',index=False)
    metric=[]
    for category,g in f.groupby('category'):
        for method in ['appearance_nn_max','D_z','map_42','full_42']:
            metric.append(dict(category=category,method=method,auroc=roc_auc_score(g.label,g[method]),ap=average_precision_score(g.label,g[method])))
    table=pd.DataFrame(metric);table.to_csv(OUT/'category_metrics.csv',index=False);macro=table.groupby('method')[['auroc','ap']].mean();macro.to_csv(OUT/'macro_metrics.csv')
    comparisons=intervals(f,[('appearance_nn_max',b) for b in ['D_z','map_42','full_42']],replicates=2000)
    pd.DataFrame(comparisons).to_csv(OUT/'paired_intervals.csv',index=False)
    official=pd.read_csv(ROOT/'datasets/VisA/split_csv/1cls.csv')
    masks={str(ROOT/'datasets/VisA'/r.image):str(ROOT/'datasets/VisA'/r.mask) for r in official.itertuples() if isinstance(r.mask,str) and r.mask}
    paths=source.set_index('image_id').path;local=[]
    for r in f[f.label==1].itertuples():
        with Image.open(masks[paths[r.image_id]]) as handle:mask=np.asarray(handle.convert('L').resize((512,512),Image.Resampling.NEAREST))>0
        local.append(dict(image_id=r.image_id,category=r.category,peak_in_mask=bool(mask[r.peak_y,r.peak_x])))
    loc=pd.DataFrame(local);loc.to_csv(OUT/'peak_diagnostic.csv',index=False);loc.groupby('category').peak_in_mask.mean().to_csv(OUT/'peak_category.csv')
    (OUT/'complete.json').write_text(json.dumps(dict(status='complete',images=2162,anomalies=1200,exact_nn_test=True,macro_peak_in_mask=float(loc.groupby('category').peak_in_mask.mean().mean()),seconds_total=float(f.seconds.sum()),limitations='Source-only normal-reference baseline; supervised MLP controls use stronger labels; no paired-sample diagnostic equivalence, no target or novelty claim.'),indent=2))
    print(macro.to_string());print(pd.DataFrame(comparisons).to_string(index=False))

if __name__=='__main__':main()
