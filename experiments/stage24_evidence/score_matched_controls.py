"""Score all existing source LOCO controls on identical saved dense matches."""
import json
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch
from local_learning import RUN,features
from local_learning_robustness import MASKS,SEEDS
from common import sha256

OUT=RUN/'20260912_v'

def main():
    OUT.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4)
    import cv2
    cv2.setNumThreads(1)
    pairs=pd.read_csv(RUN/'20260912_u/pairs.csv')
    old=pd.read_csv(RUN/'20260912_u/per_image.csv')
    d=np.load(RUN/'20260911_j/training_samples.npz')
    manifest=pd.read_csv(RUN/'20260910_a/manifest/images.csv').set_index('image_id')
    (OUT/'protocol.json').write_text(json.dumps(dict(code_sha256=sha256(__file__),pairs_sha256=sha256(RUN/'20260912_u/pairs.csv'),seeds=SEEDS,calipers=[.01,.05,.1],training=False,target_used=False,controls=list(MASKS),limitations='Fixed sampled positives and GT-assisted negative selection. Conditional pair accuracy, not image AUROC; head seeds only.'),indent=2))
    rows=[]
    for category,cp in pairs.groupby('category',sort=True):
        idx=(d['category']==category)&(d['y']==1);x=d['x'][idx];ids=d['image_id'][idx]
        base=RUN/'20260911_k'/category;scaler=np.load(base/'scaler.npz');models={}
        for variant,mask in MASKS.items():
            for seed in SEEDS:
                name=f'{variant}_{seed}';model=torch.nn.Sequential(torch.nn.Linear(7,32),torch.nn.ReLU(),torch.nn.Linear(32,1))
                model.load_state_dict(torch.load(base/f'{name}.pt',map_location='cpu',weights_only=True));model.eval();models[name]=(model,mask)
        for image_id,g in cp.groupby('image_id',sort=True):
            g=g.sort_values('positive_cache_ordinal');px=x[ids==image_id]
            assert np.array_equal(g.positive_cache_ordinal.to_numpy(),np.arange(len(px)))
            r=manifest.loc[image_id];assert r.dataset=='VisA' and r.category==category
            fx=features(SimpleNamespace(image_id=image_id,category=category,path=r.path))
            nx=fx[g.negative_y.to_numpy()*512+g.negative_x.to_numpy()]
            assert np.allclose(px[:,0],g.positive_response) and np.allclose(nx[:,0],g.negative_response)
            distance=abs(px[:,0]-nx[:,0]);assert np.allclose(distance,g.distance)
            predictions={}
            with torch.no_grad():
                for name,(model,mask) in models.items():
                    p=model(torch.tensor((px-scaler['mean'])/scaler['scale']*mask)).flatten().numpy()
                    n=model(torch.tensor((nx-scaler['mean'])/scaler['scale']*mask)).flatten().numpy()
                    predictions[name]=(np.sign(p-n)+1)/2
            for caliper in [.01,.05,.1]:
                keep=distance<=caliper
                result=dict(category=category,image_id=image_id,caliper=caliper,matched_count=int(keep.sum()),coverage=float(keep.mean()))
                for name,values in predictions.items():result[name]=float(values[keep].mean()) if keep.any() else np.nan
                check=old[(old.image_id==image_id)&(old.caliper==caliper)].iloc[0]
                assert result['matched_count']==check.matched_count
                assert np.isclose(result['full_42'],check.MLP_pair_accuracy,equal_nan=True,atol=1e-6)
                rows.append(result)
        pd.DataFrame(rows).to_csv(OUT/'per_image.partial.csv',index=False);print('COMPLETE',category,flush=True)
    f=pd.DataFrame(rows);f.to_csv(OUT/'per_image.csv',index=False)
    methods=list(models);table=f.groupby(['category','caliper'])[methods].mean();table.to_csv(OUT/'category_scores.csv')
    macro=table.groupby('caliper').mean();macro.to_csv(OUT/'macro_scores.csv')
    (OUT/'complete.json').write_text(json.dumps(dict(status='complete',images=f.image_id.nunique(),same_pairs=True,previous_full42_parity=True,limitations='Exploratory fixed matched source subset; no target generalization or causal feature attribution.'),indent=2))
    print(macro.to_string())

if __name__=='__main__':main()
