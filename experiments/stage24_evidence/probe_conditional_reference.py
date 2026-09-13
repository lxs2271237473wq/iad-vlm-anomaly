"""Fixed normal-reference controls on identical source matched pairs; no training."""
import json
from types import SimpleNamespace
import numpy as np
import pandas as pd
import cv2
from scipy.spatial import cKDTree
from local_learning import RUN,features
from common import sha256

OUT=RUN/'20260912_w'

def main():
    OUT.mkdir(parents=True,exist_ok=False);cv2.setNumThreads(1)
    cache=np.load(RUN/'20260911_m/sampled_normal_features.npz',allow_pickle=True)
    # Existing trusted audit cache includes target rows; only source rows are selected.
    source_idx=cache['dataset']=='VisA'
    normal={k:cache[k][source_idx] for k in ['features','category','role','stratum','image_id']}
    pairs=pd.read_csv(RUN/'20260912_u/pairs.csv');data=np.load(RUN/'20260911_j/training_samples.npz')
    manifest=pd.read_csv(RUN/'20260910_a/manifest/images.csv').set_index('image_id')
    protocol=dict(code_sha256=sha256(__file__),k=32,calipers=[.01,.05,.1],reference='8 source normal fit images per category, uniform plus high-response cached pixels; exact duplicate feature vectors removed',scaling='Normal-fit median/MAD, standard deviation fallback, floor 1e-6',support='95th percentile of 32nd map-neighbor distance on disjoint normal check images; descriptive only',controls=['map_nn','appearance_nn','joint_nn','conditional_appearance'],target_rows_used=False,training=False)
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
    rows=[];reference_rows=[]
    for category,cp in pairs.groupby('category',sort=True):
        fit=(normal['category']==category)&(normal['role']=='fit');check=(normal['category']==category)&(normal['role']=='check')
        assert len(set(normal['image_id'][fit]))==len(set(normal['image_id'][check]))==8
        ref=np.unique(normal['features'][fit],axis=0);center=np.median(ref,axis=0)
        scale=1.4826*np.median(abs(ref-center),axis=0);scale=np.where(scale>1e-6,scale,np.maximum(ref.std(0),1e-6))
        z=(ref-center)/scale;map_tree=cKDTree(z[:,:4]);app_tree=cKDTree(z[:,4:]);joint_tree=cKDTree(z)
        cz=(normal['features'][check]-center)/scale
        check_dist=map_tree.query(cz[:,:4],k=32)[0][:,-1];support=float(np.quantile(check_dist,.95))
        reference_rows.append(dict(category=category,reference_points=len(ref),map_support_threshold=support))
        def score(x):
            q=(x-center)/scale;dist,idx=map_tree.query(q[:,:4],k=32)
            # Same fixed normal pool for every comparator; no label-dependent reference fitting.
            values=dict(map_nn=dist[:,0],appearance_nn=app_tree.query(q[:,4:])[0],joint_nn=joint_tree.query(q)[0],conditional_appearance=np.linalg.norm(q[:,None,4:]-z[idx,4:],axis=2).mean(1))
            assert all(np.isfinite(v).all() for v in values.values())
            return values,dist[:,-1]<=support,dist[:,-1]
        pos_idx=(data['category']==category)&(data['y']==1);px_all=data['x'][pos_idx];ids=data['image_id'][pos_idx]
        assert not set(ids)&set(normal['image_id'][fit|check])
        for image_id,g in cp.groupby('image_id',sort=True):
            g=g.sort_values('positive_cache_ordinal');px=px_all[ids==image_id]
            assert np.array_equal(g.positive_cache_ordinal,np.arange(len(px)))
            r=manifest.loc[image_id];assert r.dataset=='VisA'
            fx=features(SimpleNamespace(image_id=image_id,category=category,path=r.path));nx=fx[g.negative_y.to_numpy()*512+g.negative_x.to_numpy()]
            assert np.allclose(px[:,0],g.positive_response) and np.allclose(nx[:,0],g.negative_response)
            ps,po,pd_=score(px);ns,no,nd=score(nx)
            for caliper in [.01,.05,.1]:
                keep=g.distance.to_numpy()<=caliper;both=keep&po&no
                row=dict(category=category,image_id=image_id,caliper=caliper,matched_count=int(keep.sum()),supported_count=int(both.sum()),matched_fraction=float(keep.mean()),supported_fraction_all_positives=float(both.mean()))
                for name in ps:
                    accuracy=(np.sign(ps[name]-ns[name])+1)/2
                    row[name]=float(accuracy[keep].mean()) if keep.any() else np.nan
                    row[name+'_supported']=float(accuracy[both].mean()) if both.any() else np.nan
                rows.append(row)
        pd.DataFrame(rows).to_csv(OUT/'per_image.partial.csv',index=False);print('COMPLETE',category,flush=True)
    f=pd.DataFrame(rows);f.to_csv(OUT/'per_image.csv',index=False);pd.DataFrame(reference_rows).to_csv(OUT/'reference_summary.csv',index=False)
    metrics=['matched_fraction','supported_fraction_all_positives']+list(ps)+[n+'_supported' for n in ps]
    table=f.groupby(['category','caliper'])[metrics].mean();table.to_csv(OUT/'category_scores.csv');macro=table.groupby('caliper').mean();macro.to_csv(OUT/'macro_scores.csv')
    (OUT/'complete.json').write_text(json.dumps(dict(status='complete',images=f.image_id.nunique(),limitations='GT-assisted matched source diagnostic; normals only for references; no target evaluation; support heuristic not guarantee; conditional matched accuracy not detection AUROC; no tuning or novelty claim.'),indent=2));print(macro.to_string())

if __name__=='__main__':main()
