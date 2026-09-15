"""Source-only response-stratified sampled-pixel diagnostic, not matched causal inference."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from local_learning import RUN
from common import sha256

OUT=RUN/'20260912_s_v3'

def main():
    OUT.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4)
    normal=np.load(RUN/'20260911_m/sampled_normal_features.npz',allow_pickle=True) # Trusted cache created by our earlier audit
    data=np.load(RUN/'20260911_j/training_samples.npz')
    protocol=dict(code_sha256=sha256(Path(__file__)),quantiles=[0,.5,.9,.95,.99,1],minimum_images_per_label=5,head_seed=42,
        description='Normal source fit/uniform pixels determine bins. Source evaluation sampled pixels only; frozen category-LOCO heads. No target evaluations, retraining or threshold selection.',
        caveats='Response strata are approximate matching, with residual response confounding. GT-selected pixel sample, not dense detection metrics. Tail bins unbounded. No pixel-IID significance.')
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
    rows=[];coverage=[];edges_out={}
    for category in sorted(set(data['category'])):
        ni=(normal['dataset']=='VisA')&(normal['category']==category)&(normal['role']=='fit')&(normal['stratum']=='uniform')
        assert len(set(normal['image_id'][ni]))==8
        knots=np.unique(np.quantile(normal['features'][ni,0],protocol['quantiles']))
        edges=np.r_[-np.inf,knots,np.inf];edges_out[category]=knots.tolist()
        idx=data['category']==category;x=data['x'][idx];y=data['y'][idx];ids=data['image_id'][idx]
        assert not set(ids)&set(normal['image_id'][ni])
        bins=np.searchsorted(knots,x[:,0],side='right')
        scaler=np.load(RUN/'20260911_j'/category/'scaler.npz')
        head=torch.nn.Sequential(torch.nn.Linear(7,32),torch.nn.ReLU(),torch.nn.Linear(32,1))
        head.load_state_dict(torch.load(RUN/'20260911_j'/category/'mlp.pt',map_location='cpu',weights_only=True));head.eval()
        with torch.no_grad():logits=head(torch.tensor((x-scaler['mean'])/scaler['scale'])).flatten().numpy()
        assert np.isfinite(logits).all()
        supported=np.zeros(len(y),bool)
        for b in range(len(edges)-1):
            keep=bins==b;yy=y[keep];ii=ids[keep]
            npos=len(set(ii[yy==1]));nneg=len(set(ii[yy==0]));ok=min(npos,nneg)>=5
            supported[keep]=ok
            row=dict(category=category,bin=b,lower=edges[b],upper=edges[b+1],positive_pixels=int(sum(yy==1)),negative_pixels=int(sum(yy==0)),positive_images=npos,negative_images=nneg,supported=ok)
            if ok:
                meta=pd.DataFrame(dict(image_id=ii,label=yy))
                w=1/meta.groupby(['image_id','label']).label.transform('size').to_numpy()
                row.update(response_auc=roc_auc_score(yy,x[keep,0],sample_weight=w),mlp_auc=roc_auc_score(yy,logits[keep],sample_weight=w))
                row['mlp_minus_response']=row['mlp_auc']-row['response_auc']
            rows.append(row)
        for label in [0,1]:
            use=y==label;frame=pd.DataFrame(dict(image_id=ids[use],covered=supported[use],tail=(x[use,0]>knots[-1])))
            coverage.append(dict(category=category,label=label,supported_fraction_image_balanced=float(frame.groupby('image_id').covered.mean().mean()),above_normal_max_image_balanced=float(frame.groupby('image_id')['tail'].mean().mean())))
        print('COMPLETE',category,flush=True)
    pd.DataFrame(rows).to_csv(OUT/'strata.csv',index=False)
    pd.DataFrame(coverage).to_csv(OUT/'coverage.csv',index=False)
    (OUT/'normal_bin_edges.json').write_text(json.dumps(edges_out,indent=2))
    (OUT/'complete.json').write_text(json.dumps(dict(status='complete',**protocol),indent=2))
    print(pd.DataFrame(coverage).groupby('label').mean(numeric_only=True).to_string())
    valid=pd.DataFrame(rows).query('supported')
    print(valid.groupby('category')[['response_auc','mlp_auc','mlp_minus_response']].mean().to_string())

if __name__=='__main__':main()


