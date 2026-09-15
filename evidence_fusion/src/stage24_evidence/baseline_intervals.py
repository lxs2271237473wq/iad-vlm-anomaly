"""Paired scene-group bootstrap for frozen AD2 simple fusion baselines."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd

def group_weights(groups,rng,reps):
    codes,unique=pd.factorize(groups,sort=True)
    counts=rng.multinomial(len(unique),np.full(len(unique),1/len(unique)),size=reps)
    weights=counts[:,codes].astype(np.float64)
    return weights/weights.sum(axis=1,keepdims=True)

def matrix(g,column):
    p=g[g.label==1][column].to_numpy(); n=g[g.label==0][column].to_numpy()
    d=p[:,None]-n[None,:]
    return (d>0).astype(float)+.5*(d==0)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--run',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True); args=ap.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    p=pd.read_csv(args.run/'simple_fusion/target_predictions_no_labels.csv')
    labels=pd.read_csv(args.run/'manifest/evaluation_labels.csv')
    p=p.merge(labels,on='image_id',validate='one_to_one')
    assert len(p)==1084 and p.category.nunique()==8
    rng=np.random.default_rng(42); caches=[]; groups=[]
    for category,g in p.groupby('category'):
        positive=g[g.label==1]; negative=g[g.label==0]
        cp=group_weights(positive.scene_group,rng,5000)
        cn=group_weights(negative.scene_group,rng,5000)
        caches.append((g,cp,cn))
        groups.append(dict(category=category,normal_groups=negative.scene_group.nunique(),
                           anomalous_groups=positive.scene_group.nunique(),images=len(g)))
    rows=[]
    for method,base in [('fixed_source','D_z'),('logistic_source','D_z'),
                        ('fixed_source','naive_context_top1'),('logistic_source','naive_context_top1')]:
        points=[]; samples=[]
        for g,cp,cn in caches:
            diff=matrix(g,method)-matrix(g,base)
            points.append(diff.mean()); samples.append(np.einsum('bi,ij,bj->b',cp,diff,cn,optimize=True))
        bootstrap=np.stack(samples).mean(0)
        rows.append(dict(method=method,baseline=base,delta_macro_auroc=float(np.mean(points)),
                         lower95=float(np.quantile(bootstrap,.025)),upper95=float(np.quantile(bootstrap,.975))))
    pd.DataFrame(rows).to_csv(args.out/'ad2_simple_fusion_paired_intervals.csv',index=False)
    (args.out/'protocol.json').write_text(json.dumps(dict(replicates=5000,seed=42,groups=groups,
       grouping='AD2 filename prefix, category, official split and normal/bad directory; provisional scene identity',
       stratification='category and normal/anomalous groups',fitted_models_fixed=True,categories_fixed=True),indent=2))
    print(pd.DataFrame(rows).to_string(index=False))

if __name__=='__main__': main()
