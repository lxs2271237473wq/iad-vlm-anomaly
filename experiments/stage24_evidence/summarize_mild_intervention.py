"""Image-level paired bootstrap, fixed categories/models, no pair IID assumption."""
import json
import numpy as np
import pandas as pd
from local_learning import RUN

OUT=RUN/'20260912_r'

def main():
    f=pd.read_csv(OUT/'scores.csv')
    assert len(f)==1920 and f.image_id.nunique()==384
    assert not f.duplicated(['image_id','variant']).any()
    variants=sorted(set(f.variant)-{'identity'});rng=np.random.default_rng(42)
    changes={v:[] for v in variants};draws={v:[] for v in variants}
    for category,g in f.groupby('category'):
        base=g[g.variant=='identity'];pos=base.loc[base.label==1,'image_id'].tolist();neg=base.loc[base.label==0,'image_id'].tolist()
        assert len(pos)==len(neg)==16
        wp=rng.multinomial(16,np.ones(16)/16,size=5000)/16
        wn=rng.multinomial(16,np.ones(16)/16,size=5000)/16
        def auc_matrix(variant,method):
            h=g[g.variant==variant].set_index('image_id')
            difference=h.loc[pos,method].to_numpy()[:,None]-h.loc[neg,method].to_numpy()[None,:]
            return (np.sign(difference)+1)/2
        for variant in variants:
            delta=(auc_matrix(variant,'MLP')-auc_matrix('identity','MLP'))-(auc_matrix(variant,'D')-auc_matrix('identity','D'))
            changes[variant].append(dict(category=category,variant=variant,relative_auroc_change=float(delta.mean())))
            draws[variant].append(np.einsum('bi,ij,bj->b',wp,delta,wn,optimize=True))
    rows=[]
    for variant in variants:
        sample=np.stack(draws[variant]).mean(0)
        rows.append(dict(variant=variant,relative_auroc_change=float(np.mean([r['relative_auroc_change'] for r in changes[variant]])),lower95=float(np.quantile(sample,.025)),upper95=float(np.quantile(sample,.975))))
    pd.DataFrame(rows).to_csv(OUT/'relative_change_intervals.csv',index=False)
    pd.DataFrame([r for v in variants for r in changes[v]]).to_csv(OUT/'relative_change_categories.csv',index=False)
    (OUT/'analysis_complete.json').write_text(json.dumps(dict(status='complete',replicates=5000,estimand='(MLP perturbed AUROC - MLP identity AUROC) - (D perturbed AUROC - D identity AUROC); negative means extra MLP loss',limitations='Exploratory overlapping source sample; fixed categories and head seed; image resampling does not model unknown scene dependence; no multiple-comparison correction; no causal attribution.'),indent=2))
    print(pd.DataFrame(rows).to_string(index=False))

if __name__=='__main__':main()
