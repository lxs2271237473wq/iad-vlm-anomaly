"""Paired image bootstrap of conditional-reference diagnostic differences."""
import json
import numpy as np
import pandas as pd
from local_learning import RUN

OUT=RUN/'20260912_w'

def main():
    assert (OUT/'complete.json').exists()
    f=pd.read_csv(OUT/'per_image.csv');assert f.image_id.nunique()==1200
    rng=np.random.default_rng(42);rows=[]
    for caliper,g in f.groupby('caliper'):
        for suffix in ['', '_supported']:
            method='conditional_appearance'+suffix
            for baseline in ['appearance_nn','joint_nn','map_nn']:
                base=baseline+suffix;points=[];draws=[];counts=[]
                for category,c in g.groupby('category'):
                    c=c.dropna(subset=[method,base]);counts.append(len(c))
                    if len(c)==0:continue
                    delta=(c[method]-c[base]).to_numpy();points.append(float(delta.mean()))
                    draws.append(delta[rng.integers(0,len(delta),size=(5000,len(delta)))].mean(1))
                if not points:continue
                sample=np.stack(draws).mean(0)
                rows.append(dict(caliper=caliper,subset='supported' if suffix else 'all_matched',baseline=baseline,delta=float(np.mean(points)),lower95=float(np.quantile(sample,.025)),upper95=float(np.quantile(sample,.975)),categories=len(points),categories_improved=sum(v>0 for v in points),images=sum(counts)))
    result=pd.DataFrame(rows);result.to_csv(OUT/'comparison_intervals.csv',index=False)
    (OUT/'comparison_notes.json').write_text(json.dumps(dict(status='complete',replicates=5000,unit='Paired image within fixed category, only images with finite scores for both controls',limitations='Conditional selected subset; reference fit fixed; unknown scene dependence ignored; exploratory multiple comparisons unadjusted; not detection AUROC or independent validation.'),indent=2))
    print(result.to_string(index=False))

if __name__=='__main__':main()
