import json
import numpy as np
import pandas as pd
from local_learning import ROOT
from baseline_intervals import group_weights,matrix

out=ROOT/'results/stage25_tiny_defects/dinov2_ad2_v1'
f=pd.read_csv(out/'evaluation_predictions.csv');assert len(f)==1084
rng=np.random.default_rng(42);rows=[];group_info=[]
for scope,data in [('all',f),('tiny_only',f[(f.label==0)|(f.stratum=='tiny_only')])]:
    cache=[]
    for category,g in data.groupby('category'):
        if g.label.nunique()!=2:continue
        p=g[g.label==1];n=g[g.label==0]
        cache.append((g,group_weights(p.scene_group,rng,5000),group_weights(n.scene_group,rng,5000)))
        group_info.append(dict(scope=scope,category=category,positive_images=len(p),negative_images=len(n),positive_groups=p.scene_group.nunique(),negative_groups=n.scene_group.nunique()))
    for method in ['fusion_frozen','dino_max','dino_top1']:
        points=[];samples=[]
        for g,wp,wn in cache:
            delta=matrix(g,method)-matrix(g,'D_z');points.append(float(delta.mean()));samples.append(np.einsum('bi,ij,bj->b',wp,delta,wn,optimize=True))
        draws=np.stack(samples).mean(0)
        rows.append(dict(scope=scope,method=method,baseline='D_z',delta=float(np.mean(points)),lower95=float(np.quantile(draws,.025)),upper95=float(np.quantile(draws,.975)),categories=len(cache),categories_improved=sum(p>0 for p in points)))
pd.DataFrame(rows).to_csv(out/'scene_intervals.csv',index=False);pd.DataFrame(group_info).to_csv(out/'scene_counts.csv',index=False)
(out/'scene_analysis.json').write_text(json.dumps(dict(status='complete',replicates=5000,groups='Provisional filename-derived scene groups within category and label',limitations='Fixed models/categories, public target repeatedly inspected, no multi-exploration correction. Tiny-only filters variants by observed annotation; not independent target validation.'),indent=2))
print(pd.DataFrame(rows).to_string(index=False))
