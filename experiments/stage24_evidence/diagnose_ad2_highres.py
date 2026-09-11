"""Posthoc target diagnostics; does not train or select a model."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from baseline_intervals import group_weights,matrix

root=Path('/root/private_data/iad-vlm-anomaly/results/stage24_evidence')
out=root/'20260911_e/diagnostics';out.mkdir(exist_ok=False)
f=pd.read_csv(root/'20260911_e/evaluation/predictions_with_evaluation_labels.csv')
g=pd.read_csv(root/'20260911_b/geometry_baseline_transfer/predictions_no_labels.csv')
f=f.merge(g[['image_id','logistic_geometry']],on='image_id',validate='one_to_one');assert len(f)==1084
rng=np.random.default_rng(42);points=[];draws=[]
for _,group in f.groupby('category'):
    cp=group_weights(group.loc[group.label==1,'scene_group'],rng,5000)
    cn=group_weights(group.loc[group.label==0,'scene_group'],rng,5000)
    diff=matrix(group,'D512')-matrix(group,'logistic_geometry')
    points.append(float(diff.mean()));draws.append(np.einsum('bi,ij,bj->b',cp,diff,cn,optimize=True))
samples=np.stack(draws).mean(0)
ci=dict(method='D512',baseline='previous_geometry256',delta_macro_auroc=float(np.mean(points)),lower95=float(np.quantile(samples,.025)),upper95=float(np.quantile(samples,.975)),categories_improved=sum(x>0 for x in points),replicates=5000)
(out/'strong_control_interval.json').write_text(json.dumps(ci,indent=2))
rows=[]
for p in sorted((root/'20260911_e/fixed_bank/AD2').glob('*/predictions.csv')):
    scores=pd.read_csv(p).merge(f[['image_id','label']],on='image_id',how='left',validate='one_to_one')
    for name,sel in [('calibration_normal',scores.split=='calibration'),('evaluation_normal',scores.label==0),('evaluation_anomaly',scores.label==1)]:
        v=scores.loc[sel,'D_z']
        rows.append(dict(category=p.parent.name,group=name,n=len(v),median=float(v.median()),q10=float(v.quantile(.1)),q90=float(v.quantile(.9))))
pd.DataFrame(rows).to_csv(out/'score_distributions.csv',index=False)
print(json.dumps(ci));print(pd.DataFrame(rows).to_string(index=False))
