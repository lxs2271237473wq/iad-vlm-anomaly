"""Paired AD2 public evaluation, after predictions for all eight classes exist."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score
from baseline_intervals import group_weights,matrix

ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);ap.add_argument('--high',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=False)
assert json.loads((args.high/'queue_complete.json').read_text())['status']=='complete'
high=pd.concat([pd.read_csv(p) for p in sorted((args.high/'AD2').glob('*/predictions.csv'))])
high=high[high.split=='evaluation'][['image_id','category','D_z','naive_context_top1']].rename(columns={'D_z':'D512','naive_context_top1':'naive512'})
low=pd.read_csv(args.run/'simple_fusion/target_predictions_no_labels.csv')
low=low[['image_id','D_z','naive_context_top1','fixed_source','logistic_source']].rename(columns={'D_z':'D256','naive_context_top1':'naive256'})
frame=high.merge(low,on='image_id',validate='one_to_one').merge(pd.read_csv(args.run/'manifest/evaluation_labels.csv'),on='image_id',validate='one_to_one')
assert len(frame)==1084 and frame.category.nunique()==8 and frame.image_id.is_unique
frame.to_csv(args.out/'predictions_with_evaluation_labels.csv',index=False)
methods=['D256','D512','naive256','naive512','fixed_source','logistic_source']
rows=[]
for category,g in frame.groupby('category'):
    for method in methods:rows.append(dict(category=category,method=method,auroc=roc_auc_score(g.label,g[method]),ap=average_precision_score(g.label,g[method])))
table=pd.DataFrame(rows);table.to_csv(args.out/'category_metrics.csv',index=False)
macro=table.groupby('method')[['auroc','ap']].mean();macro.to_csv(args.out/'macro_metrics.csv')
rng=np.random.default_rng(42);cache=[]
for _,g in frame.groupby('category'):
    cp=group_weights(g.loc[g.label==1,'scene_group'],rng,5000);cn=group_weights(g.loc[g.label==0,'scene_group'],rng,5000)
    cache.append((g,cp,cn))
rows=[]
for method,base in [('D512','D256'),('D512','naive256'),('D512','logistic_source'),('naive512','D512')]:
    points=[];draws=[]
    for g,cp,cn in cache:
        delta=matrix(g,method)-matrix(g,base);points.append(float(delta.mean()))
        draws.append(np.einsum('bi,ij,bj->b',cp,delta,cn,optimize=True))
    samples=np.stack(draws).mean(0)
    rows.append(dict(method=method,baseline=base,delta_macro_auroc=float(np.mean(points)),lower95=float(np.quantile(samples,.025)),upper95=float(np.quantile(samples,.975)),categories_improved=sum(x>0 for x in points)))
pd.DataFrame(rows).to_csv(args.out/'paired_intervals.csv',index=False)
(args.out/'protocol.json').write_text(json.dumps(dict(status='complete',primary_comparison='D512 versus D256',replicates=5000,
    note='Provisional filename scene groups; fixed models/categories; public target previously inspected; no new target tuning. Resolution improvement is not a novel module.'),indent=2))
print(macro.to_string());print(pd.DataFrame(rows).to_string(index=False))
