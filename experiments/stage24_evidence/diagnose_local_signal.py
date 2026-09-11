"""Source-only local signal audit and fixed, label-free pooling controls.

GT masks are used only in a separate posthoc diagnostic, never in scores.
Within-anomaly pixel AUC is not a full dataset segmentation benchmark.
"""
import json,time
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import roc_auc_score,average_precision_score
from evaluate_exchange import intervals

root=Path('/root/private_data/iad-vlm-anomaly');runs=root/'results/stage24_evidence'
out=runs/'20260911_h';out.mkdir(parents=True,exist_ok=True)
started=time.perf_counter()
manifest=pd.read_csv(runs/'20260910_a/manifest/images.csv');assert 'label' not in manifest
source=manifest[(manifest.dataset=='VisA')&(manifest.split.isin(['calibration','evaluation']))].copy()
rows=[]
for category,g in source.groupby('category'):
    base=runs/'20260911_d/fixed_bank/VisA'/category
    pred=pd.read_csv(base/'predictions.csv').set_index('image_id')
    for r in g.itertuples():
        a=np.load(base/'maps'/f'{r.image_id}.npz')['anomaly_map'].ravel()
        assert np.isfinite(a).all()
        record=dict(image_id=r.image_id,category=category,split=r.split,detector=float(pred.loc[r.image_id,'D_raw']),map_max=float(a.max()))
        for name,fraction in [('top01',.001),('top1',.01),('top5',.05)]:
            k=max(1,int(np.ceil(len(a)*fraction)));record[name]=float(np.partition(a,-k)[-k:].mean())
        rows.append(record)
    print('POOLING',category,flush=True)
scores=pd.DataFrame(rows);assert len(scores)==3026
scores.to_csv(out/'scores_no_labels.csv',index=False)
# Labels are first read after all non-oracle scores exist.
labels=pd.read_csv(runs/'20260910_a/manifest/evaluation_labels.csv')
evals=scores[scores.split=='evaluation'].merge(labels,on='image_id',validate='one_to_one');assert len(evals)==2162
metrics=[]
for category,g in evals.groupby('category'):
    for method in ['detector','map_max','top01','top1','top5']:
        metrics.append(dict(category=category,method=method,auroc=roc_auc_score(g.label,g[method]),ap=average_precision_score(g.label,g[method])))
table=pd.DataFrame(metrics);table.to_csv(out/'pooling_category_metrics.csv',index=False)
macro=table.groupby('method')[['auroc','ap']].mean();macro.to_csv(out/'pooling_macro_metrics.csv')
ci=intervals(evals,[(m,'detector') for m in ['map_max','top01','top1','top5']],replicates=2000)
pd.DataFrame(ci).to_csv(out/'pooling_intervals.csv',index=False)
visa=pd.read_csv(root/'datasets/VisA/split_csv/1cls.csv')
masks={str(root/'datasets/VisA'/r.image):str(root/'datasets/VisA'/r.mask) for r in visa.itertuples() if isinstance(r.mask,str) and r.mask}
anomalies=source.merge(labels,on='image_id',validate='one_to_one');anomalies=anomalies[(anomalies.split=='evaluation')&(anomalies.label==1)]
diagnostics=[]
for category,g in anomalies.groupby('category'):
    threshold=float(scores.loc[(scores.category==category)&(scores.split=='calibration'),'map_max'].quantile(.95))
    for r in g.itertuples():
        a=np.load(runs/'20260911_d/fixed_bank/VisA'/category/'maps'/f'{r.image_id}.npz')['anomaly_map']
        with Image.open(masks[r.path]) as handle:
            mask=np.array(handle.convert('L').resize((a.shape[1],a.shape[0]),Image.Resampling.NEAREST))>0
        row=dict(image_id=r.image_id,category=category,gt_pixels=int(mask.sum()),normal_calibration_max_q95=threshold)
        if not mask.any() or mask.all():
            row['valid']=False;diagnostics.append(row);continue
        inside=a[mask];outside=a[~mask]
        top=a>=np.quantile(a,.99)
        row.update(valid=True,within_anomaly_pixel_auc=roc_auc_score(mask.ravel(),a.ravel()),
            peak_inside_gt=bool(mask.ravel()[a.argmax()]),defect_peak=float(inside.max()),background_peak=float(outside.max()),
            defect_peak_beats_background=bool(inside.max()>outside.max()),
            defect_peak_above_normal_q95=bool(inside.max()>threshold),
            top1pct_gt_recall=float((top&mask).sum()/mask.sum()),
            top1pct_precision=float((top&mask).sum()/top.sum()))
        diagnostics.append(row)
    print('LOCAL_GT',category,flush=True)
local=pd.DataFrame(diagnostics);assert len(local)==1200
local.to_csv(out/'local_per_anomaly.csv',index=False)
fields=['within_anomaly_pixel_auc','peak_inside_gt','defect_peak_beats_background','defect_peak_above_normal_q95','top1pct_gt_recall','top1pct_precision']
summary=local[local.valid].groupby('category')[fields].mean();summary.to_csv(out/'local_category_summary.csv')
(out/'complete.json').write_text(json.dumps(dict(status='complete',source_only=True,anomalies=len(local),valid_anomalies=int(local.valid.sum()),
    seconds=time.perf_counter()-started,primary_pooling_control='top1 versus detector',
    other_pooling_controls='exploratory; no best variant selected for target evaluation',
    caveats=['Within-anomaly pixel AUC compares defect and background in the same anomalous image; not official dataset pixel AUROC.',
    'GT-conditioned quantities are oracle diagnostics, never deployable scores.',
    '95th-percentile threshold uses source normal calibration maxima, not test normals.',
    'Pooling uses existing smoothed maps; failure does not rule out every pooling method.',
    'Bootstrap conditional on fixed predictions, no correction for repeated exploration.']),indent=2))
print(macro.to_string());print(pd.DataFrame(ci).to_string(index=False));print(summary.to_string())
