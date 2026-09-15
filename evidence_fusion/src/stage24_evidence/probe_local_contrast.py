"""Fixed local-contrast controls on cached maps; no mask inputs.

Gaussian center-surround is an established operation, not a novelty claim.
"""
import json
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score
from common import robust_fit,robust_apply,sha256
from evaluate_exchange import fit_logistic,intervals

root=Path('/root/private_data/iad-vlm-anomaly/results/stage24_evidence')
out=root/'20260911_i';out.mkdir(parents=True,exist_ok=True)
manifest=pd.read_csv(root/'20260910_a/manifest/images.csv');assert 'label' not in manifest
source=manifest[(manifest.dataset=='VisA')&(manifest.split.isin(['calibration','evaluation']))]
parts=[]
for category,g in source.groupby('category',sort=True):
    dest=out/category;dest.mkdir(exist_ok=True)
    if (dest/'complete.json').exists():
        assert json.loads((dest/'complete.json').read_text())['code_sha256']==sha256(Path(__file__))
        parts.append(pd.read_csv(dest/'scores.csv'));continue
    base=root/'20260911_d/fixed_bank/VisA'/category
    old=pd.read_csv(base/'predictions.csv').set_index('image_id')
    records=[]
    for r in g.itertuples():
        a=np.load(base/'maps'/f'{r.image_id}.npz')['anomaly_map'].astype(np.float32)
        assert a.shape==(512,512)
        local=cv2.GaussianBlur(a,(0,0),4,borderType=cv2.BORDER_REFLECT_101)
        surround=cv2.GaussianBlur(a,(0,0),16,borderType=cv2.BORDER_REFLECT_101)
        variance=np.maximum(cv2.GaussianBlur(a*a,(0,0),16,borderType=cv2.BORDER_REFLECT_101)-surround*surround,0)
        contrast=local-surround
        scale=np.maximum(np.sqrt(variance),max(.1*float(a.std()),1e-6))
        records.append(dict(image_id=r.image_id,category=category,split=r.split,D=float(old.loc[r.image_id,'D_raw']),
            contrast=float(contrast.max()),standardized_contrast=float((contrast/scale).max())))
    f=pd.DataFrame(records);params={}
    for field in ['D','contrast','standardized_contrast']:
        params[field]=robust_fit(f.loc[f.split=='calibration',field]);f[field+'_z']=robust_apply(f[field],params[field])
    f.to_csv(dest/'scores.csv',index=False);parts.append(f)
    (dest/'complete.json').write_text(json.dumps(dict(code_sha256=sha256(Path(__file__)),normal_calibration=params,center_sigma=4,surround_sigma=16),indent=2))
    print('COMPLETE',category,flush=True)
f=pd.concat(parts);assert len(f)==3026;f.to_csv(out/'all_scores_no_labels.csv',index=False)
labels=pd.read_csv(root/'20260910_a/manifest/evaluation_labels.csv')
f=f[f.split=='evaluation'].merge(labels,on='image_id',validate='one_to_one');assert len(f)==2162
variants=dict(linear_D=['D_z'],linear_contrast=['D_z','contrast_z'],linear_standardized=['D_z','standardized_contrast_z'])
pieces=[]
for category in sorted(f.category.unique()):
    tr=f[f.category!=category];va=f[f.category==category].copy()
    for name,columns in variants.items():va[name]=fit_logistic(tr,va,columns)
    pieces.append(va)
oof=pd.concat(pieces);oof.to_csv(out/'oof_predictions.csv',index=False)
rows=[]
for category,g in oof.groupby('category'):
    for method in ['D_z','contrast_z','standardized_contrast_z']+list(variants):
        rows.append(dict(category=category,method=method,auroc=roc_auc_score(g.label,g[method]),ap=average_precision_score(g.label,g[method])))
table=pd.DataFrame(rows);table.to_csv(out/'category_metrics.csv',index=False)
macro=table.groupby('method')[['auroc','ap']].mean();macro.to_csv(out/'macro_metrics.csv')
ci=intervals(oof,[('linear_contrast','linear_D'),('linear_standardized','linear_D'),('linear_standardized','linear_contrast')],replicates=2000)
pd.DataFrame(ci).to_csv(out/'paired_intervals.csv',index=False)
(out/'complete.json').write_text(json.dumps(dict(status='complete',source_only=True,mask_inputs=False,
    primary_comparison='linear_contrast versus linear_D',other_variants='exploratory',
    protocol='Fixed Gaussian sigma 4/16 on 512 cached maps; normal-only calibration; source category LOCO Logistic C=1.',
    caveat='Known center-surround operation and simple regression controls; not a novel algorithm. No target evaluation.'),indent=2))
print(macro.to_string());print(pd.DataFrame(ci).to_string(index=False))
