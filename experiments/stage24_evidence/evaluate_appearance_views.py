"""Source-only photometric aggregation controls; not a novelty claim.

All four transforms are fixed from the normal-only probe. No target fitting.
"""
import gc,json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image,ImageEnhance
from anomalib.models.image.patchcore.torch_model import PatchcoreModel
from sklearn.metrics import roc_auc_score,average_precision_score
from common import image_transform,robust_fit,robust_apply,sha256
from run_baselines import seed_all
from evaluate_exchange import intervals

root=Path('/root/private_data/iad-vlm-anomaly/results/stage24_evidence')
out=root/'20260911_g';out.mkdir(parents=True,exist_ok=True)
seed_all(42);torch.set_num_threads(4);transform=image_transform(512)
manifest=pd.read_csv(root/'20260910_a/manifest/images.csv');assert 'label' not in manifest
source=manifest[(manifest.dataset=='VisA')&(manifest.split.isin(['calibration','evaluation']))]
variant_names=['brightness08','brightness12','contrast08','contrast12']
all_parts=[]
for category,g in source.groupby('category',sort=True):
    dest=out/category;dest.mkdir(exist_ok=True)
    if (dest/'complete.json').exists():
        assert json.loads((dest/'complete.json').read_text())['code_sha256']==sha256(Path(__file__))
        all_parts.append(pd.read_csv(dest/'scores.csv'));continue
    base=root/'20260911_d/fixed_bank/VisA'/category
    old=pd.read_csv(base/'predictions.csv').set_index('image_id')
    model=PatchcoreModel(layers=['layer2','layer3'],backbone='wide_resnet50_2',pre_trained=True,num_neighbors=9).cuda().eval()
    model.memory_bank=torch.load(base/'memory_bank.pt',map_location='cuda',weights_only=True)
    records=[]
    with torch.inference_mode():
        for i,r in enumerate(g.sort_values('image_id').itertuples()):
            with Image.open(r.path) as handle:im=handle.convert('RGB')
            original=float(old.loc[r.image_id,'D_raw'])
            if i==0:
                check=float(model(transform(im)[None].cuda()).pred_score.item())
                assert np.isclose(check,original,rtol=1e-4,atol=1e-4)
            views=[ImageEnhance.Brightness(im).enhance(.8),ImageEnhance.Brightness(im).enhance(1.2),ImageEnhance.Contrast(im).enhance(.8),ImageEnhance.Contrast(im).enhance(1.2)]
            values=[float(model(transform(v)[None].cuda()).pred_score.item()) for v in views]
            assert np.isfinite(values).all()
            record=dict(image_id=r.image_id,category=category,split=r.split,identity=original,**dict(zip(variant_names,values)))
            scores=[original]+values
            record.update(view_mean=float(np.mean(scores)),view_median=float(np.median(scores)),view_min=float(np.min(scores)))
            records.append(record)
            if i%50==0:print('INFER',category,i+1,len(g),flush=True)
    f=pd.DataFrame(records);parameters={}
    methods=['identity']+variant_names+['view_mean','view_median','view_min']
    for method in methods:
        parameters[method]=robust_fit(f.loc[f.split=='calibration',method]);f[method+'_z']=robust_apply(f[method],parameters[method])
    f.to_csv(dest/'scores.csv',index=False)
    (dest/'complete.json').write_text(json.dumps(dict(code_sha256=sha256(Path(__file__)),normal_calibration=parameters,images=len(f)),indent=2))
    all_parts.append(f);del model;gc.collect();torch.cuda.empty_cache()
    print('COMPLETE',category,flush=True)
scores=pd.concat(all_parts);assert len(scores)==3026
scores.to_csv(out/'all_scores_no_labels.csv',index=False)
# Read evaluation labels only after all predictions have been produced.
labels=pd.read_csv(root/'20260910_a/manifest/evaluation_labels.csv')
f=scores[scores.split=='evaluation'].merge(labels,on='image_id',validate='one_to_one');assert len(f)==2162
rows=[]
methods=['identity']+variant_names+['view_mean','view_median','view_min']
for category,g in f.groupby('category'):
    for method in methods:rows.append(dict(category=category,method=method,auroc=roc_auc_score(g.label,g[method]),ap=average_precision_score(g.label,g[method])))
table=pd.DataFrame(rows);table.to_csv(out/'category_metrics.csv',index=False)
macro=table.groupby('method')[['auroc','ap']].mean();macro.to_csv(out/'macro_metrics.csv')
ci=intervals(f,[(name,'identity') for name in ['view_mean','view_median','view_min']],replicates=2000)
pd.DataFrame(ci).to_csv(out/'paired_intervals.csv',index=False)
(out/'complete.json').write_text(json.dumps(dict(status='complete',source_only=True,additional_views_per_image=4,
    primary_control='view_mean versus identity',other_aggregations='exploratory controls; no target selection',
    note='Known test-time augmentation controls. All results retained; no claim of a new algorithm.'),indent=2))
print(macro.to_string());print(pd.DataFrame(ci).to_string(index=False))
