"""Normal-only descriptive shift audit; no evaluation label/mask reads.

Relative feature transforms fit one half of selected calibration images and
are assessed on disjoint images. Distribution alignment is not detection gain.
"""
import json
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp,wasserstein_distance
from transfer_local_heads import extract
from local_learning import NAMES,RUN
from common import sha256

OUT=RUN/'20260911_m'

def fit_reference(x):
    center=np.median(x,axis=0);scale=np.maximum(1.4826*np.median(np.abs(x-center),axis=0),1e-6)
    knots=np.quantile(x,np.linspace(0,1,257),axis=0)
    return center,scale,knots

def transform(x,reference):
    center,scale,knots=reference
    robust=(x-center)/scale
    percentile=np.empty_like(x)
    for j in range(x.shape[1]):
        values,indices=np.unique(knots[:,j],return_index=True)
        percentile[:,j]=np.interp(x[:,j],values,indices/256,left=0,right=1)
    return robust,percentile

def main():
    OUT.mkdir(parents=True,exist_ok=True);cv2.setNumThreads(1)
    manifest=pd.read_csv(RUN/'20260910_a/manifest/images.csv');assert 'label' not in manifest
    f=manifest[(manifest.dataset.isin(['VisA','AD2']))&(manifest.split=='calibration')]
    old=np.load(RUN/'20260911_l/scaler.npz');rng=np.random.default_rng(42)
    selected=[];arrays=[];meta=[]
    for (dataset,category),g in f.groupby(['dataset','category'],sort=True):
        g=g.sort_values('image_id').head(16);assert len(g)==16
        stage='20260911_d' if dataset=='VisA' else '20260911_e'
        for i,row in enumerate(g.itertuples()):
            fold='fit' if i%2==0 else 'check';selected.append(dict(image_id=row.image_id,dataset=dataset,category=category,role=fold))
            x=extract(row,RUN/stage/'fixed_bank'/dataset/category)
            for stratum,idx in [('uniform',rng.choice(len(x),1024,replace=False)),('high_response',np.argpartition(x[:,0],-128)[-128:])]:
                arrays.append(x[idx]);meta.append(dict(dataset=dataset,category=category,role=fold,stratum=stratum,image_id=row.image_id,n=len(idx)))
        print('EXTRACTED',dataset,category,flush=True)
    pd.DataFrame(selected).to_csv(OUT/'selected_normal_images.csv',index=False)
    features=np.concatenate(arrays);metadata=pd.DataFrame([row for row in meta for _ in range(row['n'])])
    assert len(metadata)==len(features) and len(selected)==320
    np.savez_compressed(OUT/'sampled_normal_features.npz',features=features,**{c:metadata[c].to_numpy() for c in ['dataset','category','role','stratum','image_id']})
    summary=[];domain_parts={};references={}
    for (dataset,category,stratum),idx in metadata.groupby(['dataset','category','stratum']).indices.items():
        m=metadata.iloc[idx];x=features[idx];fit=x[m.role.to_numpy()=='fit'];check=x[m.role.to_numpy()=='check']
        assert not set(m.loc[m.role=='fit','image_id'])&set(m.loc[m.role=='check','image_id'])
        reference=fit_reference(fit);robust,percentile=transform(check,reference)
        references[f'{dataset}/{category}/{stratum}']={k:v.tolist() for k,v in zip(['median','scale','quantile_knots'],reference)}
        for kind,values in [('raw',check),('normal_robust',robust),('normal_percentile',percentile)]:
            domain_parts.setdefault((dataset,stratum,kind),[]).append(values)
        z=(check-old['mean'])/old['scale']
        for j,name in enumerate(NAMES):
            summary.append(dict(dataset=dataset,category=category,stratum=stratum,feature=name,
                fit_images=8,check_images=8,median=float(np.median(check[:,j])),q01=float(np.quantile(check[:,j],.01)),q99=float(np.quantile(check[:,j],.99)),
                fraction_outside_source_scaler_3sd=float((np.abs(z[:,j])>3).mean()),fraction_outside_source_scaler_5sd=float((np.abs(z[:,j])>5).mean())))
    pd.DataFrame(summary).to_csv(OUT/'category_feature_summary.csv',index=False)
    (OUT/'normal_references.json').write_text(json.dumps(references))
    rows=[]
    for stratum in ['uniform','high_response']:
        for kind in ['raw','normal_robust','normal_percentile']:
            source=np.concatenate(domain_parts[('VisA',stratum,kind)]);target=np.concatenate(domain_parts[('AD2',stratum,kind)])
            for j,name in enumerate(NAMES):
                rows.append(dict(stratum=stratum,representation=kind,feature=name,
                    ks_distance=float(ks_2samp(source[:,j],target[:,j],method='asymp').statistic),
                    wasserstein=float(wasserstein_distance(source[:,j],target[:,j]))))
    table=pd.DataFrame(rows);table.to_csv(OUT/'domain_distances.csv',index=False)
    (OUT/'complete.json').write_text(json.dumps(dict(status='complete',normal_images=320,source_normal_images=192,target_normal_images=128,
        fit_check_images_per_category=[8,8],labels_read=False,code_sha256=sha256(Path(__file__)),
        caveats=['Only predefined normal calibration images; no target evaluation images or labels.',
        'Equal image and category weights within each sampling stratum.',
        'Pixel dependence means no pixel-based significance p-values are reported.',
        'KS measures marginal distribution shift, not joint alignment or anomaly detection performance.',
        'Relative feature alignment can erase defect signal; no new model is trained or selected.',
        'Source-scaler tails reference the existing supervised training scaler, not a formal OOD detector.']),indent=2))
    print(table.to_string(index=False))

if __name__=='__main__':main()
