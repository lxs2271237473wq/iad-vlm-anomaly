"""One frozen AD2 comparison of both normal-relative appearance controls."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score,average_precision_score
from local_learning import RUN,fit_model
from local_learning_robustness import dense_scores,SEEDS
from relative_appearance_controls import relative
from transfer_local_heads import extract
from baseline_intervals import group_weights,matrix
from common import sha256

OUT=RUN/'20260911_o'

def main():
    OUT.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4)
    ref_path=RUN/'20260911_m/normal_references.json'
    references=json.loads(ref_path.read_text())
    refs={key.rsplit('/',1)[0]:tuple(np.asarray(value[n]) for n in ['median','scale','quantile_knots']) for key,value in references.items() if key.endswith('/uniform')}
    data=np.load(RUN/'20260911_j/training_samples.npz');x=data['x'];y=data['y'];cats=data['category']
    manifest=pd.read_csv(RUN/'20260910_a/manifest/images.csv');assert 'label' not in manifest
    target=manifest[(manifest.dataset=='AD2')&(manifest.split=='evaluation')];assert len(target)==1084
    assert not set(data['image_id'])&set(target.image_id)
    meta=pd.DataFrame(dict(category=cats,image_id=data['image_id'],label=y))
    counts=meta.groupby(['category','label','image_id']).image_id.transform('size').to_numpy()
    images=meta.groupby(['category','label']).image_id.transform('nunique').to_numpy()
    weights=1/(counts*images);weights/=weights.mean()
    models={};scalers={};training={}
    for kind in ['percentile','robust']:
        transformed=x.copy()
        for category in sorted(set(cats)):
            idx=cats==category;transformed[idx]=relative(x[idx],refs[f'VisA/{category}'],kind)
        mean=transformed.mean(0);scale=np.maximum(transformed.std(0),1e-6);scalers[kind]=(mean,scale)
        np.savez(OUT/f'{kind}_scaler.npz',mean=mean,scale=scale);models[kind]={}
        for seed in SEEDS:
            name=f'{kind}_{seed}';model=fit_model((transformed-mean)/scale,y,weights,'mlp',seed)
            models[kind][name]=(model,torch.ones(7,device='cuda'));training[name]=model.training_report
            torch.save(model.state_dict(),OUT/f'{name}.pt')
    (OUT/'frozen_config.json').write_text(json.dumps(dict(code_sha256=sha256(Path(__file__)),reference_sha256=sha256(ref_path),
        sample_cache_sha256=sha256(RUN/'20260911_j/training_samples.npz'),seeds=SEEDS,training=training,
        primary='percentile versus D512 and same-seed map-only control',secondary='robust normal appearance',
        target_adaptation='Previously fixed 8 normal fit images per category; no refitting or hyperparameter change.',
        target_labels_used_for_training=False),indent=2))
    print('SOURCE_HEADS_FROZEN',flush=True)
    records=[]
    for category,g in target.groupby('category',sort=True):
        for row in g.sort_values('image_id').itertuples():
            fx=extract(row,RUN/'20260911_e/fixed_bank/AD2'/category);prediction={}
            for kind in models:
                prediction.update(dense_scores(models[kind],relative(fx,refs[f'AD2/{category}'],kind),*scalers[kind]))
            records.append(dict(image_id=row.image_id,category=category,**prediction))
        pd.DataFrame(records).to_csv(OUT/'predictions.partial.csv',index=False);print('TARGET_COMPLETE',category,flush=True)
    pred=pd.DataFrame(records);pred.to_csv(OUT/'predictions_no_labels.csv',index=False)
    # Evaluation begins only after every target prediction has been saved.
    base=pd.read_csv(RUN/'20260911_e/evaluation/predictions_with_evaluation_labels.csv')
    previous=pd.read_csv(RUN/'20260911_l/predictions_no_labels.csv')
    old_methods=[f'{variant}_{seed}' for variant in ['full','map'] for seed in SEEDS]
    f=pred.merge(base[['image_id','D512','label','scene_group']],on='image_id',validate='one_to_one').merge(previous[['image_id']+old_methods],on='image_id',validate='one_to_one')
    assert len(f)==1084 and f.category.nunique()==8
    f.to_csv(OUT/'evaluation_predictions.csv',index=False)
    methods=['D512']+old_methods+[f'{variant}_{seed}' for variant in models for seed in SEEDS];rows=[]
    for category,g in f.groupby('category'):
        for method in methods:rows.append(dict(category=category,method=method,auroc=roc_auc_score(g.label,g[method]),ap=average_precision_score(g.label,g[method])))
    table=pd.DataFrame(rows);table.to_csv(OUT/'category_metrics.csv',index=False)
    macro=table.groupby('method')[['auroc','ap']].mean();macro.to_csv(OUT/'macro_metrics.csv')
    rng=np.random.default_rng(42);cache=[]
    for _,g in f.groupby('category'):
        cache.append((g,group_weights(g.loc[g.label==1,'scene_group'],rng,5000),group_weights(g.loc[g.label==0,'scene_group'],rng,5000)))
    comparisons=[]
    for variant in models:
        for seed in SEEDS:
            method=f'{variant}_{seed}'
            for baseline in ['D512',f'map_{seed}',f'full_{seed}']:
                points=[];draws=[]
                for g,cp,cn in cache:
                    delta=matrix(g,method)-matrix(g,baseline);points.append(float(delta.mean()));draws.append(np.einsum('bi,ij,bj->b',cp,delta,cn,optimize=True))
                sample=np.stack(draws).mean(0)
                comparisons.append(dict(method=method,baseline=baseline,delta_macro_auroc=float(np.mean(points)),lower95=float(np.quantile(sample,.025)),upper95=float(np.quantile(sample,.975)),categories_improved=sum(p>0 for p in points)))
    pd.DataFrame(comparisons).to_csv(OUT/'paired_intervals.csv',index=False)
    rows=[]
    for variant in ['full','map','percentile','robust']:
        values=macro.loc[[f'{variant}_{s}' for s in SEEDS],'auroc'];rows.append(dict(variant=variant,mean=float(values.mean()),std=float(values.std()),minimum=float(values.min()),maximum=float(values.max())))
    pd.DataFrame(rows).to_csv(OUT/'seed_summary.csv',index=False)
    (OUT/'complete.json').write_text(json.dumps(dict(status='complete',target_images=1084,target_categories=8,
        caveats='Source pixel supervision; fixed backbone/samples; head seeds only; previously inspected public target; provisional scene bootstrap; no seed selection or multiple-testing correction.'),indent=2))
    print(macro.to_string());print(pd.DataFrame(comparisons).to_string(index=False))

if __name__=='__main__':main()
