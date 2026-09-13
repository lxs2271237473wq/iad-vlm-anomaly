"""Source-only controls, normal-relative appearance; anomaly channels intact."""
import gc,json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score,average_precision_score
from local_learning import RUN,features,fit_model
from local_learning_robustness import dense_scores,SEEDS
from audit_normal_shift import transform
from common import sha256
from evaluate_exchange import intervals

OUT=RUN/'20260911_n'

def relative(x,reference,kind):
    result=x.copy()
    sub_reference=(reference[0][4:],reference[1][4:],reference[2][:,4:])
    normalized,ranked=transform(x[:,4:],sub_reference)
    result[:,4:]=ranked if kind=='percentile' else normalized
    assert np.array_equal(result[:,:4],x[:,:4]) and np.isfinite(result).all()
    return result.astype(np.float32)

def main():
    OUT.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4)
    ref_path=RUN/'20260911_m/normal_references.json';references=json.loads(ref_path.read_text())
    refs={key.split('/')[1]:tuple(np.array(value[n]) for n in ['median','scale','quantile_knots']) for key,value in references.items() if key.startswith('VisA/') and key.endswith('/uniform')}
    assert len(refs)==12
    data=np.load(RUN/'20260911_j/training_samples.npz');x=data['x'];y=data['y'];cats=data['category'];ids=data['image_id']
    assert json.loads((RUN/'20260911_j/cache_signature.json').read_text())['code_sha256']==sha256(Path(__file__).parent/'local_learning.py')
    transformed={}
    for kind in ['percentile','robust']:
        value=x.copy()
        for category in refs:
            idx=cats==category;value[idx]=relative(x[idx],refs[category],kind)
        transformed[kind]=value
    manifest=pd.read_csv(RUN/'20260910_a/manifest/images.csv')
    source=manifest[(manifest.dataset=='VisA')&(manifest.split=='evaluation')];assert len(source)==2162
    signature=dict(code=sha256(Path(__file__)),normal_reference=sha256(ref_path))
    parts=[]
    for category,g in source.groupby('category',sort=True):
        dest=OUT/category;dest.mkdir(exist_ok=True)
        if (dest/'complete.json').exists():
            assert json.loads((dest/'complete.json').read_text())['signature']==signature
            parts.append(pd.read_csv(dest/'predictions.csv'));continue
        train=cats!=category;assert not set(ids[train])&set(g.image_id)
        meta=pd.DataFrame(dict(category=cats[train],image_id=ids[train],label=y[train]))
        weights=1/(meta.groupby(['category','label','image_id']).image_id.transform('size').to_numpy()*meta.groupby(['category','label']).image_id.transform('nunique').to_numpy());weights/=weights.mean()
        models={};scalers={};training={}
        for kind in transformed:
            value=transformed[kind][train];mean=value.mean(0);scale=np.maximum(value.std(0),1e-6);scalers[kind]=(mean,scale)
            np.savez(dest/f'{kind}_scaler.npz',mean=mean,scale=scale)
            models[kind]={}
            for seed in SEEDS:
                name=f'{kind}_{seed}';model=fit_model((value-mean)/scale,y[train],weights,'mlp',seed)
                models[kind][name]=(model,torch.ones(7,device='cuda'));training[name]=model.training_report
                torch.save(model.state_dict(),dest/f'{name}.pt')
        rows=[]
        for r in g.sort_values('image_id').itertuples():
            fx=features(r);prediction={}
            for kind in transformed:
                prediction.update(dense_scores(models[kind],relative(fx,refs[category],kind),*scalers[kind]))
            rows.append(dict(image_id=r.image_id,category=category,**prediction))
        f=pd.DataFrame(rows);f.to_csv(dest/'predictions.csv',index=False);parts.append(f)
        (dest/'complete.json').write_text(json.dumps(dict(signature=signature,training=training,heldout=category,unchanged_channels=[0,1,2,3]),indent=2))
        del models;gc.collect();torch.cuda.empty_cache();print('COMPLETE',category,flush=True)
    old=pd.read_csv(RUN/'20260911_k/oof_predictions.csv')
    f=pd.concat(parts).merge(old[['image_id','label','D_z']+[f'full_{s}' for s in SEEDS]],on='image_id',validate='one_to_one');assert len(f)==2162
    f.to_csv(OUT/'oof_predictions.csv',index=False)
    methods=['D_z']+[f'{v}_{s}' for v in ['full','percentile','robust'] for s in SEEDS];rows=[]
    for category,g in f.groupby('category'):
        for method in methods:rows.append(dict(category=category,method=method,auroc=roc_auc_score(g.label,g[method]),ap=average_precision_score(g.label,g[method])))
    table=pd.DataFrame(rows);table.to_csv(OUT/'category_metrics.csv',index=False)
    macro=table.groupby('method')[['auroc','ap']].mean();macro.to_csv(OUT/'macro_metrics.csv')
    comparisons=[(f'{v}_{s}',f'full_{s}') for v in ['percentile','robust'] for s in SEEDS]
    pd.DataFrame(intervals(f,comparisons,replicates=2000)).to_csv(OUT/'paired_intervals.csv',index=False)
    (OUT/'complete.json').write_text(json.dumps(dict(status='complete',source_only=True,seeds=SEEDS,
        primary='appearance percentile versus raw full model',secondary='robust median/MAD appearance control',
        protocol='Same source pixel samples, same MLP; source normal fit images only for references; no target evaluation.',
        caveats='Head seeds only; fixed masks/backbone; quantile mapping clips appearance tails; normal marginal alignment is not innovation or proof of detection gain.'),indent=2))
    print(macro.to_string())

if __name__=='__main__':main()
