"""Fixed-sample, frozen-backbone head-seed and feature ablation controls."""
import gc,json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.metrics import roc_auc_score,average_precision_score
from local_learning import features,fit_model,RUN
from common import sha256
from evaluate_exchange import intervals

OUT=RUN/'20260911_k'
SEEDS=[42,123,2026]
MASKS={'response':np.array([1,0,0,0,0,0,0],np.float32),'map':np.array([1,1,1,1,0,0,0],np.float32),'full':np.ones(7,np.float32)}

def convex_linear(x,y,w):
    x=x.astype(np.float64);y=y.astype(np.float64);w=w.astype(np.float64)
    def objective(beta):
        logits=x@beta[:7]+beta[7]
        loss=np.mean(w*(np.logaddexp(0,logits)-y*logits))+.0005*np.dot(beta[:7],beta[:7])
        residual=w*(expit(logits)-y)/len(y)
        grad=np.r_[x.T@residual+.001*beta[:7],residual.sum()]
        return loss,grad
    result=minimize(objective,np.zeros(8),method='L-BFGS-B',jac=True,options=dict(maxiter=500,ftol=1e-12,gtol=1e-7))
    _,grad=objective(result.x)
    report=dict(success=bool(result.success),iterations=int(result.nit),objective=float(result.fun),max_gradient=float(np.abs(grad).max()),message=str(result.message),l2=.001)
    assert result.success and report['max_gradient']<1e-5,report
    model=torch.nn.Linear(7,1).cuda()
    with torch.no_grad():model.weight.copy_(torch.tensor(result.x[:7],dtype=torch.float32)[None]);model.bias.copy_(torch.tensor(result.x[7:],dtype=torch.float32))
    return model.eval(),report

@torch.inference_mode()
def dense_scores(models,x,mean,scale):
    best={name:-np.inf for name in models}
    for start in range(0,len(x),32768):
        z=torch.as_tensor((x[start:start+32768]-mean)/scale,device='cuda')
        for name,(model,mask) in models.items():
            value=float(model(z*mask).max().item());assert np.isfinite(value)
            best[name]=max(best[name],value)
    return best

def main():
    OUT.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4)
    signature={n:sha256(Path(__file__).parent/n) for n in ['local_learning_robustness.py','local_learning.py','common.py']}
    old=RUN/'20260911_j';data=np.load(old/'training_samples.npz')
    assert json.loads((old/'cache_signature.json').read_text())['code_sha256']==sha256(Path(__file__).parent/'local_learning.py')
    x=data['x'];y=data['y'];cats=data['category'];ids=data['image_id']
    manifest=pd.read_csv(RUN/'20260910_a/manifest/images.csv')
    source=manifest[(manifest.dataset=='VisA')&(manifest.split=='evaluation')];assert len(source)==2162
    old_scores=pd.read_csv(old/'oof_predictions.csv').set_index('image_id')
    parts=[];convergence=[]
    for category,g in source.groupby('category',sort=True):
        dest=OUT/category;dest.mkdir(exist_ok=True)
        if (dest/'complete.json').exists():
            report=json.loads((dest/'complete.json').read_text());assert report['signature']==signature
            parts.append(pd.read_csv(dest/'predictions.csv'));convergence.append(dict(category=category,**report['convex_linear']));continue
        train=cats!=category;assert not set(ids[train])&set(g.image_id)
        mean=x[train].mean(0);scale=np.maximum(x[train].std(0),1e-6);tx=(x[train]-mean)/scale
        meta=pd.DataFrame(dict(category=cats[train],image_id=ids[train],label=y[train]))
        counts=meta.groupby(['category','label','image_id']).image_id.transform('size').to_numpy()
        images=meta.groupby(['category','label']).image_id.transform('nunique').to_numpy()
        w=1/(counts*images);w/=w.mean()
        linear,report=convex_linear(tx,y[train],w);convergence.append(dict(category=category,**report))
        models={'linear_convex':(linear,torch.ones(7,device='cuda'))};training={}
        for variant,mask in MASKS.items():
            for seed in SEEDS:
                name=f'{variant}_{seed}'
                if variant=='full' and seed==42:
                    model=torch.nn.Sequential(torch.nn.Linear(7,32),torch.nn.ReLU(),torch.nn.Linear(32,1)).cuda()
                    model.load_state_dict(torch.load(old/category/'mlp.pt',map_location='cuda',weights_only=True));model.eval()
                    training[name]={'reused_previous_seed42':True}
                else:
                    model=fit_model(tx*mask,y[train],w,'mlp',seed);training[name]=model.training_report
                models[name]=(model,torch.tensor(mask,device='cuda'))
                torch.save(model.state_dict(),dest/f'{name}.pt')
        torch.save(linear.state_dict(),dest/'linear_convex.pt');np.savez(dest/'scaler.npz',mean=mean,scale=scale)
        rows=[];max_difference=0.
        for r in g.sort_values('image_id').itertuples():
            prediction=dense_scores(models,features(r),mean,scale)
            difference=abs(prediction['full_42']-float(old_scores.loc[r.image_id,'mlp']));max_difference=max(max_difference,difference)
            assert difference<1e-4,(r.image_id,difference)
            rows.append(dict(image_id=r.image_id,category=category,**prediction))
        f=pd.DataFrame(rows);f.to_csv(dest/'predictions.csv',index=False);parts.append(f)
        (dest/'complete.json').write_text(json.dumps(dict(signature=signature,convex_linear=report,training=training,previous_prediction_max_difference=max_difference,heldout=category),indent=2))
        del models;gc.collect();torch.cuda.empty_cache();print('COMPLETE',category,report,flush=True)
    f=pd.concat(parts).merge(old_scores.reset_index()[['image_id','label','D_z','linear','linear_contrast']],on='image_id',validate='one_to_one');assert len(f)==2162
    f.to_csv(OUT/'oof_predictions.csv',index=False);pd.DataFrame(convergence).to_csv(OUT/'linear_convergence.csv',index=False)
    methods=['D_z','linear','linear_contrast','linear_convex']+[f'{v}_{s}' for v in MASKS for s in SEEDS]
    rows=[]
    for category,g in f.groupby('category'):
        for name in methods:rows.append(dict(category=category,method=name,auroc=roc_auc_score(g.label,g[name]),ap=average_precision_score(g.label,g[name])))
    table=pd.DataFrame(rows);table.to_csv(OUT/'category_metrics.csv',index=False)
    macro=table.groupby('method')[['auroc','ap']].mean();macro.to_csv(OUT/'macro_metrics.csv')
    comparisons=[('linear_convex','linear')]+[(f'full_{s}',base) for s in SEEDS for base in ['linear_convex',f'map_{s}',f'response_{s}']]
    ci=intervals(f,comparisons,replicates=2000);pd.DataFrame(ci).to_csv(OUT/'paired_intervals.csv',index=False)
    seed_summary=[]
    for variant in MASKS:
        a=macro.loc[[f'{variant}_{s}' for s in SEEDS],'auroc']
        seed_summary.append(dict(variant=variant,mean=float(a.mean()),std=float(a.std(ddof=1)),minimum=float(a.min()),maximum=float(a.max())))
    pd.DataFrame(seed_summary).to_csv(OUT/'seed_summary.csv',index=False)
    (OUT/'complete.json').write_text(json.dumps(dict(status='complete',seeds=SEEDS,source_only=True,
        primary='full versus map-only and converged linear, separately for every seed',
        caveats=['Fixed original sampling and backbone/coreset seed 42; varies head initialization and minibatch ordering only.',
        'Identical 7-32-1 architecture, unused standardized inputs zeroed; effective input dimensionality differs.',
        'Convex linear uses explicit weight-only L2; objective differs from AdamW decoupled decay.',
        'Pixel-supervised category transfer, not official unsupervised VisA protocol.',
        'Fixed seedwise paired intervals, no multiple-comparison adjustment or target evaluation.']),indent=2))
    print(macro.to_string());print(pd.DataFrame(seed_summary).to_string(index=False))

if __name__=='__main__':main()
