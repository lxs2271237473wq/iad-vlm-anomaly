"""Fixed-hyperparameter source category-LOCO gate pilot, no AD2 evaluation.

Bootstrap intervals condition on these fitted OOF models and the 12 source
categories; they do not measure all training uncertainty or prove target gains.
"""
import argparse,json,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,average_precision_score
from common import sha256

DM=['D_z','M_context_top1_z','disagreement']
GEOMETRY=DM+['log_roi_fraction','roi_context_occupancy','num_candidates','fallback_global']
REF=GEOMETRY+['reference_contrast_z','visual_distance_z','match_error_z','evidence_available']
EXTRA=['u_mean_z','v_mean_z','u_std_z','v_std_z']
VARIANTS={
    'gate_dm':DM,
    'gate_geometry':GEOMETRY,
    'gate_reference':REF,
    'gate_remove':REF+['u_mean_z','u_std_z'],
    'gate_exchange':REF+EXTRA,
    'gate_uncalibrated':REF+['u_mean','v_mean','u_std','v_std'],
    'gate_budget_pool':REF+['swap_margin_mean_z'],
    'gate_shuffled':REF+[f'{x}_shuffled' for x in EXTRA],
}

def pair_indices(frame):
    pos=[]; neg=[]; weight=[]
    for _,idx in frame.groupby('category').indices.items():
        y=frame.iloc[idx].label.to_numpy()
        p=np.asarray(idx)[y==1]; n=np.asarray(idx)[y==0]
        assert len(p) and len(n)
        pos.extend(np.repeat(p,len(n))); neg.extend(np.tile(n,len(p)))
        weight.extend(np.full(len(p)*len(n),1/(len(p)*len(n))))
    w=np.asarray(weight,dtype=np.float32); w/=w.sum()
    return torch.tensor(pos,dtype=torch.long),torch.tensor(neg,dtype=torch.long),torch.tensor(w)

def fit_gate(train,val,columns,epochs=150):
    scaler=StandardScaler().fit(train[columns])
    x=torch.tensor(scaler.transform(train[columns]),dtype=torch.float32)
    xv=torch.tensor(scaler.transform(val[columns]),dtype=torch.float32)
    d=torch.tensor(train.D_z.to_numpy(),dtype=torch.float32)
    m=torch.tensor(train.M_context_top1_z.to_numpy(),dtype=torch.float32)
    p,n,pw=pair_indices(train)
    layer=torch.nn.Linear(len(columns),1)
    torch.nn.init.zeros_(layer.weight); torch.nn.init.zeros_(layer.bias)
    optimizer=torch.optim.Adam(layer.parameters(),lr=.03)
    for epoch in range(epochs):
        optimizer.zero_grad()
        alpha=layer(x).squeeze(1).sigmoid()
        score=d+alpha*(m-d)
        loss=(F.softplus(-(score[p]-score[n]))*pw).sum()+.01*layer.weight.square().mean()
        if not torch.isfinite(loss): raise ValueError('Non-finite ranking loss')
        loss.backward(); optimizer.step()
    with torch.no_grad():
        alpha=layer(xv).squeeze(1).sigmoid().numpy()
    result=val.D_z.to_numpy()+alpha*(val.M_context_top1_z.to_numpy()-val.D_z.to_numpy())
    state=dict(features=columns,scaler_mean=scaler.mean_.tolist(),scaler_scale=scaler.scale_.tolist(),
               weight=layer.weight.detach().numpy().tolist(),bias=layer.bias.detach().numpy().tolist(),
               loss=float(loss.detach()),alpha_mean=float(alpha.mean()),alpha_min=float(alpha.min()),alpha_max=float(alpha.max()))
    return result,alpha,state

def fit_logistic(train,val,columns):
    scaler=StandardScaler().fit(train[columns])
    w=1/train.groupby(['category','label']).image_id.transform('count').to_numpy(); w/=w.mean()
    model=LogisticRegression(C=1,max_iter=1000,random_state=42)
    model.fit(scaler.transform(train[columns]),train.label,sample_weight=w)
    return model.decision_function(scaler.transform(val[columns]))

def auc_matrix(group,score):
    p=group.loc[group.label==1,score].to_numpy(); n=group.loc[group.label==0,score].to_numpy()
    diff=p[:,None]-n[None,:]
    return (diff>0).astype(np.float32)+.5*(diff==0)

def intervals(oof,comparisons,replicates=2000):
    rng=np.random.default_rng(42); caches=[]
    for category,g in oof.groupby('category'):
        np_,nn=int(g.label.sum()),int((g.label==0).sum())
        cp=rng.multinomial(np_,np.full(np_,1/np_),size=replicates).astype(np.float32)/np_
        cn=rng.multinomial(nn,np.full(nn,1/nn),size=replicates).astype(np.float32)/nn
        caches.append((g,cp,cn))
    output=[]
    for method,baseline in comparisons:
        draws=[]; points=[]; wins=0
        for g,cp,cn in caches:
            delta=auc_matrix(g,method)-auc_matrix(g,baseline)
            points.append(float(delta.mean())); wins+=int(delta.mean()>0)
            draws.append(np.einsum('bi,ij,bj->b',cp,delta,cn,optimize=True))
        sample=np.stack(draws).mean(0)
        output.append(dict(method=method,baseline=baseline,delta_macro_auroc=float(np.mean(points)),
            lower95=float(np.quantile(sample,.025)),upper95=float(np.quantile(sample,.975)),
            categories_improved=wins,categories=len(caches),bootstrap_replicates=replicates))
    return output

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--run',type=Path,required=True)
    ap.add_argument('--evidence',type=Path,required=True); ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args(); args.out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1); torch.manual_seed(42)
    complete=json.loads((args.evidence/'complete.json').read_text())
    assert complete=={'categories':12,'smoke':False},complete
    evidence=pd.read_csv(args.evidence/'all_source_evidence.csv')
    baseline=pd.read_csv(args.run/'simple_fusion/source_oof_predictions.csv')
    evidence=evidence[evidence.split=='evaluation']
    frame=baseline.merge(evidence.drop(columns=['category','split']),on='image_id',validate='one_to_one')
    assert len(frame)==2162 and frame.category.nunique()==12
    frame['disagreement']=(frame.D_z-frame.M_context_top1_z).abs()
    geometry=[]
    for category in sorted(frame.category.unique()):
        path=args.run/'baselines/VisA'/category/'regions.jsonl'
        for row in map(json.loads,path.read_text().splitlines()):
            fraction=0.; occupancy=0.
            if row['regions']:
                region=row['regions'][0]; x1,y1,x2,y2=region['box']; cx1,cy1,cx2,cy2=region['context_box']
                area=(x2-x1)*(y2-y1)
                fraction=area/(row['width']*row['height']); occupancy=area/((cx2-cx1)*(cy2-cy1))
            geometry.append(dict(image_id=row['image_id'],log_roi_fraction=float(np.log(max(fraction,1e-8))),roi_context_occupancy=occupancy))
    frame=frame.merge(pd.DataFrame(geometry),on='image_id',validate='one_to_one')
    assert len(frame)==2162
    for col in EXTRA: frame[col+'_shuffled']=0.
    rng=np.random.default_rng(42)
    for category,idx in frame.groupby('category').groups.items():
        order=rng.permutation(len(idx))
        frame.loc[idx,[f'{x}_shuffled' for x in EXTRA]]=frame.loc[idx,EXTRA].to_numpy()[order]
    protocol=dict(source='VisA',target_evaluated=False,epochs=150,learning_rate=.03,
        weight_regularization=.01,temperature=1.,initial_alpha=.5,seed=42,
        objective='category-balanced within-category pairwise softplus ranking loss',
        validation='leave one complete source category out; fixed hyperparameters for all variants',
        variants=VARIANTS,logistic_C=1,
        bootstrap='paired image resampling within category and label; conditional on fitted OOF models and fixed source categories',
        code_sha256=sha256(__file__),evidence_sha256=sha256(args.evidence/'all_source_evidence.csv'))
    (args.out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    oof=[]; models=[]; start=time.perf_counter()
    for category in sorted(frame.category.unique()):
        train=frame[frame.category!=category].reset_index(drop=True)
        val=frame[frame.category==category].copy()
        for name,cols in VARIANTS.items():
            val[name],val[name+'_alpha'],state=fit_gate(train,val,cols)
            models.append(dict(heldout=category,variant=name,**state))
        val['logistic_reference']=fit_logistic(train,val,REF)
        val['logistic_exchange']=fit_logistic(train,val,REF+EXTRA)
        oof.append(val)
        print('OOF',category,'seconds',round(time.perf_counter()-start,1),flush=True)
    oof=pd.concat(oof,ignore_index=True); oof.to_csv(args.out/'source_oof_predictions.csv',index=False)
    methods=['D_z','M_context_top1_z','naive_context_top1','fixed_source_oof','logistic_source_oof']+list(VARIANTS)+['logistic_reference','logistic_exchange']
    rows=[]
    for category,g in oof.groupby('category'):
        for method in methods:
            rows.append(dict(category=category,method=method,n=len(g),auroc=roc_auc_score(g.label,g[method]),ap=average_precision_score(g.label,g[method])))
    table=pd.DataFrame(rows); table.to_csv(args.out/'category_metrics.csv',index=False)
    table.groupby('method')[['auroc','ap']].mean().to_csv(args.out/'macro_metrics.csv')
    comparisons=[('gate_exchange',b) for b in ['D_z','fixed_source_oof','logistic_source_oof','gate_dm','gate_geometry','gate_reference','logistic_reference','gate_remove','gate_uncalibrated','gate_budget_pool','gate_shuffled']]
    comparisons += [('logistic_exchange','logistic_reference'),('gate_reference','fixed_source_oof')]
    ci=intervals(oof,comparisons); pd.DataFrame(ci).to_csv(args.out/'paired_intervals.csv',index=False)
    (args.out/'fold_models.json').write_text(json.dumps(models,indent=2))
    # No automatic target run: report the evidence for the research gate explicitly.
    key=[r for r in ci if r['method']=='gate_exchange' and r['baseline'] in ['fixed_source_oof','gate_reference','logistic_reference','gate_budget_pool']]
    decision=dict(source_exchange_increment_supported=all(r['lower95']>0 for r in key),
                  target_run_started=False,comparison_intervals=key,
                  note='This pilot cannot establish target generalization or publication-level novelty.')
    (args.out/'research_gate.json').write_text(json.dumps(decision,indent=2))
    (args.out/'complete.json').write_text(json.dumps(dict(status='complete',seconds=time.perf_counter()-start),indent=2))
    print(table.groupby('method')[['auroc','ap']].mean().to_string(),flush=True)
    print(json.dumps(decision,indent=2),flush=True)

if __name__=='__main__': main()
