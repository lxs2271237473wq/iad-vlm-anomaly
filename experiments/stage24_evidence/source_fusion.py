"""Strong simple fusion controls. All parameter choices use source categories only."""
import argparse,json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score,average_precision_score
from common import sha256

FEATURES=['D_z','M_context_top1_z']
GRID=np.linspace(0,1,11)
CS=[.1,1.,10.]

def macro(frame,score):
    temp=frame.assign(score=np.asarray(score))
    return float(np.mean([roc_auc_score(g.label,g.score) for _,g in temp.groupby('category')]))

def weights(frame):
    counts=frame.groupby(['category','label']).image_id.transform('count')
    w=1/counts.to_numpy(); return w/w.mean()

def fit(frame,c):
    model=make_pipeline(StandardScaler(),LogisticRegression(C=c,max_iter=1000,random_state=42))
    model.fit(frame[FEATURES],frame.label,logisticregression__sample_weight=weights(frame))
    return model

def choose_c(frame):
    values=[]
    for c in CS:
        scores=[]
        for category in sorted(frame.category.unique()):
            train=frame[frame.category!=category]; val=frame[frame.category==category]
            model=fit(train,c)
            scores.append(roc_auc_score(val.label,model.decision_function(val[FEATURES])))
        values.append(float(np.mean(scores)))
    return CS[int(np.argmax(values))],dict(zip(map(str,CS),values))

def choose_alpha(frame):
    scores=[macro(frame,(1-a)*frame.D_z+a*frame.M_context_top1_z) for a in GRID]
    # np.argmax provides a fixed tie rule (smaller VLM weight).
    return float(GRID[int(np.argmax(scores))]),scores

def read_dataset(base,dataset,count):
    markers=sorted((base/dataset).glob('*/complete.json'))
    assert len(markers)==count,f'{dataset}: {len(markers)}/{count} complete'
    frames=[]
    for marker in markers:
        status=json.loads(marker.read_text()); assert status['status']=='baseline_complete'
        path=marker.parent/'predictions.csv'
        assert sha256(path)==status['predictions_sha256']
        frames.append(pd.read_csv(path))
    all_rows=pd.concat(frames,ignore_index=True)
    assert all_rows.image_id.is_unique
    return all_rows[all_rows.split=='evaluation'].copy()

def report(frame,columns):
    rows=[]
    for category,g in frame.groupby('category'):
        for field in columns:
            rows.append(dict(category=category,method=field,n=len(g),
                auroc=roc_auc_score(g.label,g[field]),ap=average_precision_score(g.label,g[field])))
    return pd.DataFrame(rows)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--base',type=Path,required=True)
    ap.add_argument('--manifest',type=Path,required=True); ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args(); args.out.mkdir(parents=True,exist_ok=False)
    source=read_dataset(args.base,'VisA',12)
    labels=pd.read_csv(args.manifest/'evaluation_labels.csv')
    source=source.merge(labels,on='image_id',validate='one_to_one')
    assert len(source)==2162
    folds=[]; oof=[]
    for category in sorted(source.category.unique()):
        train=source[source.category!=category]; val=source[source.category==category].copy()
        alpha,_=choose_alpha(train)
        c,inner=choose_c(train)
        model=fit(train,c)
        val['fixed_source_oof']=(1-alpha)*val.D_z+alpha*val.M_context_top1_z
        val['logistic_source_oof']=model.decision_function(val[FEATURES])
        oof.append(val)
        folds.append(dict(heldout_category=category,alpha=alpha,c=c,inner_loco_scores=inner))
        print('SOURCE_FOLD',category,'alpha',alpha,'C',c,flush=True)
    oof=pd.concat(oof,ignore_index=True)
    oof.to_csv(args.out/'source_oof_predictions.csv',index=False)
    report(oof,['D_z','M_context_top1_z','naive_context_top1','fixed_source_oof','logistic_source_oof']).to_csv(args.out/'source_oof_metrics.csv',index=False)
    alpha,alpha_scores=choose_alpha(source); c,c_scores=choose_c(source); model=fit(source,c)
    protocol=dict(source='VisA',target='AD2',features=FEATURES,alpha=alpha,c=c,
        alpha_grid=GRID.tolist(),alpha_source_selection_scores=alpha_scores,c_source_loco_scores=c_scores,
        outer_folds=folds,supervision='source anomaly supervision; target normal-only calibration',
        source_prediction_ids=sorted(source.image_id.tolist()),
        target_labels_used_for_selection=False,code_sha256=sha256(__file__))
    # Freeze before reading any target predictions or target labels into evaluation.
    (args.out/'frozen_simple_fusion.json').write_text(json.dumps(protocol,indent=2))
    joblib.dump(model,args.out/'source_logistic.joblib')
    target=read_dataset(args.base,'AD2',8); assert len(target)==1084
    target['fixed_source']=(1-alpha)*target.D_z+alpha*target.M_context_top1_z
    target['logistic_source']=model.decision_function(target[FEATURES])
    target.to_csv(args.out/'target_predictions_no_labels.csv',index=False)
    target=target.merge(labels,on='image_id',validate='one_to_one')
    metrics=report(target,['D_z','M_context_top1_z','naive_context_top1','fixed_source','logistic_source'])
    metrics.to_csv(args.out/'target_category_metrics.csv',index=False)
    metrics.groupby('method')[['auroc','ap']].mean().to_csv(args.out/'target_macro_metrics.csv')
    print(metrics.groupby('method')[['auroc','ap']].mean().to_string())

if __name__=='__main__': main()
