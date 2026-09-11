"""Nested category selection of context/score branch, source only."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score
from evaluate_exchange import GEOMETRY,fit_logistic,intervals

BRANCHES=[f'{s}_{k}_z' for s in ['tight','context15','context30'] for k in ['text','visual']]

def features(frame,branch):
    f=frame.copy();f['M_context_top1_z']=f[branch]
    f['disagreement']=(f.D_z-f.M_context_top1_z).abs()
    return f

def predict(train,val,branch):
    return fit_logistic(features(train,branch),features(val,branch),GEOMETRY)

def select(train):
    scores=[]
    for branch in BRANCHES:
        auc=[]
        for category in sorted(train.category.unique()):
            tr=train[train.category!=category];va=train[train.category==category]
            auc.append(roc_auc_score(va.label,predict(tr,va,branch)))
        scores.append(float(np.mean(auc)))
    return BRANCHES[int(np.argmax(scores))],dict(zip(BRANCHES,scores))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--previous',type=Path,required=True)
    ap.add_argument('--scores',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    assert json.loads((args.scores/'complete.json').read_text())['categories']==12
    base=pd.read_csv(args.previous/'source_evaluation/source_oof_predictions.csv')
    geometry=pd.read_csv(args.previous/'geometry_control/geometry_oof_predictions.csv')[['image_id','logistic_geometry']]
    base=base.merge(geometry,on='image_id',validate='one_to_one')
    scores=pd.read_csv(args.scores/'all_source_scores.csv');scores=scores[scores.split=='evaluation']
    frame=base.merge(scores[['image_id']+BRANCHES],on='image_id',validate='one_to_one')
    assert len(frame)==2162
    folds=[];parts=[]
    for category in sorted(frame.category.unique()):
        train=frame[frame.category!=category];val=frame[frame.category==category].copy()
        selected,inner=select(train)
        for branch in BRANCHES:val['lr_'+branch]=predict(train,val,branch)
        val['nested_selected']=val['lr_'+selected]
        # Same branch and same training protocol must reproduce the geometry control.
        assert np.allclose(val['lr_context15_text_z'],val.logistic_geometry,atol=2e-3,rtol=2e-3)
        folds.append(dict(heldout=category,selected=selected,inner_scores=inner));parts.append(val)
        print('FOLD',category,selected,flush=True)
    oof=pd.concat(parts,ignore_index=True);oof.to_csv(args.out/'oof_predictions.csv',index=False)
    methods=['D_z','fixed_source_oof','logistic_geometry','nested_selected']+BRANCHES+['lr_'+b for b in BRANCHES]
    rows=[]
    for category,g in oof.groupby('category'):
        for method in methods:rows.append(dict(category=category,method=method,auroc=roc_auc_score(g.label,g[method]),ap=average_precision_score(g.label,g[method])))
    table=pd.DataFrame(rows);table.to_csv(args.out/'category_metrics.csv',index=False)
    macro=table.groupby('method')[['auroc','ap']].mean();macro.to_csv(args.out/'macro_metrics.csv')
    comparisons=[('nested_selected','logistic_geometry'),('nested_selected','D_z')]+[('lr_'+b,'logistic_geometry') for b in BRANCHES if b!='context15_text_z']
    ci=intervals(oof,comparisons,replicates=2000);pd.DataFrame(ci).to_csv(args.out/'paired_intervals.csv',index=False)
    chosen,global_cv=select(frame)
    (args.out/'source_selection.json').write_text(json.dumps(dict(selected=chosen,source_cv=global_cv,outer_folds=folds,
        target_evaluated=False,supervision='source image labels; target labels never used',C=1,features=GEOMETRY,
        note='Nested OOF is the primary selection estimate; best single-variant score is exploratory.'),indent=2))
    print(macro.to_string());print(pd.DataFrame(ci).to_string(index=False))
    (args.out/'complete.json').write_text('{"status":"complete"}')

if __name__=='__main__':main()
