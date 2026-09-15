"""Additional matched-model control after the negative exchange pilot.

No changes to the completed pilot's scores, hyperparameters or target protocol.
"""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score
from evaluate_exchange import GEOMETRY,fit_logistic,intervals

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--evaluation',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True); args=ap.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    frame=pd.read_csv(args.evaluation/'source_oof_predictions.csv')
    parts=[]
    for category in sorted(frame.category.unique()):
        train=frame[frame.category!=category]; val=frame[frame.category==category].copy()
        val['logistic_geometry']=fit_logistic(train,val,GEOMETRY)
        parts.append(val)
    oof=pd.concat(parts,ignore_index=True)
    oof[['image_id','category','label','logistic_geometry']].to_csv(args.out/'geometry_oof_predictions.csv',index=False)
    cols=['logistic_geometry','logistic_reference','logistic_exchange','visual_distance_z','reference_contrast_z']
    metrics=[]
    for category,g in oof.groupby('category'):
        for col in cols:
            metrics.append(dict(category=category,method=col,auroc=roc_auc_score(g.label,g[col]),ap=average_precision_score(g.label,g[col])))
    metrics=pd.DataFrame(metrics); metrics.to_csv(args.out/'category_metrics.csv',index=False)
    macros=metrics.groupby('method')[['auroc','ap']].mean(); macros.to_csv(args.out/'macro_metrics.csv')
    comparisons=[('logistic_reference','logistic_geometry'),('logistic_exchange','logistic_geometry'),
                 ('logistic_geometry','fixed_source_oof')]
    ci=intervals(oof,comparisons,replicates=2000); pd.DataFrame(ci).to_csv(args.out/'paired_intervals.csv',index=False)
    (args.out/'protocol.json').write_text(json.dumps(dict(status='post-pilot diagnostic',source_only=True,
        C=1,features=GEOMETRY,validation='same full-category LOCO',target_evaluated=False),indent=2))
    print(macros.to_string()); print(pd.DataFrame(ci).to_string(index=False))

if __name__=='__main__': main()
