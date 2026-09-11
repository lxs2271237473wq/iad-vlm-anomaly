"""Freeze the source-only geometry Logistic control and evaluate its AD2 transfer."""
import argparse,json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,average_precision_score
from evaluate_exchange import GEOMETRY
from baseline_intervals import group_weights,matrix
from common import sha256

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--run',type=Path,required=True)
    ap.add_argument('--source-evaluation',type=Path,required=True); ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args(); args.out.mkdir(parents=True,exist_ok=False)
    source=pd.read_csv(args.source_evaluation/'source_oof_predictions.csv')
    assert len(source)==2162
    scaler=StandardScaler().fit(source[GEOMETRY])
    weight=1/source.groupby(['category','label']).image_id.transform('count').to_numpy(); weight/=weight.mean()
    model=LogisticRegression(C=1,max_iter=1000,random_state=42).fit(scaler.transform(source[GEOMETRY]),source.label,sample_weight=weight)
    protocol=dict(source='VisA',target='AD2',C=1,features=GEOMETRY,
       selection='source-only category LOCO; geometry control; no exchange/reference inputs',
       source_file_sha256=sha256(args.source_evaluation/'source_oof_predictions.csv'),code_sha256=sha256(__file__),
       coefficients=model.coef_.tolist(),intercept=model.intercept_.tolist(),scaler_mean=scaler.mean_.tolist(),scaler_scale=scaler.scale_.tolist())
    (args.out/'frozen_protocol.json').write_text(json.dumps(protocol,indent=2))
    joblib.dump(dict(scaler=scaler,model=model),args.out/'geometry_logistic.joblib')
    target=pd.read_csv(args.run/'simple_fusion/target_predictions_no_labels.csv')
    target['disagreement']=(target.D_z-target.M_context_top1_z).abs()
    geometry=[]
    for path in sorted((args.run/'baselines/AD2').glob('*/regions.jsonl')):
        for row in map(json.loads,path.read_text().splitlines()):
            fraction=0.; occupancy=0.
            if row['regions']:
                r=row['regions'][0]; x1,y1,x2,y2=r['box']; cx1,cy1,cx2,cy2=r['context_box']
                area=(x2-x1)*(y2-y1); fraction=area/(row['width']*row['height']); occupancy=area/((cx2-cx1)*(cy2-cy1))
            geometry.append(dict(image_id=row['image_id'],log_roi_fraction=float(np.log(max(fraction,1e-8))),roi_context_occupancy=occupancy))
    target=target.merge(pd.DataFrame(geometry),on='image_id',validate='one_to_one'); assert len(target)==1084
    target['logistic_geometry']=model.decision_function(scaler.transform(target[GEOMETRY]))
    target.to_csv(args.out/'predictions_no_labels.csv',index=False)
    labels=pd.read_csv(args.run/'manifest/evaluation_labels.csv')
    target=target.merge(labels,on='image_id',validate='one_to_one')
    methods=['D_z','fixed_source','logistic_source','logistic_geometry']
    rows=[]; caches=[]; rng=np.random.default_rng(42)
    for category,g in target.groupby('category'):
        for method in methods: rows.append(dict(category=category,method=method,auroc=roc_auc_score(g.label,g[method]),ap=average_precision_score(g.label,g[method])))
        cp=group_weights(g[g.label==1].scene_group,rng,5000); cn=group_weights(g[g.label==0].scene_group,rng,5000)
        caches.append((g,cp,cn))
    table=pd.DataFrame(rows); table.to_csv(args.out/'category_metrics.csv',index=False)
    macro=table.groupby('method')[['auroc','ap']].mean(); macro.to_csv(args.out/'macro_metrics.csv')
    comparisons=[]
    for baseline in ['D_z','fixed_source','logistic_source']:
        points=[]; samples=[]
        for g,cp,cn in caches:
            diff=matrix(g,'logistic_geometry')-matrix(g,baseline)
            points.append(diff.mean()); samples.append(np.einsum('bi,ij,bj->b',cp,diff,cn,optimize=True))
        bootstrap=np.stack(samples).mean(0)
        comparisons.append(dict(method='logistic_geometry',baseline=baseline,delta_macro_auroc=float(np.mean(points)),
           lower95=float(np.quantile(bootstrap,.025)),upper95=float(np.quantile(bootstrap,.975))))
    pd.DataFrame(comparisons).to_csv(args.out/'paired_intervals.csv',index=False)
    print(macro.to_string());print(pd.DataFrame(comparisons).to_string(index=False))

if __name__=='__main__': main()
