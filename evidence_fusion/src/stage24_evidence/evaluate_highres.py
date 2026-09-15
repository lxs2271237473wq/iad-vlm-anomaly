"""Fixed source LOCO evaluation; AD2 is deliberately excluded."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score
from evaluate_exchange import GEOMETRY,fit_logistic,intervals

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--previous',type=Path,required=True)
    ap.add_argument('--high',type=Path,required=True);ap.add_argument('--residual',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    old=pd.read_csv(args.previous/'geometry_control/geometry_oof_predictions.csv')
    labels=old[['image_id','category','label','logistic_geometry']]
    assert labels.image_id.is_unique and len(labels)==2162
    residual=pd.read_csv(args.residual/'all_scores.csv')
    frame=residual[residual.split=='evaluation'].merge(labels,on=['image_id','category'],validate='one_to_one')
    assert len(frame)==2162
    rows=[];geo=[]
    for category in sorted(frame.category.unique()):
        path=args.high/'VisA'/category
        pred=pd.read_csv(path/'predictions.csv')
        rows.append(pred[['image_id','M_context_top1_z','num_candidates','fallback_global']])
        for record in map(json.loads,(path/'regions.jsonl').read_text().splitlines()):
            fraction=0.;occupancy=0.
            if record['regions']:
                r=record['regions'][0];x1,y1,x2,y2=r['box'];cx1,cy1,cx2,cy2=r['context_box']
                area=(x2-x1)*(y2-y1);fraction=area/(record['width']*record['height']);occupancy=area/((cx2-cx1)*(cy2-cy1))
            geo.append(dict(image_id=record['image_id'],log_roi_fraction=np.log(max(fraction,1e-8)),roi_context_occupancy=occupancy))
    frame=frame.merge(pd.concat(rows),on='image_id',validate='one_to_one').merge(pd.DataFrame(geo),on='image_id',validate='one_to_one')
    frame['D_z']=frame.D512_z;frame['disagreement']=(frame.D_z-frame.M_context_top1_z).abs()
    rng=np.random.default_rng(42)
    frame['shuffled_residual_z']=0.
    for _,idx in frame.groupby('category').groups.items():
        frame.loc[idx,'shuffled_residual_z']=rng.permutation(frame.loc[idx,'conditional_top1pct_z'].to_numpy())
    multi=GEOMETRY+['D256_z']
    variants=dict(geometry512=GEOMETRY,geometry_multiscale=multi,
        geometry_unconditional=multi+['unconditional_top1pct_z'],
        geometry_conditional=multi+['conditional_top1pct_z'],
        geometry_shuffled=multi+['shuffled_residual_z'])
    parts=[]
    for category in sorted(frame.category.unique()):
        tr=frame[frame.category!=category];va=frame[frame.category==category].copy()
        for name,columns in variants.items():va[name]=fit_logistic(tr,va,columns)
        parts.append(va)
    oof=pd.concat(parts);oof.to_csv(args.out/'oof_predictions.csv',index=False)
    methods=['D256_z','D512_z','fixed_multiscale','logistic_geometry','conditional_top1pct_z','unconditional_top1pct_z']+list(variants)
    rows=[]
    for category,g in oof.groupby('category'):
        for name in methods:rows.append(dict(category=category,method=name,auroc=roc_auc_score(g.label,g[name]),ap=average_precision_score(g.label,g[name])))
    table=pd.DataFrame(rows);table.to_csv(args.out/'category_metrics.csv',index=False)
    macro=table.groupby('method')[['auroc','ap']].mean();macro.to_csv(args.out/'macro_metrics.csv')
    comparisons=[('D512_z','D256_z'),('geometry512','logistic_geometry'),('geometry_conditional','geometry_multiscale'),
        ('geometry_conditional','geometry_unconditional'),('geometry_conditional','geometry_shuffled')]
    ci=pd.DataFrame(intervals(oof,comparisons,replicates=2000));ci.to_csv(args.out/'paired_intervals.csv',index=False)
    checks=ci[ci.method=='geometry_conditional']
    passed=bool(((checks.lower95>0)&(checks.delta_macro_auroc>=.01)&(checks.categories_improved>=8)).all())
    (args.out/'research_gate.json').write_text(json.dumps(dict(passed=passed,target_evaluated=False,
        rule='Exploratory triage: conditional branch +0.01 AUROC, CI lower>0, >=8/12 wins against all three controls.',
        limitations='Fixed OOF fits; not corrected for repeated research rounds. Passing requires further seeds and independent target evaluation.',
        variants=variants),indent=2))
    (args.out/'complete.json').write_text('{"status":"complete"}')
    print(macro.to_string());print(ci.to_string(index=False))

if __name__=='__main__':main()
