"""Diagnostic only: image/instance aggregation, no native AU-PRO claim."""
import argparse,csv,json
from pathlib import Path
import numpy as np

SCORES=['score_legacy_layer_independent','score_perview_four_layer_joint_independent','score_same_reference_relaxed','score_strict','score_shuffled','global_topk_soft_distance','local_topk_soft_distance']
def mean(x): return float(np.mean(x)) if len(x) else None
def main():
 p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 manifest=json.loads((a.run/'manifest.json').read_text());reports=[]
 for f in sorted(a.run.glob('*.json')):
  z=json.loads(f.read_text())
  if isinstance(z,dict) and 'output' in z and 'row' in z: reports.append(z)
 if not reports: raise RuntimeError('No scored images')
 if a.out.exists(): raise FileExistsError(a.out)
 a.out.mkdir(parents=True)
 vals=[np.load(a.run/r['output']) for r in reports if r['split']=='validation']
 if not vals: raise RuntimeError('No validation scores for independent calibration')
 thresholds={s:float(np.quantile(np.concatenate([v[s] for v in vals]),.95)) for s in SCORES}
 baseline='score_perview_four_layer_joint_independent'
 calibration=np.concatenate([v[baseline] for v in vals])
 bins=np.unique(np.quantile(calibration,np.linspace(0,1,11)))
 stats=[];matched=[];coverage=[]
 for r in reports:
  with np.load(a.run/r['output']) as z:
   ind=z[baseline];gap=z['score_strict']-ind;shuffled=z['score_shuffled']-z['score_strict']
   if gap.min() < -1e-5: raise RuntimeError('Independent<=joint violated')
   common=np.array([len(set(g).intersection(l))/len(g) for g,l in zip(z['global_topk_ids'],z['local_topk_ids'])])
   frac=z['labels_footprint_fraction'];center=z['labels_center'].astype(bool)
   groups={'pure_normal':frac==0,'any_defect_footprint':frac>0,'center_defect':center,'defect_majority':frac>=.5,'low_score_center_defect':center&(ind<=thresholds[baseline]),'low_score_any_defect':(frac>0)&(ind<=thresholds[baseline])}
   base=dict(category=r['row']['category'],split=r['split'],label=r['row']['label'],instance_id=r['row']['instance_id'],condition=r['row']['condition'],image_path=r['row']['image_path'])
   for group,sel in groups.items():
    item=dict(base,group=group,tokens=int(sel.sum()),gap_mean=mean(gap[sel]),true_pair_advantage_mean=mean(shuffled[sel]),topk_overlap_mean=mean(common[sel]))
    for s in SCORES:
     item[s+'_mean']=mean(z[s][sel]);item[s+'_above_validation95']=mean((z[s][sel]>thresholds[s]).astype(float))
    stats.append(item)
   # Matched raw-score deciles derived from validation, not anomalous labels.
   bi=np.searchsorted(bins[1:-1],ind,side='right')
   for b in range(len(bins)-1):
    for name,sel in [('pure_normal',frac==0),('center_defect',center),('any_defect_footprint',frac>0)]:
     sel=sel&(bi==b)
     matched.append(dict(base,score_bin=b,group=name,tokens=int(sel.sum()),gap_mean=mean(gap[sel]),true_pair_advantage_mean=mean(shuffled[sel]),topk_overlap_mean=mean(common[sel])))
   c=r['coverage'];components=c['components']
   coverage.append(dict(base,regions=len(components),regions_center_hit=sum(x['token_center_hits']>0 for x in components),regions_footprint_covered=sum(x['covered_pixels']>0 for x in components),gt_pixels=c['gt_pixels'],covered_gt_pixels=c['covered_gt_pixels'],encode_seconds=r['encode_seconds'],search_seconds=r['search_seconds']))
 def write(name,rows):
  with (a.out/name).open('w',newline='',encoding='utf-8') as f:
   w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 write('image_groups.csv',stats);write('matched_score_bins.csv',matched);write('coverage.csv',coverage)
 # Macro over conditions within physical instance, then report individual instances.
 ids=sorted({(r['category'],r['split'],str(r['label']),r['instance_id'],r['group']) for r in stats})
 inst=[]
 numeric=[k for k in stats[0] if k.endswith('_mean') or k.endswith('_above_validation95')]
 for cat,split,label,i,group in ids:
  rr=[r for r in stats if (r['category'],r['split'],str(r['label']),r['instance_id'],r['group'])==(cat,split,label,i,group) and r['tokens']]
  item=dict(category=cat,split=split,label=label,instance_id=i,group=group,conditions_with_support=len(rr))
  for k in numeric:item[k]=mean([r[k] for r in rr if r[k] is not None])
  inst.append(item)
 write('instance_groups.csv',inst)
 summary=dict(noncomparable=manifest['noncomparable'],scored_images=len(reports),thresholds_validation95=thresholds,score_bin_edges=bins.tolist(),notes=['Grid-level descriptive pilot, NOT native AU-PRO.','Six validation normals give unstable tail estimates, not FPR guarantees.','Normal footprints from abnormal images are diagnostic only, never calibration.','Any-defect-footprint is diluted and not equivalent to token feature seeing defect.','Physical instances, not tokens/lighting repeats, are sampling units.','No confidence claim at two abnormal instances/category.','Positive true_pair_advantage on normal and abnormal alike is not discriminative evidence.'])
 (a.out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False),encoding='utf-8')
 print(json.dumps(summary,indent=2));print('OUTPUT',a.out)
if __name__=='__main__':main()
