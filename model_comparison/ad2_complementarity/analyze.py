from pathlib import Path
import itertools,json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve
O=Path(__file__).resolve().parent
a=pd.read_csv(O.parent/'stage24_ad2_highres/evaluation/predictions_with_evaluation_labels.csv')
b=pd.read_csv(O/'dino.csv')
assert not a.image_id.duplicated().any() and not b.image_id.duplicated().any()
assert set(a.image_id)==set(b.image_id)
p=a.merge(b,on=['image_id','category'],validate='one_to_one');assert len(p)==1084
models=['D256','D512','dino_top1','dino_max']; thresholds=[]
for cat,g in p.groupby('category'):
 for m in models:
  f,t,th=roc_curve(g.label,g[m],drop_intermediate=False);i=np.flatnonzero(t>=.95)[0]
  p.loc[g.index,m+'_alarm']=g[m]>=th[i]
  thresholds.append(dict(category=cat,model=m,threshold=th[i],FPR=f[i],TPR=t[i]))
def pair(g,A,B):
 y=g.label.to_numpy()==1;a=g[A+'_alarm'].to_numpy(bool);b=g[B+'_alarm'].to_numpy(bool)
 fp=~y&a;fn=y&~a;tp=y&a;tn=~y&~a
 return dict(A=A,B=B,n=len(g),A_FP=int(fp.sum()),A_FN=int(fn.sum()),A_TP=int(tp.sum()),A_TN=int(tn.sum()),FP_corrected_by_B=int((fp&~b).sum()),FN_recovered_by_B=int((fn&b).sum()),TP_harmed_by_B=int((tp&~b).sum()),TN_harmed_by_B=int((tn&b).sum()),shared_FP=int((fp&b).sum()),shared_FN=int((fn&~b).sum()))
rows=[]
for A,B in itertools.permutations(models,2):
 rows.append(dict(category='ALL',**pair(p,A,B)))
 for c,g in p.groupby('category'): rows.append(dict(category=c,**pair(g,A,B)))
r=pd.DataFrame(rows);r.to_csv(O/'pairwise_errors.csv',index=False)
pd.DataFrame(thresholds).to_csv(O/'diagnostic_thresholds.csv',index=False)
p.to_csv(O/'aligned_predictions.csv',index=False)
print(r[(r.category=='ALL')&(r.A=='D512')].to_string(index=False))
print(r[(r.A=='D512')&(r.B=='dino_top1')].to_string(index=False))
systems=[]
for name,alarm in [('PatchCore512',p.D512_alarm.to_numpy(bool)),('DINOv2_top1',p.dino_top1_alarm.to_numpy(bool)),('AND',p.D512_alarm.to_numpy(bool)&p.dino_top1_alarm.to_numpy(bool)),('OR',p.D512_alarm.to_numpy(bool)|p.dino_top1_alarm.to_numpy(bool))]:
 y=p.label.to_numpy()==1;fp=int((~y&alarm).sum());fn=int((y&~alarm).sum())
 systems.append(dict(system=name,FP=fp,FN=fn,FPR=fp/(~y).sum(),recall=1-fn/y.sum()))
pd.DataFrame(systems).to_csv(O/'and_or_diagnostic.csv',index=False);print(systems)
(O/'protocol.json').write_text(json.dumps(dict(n=len(p),normal=int((p.label==0).sum()),anomaly=int((p.label==1).sum()),threshold='Per-category test ROC >=95% recall. Diagnostic only, not deployment or validated fusion.',efficientad='Checkpoint and aggregate metrics found; aligned per-image scores unavailable. No retraining or inference performed.',selection='Models, categories fixed; DINO top1 primary, max sensitivity. Public previously used for development.'),indent=2))
