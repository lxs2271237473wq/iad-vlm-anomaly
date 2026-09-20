from pathlib import Path
import json,itertools
import numpy as np,pandas as pd
from sklearn.metrics import roc_curve,roc_auc_score
O=Path(__file__).resolve().parent;BASE=O.parents[1]
e=pd.read_csv(O/'predictions.csv');m=pd.read_csv(BASE/'stage24_server/images_manifest.csv')
p=pd.read_csv(BASE/'stage24_ad2_highres/evaluation/predictions_with_evaluation_labels.csv')
d=pd.read_csv(O.parent/'dino.csv')
m=m[m.image_id.isin(p.image_id)][['image_id','category','sha256']]
assert not m.duplicated(['category','sha256']).any()
z=e.merge(m,on=['category','sha256'],validate='one_to_one').merge(p,on=['image_id','category','label'],validate='one_to_one').merge(d,on=['image_id','category'],validate='one_to_one')
assert len(z)==len(e)==484
models=['D512','dino_top1','efficientad'];tables=[];metrics=[];systems=[];thr=[]
for scope,g0 in [('four_public',z),('legacy_subset',z[z.legacy_test])]:
 g0=g0.copy()
 for cat,g in g0.groupby('category'):
  for model in models:
   f,t,th=roc_curve(g.label,g[model],drop_intermediate=False);i=np.flatnonzero(t>=.95)[0]
   g0.loc[g.index,model+'_alarm']=g[model]>=th[i]
   thr.append(dict(scope=scope,category=cat,model=model,threshold=th[i],FPR=f[i],recall=t[i]))
   metrics.append(dict(scope=scope,category=cat,model=model,n=len(g),AUROC=roc_auc_score(g.label,g[model])))
  metrics.append(dict(scope=scope,category=cat,model='efficientad_post',n=len(g),AUROC=roc_auc_score(g.label,g.efficientad_post)))
 for A,B in itertools.permutations(models,2):
  for cat,g in [('ALL',g0)]+list(g0.groupby('category')):
   y=g.label.to_numpy()==1;a=g[A+'_alarm'].to_numpy(bool);b=g[B+'_alarm'].to_numpy(bool)
   fp=~y&a;fn=y&~a;tp=y&a;tn=~y&~a
   tables.append(dict(scope=scope,category=cat,A=A,B=B,n=len(g),A_FP=int(fp.sum()),A_FN=int(fn.sum()),A_TP=int(tp.sum()),A_TN=int(tn.sum()),correct_FP=int((fp&~b).sum()),recover_FN=int((fn&b).sum()),harm_TP=int((tp&~b).sum()),harm_TN=int((tn&b).sum()),shared_FP=int((fp&b).sum()),shared_FN=int((fn&~b).sum())))
 for B in ['efficientad','dino_top1']:
  a=g0.D512_alarm.to_numpy(bool);b=g0[B+'_alarm'].to_numpy(bool);y=g0.label.to_numpy()==1
  for name,pred in [('D512',a),(B,b),('AND_'+B,a&b),('OR_'+B,a|b)]:
   fp=int((~y&pred).sum());fn=int((y&~pred).sum());systems.append(dict(scope=scope,system=name,FP=fp,FN=fn,FPR=fp/(~y).sum(),recall=1-fn/y.sum()))
 zcols=g0[['image_id']+[x+'_alarm' for x in models]];zcols.to_csv(O/(scope+'_decisions.csv'),index=False)
pd.DataFrame(tables).to_csv(O/'pairwise.csv',index=False);pd.DataFrame(metrics).to_csv(O/'metrics.csv',index=False)
pd.DataFrame(systems).drop_duplicates().to_csv(O/'systems.csv',index=False);pd.DataFrame(thr).to_csv(O/'thresholds.csv',index=False)
z.to_csv(O/'aligned_predictions.csv',index=False)
print(pd.DataFrame(tables).query("category=='ALL' and A=='D512'").to_string(index=False));print(pd.DataFrame(metrics).groupby(['scope','model']).AUROC.mean());print(pd.DataFrame(systems).drop_duplicates().to_string(index=False))
