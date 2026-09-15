from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

OUT=Path(__file__).resolve().parent
pred=pd.read_csv(OUT.parent/'stage24_ad2_highres/evaluation/predictions_with_evaluation_labels.csv')
cal=json.loads((OUT/'calibration.json').read_text())
cals=pd.DataFrame(cal['calibration'])
rows=[]; predictions=[]; rocrows=[]
for cat,g in pred.groupby('category'):
    norm=cal['normalization'][cat]
    values=(cals.loc[cals.category==cat,'D_raw'].to_numpy()-norm['center'])/norm['scale']
    assert len(values)==norm['n'] and np.isfinite(values).all()
    y=g.label.to_numpy(); score=g.D512.to_numpy()
    for q in [.90,.95,.99]:
        threshold=float(np.quantile(values,q,method='higher'))
        alarm=score>threshold
        tp=int(((y==1)&alarm).sum()); fp=int(((y==0)&alarm).sum())
        fn=int(((y==1)&~alarm).sum()); tn=int(((y==0)&~alarm).sum())
        rows.append(dict(category=cat,q=q,n_cal=len(values),threshold=threshold,TP=tp,FP=fp,FN=fn,TN=tn,FPR=fp/(fp+tn),FNR=fn/(fn+tp),recall=tp/(tp+fn),FP_share_errors=fp/(fp+fn) if fp+fn else None))
        if q==.95:
            a=g.copy(); a['threshold']=threshold; a['alarm']=alarm
            a['outcome']=np.where(y==1,np.where(alarm,'TP','FN'),np.where(alarm,'FP','TN'))
            predictions.append(a)
    fpr,tpr,thresholds=roc_curve(y,score,drop_intermediate=False)
    i=np.flatnonzero(tpr>=.95)[0]
    rocrows.append(dict(category=cat,FPR_at_TPR95=float(fpr[i]),achieved_TPR=float(tpr[i]),threshold=float(thresholds[i]),note='test-label ROC diagnostic only; not deployment threshold'))
df=pd.DataFrame(rows);df.to_csv(OUT/'category_confusion.csv',index=False)
pd.concat(predictions).to_csv(OUT/'predictions_q95.csv',index=False)
pd.DataFrame(rocrows).to_csv(OUT/'roc_diagnostic.csv',index=False)
summary=[]
for q,g in df.groupby('q'):
    s=g[['TP','FP','FN','TN']].sum().to_dict();tp,fp,fn,tn=[int(s[k]) for k in ['TP','FP','FN','TN']]
    summary.append(dict(q=float(q),**{k:int(v) for k,v in s.items()},FPR=fp/(fp+tn),FNR=fn/(fn+tp),recall=tp/(tp+fn),FP_share_errors=fp/(fp+fn),FN_share_errors=fn/(fp+fn),FP_share_all=fp/(tp+fp+fn+tn),FN_share_all=fn/(tp+fp+fn+tn),macro_FPR=float(g.FPR.mean()),macro_FNR=float(g.FNR.mean())))
(OUT/'summary.json').write_text(json.dumps(summary,indent=2))
print(df[df.q==.95].round(4).to_string(index=False))
print(json.dumps(summary,indent=2));print(pd.DataFrame(rocrows).to_string(index=False))
