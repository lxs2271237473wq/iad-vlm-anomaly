"""Fixed two-score logistic control; source category LOCO, no target selection."""
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,average_precision_score
from local_learning import ROOT
from evaluate_exchange import intervals

BASE=ROOT/'results/stage25_tiny_defects';OUT=BASE/'dino_complement_v1'
OUT.mkdir(parents=True,exist_ok=False)
f=pd.read_csv(BASE/'dinov2_baseline_v1_offline/evaluation_predictions.csv')
f['fusion_oof']=np.nan;f['d_only_oof']=np.nan;fits=[]
for category in sorted(f.category.unique()):
    train=f[f.category!=category];idx=f.category==category;test=f[idx]
    counts=train.groupby(['category','label']).image_id.transform('size').to_numpy();w=1/counts;w/=w.mean()
    for name,columns in [('fusion_oof',['D_z','dino_max']),('d_only_oof',['D_z'])]:
        scaler=StandardScaler().fit(train[columns]);x=scaler.transform(train[columns])
        model=LogisticRegression(C=1,solver='lbfgs',max_iter=1000,random_state=42).fit(x,train.label,sample_weight=w)
        assert model.n_iter_[0]<1000
        f.loc[idx,name]=model.decision_function(scaler.transform(test[columns]))
        fits.append(dict(heldout=category,method=name,coefficients=model.coef_.tolist(),intercept=model.intercept_.tolist(),training_categories=sorted(train.category.unique()),iterations=int(model.n_iter_[0])))
assert np.isfinite(f[['fusion_oof','d_only_oof']]).all().all()
f.to_csv(OUT/'predictions.csv',index=False);(OUT/'fits.json').write_text(json.dumps(fits,indent=2))
rows=[];cis=[]
for scope,data in [('all_source_fit',f),('tiny_only_plus_all_normal',f[(f.label==0)|(f.stratum=='tiny_only')])]:
    for category,g in data.groupby('category'):
        for method in ['D_z','dino_max','d_only_oof','fusion_oof']:rows.append(dict(scope=scope,category=category,method=method,auroc=roc_auc_score(g.label,g[method]),ap=average_precision_score(g.label,g[method])))
    for ci in intervals(data,[('fusion_oof','D_z'),('fusion_oof','d_only_oof')],replicates=2000):cis.append(dict(scope=scope,**ci))
table=pd.DataFrame(rows);table.to_csv(OUT/'category_metrics.csv',index=False);macro=table.groupby(['scope','method'])[['auroc','ap']].mean();macro.to_csv(OUT/'macro_metrics.csv');pd.DataFrame(cis).to_csv(OUT/'paired_intervals.csv',index=False)
(OUT/'complete.json').write_text(json.dumps(dict(status='complete',protocol='Fixed C=1 two-score logistic versus one-score logistic. Category-held-out scaler and training. Eight source-fit categories only.',limitations='Uses source image labels; exploratory fixed precomputed predictions; no target/new backbone tuning; ordinary fusion is not novelty.'),indent=2))
print(macro.to_string());print(pd.DataFrame(cis).to_string(index=False))
