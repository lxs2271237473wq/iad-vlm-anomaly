"""Historical-cache exploratory fusion pilot. No GPU, images or model inference.
All learned parameters and selection use VisA PatchCore source only.
Existing D/M normalization is retained; this is NOT a clean AD2 benchmark.
"""
import argparse, hashlib, json, platform, time
from pathlib import Path
import numpy as np
import pandas as pd
import scipy
from scipy.optimize import minimize
from scipy.special import expit
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

SOURCE = 'results/stage22_selective_qcr/stage22_b1_visa_patchcore_loco_predictions.csv'
TARGET = 'results/stage22_selective_qcr/stage22_b2b_ad2_frozen_predictions.csv'

def metrics(frame, predictions, dataset):
    out=[]
    for method, score in predictions.items():
        vals=[]
        for cat, idx in frame.groupby('category',sort=True).indices.items():
            auc=roc_auc_score(frame.Y.to_numpy()[idx],score[idx]); vals.append(auc)
            out.append(dict(dataset=dataset,category=cat,method=method,n=len(idx),auroc=float(auc)))
        out.append(dict(dataset=dataset,category='MACRO',method=method,n=len(frame),auroc=float(np.mean(vals))))
    return out

def predict(frame, alpha, clf, theta):
    d=frame.D.to_numpy(); m=frame.M.to_numpy()
    a=float(expit(theta[0]))
    return {'detector':d,'vlm':m,'naive_0.5':.5*(d+m),
            'source_selected_fixed_alpha':(1-alpha)*d+alpha*m,
            'source_logistic_D_M':clf.decision_function(frame[['D','M']].to_numpy()),
            'source_rank_constant_alpha':(1-a)*d+a*m,
            'historical_SRB':frame.score_S1.to_numpy()}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--repo',required=True,type=Path); ap.add_argument('--out',required=True,type=Path)
    a=ap.parse_args(); a.out.mkdir(parents=True,exist_ok=True); start=time.perf_counter()
    source=pd.read_csv(a.repo/SOURCE)
    assert source[['D','M','Y']].notna().all().all()
    # Source category-equal weighting; total weight n preserves ordinary C meaning.
    counts=source.category.value_counts(); weights=np.array([len(source)/(len(counts)*counts[c]) for c in source.category])
    grid=np.round(np.arange(11)/10,1)
    tuning=[]
    for alpha in grid:
        pred=(1-alpha)*source.D.to_numpy()+alpha*source.M.to_numpy()
        score=np.mean([roc_auc_score(source.Y.to_numpy()[idx],pred[idx]) for idx in source.groupby('category').indices.values()])
        tuning.append({'alpha':float(alpha),'source_macro_auroc':float(score)})
    # Deterministic tie: first/smallest alpha on ascending grid. No target access yet.
    alpha=tuning[int(np.argmax([r['source_macro_auroc'] for r in tuning]))]['alpha']
    clf=LogisticRegression(C=1.0,solver='lbfgs',max_iter=1000,random_state=0)
    clf.fit(source[['D','M']].to_numpy(),source.Y.to_numpy(),sample_weight=weights)
    # One-parameter convex fusion, starts at alpha=.5. Fixed T=.1 and L2=.01.
    # Full positive-negative pairs within each category; each category equal weight.
    pairs=[]
    for cat,g in source.groupby('category',sort=True):
        p=g[g.Y==1]; n=g[g.Y==0]
        dd=(p.D.to_numpy()[:,None]-n.D.to_numpy()[None,:]).ravel()
        dm=(p.M.to_numpy()[:,None]-n.M.to_numpy()[None,:]).ravel()
        pairs.append((dd,dm))
    def objective(theta):
        b=float(theta[0]); w=expit(b); losses=[]; grads=[]
        for dd,dm in pairs:
            z=-((1-w)*dd+w*dm)/.1
            losses.append(np.logaddexp(0,z).mean())
            grads.append(np.mean(expit(z)*(-(dm-dd)/.1))*w*(1-w))
        return float(np.mean(losses)+.01*b*b),np.array([np.mean(grads)+.02*b])
    fit=minimize(objective,np.zeros(1),jac=True,method='L-BFGS-B',options={'maxiter':200,'ftol':1e-12})
    if not fit.success: raise RuntimeError(fit.message)
    params={'status':'historical_cache_exploratory_not_official_benchmark',
        'source':'VisA PatchCore (2162 images,12 categories)',
        'target':'historical AD2 four-category subset (243 images)',
        'selection_policy':'Source only; alpha uses source macro AUROC; logistic and rank hyperparameters fixed before target read. No OOF or independent source validation.',
        'selected_alpha':alpha,'alpha_grid':grid.tolist(),
        'logistic':{'C':1.0,'coef':clf.coef_.tolist(),'intercept':clf.intercept_.tolist(),'category_equal_weights':True},
        'rank_constant':{'b':float(fit.x[0]),'alpha':float(expit(fit.x[0])),'temperature':.1,'l2':.01,'initial_b':0,'category_equal_pair_loss':True,'iterations':int(fit.nit)},
        'dynamic_Q':'Not run: source candidate_quality_norm and target max(candidate_score_mean) quality provenance are not established as equivalent.',
        'limitations':['Cached per-category normalization includes legacy target whole-test min/max; target is not a clean held-out pipeline.',
          'No detector/VLM inference, no geometry fix, no eight-category AD2 evaluation.',
          'Source labels supervise the learned fusions; source fitted AUROC is not cross-validation.',
          'Historical SRB parameters come from old protocol; shown for diagnostic reference only.',
          'No target-driven parameter/model selection. Prior audits have already inspected target labels/results.',
          'No confidence intervals or significance claim; no new reliability features/model implemented.'],
        'versions':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__}}
    # Persist frozen source configuration BEFORE reading any target samples in this run.
    (a.out/'source_frozen_parameters.json').write_text(json.dumps(params,indent=2),encoding='utf-8')
    pd.DataFrame(tuning).to_csv(a.out/'source_alpha_selection.csv',index=False)
    target=pd.read_csv(a.repo/TARGET)
    assert target[['D','M','Y']].notna().all().all()
    rows=[]
    for name,frame in [('source_fitted',source),('target_AD2_4',target)]:
        preds=predict(frame,alpha,clf,fit.x); rows.extend(metrics(frame,preds,name))
        identifiers=[c for c in ['category','image_key','image_path','Y'] if c in frame]
        output=frame[identifiers].copy()
        for k,v in preds.items():output[k]=v
        output.to_csv(a.out/(name+'_predictions.csv'),index=False)
    result=pd.DataFrame(rows);result.to_csv(a.out/'metrics.csv',index=False)
    provenance={'elapsed_seconds':time.perf_counter()-start,'input_sha256':{p:hashlib.sha256((a.repo/p).read_bytes()).hexdigest() for p in [SOURCE,TARGET]},'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (a.out/'provenance.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
    table=result[result.category=='MACRO'].pivot(index='method',columns='dataset',values='auroc')
    print(table.to_string(float_format=lambda x:f'{x:.8f}'))
    print(json.dumps({'selected_alpha':alpha,'rank_alpha':float(expit(fit.x[0])),**provenance},indent=2))
    report=['# AD2 历史缓存 CPU 融合探索','',
      '本次确实运行了源域融合拟合与目标缓存评测，但未运行检测器/VLM，也不是完整八类 AD2 正式实验。',
      '使用 VisA PatchCore 2162 张、12 类为源，历史 AD2 四类 243 张为目标。当前缓存含旧目标集全体 min-max 归一化；结果只能诊断融合假设。','',
      '所有新参数仅由源域确定。固定权重按源类别宏 AUROC 在 0～1（步长0.1）选择；logistic 固定 C=1、源类别等权；常数凸融合用类别等权排序损失，固定温度0.1、L2=0.01，从 alpha=0.5 初始化。未做源域 OOF，source_fitted 是拟合数据指标，不能解释为泛化成绩。',
      '未训练质量动态门控：源目标 Q 的质量特征来源未能确认同义，不能把两个同名字段直接当作可靠性变量。', '',
      f'源选择固定 alpha={alpha:.1f}；排序训练常数 alpha={float(expit(fit.x[0])):.8f}。','',
      '| 方法 | 源拟合宏 AUROC | AD2 四类宏 AUROC |','|---|---:|---:|']
    for name,row in table.iterrows():report.append(f'| {name} | {row.source_fitted:.8f} | {row.target_AD2_4:.8f} |')
    report += ['',f'计算耗时：{provenance["elapsed_seconds"]:.3f} 秒（不含环境和数据检查）。',
       '全部方法均报告；没有根据目标成绩修改参数或挑选方法。旧 SRB 为历史诊断参照。未计算显著性；本轮不能声称新方法已优于朴素融合或可支撑论文。', '',
       '重跑：python run_cached_pilot.py --repo REPOSITORY_PATH --out OUTPUT_DIRECTORY',
       '逐类指标见 metrics.csv；预测见 target_AD2_4_predictions.csv；源选参、冻结参数与输入哈希分别记录在同目录 JSON/CSV。']
    (a.out/'历史缓存试跑报告.md').write_text('\n'.join(report),encoding='utf-8')

if __name__=='__main__':main()
