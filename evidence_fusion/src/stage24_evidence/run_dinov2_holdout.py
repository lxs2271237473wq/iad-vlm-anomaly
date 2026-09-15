"""New frozen representation baseline; not an AnomalyDINO reproduction."""
import json,time,math
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModel
from local_learning import ROOT,RUN
from common import image_transform,sha256
from run_baselines import seed_all
from evaluate_exchange import intervals
from sklearn.metrics import roc_auc_score,average_precision_score

BASE=ROOT/'results/stage25_tiny_defects';OUT=BASE/'dinov2_holdout_v1'

@torch.inference_mode()
def main():
    OUT.mkdir(parents=True,exist_ok=False);seed_all(42);torch.set_num_threads(4)
    (OUT/'protocol.json').write_text(json.dumps(dict(code_sha256=sha256(__file__),model='facebook/dinov2-base',input_size=518,reference_count=32,reference_selection='First 32 normal train image IDs per category; every patch retained',feature='L2 normalized last hidden patch tokens, exclude CLS',primary='fusion_frozen versus D_z; fixed maximum cosine distance for DINO score',secondary='mean largest 1% patch distances',scope='400 source_development_holdout images; same eight source_fit categories and frozen coefficients',training=False,limitations='New backbone baseline only, not novel algorithm or exact AnomalyDINO reproduction; reference budget differs from original WRN50.'),indent=2))
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    training=pd.read_csv(BASE/'dinov2_baseline_v1_offline/evaluation_predictions.csv')
    columns=['D_z','dino_max'];scaler=StandardScaler().fit(training[columns])
    weights=1/training.groupby(['category','label']).image_id.transform('size').to_numpy();weights/=weights.mean()
    fusion=LogisticRegression(C=1,solver='lbfgs',max_iter=1000,random_state=42).fit(scaler.transform(training[columns]),training.label,sample_weight=weights)
    assert fusion.n_iter_[0]<1000
    (OUT/'frozen_fusion.json').write_text(json.dumps(dict(training_categories=sorted(training.category.unique()),mean=scaler.mean_.tolist(),scale=scaler.scale_.tolist(),coefficients=fusion.coef_.tolist(),intercept=fusion.intercept_.tolist(),C=1,validation_labels_used=False),indent=2))
    frozen=json.loads((BASE/'dinov2_validation_v1/frozen_fusion.json').read_text())
    assert np.allclose(fusion.coef_,frozen['coefficients'],rtol=0,atol=1e-12)
    assert np.allclose(fusion.intercept_,frozen['intercept'],rtol=0,atol=1e-12)
    assert np.allclose(scaler.mean_,frozen['mean'],rtol=0,atol=1e-12)
    assert np.allclose(scaler.scale_,frozen['scale'],rtol=0,atol=1e-12)
    print('FUSION_FROZEN_PARITY_PASSED',flush=True)
    print('LOADING_DINOV2',flush=True)
    model=AutoModel.from_pretrained(str(ROOT/'models/dinov2-base'),use_safetensors=True,local_files_only=True).cuda().eval()
    assert model.config.patch_size==14
    (OUT/'model_config.json').write_text(model.config.to_json_string())
    print('MODEL_READY',flush=True);transform=image_transform(518)
    def embed(path):
        with Image.open(path) as h:im=h.convert('RGB')
        output=model(pixel_values=transform(im)[None].cuda()).last_hidden_state[:,1:]
        assert output.shape[1]==1369 and torch.isfinite(output).all()
        return F.normalize(output[0],dim=1)
    manifest=pd.read_csv(BASE/'v1/inference_manifest.csv');evals=manifest[manifest.role=='source_development_holdout'];assert len(evals)==400
    rows=[]
    for category,g in evals.groupby('category',sort=True):
        ref=manifest[(manifest.dataset=='VisA')&(manifest.category==category)&(manifest.original_split=='train')].sort_values('image_id').head(32)
        assert len(ref)==32 and not set(ref.image_id)&set(g.image_id)
        dest=OUT/category;dest.mkdir();ref.to_csv(dest/'reference_images.csv',index=False)
        bank=torch.cat([embed(r.path) for r in ref.itertuples()]);torch.save(bank.cpu(),dest/'memory_bank.pt')
        assert torch.allclose((bank[:4]@bank.T).max(1).values,torch.ones(4,device='cuda'),atol=1e-5)
        print('BANK_READY',category,list(bank.shape),flush=True)
        for i,r in enumerate(g.sort_values('image_id').itertuples()):
            start=time.perf_counter();x=embed(r.path)
            values=torch.cat([(1-chunk@bank.T).min(1).values.clamp_min(0) for chunk in x.split(256)])
            k=math.ceil(len(values)*.01);torch.cuda.synchronize()
            rows.append(dict(image_id=r.image_id,category=category,dino_max=float(values.max()),dino_top1=float(values.topk(k).values.mean()),seconds=time.perf_counter()-start))
            if i%50==0:print('INFER',category,i+1,len(g),flush=True)
        pd.DataFrame(rows).to_csv(OUT/'predictions.partial.csv',index=False);del bank;print('COMPLETE',category,flush=True)
    f=pd.DataFrame(rows);f.to_csv(OUT/'predictions_no_labels.csv',index=False)
    labels=pd.read_csv(BASE/'v1/partition_manifest.csv')[['image_id','label','stratum']]
    baseline=pd.read_csv(RUN/'20260911_k/oof_predictions.csv')[['image_id','D_z']]
    f=f.merge(labels,on='image_id',validate='one_to_one').merge(baseline,on='image_id',validate='one_to_one');f['fusion_frozen']=fusion.decision_function(scaler.transform(f[columns]));f.to_csv(OUT/'evaluation_predictions.csv',index=False)
    rows=[];cis=[]
    for scope,data in [('source_development_holdout',f),('tiny_only_plus_all_normal',f[(f.label==0)|(f.stratum=='tiny_only')])]:
        for category,g in data.groupby('category'):
            for method in ['D_z','dino_max','dino_top1','fusion_frozen']:rows.append(dict(scope=scope,category=category,method=method,auroc=roc_auc_score(g.label,g[method]),ap=average_precision_score(g.label,g[method])))
        for ci in intervals(data,[('fusion_frozen','D_z'),('dino_max','D_z'),('dino_top1','D_z')],replicates=2000):cis.append(dict(scope=scope,**ci))
    table=pd.DataFrame(rows);table.to_csv(OUT/'category_metrics.csv',index=False);macro=table.groupby(['scope','method'])[['auroc','ap']].mean();macro.to_csv(OUT/'macro_metrics.csv')
    pd.DataFrame(cis).to_csv(OUT/'paired_intervals.csv',index=False)
    (OUT/'complete.json').write_text(json.dumps(dict(status='complete',images=len(f),new_representation=True,new_algorithm=False,target_evaluated=False),indent=2));print(macro.to_string())

if __name__=='__main__':main()



