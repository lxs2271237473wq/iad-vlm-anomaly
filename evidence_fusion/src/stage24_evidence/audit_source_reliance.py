"""Frozen source LOCO heads: sampled-pixel reliance, not image detection metrics."""
import json
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from local_learning import RUN

OUT=RUN/'20260911_p'

def main():
    OUT.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4)
    data=np.load(RUN/'20260911_j/training_samples.npz')
    rows=[];gradients=[]
    for category in sorted(set(data['category'])):
        idx=data['category']==category
        x=data['x'][idx];y=data['y'][idx];ids=data['image_id'][idx]
        scaler=np.load(RUN/'20260911_j'/category/'scaler.npz')
        z=(x-scaler['mean'])/scaler['scale']
        model=torch.nn.Sequential(torch.nn.Linear(7,32),torch.nn.ReLU(),torch.nn.Linear(32,1))
        model.load_state_dict(torch.load(RUN/'20260911_j'/category/'mlp.pt',map_location='cpu',weights_only=True));model.eval()
        meta=pd.DataFrame(dict(image_id=ids,label=y))
        w=1/(meta.groupby(['image_id','label']).label.transform('size').to_numpy()*meta.groupby('label').image_id.transform('nunique').to_numpy())
        t=torch.tensor(z,requires_grad=True);logits=model(t).squeeze(1)
        grad=torch.autograd.grad(logits.sum(),t)[0].detach().numpy()
        gradients.append(dict(category=category,response_negative_fraction=float(np.average(grad[:,0]<-1e-8,weights=w)),appearance_abs_gradient_share=float(np.sum(w[:,None]*abs(grad[:,4:]))/max(np.sum(w[:,None]*abs(grad)),1e-12))))
        original=roc_auc_score(y,logits.detach().numpy(),sample_weight=w)
        groups=[np.flatnonzero(ids==image_id) for image_id in sorted(set(ids))]
        for variant,columns in [('appearance',[4,5,6]),('map',[0,1,2,3])]:
            for seed in range(10):
                rng=np.random.default_rng(seed);changed=z.copy()
                for group in groups:
                    changed[np.ix_(group,columns)]=z[np.ix_(rng.permutation(group),columns)]
                assert np.array_equal(changed[:,[i for i in range(7) if i not in columns]],z[:,[i for i in range(7) if i not in columns]])
                with torch.no_grad():score=model(torch.tensor(changed)).squeeze(1).numpy()
                auc=roc_auc_score(y,score,sample_weight=w)
                rows.append(dict(category=category,permuted=variant,seed=seed,original_sample_auc=original,permuted_sample_auc=auc,drop=original-auc))
        print('COMPLETE',category,flush=True)
    pd.DataFrame(rows).to_csv(OUT/'permutation.csv',index=False)
    pd.DataFrame(gradients).to_csv(OUT/'gradients.csv',index=False)
    (OUT/'complete.json').write_text(json.dumps(dict(status='complete',source_only=True,head_seed=42,permutations=10,limitations='Mask-selected sampled pixels, not dense image or official pixel AUROC; perturbations can break joint support; derivative is sensitivity not causal importance; permutations are not training seeds.'),indent=2))
    print(pd.DataFrame(rows).groupby('permuted')[['original_sample_auc','permuted_sample_auc','drop']].mean().to_string())
    print(pd.DataFrame(gradients).to_string(index=False))

if __name__=='__main__':main()
