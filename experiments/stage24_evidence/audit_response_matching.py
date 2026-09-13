"""Within-image nearest-response matching of existing source sampled pixels."""
import json
import numpy as np
import pandas as pd
import torch
from local_learning import RUN

OUT=RUN/'20260912_t'

def main():
    OUT.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4)
    d=np.load(RUN/'20260911_j/training_samples.npz');rows=[]
    (OUT/'protocol.json').write_text(json.dumps(dict(calipers=[.01,.05,.1],units='Existing category-normal calibrated response units',matching='Within same image nearest negative response, with replacement; ties take first sorted point',heads='Frozen source category-LOCO seed 42',limitations='GT-selected existing sample, not dense pixels. Calipers are sensitivity analysis, not tuned. No training or target evaluation.'),indent=2))
    for category in sorted(set(d['category'])):
        ix=d['category']==category;x=d['x'][ix];y=d['y'][ix];ids=d['image_id'][ix]
        scaler=np.load(RUN/'20260911_j'/category/'scaler.npz')
        model=torch.nn.Sequential(torch.nn.Linear(7,32),torch.nn.ReLU(),torch.nn.Linear(32,1))
        model.load_state_dict(torch.load(RUN/'20260911_j'/category/'mlp.pt',map_location='cpu',weights_only=True));model.eval()
        with torch.no_grad():z=model(torch.tensor((x-scaler['mean'])/scaler['scale'])).flatten().numpy()
        for image_id in sorted(set(ids[y==1])):
            pos=np.flatnonzero((ids==image_id)&(y==1));neg=np.flatnonzero((ids==image_id)&(y==0))
            assert len(neg)>0
            distance=np.abs(x[pos,0,None]-x[neg,0][None,:]);nearest=distance.argmin(1);chosen=neg[nearest];delta=distance[np.arange(len(pos)),nearest]
            assert np.allclose(delta,np.abs(x[pos,0]-x[chosen,0]))
            for caliper in [.01,.05,.1]:
                keep=delta<=caliper;n=int(keep.sum())
                row=dict(category=category,image_id=image_id,caliper=caliper,positive_count=len(pos),matched_count=n,coverage=n/len(pos),median_nearest_distance=float(np.median(delta)),unique_negatives=len(set(chosen[keep])),reuse_fraction=(1-len(set(chosen[keep]))/n) if n else np.nan)
                for name,score in [('response',x[:,0]),('MLP',z)]:
                    row[name+'_pair_accuracy']=float(np.mean((np.sign(score[pos[keep]]-score[chosen[keep]])+1)/2)) if n else np.nan
                rows.append(row)
        print('COMPLETE',category,flush=True)
    f=pd.DataFrame(rows);f.to_csv(OUT/'per_image.csv',index=False)
    summary=f.groupby(['category','caliper']).agg(images=('image_id','size'),images_with_matches=('matched_count',lambda s:int((s>0).sum())),coverage=('coverage','mean'),median_nearest_distance=('median_nearest_distance','median'),reuse_fraction=('reuse_fraction','mean'),response_pair_accuracy=('response_pair_accuracy','mean'),MLP_pair_accuracy=('MLP_pair_accuracy','mean'))
    summary.to_csv(OUT/'category_summary.csv')
    macro=summary.groupby('caliper').mean();macro.to_csv(OUT/'macro_summary.csv');print(macro.to_string())
    (OUT/'complete.json').write_text(json.dumps(dict(status='complete',caveats='Accuracy conditional on matched samples, equal images then categories; unmatched images excluded from accuracy but retained in coverage. No independent-pair tests. Normal response units not calibrated probability. Source pixels only.'),indent=2))

if __name__=='__main__':main()
