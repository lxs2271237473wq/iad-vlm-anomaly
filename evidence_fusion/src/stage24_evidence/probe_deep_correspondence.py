"""Frozen deep context matching feasibility pilot; not a novel-method claim."""
import json,time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from anomalib.models.image.patchcore.torch_model import PatchcoreModel
from local_learning import RUN
from common import image_transform,sha256
from run_baselines import seed_all
from evaluate_exchange import intervals
from sklearn.metrics import roc_auc_score,average_precision_score

OUT=RUN/'20260912_y'

def ring(x):
    # Eight neighboring cells, exclude center, normalize by valid border cells.
    kernel=torch.ones((x.shape[1],1,3,3),device=x.device,dtype=x.dtype);kernel[:,:,1,1]=0
    total=F.conv2d(x,kernel,padding=1,groups=x.shape[1])
    count=F.conv2d(torch.ones_like(x[:,:1]),kernel[:1],padding=1)
    return total/count

@torch.inference_mode()
def main():
    OUT.mkdir(parents=True,exist_ok=False);seed_all(42);torch.set_num_threads(4)
    test=torch.ones((1,2,5,5),device='cuda');assert torch.allclose(ring(test),test)
    impulse=torch.zeros_like(test);impulse[:,:,2,2]=1;assert (ring(impulse)[:,:,2,2]==0).all()
    manifest=pd.read_csv(RUN/'20260910_a/manifest/images.csv');source=manifest[manifest.dataset=='VisA']
    # Stratification is exploratory and disclosed; scoring has no labels/masks.
    labels=pd.read_csv(RUN/'20260910_a/manifest/evaluation_labels.csv')
    selected=source[source.split=='evaluation'].merge(labels,on='image_id',validate='one_to_one').sort_values('image_id').groupby(['category','label']).head(16)
    assert len(selected)==384
    selected.to_csv(OUT/'selected_images.csv',index=False)
    refs=source[source.split=='train'].sort_values('image_id').groupby('category').head(4)
    assert len(refs)==48 and not set(refs.image_id)&set(selected.image_id)
    refs.to_csv(OUT/'reference_images.csv',index=False)
    protocol=dict(code_sha256=sha256(__file__),images=384,reference_images_per_category=4,grid=32,input_size=512,backbone='WRN50_2 layer2/3, existing PatchCore pooling; adaptive average to 32x32',features='L2 normalized center and eight-neighbor mean; cosine distance',primary='context_matched_center versus center_nn with identical reference pool',controls=['same_position','shuffled_correspondence','D_z'],aggregation='full-grid maximum',no_training=True,seed=42,limitations='Source exploratory subset overlaps prior probes; 32-grid may erase small defects; neural receptive fields overlap despite excluding center cell; not learned registration or RegAD reproduction; D_z uses larger reference bank.')
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
    model=PatchcoreModel(layers=['layer2','layer3'],backbone='wide_resnet50_2',pre_trained=True,num_neighbors=9).cuda().eval();transform=image_transform(512)
    def embed(path):
        with Image.open(path) as h:im=h.convert('RGB')
        feature=model.feature_extractor(transform(im)[None].cuda())
        feature={k:model.feature_pooler(v) for k,v in feature.items()}
        grid=F.adaptive_avg_pool2d(model.generate_embedding(feature),(32,32))
        center=F.normalize(grid.flatten(2).transpose(1,2)[0],dim=1)
        context=F.normalize(ring(grid).flatten(2).transpose(1,2)[0],dim=1)
        return center,context
    rows=[]
    for category,g in selected.groupby('category',sort=True):
        reference=[embed(r.path) for r in refs[refs.category==category].itertuples()]
        rc=torch.cat([v[0] for v in reference]);rr=torch.cat([v[1] for v in reference]);permutation=torch.randperm(len(rc),device='cuda')
        # Exact center nearest-neighbor test on a small reference subset.
        small=rc[:7];fast=(1-small@rc.T).min(1).values
        brute=(.5*((small[:,None]-rc[None])**2).sum(2)).min(1).values
        assert torch.allclose(fast,brute,atol=1e-5)
        for r in g.drop(columns=['label','scene_group']).sort_values('image_id').itertuples():
            start=time.perf_counter();qc,qr=embed(r.path)
            center_score=(1-qc@rc.T).min(1).values.clamp_min(0)
            matched=(qr@rr.T).argmax(1)
            context_score=(1-(qc*rc[matched]).sum(1)).clamp_min(0)
            same=(1-(qc[None]*rc.reshape(4,1024,-1)).sum(2)).min(0).values.clamp_min(0)
            shuffled=(1-(qc*rc[permutation[matched]]).sum(1)).clamp_min(0)
            values=dict(center_nn=center_score,context_matched_center=context_score,same_position=same,shuffled_correspondence=shuffled)
            assert all(torch.isfinite(v).all() for v in values.values())
            peak=int(context_score.argmax());ref_index=int(matched[peak]);torch.cuda.synchronize()
            rows.append(dict(image_id=r.image_id,category=category,**{k:float(v.max()) for k,v in values.items()},context_peak_y=peak//32,context_peak_x=peak%32,reference_number=ref_index//1024,reference_y=(ref_index%1024)//32,reference_x=ref_index%32,seconds=time.perf_counter()-start))
        pd.DataFrame(rows).to_csv(OUT/'predictions.partial.csv',index=False);print('COMPLETE',category,flush=True)
    pred=pd.DataFrame(rows);pred.to_csv(OUT/'predictions_no_labels.csv',index=False)
    baseline=pd.read_csv(RUN/'20260911_k/oof_predictions.csv')
    f=pred.merge(baseline[['image_id','label','D_z']],on='image_id',validate='one_to_one');assert len(f)==384
    f.to_csv(OUT/'evaluation_predictions.csv',index=False)
    rows=[];methods=list(values)+['D_z']
    for category,g in f.groupby('category'):
        for method in methods:rows.append(dict(category=category,method=method,auroc=roc_auc_score(g.label,g[method]),ap=average_precision_score(g.label,g[method])))
    table=pd.DataFrame(rows);table.to_csv(OUT/'category_metrics.csv',index=False);macro=table.groupby('method')[['auroc','ap']].mean();macro.to_csv(OUT/'macro_metrics.csv')
    ci=intervals(f,[('context_matched_center',b) for b in ['center_nn','same_position','shuffled_correspondence','D_z']],replicates=2000)
    pd.DataFrame(ci).to_csv(OUT/'paired_intervals.csv',index=False)
    (OUT/'complete.json').write_text(json.dumps(dict(status='complete',ring_test=True,nearest_test=True,images=384,seconds=float(pred.seconds.sum()),limitations=protocol['limitations']),indent=2))
    print(macro.to_string());print(pd.DataFrame(ci).to_string(index=False))

if __name__=='__main__':main()
