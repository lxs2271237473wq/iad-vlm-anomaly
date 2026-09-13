"""Source-only paired photometric diagnostic; frozen detector and LOCO head."""
import gc,json
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image,ImageEnhance
from anomalib.models.image.patchcore.torch_model import PatchcoreModel
from common import image_transform,sha256
from local_learning import RUN,ROOT,features
from run_baselines import seed_all

OUT=RUN/'20260911_q'

def feature_array(a,im,p):
    a=a.astype(np.float32);small=cv2.GaussianBlur(a,(0,0),4);large=cv2.GaussianBlur(a,(0,0),16)
    std=np.sqrt(np.maximum(cv2.GaussianBlur(a*a,(0,0),16)-large*large,0))
    gray=np.asarray(im.convert('L').resize((512,512),Image.Resampling.BILINEAR),dtype=np.float32)/255
    mean=cv2.GaussianBlur(gray,(0,0),16)
    texture=np.sqrt(np.maximum(cv2.GaussianBlur(gray*gray,(0,0),16)-mean*mean,0))
    gx=cv2.Sobel(gray,cv2.CV_32F,1,0);gy=cv2.Sobel(gray,cv2.CV_32F,0,1)
    gradient=cv2.GaussianBlur(np.sqrt(gx*gx+gy*gy),(0,0),4)
    return np.stack([(a-p['center'])/p['scale'],(small-large)/p['scale'],(large-p['center'])/p['scale'],std/p['scale'],mean,texture,gradient],axis=-1).reshape(-1,7)

def main():
    OUT.mkdir(parents=True,exist_ok=False);seed_all(42);torch.set_num_threads(4);cv2.setNumThreads(1)
    manifest=pd.read_csv(RUN/'20260910_a/manifest/images.csv')
    source=manifest[(manifest.dataset=='VisA')&(manifest.split=='evaluation')].merge(pd.read_csv(RUN/'20260910_a/manifest/evaluation_labels.csv'),on='image_id',validate='one_to_one')
    selected=source.sort_values('image_id').groupby(['category','label']).head(4)
    assert len(selected)==96 and selected.category.nunique()==12
    selected.to_csv(OUT/'selected_images.csv',index=False)
    protocol=dict(code_sha256=sha256(Path(__file__)),images=96,views=480,selection='first four image IDs per source category and image label',factors=[.8,1.2],head_seed=42,target_used=False,limitations='Exploratory small source subset; masks diagnostic only; image transforms need not preserve defect visibility; no causal or novelty claim.')
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
    official=pd.read_csv(ROOT/'datasets/VisA/split_csv/1cls.csv')
    masks={str(ROOT/'datasets/VisA'/r.image):str(ROOT/'datasets/VisA'/r.mask) for r in official.itertuples() if isinstance(r.mask,str) and r.mask}
    records=[];transform=image_transform(512)
    for category,g in selected.groupby('category',sort=True):
        base=RUN/'20260911_d/fixed_bank/VisA'/category
        detector=PatchcoreModel(layers=['layer2','layer3'],backbone='wide_resnet50_2',pre_trained=True,num_neighbors=9).cuda().eval()
        detector.memory_bank=torch.load(base/'memory_bank.pt',map_location='cuda',weights_only=True)
        head=torch.nn.Sequential(torch.nn.Linear(7,32),torch.nn.ReLU(),torch.nn.Linear(32,1)).cuda().eval()
        head.load_state_dict(torch.load(RUN/'20260911_j'/category/'mlp.pt',map_location='cuda',weights_only=True))
        scaler=np.load(RUN/'20260911_j'/category/'scaler.npz');p=json.loads((base/'normal_calibration.json').read_text())['D_raw']
        old=pd.read_csv(base/'predictions.csv').set_index('image_id')
        old_head=pd.read_csv(RUN/'20260911_j'/category/'predictions.csv').set_index('image_id')
        for r in g.itertuples():
            with Image.open(r.path) as handle:im=handle.convert('RGB')
            mask=np.zeros((512,512),bool)
            if r.label:
                with Image.open(masks[r.path]) as handle:mask=np.asarray(handle.convert('L').resize((512,512),Image.Resampling.NEAREST))>0
                assert mask.any()
            outside=~(cv2.dilate(mask.astype(np.uint8),np.ones((17,17),np.uint8))>0)
            views=[('identity',im)]+[(f'{kind}_{factor}',enhancer(im).enhance(factor)) for kind,enhancer in [('brightness',ImageEnhance.Brightness),('contrast',ImageEnhance.Contrast)] for factor in [.8,1.2]]
            original_pixels=np.asarray(im)
            for name,view in views:
                with torch.inference_mode():
                    result=detector(transform(view)[None].cuda());a=result.anomaly_map[0,0].cpu().numpy();d=float(result.pred_score.item())
                    x=feature_array(a,view,p)
                    logits=torch.cat([head(torch.as_tensor((chunk-scaler['mean'])/scaler['scale'],device='cuda')).flatten().cpu() for chunk in np.array_split(x,8)]).numpy().reshape(512,512)
                assert np.isfinite(x).all() and np.isfinite(logits).all()
                if name=='identity':
                    assert np.isclose(d,old.loc[r.image_id,'D_raw'],rtol=1e-4,atol=1e-4)
                    assert np.allclose(x,features(r),rtol=1e-3,atol=2e-3)
                    assert np.isclose(logits.max(),old_head.loc[r.image_id,'mlp'],rtol=1e-4,atol=1e-4)
                pixels=np.asarray(view)
                record=dict(category=category,image_id=r.image_id,label=r.label,variant=name,D=d,MLP=float(logits.max()),new_saturation_fraction=float(np.mean(((pixels==0)|(pixels==255))&((original_pixels>0)&(original_pixels<255)))))
                for key,grid in [('D',a),('MLP',logits)]:
                    record[key+'_peak_in_mask']=bool(mask.flat[grid.argmax()]) if r.label else np.nan
                    record[key+'_defect_background_margin']=float(grid[mask].max()-grid[outside].max()) if r.label and outside.any() else np.nan
                records.append(record)
            pd.DataFrame(records).to_csv(OUT/'scores.partial.csv',index=False)
        print('COMPLETE',category,flush=True);del detector,head;gc.collect();torch.cuda.empty_cache()
    f=pd.DataFrame(records);f.to_csv(OUT/'scores.csv',index=False)
    ranking=[]
    for category,g in f.groupby('category'):
        for method in ['D','MLP']:
            baseline=g[g.variant=='identity'];a=baseline[baseline.label==1].set_index('image_id')[method];n=baseline[baseline.label==0].set_index('image_id')[method]
            original=np.sign(a.to_numpy()[:,None]-n.to_numpy()[None,:])
            for variant,v in g.groupby('variant'):
                av=v[v.label==1].set_index('image_id').loc[a.index,method].to_numpy();nv=v[v.label==0].set_index('image_id').loc[n.index,method].to_numpy()
                current=np.sign(av[:,None]-nv[None,:])
                ranking.append(dict(category=category,method=method,variant=variant,pair_accuracy=float(np.mean((current+1)/2)),fraction_correct_to_wrong=float(np.mean((original>0)&(current<0))),fraction_wrong_to_correct=float(np.mean((original<0)&(current>0))),changed_pair_fraction=float(np.mean(original!=current))))
    pd.DataFrame(ranking).to_csv(OUT/'pair_rankings.csv',index=False)
    (OUT/'complete.json').write_text(json.dumps(dict(status='complete',identity_parity=True,**protocol),indent=2))

if __name__=='__main__':main()
