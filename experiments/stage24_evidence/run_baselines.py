"""Stage24 raw PatchCore + fixed CLIP; no Lightning validation/postprocessing.

All test labels are read only after all category predictions have been written.
This runner is the repaired full-field baseline, not the proposed exchange model.
"""
import argparse
import gc
import importlib.metadata as md
import json
import os
import random
import subprocess
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image,ImageDraw
from torch.utils.data import Dataset,DataLoader
from anomalib.models.image.patchcore.torch_model import PatchcoreModel
import open_clip
from common import image_transform,candidates,prompts,robust_fit,robust_apply,sha256

class Images(Dataset):
    def __init__(self,records,size): self.records=records; self.transform=image_transform(size)
    def __len__(self): return len(self.records)
    def __getitem__(self,index):
        row=self.records[index]
        with Image.open(row['path']) as im: tensor=self.transform(im.convert('RGB'))
        return tensor,index

def json_save(path,obj):
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(obj,indent=2)); temp.replace(path)

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False

def clip_init(checkpoint):
    model,_,_=open_clip.create_model_and_transforms('ViT-B-32',pretrained=str(checkpoint))
    return model.cuda().eval(),open_clip.get_tokenizer('ViT-B-32'),image_transform(224,clip=True)

@torch.inference_mode()
def encode_text(model,tokenizer,category):
    features=F.normalize(model.encode_text(tokenizer(prompts(category)).cuda()),dim=-1)
    return torch.stack([F.normalize(features[:3].mean(0),dim=0),F.normalize(features[3:].mean(0),dim=0)])

@torch.inference_mode()
def clip_scores(model,transform,images,text):
    tensor=torch.stack([transform(im) for im in images]).cuda()
    emb=F.normalize(model.encode_image(tensor),dim=-1)
    sim=emb@text.T
    return (sim[:,1]-sim[:,0]).float().cpu().numpy()

def evaluate(out,manifest,frame):
    from sklearn.metrics import roc_auc_score,average_precision_score
    labels=pd.read_csv(manifest/'evaluation_labels.csv')
    pred=pd.read_csv(out/'predictions.csv')
    assert pred.image_id.is_unique
    joined=pred.merge(labels,on='image_id',validate='one_to_one')
    assert len(joined)==len(pred)
    evaluated=joined[joined.split=='evaluation']
    metrics=[]
    score_cols=['D_raw','M_global','M_tight_top1','M_context_top1','M_context_top3max',
                'D_z','M_context_top1_z','naive_context_top1','naive_context_top3max']
    for col in score_cols:
        if evaluated.label.nunique()==2:
            metrics.append(dict(method=col,n=len(evaluated),normal=int((evaluated.label==0).sum()),
                                anomaly=int(evaluated.label.sum()),auroc=roc_auc_score(evaluated.label,evaluated[col]),
                                ap=average_precision_score(evaluated.label,evaluated[col])))
    pd.DataFrame(metrics).to_csv(out/'metrics.csv',index=False)
    return metrics

def run_category(args,manifest,frame,dataset,category,clip_model,tokenizer,clip_transform):
    out=args.out/dataset/category
    config=dict(dataset=dataset,category=category,seed=args.seed,image_size=args.image_size,
                coreset_ratio=args.coreset_ratio,backbone='wide_resnet50_2',layers=['layer2','layer3'],
                neighbors=9,preprocess='RGB full-field square resize; no center crop',
                clip='ViT-B-32/OpenAI cached weights',clip_sha256=args.clip_sha256,
                clip_size=224,prompts=prompts(category),candidate_quantile=.97,min_map_area=4,
                top_k=3,context_factor=1.5,normal_calibration='median / 1.4826 MAD; std fallback; no clipping',
                manifest_sha256=args.manifest_sha256,code_sha256=args.code_hashes,smoke=args.smoke,
                packages={p:md.version(p) for p in ['torch','torchvision','anomalib','open_clip_torch','timm']})
    if (out/'complete.json').exists():
        assert json.loads((out/'config.json').read_text())==config,'Resume configuration changed'
        print('SKIP_COMPLETE',dataset,category,flush=True); return
    out.mkdir(parents=True,exist_ok=True); (out/'maps').mkdir(exist_ok=True)
    if (out/'config.json').exists(): assert json.loads((out/'config.json').read_text())==config
    json_save(out/'config.json',config)
    seed_all(args.seed); start=time.perf_counter()
    rows=frame[(frame.dataset==dataset)&(frame.category==category)].sort_values('image_id')
    training=rows[rows.split=='train'].to_dict('records')
    calibration=rows[rows.split=='calibration'].to_dict('records')
    evaluation=rows[rows.split=='evaluation'].to_dict('records')
    if args.smoke: training=training[:8]; calibration=calibration[:4]; evaluation=evaluation[:4]
    assert training and calibration and evaluation
    infer=calibration+evaluation
    json_save(out/'used_image_ids.json',{s:[r['image_id'] for r in rr] for s,rr in [('train',training),('calibration',calibration),('evaluation',evaluation)]})
    print('START',dataset,category,'train/cal/eval',len(training),len(calibration),len(evaluation),flush=True)
    model=PatchcoreModel(layers=['layer2','layer3'],backbone='wide_resnet50_2',pre_trained=True,num_neighbors=9).cuda()
    for parameter in model.parameters(): parameter.requires_grad_(False)
    bankfile=out/'memory_bank.pt'
    if bankfile.exists():
        model.memory_bank=torch.load(bankfile,map_location='cuda',weights_only=True)
        print('RESUMED_BANK',list(model.memory_bank.shape),flush=True)
    else:
        model.train(); model.feature_extractor.eval()
        loader=DataLoader(Images(training,args.image_size),batch_size=args.batch_size,num_workers=args.workers,shuffle=False)
        with torch.inference_mode():
            for i,(batch,_) in enumerate(loader):
                model(batch.cuda())
                if i%10==0: print('EMBED',dataset,category,i+1,'/',len(loader),flush=True)
        print('CORESET_START',dataset,category,'patches',sum(len(t) for t in model.embedding_store),flush=True)
        model.subsample_embedding(sampling_ratio=args.coreset_ratio)
        torch.save(model.memory_bank.cpu(),out/'memory_bank.pt.tmp')
        (out/'memory_bank.pt.tmp').replace(bankfile)
    model.eval()
    text=encode_text(clip_model,tokenizer,category)
    predictions=[]; rois=[]
    loader=DataLoader(Images(infer,args.image_size),batch_size=args.batch_size,num_workers=args.workers,shuffle=False)
    smoke_checked=False
    with torch.inference_mode():
        for batch_index,(batch,indices) in enumerate(loader):
            torch.cuda.synchronize(); t=time.perf_counter()
            batch=batch.cuda(); output=model(batch)
            torch.cuda.synchronize(); detector_sec=(time.perf_counter()-t)/len(indices)
            if not smoke_checked:
                # Batch-composition invariance: first sample alone vs same sample in batch.
                single=model(batch[:1])
                delta=float((single.pred_score[0]-output.pred_score[0]).abs())
                assert torch.allclose(single.pred_score[0],output.pred_score[0],rtol=1e-4,atol=1e-4),delta
                json_save(out/'inference_checks.json',dict(batch_composition_score_delta=delta,
                    labels_excluded_from_model_inputs=True,post_processor_used=False,
                    center_crop_used=False,finite_outputs=True))
                smoke_checked=True
            scores=output.pred_score.float().cpu().numpy()
            maps=output.anomaly_map[:,0].float().cpu().numpy()
            assert np.isfinite(scores).all() and np.isfinite(maps).all()
            for offset,index in enumerate(indices.tolist()):
                row=infer[index]
                with Image.open(row['path']) as handle: im=handle.convert('RGB')
                regions=candidates(maps[offset],im.size)
                views=[im]
                for region in regions: views.extend([im.crop(region['box']),im.crop(region['context_box'])])
                torch.cuda.synchronize(); t=time.perf_counter()
                margins=clip_scores(clip_model,clip_transform,views,text)
                torch.cuda.synchronize(); clip_sec=time.perf_counter()-t
                tight=margins[1::2]; context=margins[2::2]
                pred=dict(image_id=row['image_id'],dataset=dataset,category=category,split=row['split'],
                    D_raw=float(scores[offset]),M_global=float(margins[0]),
                    M_tight_top1=float(tight[0] if len(tight) else margins[0]),
                    M_context_top1=float(context[0] if len(context) else margins[0]),
                    M_context_top3max=float(context.max() if len(context) else margins[0]),
                    num_candidates=len(regions),fallback_global=int(not regions),
                    detector_batch_amortized_sec=detector_sec,clip_sec=clip_sec,clip_views=len(views))
                predictions.append(pred)
                rois.append(dict(image_id=row['image_id'],width=im.width,height=im.height,
                                 regions=regions,clip_margins=margins.tolist()))
                np.savez_compressed(out/'maps'/f'{row["image_id"]}.npz',anomaly_map=maps[offset])
                if index in [0,len(calibration)]:
                    overlay=im.copy(); draw=ImageDraw.Draw(overlay)
                    for reg in regions:
                        draw.rectangle(reg['context_box'],outline='lime',width=max(2,im.width//400))
                        draw.rectangle(reg['box'],outline='red',width=max(2,im.width//400))
                    overlay.thumbnail((1200,1200)); overlay.save(out/f'roi_check_{row["split"]}.jpg')
            if batch_index%5==0: print('INFER',dataset,category,len(predictions),'/',len(infer),flush=True)
    raw=pd.DataFrame(predictions)
    raw.to_csv(out/'raw_predictions.csv',index=False)
    with open(out/'regions.jsonl','w') as f:
        for r in rois: f.write(json.dumps(r)+'\n')
    cal=raw[raw.split=='calibration']; params={}
    for field in ['D_raw','M_global','M_tight_top1','M_context_top1','M_context_top3max']:
        params[field]=robust_fit(cal[field])
        raw['D_z' if field=='D_raw' else field+'_z']=robust_apply(raw[field],params[field])
    raw['naive_context_top1']=.5*raw.D_z+.5*raw.M_context_top1_z
    raw['naive_context_top3max']=.5*raw.D_z+.5*raw.M_context_top3max_z
    raw.to_csv(out/'predictions.csv',index=False)
    json_save(out/'normal_calibration.json',params)
    metrics=evaluate(out,manifest,frame)
    completed=dict(status='smoke_complete' if args.smoke else 'baseline_complete',
                   train_images=len(training),calibration_images=len(calibration),evaluation_images=len(evaluation),
                   seconds=time.perf_counter()-start,memory_bank_shape=list(model.memory_bank.shape),
                   memory_bank_sha256=sha256(bankfile),predictions_sha256=sha256(out/'predictions.csv'))
    json_save(out/'complete.json',completed)
    print('COMPLETE',dataset,category,json.dumps(completed),flush=True)
    print('METRICS',dataset,category,json.dumps(metrics),flush=True)
    del model; gc.collect(); torch.cuda.empty_cache()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--manifest',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--clip-checkpoint',type=Path,required=True)
    ap.add_argument('--datasets',nargs='+',default=['AD2','VisA'])
    ap.add_argument('--categories',nargs='+')
    ap.add_argument('--image-size',type=int,default=256)
    ap.add_argument('--coreset-ratio',type=float,default=.1)
    ap.add_argument('--batch-size',type=int,default=4)
    ap.add_argument('--workers',type=int,default=4)
    ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--smoke',action='store_true')
    args=ap.parse_args()
    torch.set_num_threads(4)
    args.out.mkdir(parents=True,exist_ok=True)
    report=json.loads((args.manifest/'manifest_report.json').read_text())
    assert not report['cross_split_duplicates']
    args.manifest_sha256=sha256(args.manifest/'images.csv')
    args.clip_sha256=sha256(args.clip_checkpoint)
    args.code_hashes={p.name:sha256(p) for p in sorted(Path(__file__).parent.glob('*.py')) if p.name in ['common.py','run_baselines.py','build_manifest.py']}
    frame=pd.read_csv(args.manifest/'images.csv')
    assert 'label' not in frame
    model,tokenizer,transform=clip_init(args.clip_checkpoint)
    for dataset in args.datasets:
        categories=args.categories or sorted(frame[frame.dataset==dataset].category.unique())
        for category in categories:
            run_category(args,args.manifest,frame,dataset,category,model,tokenizer,transform)
    json_save(args.out/'queue_complete.json',dict(status='complete',datasets=args.datasets,smoke=args.smoke))

if __name__=='__main__': main()
