"""Source-only bidirectional patch exchange pilot, without evaluation labels/masks.

Reference matching uses RGB context outside the candidate rectangle. It is a
fixed translation search, not a learned alignment or a causal-effect estimator.
"""
import argparse,json,math,time
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image,ImageDraw
from common import context_box,sha256,robust_fit,robust_apply
from run_baselines import clip_init,encode_text,json_save,seed_all

FIELDS=['u_mean','v_mean','bidirectional_mean','u_std','v_std','reference_contrast',
        'visual_distance','match_error','reference_margin','swap_margin_mean']

def canonical_box(box,size,side=128):
    w,h=size; x1,y1,x2,y2=box
    return (max(0,math.floor(x1*side/w)),max(0,math.floor(y1*side/h)),
            min(side,math.ceil(x2*side/w)),min(side,math.ceil(y2*side/h)))

def search_template(im,box,side=128):
    ring=context_box(box,im.size,3.0)
    mx1,my1,mx2,my2=canonical_box(ring,im.size,side)
    if mx2-mx1<16:
        mx1=max(0,min(side-16,(mx1+mx2)//2-8)); mx2=mx1+16
    if my2-my1<16:
        my1=max(0,min(side-16,(my1+my2)//2-8)); my2=my1+16
    small=cv2.resize(np.asarray(im),(side,side),interpolation=cv2.INTER_AREA).astype(np.float32)/255
    template=small[my1:my2,mx1:mx2].copy()
    tx1,ty1,tx2,ty2=canonical_box(box,im.size,side)
    mask=np.ones(template.shape[:2],dtype=np.uint8)
    # Cover all downsampling cells touched by the candidate, plus one-cell guard.
    mask[max(0,ty1-my1-1):min(mask.shape[0],ty2-my1+1),
         max(0,tx1-mx1-1):min(mask.shape[1],tx2-mx1+1)]=0
    return template,mask,(mx1,my1,mx2,my2)

def match_references(im,box,score_box,references,side=128):
    template,mask,search_box=search_template(im,box,side)
    valid=int(mask.sum())
    if valid<12: return [],'insufficient_context_pixels'
    candidates=[]
    for ref in references:
        response=cv2.matchTemplate(ref['small'],template,cv2.TM_SQDIFF,mask=mask)
        if not np.isfinite(response).all(): raise ValueError('Non-finite reference matching response')
        minimum,_,location,_=cv2.minMaxLoc(response)
        candidates.append((max(0,float(minimum))/(valid*3),ref['image_id'],location,ref))
    candidates.sort(key=lambda r:(r[0],r[1],r[2]))
    results=[]
    for error,ref_id,(rx,ry),ref in candidates[:2]:
        w,h=im.size; rw,rh=ref['image'].size; mx,my,_,_=search_box
        sx1,sy1,sx2,sy2=score_box
        rb=(max(0,math.floor((rx+sx1*side/w-mx)*rw/side)),
            max(0,math.floor((ry+sy1*side/h-my)*rh/side)),
            min(rw,math.ceil((rx+sx2*side/w-mx)*rw/side)),
            min(rh,math.ceil((ry+sy2*side/h-my)*rh/side)))
        assert rb[2]>rb[0] and rb[3]>rb[1]
        crop=ref['image'].crop(rb).resize((sx2-sx1,sy2-sy1),Image.Resampling.BICUBIC)
        results.append(dict(image=crop,error=error,image_id=ref_id,reference_box=rb))
    return results,'ok'

def swap_pair(query,reference,local_box):
    removed=query.copy(); inserted=reference.copy()
    removed.paste(reference.crop(local_box),local_box[:2])
    inserted.paste(query.crop(local_box),local_box[:2])
    return removed,inserted

@torch.inference_mode()
def score_views(model,transform,views,text):
    features=F.normalize(model.encode_image(torch.stack([transform(im) for im in views]).cuda()),dim=-1)
    similarity=features@text.T
    return (similarity[:,1]-similarity[:,0]).cpu().numpy(),features.cpu().numpy()

def montage(path,views,margins):
    names=['query','reference 1','query removed 1','reference inserted 1',
           'reference 2','query removed 2','reference inserted 2']
    canvas=Image.new('RGB',(224*len(views),254),'white'); draw=ImageDraw.Draw(canvas)
    for i,(im,name,margin) in enumerate(zip(views,names,margins)):
        canvas.paste(im.resize((224,224)),(224*i,30))
        draw.text((224*i+3,3),f'{name}: {margin:.4f}',fill='black')
    canvas.save(path)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--run',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True); ap.add_argument('--clip-checkpoint',type=Path,required=True)
    ap.add_argument('--categories',nargs='+'); ap.add_argument('--smoke',action='store_true')
    args=ap.parse_args(); args.out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4); cv2.setNumThreads(1); seed_all(42)
    frame=pd.read_csv(args.run/'manifest/images.csv'); assert 'label' not in frame
    frame=frame[frame.dataset=='VisA']
    model,tokenizer,transform=clip_init(args.clip_checkpoint)
    config=dict(dataset='VisA',reference_pool_size=32,reference_count=2,roi='top1 detector rectangle',
        reference_search='masked RGB SQDIFF translation at 128x128; context factor 3, minimum 16x16 cells; candidate+1-cell excluded',
        scoring_context_factor=1.5,normal_calibration='independent held-out normal; median/MAD; no clipping',
        exchange='hard rectangular patch swap; reference scored crop resized to query scored crop size',
        additional_views='two references plus four exchanged views; query rescored for parity check',
        smoke=args.smoke,clip_sha256=sha256(args.clip_checkpoint),
        inference_manifest_sha256=sha256(args.run/'manifest/images.csv'),
        code_sha256={name:sha256(Path(__file__).parent/name) for name in ['extract_exchange.py','common.py','run_baselines.py']})
    all_frames=[]
    for category in args.categories or sorted(frame.category.unique()):
        out=args.out/category; out.mkdir(exist_ok=True)
        if (out/'config.json').exists(): assert json.loads((out/'config.json').read_text())==config
        json_save(out/'config.json',config)
        if (out/'complete.json').exists():
            assert sha256(out/'evidence.csv')==json.loads((out/'complete.json').read_text())['evidence_sha256']
            all_frames.append(pd.read_csv(out/'evidence.csv')); print('SKIP',category,flush=True); continue
        start=time.perf_counter()
        source=frame[frame.category==category].sort_values('image_id')
        training=source[source.split=='train'].head(32)
        reference_ids=set(training.image_id)
        query=source[source.split!='train']
        if args.smoke: query=query.groupby('split',group_keys=False).head(3)
        assert not reference_ids.intersection(query.image_id)
        references=[]
        for r in training.itertuples():
            with Image.open(r.path) as handle: image=handle.convert('RGB')
            small=cv2.resize(np.asarray(image),(128,128),interpolation=cv2.INTER_AREA).astype(np.float32)/255
            references.append(dict(image_id=r.image_id,image=image,small=small))
        baseline=args.run/'baselines/VisA'/category
        baseline_pred=pd.read_csv(baseline/'predictions.csv').set_index('image_id')
        roi={r['image_id']:r for r in map(json.loads,(baseline/'regions.jsonl').read_text().splitlines())}
        text=encode_text(model,tokenizer,category); records=[]; provenance=[]; shown=set()
        print('START',category,'queries',len(query),'references',len(references),flush=True)
        for n,r in enumerate(query.itertuples(),1):
            record=dict(image_id=r.image_id,category=category,split=r.split,evidence_available=0,
                        encoded_views=0,matching_reason='no_candidate',**{name:0. for name in FIELDS})
            region_list=roi[r.image_id]['regions']
            if region_list:
                region=region_list[0]; box=region['box']; score_box=region['context_box']
                with Image.open(r.path) as handle: im=handle.convert('RGB')
                chosen,reason=match_references(im,box,score_box,references)
                record['matching_reason']=reason
                if chosen:
                    crop=im.crop(score_box)
                    local=(box[0]-score_box[0],box[1]-score_box[1],box[2]-score_box[0],box[3]-score_box[1])
                    views=[crop]
                    for ref in chosen:
                        removed,inserted=swap_pair(crop,ref['image'],local)
                        views.extend([ref['image'],removed,inserted])
                    scores,embeddings=score_views(model,transform,views,text)
                    delta=abs(float(scores[0])-float(baseline_pred.loc[r.image_id,'M_context_top1']))
                    assert delta<2e-5,('baseline query score changed',delta,r.image_id)
                    reference=scores[[1,4]]; removed=scores[[2,5]]; inserted=scores[[3,6]]
                    u=scores[0]-removed; v=inserted-reference
                    record.update(evidence_available=1,encoded_views=7,u_mean=float(u.mean()),v_mean=float(v.mean()),
                        bidirectional_mean=float((u.mean()+v.mean())/2),u_std=float(u.std()),v_std=float(v.std()),
                        reference_contrast=float(scores[0]-reference.mean()),reference_margin=float(reference.mean()),
                        visual_distance=float(np.mean(1-embeddings[[1,4]]@embeddings[0])),
                        match_error=float(np.mean([ref['error'] for ref in chosen])),
                        swap_margin_mean=float(scores[[2,3,5,6]].mean()))
                    provenance.append(dict(image_id=r.image_id,query_box=box,query_context_box=score_box,
                        references=[{k:v for k,v in ref.items() if k!='image'} for ref in chosen],scores=scores.tolist(),
                        query_parity_absolute_error=delta))
                    if r.split not in shown:
                        montage(out/f'views_{r.split}.jpg',views,scores); shown.add(r.split)
            records.append(record)
            if n%25==0: print('EXTRACT',category,n,'/',len(query),flush=True)
        raw=pd.DataFrame(records); raw.to_csv(out/'raw_evidence.csv',index=False)
        normal=raw[(raw.split=='calibration')&(raw.evidence_available==1)]
        assert len(normal)>=2,('insufficient valid normal controls',category)
        params={}
        for name in FIELDS:
            params[name]=robust_fit(normal[name])
            raw[name+'_z']=np.where(raw.evidence_available,robust_apply(raw[name],params[name]),0.)
        raw.to_csv(out/'evidence.csv',index=False)
        with open(out/'provenance.jsonl','w') as f:
            for record in provenance: f.write(json.dumps(record)+'\n')
        json_save(out/'normal_control_calibration.json',params)
        json_save(out/'reference_pool.json',dict(image_ids=sorted(reference_ids)))
        completion=dict(queries=len(raw),valid_evidence=int(raw.evidence_available.sum()),
                        normal_controls=len(normal),seconds=time.perf_counter()-start,
                        encoded_views=int(raw.encoded_views.sum()),evidence_sha256=sha256(out/'evidence.csv'))
        json_save(out/'complete.json',completion)
        all_frames.append(raw); print('COMPLETE',category,json.dumps(completion),flush=True)
    pd.concat(all_frames,ignore_index=True).to_csv(args.out/'all_source_evidence.csv',index=False)
    json_save(args.out/'complete.json',dict(categories=len(all_frames),smoke=args.smoke))

if __name__=='__main__': main()
