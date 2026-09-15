"""Source-only matched-context scale probe; no masks or evaluation labels in inference."""
import argparse,json,time
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from common import context_box,robust_fit,robust_apply,sha256
from run_baselines import clip_init,encode_text,seed_all,json_save
from extract_exchange import match_references,score_views

SCALES={'tight':1.,'context15':1.5,'context30':3.}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True);ap.add_argument('--clip-checkpoint',type=Path,required=True)
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    seed_all(42);torch.set_num_threads(4);cv2.setNumThreads(1)
    frame=pd.read_csv(args.run/'manifest/images.csv');assert 'label' not in frame
    frame=frame[frame.dataset=='VisA']
    model,tokenizer,transform=clip_init(args.clip_checkpoint)
    config=dict(scales=SCALES,roi='fixed top1 from 256 full-field detector',reference_pool=32,selected_references=2,
        matcher='same masked RGB translation search as source exchange pilot',
        reference_extraction='retrieve at context3, then take aligned subwindows for each scale',
        query_views=3,reference_views=6,labels_used=False,mask_used=False,
        weight_sha256=sha256(args.clip_checkpoint),code_sha256=sha256(__file__),
        helper_sha256={p:sha256(Path(__file__).parent/p) for p in ['common.py','run_baselines.py','extract_exchange.py']})
    all_rows=[]
    for category in sorted(frame.category.unique()):
        out=args.out/category;out.mkdir(exist_ok=True)
        if (out/'complete.json').exists():
            assert json.loads((out/'config.json').read_text())==config
            all_rows.append(pd.read_csv(out/'scores.csv'));continue
        json_save(out/'config.json',config);start=time.perf_counter()
        rows=frame[frame.category==category].sort_values('image_id')
        refrows=rows[rows.split=='train'].head(32);queries=rows[rows.split!='train']
        assert not set(refrows.image_id)&set(queries.image_id)
        references=[]
        for r in refrows.itertuples():
            with Image.open(r.path) as handle:im=handle.convert('RGB')
            references.append(dict(image_id=r.image_id,image=im,small=cv2.resize(np.asarray(im),(128,128),interpolation=cv2.INTER_AREA).astype(np.float32)/255))
        base=args.run/'baselines/VisA'/category
        predictions=pd.read_csv(base/'predictions.csv').set_index('image_id')
        rois={r['image_id']:r for r in map(json.loads,(base/'regions.jsonl').read_text().splitlines())}
        text=encode_text(model,tokenizer,category);result=[]
        print('START',category,len(queries),flush=True)
        for i,r in enumerate(queries.itertuples(),1):
            with Image.open(r.path) as handle:im=handle.convert('RGB')
            candidates=rois[r.image_id]['regions']
            record=dict(image_id=r.image_id,category=category,split=r.split,reference_available=0,encoded_views=0)
            if candidates:
                tight=candidates[0]['box'];big=context_box(tight,im.size,3.)
                chosen,reason=match_references(im,tight,big,references)
                record['reference_available']=int(bool(chosen))
            else:
                tight=(0,0,*im.size);big=tight;chosen=[];reason='no_candidate'
            record['matching_reason']=reason
            views=[]; offsets={}
            for name,scale in SCALES.items():
                box=(tight if scale==1. else context_box(tight,im.size,scale)) if candidates else tight
                offsets[name]=len(views);views.append(im.crop(box))
                local=(box[0]-big[0],box[1]-big[1],box[2]-big[0],box[3]-big[1])
                if chosen:views.extend([ref['image'].crop(local) for ref in chosen])
                record[name+'_area_fraction']=(box[2]-box[0])*(box[3]-box[1])/(im.width*im.height)
            margins,embeddings=score_views(model,transform,views,text)
            for name,index in offsets.items():
                record[name+'_text']=float(margins[index])
                record[name+'_visual']=float(np.mean(1-embeddings[index+1:index+3]@embeddings[index])) if chosen else 0.
            assert abs(record['context15_text']-predictions.loc[r.image_id,'M_context_top1'])<2e-5
            assert abs(record['tight_text']-predictions.loc[r.image_id,'M_tight_top1'])<2e-5
            record['encoded_views']=len(views);result.append(record)
            if i%50==0:print('SCORE',category,i,'/',len(queries),flush=True)
        df=pd.DataFrame(result);parameters={}
        for field in [f'{name}_{kind}' for name in SCALES for kind in ['text','visual']]:
            normal=df[df.split=='calibration']
            if field.endswith('visual'):normal=normal[normal.reference_available==1]
            parameters[field]=robust_fit(normal[field]);df[field+'_z']=robust_apply(df[field],parameters[field])
            if field.endswith('visual'):df.loc[df.reference_available==0,field+'_z']=0.
        df.to_csv(out/'scores.csv',index=False);json_save(out/'normal_calibration.json',parameters)
        json_save(out/'complete.json',dict(queries=len(df),seconds=time.perf_counter()-start,sha256=sha256(out/'scores.csv')))
        all_rows.append(df);print('COMPLETE',category,flush=True)
    pd.concat(all_rows,ignore_index=True).to_csv(args.out/'all_source_scores.csv',index=False)
    json_save(args.out/'complete.json',dict(categories=len(all_rows)))

if __name__=='__main__':main()
