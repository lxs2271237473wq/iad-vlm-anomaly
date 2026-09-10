"""Evaluation-only GT inspection. This file is never imported by inference."""
import argparse,json
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from PIL import Image,ImageDraw

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--run',type=Path,required=True); args=ap.parse_args()
    images=pd.read_csv(args.run/'manifest/images.csv')
    labels=pd.read_csv(args.run/'manifest/evaluation_labels.csv')
    evaluated=images.merge(labels,on='image_id',validate='one_to_one')
    evaluated=evaluated[(evaluated.split=='evaluation')&(evaluated.label==1)]
    visa_split=pd.read_csv(args.root/'datasets/VisA/split_csv/1cls.csv')
    visa_masks={str(args.root/'datasets/VisA'/r.image):str(args.root/'datasets/VisA'/r.mask)
                for r in visa_split.itertuples() if isinstance(r.mask,str) and r.mask}
    rows=[]; output=args.run/'roi_evaluation'; output.mkdir(exist_ok=True)
    for marker in sorted((args.run/'baselines').glob('*/*/complete.json')):
        dataset,category=marker.parent.parts[-2:]
        regions={r['image_id']:r for r in map(json.loads,(marker.parent/'regions.jsonl').read_text().splitlines())}
        for j,row in enumerate(evaluated[(evaluated.dataset==dataset)&(evaluated.category==category)].sort_values('image_id').itertuples()):
            p=Path(row.path)
            mask_path=(args.root/'datasets/MVTec_AD_2'/category/'test_public/ground_truth/bad'/f'{p.stem}_mask.png'
                       if dataset=='AD2' else Path(visa_masks[str(p)]))
            with Image.open(mask_path) as handle: mask=np.asarray(handle.convert('L'))>0
            assert mask.shape==(row.height,row.width)
            total=int(mask.sum()); assert total>0,(row.image_id,mask_path)
            candidates=regions[row.image_id]['regions']
            covered=np.zeros_like(mask); top1=0; top1_context=0
            for rank,candidate in enumerate(candidates):
                x1,y1,x2,y2=candidate['box']
                cx1,cy1,cx2,cy2=candidate['context_box']
                covered[cy1:cy2,cx1:cx2]=True
                if rank==0:
                    top1=int(mask[y1:y2,x1:x2].sum())
                    top1_context=int(mask[cy1:cy2,cx1:cx2].sum())
            union=int((mask&covered).sum())
            rows.append(dict(image_id=row.image_id,dataset=dataset,category=category,gt_pixels=total,
                top1_tight_fraction=top1/total,top1_context_fraction=top1_context/total,
                top3_context_union_fraction=union/total,top1_tight_hit=int(top1>0),
                top1_context_hit=int(top1_context>0),top3_context_hit=int(union>0)))
            if j==0:
                with Image.open(p) as handle: im=handle.convert('RGB')
                array=np.array(im)
                contours,_=cv2.findContours(mask.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(array,contours,-1,(0,128,255),max(3,im.width//300))
                overlay=Image.fromarray(array); draw=ImageDraw.Draw(overlay)
                for candidate in candidates:
                    draw.rectangle(candidate['context_box'],outline='lime',width=max(2,im.width//400))
                    draw.rectangle(candidate['box'],outline='red',width=max(2,im.width//400))
                overlay.thumbnail((1400,1400)); overlay.save(output/f'{category}_first_anomaly.jpg')
    if not rows: return
    frame=pd.DataFrame(rows); frame.to_csv(output/'per_image_coverage.csv',index=False)
    fields=['top1_tight_fraction','top1_context_fraction','top3_context_union_fraction',
            'top1_tight_hit','top1_context_hit','top3_context_hit']
    summary=frame.groupby(['dataset','category'])[fields].mean(); summary['anomalous_images']=frame.groupby(['dataset','category']).size()
    summary.to_csv(output/'category_coverage.csv')
    print(summary.to_string())
    print('GT used for evaluation only; no inference or parameter selection performed.')

if __name__=='__main__': main()
