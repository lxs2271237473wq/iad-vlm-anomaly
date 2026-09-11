"""GT evaluation only; never import this module into inference/selection."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from common import context_box

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--run',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    im=pd.read_csv(args.run/'manifest/images.csv');labels=pd.read_csv(args.run/'manifest/evaluation_labels.csv')
    f=im.merge(labels,on='image_id',validate='one_to_one')
    f=f[(f.dataset=='VisA')&(f.split=='evaluation')&(f.label==1)]
    split=pd.read_csv(args.root/'datasets/VisA/split_csv/1cls.csv')
    masks={str(args.root/'datasets/VisA'/r.image):str(args.root/'datasets/VisA'/r.mask) for r in split.itertuples() if isinstance(r.mask,str) and r.mask}
    rows=[]
    for category,g in f.groupby('category'):
        base=args.run/'baselines/VisA'/category
        rois={r['image_id']:r for r in map(json.loads,(base/'regions.jsonl').read_text().splitlines())}
        for r in g.itertuples():
            with Image.open(masks[r.path]) as handle:mask=np.asarray(handle.convert('L'))>0
            assert mask.shape==(r.height,r.width) and mask.sum()>0
            total=int(mask.sum());record=dict(image_id=r.image_id,category=category,original_gt_pixels=total)
            original_mask=Image.fromarray(mask.astype(np.uint8))
            for side in [256,512]:
                resized=np.asarray(original_mask.resize((side,side),Image.Resampling.NEAREST))
                record[f'gt_pixels_{side}']=int(resized.sum());record[f'gt_vanished_{side}']=int(not resized.any())
            regions=rois[r.image_id]['regions']
            for name,scale in [('tight',1.),('context15',1.5),('context30',3.)]:
                hit=0;area=0;covered=0
                if regions:
                    box=regions[0]['box'] if scale==1 else context_box(regions[0]['box'],(r.width,r.height),scale)
                    x1,y1,x2,y2=box;covered=int(mask[y1:y2,x1:x2].sum());hit=int(covered>0);area=(x2-x1)*(y2-y1)
                record[name+'_hit']=hit;record[name+'_gt_fraction']=covered/total
                record[name+'_image_fraction']=area/(r.width*r.height)
            rows.append(record)
        print('COVERAGE',category,len(g),flush=True)
    data=pd.DataFrame(rows);data.to_csv(args.out/'per_image.csv',index=False)
    data.groupby('category').mean(numeric_only=True).to_csv(args.out/'category_means.csv')
    (args.out/'protocol.json').write_text(json.dumps(dict(source_only=True,gt_used_for_evaluation_only=True,
        image_count=len(data),caveat='Nearest-neighbor mask survival is a geometric sampling diagnostic, not evidence that RGB information or model sensitivity is absent.'),indent=2))
    print(data.drop(columns=['image_id','category']).mean().to_string())

if __name__=='__main__':main()
