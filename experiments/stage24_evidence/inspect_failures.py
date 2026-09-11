"""Posthoc visual audit. Labels/masks select examples, never train a model."""
from pathlib import Path
import json
import cv2
import numpy as np
import pandas as pd
from PIL import Image,ImageDraw

root=Path('/root/private_data/iad-vlm-anomaly');runs=root/'results/stage24_evidence'
out=runs/'20260911_f/visual_audit';out.mkdir(parents=True,exist_ok=False)
manifest=pd.read_csv(runs/'20260910_a/manifest/images.csv').merge(pd.read_csv(runs/'20260910_a/manifest/evaluation_labels.csv'),on='image_id',validate='one_to_one')
visa=pd.read_csv(root/'datasets/VisA/split_csv/1cls.csv')
masks={str(root/'datasets/VisA'/r.image):str(root/'datasets/VisA'/r.mask) for r in visa.itertuples() if isinstance(r.mask,str) and r.mask}
records=[]
for dataset,category,stage in [('AD2','can','20260911_e'),('AD2','wallplugs','20260911_e'),('AD2','sheet_metal','20260911_e'),('VisA','macaroni2','20260911_d')]:
    base=runs/stage/'fixed_bank'/dataset/category
    pred=pd.read_csv(base/'predictions.csv')[['image_id','D_raw','D_z']]
    f=manifest[(manifest.dataset==dataset)&(manifest.category==category)&(manifest.split=='evaluation')].merge(pred,on='image_id',validate='one_to_one')
    selected=pd.concat([f[f.label==0].sort_values(['D_raw','image_id'],ascending=[False,True]).head(3),f[f.label==1].sort_values(['D_raw','image_id']).head(3)])
    canvas=Image.new('RGB',(960,6*340),(255,255,255));draw=ImageDraw.Draw(canvas)
    for i,r in enumerate(selected.itertuples()):
        original=np.array(Image.open(r.path).convert('RGB').resize((320,300)))
        marked=original.copy()
        if r.label==1:
            path=root/'datasets/MVTec_AD_2'/category/'test_public/ground_truth/bad'/f'{Path(r.path).stem}_mask.png' if dataset=='AD2' else Path(masks[r.path])
            mask=np.array(Image.open(path).convert('L').resize((320,300),Image.Resampling.NEAREST))>0
            contours,_=cv2.findContours(mask.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(marked,contours,-1,(0,255,0),2)
        a=np.load(base/'maps'/f'{r.image_id}.npz')['anomaly_map']
        a=cv2.resize(a,(320,300));a=(a-a.min())/max(float(a.max()-a.min()),1e-8)
        heat=cv2.cvtColor(cv2.applyColorMap(np.uint8(255*a),cv2.COLORMAP_TURBO),cv2.COLOR_BGR2RGB)
        overlay=np.uint8(.55*original+.45*heat)
        for j,arr in enumerate([original,marked,overlay]):canvas.paste(Image.fromarray(arr),(j*320,i*340+35))
        draw.text((5,i*340+3),f'{category} label={r.label} D_z={r.D_z:.3f} id={r.image_id}',fill='black')
        records.append(dict(dataset=dataset,category=category,image_id=r.image_id,label=r.label,D_z=r.D_z,path=r.path))
    canvas.save(out/f'{dataset}_{category}.jpg',quality=93)
pd.DataFrame(records).to_csv(out/'selected_images.csv',index=False)
(out/'protocol.json').write_text(json.dumps(dict(selection='3 highest scoring normals and 3 lowest scoring anomalies per specified category',columns=['original','GT contour if anomalous','per-image normalized heatmap'],warning='Posthoc diagnostic selection; heatmap colors cannot compare absolute scores across images.')))
