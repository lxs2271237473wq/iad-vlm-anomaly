import hashlib
import math
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms as T

def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''): h.update(b)
    return h.hexdigest()

def image_transform(size, clip=False):
    mean=(.48145466,.4578275,.40821073) if clip else (.485,.456,.406)
    std=(.26862954,.26130258,.27577711) if clip else (.229,.224,.225)
    # Explicit full-field resize: no center crop and no hidden coordinate offset.
    return T.Compose([T.Resize((size,size),interpolation=T.InterpolationMode.BICUBIC if clip else T.InterpolationMode.BILINEAR),T.ToTensor(),T.Normalize(mean,std)])

def map_box_to_original(box, map_shape, image_size):
    mh,mw=map_shape; w,h=image_size
    x1,y1,x2,y2=box
    return (max(0,min(w-1,math.floor(x1*w/mw))),max(0,min(h-1,math.floor(y1*h/mh))),
            max(1,min(w,math.ceil(x2*w/mw))),max(1,min(h,math.ceil(y2*h/mh))))

def context_box(box,image_size,factor=1.5):
    x1,y1,x2,y2=box; w,h=image_size
    cx,cy=(x1+x2)/2,(y1+y2)/2
    bw,bh=max(8,x2-x1)*factor,max(8,y2-y1)*factor
    return max(0,math.floor(cx-bw/2)),max(0,math.floor(cy-bh/2)),min(w,math.ceil(cx+bw/2)),min(h,math.ceil(cy+bh/2))

def candidates(anomaly_map,image_size,top_k=3):
    a=np.asarray(anomaly_map,dtype=np.float32)
    if a.ndim!=2 or not np.isfinite(a).all(): raise ValueError('Invalid anomaly map')
    mask=(a>np.quantile(a,.97)).astype(np.uint8)
    n,ids,stats,_=cv2.connectedComponentsWithStats(mask,8)
    regions=[]
    for idx in range(1,n):
        x,y,w,h,area=map(int,stats[idx])
        if area<4: continue
        values=a[ids==idx]
        regions.append((float(values.max()),float(values.mean()),(x,y,x+w,y+h),area))
    regions.sort(key=lambda r:(-r[0],-r[1],r[2]))
    rows=[]
    for rank,(peak,mean,box,area) in enumerate(regions[:top_k],1):
        original=map_box_to_original(box,a.shape,image_size)
        rows.append(dict(rank=rank,map_box=list(box),box=list(original),context_box=list(context_box(original,image_size)),peak=peak,mean=mean,area=area))
    return rows

def robust_fit(values):
    v=np.asarray(values,dtype=float)
    if len(v)<2 or not np.isfinite(v).all(): raise ValueError('Invalid calibration data')
    center=float(np.median(v)); mad=float(1.4826*np.median(np.abs(v-center)))
    scale=mad if mad>1e-8 else max(float(v.std()),1e-8)
    return dict(center=center,scale=scale,n=len(v),mad=mad,fallback_std=mad<=1e-8)

def robust_apply(values,params):
    return (np.asarray(values)-params['center'])/params['scale']

OBJECTS={'fruit_jelly':'fruit jelly','sheet_metal':'sheet metal','wallplugs':'wall plugs',
         'macaroni1':'macaroni','macaroni2':'macaroni','pcb1':'printed circuit board',
         'pcb2':'printed circuit board','pcb3':'printed circuit board','pcb4':'printed circuit board',
         'pipe_fryum':'pipe-shaped fryum snack','chewinggum':'chewing gum','cashew':'cashew nut'}

def prompts(category):
    obj=OBJECTS.get(category,category)
    return [f'a normal {obj}',f'a defect-free {obj}',f'a clean undamaged {obj}',
            f'a defective {obj}',f'an anomalous {obj}',f'a damaged or contaminated {obj}']
