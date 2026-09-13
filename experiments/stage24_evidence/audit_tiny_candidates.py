"""Mask-free fixed-budget candidates, followed by native-resolution coverage audit."""
import json,math
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from local_learning import RUN,ROOT
from common import sha256

BASE=ROOT/'results/stage25_tiny_defects'
OUT=BASE/'candidate_v1'

def box(x,y):
    x=max(0,min(384,int(x)-64));y=max(0,min(384,int(y)-64))
    return (x,y,x+128,y+128)

def iou(a,b):
    area=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    return area/(32768-area)

def candidates(a):
    assert a.shape==(512,512) and np.isfinite(a).all()
    work=a.copy();top=[]
    # Deterministic row-major tie order; NMS acts on centers, IoU removes duplicates.
    for _ in range(64):
        index=int(work.argmax());y,x=divmod(index,512)
        if not np.isfinite(work[y,x]):break
        candidate=box(x,y)
        if all(iou(candidate,v)<.5 for v in top):top.append(candidate)
        work[max(0,y-64):min(512,y+65),max(0,x-64):min(512,x+65)]=-np.inf
        if len(top)==8:break
    fallback=[box(x,y) for y in [64,192,320,448] for x in [64,192,320,448]]
    def fill(values):
        result=[]
        for value in values+fallback:
            if all(iou(value,v)<.5 for v in result):result.append(value)
            if len(result)==8:return result
        raise AssertionError('Unable to form eight windows')
    grid4=[box(x,y) for y in [128,384] for x in [128,384]]
    grid8=[box(x,y) for y in [128,384] for x in [64,192,320,448]]
    return dict(top8=fill(top),grid8=grid8,mixed8=fill(top[:4]+grid4))

def native(b,w,h):return (math.floor(b[0]*w/512),math.floor(b[1]*h/512),math.ceil(b[2]*w/512),math.ceil(b[3]*h/512))

def main():
    # Bounds, deterministic ties, budget, and exact quarter-field dimensions.
    for a in [np.zeros((512,512),np.float32),np.random.default_rng(42).random((512,512)).astype(np.float32)]:
        out=candidates(a);assert out==candidates(a)
        for boxes in out.values():
            assert len(boxes)==8 and len(set(boxes))==8
            assert all(0<=x<r<=512 and 0<=y<t<=512 and r-x==t-y==128 for x,y,r,t in boxes)
    OUT.mkdir(parents=True,exist_ok=False);cv2.setNumThreads(1)
    f=pd.read_csv(BASE/'v1/inference_manifest.csv');f=f[f.role=='source_fit']
    assert f.dataset.eq('VisA').all() and len(f)==1462
    protocol=dict(code_sha256=sha256(__file__),images=len(f),scope='All source_fit images; existing maps make full coverage audit affordable',budget=8,window='128x128 on 512 grid; corresponding native rectangle',top_nms_radius=64,duplicate_iou=.5,primary='mixed8',controls=['top8','grid8'],recall='A single window covers >=50% or >=90% native component pixels; union coverage reported separately',continue_gate='mixed8 tiny component 50%-coverage recall, image then category macro >=0.90; project gate not paper standard',target_used=False,tests_passed=True)
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
    rows=[];lookup={}
    for r in f.sort_values(['category','image_id']).itertuples():
        a=np.load(RUN/'20260911_d/fixed_bank/VisA'/r.category/'maps'/f'{r.image_id}.npz')['anomaly_map']
        choices=candidates(a);lookup[r.image_id]=choices
        for method,boxes in choices.items():
            for i,b in enumerate(boxes):
                n=native(b,r.width,r.height);rows.append(dict(image_id=r.image_id,category=r.category,method=method,candidate=i,x=b[0],y=b[1],right=b[2],bottom=b[3],native_x=n[0],native_y=n[1],native_right=n[2],native_bottom=n[3]))
    pd.DataFrame(rows).to_csv(OUT/'candidates_no_labels.csv',index=False)
    # GT is first accessed after every candidate has been persisted.
    gt=pd.read_csv(BASE/'v1/partition_manifest.csv');gt=gt[(gt.role=='source_fit')&(gt.label==1)]
    comp=pd.read_csv(BASE/'v1/components.csv');comp=comp[comp.role=='source_fit']
    records=[]
    for category,g in gt.groupby('category'):
        for r in g.itertuples():
            with Image.open(r.mask_path) as h:mask=np.asarray(h.convert('L'))>0
            count,labels,stats,_=cv2.connectedComponentsWithStats(mask.astype(np.uint8),connectivity=8)
            expected=comp[comp.image_id==r.image_id].set_index('component_id');assert count-1==len(expected)
            for method,boxes in lookup[r.image_id].items():
                best=np.zeros(count);covered=np.zeros(mask.shape,bool)
                for b in boxes:
                    x,y,right,bottom=native(b,r.width,r.height)
                    hits=np.bincount(labels[y:bottom,x:right].ravel(),minlength=count)
                    best=np.maximum(best,hits);covered[y:bottom,x:right]=True
                union=np.bincount(labels[covered],minlength=count)
                for c in range(1,count):
                    area=int(stats[c,cv2.CC_STAT_AREA]);assert area==int(expected.loc[c,'area_native'])
                    records.append(dict(image_id=r.image_id,category=category,stratum=r.stratum,component_id=c,kind=expected.loc[c,'kind'],method=method,best_window_coverage=float(best[c]/area),union_coverage=float(union[c]/area),recalled50=bool(best[c]>=.5*area),recalled90=bool(best[c]>=.9*area)))
        pd.DataFrame(records).to_csv(OUT/'coverage.partial.csv',index=False);print('COMPLETE',category,flush=True)
    coverage=pd.DataFrame(records);coverage.to_csv(OUT/'component_coverage.csv',index=False)
    tiny=coverage[coverage.kind=='tiny_compact'];metrics=['best_window_coverage','union_coverage','recalled50','recalled90']
    perimage=tiny.groupby(['category','image_id','method'])[metrics].mean();perimage.to_csv(OUT/'tiny_per_image.csv')
    table=perimage.groupby(['category','method']).mean();table.to_csv(OUT/'tiny_category.csv');macro=table.groupby('method').mean();macro.to_csv(OUT/'tiny_macro.csv')
    score=float(macro.loc['mixed8','recalled50'])
    (OUT/'complete.json').write_text(json.dumps(dict(status='complete',images=len(f),anomalous_images=len(gt),tiny_instances=int(len(tiny)/3),gate_passed=score>=.90,gate_value=score,limitations='Source-fit feasibility only; GT coverage not detection recall; fixed teacher and budgets; exact connected regions are not guaranteed physical instances.'),indent=2))
    print(macro.to_string())

if __name__=='__main__':main()
