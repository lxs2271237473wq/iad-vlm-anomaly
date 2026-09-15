"""Non-destructive manifests for tiny defect vs normal nuisance research."""
import hashlib,json
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from local_learning import RUN,ROOT
from common import sha256

OUT=ROOT/'results/stage25_tiny_defects/v1'

def main():
    OUT.mkdir(parents=True,exist_ok=False);cv2.setNumThreads(1)
    f=pd.read_csv(RUN/'20260910_a/manifest/images.csv').merge(pd.read_csv(RUN/'20260910_a/manifest/evaluation_labels.csv'),on='image_id',validate='one_to_one')
    cats=sorted(f[f.dataset=='VisA'].category.unique(),key=lambda x:hashlib.sha256(('tiny-v1/'+x).encode()).hexdigest())
    roles={c:('source_fit' if i<8 else 'source_validation' if i<10 else 'source_development_holdout') for i,c in enumerate(cats)}
    protocol=dict(code_sha256=sha256(__file__),version='v1',primary='compact tiny components: native area/image area <=0.001 AND bbox max side after square resize512 <=32',secondary='area<=0.001 but max side>32: thin_or_extended_small_area',connectivity=8,no_component_removal=True,source_categories=roles,source_split='8/2/2 category-disjoint hash order; all previously inspected, not independent test',target_role='AD2 public exploration only; no target anomaly training',hard_normal='D_raw above source/category original normal-calibration q95, fixed teacher tag only; not physical nuisance annotation',future_test='Not present: external untouched data or locked official private evaluation required',gt_use='Partition and evaluation only; excluded from inference manifest',original_images_unchanged=True)
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
    official=pd.read_csv(ROOT/'datasets/VisA/split_csv/1cls.csv')
    visa_masks={str(ROOT/'datasets/VisA'/r.image):str(ROOT/'datasets/VisA'/r.mask) for r in official.itertuples() if isinstance(r.mask,str) and r.mask}
    rows=[];components=[]
    for (dataset,category),g in f.groupby(['dataset','category'],sort=True):
        base=RUN/('20260911_d' if dataset=='VisA' else '20260911_e')/'fixed_bank'/dataset/category
        pred=pd.read_csv(base/'predictions.csv').set_index('image_id')
        cal_ids=g.loc[g.split=='calibration','image_id'];threshold=float(pred.loc[cal_ids,'D_raw'].quantile(.95))
        for r in g.itertuples():
            role=('normal_reference' if r.split=='train' else 'normal_calibration' if r.split=='calibration' else roles[category] if dataset=='VisA' else 'target_public_development')
            item=dict(image_id=r.image_id,dataset=dataset,category=category,original_split=r.split,role=role,path=r.path,label=r.label,scene_group=r.scene_group,width=r.width,height=r.height,stratum='normal',component_count=0,tiny_count=0,thin_count=0,mask_path='',hard_normal=False,teacher_q95=threshold)
            if r.split=='evaluation' and r.label==0:item['hard_normal']=bool(pred.loc[r.image_id,'D_raw']>threshold)
            if r.label==1:
                assert r.split=='evaluation'
                mask_path=visa_masks[r.path] if dataset=='VisA' else str(ROOT/'datasets/MVTec_AD_2'/category/'test_public/ground_truth/bad'/(Path(r.path).stem+'_mask.png'))
                with Image.open(mask_path) as h:mask=np.asarray(h.convert('L'))>0
                assert mask.shape==(r.height,r.width),(mask_path,mask.shape,(r.height,r.width))
                count,_,stats,_=cv2.connectedComponentsWithStats(mask.astype(np.uint8),connectivity=8)
                assert count>1;types=[]
                for component in range(1,count):
                    x,y,w,h,area=map(int,stats[component]);fraction=area/(r.width*r.height);span=max(w*512/r.width,h*512/r.height)
                    kind='tiny_compact' if fraction<=.001 and span<=32 else 'thin_or_extended_small_area' if fraction<=.001 else 'larger'
                    types.append(kind);components.append(dict(image_id=r.image_id,dataset=dataset,category=category,role=role,component_id=component,kind=kind,area_native=area,area_fraction=fraction,equivalent_area_512=fraction*512**2,max_bbox_side_512=span,x=x,y=y,width=w,height=h))
                tiny=types.count('tiny_compact');thin=types.count('thin_or_extended_small_area')
                stratum='tiny_only' if tiny==len(types) else 'mixed_tiny' if tiny else 'thin_only' if thin==len(types) else 'other_anomaly'
                item.update(stratum=stratum,component_count=len(types),tiny_count=tiny,thin_count=thin,mask_path=mask_path)
            rows.append(item)
        print('COMPLETE',dataset,category,flush=True)
    table=pd.DataFrame(rows);table.to_csv(OUT/'partition_manifest.csv',index=False)
    pd.DataFrame(components).to_csv(OUT/'components.csv',index=False)
    table[['image_id','dataset','category','original_split','role','path','width','height']].to_csv(OUT/'inference_manifest.csv',index=False)
    evals=table[table.original_split=='evaluation']
    evals.groupby(['dataset','category','role','stratum']).size().rename('images').to_csv(OUT/'stratum_counts.csv')
    evals[evals.label==0].groupby(['dataset','category','role']).agg(normal_images=('image_id','size'),teacher_hard_normals=('hard_normal','sum')).to_csv(OUT/'normal_counts.csv')
    for role,g in evals.groupby('role'):
        g.to_csv(OUT/f'{role}.csv',index=False)
    # Every normal retained in primary tiny evaluation; mixed images only auxiliary.
    evals[(evals.label==0)|(evals.stratum=='tiny_only')].to_csv(OUT/'primary_tiny_evaluation.csv',index=False)
    assert table.image_id.is_unique
    assert evals.groupby(['dataset','category','scene_group']).role.nunique().max()==1
    assert not set(evals[evals.role=='source_fit'].category)&set(evals[evals.role=='source_validation'].category)
    (OUT/'complete.json').write_text(json.dumps(dict(status='complete',images=len(table),components=len(components),evaluation_images=len(evals),source_roles=roles,independent_test_created=False),indent=2))
    print(evals.groupby(['role','stratum']).size().to_string())

if __name__=='__main__':main()
