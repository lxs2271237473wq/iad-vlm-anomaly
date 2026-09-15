"""Controlled SuperADD-inspired B adaptation; not official reproduction."""
import ast,csv,json,time,hashlib
from pathlib import Path
from math import ceil,floor
from itertools import product
import numpy as np
import torch
import torch.nn.functional as F
import timm
from PIL import Image
from torchvision.transforms.functional import to_tensor

ROOT=Path('/root/private_data/iad-vlm-anomaly')
DATA=ROOT/'datasets/MVTec_AD_2'
A1=ROOT/'orbitad/results/a1_anomaly_score_sensitivity'
OUT=ROOT/'orbitad/results/a10_multilayer_v1'
LAYERS=[2,5,8,11]

def main():
    torch.set_num_threads(8); torch.manual_seed(42); np.random.seed(42)
    torch.backends.cuda.matmul.allow_tf32=False
    OUT.mkdir(parents=True,exist_ok=False)
    source=ROOT/'orbitad/third_party/SuperADD/tracks/industrial/src/industrial/model.py'
    tree=ast.parse(source.read_text()); classes=[n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='PatchedExecution']
    exec(compile(ast.Module(body=classes,type_ignores=[]),str(source),'exec'),globals())
    patcher=PatchedExecution(640,128,16)
    meta={s:list(csv.DictReader(open(A1/f'{s}_patch_maps.csv'))) for s in ('validation','public')}
    # Inference only receives image paths/category/index; labels are copied for later evaluation.
    for s,rows in meta.items():
        with open(OUT/f'{s}_meta.csv','w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    protocol={'model':'vit_base_patch16_dinov3.lvd1689m','layers':LAYERS,'methods':['full512','patch640'],'patch_overlap':128,'patch_resize_factor':0.625,'memory_per_image_per_layer':64,'memory_limit_per_layer':32768,'sampling':'fixed 8x8 spatial grid, paired across layers; seed20260913 cap','score':'mean over 4 layers of Euclidean NN distance / 768','augmentation':False,'postprocessing':False,'train':'train/good all','calibration':'validation/good','test':'public development only','official_reproduction':False,'budget_note':'matched memory count; patch method uses more computation, measured separately','code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'upstream_model_sha256':hashlib.sha256(source.read_bytes()).hexdigest()}
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
    mean=torch.tensor([.485,.456,.406],device='cuda')[None,:,None,None]
    std=torch.tensor([.229,.224,.225],device='cuda')[None,:,None,None]
    progress=[]
    for method in ('full512','patch640'):
        model=timm.create_model(protocol['model'],pretrained=True,num_classes=0,img_size=512 if method=='full512' else 640).eval().cuda()
        @torch.inference_mode()
        def backbone(x):
            result=[[] for _ in LAYERS]
            for part in x.split(1):
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    out=model.forward_intermediates(part,indices=LAYERS,norm=False,output_fmt='NLC',intermediates_only=True)
                for dest,f in zip(result,out): dest.append(f.float())
            return [torch.cat(v) for v in result]
        @torch.inference_mode()
        def extract(path):
            with Image.open(path) as im:
                im=im.convert('RGB')
                if method=='full512': im=im.resize((512,512),Image.Resampling.BICUBIC)
                x=to_tensor(im)[None].cuda()
            if method=='patch640':
                shape=(int(x.shape[-2]*.625),int(x.shape[-1]*.625))
                assert min(shape)>=640, f'Image too small for official patch policy: {path} {shape}'
                x=F.interpolate(x,size=shape,mode='bicubic',align_corners=False,antialias=True)
            x=(x-mean)/std
            if method=='full512': return [f.reshape(1,32,32,768).cpu().numpy() for f in backbone(x)]
            return patcher(x,backbone)
        @torch.inference_mode()
        def score(features,memories):
            maps=[]
            for arr,m in zip(features,memories):
                h,w=arr.shape[1:3]; q=torch.from_numpy(arr.reshape(-1,768)).cuda(); vals=[]
                for qc in q.split(256):
                    best=torch.full((len(qc),),float('inf'),device='cuda')
                    for mc in m.split(4096):
                        best=torch.minimum(best,torch.cdist(qc,mc,compute_mode='use_mm_for_euclid_dist').min(1).values)
                    vals.append(best)
                maps.append(torch.cat(vals).reshape(h,w)/768)
            a=torch.stack(maps).mean(0).cpu().numpy()
            assert np.isfinite(a).all()
            return a
        for cat in sorted({r['category'] for r in meta['public']}):
            start=time.time(); torch.cuda.reset_peak_memory_stats()
            dest=OUT/method/cat; dest.mkdir(parents=True)
            paths=sorted(p for p in (DATA/cat/'train/good').iterdir() if p.suffix.lower() in {'.png','.jpg','.jpeg','.tif','.tiff','.bmp'})
            bank=[[] for _ in LAYERS]; refs=[]
            for n,path in enumerate(paths):
                features=extract(path); h,w=features[0].shape[1:3]
                ys=np.rint(np.linspace(0,h-1,8)).astype(int); xs=np.rint(np.linspace(0,w-1,8)).astype(int)
                idx=np.array([y*w+x for y in ys for x in xs])
                assert len(np.unique(idx))==64
                for b,f in zip(bank,features): b.append(f.reshape(-1,768)[idx].astype('float16'))
                refs.extend({'path':str(path.relative_to(DATA)),'patch':int(j),'grid':[h,w]} for j in idx)
                if (n+1)%50==0: print(method,cat,'memory',n+1,len(paths),flush=True)
            banks=[np.concatenate(b) for b in bank]
            if len(banks[0])>32768:
                idx=np.sort(np.random.default_rng(20260913).choice(len(banks[0]),32768,replace=False))
                banks=[b[idx] for b in banks]; refs=[refs[j] for j in idx]
            np.savez(dest/'memory.npz',**{str(l):b for l,b in zip(LAYERS,banks)})
            (dest/'references.json').write_text(json.dumps(refs))
            memories=[torch.from_numpy(b.astype('float32')).cuda() for b in banks]
            build_seconds=time.time()-start; timings={}
            for split,rows in meta.items():
                tick=time.time(); subset=[r for r in rows if r['category']==cat]
                (dest/split).mkdir()
                for n,row in enumerate(subset):
                    a=score(extract(DATA/row['image_path']),memories)
                    np.save(dest/split/f"{int(row['map_index']):05d}.npy",a)
                    if (n+1)%30==0: print(method,cat,split,n+1,len(subset),flush=True)
                timings[split]={'images':len(subset),'seconds':time.time()-tick}
            record={'method':method,'category':cat,'memory_count_per_layer':len(banks[0]),'memory_build_seconds':build_seconds,'inference':timings,'total_seconds':time.time()-start,'peak_gpu_bytes':torch.cuda.max_memory_allocated()}
            progress.append(record); (OUT/'progress.json').write_text(json.dumps(progress,indent=2)); print(record,flush=True)
            del memories,banks,bank,features; torch.cuda.empty_cache()
        del model; torch.cuda.empty_cache()
    (OUT/'MAPS_COMPLETE.json').write_text(json.dumps({'categories':8,'methods':2,'status':'maps_complete'}))
    print('MAPS_COMPLETE',flush=True)

if __name__=='__main__': main()
