"""Resume A18 after completed state extraction; generate routed patch maps only."""
from pathlib import Path
from math import ceil, floor
from itertools import product
import ast
import csv
import json
import time

import numpy as np
import torch
import torch.nn.functional as F
import timm
from PIL import Image
from torchvision.transforms.functional import to_tensor


ROOT=Path('/root/private_data/iad-vlm-anomaly')
DATA=ROOT/'datasets/MVTec_AD_2'
A10=ROOT/'orbitad/results/a10_multilayer_v1'
OUT=ROOT/'orbitad/results/a18_state_routed_memory_v1'
LAYERS=[2,5,8,11]
K_STATES=4


def main():
    torch.set_num_threads(8); torch.manual_seed(42); np.random.seed(42)
    torch.backends.cuda.matmul.allow_tf32=False
    assert (OUT/'STATES_COMPLETE.json').exists()
    assert not (OUT/'MAPS_COMPLETE.json').exists()
    meta={split:list(csv.DictReader(open(OUT/f'{split}_meta.csv'))) for split in ('validation','public')}
    categories=sorted({r['category'] for r in meta['public']})

    source=ROOT/'orbitad/third_party/SuperADD/tracks/industrial/src/industrial/model.py'
    tree=ast.parse(source.read_text())
    classes=[n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='PatchedExecution']
    exec(compile(ast.Module(body=classes,type_ignores=[]),str(source),'exec'),globals())
    patcher=PatchedExecution(640,128,16)
    model=timm.create_model('vit_base_patch16_dinov3.lvd1689m',pretrained=True,num_classes=0,img_size=640).eval().cuda()
    mean=torch.tensor([.485,.456,.406],device='cuda')[None,:,None,None]
    std=torch.tensor([.229,.224,.225],device='cuda')[None,:,None,None]

    @torch.inference_mode()
    def backbone(x):
        result=[[] for _ in LAYERS]
        for part in x.split(1):
            with torch.autocast('cuda',dtype=torch.bfloat16):
                out=model.forward_intermediates(part,indices=LAYERS,norm=False,output_fmt='NLC',intermediates_only=True)
            for dest,feature in zip(result,out): dest.append(feature.float())
        return [torch.cat(v) for v in result]

    @torch.inference_mode()
    def extract(path):
        with Image.open(path) as im: x=to_tensor(im.convert('RGB'))[None].cuda()
        shape=(int(x.shape[-2]*.625),int(x.shape[-1]*.625))
        assert min(shape)>=640,(path,shape)
        x=F.interpolate(x,size=shape,mode='bicubic',align_corners=False,antialias=True)
        return patcher((x-mean)/std,backbone)

    @torch.inference_mode()
    def score(features,state_memories,states):
        maps=[]
        for li,feature in enumerate(features):
            h,w=feature.shape[1:3]; query=torch.from_numpy(feature.reshape(-1,768)).cuda(); values=[]
            for qc in query.split(256):
                best=torch.full((len(qc),),float('inf'),device='cuda')
                for state in states:
                    for mc in state_memories[int(state)][li].split(4096):
                        best=torch.minimum(best,torch.cdist(qc,mc,compute_mode='use_mm_for_euclid_dist').min(1).values)
                values.append(best)
            maps.append(torch.cat(values).reshape(h,w)/768)
        result=torch.stack(maps).mean(0).cpu().numpy()
        assert np.isfinite(result).all()
        return result

    progress=json.loads((OUT/'progress.json').read_text())
    for category in categories:
        tick=time.time(); dest=OUT/category/'state_patch640'
        if (dest/'CATEGORY_COMPLETE.json').exists():
            print('SKIP_COMPLETE',category,flush=True); continue
        archive=np.load(A10/'patch640'/category/'memory.npz')
        references=json.loads((A10/'patch640'/category/'references.json').read_text())
        state_archive=np.load(OUT/category/'global_state.npz')
        centers=state_archive['centers']; train_desc=state_archive['train_desc']
        train_paths=sorted(p for p in (DATA/category/'train/good').iterdir() if p.suffix.lower() in {'.png','.jpg','.jpeg','.tif','.tiff','.bmp'})
        assert len(train_paths)==len(train_desc)
        labels=np.argmax(train_desc@centers.T,axis=1)
        train_state={str(p.relative_to(DATA)).replace('\\','/'):int(s) for p,s in zip(train_paths,labels)}
        token_states=np.asarray([train_state[r['path']] for r in references],dtype=np.int64)
        state_memories=[]; counts={}
        for state in range(K_STATES):
            idx=np.flatnonzero(token_states==state); assert len(idx)>0
            counts[state]=int(len(idx))
            state_memories.append([torch.from_numpy(archive[str(layer)][idx].astype('float32')).cuda() for layer in LAYERS])
        state_meta=json.loads((OUT/category/'state_meta.json').read_text())
        timings={}
        for split in ('validation','public'):
            rows=[r for r in meta[split] if r['category']==category]
            split_dest=dest/split; split_dest.mkdir(parents=True,exist_ok=True); start=time.time()
            routes=state_meta[f'{split}_routes']
            for number,row in enumerate(rows,1):
                index=int(row['map_index']); route=routes[str(index)]
                np.save(split_dest/f'{index:05d}.npy',score(extract(DATA/row['image_path']),state_memories,route))
                if number%30==0: print(category,split,number,len(rows),route,flush=True)
            timings[split]={'images':len(rows),'seconds':time.time()-start}
        record={'phase':'state_patch_maps','category':category,'state_memory_counts':counts,'timings':timings,'seconds':time.time()-tick}
        progress.append(record); (OUT/'progress.json').write_text(json.dumps(progress,indent=2))
        (dest/'CATEGORY_COMPLETE.json').write_text(json.dumps(record,indent=2)); print(record,flush=True)
        del state_memories; torch.cuda.empty_cache()
    (OUT/'MAPS_COMPLETE.json').write_text(json.dumps({'status':'complete','categories':len(categories),'resumed':True}))
    print('A18_MAPS_COMPLETE',flush=True)


if __name__=='__main__': main()
