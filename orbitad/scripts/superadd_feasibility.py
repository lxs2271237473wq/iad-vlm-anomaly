"""Hardware smoke only; NOT a SuperADD accuracy reproduction."""
import ast,json,time
from pathlib import Path
from math import ceil,floor
from itertools import product
import numpy as np
import torch
import timm
from PIL import Image
from torchvision.transforms.functional import to_tensor

root=Path('/root/private_data/iad-vlm-anomaly')
source=root/'orbitad/third_party/SuperADD/tracks/industrial/src/industrial/model.py'
torch.set_num_threads(8)
tree=ast.parse(source.read_text())
classes=[n for n in tree.body if isinstance(n,ast.ClassDef) and n.name in ('PreProcessing','PatchedExecution')]
exec(compile(ast.Module(body=classes,type_ignores=[]),str(source),'exec'))
torch.manual_seed(42); torch.backends.cuda.matmul.allow_tf32=False
start=time.time()
model=timm.create_model('vit_base_patch16_dinov3.lvd1689m',pretrained=True,num_classes=0,img_size=640).eval().cuda()
layers=[2,5,8,11]
patcher=PatchedExecution(640,128,16)
pre=PreProcessing('cuda',0.625)
@torch.inference_mode()
def backbone(x):
    result=[[] for _ in layers]
    for part in x.split(1):
        with torch.autocast('cuda',dtype=torch.bfloat16):
            out=model.forward_intermediates(part,indices=layers,norm=False,output_fmt='NLC',intermediates_only=True)
        for dest,f in zip(result,out): dest.append(f.float())
    return [torch.cat(v) for v in result]
report={'kind':'hardware_smoke_not_accuracy','official_commit':'44cf25144442fbbc1334ea59d1632327a4376d1a','backbone':'DINOv3-B adaptation; official H+ unavailable locally','layers':layers,'patch':640,'overlap':128,'resize_factor':0.625,'cases':[]}
for cat in ('can','wallplugs'):
    path=sorted((root/'datasets/MVTec_AD_2'/cat/'train/good').glob('*.png'))[0]
    with Image.open(path) as im: x=to_tensor(im.convert('RGB'))[None].cuda()
    tick=time.time(); features=patcher(pre(x),backbone)
    assert all(np.isfinite(f).all() for f in features)
    # Verify chunked nearest distance against full matrix on a small random slice.
    q=torch.from_numpy(features[0].reshape(-1,768)[:64]).cuda()
    keys=torch.from_numpy(features[0].reshape(-1,768)[64:320]).cuda()
    exact=torch.cdist(q,keys).min(1).values
    chunked=torch.stack([torch.cdist(q,k).min(1).values for k in keys.split(64)]).min(0).values
    assert torch.allclose(exact,chunked,atol=1e-3,rtol=1e-4)
    case={'category':cat,'input_shape':list(x.shape),'feature_shapes':[list(f.shape) for f in features],'seconds':time.time()-tick,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'nn_chunk_max_error':float((exact-chunked).abs().max())}
    report['cases'].append(case); print(case,flush=True)
    del features,x; torch.cuda.empty_cache()
report['total_seconds']=time.time()-start
out=root/'orbitad/results/superadd_feasibility'; out.mkdir(parents=True,exist_ok=True)
(out/'smoke.json').write_text(json.dumps(report,indent=2))
print('SMOKE_COMPLETE',flush=True)
