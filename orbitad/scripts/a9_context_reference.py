"""Frozen A9 pilot: matched center memory, ring retrieval, center scoring."""
import csv, json, time, hashlib
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import timm
from timm.data import resolve_model_data_config
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, InterpolationMode

ROOT=Path('/root/private_data/iad-vlm-anomaly')
A1=ROOT/'orbitad/results/a1_anomaly_score_sensitivity'
OUT=ROOT/'orbitad/results/a9_context_reference_v1'
DATA=ROOT/'datasets/MVTec_AD_2'
METHODS=['raw','pca16','context_top1','context_top8_min','center_top8_min']

def ring(z):
    # Valid neighbor count avoids reflecting the center back into edge contexts.
    x=z.reshape(-1,32,32,768).permute(0,3,1,2)
    count=F.avg_pool2d(torch.ones_like(x[:,:1]),5,1,2,count_include_pad=True)*25-1
    pooled=(F.avg_pool2d(x,5,1,2,count_include_pad=True)*25-x)/count
    return F.normalize(pooled.permute(0,2,3,1).reshape(-1,1024,768),dim=-1)

def verify_ring():
    # Changing a center must not change its own ring descriptor.
    z=torch.randn(1,1024,768)
    for p in (0,16*32+16,1023):
        other=z.clone(); other[:,p]+=3
        assert torch.allclose(ring(z)[:,p],ring(other)[:,p],atol=2e-6)

def main():
    verify_ring()
    torch.manual_seed(20260914); np.random.seed(20260914)
    torch.backends.cuda.matmul.allow_tf32=False
    OUT.mkdir(parents=True,exist_ok=False)
    device='cuda'
    model=timm.create_model('vit_base_patch16_dinov3.lvd1689m',pretrained=True,num_classes=0,img_size=512).eval().to(device)
    cfg=resolve_model_data_config(model)
    transform=Compose([Resize((512,512),interpolation=InterpolationMode.BICUBIC,antialias=True),ToTensor(),Normalize(cfg['mean'],cfg['std'])])
    @torch.inference_mode()
    def features(paths):
        xs=[]
        for path in paths:
            with Image.open(path) as im: xs.append(transform(im.convert('RGB')))
        with torch.autocast('cuda',dtype=torch.bfloat16):
            f=model.forward_features(torch.stack(xs).to(device))
        if isinstance(f,dict):
            f=f['x_norm_patchtokens'] if 'x_norm_patchtokens' in f else f['x'][:,model.num_prefix_tokens:]
        else: f=f[:,model.num_prefix_tokens:]
        assert f.shape[1:]==(1024,768)
        return F.normalize(f.float(),dim=-1)
    meta={s:list(csv.DictReader(open(A1/f'{s}_patch_maps.csv'))) for s in ['validation','public']}
    maps={s:{m:np.lib.format.open_memmap(OUT/f'{s}_{m}.npy',mode='w+',dtype='float32',shape=(len(rows),32,32)) for m in METHODS} for s,rows in meta.items()}
    for s,rows in meta.items():
        with open(OUT/f'{s}_meta.csv','w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    protocol={'methods':METHODS,'input':512,'grid':32,'ring':'5x5 minus center; valid count','context_candidates':8,'seed':20260914,'memory_limit':32768,'normal_only_memory':True,'query_gt_used':False,'note':'Ring excludes center token, not global attention influence. Extra context-memory storage reported separately.'}
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
    coords=np.rint(np.linspace(0,31,8)).astype(int); idx=np.array([y*32+x for y in coords for x in coords])
    stats=[]
    for cat in sorted({r['category'] for r in meta['public']}):
        start=time.time()
        paths=sorted(p for p in (DATA/cat/'train/good').iterdir() if p.suffix.lower() in {'.png','.jpg','.jpeg','.bmp','.tif','.tiff'})
        centers=[]; contexts=[]; ids=[]
        for i in range(0,len(paths),4):
            z=features(paths[i:i+4]); c=ring(z)
            centers.append(z[:,idx].cpu().numpy().astype('float16')); contexts.append(c[:,idx].cpu().numpy().astype('float16'))
            ids.extend({'image':str(p.relative_to(DATA)),'patch':int(j)} for p in paths[i:i+4] for j in idx)
        center=np.concatenate(centers).reshape(-1,768); context=np.concatenate(contexts).reshape(-1,768)
        if len(center)>32768:
            select=np.sort(np.random.default_rng(20260913).choice(len(center),32768,replace=False))
            center=center[select]; context=context[select]; ids=[ids[j] for j in select]
        old=np.load(A1/'memory'/f'{cat}.npy')
        err=float(np.max(np.abs(old.astype('float32')-center.astype('float32')))) if old.shape==center.shape else None
        np.savez(OUT/f'{cat}_memory.npz',center=center,context=context)
        (OUT/f'{cat}_references.json').write_text(json.dumps(ids))
        mem=F.normalize(torch.from_numpy(center.astype('float32')).to(device),dim=-1)
        ctx=F.normalize(torch.from_numpy(context.astype('float32')).to(device),dim=-1)
        centered=mem-mem.mean(0); _,v=torch.linalg.eigh(centered.T@centered/max(len(mem)-1,1)); u=v[:,-16:]
        pm=mem-(mem@u)@u.T
        @torch.inference_mode()
        def score(z):
            rc=ring(z[None])[0]; out={m:[] for m in METHODS}
            for j in range(0,1024,128):
                q=z[j:j+128]; sims=q@mem.T
                best=sims.max(1).values
                out['raw'].append((1-best).clamp_min(0))
                out['center_top8_min'].append((1-sims.topk(8,dim=1).values).clamp_min(0).min(1).values)
                pq=q-(q@u)@u.T
                d=(pq*pq).sum(1)[:,None]+(pm*pm).sum(1)[None]-2*pq@pm.T
                out['pca16'].append(d.clamp_min(0).min(1).values)
                neighbors=(rc[j:j+128]@ctx.T).topk(8,dim=1).indices
                dist=(1-(q[:,None]*mem[neighbors]).sum(-1)).clamp_min(0)
                out['context_top1'].append(dist[:,0]); out['context_top8_min'].append(dist.min(1).values)
            return {m:torch.cat(v).cpu().numpy().reshape(32,32) for m,v in out.items()}
        for split,rows in meta.items():
            subset=[r for r in rows if r['category']==cat]
            for i in range(0,len(subset),4):
                batch=subset[i:i+4]; zz=features([DATA/r['image_path'] for r in batch])
                for row,z in zip(batch,zz):
                    scores=score(z)
                    assert np.allclose(scores['raw'],scores['center_top8_min'],atol=2e-6)
                    for method,a in scores.items():
                        assert np.isfinite(a).all()
                        maps[split][method][int(row['map_index'])]=a
            for m in METHODS: maps[split][m].flush()
        stats.append({'category':cat,'seconds':time.time()-start,'memory_count':len(mem),'old_memory_max_error':err,'peak_gpu_bytes':torch.cuda.max_memory_allocated()})
        (OUT/'progress.json').write_text(json.dumps(stats,indent=2)); print(stats[-1],flush=True)
    (OUT/'COMPLETE.json').write_text(json.dumps({'categories':len(stats),'status':'maps_complete','evaluation_pending':True},indent=2))

if __name__=='__main__': main()
