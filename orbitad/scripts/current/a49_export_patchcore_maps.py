"""Export raw anomaly maps from the already trained Stage-11 PatchCore checkpoints."""
import gc
import inspect
import json
from pathlib import Path

import numpy as np
import tifffile
import torch


ROOT=Path('/root/private_data/iad-vlm-anomaly')
SRB=ROOT/'srb_qcr'
ADAPTER=ROOT/'datasets/MVTec_AD_2_anomalib_all'
CKPT=SRB/'runs/stage11_mvtecad2_multicategory/patchcore_baseline'
OUT=ROOT/'ad2_model_zoo/results/a49_patchcore256_public_maps_v1'
CATS=['can','fabric','fruit_jelly','rice','sheet_metal','vial','wallplugs','walnuts']


def filter_kwargs(fn,kwargs):
    sig=inspect.signature(fn); return {k:v for k,v in kwargs.items() if k in sig.parameters}


def field(batch,name):
    return batch.get(name) if isinstance(batch,dict) else getattr(batch,name,None)


def main():
    from anomalib.data import Folder
    from anomalib.data.utils import ValSplitMode
    from anomalib.engine import Engine
    from anomalib.models import Patchcore
    OUT.mkdir(parents=True,exist_ok=True)
    progress=[]
    for cat in CATS:
        marker=OUT/cat/'COMPLETE.json'
        data_root=ADAPTER/f'{cat}_folder'
        expected=sum(1 for p in (data_root/'test').glob('*/*') if p.is_file())
        existing=sum(1 for p in (OUT/cat).glob('*/*.tiff'))
        if marker.exists() and existing == expected:
            print('SKIP_COMPLETE',cat,flush=True); progress.append(json.loads(marker.read_text())); continue
        dm=Folder(**filter_kwargs(Folder,{'name':f'mvtecad2_{cat}_folder','root':str(data_root),'normal_dir':'train/good','abnormal_dir':'test/bad','normal_test_dir':'test/good','mask_dir':'ground_truth/bad','train_batch_size':8,'eval_batch_size':4,'num_workers':0,'val_split_mode':ValSplitMode.NONE}))
        # The checkpoint contains the complete backbone and memory bank.  Avoid a
        # network-dependent ImageNet download before Lightning restores it.
        model=Patchcore(backbone='wide_resnet50_2',layers=['layer2','layer3'],pre_trained=False,num_neighbors=9,visualizer=False)
        engine=Engine(default_root_dir=str(OUT/'engine'/cat),accelerator='gpu',devices=1,logger=False)
        ckpts=list((CKPT/cat).rglob('model.ckpt')); assert len(ckpts)==1,ckpts
        print('EXPORT_START',cat,ckpts[0],flush=True)
        outputs=engine.predict(model=model,datamodule=dm,return_predictions=True,ckpt_path=str(ckpts[0]))
        count=0
        for batch in outputs:
            paths=field(batch,'image_path'); maps=field(batch,'anomaly_map')
            for i,path in enumerate(paths):
                p=Path(path); label=p.parent.name
                stem=p.stem.split('_',1)[1]
                array=maps[i].detach().float().cpu().numpy().squeeze().astype(np.float32)
                dest=OUT/cat/label/f'{stem}.tiff'; dest.parent.mkdir(parents=True,exist_ok=True)
                tifffile.imwrite(dest,array); count+=1
        total=sum(1 for p in (OUT/cat).glob('*/*.tiff'))
        assert total == expected, (cat,total,expected)
        item={'category':cat,'maps':total,'checkpoint':str(ckpts[0])}
        marker.parent.mkdir(parents=True,exist_ok=True); marker.write_text(json.dumps(item,indent=2)); progress.append(item)
        del outputs,engine,model,dm; torch.cuda.empty_cache(); gc.collect()
        print('EXPORT_COMPLETE',item,flush=True)
    (OUT/'MAPS_COMPLETE.json').write_text(json.dumps({'status':'complete','categories':progress},indent=2))


if __name__=='__main__': main()
