import os,sys,json,time,hashlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import torch

ZOO=Path('/root/private_data/iad-vlm-anomaly/ad2_model_zoo')
DATA=ZOO.parent/'datasets/MVTec_AD_2'
REPO=ZOO/'repos/RoBiS'
os.chdir(REPO); sys.path.insert(0,str(REPO)); torch.set_num_threads(8)
from models.modules._swin_cropping import generate_sliding_window_images
from models.robis import RoBiS

def main():
    cats=['sheet_metal','vial','wallplugs','walnuts','can','fabric','fruit_jelly','rice']
    cfg={'device':0,'dataset':{'dataset_name':'MVTec-AD-2','original_data_path':str(DATA),'data_path':str(ZOO/'data/robis_train_crops'),'mvtecad2_class_list':cats,'test_type':'validation'},'save':{'save_dir':str(ZOO/'weights/RoBiS'),'amap_savedir':str(ZOO/'results/RoBiS')},'model':{'encoder':'dinov2reg_vit_base_14','input_size':518,'crop_size':518,'INP_num':6},'training':{'total_epochs':200,'batch_size':16,'train_state':True},'testing':{'test_state':False}}
    (ZOO/'configs/robis_train.json').write_text(json.dumps(cfg,indent=2))
    for cat in cats:
        weight=ZOO/'weights/RoBiS'/cat/'model.pth'
        if weight.exists(): continue
        dest=ZOO/'data/robis_train_crops'/cat/'train/good'; dest.mkdir(parents=True,exist_ok=True)
        marker=dest.parent/'CROPS_COMPLETE.json'
        if not marker.exists():
            paths=sorted((DATA/cat/'train/good').glob('*.png'))
            assert paths
            def crop(p): generate_sliding_window_images(str(p),str(dest),1024,.1)
            with ThreadPoolExecutor(max_workers=4) as pool: list(pool.map(crop,paths))
            marker.write_text(json.dumps({'source_images':len(paths),'crops':len(list(dest.glob('*.png'))),'normal_only':True}))
        cfg['dataset']['mvtecad2_class_list']=[cat]
        print('TRAIN_START',cat,flush=True)
        RoBiS(cfg).train()
        assert weight.exists()
        manifest={'category':cat,'epochs':200,'bytes':weight.stat().st_size,'sha256':hashlib.sha256(weight.read_bytes()).hexdigest(),'status':'trained'}
        (weight.parent/'manifest.json').write_text(json.dumps(manifest,indent=2))
        print('TRAIN_COMPLETE',manifest,flush=True)
    (ZOO/'results/ROBIS_COMPLETE.json').write_text(json.dumps({'status':'complete','categories':cats}))

if __name__=='__main__': main()
