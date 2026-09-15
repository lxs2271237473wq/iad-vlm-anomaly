"""Official six-category INP branch only. CPR branch tracked separately."""
import os,sys,runpy,subprocess,json,time,hashlib
from pathlib import Path
Z=Path('/root/private_data/iad-vlm-anomaly/ad2_model_zoo'); R=Z/'repos/ISVL'; D=Z.parent/'datasets/MVTec_AD_2'
def main():
    os.chdir(R); sys.path.insert(0,str(R))
    splitter=runpy.run_path(str(R/'1_image_splitter.py'))
    specs={'can':[('long_edge',(2,2),2),('grid',(4,2),2)],'fabric':[('grid',(2,2),2)],'rice':[('grid',(2,2),2),('grid',(4,4),2)],'sheet_metal':[('long_edge',(2,2),4),('grid',(8,2),2)],'wallplugs':[],'walnuts':[('grid',(2,2),2)]}
    dest=Z/'data/isvl_inp_train'
    for cat,ops in specs.items():
        marker=dest/cat/'PREPARED.json'
        if not marker.exists():
            for mode,grid,n in ops:
                splitter['process_dataset'](input_dir=str(D/cat/'train'),output_dir=str(dest/cat/'train'),split_mode=mode,grid_size=grid,num_splits=n,save_format='png')
            if cat!='can': splitter['copy_images_with_structure'](str(D/cat/'train'),str(dest/cat/'train'))
            marker.parent.mkdir(parents=True,exist_ok=True); marker.write_text(json.dumps({'normal_train_only':True,'ops':ops}))
    # CPU data prep can finish while RoBiS owns the GPU. No simultaneous trainers.
    while not (Z/'results/ROBIS_COMPLETE.json').exists():
        try: os.kill(70323,0)
        except ProcessLookupError: raise RuntimeError('RoBiS stopped before completion; GPU queue paused')
        time.sleep(30)
    pretrained=Z/'pretrained/dinov2_vitl14_reg4_pretrain.pth'
    ready=pretrained.with_suffix('.ready.json')
    if not ready.exists(): raise RuntimeError('Verified ISVL backbone is not ready')
    target=R/'backbones/weights'/pretrained.name; target.parent.mkdir(parents=True,exist_ok=True)
    if not target.exists(): target.symlink_to(pretrained)
    for cat in specs:
        command=[sys.executable,'-u',str(Z/'runners/isvl_train_only.py'),'--data_path',str(dest),'--save_dir',str(Z/'weights/ISVL_INP'),'--item_list',cat,'--total_epochs','10','--phase','train']
        print('START',command,flush=True); subprocess.run(command,cwd=R,check=True,env={**os.environ,'OMP_NUM_THREADS':'8','MKL_NUM_THREADS':'8','PYTHONPATH':str(R)})
    (Z/'results/ISVL_INP_COMPLETE.json').write_text(json.dumps({'status':'six_INP_categories_trained','CPR_two_categories':'not_yet_run'}))
if __name__=='__main__': main()
