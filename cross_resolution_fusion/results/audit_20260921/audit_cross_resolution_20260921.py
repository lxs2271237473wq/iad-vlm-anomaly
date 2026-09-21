"""Read-only checks of the published cross-resolution experiment."""
from pathlib import Path
from collections import Counter, defaultdict
import csv, json, subprocess, hashlib, importlib.util
import numpy as np
from PIL import Image

ROOT = Path('/root/private_data/iad-vlm-anomaly')
P = ROOT/'cross_resolution_fusion'
out = {'commit': subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()}
meta = list(csv.DictReader((ROOT/'orbitad/results/a10_multilayer_v1/public_meta.csv').open()))
out['metadata'] = {'rows':len(meta), 'unique_image_paths':len({r['image_path'] for r in meta}),
 'category_condition_label':dict(Counter('|'.join((r['category'],r['condition'],r['label'])) for r in meta)),
 'unique_category_label_instance':len({(r['category'],r['label'],r['instance_id']) for r in meta})}
data = ROOT/'datasets/MVTec_AD_2'
actual = {str(p.relative_to(data)) for cat in {r['category'] for r in meta} for label_dir in ('good','bad') for p in (data/cat/'test_public'/label_dir).glob('*.png')}
out['metadata']['dataset_vs_meta_missing'] = sorted(actual-{r['image_path'] for r in meta})
out['metadata']['meta_vs_dataset_extra'] = sorted({r['image_path'] for r in meta}-actual)
out['metadata']['missing_masks'] = [r['mask_path'] for r in meta if int(r['label']) and not (data/r['mask_path']).is_file()]
out['maps']={}
for tag,base in [('normal',ROOT/'ad2_model_zoo/results/a67_normal_validation_dual_resolution_v1'),('test',ROOT/'ad2_model_zoo/results/a74_ad2_matched_dual_resolution_v1')]:
 out['maps'][tag]={c:{k:len(list((base/c/'component_maps/seed=0'/c).rglob('*_'+k+'.tiff'))) for k in ['global','tiled']} for c in sorted({r['category'] for r in meta})}
details=list(csv.DictReader((P/'results/ad2_matched/method_category_condition_size.csv').open()))
summary=json.loads((P/'results/ad2_matched/summary.json').read_text())
recalc={}
for method in summary['summary']:
 cats=defaultdict(list)
 for r in details:
  if r['method']==method and r['size_band']=='all' and np.isfinite(float(r['aupro_0_05'])): cats[r['category']].append(float(r['aupro_0_05']))
 recalc[method]={c:float(np.mean(v)) for c,v in cats.items()}
out['a74_recomputed_macro']={m:float(np.mean(list(v.values()))) for m,v in recalc.items()}
out['a74_max_macro_abs_error']=max(abs(out['a74_recomputed_macro'][m]-summary['summary'][m]['all']) for m in recalc)
out['a74_category_scores']=recalc
out['a74_size_band_categories']={band:sorted({r['category'] for r in details if r['method']=='raw_mean' and r['size_band']==band and np.isfinite(float(r['aupro_0_05']))}) for band in ['all','tiny_le_0.1pct','small_0.1_to_1pct','large_gt_1pct']}
a73=list(csv.DictReader((P/'results/mvtec14_external/category_metrics.csv').open()))
js=json.loads((P/'results/mvtec14_external/summary.json').read_text())
out['a73_max_macro_abs_error']=max(abs(float(np.mean([float(r[metric]) for r in a73 if r['method']==m]))-js['macro'][m][metric]['mean']) for m in js['macro'] for metric in ['aupro_0_05','pixel_auroc','image_auroc'])
out['a73_missing_categories']=sorted({p.name for p in (ROOT/'datasets/MVTecAD').iterdir() if p.is_dir()}-{r['category'] for r in a73})
out['geometry']={}
for c in sorted({r['category'] for r in meta}):
 row=next(r for r in meta if r['category']==c)
 with Image.open(data/row['image_path']) as im: w,h=im.size
 scale=448/min(w,h)
 rw,rh=(448,int(448*h/w)) if w<=h else (int(448*w/h),448)
 out['geometry'][c]={'original_wh':[w,h], 'resize_wh':[rw,rh], 'cropped_wh':[rw-rw%14,rh-rh%14], 'lost_original_pixels_wh':[rw%14/scale,rh%14/scale], 'tiled_crops':int(np.ceil(max(w,h)/min(w,h)))}
tracked=subprocess.check_output(['git','ls-files','cross_resolution_fusion'],cwd=ROOT,text=True).splitlines()
out['publication']={'tracked_files':len(tracked),'bytes':sum((ROOT/p).stat().st_size for p in tracked),
 'forbidden_files':[p for p in tracked if Path(p).suffix.lower() in {'.pth','.pt','.tif','.tiff','.npy','.npz','.safetensors','.ckpt'}],
 'legacy_meta_tracked':bool(subprocess.check_output(['git','ls-files','orbitad/results/a10_multilayer_v1/public_meta.csv'],cwd=ROOT,text=True).strip()),
 'faiss_fallback_tracked':any('faiss' in p for p in tracked)}
manifest=json.loads((P/'results/provenance/environment_manifest.json').read_text())
out['manifest_count_check']={c:{'reported_evaluation':v['evaluation'],'actual_test_images':sum(r['category']==c for r in meta)} for c,v in manifest['dataset_counts']['MVTec_AD_2'].items()}
out['publication']['patch_hash_matches']=hashlib.sha256((P/'patches/superad_cross_resolution.patch').read_bytes()).hexdigest()==manifest['superad']['integration_patch_sha256']
# Validate the A73 histogram integration against an independent sorted-score ROC
# calculation using region-normalized positive weights, including many score ties.
spec=importlib.util.spec_from_file_location('audit_a73',P/'scripts/a73_evaluate_mvtec14_unified.py')
ev=importlib.util.module_from_spec(spec);spec.loader.exec_module(ev)
def exact(scores,negative,weights,nregions):
 order=np.argsort(-scores,kind='stable'); s=scores[order]
 cuts=np.r_[np.flatnonzero(s[1:]!=s[:-1]),len(s)-1]
 x=np.r_[0,np.cumsum(negative[order])[cuts]/negative.sum()]
 y=np.r_[0,np.cumsum(weights[order])[cuts]/nregions]
 j=np.searchsorted(x,.05,side='right'); xx=x[:j]; yy=y[:j]
 if xx[-1]<.05 and j<len(x):
  yy=np.r_[yy,y[j-1]+(.05-x[j-1])/(x[j]-x[j-1])*(y[j]-y[j-1])];xx=np.r_[xx,.05]
 return float(np.trapezoid(yy,xx)/.05)
rng=np.random.default_rng(42);errors=[]
for rep in range(100):
 scores=rng.integers(0,64,400); neg=np.ones(400,bool);neg[:40]=False
 weights=np.zeros(400);weights[:10]=.1;weights[10:40]=1/30
 nh=np.bincount(scores[neg],minlength=64);ph=np.bincount(scores,weights=weights,minlength=64)
 errors.append(abs(ev.aupro(nh,ph,2)-exact(scores,neg,weights,2)))
out['histogram_integration_test']={'cases':100,'maximum_absolute_error':max(errors),'scope':'synthetic integration only; does not establish parity with official evaluation or real-map binning accuracy'}
out['condition_aggregation_counterexample']={'per_condition_mean':1.0,'pooled_score_aupro':exact(np.array([0.,.1,.2,.8,.9,1.]),np.array([1,1,0,1,1,0],bool),np.array([0,0,1,0,0,1.]),2)}
print(json.dumps(out,indent=2))
