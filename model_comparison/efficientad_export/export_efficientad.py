from pathlib import Path
import os,json,time,hashlib
import numpy as np,pandas as pd,torch
from PIL import Image
from torchvision.transforms.functional import to_tensor
from anomalib.models import EfficientAd
R=Path('/root/private_data/iad-vlm-anomaly');O=R/'model_comparison/efficientad_export';O.mkdir(parents=True,exist_ok=True)
torch.set_num_threads(8)
manifest=pd.read_csv(R/'model_comparison/stage24_ad2_highres/evaluation/predictions_with_evaluation_labels.csv')
# Original Stage24 identity manifest provides source paths and content digests.
paths=list((R/'evidence_fusion/results/stage24_evidence/20260910_a').rglob('*manifest*.csv'))
print('manifests',paths,flush=True)
rows=[];meta=[]
for cat in ['fruit_jelly','sheet_metal','vial','walnuts']:
 root=R/'srb_qcr/runs'/('stage17_defensive_sensitivity' if cat=='fruit_jelly' else 'stage19_efficientad100_sensitivity')
 checkpoints=[p for p in root.rglob('model.ckpt') if cat in str(p)]
 assert len(checkpoints)==1,checkpoints
 ck=checkpoints[0]; c=torch.load(ck,map_location='cpu',weights_only=False)
 model=EfficientAd(**c['hyper_parameters']);model.load_state_dict(c['state_dict'],strict=True);model.eval().cuda();model.requires_grad_(False)
 original=R/'datasets/MVTec_AD_2'/cat/'test_public'
 originals={}
 for label in ['good','bad']:
  for p in sorted((original/label).glob('*.png')):originals[hashlib.sha256(p.read_bytes()).hexdigest()]=(p,int(label=='bad'))
 legacy=[];unknown=[]
 for p in (R/'datasets/MVTec_AD_2_anomalib_all'/f'{cat}_folder'/'test').rglob('*.png'):
  sha=hashlib.sha256(p.read_bytes()).hexdigest()
  if sha in originals:legacy.append(sha)
  else:unknown.append(str(p))
 assert not unknown,unknown[:3]
 start=time.time()
 for i,(sha,(p,label)) in enumerate(originals.items()):
  with torch.inference_mode():
   x=to_tensor(Image.open(p).convert('RGB')).unsqueeze(0)
   x=model.pre_processor(x).cuda();raw=model.model(x)
   score=float(raw.pred_score.item());processed=model.post_processor(raw);post=float(processed.pred_score.item())
  assert np.isfinite(score) and np.isfinite(post)
  rows.append(dict(category=cat,label=label,path=str(p),sha256=sha,efficientad=score,efficientad_post=post,legacy_test=sha in legacy))
  if i%40==0:print(cat,i,len(originals),flush=True)
 pd.DataFrame(rows).to_csv(O/'predictions.partial.csv',index=False)
 meta.append(dict(category=cat,checkpoint=str(ck),sha256=hashlib.sha256(ck.read_bytes()).hexdigest(),epoch=c['epoch'],images=len(originals),legacy_test=len(legacy),seconds=time.time()-start))
 (O/'checkpoints.json').write_text(json.dumps(meta,indent=2));del model;torch.cuda.empty_cache()
pd.DataFrame(rows).to_csv(O/'predictions.csv',index=False)
(O/'complete.json').write_text(json.dumps(dict(training=False,inference='fp32, original checkpoint preprocessor + model + postprocessor; raw score retained',categories=meta),indent=2))
print('EXPORT_COMPLETE',len(rows),flush=True)
