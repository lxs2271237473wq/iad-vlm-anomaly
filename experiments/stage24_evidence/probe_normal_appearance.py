"""Frozen-detector sensitivity on source normal calibration images only."""
import gc,json,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image,ImageEnhance
from anomalib.models.image.patchcore.torch_model import PatchcoreModel
from common import image_transform
from run_baselines import seed_all

root=Path('/root/private_data/iad-vlm-anomaly/results/stage24_evidence')
out=root/'20260911_f/appearance_probe';out.mkdir(parents=True,exist_ok=False)
seed_all(42);torch.set_num_threads(4)
f=pd.read_csv(root/'20260910_a/manifest/images.csv');assert 'label' not in f
f=f[(f.dataset=='VisA')&(f.split=='calibration')]
transform=image_transform(512);rows=[];started=time.perf_counter()
variants=['identity','brightness_08','brightness_12','contrast_08','contrast_12']
for category,g in f.groupby('category',sort=True):
    base=root/'20260911_d/fixed_bank/VisA'/category
    model=PatchcoreModel(layers=['layer2','layer3'],backbone='wide_resnet50_2',pre_trained=True,num_neighbors=9).cuda().eval()
    model.memory_bank=torch.load(base/'memory_bank.pt',map_location='cuda',weights_only=True)
    previous=pd.read_csv(base/'predictions.csv').set_index('image_id')
    cal=json.loads((base/'normal_calibration.json').read_text())['D_raw']
    for r in g.sort_values('image_id').head(4).itertuples():
        original=Image.open(r.path).convert('RGB')
        views=[original,ImageEnhance.Brightness(original).enhance(.8),ImageEnhance.Brightness(original).enhance(1.2),ImageEnhance.Contrast(original).enhance(.8),ImageEnhance.Contrast(original).enhance(1.2)]
        values=[];locations=[]
        with torch.inference_mode():
            for view in views:
                output=model(transform(view)[None].cuda())
                score=float(output.pred_score.item());assert np.isfinite(score)
                values.append(score)
                a=output.anomaly_map[0,0].cpu().numpy();locations.append(np.unravel_index(a.argmax(),a.shape))
        expected=float(previous.loc[r.image_id,'D_raw'])
        assert np.isclose(values[0],expected,rtol=1e-4,atol=1e-4),(category,r.image_id,values[0],expected)
        for variant,value,location in zip(variants,values,locations):
            rows.append(dict(category=category,image_id=r.image_id,variant=variant,score=value,
                normal_z=(value-cal['center'])/cal['scale'],delta_normal_units=(value-values[0])/cal['scale'],
                peak_displacement_pixels=float(np.linalg.norm(np.array(location)-locations[0]))))
    pd.DataFrame(rows).to_csv(out/'scores.partial.csv',index=False)
    print('COMPLETE',category,flush=True)
    del model;gc.collect();torch.cuda.empty_cache()
table=pd.DataFrame(rows);table.to_csv(out/'scores.csv',index=False)
table['abs_delta_normal_units']=table.delta_normal_units.abs()
summary=table.groupby(['category','variant']).agg(mean_absolute_shift=('abs_delta_normal_units','mean'),median_shift=('delta_normal_units','median'),mean_peak_displacement=('peak_displacement_pixels','mean'))
summary.to_csv(out/'summary.csv')
(out/'complete.json').write_text(json.dumps(dict(status='complete',normal_images=48,views=240,seconds=time.perf_counter()-started,
    labels_used=False,identity_parity=True,selection='first 4 source normal calibration images by image_id per category',
    limitation='Controlled photometric sensitivity only; no anomaly performance, real illumination equivalence or novelty claim.'),indent=2))
print(summary.to_string(),flush=True)
