import json
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from local_learning import RUN,ROOT

p=RUN/'20260912_z';records=json.loads((p/'provenance.json').read_text())
official=pd.read_csv(ROOT/'datasets/VisA/split_csv/1cls.csv')
masks={str(ROOT/'datasets/VisA'/r.image):str(ROOT/'datasets/VisA'/r.mask) for r in official.itertuples() if isinstance(r.mask,str) and r.mask}
fig,axes=plt.subplots(4,4,figsize=(16,16));metrics=[]
for i,r in enumerate(records):
    query=np.asarray(Image.open(r['query_path']).convert('RGB').resize((512,512)))
    reference=np.asarray(Image.open(r['reference_path']).convert('RGB').resize((512,512)))
    qx=(r['raw_patch_x']+.5)*512/r['patch_grid'][1];qy=(r['raw_patch_y']+.5)*512/r['patch_grid'][0]
    rx=(r['reference_patch_x']+.5)*512/r['patch_grid'][1];ry=(r['reference_patch_y']+.5)*512/r['patch_grid'][0]
    axes[i,0].imshow(query);axes[i,0].plot(qx,qy,'r+',ms=12);axes[i,0].add_patch(Rectangle((qx-40,qy-40),80,80,fill=False,edgecolor='yellow'))
    axes[i,0].set_title(f"{r['category']} label={r['label']} D_z={r['D_z']:.3f}")
    if r['label']:
        mask=np.asarray(Image.open(masks[r['query_path']]).convert('L').resize((512,512),Image.Resampling.NEAREST))>0
        axes[i,0].contour(mask,levels=[.5],colors=['lime'],linewidths=1)
        metrics.append(dict(image_id=r['image_id'],raw_center_in_mask=bool(mask[int(qy),int(qx)]),smoothed_peak_in_mask=bool(mask[r['smoothed_peak_y'],r['smoothed_peak_x']]),mask_fraction=float(mask.mean())))
    axes[i,1].imshow(query);axes[i,1].set_xlim(max(0,qx-40),min(512,qx+40));axes[i,1].set_ylim(min(512,qy+40),max(0,qy-40));axes[i,1].plot(qx,qy,'r+');axes[i,1].set_title('Query 80px context (not receptive field)')
    axes[i,2].imshow(reference);axes[i,2].plot(rx,ry,'r+',ms=12);axes[i,2].add_patch(Rectangle((rx-40,ry-40),80,80,fill=False,edgecolor='yellow'));axes[i,2].set_title('Recovered normal reference')
    axes[i,3].imshow(reference);axes[i,3].set_xlim(max(0,rx-40),min(512,rx+40));axes[i,3].set_ylim(min(512,ry+40),max(0,ry-40));axes[i,3].plot(rx,ry,'r+');axes[i,3].set_title('Reference 80px context')
    for ax in axes[i]:ax.axis('off')
fig.tight_layout();fig.savefig(p/'neighbor_audit.png',dpi=110);pd.DataFrame(metrics).to_csv(p/'mask_checks.csv',index=False)
print(pd.DataFrame(metrics).to_string(index=False))
