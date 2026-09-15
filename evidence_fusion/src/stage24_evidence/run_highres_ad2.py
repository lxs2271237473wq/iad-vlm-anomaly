"""Frozen source-validated 512 configuration on all eight AD2 categories."""
import argparse,json
from pathlib import Path
import pandas as pd
import torch
import run_baselines as baseline
from run_highres import CPUStorePatchcore
from common import sha256

ap=argparse.ArgumentParser()
ap.add_argument('--manifest',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
ap.add_argument('--clip-checkpoint',type=Path,required=True)
args=ap.parse_args()
args.seed=42;args.batch_size=4;args.workers=4;args.image_size=512;args.coreset_ratio=.025;args.smoke=False
torch.set_num_threads(4);args.out.mkdir(parents=True,exist_ok=True)
frame=pd.read_csv(args.manifest/'images.csv');assert 'label' not in frame
assert not json.loads((args.manifest/'manifest_report.json').read_text())['cross_split_duplicates']
args.manifest_sha256=sha256(args.manifest/'images.csv');args.clip_sha256=sha256(args.clip_checkpoint)
names=['run_highres_ad2.py','run_highres.py','memory_safe_coreset.py','run_baselines.py','common.py']
args.code_hashes={name:sha256(Path(__file__).parent/name) for name in names}
baseline.PatchcoreModel=CPUStorePatchcore
model,tokenizer,transform=baseline.clip_init(args.clip_checkpoint)
categories=sorted(frame[frame.dataset=='AD2'].category.unique());assert len(categories)==8
for category in categories:
    CPUStorePatchcore.training_images=int(((frame.dataset=='AD2')&(frame.category==category)&(frame.split=='train')).sum())
    baseline.run_category(args,args.manifest,frame,'AD2',category,model,tokenizer,transform)
baseline.json_save(args.out/'queue_complete.json',dict(status='complete',categories=categories,
    image_size=512,coreset_ratio=.025,protocol='Frozen source configuration; all target train normals; normal-only calibration; no conditional residual.'))
