"""Memory feasibility on eight source NORMAL TRAIN images; not a trained baseline."""
import argparse,json
from pathlib import Path
import pandas as pd
import torch
from PIL import Image
from anomalib.models.image.patchcore.torch_model import PatchcoreModel
from common import image_transform
from run_baselines import seed_all

ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
args=ap.parse_args();args.out.parent.mkdir(parents=True,exist_ok=True)
seed_all(42);torch.set_num_threads(4)
frame=pd.read_csv(args.run/'manifest/images.csv')
train=frame[(frame.dataset=='VisA')&(frame.split=='train')]
category=train.groupby('category').size().idxmax();subset=train[train.category==category].sort_values('image_id')
model=PatchcoreModel(layers=['layer2','layer3'],backbone='wide_resnet50_2',pre_trained=True).cuda()
model.train();model.feature_extractor.eval();transform=image_transform(512)
torch.cuda.reset_peak_memory_stats();shapes=[]
with torch.inference_mode():
    for start in [0,4]:
        tensors=[]
        for row in subset.iloc[start:start+4].itertuples():
            with Image.open(row.path) as handle:tensors.append(transform(handle.convert('RGB')))
        embedding=model(torch.stack(tensors).cuda());shapes.append(list(embedding.shape))
        model.embedding_store.clear();del embedding
torch.cuda.synchronize()
bytes_per_image=shapes[0][0]//4*shapes[0][1]*4
raw=len(subset)*bytes_per_image
report=dict(status='512 feature extraction preflight passed',category=category,normal_training_images=len(subset),
    sampled_normal_images=8,input_size=512,batch_size=4,embedding_shapes=shapes,
    peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,
    peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3,
    device_total_gib=torch.cuda.get_device_properties(0).total_memory/1024**3,
    estimated_full_training_embeddings_gib=raw/1024**3,
    estimated_embeddings_plus_vstack_gib=2*raw/1024**3,
    coreset_selection_run=False,anomaly_evaluation_run=False,
    note='Capacity estimate excludes further projection and inference allocations. Existing GPU embedding_store + vstack needs redesign/offload or more memory before full 512 training.')
args.out.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
