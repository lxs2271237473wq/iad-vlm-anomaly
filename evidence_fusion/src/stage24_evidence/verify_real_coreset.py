"""Real normal-image parity check before high-resolution training."""
import argparse, json, time
from pathlib import Path
import pandas as pd
import torch
from anomalib.models.image.patchcore.torch_model import PatchcoreModel
from anomalib.models.components.sampling.k_center_greedy import KCenterGreedy
from torch.utils.data import DataLoader
from run_baselines import Images, seed_all
from memory_safe_coreset import select_cpu_backed

ap=argparse.ArgumentParser(); ap.add_argument('--manifest',type=Path,required=True); ap.add_argument('--out',type=Path,required=True)
args=ap.parse_args(); args.out.parent.mkdir(parents=True,exist_ok=True)
seed_all(42); torch.set_num_threads(4)
frame=pd.read_csv(args.manifest/'images.csv')
rows=frame[(frame.dataset=='VisA')&(frame.category=='pcb3')&(frame.split=='train')].sort_values('image_id').head(8).to_dict('records')
model=PatchcoreModel(layers=['layer2','layer3'],backbone='wide_resnet50_2',pre_trained=True,num_neighbors=9).cuda()
model.train(); model.feature_extractor.eval(); store=None; offset=0
torch.cuda.reset_peak_memory_stats()
with torch.inference_mode():
    for batch,_ in DataLoader(Images(rows,512),batch_size=4,shuffle=False):
        last=batch.cuda(); emb=model(last)
        if store is None: store=torch.empty((len(rows)*(len(emb)//len(batch)),emb.shape[1]),dtype=emb.dtype)
        store[offset:offset+len(emb)].copy_(emb.cpu()); offset+=len(emb)
        model.embedding_store.clear(); del emb
    seed_all(42); start=time.perf_counter()
    expected=KCenterGreedy(store.cuda(),.025).select_coreset_idxs()
    torch.cuda.synchronize(); original_seconds=time.perf_counter()-start
    seed_all(42); start=time.perf_counter()
    actual=select_cpu_backed(store,.025)
    torch.cuda.synchronize(); safe_seconds=time.perf_counter()-start
    assert actual==expected, 'Real-feature coreset index mismatch'
    model.memory_bank=store[expected].cuda(); model.eval(); a=model(last)
    model.memory_bank=store[actual].cuda(); b=model(last)
    torch.testing.assert_close(a.pred_score,b.pred_score,rtol=0,atol=0)
    torch.testing.assert_close(a.anomaly_map,b.anomaly_map,rtol=0,atol=0)
result=dict(status='passed',normal_images=len(rows),embedding_shape=list(store.shape),selected_centers=len(actual),
    indices_identical=True,scores_and_maps_identical=True,original_seconds=original_seconds,safe_seconds=safe_seconds,
    peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,image_ids=[r['image_id'] for r in rows],
    note='Normal-only implementation parity test. No anomaly performance claim.')
args.out.write_text(json.dumps(result,indent=2)); print(json.dumps(result),flush=True)
