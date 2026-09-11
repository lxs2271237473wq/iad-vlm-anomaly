"""Source-only 512 PatchCore, CPU-preallocated embeddings, fixed bank budget.

At 512, ratio .025 matches the number of centers at 256 ratio .1.
All training images are retained; no pre-sampling of training patches.
"""
import argparse, json
from pathlib import Path
import pandas as pd
import torch
import run_baselines as baseline
from memory_safe_coreset import select_cpu_backed
from common import sha256

OriginalPatchcore=baseline.PatchcoreModel

class CPUStorePatchcore(OriginalPatchcore):
    training_images=0
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.cpu_store=None
        self.cpu_offset=0

    def forward(self,input_tensor):
        output=super().forward(input_tensor)
        if self.training:
            if self.cpu_store is None:
                patches=len(output)//len(input_tensor)
                self.cpu_store=torch.empty((self.training_images*patches,output.shape[1]),dtype=output.dtype,device='cpu')
            end=self.cpu_offset+len(output)
            assert end<=len(self.cpu_store)
            self.cpu_store[self.cpu_offset:end].copy_(output.detach().cpu())
            self.cpu_offset=end
            self.embedding_store.clear()
        return output

    def subsample_embedding(self,sampling_ratio,embeddings=None):
        assert self.cpu_store is not None and self.cpu_offset==len(self.cpu_store)
        print('CPU_CORESET_START',list(self.cpu_store.shape),sampling_ratio,flush=True)
        indices=select_cpu_backed(self.cpu_store,sampling_ratio)
        self.memory_bank=self.cpu_store[indices].to('cuda')
        self.cpu_store=None

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--manifest',type=Path,required=True); ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--clip-checkpoint',type=Path,required=True)
    ap.add_argument('--categories',nargs='+'); ap.add_argument('--coreset-ratio',type=float,default=.025)
    ap.add_argument('--image-size',type=int,default=512); ap.add_argument('--smoke',action='store_true')
    args=ap.parse_args(); args.seed=42; args.batch_size=4; args.workers=4
    torch.set_num_threads(4); args.out.mkdir(parents=True,exist_ok=True)
    frame=pd.read_csv(args.manifest/'images.csv'); assert 'label' not in frame
    assert not json.loads((args.manifest/'manifest_report.json').read_text())['cross_split_duplicates']
    args.manifest_sha256=sha256(args.manifest/'images.csv'); args.clip_sha256=sha256(args.clip_checkpoint)
    names=['run_highres.py','memory_safe_coreset.py','run_baselines.py','common.py']
    args.code_hashes={name:sha256(Path(__file__).parent/name) for name in names}
    baseline.PatchcoreModel=CPUStorePatchcore
    model,tokenizer,transform=baseline.clip_init(args.clip_checkpoint)
    categories=args.categories or sorted(frame[frame.dataset=='VisA'].category.unique())
    for category in categories:
        count=int(((frame.dataset=='VisA')&(frame.category==category)&(frame.split=='train')).sum())
        CPUStorePatchcore.training_images=min(count,8) if args.smoke else count
        baseline.run_category(args,args.manifest,frame,'VisA',category,model,tokenizer,transform)
    baseline.json_save(args.out/'queue_complete.json',dict(status='complete',categories=categories,smoke=args.smoke,
        protocol='source only; unchanged training images; fixed seed; normal-only calibration; CPU feature store',
        image_size=args.image_size,coreset_ratio=args.coreset_ratio))

if __name__=='__main__': main()
