import importlib.metadata as md
import json,subprocess,platform
from pathlib import Path
import torch
from anomalib.models.image.patchcore.torch_model import PatchcoreModel
from common import sha256

root=Path('/root/private_data/iad-vlm-anomaly')
model=PatchcoreModel(layers=['layer2','layer3'],backbone='wide_resnet50_2',pre_trained=True)
cache=Path('/root/.cache/huggingface/hub')
weights={str(p):sha256(p) for directory in ['models--timm--wide_resnet50_2.racm_in1k','models--timm--vit_base_patch32_clip_224.openai']
         for p in (cache/directory/'snapshots').glob('*/*') if p.suffix in ['.safetensors','.bin','.pth']}
report=dict(python=platform.python_version(),gpu=torch.cuda.get_device_name(0),cuda=torch.version.cuda,
    base_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
    git_status=subprocess.check_output(['git','status','--short'],cwd=root,text=True),
    packages={p:md.version(p) for p in ['torch','torchvision','anomalib','open_clip_torch','timm','numpy','pandas','scikit-learn']},
    cached_pretrained_weight_hashes=weights,
    dropout_modules=[n for n,m in model.named_modules() if isinstance(m,torch.nn.modules.dropout._DropoutNd)],
    protocol='Source anomaly supervision / target normal-only adaptation; first run seed 42; full-field 256 detector and 224 CLIP; default 10% coreset; 9 neighbors',
    new_exchange_method_status='not executed; baseline prerequisites first')
(root/'results/stage24_evidence/20260910_a/environment.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
