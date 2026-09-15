from pathlib import Path
import csv
import random

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import timm
from timm.data import resolve_model_data_config, create_transform


DATA_ROOT = Path(
    "/root/private_data/iad-vlm-anomaly/datasets/MVTec_AD_2"
)

GOOD_MANIFEST = Path(
    "orbitad/data/manifests/test_public_good.csv"
)

BAD_MANIFEST = Path(
    "orbitad/data/manifests/test_public_bad.csv"
)

OUT_ROOT = Path(
    "orbitad/results/a1_feature_sensitivity"
)

OUT_ROOT.mkdir(parents=True, exist_ok=True)

FEATURES_NPY = OUT_ROOT / "dinov3b_public_features.npy"
META_CSV = OUT_ROOT / "dinov3b_public_features.csv"

MODEL_NAME = "vit_base_patch16_dinov3.lvd1689m"

BATCH_SIZE = 32
NUM_WORKERS = 8
SEED = 20260913


random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def read_manifest(path, set_type):
    rows = []

    with path.open() as f:
        reader = csv.DictReader(f)

        for row in reader:
            rows.append({
                "set_type": set_type,
                "category": row["category"],
                "instance_id": row["instance_id"],
                "condition": row["condition"],
                "image_path": row["image_path"],
            })

    return rows


rows = (
    read_manifest(GOOD_MANIFEST, "normal")
    +
    read_manifest(BAD_MANIFEST, "defect")
)

print("=" * 100)
print("OrbitAD A1.1 — FROZEN DINOv3 PUBLIC FEATURE EXTRACTION")
print("=" * 100)
print("normal images :", sum(r["set_type"] == "normal" for r in rows))
print("defect images :", sum(r["set_type"] == "defect" for r in rows))
print("total images  :", len(rows))


if len(rows) != 1084:
    raise RuntimeError(
        f"Expected 1084 public images, found {len(rows)}"
    )


device = torch.device("cuda")

print("\nLoading model:", MODEL_NAME)

model = timm.create_model(
    MODEL_NAME,
    pretrained=True,
    num_classes=0,
)

model = model.eval().to(device)

data_config = resolve_model_data_config(model)
transform = create_transform(
    **data_config,
    is_training=False,
)

print("input size :", data_config.get("input_size"))
print("mean       :", data_config.get("mean"))
print("std        :", data_config.get("std"))


class PublicDataset(Dataset):

    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        row = self.records[idx]

        path = DATA_ROOT / row["image_path"]

        with Image.open(path) as im:
            image = im.convert("RGB")

        image = transform(image)

        return image, idx


dataset = PublicDataset(rows)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    persistent_workers=(NUM_WORKERS > 0),
)


all_features = np.zeros(
    (len(rows), 768),
    dtype=np.float32
)

seen = 0

with torch.inference_mode():

    for batch_idx, (images, indices) in enumerate(loader):

        images = images.to(
            device,
            non_blocking=True
        )

        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
        ):
            features = model(images)

        features = F.normalize(
            features.float(),
            dim=-1
        )

        features = features.cpu().numpy()

        idx_np = indices.numpy()

        all_features[idx_np] = features

        seen += len(indices)

        if (
            batch_idx % 10 == 0
            or seen == len(rows)
        ):
            print(
                f"[{seen:4d}/{len(rows)}] "
                f"GPU allocated="
                f"{torch.cuda.memory_allocated()/1024**3:.2f} GB"
            )


if not np.isfinite(all_features).all():
    raise RuntimeError(
        "Non-finite feature found."
    )


norms = np.linalg.norm(
    all_features,
    axis=1
)

print()
print("feature norm min :", norms.min())
print("feature norm max :", norms.max())
print("feature shape    :", all_features.shape)


np.save(
    FEATURES_NPY,
    all_features
)


with META_CSV.open("w", newline="") as f:

    fields = [
        "feature_index",
        "set_type",
        "category",
        "instance_id",
        "condition",
        "image_path",
    ]

    writer = csv.DictWriter(
        f,
        fieldnames=fields
    )

    writer.writeheader()

    for idx, row in enumerate(rows):
        writer.writerow({
            "feature_index": idx,
            **row,
        })


print()
print("FEATURES :", FEATURES_NPY)
print("METADATA :", META_CSV)
print("STATUS   : PASS")
