from pathlib import Path
from collections import defaultdict
import csv
import json
import math
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

MODEL_NAME = "vit_base_patch16_dinov3.lvd1689m"

PATCH_FEATURES = OUT_ROOT / "dinov3b_public_patch_features.npy"
PATCH_META = OUT_ROOT / "dinov3b_public_patch_features.csv"

PAIR_CSV = OUT_ROOT / "patch_orbit_drift.csv"
COND_CSV = OUT_ROOT / "patch_orbit_condition_summary.csv"
CAT_CSV = OUT_ROOT / "patch_orbit_category_summary.csv"
SUMMARY_JSON = OUT_ROOT / "patch_orbit_summary.json"

SEED = 20260913
BATCH_SIZE = 16
NUM_WORKERS = 8

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def read_manifest(path, set_type):

    rows = []

    with path.open() as f:

        for row in csv.DictReader(f):

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

if len(rows) != 1084:
    raise RuntimeError(
        f"Expected 1084 images, found {len(rows)}"
    )


# ============================================================
# Model
# ============================================================

device = torch.device("cuda")

print("=" * 110)
print("OrbitAD A1.2 — PATCH-LEVEL ACQUISITION INSTABILITY")
print("=" * 110)

print("Loading:", MODEL_NAME)

model = timm.create_model(
    MODEL_NAME,
    pretrained=True,
    num_classes=0,
).eval().to(device)

data_config = resolve_model_data_config(model)

transform = create_transform(
    **data_config,
    is_training=False,
)

print("input size        :", data_config.get("input_size"))
print("num prefix tokens :", getattr(model, "num_prefix_tokens", "unknown"))


# ============================================================
# Dataset
# ============================================================

class PublicDataset(Dataset):

    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):

        row = self.records[idx]

        path = DATA_ROOT / row["image_path"]

        with Image.open(path) as im:
            image = transform(
                im.convert("RGB")
            )

        return image, idx


loader = DataLoader(
    PublicDataset(rows),
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    persistent_workers=(NUM_WORKERS > 0),
)


def extract_patch_tokens(images):

    tokens = model.forward_features(images)

    if isinstance(tokens, dict):

        if "x_norm_patchtokens" in tokens:
            tokens = tokens["x_norm_patchtokens"]
            prefix = 0

        elif "x" in tokens:
            tokens = tokens["x"]
            prefix = getattr(
                model,
                "num_prefix_tokens",
                1
            )

        else:
            raise RuntimeError(
                f"Unsupported forward_features dict keys: "
                f"{list(tokens.keys())}"
            )

    else:

        prefix = getattr(
            model,
            "num_prefix_tokens",
            1
        )

    if tokens.ndim != 3:

        raise RuntimeError(
            f"Expected [B,N,D], got "
            f"{tuple(tokens.shape)}"
        )

    if prefix > 0:
        tokens = tokens[:, prefix:, :]

    n_patches = tokens.shape[1]

    grid = int(round(
        math.sqrt(n_patches)
    ))

    if grid * grid != n_patches:

        raise RuntimeError(
            f"Patch token count {n_patches} "
            f"is not square."
        )

    tokens = F.normalize(
        tokens.float(),
        dim=-1
    )

    return tokens, grid


# ============================================================
# Extract
# ============================================================

feature_store = None
grid_size = None
feature_dim = None

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
            patches, current_grid = extract_patch_tokens(
                images
            )

        patches = patches.cpu().numpy()

        if feature_store is None:

            grid_size = current_grid
            feature_dim = patches.shape[-1]
            n_patches = patches.shape[1]

            print("patch grid        :", grid_size, "x", grid_size)
            print("patch tokens/image:", n_patches)
            print("feature dim       :", feature_dim)

            feature_store = np.lib.format.open_memmap(
                PATCH_FEATURES,
                mode="w+",
                dtype=np.float16,
                shape=(
                    len(rows),
                    n_patches,
                    feature_dim,
                ),
            )

        if current_grid != grid_size:
            raise RuntimeError(
                "Patch grid changed between batches."
            )

        idx_np = indices.numpy()

        feature_store[idx_np] = patches.astype(
            np.float16
        )

        seen += len(indices)

        if batch_idx % 10 == 0 or seen == len(rows):

            print(
                f"[{seen:4d}/{len(rows)}] "
                f"GPU={torch.cuda.memory_allocated()/1024**3:.2f} GB"
            )


feature_store.flush()

with PATCH_META.open("w", newline="") as f:

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


# ============================================================
# Reload feature mmap
# ============================================================

features = np.load(
    PATCH_FEATURES,
    mmap_mode="r"
)


# ============================================================
# Index structure
# ============================================================

groups = defaultdict(list)
regular_index = defaultdict(list)

for idx, row in enumerate(rows):

    row = {
        **row,
        "feature_index": idx,
    }

    groups[
        (
            row["set_type"],
            row["category"],
            row["instance_id"],
        )
    ].append(row)

    if row["condition"] == "regular":

        regular_index[
            (
                row["set_type"],
                row["category"],
            )
        ].append(row)


def summary(x):

    x = np.asarray(
        x,
        dtype=np.float64
    )

    x = x[np.isfinite(x)]

    if len(x) == 0:
        return {
            "n": 0,
            "mean": np.nan,
            "median": np.nan,
            "p90": np.nan,
            "p95": np.nan,
            "max": np.nan,
        }

    return {
        "n": len(x),
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "p90": float(np.percentile(x, 90)),
        "p95": float(np.percentile(x, 95)),
        "max": float(np.max(x)),
    }


pair_rows = []


# ============================================================
# Patch-level acquisition vs identity
# ============================================================

for (
    set_type,
    category,
    instance_id,
), members in sorted(groups.items()):

    regular = [
        x for x in members
        if x["condition"] == "regular"
    ]

    if len(regular) != 1:
        raise RuntimeError(
            f"{set_type}/{category}/{instance_id}: "
            f"regular count={len(regular)}"
        )

    reg_row = regular[0]

    reg = np.asarray(
        features[
            reg_row["feature_index"]
        ],
        dtype=np.float32,
    )

    other_rows = [
        x
        for x in regular_index[
            (set_type, category)
        ]
        if x["instance_id"] != instance_id
    ]

    if len(other_rows) == 0:
        continue

    others = np.stack(
        [
            np.asarray(
                features[x["feature_index"]],
                dtype=np.float32
            )
            for x in other_rows
        ],
        axis=0
    )

    # same spatial location, other physical instances
    dots_id = np.einsum(
        "pd,kpd->kp",
        reg,
        others,
    )

    d_id = np.median(
        1.0 - dots_id,
        axis=0
    )

    for current in members:

        condition = current["condition"]

        if condition == "regular":
            continue

        cur = np.asarray(
            features[
                current["feature_index"]
            ],
            dtype=np.float32,
        )

        d_acq = (
            1.0
            -
            np.sum(
                reg * cur,
                axis=1
            )
        )

        valid = d_id > 1e-6

        acr = np.full_like(
            d_acq,
            np.nan,
            dtype=np.float32
        )

        acr[valid] = (
            d_acq[valid]
            /
            d_id[valid]
        )

        fod_stats = summary(
            d_acq
        )

        acr_stats = summary(
            acr
        )

        fraction_ge_1 = float(
            np.mean(
                acr[valid] >= 1.0
            )
        )

        fraction_ge_05 = float(
            np.mean(
                acr[valid] >= 0.5
            )
        )

        pair_rows.append({
            "set_type": set_type,
            "category": category,
            "instance_id": instance_id,
            "condition": condition,

            "patch_count":
                int(valid.sum()),

            "patch_fod_mean":
                fod_stats["mean"],

            "patch_fod_median":
                fod_stats["median"],

            "patch_fod_p90":
                fod_stats["p90"],

            "patch_fod_p95":
                fod_stats["p95"],

            "patch_acr_mean":
                acr_stats["mean"],

            "patch_acr_median":
                acr_stats["median"],

            "patch_acr_p90":
                acr_stats["p90"],

            "patch_acr_p95":
                acr_stats["p95"],

            "fraction_patch_acr_ge_0_5":
                fraction_ge_05,

            "fraction_patch_acr_ge_1":
                fraction_ge_1,
        })


# ============================================================
# Save pair-level
# ============================================================

with PAIR_CSV.open("w", newline="") as f:

    writer = csv.DictWriter(
        f,
        fieldnames=list(
            pair_rows[0].keys()
        )
    )

    writer.writeheader()
    writer.writerows(pair_rows)


# ============================================================
# Aggregate helper
# ============================================================

def aggregate(members):

    return {
        "n_pairs": len(members),

        "patch_fod_median":
            float(np.median([
                x["patch_fod_median"]
                for x in members
            ])),

        "patch_fod_p90":
            float(np.median([
                x["patch_fod_p90"]
                for x in members
            ])),

        "patch_acr_median":
            float(np.median([
                x["patch_acr_median"]
                for x in members
            ])),

        "patch_acr_p90":
            float(np.median([
                x["patch_acr_p90"]
                for x in members
            ])),

        "fraction_patch_acr_ge_0_5":
            float(np.mean([
                x[
                    "fraction_patch_acr_ge_0_5"
                ]
                for x in members
            ])),

        "fraction_patch_acr_ge_1":
            float(np.mean([
                x[
                    "fraction_patch_acr_ge_1"
                ]
                for x in members
            ])),
    }


# ============================================================
# Condition summary
# ============================================================

cond_groups = defaultdict(list)

for row in pair_rows:

    cond_groups[
        (
            row["set_type"],
            row["condition"],
        )
    ].append(row)


cond_rows = []

for (
    set_type,
    condition,
), members in sorted(cond_groups.items()):

    cond_rows.append({
        "set_type": set_type,
        "condition": condition,
        **aggregate(members),
    })


with COND_CSV.open("w", newline="") as f:

    writer = csv.DictWriter(
        f,
        fieldnames=list(
            cond_rows[0].keys()
        )
    )

    writer.writeheader()
    writer.writerows(cond_rows)


# ============================================================
# Category summary
# ============================================================

cat_groups = defaultdict(list)

for row in pair_rows:

    cat_groups[
        (
            row["set_type"],
            row["category"],
        )
    ].append(row)


cat_rows = []

for (
    set_type,
    category,
), members in sorted(cat_groups.items()):

    cat_rows.append({
        "set_type": set_type,
        "category": category,
        **aggregate(members),
    })


with CAT_CSV.open("w", newline="") as f:

    writer = csv.DictWriter(
        f,
        fieldnames=list(
            cat_rows[0].keys()
        )
    )

    writer.writeheader()
    writer.writerows(cat_rows)


# ============================================================
# Global
# ============================================================

payload = {
    "model": MODEL_NAME,
    "grid_size": grid_size,
    "patch_tokens_per_image":
        grid_size * grid_size,
}

for set_type in [
    "normal",
    "defect",
]:

    members = [
        x for x in pair_rows
        if x["set_type"] == set_type
    ]

    payload[set_type] = aggregate(
        members
    )


with SUMMARY_JSON.open("w") as f:

    json.dump(
        payload,
        f,
        indent=2
    )


# ============================================================
# Console
# ============================================================

print()
print("=" * 110)
print("PATCH-LEVEL RESULTS")
print("=" * 110)

for set_type in [
    "normal",
    "defect",
]:

    x = payload[set_type]

    print()
    print(f"[{set_type.upper()} PATCH ORBIT]")

    print(
        f"  pair count               : "
        f"{x['n_pairs']}"
    )

    print(
        f"  patch FOD median         : "
        f"{x['patch_fod_median']:.6f}"
    )

    print(
        f"  patch FOD p90            : "
        f"{x['patch_fod_p90']:.6f}"
    )

    print(
        f"  patch ACR median         : "
        f"{x['patch_acr_median']:.6f}"
    )

    print(
        f"  patch ACR p90            : "
        f"{x['patch_acr_p90']:.6f}"
    )

    print(
        f"  P(patch ACR >= 0.5)      : "
        f"{x['fraction_patch_acr_ge_0_5']:.4f}"
    )

    print(
        f"  P(patch ACR >= 1.0)      : "
        f"{x['fraction_patch_acr_ge_1']:.4f}"
    )


print()
print("[CONDITION-WISE]")

print(
    f"{'set':8s} "
    f"{'condition':16s} "
    f"{'N':>5s} "
    f"{'ACR-med':>10s} "
    f"{'ACR-p90':>10s} "
    f"{'P>=0.5':>9s} "
    f"{'P>=1':>9s}"
)

print("-" * 85)

for r in cond_rows:

    print(
        f"{r['set_type']:8s} "
        f"{r['condition']:16s} "
        f"{r['n_pairs']:5d} "
        f"{r['patch_acr_median']:10.4f} "
        f"{r['patch_acr_p90']:10.4f} "
        f"{r['fraction_patch_acr_ge_0_5']:9.4f} "
        f"{r['fraction_patch_acr_ge_1']:9.4f}"
    )


print()
print("PATCH FEATURES :", PATCH_FEATURES)
print("PAIR CSV       :", PAIR_CSV)
print("CONDITION CSV  :", COND_CSV)
print("CATEGORY CSV   :", CAT_CSV)
print("SUMMARY JSON   :", SUMMARY_JSON)
print("STATUS         : PASS")
print("=" * 110)
