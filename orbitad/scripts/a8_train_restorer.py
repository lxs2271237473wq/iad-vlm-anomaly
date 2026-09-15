from pathlib import Path
from collections import defaultdict
import csv
import json
import random

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torchvision.transforms.functional import (
    resize,
    to_tensor,
)
from torchvision.transforms import InterpolationMode

import timm
from timm.data import resolve_model_data_config


PROJECT = Path(
    "/root/private_data/iad-vlm-anomaly/orbitad"
)

import sys
sys.path.insert(
    0,
    str(PROJECT / "src")
)

from orbitad.illumination.spatial_orbit import (
    directional_gradient,
    spotlight,
    shadow,
    low_frequency_field,
)


DATA_ROOT = Path(
    "/root/private_data/iad-vlm-anomaly/datasets/MVTec_AD_2"
)

A2_ROOT = Path(
    "orbitad/results/a2_counterfactual_persistence"
)

OUT_ROOT = Path(
    "orbitad/results/a8_spatial_drift"
)

OUT_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)

CHECKPOINT = (
    OUT_ROOT / "a8_1_restorer.pt"
)

DETAIL_CSV = (
    OUT_ROOT / "a8_1_real_shift_pairs.csv"
)

CATEGORY_CSV = (
    OUT_ROOT / "a8_1_category_summary.csv"
)

SUMMARY_JSON = (
    OUT_ROOT / "a8_1_summary.json"
)


MODEL_NAME = (
    "vit_base_patch16_dinov3.lvd1689m"
)

INPUT_SIZE = 512
GRID = 32
COARSE = 8
FEATURE_DIM = 768
HIDDEN_DIM = 128

BATCH_SIZE = 2
NUM_WORKERS = 6

EPOCHS = 3
LR = 1e-3
WEIGHT_DECAY = 1e-4

SEED = 20260913


CATEGORIES = [
    "can",
    "fabric",
    "fruit_jelly",
    "rice",
    "sheet_metal",
    "vial",
    "wallplugs",
    "walnuts",
]


PRIMARY_CONDITIONS = [
    "shift_1",
    "shift_2",
    "shift_3",
]


VARIANTS = [
    "exposure_070",
    "exposure_130",
    "gradient_045",
    "gradient_135",
    "spotlight",
    "shadow",
    "lowfreq",
]


random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA unavailable"
    )

device = torch.device(
    "cuda"
)


# ============================================================
# Training files
# ============================================================

VALID_SUFFIXES = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
}


train_paths = []

for category in CATEGORIES:

    root = (
        DATA_ROOT
        / category
        / "train"
        / "good"
    )

    paths = sorted([
        p
        for p in root.rglob("*")
        if (
            p.is_file()
            and
            p.suffix.lower()
            in VALID_SUFFIXES
        )
    ])

    if not paths:
        raise RuntimeError(
            f"No train/good images: {root}"
        )

    train_paths.extend(paths)


print("=" * 112)
print("A8.1 — COARSE ACQUISITION DRIFT RESTORER")
print("=" * 112)

print(
    "train images :",
    len(train_paths)
)

print(
    "epochs       :",
    EPOCHS
)

print(
    "coarse grid  :",
    f"{COARSE}x{COARSE}"
)


# ============================================================
# Backbone
# ============================================================

backbone = timm.create_model(
    MODEL_NAME,
    pretrained=True,
    num_classes=0,
    img_size=INPUT_SIZE,
).eval().to(device)


for parameter in backbone.parameters():
    parameter.requires_grad_(False)


cfg = resolve_model_data_config(
    backbone
)


mean = torch.tensor(
    cfg["mean"],
    dtype=torch.float32,
    device=device,
).view(
    1,
    3,
    1,
    1,
)

std = torch.tensor(
    cfg["std"],
    dtype=torch.float32,
    device=device,
).view(
    1,
    3,
    1,
    1,
)


def load_raw(path):

    with Image.open(path) as im:

        im = im.convert("RGB")

        im = resize(
            im,
            [INPUT_SIZE, INPUT_SIZE],
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        )

        return to_tensor(im)


def extract_features(raw):

    x = raw.to(
        device,
        non_blocking=True,
    )

    x = (
        x - mean
    ) / std


    with torch.inference_mode():

        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
        ):

            out = backbone.forward_features(
                x
            )


    if isinstance(out, dict):

        if "x_norm_patchtokens" in out:

            p = out[
                "x_norm_patchtokens"
            ]

        elif "x" in out:

            prefix = getattr(
                backbone,
                "num_prefix_tokens",
                1,
            )

            p = out["x"][
                :,
                prefix:,
                :
            ]

        else:

            raise RuntimeError(
                str(list(out.keys()))
            )

    else:

        prefix = getattr(
            backbone,
            "num_prefix_tokens",
            1,
        )

        p = out[
            :,
            prefix:,
            :
        ]


    p = F.normalize(
        p.float(),
        dim=-1,
    )


    if p.shape[1:] != (
        1024,
        FEATURE_DIM,
    ):

        raise RuntimeError(
            f"Unexpected feature shape: "
            f"{tuple(p.shape)}"
        )


    return (
        p.reshape(
            p.shape[0],
            GRID,
            GRID,
            FEATURE_DIM,
        )
        .permute(
            0,
            3,
            1,
            2,
        )
        .contiguous()
    )


# ============================================================
# Frozen synthetic acquisition family
# ============================================================

def make_variant(
    raw,
    name,
    seed,
):

    if name == "exposure_070":

        return (
            raw * 0.70
        ).clamp(
            0.0,
            1.0,
        )


    if name == "exposure_130":

        return (
            raw * 1.30
        ).clamp(
            0.0,
            1.0,
        )


    if name == "gradient_045":

        return directional_gradient(
            raw,
            0.30,
            45,
        )


    if name == "gradient_135":

        return directional_gradient(
            raw,
            0.30,
            135,
        )


    if name == "spotlight":

        return spotlight(
            raw,
            0.30,
            center_x=-0.35,
            center_y=0.20,
            sigma_fraction=0.30,
        )


    if name == "shadow":

        return shadow(
            raw,
            0.30,
            center_x=0.35,
            center_y=-0.20,
            sigma_fraction=0.30,
        )


    if name == "lowfreq":

        generator = torch.Generator(
            device=device
        )

        generator.manual_seed(
            seed
        )

        return low_frequency_field(
            raw,
            0.10,
            8,
            generator=generator,
        )


    raise ValueError(
        name
    )


# ============================================================
# Fixed low-frequency target
# ============================================================

def lowpass8(x):

    coarse = F.adaptive_avg_pool2d(
        x,
        (
            COARSE,
            COARSE,
        ),
    )

    return F.interpolate(
        coarse,
        size=(
            GRID,
            GRID,
        ),
        mode="bilinear",
        align_corners=False,
    )


# ============================================================
# Tiny coarse drift predictor
# ============================================================

class CoarseDriftRestorer(nn.Module):

    def __init__(self):

        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(
                FEATURE_DIM,
                HIDDEN_DIM,
                kernel_size=1,
                bias=True,
            ),

            nn.GELU(),

            nn.Conv2d(
                HIDDEN_DIM,
                HIDDEN_DIM,
                kernel_size=3,
                padding=1,
                groups=HIDDEN_DIM,
                bias=True,
            ),

            nn.GELU(),

            nn.Conv2d(
                HIDDEN_DIM,
                FEATURE_DIM,
                kernel_size=1,
                bias=True,
            ),
        )


    def forward(
        self,
        feature,
    ):

        coarse = (
            F.adaptive_avg_pool2d(
                feature,
                (
                    COARSE,
                    COARSE,
                ),
            )
        )

        drift = self.net(
            coarse
        )

        return F.interpolate(
            drift,
            size=(
                GRID,
                GRID,
            ),
            mode="bilinear",
            align_corners=False,
        )


restorer = CoarseDriftRestorer().to(
    device
)


n_params = sum(
    p.numel()
    for p in restorer.parameters()
)


print(
    "restorer parameters :",
    n_params
)


# ============================================================
# Dataset
# ============================================================

class TrainDataset(Dataset):

    def __init__(
        self,
        paths,
    ):

        self.paths = paths


    def __len__(self):

        return len(
            self.paths
        )


    def __getitem__(
        self,
        idx,
    ):

        return load_raw(
            self.paths[idx]
        )


generator = torch.Generator()
generator.manual_seed(SEED)


loader = DataLoader(
    TrainDataset(
        train_paths
    ),

    batch_size=BATCH_SIZE,

    shuffle=True,

    num_workers=NUM_WORKERS,

    pin_memory=True,

    persistent_workers=(
        NUM_WORKERS > 0
    ),

    generator=generator,
)


optimizer = torch.optim.AdamW(
    restorer.parameters(),
    lr=LR,
    weight_decay=WEIGHT_DECAY,
)


# ============================================================
# Fixed three-epoch training
# ============================================================

global_step = 0


for epoch in range(
    EPOCHS
):

    restorer.train()

    losses = []
    drift_losses = []
    identity_losses = []


    for step, raw in enumerate(
        loader
    ):

        raw = raw.to(
            device,
            non_blocking=True,
        )


        variant_name = VARIANTS[
            global_step
            %
            len(VARIANTS)
        ]


        transformed = make_variant(
            raw,
            variant_name,
            SEED
            +
            global_step
            *
            7919,
        )


        # One frozen-backbone call.
        stacked = torch.cat([
            raw,
            transformed,
        ], dim=0)


        features = extract_features(
            stacked
        )


        B = raw.shape[0]

        f0 = features[
            :B
        ]

        ft = features[
            B:
        ]


        target = lowpass8(
            ft - f0
        )


        pred_t = restorer(
            ft.detach()
        )

        pred_0 = restorer(
            f0.detach()
        )


        loss_drift = F.mse_loss(
            pred_t,
            target.detach(),
        )


        loss_identity = F.mse_loss(
            pred_0,
            torch.zeros_like(
                pred_0
            ),
        )


        # Equal-weight two-part objective.
        loss = (
            0.5
            *
            (
                loss_drift
                +
                loss_identity
            )
        )


        optimizer.zero_grad(
            set_to_none=True
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            restorer.parameters(),
            max_norm=1.0,
        )

        optimizer.step()


        losses.append(
            float(
                loss.item()
            )
        )

        drift_losses.append(
            float(
                loss_drift.item()
            )
        )

        identity_losses.append(
            float(
                loss_identity.item()
            )
        )


        global_step += 1


        if (
            (step + 1) % 100 == 0
            or
            step + 1 == len(loader)
        ):

            print(
                f"epoch {epoch+1}/{EPOCHS} "
                f"step {step+1}/{len(loader)} "
                f"loss={np.mean(losses[-100:]):.6f}"
            )


    print(
        f"[EPOCH {epoch+1}] "
        f"loss={np.mean(losses):.6f} "
        f"drift={np.mean(drift_losses):.6f} "
        f"id={np.mean(identity_losses):.6f}"
    )


torch.save(
    {
        "state_dict":
            restorer.state_dict(),

        "model_name":
            MODEL_NAME,

        "coarse_grid":
            COARSE,

        "hidden_dim":
            HIDDEN_DIM,

        "epochs":
            EPOCHS,

        "seed":
            SEED,
    },
    CHECKPOINT,
)


print()
print(
    "checkpoint:",
    CHECKPOINT
)


# ============================================================
# REAL public-normal shift audit
#
# Model is frozen before public data are inspected.
# No public result affects training.
# ============================================================

restorer.eval()


PUBLIC_META = (
    A2_ROOT / "public_orbit_meta.csv"
)


rows = []

with PUBLIC_META.open() as f:

    for row in csv.DictReader(f):

        row["label"] = int(
            row["label"]
        )

        rows.append(row)


normal_rows = [
    row
    for row in rows
    if row["label"] == 0
]


pairs = defaultdict(dict)


for row in normal_rows:

    pairs[
        (
            row["category"],
            row["instance_id"],
        )
    ][
        row["condition"]
    ] = row


detail = []


def patch_distance(
    a,
    b,
):

    # [1,D,H,W]
    d = torch.sqrt(
        (
            (
                a - b
            ) ** 2
        ).sum(
            dim=1
        ).clamp_min(
            0.0
        )
    )

    return float(
        d.mean().item()
    )


def mean_patch_norm(
    x
):

    return float(
        torch.sqrt(
            (
                x ** 2
            ).sum(
                dim=1
            ).clamp_min(
                0.0
            )
        ).mean().item()
    )


with torch.inference_mode():

    for (
        category,
        instance_id,
    ), conditions in sorted(
        pairs.items()
    ):

        if "regular" not in conditions:
            continue


        regular_raw = load_raw(
            DATA_ROOT
            /
            conditions[
                "regular"
            ][
                "image_path"
            ]
        )[None]


        f_regular = extract_features(
            regular_raw
        )


        pred_regular = restorer(
            f_regular
        )


        identity_norm = (
            mean_patch_norm(
                pred_regular
            )
        )


        for condition in (
            PRIMARY_CONDITIONS
        ):

            if condition not in conditions:
                continue


            shifted_raw = load_raw(
                DATA_ROOT
                /
                conditions[
                    condition
                ][
                    "image_path"
                ]
            )[None]


            f_shift = extract_features(
                shifted_raw
            )


            raw_distance = patch_distance(
                f_shift,
                f_regular,
            )


            predicted_drift = restorer(
                f_shift
            )


            canonical = F.normalize(
                f_shift
                -
                predicted_drift,
                dim=1,
            )


            restored_distance = (
                patch_distance(
                    canonical,
                    f_regular,
                )
            )


            detail.append({
                "category":
                    category,

                "instance_id":
                    instance_id,

                "condition":
                    condition,

                "raw_distance":
                    raw_distance,

                "restored_distance":
                    restored_distance,

                "distance_ratio":
                    (
                        restored_distance
                        /
                        max(
                            raw_distance,
                            1e-12,
                        )
                    ),

                "identity_drift_norm":
                    identity_norm,
            })


with DETAIL_CSV.open(
    "w",
    newline="",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=list(
            detail[0].keys()
        ),
    )

    writer.writeheader()

    writer.writerows(
        detail
    )


# ============================================================
# Category summary
# ============================================================

category_rows = []


for category in CATEGORIES:

    members = [
        r
        for r in detail
        if r[
            "category"
        ] == category
    ]


    raw_mean = float(
        np.mean([
            r[
                "raw_distance"
            ]
            for r in members
        ])
    )


    restored_mean = float(
        np.mean([
            r[
                "restored_distance"
            ]
            for r in members
        ])
    )


    ratio = (
        restored_mean
        /
        max(
            raw_mean,
            1e-12,
        )
    )


    identity = float(
        np.mean([
            r[
                "identity_drift_norm"
            ]
            for r in members
        ])
    )


    category_rows.append({
        "category":
            category,

        "N":
            len(members),

        "raw_distance":
            raw_mean,

        "restored_distance":
            restored_mean,

        "distance_ratio":
            ratio,

        "improved":
            bool(
                ratio < 1.0
            ),

        "identity_drift_norm":
            identity,
    })


with CATEGORY_CSV.open(
    "w",
    newline="",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=list(
            category_rows[0].keys()
        ),
    )

    writer.writeheader()

    writer.writerows(
        category_rows
    )


# ============================================================
# Pre-registered decision
# ============================================================

mean_raw = float(
    np.mean([
        r[
            "raw_distance"
        ]
        for r in detail
    ])
)


mean_restored = float(
    np.mean([
        r[
            "restored_distance"
        ]
        for r in detail
    ])
)


mean_ratio = (
    mean_restored
    /
    max(
        mean_raw,
        1e-12,
    )
)


mean_identity = float(
    np.mean([
        r[
            "identity_drift_norm"
        ]
        for r in detail
    ])
)


category_wins = sum(
    r[
        "improved"
    ]
    for r in category_rows
)


signal_restoration = (
    mean_ratio <= 0.85
)


signal_categories = (
    category_wins >= 6
)


signal_identity = (
    mean_identity <= 0.05
)


decision = (
    "GO"
    if (
        signal_restoration
        and
        signal_categories
        and
        signal_identity
    )
    else "NO_GO"
)


payload = {
    "training": {
        "train_good_only":
            True,

        "synthetic_pairs_only":
            True,

        "epochs":
            EPOCHS,

        "coarse_grid":
            COARSE,

        "restorer_parameters":
            n_params,

        "public_data_used_for_training":
            False,
    },

    "pre_registered": {
        "maximum_real_shift_distance_ratio":
            0.85,

        "minimum_categories_improved":
            6,

        "maximum_identity_drift_norm":
            0.05,
    },

    "observed": {
        "mean_raw_distance":
            mean_raw,

        "mean_restored_distance":
            mean_restored,

        "restored_over_raw":
            mean_ratio,

        "relative_reduction":
            (
                1.0 - mean_ratio
            ),

        "categories_improved":
            int(
                category_wins
            ),

        "mean_identity_drift_norm":
            mean_identity,
    },

    "signals": {
        "real_shift_restoration":
            bool(
                signal_restoration
            ),

        "category_consistency":
            bool(
                signal_categories
            ),

        "identity_preservation":
            bool(
                signal_identity
            ),
    },

    "decision":
        decision,
}


with SUMMARY_JSON.open(
    "w"
) as f:

    json.dump(
        payload,
        f,
        indent=2,
    )


# ============================================================
# Console
# ============================================================

print()
print("=" * 112)
print("A8.1 REAL-SHIFT RESTORATION RESULTS")
print("=" * 112)

print(
    f"{'category':15s}"
    f"{'RawDist':>11s}"
    f"{'RestDist':>11s}"
    f"{'Ratio':>10s}"
    f"{'Identity':>11s}"
)

print("-" * 58)


for r in category_rows:

    print(
        f"{r['category']:15s}"
        f"{r['raw_distance']:11.4f}"
        f"{r['restored_distance']:11.4f}"
        f"{r['distance_ratio']:10.4f}"
        f"{r['identity_drift_norm']:11.4f}"
    )


print()
print("[AGGREGATE]")

print(
    "raw distance          :",
    f"{mean_raw:.4f}"
)

print(
    "restored distance     :",
    f"{mean_restored:.4f}"
)

print(
    "restored / raw        :",
    f"{mean_ratio:.4f}"
)

print(
    "relative reduction    :",
    f"{100*(1-mean_ratio):.2f}%"
)

print(
    "categories improved   :",
    f"{category_wins}/8"
)

print(
    "identity drift norm   :",
    f"{mean_identity:.4f}"
)


print()
print("[PRE-REGISTERED A8.1 GO / NO-GO]")

print(
    "distance ratio <= .85 :",
    signal_restoration
)

print(
    "improved in >=6/8     :",
    signal_categories
)

print(
    "identity drift <= .05 :",
    signal_identity
)

print()
print(
    "DECISION :",
    decision
)

print()
print(
    "CHECKPOINT:",
    CHECKPOINT
)

print(
    "SUMMARY   :",
    SUMMARY_JSON
)

print("=" * 112)
