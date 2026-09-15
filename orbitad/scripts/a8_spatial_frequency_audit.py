from pathlib import Path
from collections import defaultdict
import csv
import json

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision.transforms.functional import resize, to_tensor
from torchvision.transforms import InterpolationMode

import timm
from timm.data import resolve_model_data_config


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

PUBLIC_META = (
    A2_ROOT / "public_orbit_meta.csv"
)

DETAIL_CSV = (
    OUT_ROOT / "a8_0_frequency_pairs.csv"
)

CATEGORY_CSV = (
    OUT_ROOT / "a8_0_frequency_category.csv"
)

SUMMARY_JSON = (
    OUT_ROOT / "a8_0_summary.json"
)


MODEL_NAME = (
    "vit_base_patch16_dinov3.lvd1689m"
)

INPUT_SIZE = 512
GRID = 32
FEATURE_DIM = 768

# Pre-registered spatial cutoff.
LOW_GRID = 8

PRIMARY_CONDITIONS = [
    "shift_1",
    "shift_2",
    "shift_3",
]


if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA unavailable"
    )

device = torch.device(
    "cuda"
)


# ============================================================
# Metadata
# ============================================================

rows = []

with PUBLIC_META.open() as f:

    for r in csv.DictReader(f):

        r["label"] = int(
            r["label"]
        )

        rows.append(r)


# A8.0 uses only normal public scenes.
normal_rows = [
    r
    for r in rows
    if r["label"] == 0
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


CATEGORIES = sorted({
    r["category"]
    for r in normal_rows
})


# ============================================================
# Backbone
# ============================================================

print("=" * 112)
print("A8.0 — REAL ACQUISITION DRIFT SPATIAL-FREQUENCY AUDIT")
print("=" * 112)


model = timm.create_model(
    MODEL_NAME,
    pretrained=True,
    num_classes=0,
    img_size=INPUT_SIZE,
).eval().to(device)


cfg = resolve_model_data_config(
    model
)


mean = torch.tensor(
    cfg["mean"],
    dtype=torch.float32,
    device=device,
).view(
    1, 3, 1, 1
)

std = torch.tensor(
    cfg["std"],
    dtype=torch.float32,
    device=device,
).view(
    1, 3, 1, 1
)


def load_image(rel_path):

    path = (
        DATA_ROOT / rel_path
    )

    with Image.open(path) as im:

        im = im.convert("RGB")

        im = resize(
            im,
            [INPUT_SIZE, INPUT_SIZE],
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        )

        return to_tensor(im)


def extract_feature(raw):

    x = raw[None].to(device)

    x = (
        x - mean
    ) / std

    with torch.inference_mode():

        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
        ):

            out = model.forward_features(x)


    if isinstance(out, dict):

        if "x_norm_patchtokens" in out:

            p = out["x_norm_patchtokens"]

        elif "x" in out:

            prefix = getattr(
                model,
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
            model,
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


    return p[
        0
    ].reshape(
        GRID,
        GRID,
        FEATURE_DIM,
    ).cpu()


# ============================================================
# Cache
# ============================================================

cache = {}


needed = []

for (
    category,
    instance_id,
), conditions in sorted(
    pairs.items()
):

    for condition in (
        ["regular"]
        + PRIMARY_CONDITIONS
    ):

        if condition in conditions:

            needed.append(
                (
                    category,
                    instance_id,
                    condition,
                    conditions[condition],
                )
            )


for i, (
    category,
    instance_id,
    condition,
    row,
) in enumerate(
    needed,
    1,
):

    cache[
        (
            category,
            instance_id,
            condition,
        )
    ] = extract_feature(
        load_image(
            row["image_path"]
        )
    )


    if (
        i % 25 == 0
        or
        i == len(needed)
    ):

        print(
            f"features "
            f"{i}/{len(needed)}"
        )


# ============================================================
# Spatial low-frequency projection
#
# Fixed 8x8 adaptive-average representation,
# bilinearly restored to 32x32.
#
# This is deliberately identical in spirit to
# the proposed future coarse drift predictor.
# ============================================================

def lowpass(delta):

    # [32,32,D] -> [1,D,32,32]
    x = delta.permute(
        2, 0, 1
    )[None]


    coarse = F.adaptive_avg_pool2d(
        x,
        (
            LOW_GRID,
            LOW_GRID,
        ),
    )


    restored = F.interpolate(
        coarse,
        size=(
            GRID,
            GRID,
        ),
        mode="bilinear",
        align_corners=False,
    )


    return restored[
        0
    ].permute(
        1, 2, 0
    )


def energy_fraction(delta):

    lp = lowpass(
        delta
    )

    total = float(
        (
            delta ** 2
        ).sum().item()
    )

    low = float(
        (
            lp ** 2
        ).sum().item()
    )


    return (
        low
        /
        max(
            total,
            1e-12,
        )
    )


def reconstruction_explained(delta):

    lp = lowpass(
        delta
    )

    residual = (
        delta - lp
    )

    total = float(
        (
            delta ** 2
        ).sum().item()
    )

    unexplained = float(
        (
            residual ** 2
        ).sum().item()
    )


    return (
        1.0
        -
        unexplained
        /
        max(
            total,
            1e-12,
        )
    )


# ============================================================
# Deterministic same-shift vs other-normal control
# ============================================================

detail = []


for category in CATEGORIES:

    for condition in PRIMARY_CONDITIONS:

        valid = sorted([
            instance_id
            for (
                cat,
                instance_id
            ), conditions
            in pairs.items()

            if (
                cat == category
                and
                "regular" in conditions
                and
                condition in conditions
            )
        ])


        if len(valid) < 2:
            continue


        for idx, instance_id in enumerate(
            valid
        ):

            other_id = valid[
                (
                    idx + 1
                )
                %
                len(valid)
            ]


            regular = cache[
                (
                    category,
                    instance_id,
                    "regular",
                )
            ]


            shifted = cache[
                (
                    category,
                    instance_id,
                    condition,
                )
            ]


            other_regular = cache[
                (
                    category,
                    other_id,
                    "regular",
                )
            ]


            delta_shift = (
                shifted
                -
                regular
            )


            delta_other = (
                other_regular
                -
                regular
            )


            shift_lfr = energy_fraction(
                delta_shift
            )

            other_lfr = energy_fraction(
                delta_other
            )


            shift_explained = (
                reconstruction_explained(
                    delta_shift
                )
            )


            detail.append({
                "category":
                    category,

                "instance_id":
                    instance_id,

                "condition":
                    condition,

                "other_instance":
                    other_id,

                "shift_lowfreq":
                    shift_lfr,

                "other_lowfreq":
                    other_lfr,

                "lowfreq_margin":
                    (
                        shift_lfr
                        -
                        other_lfr
                    ),

                "shift_explained":
                    shift_explained,
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
        if r["category"] == category
    ]


    shift_lfr = float(
        np.mean([
            r["shift_lowfreq"]
            for r in members
        ])
    )


    other_lfr = float(
        np.mean([
            r["other_lowfreq"]
            for r in members
        ])
    )


    explained = float(
        np.mean([
            r["shift_explained"]
            for r in members
        ])
    )


    category_rows.append({
        "category":
            category,

        "N":
            len(members),

        "shift_lowfreq":
            shift_lfr,

        "other_lowfreq":
            other_lfr,

        "margin":
            shift_lfr
            -
            other_lfr,

        "shift_explained":
            explained,

        "shift_more_lowfreq":
            bool(
                shift_lfr
                >
                other_lfr
            ),
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
# Aggregate + pre-registered decision
# ============================================================

mean_shift_lfr = float(
    np.mean([
        r["shift_lowfreq"]
        for r in detail
    ])
)


mean_other_lfr = float(
    np.mean([
        r["other_lowfreq"]
        for r in detail
    ])
)


mean_margin = (
    mean_shift_lfr
    -
    mean_other_lfr
)


mean_explained = float(
    np.mean([
        r["shift_explained"]
        for r in detail
    ])
)


category_wins = sum(
    r["shift_more_lowfreq"]
    for r in category_rows
)


signal_margin = (
    mean_margin >= 0.10
)

signal_categories = (
    category_wins >= 6
)

signal_explained = (
    mean_explained >= 0.50
)


decision = (
    "GO"
    if (
        signal_margin
        and
        signal_categories
        and
        signal_explained
    )
    else "NO_GO"
)


payload = {
    "protocol": {
        "backbone":
            MODEL_NAME,

        "feature_grid":
            "32x32",

        "low_frequency_grid":
            "8x8",

        "public_bad_or_gt_used":
            False,
    },

    "pre_registered": {
        "minimum_lowfreq_margin":
            0.10,

        "minimum_categories":
            6,

        "minimum_shift_energy_explained":
            0.50,
    },

    "observed": {
        "shift_lowfreq":
            mean_shift_lfr,

        "other_normal_lowfreq":
            mean_other_lfr,

        "lowfreq_margin":
            mean_margin,

        "shift_energy_explained":
            mean_explained,

        "categories_shift_more_lowfreq":
            int(category_wins),
    },

    "signals": {
        "frequency_separation":
            bool(signal_margin),

        "category_consistency":
            bool(signal_categories),

        "coarse_reconstructability":
            bool(signal_explained),
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
print("A8.0 SPATIAL-FREQUENCY RESULTS")
print("=" * 112)

print(
    f"{'category':15s}"
    f"{'Shift-LF':>11s}"
    f"{'Other-LF':>11s}"
    f"{'Margin':>10s}"
    f"{'Explained':>12s}"
)

print("-" * 59)


for r in category_rows:

    print(
        f"{r['category']:15s}"
        f"{r['shift_lowfreq']:11.4f}"
        f"{r['other_lowfreq']:11.4f}"
        f"{r['margin']:+10.4f}"
        f"{r['shift_explained']:12.4f}"
    )


print()
print("[AGGREGATE]")

print(
    "shift low-frequency fraction :",
    f"{mean_shift_lfr:.4f}"
)

print(
    "other-normal fraction        :",
    f"{mean_other_lfr:.4f}"
)

print(
    "low-frequency margin         :",
    f"{mean_margin:+.4f}"
)

print(
    "coarse energy explained      :",
    f"{mean_explained:.4f}"
)

print(
    "categories positive          :",
    f"{category_wins}/8"
)


print()
print("[PRE-REGISTERED A8.0 GO / NO-GO]")

print(
    "LF margin >= +0.10       :",
    signal_margin
)

print(
    "positive in >=6/8 cats   :",
    signal_categories
)

print(
    "energy explained >= .50  :",
    signal_explained
)

print()
print(
    "DECISION :",
    decision
)

print()
print(
    "SUMMARY :",
    SUMMARY_JSON
)

print("=" * 112)
