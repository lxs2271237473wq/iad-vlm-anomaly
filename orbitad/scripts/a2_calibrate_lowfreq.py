from pathlib import Path
import sys
import csv
import json
import math

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision.transforms.functional import (
    to_tensor,
    normalize,
)
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import resize

import timm
from timm.data import resolve_model_data_config


PROJECT_ROOT = Path(
    "/root/private_data/iad-vlm-anomaly/orbitad"
)

sys.path.insert(
    0,
    str(PROJECT_ROOT / "src")
)

from orbitad.illumination.spatial_orbit import (
    exposure,
    directional_gradient,
    spotlight,
    shadow,
    low_frequency_field,
)


DATA_ROOT = Path(
    "/root/private_data/iad-vlm-anomaly/datasets/MVTec_AD_2"
)

ROOT = Path(
    "orbitad/results/a2_synthetic_alignment"
)

ROOT.mkdir(
    parents=True,
    exist_ok=True,
)

MEMORY_ROOT = Path(
    "orbitad/results/a1_anomaly_score_sensitivity/memory"
)

VAL_MAPS = Path(
    "orbitad/results/a1_anomaly_score_sensitivity/"
    "validation_patch_maps.npy"
)

VAL_META = Path(
    "orbitad/results/a1_anomaly_score_sensitivity/"
    "validation_patch_maps.csv"
)

DETAIL_CSV = ROOT / "lowfreq_calibration_scores.csv"
SUMMARY_CSV = ROOT / "lowfreq_calibration_summary.csv"
SUMMARY_JSON = ROOT / "a2_0b_lowfreq_summary.json"

MODEL_NAME = "vit_base_patch16_dinov3.lvd1689m"

INPUT_SIZE = 512
TOPK = 10

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

SEED = 20260913


# ============================================================
# Model
# ============================================================

if not torch.cuda.is_available():
    raise RuntimeError("CUDA unavailable.")

device = torch.device("cuda")

print("=" * 105)
print("OrbitAD A2.0 — SYNTHETIC-TO-REAL ACQUISITION ALIGNMENT")
print("=" * 105)

model = timm.create_model(
    MODEL_NAME,
    pretrained=True,
    num_classes=0,
    img_size=INPUT_SIZE,
).eval().to(device)

data_config = resolve_model_data_config(model)

MEAN = data_config["mean"]
STD = data_config["std"]


def extract_patch_tokens(x):

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
            p = out["x"][:, prefix:, :]

        else:
            raise RuntimeError(
                f"Unsupported keys: {list(out.keys())}"
            )

    else:

        prefix = getattr(
            model,
            "num_prefix_tokens",
            1,
        )

        p = out[:, prefix:, :]

    p = F.normalize(
        p.float(),
        dim=-1,
    )

    if p.shape[1:] != (1024, 768):
        raise RuntimeError(
            f"Unexpected patch shape: {tuple(p.shape)}"
        )

    return p


# ============================================================
# Validation metadata + regular scores
# ============================================================

val_maps = np.load(
    VAL_MAPS,
    mmap_mode="r",
)

val_rows = []

with VAL_META.open() as f:

    for row in csv.DictReader(f):

        row["map_index"] = int(
            row["map_index"]
        )

        val_rows.append(row)

if len(val_rows) != 302:
    raise RuntimeError(
        f"Expected 302 validation rows, got {len(val_rows)}"
    )


def top10_score(score_map):

    x = np.asarray(
        score_map,
        dtype=np.float32,
    ).reshape(-1)

    if len(x) < TOPK:
        return float(x.mean())

    idx = np.argpartition(
        x,
        -TOPK,
    )[-TOPK:]

    return float(
        x[idx].mean()
    )


regular_scores = {
    i: top10_score(
        val_maps[
            row["map_index"]
        ]
    )
    for i, row in enumerate(val_rows)
}


# ============================================================
# Image preparation
# ============================================================

def load_base_image(rel_path):

    with Image.open(
        DATA_ROOT / rel_path
    ) as im:

        im = im.convert("RGB")

        im = resize(
            im,
            [INPUT_SIZE, INPUT_SIZE],
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        )

    return to_tensor(im)


def normalize_batch(x):

    return torch.stack([
        normalize(
            image,
            MEAN,
            STD,
        )
        for image in x
    ])


# ============================================================
# Synthetic variants
# ============================================================

SPATIAL_VARIANTS = {

    "lowfreq_g2_a10":
        lambda x, g: low_frequency_field(
            x, 0.10, 2, generator=g
        ),

    "lowfreq_g2_a15":
        lambda x, g: low_frequency_field(
            x, 0.15, 2, generator=g
        ),

    "lowfreq_g2_a20":
        lambda x, g: low_frequency_field(
            x, 0.20, 2, generator=g
        ),

    "lowfreq_g4_a10":
        lambda x, g: low_frequency_field(
            x, 0.10, 4, generator=g
        ),

    "lowfreq_g4_a15":
        lambda x, g: low_frequency_field(
            x, 0.15, 4, generator=g
        ),

    "lowfreq_g4_a20":
        lambda x, g: low_frequency_field(
            x, 0.20, 4, generator=g
        ),

    "lowfreq_g8_a10":
        lambda x, g: low_frequency_field(
            x, 0.10, 8, generator=g
        ),

    "lowfreq_g8_a15":
        lambda x, g: low_frequency_field(
            x, 0.15, 8, generator=g
        ),

    "lowfreq_g8_a20":
        lambda x, g: low_frequency_field(
            x, 0.20, 8, generator=g
        ),
}

CONTROL_VARIANTS = {}

VARIANTS = {
    **SPATIAL_VARIANTS,
}


# ============================================================
# Nearest-normal scoring
# ============================================================

def score_patch_features(
    patch_features,
    memory_gpu,
    query_chunk=512,
):

    # patch_features: [V,1024,768]
    V = patch_features.shape[0]

    flat = patch_features.reshape(
        -1,
        patch_features.shape[-1],
    )

    scores = []

    for start in range(
        0,
        flat.shape[0],
        query_chunk,
    ):

        q = flat[
            start:start + query_chunk
        ].to(
            dtype=torch.float16
        )

        sim = (
            q @ memory_gpu.T
        )

        best = sim.max(
            dim=1
        ).values

        d = (
            1.0
            - best.float()
        ).clamp_min(0.0)

        scores.append(
            d.cpu()
        )

    scores = torch.cat(
        scores
    ).reshape(
        V,
        1024,
    )

    topk = torch.topk(
        scores,
        k=TOPK,
        dim=1,
    ).values

    return (
        topk
        .mean(dim=1)
        .numpy()
    )


# ============================================================
# Main
# ============================================================

detail_rows = []

rows_by_category = {
    cat: []
    for cat in CATEGORIES
}

for idx, row in enumerate(val_rows):
    rows_by_category[
        row["category"]
    ].append(
        (idx, row)
    )


for category in CATEGORIES:

    print()
    print(
        "=" * 105
    )
    print(
        "CATEGORY:",
        category
    )
    print(
        "=" * 105
    )

    memory = np.load(
        MEMORY_ROOT / f"{category}.npy"
    )

    memory_gpu = torch.from_numpy(
        memory.astype(
            np.float16,
            copy=False,
        )
    ).to(device)

    cat_rows = rows_by_category[
        category
    ]

    for local_n, (
        global_idx,
        row,
    ) in enumerate(cat_rows):

        base = load_base_image(
            row["image_path"]
        )[None].to(device)

        variant_names = []
        variant_tensors = []

        for variant_idx, (
            name,
            fn,
        ) in enumerate(
            VARIANTS.items()
        ):

            generator = torch.Generator(
                device=device
            )

            generator.manual_seed(
                SEED
                + global_idx * 100
                + variant_idx
            )

            transformed = fn(
                base,
                generator,
            )

            variant_names.append(
                name
            )

            variant_tensors.append(
                transformed[0]
            )

        batch_raw = torch.stack(
            variant_tensors,
            dim=0,
        )

        batch = normalize_batch(
            batch_raw
        )

        with torch.inference_mode():

            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
            ):

                patches = extract_patch_tokens(
                    batch
                )

        synthetic_scores = score_patch_features(
            patches,
            memory_gpu,
        )

        ref = regular_scores[
            global_idx
        ]

        for name, score in zip(
            variant_names,
            synthetic_scores,
        ):

            ratio = (
                float(score)
                /
                max(
                    ref,
                    1e-8,
                )
            )

            change_pct = (
                100.0
                * (
                    float(score)
                    - ref
                )
                /
                max(
                    abs(ref),
                    1e-8,
                )
            )

            family = (
                "exposure"
                if name.startswith(
                    "exposure_"
                )
                else "spatial"
            )

            detail_rows.append({
                "category":
                    category,

                "instance_id":
                    row["instance_id"],

                "variant":
                    name,

                "family":
                    family,

                "regular_score":
                    ref,

                "synthetic_score":
                    float(score),

                "score_ratio":
                    ratio,

                "score_change_pct":
                    change_pct,

                "increase_ge_30pct":
                    ratio >= 1.30,
            })

        if (
            (local_n + 1) % 10 == 0
            or
            local_n + 1 == len(cat_rows)
        ):
            print(
                f"{local_n + 1:4d}/"
                f"{len(cat_rows):4d}"
            )

    del memory_gpu
    torch.cuda.empty_cache()


# ============================================================
# Save details
# ============================================================

with DETAIL_CSV.open(
    "w",
    newline="",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=list(
            detail_rows[0].keys()
        )
    )

    writer.writeheader()
    writer.writerows(
        detail_rows
    )


# ============================================================
# Aggregate
# ============================================================

summary_rows = []

for name in VARIANTS:

    members = [
        x
        for x in detail_rows
        if x["variant"] == name
    ]

    changes = np.asarray(
        [
            x["score_change_pct"]
            for x in members
        ],
        dtype=np.float64,
    )

    ratios = np.asarray(
        [
            x["score_ratio"]
            for x in members
        ],
        dtype=np.float64,
    )

    frac30 = float(
        np.mean(
            ratios >= 1.30
        )
    )

    family = members[0][
        "family"
    ]

    summary_rows.append({
        "variant":
            name,

        "family":
            family,

        "n":
            len(members),

        "median_score_ratio":
            float(
                np.median(ratios)
            ),

        "median_score_change_pct":
            float(
                np.median(changes)
            ),

        "p90_score_change_pct":
            float(
                np.percentile(
                    changes,
                    90,
                )
            ),

        "fraction_increase_ge_30pct":
            frac30,
    })


with SUMMARY_CSV.open(
    "w",
    newline="",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=list(
            summary_rows[0].keys()
        )
    )

    writer.writeheader()
    writer.writerows(
        summary_rows
    )


# ============================================================
# A2.0b moderate-orbit calibration
# ============================================================

eligible = []

for row in summary_rows:

    median_change = row[
        "median_score_change_pct"
    ]

    tail30 = row[
        "fraction_increase_ge_30pct"
    ]

    row["moderate_candidate"] = bool(
        3.0 <= median_change <= 12.0
        and
        tail30 <= 0.30
    )

    if row["moderate_candidate"]:
        eligible.append(row)


# Prefer a middle-strength candidate rather than
# the most aggressive candidate.
#
# Ranking is deterministic:
# first closer to 7% median increase,
# then lower +30% tail.
if eligible:

    selected = sorted(
        eligible,
        key=lambda r: (
            abs(
                r["median_score_change_pct"]
                - 7.0
            ),
            r[
                "fraction_increase_ge_30pct"
            ],
            r["variant"],
        )
    )[0]

    decision = "PASS"

else:

    selected = None
    decision = "FAIL"


payload = {
    "protocol": {
        "source":
            "validation/good regular only",

        "public_gt_used":
            False,
    },

    "moderate_orbit_definition": {
        "median_score_change_pct_min":
            3.0,

        "median_score_change_pct_max":
            12.0,

        "fraction_increase_ge_30pct_max":
            0.30,
    },

    "eligible_variants": [
        r["variant"]
        for r in eligible
    ],

    "selected_variant": (
        selected["variant"]
        if selected is not None
        else None
    ),

    "selected_statistics": (
        selected
        if selected is not None
        else {}
    ),

    "decision":
        decision,
}


with SUMMARY_JSON.open("w") as f:

    json.dump(
        payload,
        f,
        indent=2,
    )


# ============================================================
# Console
# ============================================================

print()
print("=" * 105)
print("A2.0b LOW-FREQUENCY ORBIT CALIBRATION")
print("=" * 105)

print(
    f"{'variant':20s}"
    f"{'N':>6s}"
    f"{'ratio-med':>12s}"
    f"{'change%':>11s}"
    f"{'p90%':>11s}"
    f"{'P(+30%)':>11s}"
    f"{'eligible':>11s}"
)

print("-" * 82)

for r in summary_rows:

    eligible_flag = (
        3.0
        <= r["median_score_change_pct"]
        <= 12.0
        and
        r["fraction_increase_ge_30pct"]
        <= 0.30
    )

    print(
        f"{r['variant']:20s}"
        f"{r['n']:6d}"
        f"{r['median_score_ratio']:12.4f}"
        f"{r['median_score_change_pct']:11.2f}"
        f"{r['p90_score_change_pct']:11.2f}"
        f"{r['fraction_increase_ge_30pct']:11.4f}"
        f"{str(eligible_flag):>11s}"
    )


print()
print("[CALIBRATION]")

print(
    "eligible variants :",
    [
        r["variant"]
        for r in eligible
    ]
)

print(
    "selected variant  :",
    (
        selected["variant"]
        if selected is not None
        else "NONE"
    )
)

print()
print("DECISION :", decision)

print()
print("DETAIL  :", DETAIL_CSV)
print("SUMMARY :", SUMMARY_CSV)
print("JSON    :", SUMMARY_JSON)

print("=" * 105)
