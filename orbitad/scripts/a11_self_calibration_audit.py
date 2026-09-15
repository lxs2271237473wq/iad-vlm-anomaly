from pathlib import Path
from collections import defaultdict
import csv
import json
import re

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

A1_ROOT = Path(
    "orbitad/results/a1_anomaly_score_sensitivity"
)

MEM_ROOT = Path(
    "orbitad/results/a3_resolution_audit/memory"
)

OUT = Path(
    "orbitad/results/a11_self_calibration"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)


PUBLIC_META = (
    A1_ROOT / "public_patch_maps.csv"
)

DETAIL_CSV = (
    OUT / "a11_0a_pairs.csv"
)

CATEGORY_CSV = (
    OUT / "a11_0a_category.csv"
)

SUMMARY_JSON = (
    OUT / "a11_0a_summary.json"
)


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

SHIFTS = [
    "shift_1",
    "shift_2",
    "shift_3",
]


MODEL_NAME = (
    "vit_base_patch16_dinov3.lvd1689m"
)

INPUT_SIZE = 512
GRID = 32
DIM = 768

RETAIN = 0.70
COARSE = 8


if not torch.cuda.is_available():
    raise RuntimeError("CUDA unavailable")

device = torch.device("cuda")


# ------------------------------------------------------------
# metadata
# ------------------------------------------------------------

with PUBLIC_META.open() as f:
    rows = list(csv.DictReader(f))

for r in rows:
    r["label"] = int(r["label"])


good = [
    r for r in rows
    if (
        r["label"] == 0
        and r["condition"] in ["regular"] + SHIFTS
    )
]


COND_RE = re.compile(
    r"_(regular|shift_[1-4]|overexposed|underexposed)$"
)


def get_iid(row):

    if row.get("instance_id", ""):
        return row["instance_id"]

    return COND_RE.sub(
        "",
        Path(row["image_path"]).stem,
    )


groups = defaultdict(dict)

for r in good:
    groups[
        (
            r["category"],
            get_iid(r),
        )
    ][
        r["condition"]
    ] = r


# ------------------------------------------------------------
# backbone
# ------------------------------------------------------------

model = timm.create_model(
    MODEL_NAME,
    pretrained=True,
    num_classes=0,
    img_size=INPUT_SIZE,
).eval().to(device)

for p in model.parameters():
    p.requires_grad_(False)


cfg = resolve_model_data_config(model)

mean = torch.tensor(
    cfg["mean"],
    device=device,
).view(1, 3, 1, 1)

std = torch.tensor(
    cfg["std"],
    device=device,
).view(1, 3, 1, 1)


def load_image(rel):

    path = DATA_ROOT / rel

    with Image.open(path) as im:

        im = im.convert("RGB")

        im = resize(
            im,
            [INPUT_SIZE, INPUT_SIZE],
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        )

        return to_tensor(im)[None]


def features(raw):

    x = raw.to(device)
    x = (x - mean) / std

    with torch.inference_mode():

        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
        ):

            out = model.forward_features(x)

    if isinstance(out, dict):

        if "x_norm_patchtokens" in out:
            p = out["x_norm_patchtokens"]

        else:
            prefix = getattr(
                model,
                "num_prefix_tokens",
                1,
            )
            p = out["x"][:, prefix:, :]

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

    assert p.shape == (1, 1024, DIM)

    return p[0]


# ------------------------------------------------------------
# exact NN
# ------------------------------------------------------------

def nearest(query, memory, chunk=4096):

    best_sim = torch.full(
        (query.shape[0],),
        -1e9,
        device=device,
    )

    best_idx = torch.zeros(
        query.shape[0],
        dtype=torch.long,
        device=device,
    )

    for j in range(
        0,
        memory.shape[0],
        chunk,
    ):

        m = memory[
            j:j+chunk
        ]

        sim = query @ m.T

        val, idx = sim.max(dim=1)

        mask = val > best_sim

        best_sim[mask] = val[mask]

        best_idx[mask] = (
            idx[mask] + j
        )

    return best_sim, best_idx


# ------------------------------------------------------------
# one-pass self-calibration
# ------------------------------------------------------------

def calibrate(query, memory):

    sim, idx = nearest(
        query,
        memory,
    )

    raw_score = (
        1.0 - sim
    )

    matched = memory[idx]

    residual = (
        query - matched
    )


    threshold = torch.quantile(
        raw_score,
        RETAIN,
    )

    weight = (
        raw_score <= threshold
    ).float()


    R = (
        residual
        .reshape(
            GRID,
            GRID,
            DIM,
        )
        .permute(
            2, 0, 1
        )[None]
    )

    W = (
        weight
        .reshape(
            1, 1,
            GRID,
            GRID,
        )
    )


    kernel = (
        GRID // COARSE
    )


    num = F.avg_pool2d(
        R * W,
        kernel_size=kernel,
        stride=kernel,
    )

    den = F.avg_pool2d(
        W,
        kernel_size=kernel,
        stride=kernel,
    )


    global_residual = (
        residual
        *
        weight[:, None]
    ).sum(dim=0) / (
        weight.sum().clamp_min(1.0)
    )


    coarse = (
        num
        /
        den.clamp_min(1e-6)
    )


    empty = (
        den <= 1e-6
    ).expand_as(coarse)


    fallback = (
        global_residual
        .view(
            1,
            DIM,
            1,
            1,
        )
        .expand_as(coarse)
    )


    coarse = torch.where(
        empty,
        fallback,
        coarse,
    )


    field = F.interpolate(
        coarse,
        size=(GRID, GRID),
        mode="bilinear",
        align_corners=False,
    )


    field = (
        field[0]
        .permute(
            1, 2, 0
        )
        .reshape(
            -1,
            DIM,
        )
    )


    corrected = F.normalize(
        query - field,
        dim=-1,
    )


    corrected_sim, _ = nearest(
        corrected,
        memory,
    )


    corrected_score = (
        1.0
        -
        corrected_sim
    )


    return (
        corrected,
        raw_score,
        corrected_score,
        field,
    )


def top10(x):

    k = min(
        10,
        x.numel(),
    )

    return float(
        torch.topk(
            x,
            k=k,
        ).values.mean().item()
    )


def feature_distance(a, b):

    return float(
        (
            1.0
            -
            (a * b).sum(dim=-1)
        ).mean().item()
    )


# ------------------------------------------------------------
# evaluate paired good scenes
# ------------------------------------------------------------

detail = []


print("=" * 110)
print("A11.0a — SINGLE-IMAGE SELF-CALIBRATION")
print("=" * 110)


for category in CATEGORIES:

    memory_np = np.load(
        MEM_ROOT
        / f"r512_{category}.npy"
    ).astype(
        np.float32,
        copy=False,
    )

    memory = torch.from_numpy(
        memory_np
    ).to(device)

    memory = F.normalize(
        memory,
        dim=-1,
    )


    cat_groups = [
        (key, conds)
        for key, conds in groups.items()
        if key[0] == category
    ]


    n_pairs = 0


    for (
        _,
        iid,
    ), conds in cat_groups:

        if "regular" not in conds:
            continue


        f_reg = features(
            load_image(
                conds[
                    "regular"
                ][
                    "image_path"
                ]
            )
        )


        (
            fc_reg,
            s_reg,
            sc_reg,
            field_reg,
        ) = calibrate(
            f_reg,
            memory,
        )


        for condition in SHIFTS:

            if condition not in conds:
                continue


            f_shift = features(
                load_image(
                    conds[
                        condition
                    ][
                        "image_path"
                    ]
                )
            )


            (
                fc_shift,
                s_shift,
                sc_shift,
                field_shift,
            ) = calibrate(
                f_shift,
                memory,
            )


            raw_fd = feature_distance(
                f_reg,
                f_shift,
            )


            corr_fd = feature_distance(
                fc_reg,
                fc_shift,
            )


            raw_drift = abs(
                top10(s_shift)
                -
                top10(s_reg)
            )


            corr_drift = abs(
                top10(sc_shift)
                -
                top10(sc_reg)
            )


            detail.append({
                "category":
                    category,

                "instance_id":
                    iid,

                "condition":
                    condition,

                "raw_feature_distance":
                    raw_fd,

                "corrected_feature_distance":
                    corr_fd,

                "feature_ratio":
                    corr_fd
                    /
                    max(raw_fd, 1e-12),

                "raw_score_drift":
                    raw_drift,

                "corrected_score_drift":
                    corr_drift,

                "score_ratio":
                    corr_drift
                    /
                    max(raw_drift, 1e-12),

                "regular_field_norm":
                    float(
                        field_reg.norm(
                            dim=-1
                        ).mean().item()
                    ),

                "shift_field_norm":
                    float(
                        field_shift.norm(
                            dim=-1
                        ).mean().item()
                    ),
            })


            n_pairs += 1


    print(
        f"{category:15s}: "
        f"{n_pairs:4d} pairs"
    )


    del memory
    torch.cuda.empty_cache()


# ------------------------------------------------------------
# category aggregation
# ------------------------------------------------------------

cat_rows = []


for category in CATEGORIES:

    x = [
        r for r in detail
        if r["category"] == category
    ]


    raw_fd = np.mean([
        r["raw_feature_distance"]
        for r in x
    ])

    corr_fd = np.mean([
        r["corrected_feature_distance"]
        for r in x
    ])


    raw_sd = np.mean([
        r["raw_score_drift"]
        for r in x
    ])

    corr_sd = np.mean([
        r["corrected_score_drift"]
        for r in x
    ])


    cat_rows.append({
        "category":
            category,

        "N":
            len(x),

        "raw_feature_distance":
            float(raw_fd),

        "corrected_feature_distance":
            float(corr_fd),

        "feature_ratio":
            float(
                corr_fd
                /
                max(raw_fd, 1e-12)
            ),

        "raw_score_drift":
            float(raw_sd),

        "corrected_score_drift":
            float(corr_sd),

        "score_ratio":
            float(
                corr_sd
                /
                max(raw_sd, 1e-12)
            ),
    })


macro_feature_ratio = float(
    np.mean([
        r["feature_ratio"]
        for r in cat_rows
    ])
)

macro_score_ratio = float(
    np.mean([
        r["score_ratio"]
        for r in cat_rows
    ])
)


feature_wins = sum(
    r["feature_ratio"] < 1.0
    for r in cat_rows
)

score_wins = sum(
    r["score_ratio"] < 1.0
    for r in cat_rows
)


signal_feature = (
    macro_feature_ratio <= 0.85
)

signal_score = (
    macro_score_ratio <= 0.80
)

signal_feature_cat = (
    feature_wins >= 6
)

signal_score_cat = (
    score_wins >= 6
)


decision = (
    "GO"
    if (
        signal_feature
        and
        signal_score
        and
        signal_feature_cat
        and
        signal_score_cat
    )
    else
    "NO_GO"
)


# ------------------------------------------------------------
# save
# ------------------------------------------------------------

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
    writer.writerows(detail)


with CATEGORY_CSV.open(
    "w",
    newline="",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=list(
            cat_rows[0].keys()
        ),
    )

    writer.writeheader()
    writer.writerows(cat_rows)


summary = {
    "protocol": {
        "retain_fraction":
            RETAIN,

        "coarse_grid":
            COARSE,

        "public_good_only":
            True,

        "public_bad_used":
            False,

        "ground_truth_used":
            False,

        "training":
            False,

        "iterations":
            1,
    },

    "observed": {
        "macro_feature_ratio":
            macro_feature_ratio,

        "macro_score_ratio":
            macro_score_ratio,

        "feature_categories_improved":
            int(feature_wins),

        "score_categories_improved":
            int(score_wins),
    },

    "signals": {
        "feature_ratio_le_085":
            bool(signal_feature),

        "score_ratio_le_080":
            bool(signal_score),

        "feature_categories_ge_6":
            bool(signal_feature_cat),

        "score_categories_ge_6":
            bool(signal_score_cat),
    },

    "decision":
        decision,
}


with SUMMARY_JSON.open("w") as f:

    json.dump(
        summary,
        f,
        indent=2,
    )


# ------------------------------------------------------------
# print
# ------------------------------------------------------------

print()
print("=" * 110)
print("A11.0a RESULTS")
print("=" * 110)

print(
    f"{'category':15s}"
    f"{'FeatRatio':>12s}"
    f"{'ScoreRatio':>12s}"
)

print("-" * 39)


for r in cat_rows:

    print(
        f"{r['category']:15s}"
        f"{r['feature_ratio']:12.4f}"
        f"{r['score_ratio']:12.4f}"
    )


print()
print("[AGGREGATE]")

print(
    "macro feature ratio :",
    f"{macro_feature_ratio:.4f}"
)

print(
    "macro score ratio   :",
    f"{macro_score_ratio:.4f}"
)

print(
    "feature wins        :",
    f"{feature_wins}/8"
)

print(
    "score wins          :",
    f"{score_wins}/8"
)


print()
print("[PRE-REGISTERED A11.0a]")

print(
    "feature ratio <= .85 :",
    signal_feature
)

print(
    "score ratio <= .80   :",
    signal_score
)

print(
    "feature wins >= 6/8  :",
    signal_feature_cat
)

print(
    "score wins >= 6/8    :",
    signal_score_cat
)

print()
print(
    "DECISION :",
    decision
)

print()
print(
    "SUMMARY:",
    SUMMARY_JSON
)

print("=" * 110)
