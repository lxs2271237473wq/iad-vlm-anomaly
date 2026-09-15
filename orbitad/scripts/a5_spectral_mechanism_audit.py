from pathlib import Path
from collections import defaultdict
import csv
import json

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms import InterpolationMode

import timm
from timm.data import resolve_model_data_config


DATA_ROOT = Path(
    "/root/private_data/iad-vlm-anomaly/datasets/MVTec_AD_2"
)

A1_ROOT = Path(
    "orbitad/results/a1_anomaly_score_sensitivity"
)

A2_ROOT = Path(
    "orbitad/results/a2_counterfactual_persistence"
)

ANT_ROOT = Path(
    "orbitad/results/a5_tangent_alignment"
)

PCA_ROOT = Path(
    "orbitad/results/a5_tangent_specificity"
)

OUT_ROOT = Path(
    "orbitad/results/a5_spectral_audit"
)

OUT_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)

MEMORY_ROOT = (
    A1_ROOT / "memory"
)

PUBLIC_META = (
    A2_ROOT / "public_orbit_meta.csv"
)

DETAIL_CSV = (
    OUT_ROOT / "pair_alignment.csv"
)

CATEGORY_CSV = (
    OUT_ROOT / "category_spectral_audit.csv"
)

SUMMARY_JSON = (
    OUT_ROOT / "a5_3_summary.json"
)


MODEL_NAME = (
    "vit_base_patch16_dinov3.lvd1689m"
)

INPUT_SIZE = 512
FEATURE_DIM = 768
RANK = 16

PRIMARY_CONDITIONS = [
    "shift_1",
    "shift_2",
    "shift_3",
]


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


WRONG_CATEGORY = {
    "can": "fabric",
    "fabric": "fruit_jelly",
    "fruit_jelly": "rice",
    "rice": "sheet_metal",
    "sheet_metal": "vial",
    "vial": "wallplugs",
    "wallplugs": "walnuts",
    "walnuts": "can",
}


if not torch.cuda.is_available():
    raise RuntimeError("CUDA unavailable")

device = torch.device("cuda")


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


normal_rows = [
    r
    for r in rows
    if r["label"] == 0
]


pairs = defaultdict(dict)

for r in normal_rows:

    key = (
        r["category"],
        r["instance_id"],
    )

    pairs[key][
        r["condition"]
    ] = r


# ============================================================
# Model
# ============================================================

model = timm.create_model(
    MODEL_NAME,
    pretrained=True,
    num_classes=0,
    img_size=INPUT_SIZE,
).eval().to(device)


cfg = resolve_model_data_config(
    model
)


transform = transforms.Compose([
    transforms.Resize(
        (
            INPUT_SIZE,
            INPUT_SIZE,
        ),
        interpolation=InterpolationMode.BICUBIC,
        antialias=True,
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=cfg["mean"],
        std=cfg["std"],
    ),
])


def load_image(rel_path):

    path = (
        DATA_ROOT
        / rel_path
    )

    with Image.open(path) as im:

        return transform(
            im.convert("RGB")
        )


def extract_tokens(batch):

    batch = batch.to(
        device
    )

    with torch.inference_mode():

        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
        ):

            out = model.forward_features(
                batch
            )


    if isinstance(out, dict):

        if "x_norm_patchtokens" in out:

            p = out[
                "x_norm_patchtokens"
            ]

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
                str(
                    list(out.keys())
                )
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


    return F.normalize(
        p.float(),
        dim=-1,
    )


# ============================================================
# Load subspaces
# ============================================================

ANT = {}
PCA = {}


for cat in CATEGORIES:

    ANT[cat] = torch.from_numpy(
        np.load(
            ANT_ROOT
            / f"{cat}_tangent_rank16.npy"
        )
    ).to(
        device=device,
        dtype=torch.float32,
    )


    PCA[cat] = torch.from_numpy(
        np.load(
            PCA_ROOT
            / f"{cat}_normal_pca_rank16.npy"
        )
    ).to(
        device=device,
        dtype=torch.float32,
    )


# ============================================================
# Static category geometry
# ============================================================

category_static = {}


for cat in CATEGORIES:

    Ua = ANT[
        cat
    ]

    Up = PCA[
        cat
    ]


    # ----------------------------------------
    # Subspace overlap
    #
    # Random rank16/rank16 expected overlap:
    # 16 / 768 = 0.020833...
    # ----------------------------------------

    cross = (
        Ua.T
        @ Up
    )

    overlap = float(
        (
            cross ** 2
        ).sum().item()
        /
        RANK
    )


    singular = torch.linalg.svdvals(
        cross
    )

    cos2 = (
        singular ** 2
    )


    # ----------------------------------------
    # Normal-memory variance captured
    # ----------------------------------------

    memory = torch.from_numpy(
        np.load(
            MEMORY_ROOT
            / f"{cat}.npy"
        ).astype(
            np.float32
        )
    ).to(
        device
    )

    memory = F.normalize(
        memory,
        dim=-1,
    )

    centered = (
        memory
        -
        memory.mean(
            dim=0,
            keepdim=True,
        )
    )

    total = float(
        (
            centered ** 2
        ).sum().item()
    )


    ant_energy = float(
        (
            centered
            @ Ua
        ).pow(2)
        .sum()
        .item()
        /
        max(
            total,
            1e-12,
        )
    )


    pca_energy = float(
        (
            centered
            @ Up
        ).pow(2)
        .sum()
        .item()
        /
        max(
            total,
            1e-12,
        )
    )


    category_static[
        cat
    ] = {
        "subspace_overlap":
            overlap,

        "overlap_vs_random":
            overlap
            /
            (
                RANK
                /
                FEATURE_DIM
            ),

        "principal_cos2_max":
            float(
                cos2.max().item()
            ),

        "principal_cos2_median":
            float(
                cos2.median().item()
            ),

        "normal_variance_ant":
            ant_energy,

        "normal_variance_pca":
            pca_energy,
    }


# ============================================================
# Real normal acquisition shift alignment
# ============================================================

def alignment(
    delta,
    U,
):

    projected_energy = (
        (
            delta
            @ U
        ) ** 2
    ).sum()

    total_energy = (
        delta ** 2
    ).sum().clamp_min(
        1e-12
    )

    return float(
        (
            projected_energy
            /
            total_energy
        ).item()
    )


detail = []


for (
    cat,
    instance_id,
), conditions in sorted(
    pairs.items()
):

    if "regular" not in conditions:
        continue


    available = [
        c
        for c in PRIMARY_CONDITIONS
        if c in conditions
    ]

    if not available:
        continue


    images = [
        load_image(
            conditions[
                "regular"
            ][
                "image_path"
            ]
        )
    ]


    for condition in available:

        images.append(
            load_image(
                conditions[
                    condition
                ][
                    "image_path"
                ]
            )
        )


    features = extract_tokens(
        torch.stack(
            images,
            dim=0,
        )
    )


    regular = features[
        0
    ]


    for j, condition in enumerate(
        available,
        start=1,
    ):

        delta = (
            features[
                j
            ]
            -
            regular
        )


        ant_alignment = alignment(
            delta,
            ANT[
                cat
            ],
        )


        pca_alignment = alignment(
            delta,
            PCA[
                cat
            ],
        )


        wrong_alignment = alignment(
            delta,
            ANT[
                WRONG_CATEGORY[
                    cat
                ]
            ],
        )


        detail.append({
            "category":
                cat,

            "instance_id":
                instance_id,

            "condition":
                condition,

            "ant_alignment":
                ant_alignment,

            "pca_alignment":
                pca_alignment,

            "wrongcat_alignment":
                wrong_alignment,

            "ant_minus_pca":
                ant_alignment
                -
                pca_alignment,

            "ant_minus_wrong":
                ant_alignment
                -
                wrong_alignment,
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


for cat in CATEGORIES:

    members = [
        r
        for r in detail
        if r[
            "category"
        ] == cat
    ]


    ant_mean = float(
        np.mean([
            r[
                "ant_alignment"
            ]
            for r in members
        ])
    )


    pca_mean = float(
        np.mean([
            r[
                "pca_alignment"
            ]
            for r in members
        ])
    )


    wrong_mean = float(
        np.mean([
            r[
                "wrongcat_alignment"
            ]
            for r in members
        ])
    )


    s = category_static[
        cat
    ]


    category_rows.append({
        "category":
            cat,

        "pairs":
            len(
                members
            ),

        "real_shift_ant":
            ant_mean,

        "real_shift_pca":
            pca_mean,

        "real_shift_wrongcat":
            wrong_mean,

        "ant_minus_pca":
            ant_mean
            -
            pca_mean,

        "ant_minus_wrongcat":
            ant_mean
            -
            wrong_mean,

        **s,
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
# Aggregate
# ============================================================

def mean_column(
    key
):

    return float(
        np.mean([
            r[key]
            for r in category_rows
        ])
    )


random_overlap = (
    RANK
    /
    FEATURE_DIM
)


payload = {
    "rank":
        RANK,

    "random_rank16_overlap_expectation":
        random_overlap,

    "aggregate": {
        "mean_subspace_overlap":
            mean_column(
                "subspace_overlap"
            ),

        "mean_overlap_vs_random":
            mean_column(
                "overlap_vs_random"
            ),

        "mean_normal_variance_ant":
            mean_column(
                "normal_variance_ant"
            ),

        "mean_normal_variance_pca":
            mean_column(
                "normal_variance_pca"
            ),

        "mean_real_shift_ant":
            mean_column(
                "real_shift_ant"
            ),

        "mean_real_shift_pca":
            mean_column(
                "real_shift_pca"
            ),

        "mean_real_shift_wrongcat":
            mean_column(
                "real_shift_wrongcat"
            ),
    },

    "public_bad_or_gt_used":
        False,
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
print("=" * 120)
print("A5.3 SPECTRAL MECHANISM AUDIT")
print("=" * 120)

print(
    f"{'category':15s}"
    f"{'ANT-shift':>11s}"
    f"{'PCA-shift':>11s}"
    f"{'Wrong':>11s}"
    f"{'Overlap':>11s}"
    f"{'xRnd':>9s}"
    f"{'Nvar-ANT':>11s}"
    f"{'Nvar-PCA':>11s}"
)

print("-" * 91)


for r in category_rows:

    print(
        f"{r['category']:15s}"
        f"{r['real_shift_ant']:11.4f}"
        f"{r['real_shift_pca']:11.4f}"
        f"{r['real_shift_wrongcat']:11.4f}"
        f"{r['subspace_overlap']:11.4f}"
        f"{r['overlap_vs_random']:9.2f}"
        f"{r['normal_variance_ant']:11.4f}"
        f"{r['normal_variance_pca']:11.4f}"
    )


a = payload[
    "aggregate"
]


print()
print("[AGGREGATE]")

print(
    "random subspace overlap expectation :",
    f"{random_overlap:.4f}"
)

print(
    "ANT/PCA mean overlap                :",
    f"{a['mean_subspace_overlap']:.4f}"
)

print(
    "overlap enrichment vs random        :",
    f"{a['mean_overlap_vs_random']:.2f}x"
)

print()

print(
    "real shift alignment — ANT          :",
    f"{a['mean_real_shift_ant']:.4f}"
)

print(
    "real shift alignment — PCA16        :",
    f"{a['mean_real_shift_pca']:.4f}"
)

print(
    "real shift alignment — wrong ANT    :",
    f"{a['mean_real_shift_wrongcat']:.4f}"
)

print()

print(
    "normal variance captured — ANT      :",
    f"{a['mean_normal_variance_ant']:.4f}"
)

print(
    "normal variance captured — PCA16    :",
    f"{a['mean_normal_variance_pca']:.4f}"
)

print()
print(
    "SUMMARY:",
    SUMMARY_JSON
)

print("=" * 120)
