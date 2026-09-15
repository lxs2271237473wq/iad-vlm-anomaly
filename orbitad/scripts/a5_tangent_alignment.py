from pathlib import Path
from collections import defaultdict
import csv
import gc
import json

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
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
    "orbitad/results/a5_tangent_alignment"
)

OUT_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)


VAL_META = (
    A2_ROOT / "validation_orbit_meta.csv"
)

PUBLIC_META = (
    A2_ROOT / "public_orbit_meta.csv"
)


DETAIL_CSV = (
    OUT_ROOT /
    "real_shift_tangent_alignment.csv"
)

CATEGORY_CSV = (
    OUT_ROOT /
    "category_alignment.csv"
)

CONDITION_CSV = (
    OUT_ROOT /
    "condition_alignment.csv"
)

SUMMARY_JSON = (
    OUT_ROOT /
    "a5_0_summary.json"
)


MODEL_NAME = (
    "vit_base_patch16_dinov3.lvd1689m"
)

INPUT_SIZE = 512
FEATURE_DIM = 768

RANK = 16

SYNTHETIC_PATCHES_PER_STATE = 64

SEED = 20260913


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


torch.manual_seed(
    SEED
)

np.random.seed(
    SEED
)


# ============================================================
# Metadata
# ============================================================

def read_meta(path):

    rows = []

    with path.open() as f:

        for r in csv.DictReader(f):

            r["orbit_index"] = int(
                r["orbit_index"]
            )

            r["label"] = int(
                r["label"]
            )

            rows.append(r)

    return rows


val_rows = read_meta(
    VAL_META
)

public_rows = read_meta(
    PUBLIC_META
)


CATEGORIES = sorted({
    r["category"]
    for r in val_rows
})


# ============================================================
# Model
# ============================================================

print("=" * 110)
print(
    "A5.0 — SYNTHETIC-TO-REAL "
    "NUISANCE TANGENT ALIGNMENT"
)
print("=" * 110)


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


def load_raw(
    rel_path
):

    path = (
        DATA_ROOT
        / rel_path
    )

    with Image.open(path) as im:

        im = im.convert(
            "RGB"
        )

        im = resize(
            im,
            [
                INPUT_SIZE,
                INPUT_SIZE,
            ],
            interpolation=(
                InterpolationMode.BICUBIC
            ),
            antialias=True,
        )

        x = to_tensor(
            im
        )

    return x


def extract_tokens(
    raw_batch
):

    raw_batch = raw_batch.to(
        device,
        non_blocking=True,
    )

    x = (
        raw_batch
        - mean
    ) / std

    with torch.inference_mode():

        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
        ):

            out = model.forward_features(
                x
            )


    if isinstance(
        out,
        dict,
    ):

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
                    list(
                        out.keys()
                    )
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


    return p


# ============================================================
# 8x8 uniformly distributed spatial sample
# = 64 patch positions/state/image
# ============================================================

coords = np.rint(
    np.linspace(
        0,
        31,
        8,
    )
).astype(
    np.int64
)


sample_indices = np.asarray([
    y * 32 + x
    for y in coords
    for x in coords
], dtype=np.int64)


if len(
    np.unique(
        sample_indices
    )
) != SYNTHETIC_PATCHES_PER_STATE:

    raise RuntimeError(
        "Spatial sampling error"
    )


sample_indices_t = torch.tensor(
    sample_indices,
    dtype=torch.long,
    device=device,
)


# ============================================================
# Frozen synthetic orbit
# ============================================================

def synthetic_orbit(
    raw,
    seed,
):

    # raw: [1,3,512,512]

    generator = torch.Generator(
        device=device
    )

    generator.manual_seed(
        seed
    )


    return torch.cat([
        raw,

        directional_gradient(
            raw,
            0.30,
            45,
        ),

        directional_gradient(
            raw,
            0.30,
            135,
        ),

        spotlight(
            raw,
            0.30,
            center_x=-0.35,
            center_y=0.20,
            sigma_fraction=0.30,
        ),

        shadow(
            raw,
            0.30,
            center_x=0.35,
            center_y=-0.20,
            sigma_fraction=0.30,
        ),

        low_frequency_field(
            raw,
            0.10,
            8,
            generator=generator,
        ),
    ], dim=0)


# ============================================================
# Build category-specific nuisance tangent subspaces
#
# ONLY validation/good normal images are used.
# ============================================================

val_by_cat = defaultdict(
    list
)


for row in val_rows:

    val_by_cat[
        row["category"]
    ].append(row)


subspaces = {}

subspace_info = {}


for category in CATEGORIES:

    print()
    print(
        "=" * 110
    )
    print(
        "BUILD TANGENT:",
        category
    )
    print(
        "=" * 110
    )

    delta_chunks = []


    members = sorted(
        val_by_cat[
            category
        ],
        key=lambda r: (
            int(
                r["instance_id"]
            ),
            r["image_path"],
        )
    )


    for i, row in enumerate(
        members
    ):

        raw = load_raw(
            row["image_path"]
        )[None].to(
            device
        )


        orbit = synthetic_orbit(
            raw,
            SEED
            + row[
                "orbit_index"
            ] * 1009,
        )


        features = extract_tokens(
            orbit
        )


        regular = features[
            0
        ]


        # states 1..5
        for state in range(
            1,
            6,
        ):

            delta = (
                features[
                    state,
                    sample_indices_t,
                    :
                ]
                -
                regular[
                    sample_indices_t,
                    :
                ]
            )


            delta_chunks.append(
                delta
                .cpu()
                .numpy()
                .astype(
                    np.float16
                )
            )


        if (
            (i + 1) % 10 == 0
            or i + 1 == len(
                members
            )
        ):

            print(
                f"{i+1:3d}/"
                f"{len(members):3d}"
            )


    delta_matrix = np.concatenate(
        delta_chunks,
        axis=0,
    ).astype(
        np.float32
    )


    A = torch.from_numpy(
        delta_matrix
    ).to(
        device
    )


    total_energy = float(
        (
            A * A
        ).sum().item()
    )


    # Uncentered low-rank SVD:
    # mean nuisance direction must NOT be removed.
    U_sample, S, V = torch.pca_lowrank(
        A,
        q=RANK,
        center=False,
        niter=4,
    )


    tangent = V[
        :,
        :RANK
    ].float()


    captured = float(
        (
            S[
                :RANK
            ] ** 2
        ).sum().item()
        /
        max(
            total_energy,
            1e-12,
        )
    )


    subspaces[
        category
    ] = tangent


    subspace_info[
        category
    ] = {
        "synthetic_delta_vectors":
            int(
                len(
                    delta_matrix
                )
            ),

        "rank":
            RANK,

        "synthetic_energy_captured":
            captured,
    }


    np.save(
        OUT_ROOT
        / f"{category}_tangent_rank16.npy",
        tangent
        .cpu()
        .numpy()
        .astype(
            np.float32
        ),
    )


    print(
        "delta vectors            :",
        len(
            delta_matrix
        )
    )

    print(
        "synthetic energy captured:",
        f"{captured:.4f}"
    )


    del A
    del U_sample
    del S
    del V

    torch.cuda.empty_cache()


# ============================================================
# Construct paired REAL normal public acquisition instances
#
# No defect masks.
# No bad images.
# ============================================================

normal_public = [
    r
    for r in public_rows
    if r["label"] == 0
]


pairs = defaultdict(
    dict
)


for row in normal_public:

    key = (
        row["category"],
        row["instance_id"],
    )

    pairs[
        key
    ][
        row["condition"]
    ] = row


pair_records = []


for (
    category,
    instance_id,
), conditions in sorted(
    pairs.items()
):

    if "regular" not in conditions:
        continue


    for condition in PRIMARY_CONDITIONS:

        if condition not in conditions:
            continue


        pair_records.append({
            "category":
                category,

            "instance_id":
                instance_id,

            "condition":
                condition,

            "regular_path":
                conditions[
                    "regular"
                ][
                    "image_path"
                ],

            "shift_path":
                conditions[
                    condition
                ][
                    "image_path"
                ],
        })


print()
print("=" * 110)
print(
    "REAL NORMAL ACQUISITION PAIRS:",
    len(
        pair_records
    )
)
print("=" * 110)


# ============================================================
# Alignment
# ============================================================

RANDOM_EXPECTATION = (
    RANK
    /
    FEATURE_DIM
)


results = []


for n, pair in enumerate(
    pair_records,
    1,
):

    category = pair[
        "category"
    ]


    regular_raw = load_raw(
        pair[
            "regular_path"
        ]
    )


    shift_raw = load_raw(
        pair[
            "shift_path"
        ]
    )


    raw_batch = torch.stack([
        regular_raw,
        shift_raw,
    ], dim=0)


    features = extract_tokens(
        raw_batch
    )


    delta = (
        features[1]
        -
        features[0]
    )


    tangent = subspaces[
        category
    ]


    coeff = (
        delta
        @ tangent
    )


    tangent_energy = (
        coeff ** 2
    ).sum(
        dim=1
    )


    total_energy = (
        delta ** 2
    ).sum(
        dim=1
    ).clamp_min(
        1e-12
    )


    patch_ratio = (
        tangent_energy
        /
        total_energy
    )


    # Energy-weighted pair-level alignment.
    weighted_ratio = float(
        tangent_energy.sum().item()
        /
        total_energy.sum().item()
    )


    median_patch_ratio = float(
        patch_ratio
        .median()
        .item()
    )


    p90_patch_ratio = float(
        torch.quantile(
            patch_ratio,
            0.90,
        ).item()
    )


    results.append({
        "category":
            category,

        "instance_id":
            pair[
                "instance_id"
            ],

        "condition":
            pair[
                "condition"
            ],

        "alignment_energy":
            weighted_ratio,

        "median_patch_alignment":
            median_patch_ratio,

        "p90_patch_alignment":
            p90_patch_ratio,

        "random_rank16_expectation":
            RANDOM_EXPECTATION,

        "enrichment_vs_random":
            weighted_ratio
            /
            RANDOM_EXPECTATION,
    })


    if (
        n % 20 == 0
        or n == len(
            pair_records
        )
    ):

        print(
            f"{n:3d}/"
            f"{len(pair_records):3d}"
        )


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
            results[0].keys()
        ),
    )

    writer.writeheader()
    writer.writerows(
        results
    )


# ============================================================
# Aggregates
# ============================================================

category_rows = []


for category in CATEGORIES:

    members = [
        r
        for r in results
        if r[
            "category"
        ] == category
    ]


    values = np.asarray(
        [
            r[
                "alignment_energy"
            ]
            for r in members
        ],
        dtype=np.float64,
    )


    if len(values) == 0:
        continue


    mean_alignment = float(
        values.mean()
    )


    category_rows.append({
        "category":
            category,

        "pairs":
            len(values),

        "mean_alignment":
            mean_alignment,

        "median_alignment":
            float(
                np.median(
                    values
                )
            ),

        "enrichment_vs_random":
            mean_alignment
            /
            RANDOM_EXPECTATION,

        "alignment_ge_0_06":
            bool(
                mean_alignment
                >= 0.06
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


condition_rows = []


for condition in PRIMARY_CONDITIONS:

    members = [
        r
        for r in results
        if r[
            "condition"
        ] == condition
    ]


    if not members:
        continue


    values = np.asarray(
        [
            r[
                "alignment_energy"
            ]
            for r in members
        ],
        dtype=np.float64,
    )


    condition_rows.append({
        "condition":
            condition,

        "pairs":
            len(values),

        "mean_alignment":
            float(
                values.mean()
            ),

        "median_alignment":
            float(
                np.median(
                    values
                )
            ),
    })


with CONDITION_CSV.open(
    "w",
    newline="",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=list(
            condition_rows[0].keys()
        ),
    )

    writer.writeheader()
    writer.writerows(
        condition_rows
    )


all_alignment = np.asarray(
    [
        r[
            "alignment_energy"
        ]
        for r in results
    ],
    dtype=np.float64,
)


mean_alignment = float(
    all_alignment.mean()
)


enrichment = (
    mean_alignment
    /
    RANDOM_EXPECTATION
)


category_support = sum(
    r[
        "mean_alignment"
    ] >= 0.06
    for r in category_rows
)


signal_absolute = (
    mean_alignment
    >= 0.10
)


signal_enrichment = (
    enrichment
    >= 3.0
)


signal_categories = (
    category_support
    >= 6
)


decision = (
    "GO"
    if (
        signal_absolute
        and
        signal_enrichment
        and
        signal_categories
    )
    else "NO_GO"
)


payload = {
    "method":
        "Acquisition Nuisance Tangent",

    "rank":
        RANK,

    "random_subspace_expectation":
        RANDOM_EXPECTATION,

    "synthetic_source":
        "validation/good only",

    "real_alignment_source":
        "test_public/good paired regular vs shift_1/2/3",

    "public_bad_or_masks_used":
        False,

    "pre_registered": {
        "minimum_mean_alignment":
            0.10,

        "minimum_enrichment_vs_random":
            3.0,

        "minimum_categories_alignment_ge_0_06":
            6,
    },

    "observed": {
        "mean_alignment":
            mean_alignment,

        "median_pair_alignment":
            float(
                np.median(
                    all_alignment
                )
            ),

        "enrichment_vs_random":
            enrichment,

        "categories_alignment_ge_0_06":
            int(
                category_support
            ),
    },

    "signals": {
        "absolute_alignment":
            bool(
                signal_absolute
            ),

        "random_enrichment":
            bool(
                signal_enrichment
            ),

        "category_support":
            bool(
                signal_categories
            ),
    },

    "subspace_info":
        subspace_info,

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
print("=" * 110)
print("A5.0 NUISANCE TANGENT ALIGNMENT RESULTS")
print("=" * 110)

print(
    f"{'category':15s}"
    f"{'pairs':>8s}"
    f"{'alignment':>12s}"
    f"{'x-random':>12s}"
    f"{'>=0.06':>10s}"
)

print("-" * 57)


for r in category_rows:

    print(
        f"{r['category']:15s}"
        f"{r['pairs']:8d}"
        f"{r['mean_alignment']:12.4f}"
        f"{r['enrichment_vs_random']:12.2f}"
        f"{str(r['alignment_ge_0_06']):>10s}"
    )


print()
print("[CONDITION]")

for r in condition_rows:

    print(
        f"{r['condition']:10s} "
        f"N={r['pairs']:3d} "
        f"mean={r['mean_alignment']:.4f} "
        f"median={r['median_alignment']:.4f}"
    )


print()
print("[AGGREGATE]")

print(
    "random rank-16 expectation :",
    f"{RANDOM_EXPECTATION:.4f}"
)

print(
    "mean real-shift alignment  :",
    f"{mean_alignment:.4f}"
)

print(
    "enrichment vs random       :",
    f"{enrichment:.2f}x"
)

print(
    "categories >= 0.06         :",
    f"{category_support}/"
    f"{len(category_rows)}"
)


print()
print("[PRE-REGISTERED A5.0 GO / NO-GO]")

print(
    "mean alignment >= 0.10 :",
    signal_absolute
)

print(
    "enrichment >= 3x       :",
    signal_enrichment
)

print(
    ">=6 categories         :",
    signal_categories
)

print()
print(
    "DECISION :",
    decision
)

print()
print(
    "DETAIL    :",
    DETAIL_CSV
)

print(
    "CATEGORY  :",
    CATEGORY_CSV
)

print(
    "CONDITION :",
    CONDITION_CSV
)

print(
    "SUMMARY   :",
    SUMMARY_JSON
)

print("=" * 110)
