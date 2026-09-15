from pathlib import Path
from collections import defaultdict
import csv
import json
import math
import sys

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import (
    resize,
    to_tensor,
)
from torchvision.transforms import InterpolationMode

import timm
from timm.data import resolve_model_data_config


# ============================================================
# Paths
# ============================================================

DATA_ROOT = Path(
    "/root/private_data/iad-vlm-anomaly/datasets/MVTec_AD_2"
)

PROJECT = Path(
    "/root/private_data/iad-vlm-anomaly/orbitad"
)

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


A1_ROOT = Path(
    "orbitad/results/a1_anomaly_score_sensitivity"
)

ANT_ROOT = Path(
    "orbitad/results/a5_tangent_alignment"
)

PCA_ROOT = Path(
    "orbitad/results/a5_tangent_specificity"
)

OUT_ROOT = Path(
    "orbitad/results/a6_rns"
)

OUT_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)

MEMORY_ROOT = (
    A1_ROOT / "memory"
)

PUBLIC_META = (
    A1_ROOT / "public_patch_maps.csv"
)

VAL_META = (
    A1_ROOT / "validation_patch_maps.csv"
)

SUMMARY_JSON = (
    OUT_ROOT / "a6_0_spectrum_summary.json"
)


# ============================================================
# Fixed protocol
# ============================================================

MODEL_NAME = (
    "vit_base_patch16_dinov3.lvd1689m"
)

INPUT_SIZE = 512
FEATURE_DIM = 768

BATCH_SIZE = 2
NUM_WORKERS = 6

BETA = 1.0

# purely numerical covariance regularization
EPS_RELATIVE = 1e-3

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

            r["map_index"] = int(
                r["map_index"]
            )

            rows.append(r)

    return rows


public_rows = read_meta(
    PUBLIC_META
)

val_rows = read_meta(
    VAL_META
)


if len(public_rows) != 1084:
    raise RuntimeError(
        f"Expected 1084 public rows, "
        f"got {len(public_rows)}"
    )

if len(val_rows) != 302:
    raise RuntimeError(
        f"Expected 302 validation rows, "
        f"got {len(val_rows)}"
    )


val_by_category = defaultdict(
    list
)

for row in val_rows:

    val_by_category[
        row["category"]
    ].append(row)


# ============================================================
# DINOv3
# ============================================================

print("=" * 115)
print("A6.0-1 — RELATIVE NUISANCE SPECTRUM MAP GENERATION")
print("=" * 115)

print(
    "beta               :",
    BETA
)

print(
    "covariance epsilon :",
    EPS_RELATIVE
)


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

    with Image.open(
        path
    ) as im:

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

        return to_tensor(
            im
        )


def extract_tokens(
    raw_batch
):

    raw_batch = raw_batch.to(
        device,
        non_blocking=True,
    )

    x = (
        raw_batch
        -
        mean
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
            f"Unexpected patch shape: "
            f"{tuple(p.shape)}"
        )


    return p


# ============================================================
# Frozen synthetic acquisition orbit
#
# EXACTLY the same nuisance family used in A5.0.
# No new perturbation tuning.
# ============================================================

def synthetic_orbit(
    raw,
    seed,
):

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
# Same 8x8 = 64 spatial samples/state as A5.0
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


sample_indices_t = torch.tensor(
    sample_indices,
    dtype=torch.long,
    device=device,
)


# ============================================================
# Load PCA + ANT and construct union basis
# ============================================================

def build_union_basis(
    category
):

    pca = torch.from_numpy(
        np.load(
            PCA_ROOT
            / f"{category}_normal_pca_rank16.npy"
        )
    ).to(
        device=device,
        dtype=torch.float32,
    )


    ant = torch.from_numpy(
        np.load(
            ANT_ROOT
            / f"{category}_tangent_rank16.npy"
        )
    ).to(
        device=device,
        dtype=torch.float32,
    )


    joined = torch.cat([
        pca,
        ant,
    ], dim=1)


    U, S, _ = torch.linalg.svd(
        joined,
        full_matrices=False,
    )


    keep = (
        S > 1e-4
    )


    B = U[
        :,
        keep
    ].contiguous()


    if B.shape[1] < 16:
        raise RuntimeError(
            f"Union rank too low for {category}: "
            f"{B.shape[1]}"
        )


    return (
        B,
        S.detach()
        .cpu()
        .numpy(),
    )


# ============================================================
# Fit RNS metric
#
# C_N:
#   intrinsic normal covariance
#
# C_A:
#   synthetic acquisition displacement second moment
#
# Relative spectrum:
#
#   R = C_N^{-1/2} C_A C_N^{-1/2}
#
# NWS:
#   normal-whitened control
#
# RNS:
#   NWS + soft shrinkage
#
#   g(lambda)=1/sqrt(1 + beta*lambda)
# ============================================================

metrics = {}
summary = {}


for category in CATEGORIES:

    print()
    print("#" * 115)
    print(
        "FIT RNS:",
        category
    )
    print("#" * 115)


    B, union_singular = build_union_basis(
        category
    )

    d = B.shape[
        1
    ]


    # --------------------------------------------------------
    # Normal covariance from train/good memory
    # --------------------------------------------------------

    memory_np = np.load(
        MEMORY_ROOT
        / f"{category}.npy"
    ).astype(
        np.float32,
        copy=False,
    )


    memory = torch.from_numpy(
        memory_np
    ).to(
        device=device,
        dtype=torch.float32,
    )


    memory = F.normalize(
        memory,
        dim=-1,
    )


    memory_mean = memory.mean(
        dim=0,
        keepdim=True,
    )


    normal_coordinates = (
        (
            memory
            -
            memory_mean
        )
        @ B
    )


    Cn = (
        normal_coordinates.T
        @ normal_coordinates
    ) / max(
        normal_coordinates.shape[0]
        - 1,
        1,
    )


    mean_normal_variance = float(
        torch.trace(
            Cn
        ).item()
        /
        d
    )


    eps = (
        EPS_RELATIVE
        *
        mean_normal_variance
    )


    Cn_reg = (
        Cn
        +
        eps
        * torch.eye(
            d,
            device=device,
            dtype=torch.float32,
        )
    )


    normal_eigvals, normal_eigvecs = (
        torch.linalg.eigh(
            Cn_reg
        )
    )


    if (
        normal_eigvals.min()
        <= 0
    ):

        raise RuntimeError(
            f"Non-positive Cn for {category}"
        )


    Cn_inv_sqrt = (
        normal_eigvecs
        @ torch.diag(
            torch.rsqrt(
                normal_eigvals
            )
        )
        @ normal_eigvecs.T
    )


    # Scale keeps the whitened block roughly
    # on the same Euclidean energy scale.
    energy_scale = (
        torch.trace(
            Cn
        )
        /
        d
    )


    # --------------------------------------------------------
    # Acquisition covariance in the SAME union basis
    # --------------------------------------------------------

    Ca_sum = torch.zeros(
        (
            d,
            d,
        ),
        device=device,
        dtype=torch.float32,
    )

    delta_count = 0


    members = sorted(
        val_by_category[
            category
        ],
        key=lambda r: (
            r["image_path"],
            r["map_index"],
        )
    )


    for i, row in enumerate(
        members
    ):

        raw = load_raw(
            row[
                "image_path"
            ]
        )[None].to(
            device
        )


        orbit = synthetic_orbit(
            raw,
            SEED
            +
            row[
                "map_index"
            ]
            * 1009,
        )


        features = extract_tokens(
            orbit
        )


        regular = features[
            0
        ]


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


            delta_B = (
                delta
                @ B
            )


            Ca_sum += (
                delta_B.T
                @ delta_B
            )


            delta_count += (
                delta_B.shape[0]
            )


        if (
            (i + 1) % 10 == 0
            or
            i + 1 == len(
                members
            )
        ):

            print(
                f"covariance "
                f"{i+1:3d}/"
                f"{len(members):3d}"
            )


    Ca = (
        Ca_sum
        /
        max(
            delta_count,
            1,
        )
    )


    # --------------------------------------------------------
    # Relative nuisance spectrum
    # --------------------------------------------------------

    R = (
        Cn_inv_sqrt
        @ Ca
        @ Cn_inv_sqrt
    )


    R = (
        0.5
        *
        (
            R
            +
            R.T
        )
    )


    lambdas, Q = torch.linalg.eigh(
        R
    )


    order = torch.argsort(
        lambdas,
        descending=True,
    )


    lambdas = lambdas[
        order
    ].clamp_min(
        0.0
    )


    Q = Q[
        :,
        order
    ]


    gains = torch.rsqrt(
        1.0
        +
        BETA
        * lambdas
    )


    # --------------------------------------------------------
    # Linear transforms in B coordinates
    #
    # NWS:
    #   sqrt(scale) C_N^{-1/2}
    #
    # RNS:
    #   sqrt(scale) C_N^{-1/2} Q G
    #
    # For row vectors:
    #   z = y @ transform
    # --------------------------------------------------------

    sqrt_scale = torch.sqrt(
        energy_scale
    )


    A_nws = (
        sqrt_scale
        *
        Cn_inv_sqrt
    )


    A_rns = (
        sqrt_scale
        *
        Cn_inv_sqrt
        @ Q
        @ torch.diag(
            gains
        )
    )


    # --------------------------------------------------------
    # Save metric
    # --------------------------------------------------------

    np.savez(
        OUT_ROOT
        / f"{category}_rns_metric.npz",

        B=B.detach()
        .cpu()
        .numpy()
        .astype(
            np.float32
        ),

        A_nws=A_nws.detach()
        .cpu()
        .numpy()
        .astype(
            np.float32
        ),

        A_rns=A_rns.detach()
        .cpu()
        .numpy()
        .astype(
            np.float32
        ),

        lambdas=lambdas.detach()
        .cpu()
        .numpy()
        .astype(
            np.float32
        ),

        gains=gains.detach()
        .cpu()
        .numpy()
        .astype(
            np.float32
        ),

        Cn=Cn.detach()
        .cpu()
        .numpy()
        .astype(
            np.float32
        ),

        Ca=Ca.detach()
        .cpu()
        .numpy()
        .astype(
            np.float32
        ),
    )


    metrics[
        category
    ] = {
        "B":
            B,

        "A_nws":
            A_nws,

        "A_rns":
            A_rns,
    }


    summary[
        category
    ] = {
        "union_dimension":
            int(
                d
            ),

        "synthetic_delta_vectors":
            int(
                delta_count
            ),

        "mean_normal_variance":
            mean_normal_variance,

        "epsilon":
            float(
                eps
            ),

        "lambda_max":
            float(
                lambdas.max()
                .item()
            ),

        "lambda_median":
            float(
                lambdas.median()
                .item()
            ),

        "lambda_min":
            float(
                lambdas.min()
                .item()
            ),

        "gain_min":
            float(
                gains.min()
                .item()
            ),

        "gain_median":
            float(
                gains.median()
                .item()
            ),

        "gain_max":
            float(
                gains.max()
                .item()
            ),
    }


    print(
        "union dimension :",
        d
    )

    print(
        "lambda max/med  :",
        f"{lambdas.max().item():.4f}",
        "/",
        f"{lambdas.median().item():.4f}"
    )

    print(
        "gain min/med    :",
        f"{gains.min().item():.4f}",
        "/",
        f"{gains.median().item():.4f}"
    )


    del memory
    del memory_mean
    del normal_coordinates

    torch.cuda.empty_cache()


with SUMMARY_JSON.open(
    "w"
) as f:

    json.dump(
        {
            "beta":
                BETA,

            "epsilon_relative":
                EPS_RELATIVE,

            "categories":
                summary,
        },
        f,
        indent=2,
    )


# ============================================================
# Full-space linear metric transform
# ============================================================

def apply_metric(
    x,
    B,
    A,
):

    # x: [N,768]
    #
    # y       : coordinates inside union subspace
    # x_perp  : untouched orthogonal complement
    # z       : transformed union coordinates

    y = (
        x
        @ B
    )

    x_perp = (
        x
        -
        y
        @ B.T
    )

    z = (
        y
        @ A
    )

    return (
        x_perp
        +
        z
        @ B.T
    )


# ============================================================
# Exact Euclidean nearest-neighbour scoring
# ============================================================

def nn_scores(
    query,
    memory,
    query_chunk=256,
    memory_chunk=4096,
):

    q_norm = (
        query
        *
        query
    ).sum(
        dim=1
    )


    best = torch.full(
        (
            query.shape[0],
        ),
        float(
            "inf"
        ),
        dtype=torch.float32,
        device=device,
    )


    for m0 in range(
        0,
        memory.shape[0],
        memory_chunk,
    ):

        m = memory[
            m0:
            m0
            +
            memory_chunk
        ]


        m_norm = (
            m
            *
            m
        ).sum(
            dim=1
        )


        for q0 in range(
            0,
            query.shape[0],
            query_chunk,
        ):

            q = query[
                q0:
                q0
                +
                query_chunk
            ]


            d2 = (
                q_norm[
                    q0:
                    q0
                    +
                    len(
                        q
                    )
                ][
                    :,
                    None
                ]
                +
                m_norm[
                    None,
                    :
                ]
                -
                2.0
                *
                (
                    q
                    @ m.T
                )
            )


            d2 = d2.clamp_min(
                0.0
            )


            current = d2.min(
                dim=1
            ).values


            best[
                q0:
                q0
                +
                len(
                    q
                )
            ] = torch.minimum(
                best[
                    q0:
                    q0
                    +
                    len(
                        q
                    )
                ],
                current,
            )


    return (
        0.5
        *
        best
    )


# ============================================================
# Dataset
# ============================================================

class ImageDataset(Dataset):

    def __init__(
        self,
        rows
    ):

        self.rows = rows

    def __len__(
        self
    ):

        return len(
            self.rows
        )

    def __getitem__(
        self,
        idx
    ):

        row = self.rows[
            idx
        ]

        return (
            load_raw(
                row[
                    "image_path"
                ]
            ),
            idx,
        )


# ============================================================
# Generate NWS32 + RNS32 maps
# ============================================================

METHODS = [
    "nws32",
    "rns32",
]


def process_split(
    split_name,
    rows,
):

    print()
    print("=" * 115)
    print(
        "GENERATE:",
        split_name
    )
    print("=" * 115)


    outputs = {
        method:
            np.lib.format.open_memmap(
                OUT_ROOT
                / f"{split_name}_{method}_maps.npy",

                mode="w+",

                dtype=np.float16,

                shape=(
                    len(
                        rows
                    ),
                    32,
                    32,
                ),
            )

        for method in METHODS
    }


    for category in CATEGORIES:

        global_ids = [
            i
            for i, row
            in enumerate(
                rows
            )
            if row[
                "category"
            ] == category
        ]


        category_rows = [
            rows[i]
            for i in global_ids
        ]


        memory_np = np.load(
            MEMORY_ROOT
            / f"{category}.npy"
        ).astype(
            np.float32,
            copy=False,
        )


        memory = torch.from_numpy(
            memory_np
        ).to(
            device=device,
            dtype=torch.float32,
        )


        memory = F.normalize(
            memory,
            dim=-1,
        )


        metric = metrics[
            category
        ]

        B = metric[
            "B"
        ]

        A_nws = metric[
            "A_nws"
        ]

        A_rns = metric[
            "A_rns"
        ]


        memory_nws = apply_metric(
            memory,
            B,
            A_nws,
        )


        memory_rns = apply_metric(
            memory,
            B,
            A_rns,
        )


        loader = DataLoader(
            ImageDataset(
                category_rows
            ),

            batch_size=BATCH_SIZE,

            shuffle=False,

            num_workers=NUM_WORKERS,

            pin_memory=True,

            persistent_workers=(
                NUM_WORKERS
                >
                0
            ),
        )


        seen = 0


        for images, local_ids in loader:

            features = extract_tokens(
                images
            )


            for b in range(
                features.shape[
                    0
                ]
            ):

                query = features[
                    b
                ].float()


                q_nws = apply_metric(
                    query,
                    B,
                    A_nws,
                )


                q_rns = apply_metric(
                    query,
                    B,
                    A_rns,
                )


                score_nws = nn_scores(
                    q_nws,
                    memory_nws,
                )


                score_rns = nn_scores(
                    q_rns,
                    memory_rns,
                )


                local_idx = int(
                    local_ids[
                        b
                    ]
                )


                global_idx = global_ids[
                    local_idx
                ]


                outputs[
                    "nws32"
                ][
                    global_idx
                ] = (
                    score_nws
                    .reshape(
                        32,
                        32,
                    )
                    .cpu()
                    .numpy()
                    .astype(
                        np.float16
                    )
                )


                outputs[
                    "rns32"
                ][
                    global_idx
                ] = (
                    score_rns
                    .reshape(
                        32,
                        32,
                    )
                    .cpu()
                    .numpy()
                    .astype(
                        np.float16
                    )
                )


                seen += 1


        print(
            f"{category:15s}: "
            f"{seen:4d}"
        )


        del memory
        del memory_nws
        del memory_rns

        torch.cuda.empty_cache()


    for mmap in outputs.values():

        mmap.flush()


process_split(
    "public",
    public_rows,
)

process_split(
    "validation",
    val_rows,
)


# ============================================================
# Final integrity audit
# ============================================================

print()
print("=" * 115)
print("A6.0-1 OUTPUT AUDIT")
print("=" * 115)


for split_name, rows in [
    (
        "public",
        public_rows,
    ),
    (
        "validation",
        val_rows,
    ),
]:

    for method in METHODS:

        path = (
            OUT_ROOT
            / f"{split_name}_{method}_maps.npy"
        )


        x = np.load(
            path,
            mmap_mode="r",
        )


        expected = (
            len(
                rows
            ),
            32,
            32,
        )


        if x.shape != expected:

            raise RuntimeError(
                f"{path}: "
                f"{x.shape} "
                f"!= {expected}"
            )


        if not np.isfinite(
            x
        ).all():

            raise RuntimeError(
                f"Nonfinite map: {path}"
            )


        print(
            f"{split_name:10s} "
            f"{method:8s}: "
            f"{x.shape}"
        )


print()
print(
    "SPECTRUM SUMMARY:",
    SUMMARY_JSON
)

print()
print("STATUS: PASS")
print("=" * 115)
