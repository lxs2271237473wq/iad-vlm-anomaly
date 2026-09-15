from pathlib import Path
import csv

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


DATA_ROOT = Path(
    "/root/private_data/iad-vlm-anomaly/datasets/MVTec_AD_2"
)

A1_ROOT = Path(
    "orbitad/results/a1_anomaly_score_sensitivity"
)

OUT_ROOT = Path(
    "orbitad/results/a7_relational"
)

MEMORY_ROOT = (
    OUT_ROOT / "memory"
)

MEMORY_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)


PUBLIC_META = (
    A1_ROOT / "public_patch_maps.csv"
)

VAL_META = (
    A1_ROOT / "validation_patch_maps.csv"
)


MODEL_NAME = (
    "vit_base_patch16_dinov3.lvd1689m"
)

INPUT_SIZE = 512
GRID = 32
FEATURE_DIM = 768
REL_DIM = 24

BATCH_SIZE = 4
NUM_WORKERS = 6


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
        f"public rows={len(public_rows)}"
    )

if len(val_rows) != 302:
    raise RuntimeError(
        f"validation rows={len(val_rows)}"
    )


# ============================================================
# Backbone
# ============================================================

print("=" * 112)
print(
    "A7.1-1 — RELATIONAL NORMAL MEMORY "
    "AND ANOMALY MAP GENERATION"
)
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


    p = F.normalize(
        p.float(),
        dim=-1,
    )


    if p.shape[1:] != (
        1024,
        FEATURE_DIM,
    ):

        raise RuntimeError(
            f"feature shape={tuple(p.shape)}"
        )


    return p


# ============================================================
# EXACT A7.0 descriptor
# ============================================================

def relational_signature(
    tokens
):

    B = tokens.shape[0]


    z = tokens.reshape(
        B,
        GRID,
        GRID,
        FEATURE_DIM,
    )


    z_chw = z.permute(
        0,
        3,
        1,
        2,
    )


    padded = F.pad(
        z_chw,
        (
            2,
            2,
            2,
            2,
        ),
        mode="reflect",
    )


    windows = (
        padded
        .unfold(
            2,
            5,
            1,
        )
        .unfold(
            3,
            5,
            1,
        )
    )


    center = z_chw[
        :,
        :,
        :,
        :,
        None,
        None,
    ]


    sim = (
        windows
        *
        center
    ).sum(
        dim=1
    )


    sim = sim.reshape(
        B,
        GRID,
        GRID,
        25,
    )


    descriptor = torch.cat([
        sim[
            ...,
            :12
        ],
        sim[
            ...,
            13:
        ],
    ], dim=-1)


    mu = descriptor.mean(
        dim=-1,
        keepdim=True,
    )


    sigma = descriptor.std(
        dim=-1,
        keepdim=True,
        unbiased=False,
    ).clamp_min(
        1e-6
    )


    descriptor = (
        descriptor
        -
        mu
    ) / sigma


    return descriptor.reshape(
        B,
        1024,
        REL_DIM,
    )


# ============================================================
# Fixed 8x8 spatial sampling:
# 64 relational patches / train image.
#
# No random sampling.
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
    y * GRID + x
    for y in coords
    for x in coords
], dtype=np.int64)


if len(
    np.unique(
        sample_indices
    )
) != 64:

    raise RuntimeError(
        "sample grid error"
    )


sample_indices_t = torch.tensor(
    sample_indices,
    dtype=torch.long,
    device=device,
)


# ============================================================
# Training image discovery
# ============================================================

VALID_SUFFIXES = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
}


def train_files(
    category
):

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
            f"No training images: {root}"
        )


    return paths


# ============================================================
# Build relational memory
# ============================================================

memories = {}


print()
print("[BUILD RELATIONAL MEMORY]")


for category in CATEGORIES:

    paths = train_files(
        category
    )

    chunks = []


    for i, path in enumerate(
        paths,
        1,
    ):

        raw = load_raw(
            path
        )[None]


        tokens = extract_tokens(
            raw
        )


        relation = relational_signature(
            tokens
        )[
            0,
            sample_indices_t,
            :
        ]


        chunks.append(
            relation
            .cpu()
            .numpy()
            .astype(
                np.float16
            )
        )


        if (
            i % 50 == 0
            or
            i == len(paths)
        ):

            print(
                f"{category:15s} "
                f"{i:4d}/"
                f"{len(paths):4d}"
            )


    memory = np.concatenate(
        chunks,
        axis=0,
    )


    # With AD2 train sizes this should remain
    # below the old 32768-patch memory budget.
    if memory.shape[0] > 32768:

        # Deterministic uniform reduction.
        ids = np.linspace(
            0,
            memory.shape[0] - 1,
            32768,
        ).round().astype(
            np.int64
        )

        memory = memory[
            ids
        ]


    if memory.shape[1] != REL_DIM:

        raise RuntimeError(
            f"{category}: {memory.shape}"
        )


    np.save(
        MEMORY_ROOT
        / f"{category}_relation.npy",
        memory,
    )


    memories[
        category
    ] = torch.from_numpy(
        memory.astype(
            np.float32
        )
    ).to(
        device
    )


    print(
        f"{category:15s} "
        f"memory={memory.shape}"
    )


# ============================================================
# Relational nearest-normal score
#
# s = min_m mean_d (r - m)^2
# ============================================================

def relation_nn_score(
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
        float("inf"),
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
                    len(q)
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
                len(q)
            ] = torch.minimum(
                best[
                    q0:
                    q0
                    +
                    len(q)
                ],
                current,
            )


    return (
        best
        /
        REL_DIM
    )


# ============================================================
# Evaluation datasets
# ============================================================

class MetaDataset(Dataset):

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

        path = (
            DATA_ROOT
            / row[
                "image_path"
            ]
        )


        return (
            load_raw(
                path
            ),
            idx,
        )


def process_split(
    name,
    rows
):

    output = np.lib.format.open_memmap(
        OUT_ROOT
        / f"{name}_relation_maps.npy",

        mode="w+",

        dtype=np.float16,

        shape=(
            len(rows),
            GRID,
            GRID,
        ),
    )


    print()
    print("=" * 112)
    print(
        "GENERATE:",
        name
    )
    print("=" * 112)


    for category in CATEGORIES:

        ids = [
            i
            for i, r
            in enumerate(
                rows
            )
            if r[
                "category"
            ] == category
        ]


        cat_rows = [
            rows[i]
            for i in ids
        ]


        loader = DataLoader(
            MetaDataset(
                cat_rows
            ),

            batch_size=BATCH_SIZE,

            shuffle=False,

            num_workers=NUM_WORKERS,

            pin_memory=True,

            persistent_workers=(
                NUM_WORKERS > 0
            ),
        )


        memory = memories[
            category
        ]


        seen = 0


        for raw, local_ids in loader:

            tokens = extract_tokens(
                raw
            )


            relation = relational_signature(
                tokens
            )


            for b in range(
                relation.shape[0]
            ):

                score = relation_nn_score(
                    relation[
                        b
                    ],
                    memory,
                )


                local_idx = int(
                    local_ids[
                        b
                    ]
                )


                global_idx = ids[
                    local_idx
                ]


                output[
                    global_idx
                ] = (
                    score
                    .reshape(
                        GRID,
                        GRID,
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


    output.flush()


process_split(
    "public",
    public_rows,
)

process_split(
    "validation",
    val_rows,
)


# ============================================================
# Audit
# ============================================================

print()
print("=" * 112)
print("A7.1-1 OUTPUT AUDIT")
print("=" * 112)


for split_name, expected_n in [
    (
        "public",
        1084,
    ),
    (
        "validation",
        302,
    ),
]:

    path = (
        OUT_ROOT
        / f"{split_name}_relation_maps.npy"
    )


    x = np.load(
        path,
        mmap_mode="r",
    )


    expected = (
        expected_n,
        GRID,
        GRID,
    )


    if x.shape != expected:

        raise RuntimeError(
            f"{path}: "
            f"{x.shape} != {expected}"
        )


    if not np.isfinite(
        x
    ).all():

        raise RuntimeError(
            f"Non-finite values: "
            f"{path}"
        )


    print(
        f"{split_name:10s}: "
        f"{x.shape}"
    )


print()
print("RELATIONAL MEMORY")

for category in CATEGORIES:

    x = np.load(
        MEMORY_ROOT
        / f"{category}_relation.npy",
        mmap_mode="r",
    )

    print(
        f"{category:15s}: "
        f"{x.shape}"
    )


print()
print("STATUS: PASS")
print("=" * 112)
