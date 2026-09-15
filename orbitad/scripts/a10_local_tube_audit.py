from pathlib import Path
from collections import defaultdict
import csv
import json
import re

import cv2
import numpy as np
from PIL import Image
from scipy.stats import rankdata

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

A1_ROOT = Path(
    "orbitad/results/a1_anomaly_score_sensitivity"
)

A5_PCA_ROOT = Path(
    "orbitad/results/a5_tangent_specificity"
)

A5_SCORE_ROOT = Path(
    "orbitad/results/a5_orthogonal_scoring"
)

OUT_ROOT = Path(
    "orbitad/results/a10_local_tube"
)

OUT_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)


PUBLIC_META = (
    A1_ROOT / "public_patch_maps.csv"
)

VAL_META = (
    A1_ROOT / "validation_patch_maps.csv"
)

MEMORY_ROOT = (
    A1_ROOT / "memory"
)


MAP_PATHS = {
    "validation": {
        "raw":
            A5_SCORE_ROOT
            / "validation_raw_maps.npy",

        "pca16":
            A5_PCA_ROOT
            / "validation_normalpca16_maps.npy",
    },

    "public": {
        "raw":
            A5_SCORE_ROOT
            / "public_raw_maps.npy",

        "pca16":
            A5_PCA_ROOT
            / "public_normalpca16_maps.npy",
    },
}


PAIR_CSV = (
    OUT_ROOT / "a10_0_normal_shift_pairs.csv"
)

PATCH_CSV = (
    OUT_ROOT / "a10_0_defect_patches.csv"
)

CATEGORY_CSV = (
    OUT_ROOT / "a10_0_category_summary.csv"
)

SUMMARY_JSON = (
    OUT_ROOT / "a10_0_summary.json"
)


# ============================================================
# Fixed protocol
# ============================================================

MODEL_NAME = (
    "vit_base_patch16_dinov3.lvd1689m"
)

INPUT_SIZE = 512
GRID = 32
FEATURE_DIM = 768

K_NEIGHBORS = 16
LOCAL_RANK = 4

# Fixed 8x8 probe grid = 64 patches/image
PROBE_SIDE = 8

# Defect audit:
# at most 16 defect + 16 background patches/image
MAX_DEFECT_PATCHES = 16
MAX_BACKGROUND_PATCHES = 16

BATCH_SIZE = 4
NUM_WORKERS = 6


PRIMARY_SHIFTS = [
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

            r["label"] = int(
                r.get(
                    "label",
                    0,
                )
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
# Existing baseline maps
# ============================================================

maps = {
    split: {}
    for split in [
        "validation",
        "public",
    ]
}


for split in maps:

    expected_n = (
        302
        if split == "validation"
        else 1084
    )


    for method, path in (
        MAP_PATHS[
            split
        ].items()
    ):

        x = np.load(
            path,
            mmap_mode="r",
        )


        if x.shape != (
            expected_n,
            32,
            32,
        ):

            raise RuntimeError(
                f"{path}: {x.shape}"
            )


        maps[
            split
        ][
            method
        ] = x


# ============================================================
# Probe locations
# ============================================================

coords = np.rint(
    np.linspace(
        0,
        GRID - 1,
        PROBE_SIDE,
    )
).astype(
    np.int64
)


probe_indices = np.asarray([
    y * GRID + x
    for y in coords
    for x in coords
], dtype=np.int64)


probe_indices_t = torch.tensor(
    probe_indices,
    dtype=torch.long,
    device=device,
)


print("=" * 112)
print("A10.0 — LOCAL NORMAL TUBE MECHANISM AUDIT")
print("=" * 112)

print(
    "K neighbours :",
    K_NEIGHBORS
)

print(
    "local rank   :",
    LOCAL_RANK
)

print(
    "probe patches:",
    len(probe_indices)
)


# ============================================================
# Backbone
# ============================================================

model = timm.create_model(
    MODEL_NAME,
    pretrained=True,
    num_classes=0,
    img_size=INPUT_SIZE,
).eval().to(device)


for p in model.parameters():

    p.requires_grad_(False)


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


def load_image(
    rel_path
):

    path = (
        DATA_ROOT
        /
        rel_path
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
    raw
):

    raw = raw.to(
        device,
        non_blocking=True,
    )

    x = (
        raw
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
            f"bad token shape "
            f"{tuple(p.shape)}"
        )


    return p


# ============================================================
# Exact top-K cosine search
# ============================================================

def exact_topk(
    query,
    memory,
    k=K_NEIGHBORS,
    memory_chunk=4096,
):

    Q = query.shape[
        0
    ]


    best_val = torch.full(
        (
            Q,
            k,
        ),
        -float("inf"),
        dtype=torch.float32,
        device=device,
    )


    best_idx = torch.full(
        (
            Q,
            k,
        ),
        -1,
        dtype=torch.long,
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


        sim = (
            query
            @
            m.T
        )


        local_k = min(
            k,
            m.shape[0],
        )


        vals, ids = torch.topk(
            sim,
            k=local_k,
            dim=1,
            largest=True,
            sorted=False,
        )


        ids = (
            ids
            +
            m0
        )


        candidate_val = torch.cat([
            best_val,
            vals,
        ], dim=1)


        candidate_idx = torch.cat([
            best_idx,
            ids,
        ], dim=1)


        new_val, order = torch.topk(
            candidate_val,
            k=k,
            dim=1,
            largest=True,
            sorted=False,
        )


        new_idx = torch.gather(
            candidate_idx,
            1,
            order,
        )


        best_val = new_val
        best_idx = new_idx


    return best_idx


# ============================================================
# Local Normal Tube score
#
# Efficient small-Gram formulation:
#
# X: centered K neighbours, K x D
# G = X X^T, K x K
#
# Projection energy onto local PCA directions can be
# obtained without constructing a 768-D basis explicitly.
# ============================================================

def local_tube_score(
    query,
    memory,
    query_chunk=128,
):

    outputs = []


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


        ids = exact_topk(
            q,
            memory,
        )


        neighbours = memory[
            ids
        ]


        mu = neighbours.mean(
            dim=1
        )


        X = (
            neighbours
            -
            mu[
                :,
                None,
                :
            ]
        )


        v = (
            q
            -
            mu
        )


        G = torch.bmm(
            X,
            X.transpose(
                1,
                2,
            ),
        )


        eigenvalues, eigenvectors = (
            torch.linalg.eigh(
                G
            )
        )


        eigenvalues = eigenvalues[
            :,
            -LOCAL_RANK:
        ]


        U = eigenvectors[
            :,
            :,
            -LOCAL_RANK:
        ]


        Xv = torch.bmm(
            X,
            v[
                :,
                :,
                None,
            ],
        )[
            :,
            :,
            0
        ]


        coeff = torch.einsum(
            "bkr,bk->br",
            U,
            Xv,
        )


        valid = (
            eigenvalues
            >
            1e-8
        )


        projection_energy = torch.where(
            valid,
            coeff.pow(2)
            /
            eigenvalues.clamp_min(
                1e-8
            ),
            torch.zeros_like(
                coeff
            ),
        ).sum(
            dim=1
        )


        total_energy = (
            v.pow(2)
            .sum(
                dim=1
            )
        )


        residual = (
            total_energy
            -
            projection_energy
        ).clamp_min(
            0.0
        )


        outputs.append(
            residual
        )


    return torch.cat(
        outputs,
        dim=0,
    )


# ============================================================
# Dataset
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


        return (
            load_image(
                row[
                    "image_path"
                ]
            ),
            idx,
        )


# ============================================================
# Calibration scores
# ============================================================

METHODS = [
    "raw",
    "pca16",
    "lnt",
]


calibration = {
    category: {
        method: []
        for method in METHODS
    }
    for category in CATEGORIES
}


def flatten_map_values(
    score_map,
    indices,
):

    return (
        np.asarray(
            score_map,
            dtype=np.float32,
        )
        .reshape(
            -1
        )[
            indices
        ]
    )


print()
print("[1/3] VALIDATION CALIBRATION")


for category in CATEGORIES:

    cat_rows = [
        r
        for r in val_rows
        if r[
            "category"
        ] == category
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


    processed = 0


    for raw, local_ids in loader:

        tokens = extract_tokens(
            raw
        )


        query = tokens[
            :,
            probe_indices_t,
            :
        ].reshape(
            -1,
            FEATURE_DIM,
        )


        lnt = local_tube_score(
            query,
            memory,
        ).reshape(
            tokens.shape[0],
            -1,
        )


        for b in range(
            tokens.shape[0]
        ):

            local_idx = int(
                local_ids[b]
            )

            row = cat_rows[
                local_idx
            ]

            map_idx = row[
                "map_index"
            ]


            calibration[
                category
            ][
                "lnt"
            ].extend(
                lnt[
                    b
                ].detach()
                .cpu()
                .numpy()
                .tolist()
            )


            for method in [
                "raw",
                "pca16",
            ]:

                values = flatten_map_values(
                    maps[
                        "validation"
                    ][
                        method
                    ][
                        map_idx
                    ],
                    probe_indices,
                )


                calibration[
                    category
                ][
                    method
                ].extend(
                    values.tolist()
                )


            processed += 1


    print(
        f"{category:15s}: "
        f"{processed:4d} images"
    )


    del memory
    torch.cuda.empty_cache()


# Sort empirical calibration CDFs.

for category in CATEGORIES:

    for method in METHODS:

        x = np.asarray(
            calibration[
                category
            ][
                method
            ],
            dtype=np.float64,
        )


        if len(x) == 0:

            raise RuntimeError(
                f"empty calibration "
                f"{category} {method}"
            )


        calibration[
            category
        ][
            method
        ] = np.sort(
            x
        )


def percentile(
    category,
    method,
    values,
):

    ref = calibration[
        category
    ][
        method
    ]


    values = np.asarray(
        values,
        dtype=np.float64,
    )


    return (
        np.searchsorted(
            ref,
            values,
            side="right",
        )
        /
        len(ref)
    )


# ============================================================
# Helper: instance ID
# ============================================================

COND_RE = re.compile(
    r"_(regular|shift_[1-4]|overexposed|underexposed)$"
)


def instance_id(
    row
):

    stem = Path(
        row[
            "image_path"
        ]
    ).stem


    return COND_RE.sub(
        "",
        stem,
    )


# ============================================================
# 2/3 Normal real-shift score stability
# ============================================================

print()
print("[2/3] PUBLIC GOOD REAL-SHIFT STABILITY")


good_rows = [
    r
    for r in public_rows
    if (
        r[
            "label"
        ] == 0
        and
        r[
            "condition"
        ]
        in (
            ["regular"]
            +
            PRIMARY_SHIFTS
        )
    )
]


normal_score = {}


for category in CATEGORIES:

    cat_rows = [
        r
        for r in good_rows
        if r[
            "category"
        ] == category
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


    for raw, local_ids in loader:

        tokens = extract_tokens(
            raw
        )


        query = tokens[
            :,
            probe_indices_t,
            :
        ].reshape(
            -1,
            FEATURE_DIM,
        )


        lnt = local_tube_score(
            query,
            memory,
        ).reshape(
            tokens.shape[0],
            -1,
        )


        for b in range(
            tokens.shape[0]
        ):

            local_idx = int(
                local_ids[b]
            )

            row = cat_rows[
                local_idx
            ]

            map_idx = row[
                "map_index"
            ]

            iid = instance_id(
                row
            )

            condition = row[
                "condition"
            ]


            lnt_p = percentile(
                category,
                "lnt",
                lnt[
                    b
                ]
                .detach()
                .cpu()
                .numpy(),
            )


            normal_score[
                (
                    "lnt",
                    category,
                    iid,
                    condition,
                )
            ] = float(
                np.mean(
                    lnt_p
                )
            )


            for method in [
                "raw",
                "pca16",
            ]:

                values = flatten_map_values(
                    maps[
                        "public"
                    ][
                        method
                    ][
                        map_idx
                    ],
                    probe_indices,
                )


                p = percentile(
                    category,
                    method,
                    values,
                )


                normal_score[
                    (
                        method,
                        category,
                        iid,
                        condition,
                    )
                ] = float(
                    np.mean(
                        p
                    )
                )


    print(
        f"{category:15s}: "
        f"{len(cat_rows):4d} good images"
    )


    del memory
    torch.cuda.empty_cache()


pair_rows = []


for method in METHODS:

    for category in CATEGORIES:

        instance_ids = sorted({
            key[2]
            for key in normal_score
            if (
                key[0] == method
                and
                key[1] == category
            )
        })


        for iid in instance_ids:

            regular_key = (
                method,
                category,
                iid,
                "regular",
            )


            if regular_key not in normal_score:
                continue


            regular_value = normal_score[
                regular_key
            ]


            for condition in PRIMARY_SHIFTS:

                shift_key = (
                    method,
                    category,
                    iid,
                    condition,
                )


                if shift_key not in normal_score:
                    continue


                shifted_value = normal_score[
                    shift_key
                ]


                pair_rows.append({
                    "method":
                        method,

                    "category":
                        category,

                    "instance_id":
                        iid,

                    "condition":
                        condition,

                    "regular_mean_percentile":
                        regular_value,

                    "shift_mean_percentile":
                        shifted_value,

                    "absolute_score_drift":
                        abs(
                            shifted_value
                            -
                            regular_value
                        ),
                })


with PAIR_CSV.open(
    "w",
    newline="",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=list(
            pair_rows[0].keys()
        ),
    )

    writer.writeheader()

    writer.writerows(
        pair_rows
    )


# ============================================================
# 3/3 Defect-vs-background sampled discrimination
# ============================================================

print()
print("[3/3] PUBLIC BAD DEFECT PRESERVATION")


bad_rows = [
    r
    for r in public_rows
    if (
        r[
            "label"
        ] == 1
        and
        r[
            "condition"
        ]
        in (
            ["regular"]
            +
            PRIMARY_SHIFTS
        )
    )
]


def get_mask_path(
    row
):

    value = row.get(
        "mask_path",
        "",
    )


    if value:

        path = Path(
            value
        )

        if not path.is_absolute():

            path = (
                DATA_ROOT
                /
                path
            )


        if path.exists():

            return path


    image_path = Path(
        row[
            "image_path"
        ]
    )


    candidate = (
        DATA_ROOT
        /
        row[
            "category"
        ]
        /
        "test_public"
        /
        "ground_truth"
        /
        "bad"
        /
        (
            image_path.stem
            +
            "_mask.png"
        )
    )


    if not candidate.exists():

        raise RuntimeError(
            f"GT missing: {candidate}"
        )


    return candidate


def mask_to_patch_fraction(
    path
):

    mask = cv2.imread(
        str(path),
        cv2.IMREAD_GRAYSCALE,
    )


    if mask is None:

        raise RuntimeError(
            f"cannot read {path}"
        )


    mask = (
        mask > 0
    ).astype(
        np.float32
    )


    # INTER_AREA preserves tiny positive occupancy.
    small = cv2.resize(
        mask,
        (
            GRID,
            GRID,
        ),
        interpolation=cv2.INTER_AREA,
    )


    return small.reshape(
        -1
    )


def uniform_take(
    ids,
    n,
):

    ids = np.asarray(
        ids,
        dtype=np.int64,
    )


    if len(ids) <= n:

        return ids


    pos = np.linspace(
        0,
        len(ids) - 1,
        n,
    ).round().astype(
        np.int64
    )


    return ids[
        pos
    ]


patch_rows = []


for category in CATEGORIES:

    cat_rows = [
        r
        for r in bad_rows
        if r[
            "category"
        ] == category
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


    processed = 0


    for raw, local_ids in loader:

        tokens = extract_tokens(
            raw
        )


        for b in range(
            tokens.shape[0]
        ):

            local_idx = int(
                local_ids[b]
            )

            row = cat_rows[
                local_idx
            ]


            occupancy = mask_to_patch_fraction(
                get_mask_path(
                    row
                )
            )


            defect_ids = np.where(
                occupancy > 0
            )[0]


            background_ids = np.where(
                occupancy == 0
            )[0]


            if (
                len(defect_ids) == 0
                or
                len(background_ids) == 0
            ):

                continue


            defect_ids = uniform_take(
                defect_ids,
                MAX_DEFECT_PATCHES,
            )


            background_ids = uniform_take(
                background_ids,
                MAX_BACKGROUND_PATCHES,
            )


            selected = np.concatenate([
                defect_ids,
                background_ids,
            ])


            labels = np.concatenate([
                np.ones(
                    len(defect_ids),
                    dtype=np.int64,
                ),

                np.zeros(
                    len(background_ids),
                    dtype=np.int64,
                ),
            ])


            selected_t = torch.tensor(
                selected,
                dtype=torch.long,
                device=device,
            )


            lnt_scores = local_tube_score(
                tokens[
                    b,
                    selected_t,
                    :
                ],
                memory,
            ).detach().cpu().numpy()


            map_idx = row[
                "map_index"
            ]


            method_values = {
                "lnt":
                    lnt_scores,

                "raw":
                    flatten_map_values(
                        maps[
                            "public"
                        ][
                            "raw"
                        ][
                            map_idx
                        ],
                        selected,
                    ),

                "pca16":
                    flatten_map_values(
                        maps[
                            "public"
                        ][
                            "pca16"
                        ][
                            map_idx
                        ],
                        selected,
                    ),
            }


            for method in METHODS:

                calibrated = percentile(
                    category,
                    method,
                    method_values[
                        method
                    ],
                )


                for j in range(
                    len(selected)
                ):

                    patch_rows.append({
                        "method":
                            method,

                        "category":
                            category,

                        "condition":
                            row[
                                "condition"
                            ],

                        "image_path":
                            row[
                                "image_path"
                            ],

                        "patch_index":
                            int(
                                selected[j]
                            ),

                        "label":
                            int(
                                labels[j]
                            ),

                        "percentile_score":
                            float(
                                calibrated[j]
                            ),
                    })


            processed += 1


    print(
        f"{category:15s}: "
        f"{processed:4d} bad images"
    )


    del memory
    torch.cuda.empty_cache()


with PATCH_CSV.open(
    "w",
    newline="",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=list(
            patch_rows[0].keys()
        ),
    )

    writer.writeheader()

    writer.writerows(
        patch_rows
    )


# ============================================================
# AUC
# ============================================================

def auc_rank(
    labels,
    scores,
):

    labels = np.asarray(
        labels,
        dtype=np.int64,
    )

    scores = np.asarray(
        scores,
        dtype=np.float64,
    )


    pos = (
        labels == 1
    )

    neg = (
        labels == 0
    )


    n_pos = int(
        pos.sum()
    )

    n_neg = int(
        neg.sum()
    )


    if (
        n_pos == 0
        or
        n_neg == 0
    ):

        return np.nan


    ranks = rankdata(
        scores,
        method="average",
    )


    pos_rank_sum = float(
        ranks[
            pos
        ].sum()
    )


    return (
        pos_rank_sum
        -
        n_pos
        *
        (
            n_pos + 1
        )
        /
        2.0
    ) / (
        n_pos
        *
        n_neg
    )


# ============================================================
# Aggregate results
# ============================================================

shift_drift = {}

patch_auc = {}


for method in METHODS:

    drifts = [
        r[
            "absolute_score_drift"
        ]
        for r in pair_rows
        if r[
            "method"
        ] == method
    ]


    shift_drift[
        method
    ] = float(
        np.median(
            drifts
        )
    )


    method_patches = [
        r
        for r in patch_rows
        if r[
            "method"
        ] == method
    ]


    category_aucs = []


    for category in CATEGORIES:

        members = [
            r
            for r in method_patches
            if r[
                "category"
            ] == category
        ]


        auc = auc_rank(
            [
                r[
                    "label"
                ]
                for r in members
            ],

            [
                r[
                    "percentile_score"
                ]
                for r in members
            ],
        )


        if np.isfinite(
            auc
        ):

            category_aucs.append(
                auc
            )


    patch_auc[
        method
    ] = float(
        np.mean(
            category_aucs
        )
    )


# ============================================================
# Category table
# ============================================================

category_rows = []


for category in CATEGORIES:

    row = {
        "category":
            category,
    }


    for method in METHODS:

        drifts = [
            r[
                "absolute_score_drift"
            ]
            for r in pair_rows
            if (
                r[
                    "method"
                ] == method
                and
                r[
                    "category"
                ] == category
            )
        ]


        row[
            f"{method}_shift_drift"
        ] = float(
            np.median(
                drifts
            )
        )


        members = [
            r
            for r in patch_rows
            if (
                r[
                    "method"
                ] == method
                and
                r[
                    "category"
                ] == category
            )
        ]


        row[
            f"{method}_patch_auc"
        ] = auc_rank(
            [
                r[
                    "label"
                ]
                for r in members
            ],

            [
                r[
                    "percentile_score"
                ]
                for r in members
            ],
        )


    row[
        "lnt_shift_better"
    ] = bool(
        row[
            "lnt_shift_drift"
        ]
        <
        row[
            "pca16_shift_drift"
        ]
    )


    row[
        "lnt_auc_safe"
    ] = bool(
        row[
            "lnt_patch_auc"
        ]
        >=
        row[
            "pca16_patch_auc"
        ]
        -
        0.03
    )


    category_rows.append(
        row
    )


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
# Pre-registered GO / NO-GO
# ============================================================

pca_drift = shift_drift[
    "pca16"
]

lnt_drift = shift_drift[
    "lnt"
]


pca_auc = patch_auc[
    "pca16"
]

lnt_auc = patch_auc[
    "lnt"
]


drift_ratio = (
    lnt_drift
    /
    max(
        pca_drift,
        1e-12,
    )
)


shift_category_wins = sum(
    r[
        "lnt_shift_better"
    ]
    for r in category_rows
)


auc_safe_categories = sum(
    r[
        "lnt_auc_safe"
    ]
    for r in category_rows
)


signal_drift = (
    drift_ratio
    <=
    0.80
)


signal_auc = (
    lnt_auc
    >=
    pca_auc
    -
    0.01
)


signal_shift_categories = (
    shift_category_wins
    >=
    5
)


signal_auc_categories = (
    auc_safe_categories
    >=
    6
)


decision = (
    "GO"
    if (
        signal_drift
        and
        signal_auc
        and
        signal_shift_categories
        and
        signal_auc_categories
    )
    else "NO_GO"
)


payload = {
    "protocol": {
        "k_neighbors":
            K_NEIGHBORS,

        "local_rank":
            LOCAL_RANK,

        "validation_percentile_calibration":
            True,

        "probe_grid":
            "8x8",

        "public_gt_used_only_for_audit":
            True,

        "parameter_search":
            False,
    },

    "pre_registered": {
        "maximum_lnt_over_pca_shift_drift":
            0.80,

        "maximum_patch_auc_loss_vs_pca":
            0.01,

        "minimum_shift_categories_better":
            5,

        "minimum_auc_safe_categories":
            6,
    },

    "observed": {
        "raw_shift_drift":
            shift_drift[
                "raw"
            ],

        "pca16_shift_drift":
            pca_drift,

        "lnt_shift_drift":
            lnt_drift,

        "lnt_over_pca_shift_drift":
            drift_ratio,

        "raw_patch_auc":
            patch_auc[
                "raw"
            ],

        "pca16_patch_auc":
            pca_auc,

        "lnt_patch_auc":
            lnt_auc,

        "lnt_minus_pca_patch_auc":
            (
                lnt_auc
                -
                pca_auc
            ),

        "shift_categories_better":
            int(
                shift_category_wins
            ),

        "auc_safe_categories":
            int(
                auc_safe_categories
            ),
    },

    "signals": {
        "shift_stability":
            bool(
                signal_drift
            ),

        "defect_preservation":
            bool(
                signal_auc
            ),

        "shift_category_consistency":
            bool(
                signal_shift_categories
            ),

        "auc_category_consistency":
            bool(
                signal_auc_categories
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
print("A10.0 LOCAL NORMAL TUBE RESULTS")
print("=" * 112)

print()
print("[GLOBAL]")

print(
    f"{'method':12s}"
    f"{'shift drift':>15s}"
    f"{'patch AUC':>15s}"
)

print("-" * 42)


for method in METHODS:

    print(
        f"{method:12s}"
        f"{shift_drift[method]:15.4f}"
        f"{patch_auc[method]:15.4f}"
    )


print()
print("[CATEGORY]")

print(
    f"{'category':15s}"
    f"{'PCA-drift':>11s}"
    f"{'LNT-drift':>11s}"
    f"{'PCA-AUC':>10s}"
    f"{'LNT-AUC':>10s}"
)

print("-" * 59)


for r in category_rows:

    print(
        f"{r['category']:15s}"
        f"{r['pca16_shift_drift']:11.4f}"
        f"{r['lnt_shift_drift']:11.4f}"
        f"{r['pca16_patch_auc']:10.4f}"
        f"{r['lnt_patch_auc']:10.4f}"
    )


print()
print("[PRE-REGISTERED A10.0 GO / NO-GO]")

print(
    "LNT/PCA shift drift <= .80 :",
    signal_drift,
    f"({drift_ratio:.4f})"
)

print(
    "LNT patch AUC within -.01  :",
    signal_auc,
    f"({lnt_auc-pca_auc:+.4f})"
)

print(
    "shift better in >=5/8      :",
    signal_shift_categories,
    f"({shift_category_wins}/8)"
)

print(
    "AUC safe in >=6/8          :",
    signal_auc_categories,
    f"({auc_safe_categories}/8)"
)

print()
print(
    "DECISION :",
    decision
)

print()
print(
    "PAIR     :",
    PAIR_CSV
)

print(
    "PATCH    :",
    PATCH_CSV
)

print(
    "CATEGORY :",
    CATEGORY_CSV
)

print(
    "SUMMARY  :",
    SUMMARY_JSON
)

print("=" * 112)
