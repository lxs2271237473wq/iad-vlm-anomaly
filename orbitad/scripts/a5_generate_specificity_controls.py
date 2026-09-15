from pathlib import Path
import csv
import gc

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
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

TANGENT_ROOT = Path(
    "orbitad/results/a5_tangent_alignment"
)

OUT_ROOT = Path(
    "orbitad/results/a5_tangent_specificity"
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


MODEL_NAME = (
    "vit_base_patch16_dinov3.lvd1689m"
)

INPUT_SIZE = 512
FEATURE_DIM = 768
RANK = 16

NUM_WORKERS = 6
BATCH_SIZE = 2


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


# Fixed derangement.
#
# Test category memory remains unchanged.
# ONLY the nuisance tangent comes from another category.
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


METHODS = [
    "normalpca16",
    "wrongcat",
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
        f"Expected 1084 public rows, "
        f"got {len(public_rows)}"
    )

if len(val_rows) != 302:
    raise RuntimeError(
        f"Expected 302 validation rows, "
        f"got {len(val_rows)}"
    )


# ============================================================
# Model
# ============================================================

print("=" * 110)
print("A5.2-1 — GENERATE TANGENT SPECIFICITY CONTROLS")
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


class ImageDS(Dataset):

    def __init__(
        self,
        rows,
    ):
        self.rows = rows

    def __len__(self):
        return len(
            self.rows
        )

    def __getitem__(
        self,
        idx,
    ):

        row = self.rows[
            idx
        ]

        path = (
            DATA_ROOT
            / row["image_path"]
        )

        with Image.open(path) as im:

            x = transform(
                im.convert("RGB")
            )

        return x, idx


def extract_tokens(
    images,
):

    out = model.forward_features(
        images
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
# Projection
# ============================================================

def project_perp(
    x,
    U,
):

    return (
        x
        -
        (
            x
            @ U
        )
        @ U.T
    )


def nn_scores(
    query,
    memory,
    query_chunk=256,
    memory_chunk=4096,
):

    q_norm = (
        query
        * query
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
            m0 + memory_chunk
        ]

        m_norm = (
            m
            * m
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
                q0 + query_chunk
            ]

            d2 = (
                q_norm[
                    q0:
                    q0 + len(q)
                ][:, None]
                +
                m_norm[
                    None,
                    :
                ]
                -
                2.0
                * (
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
                q0 + len(q)
            ] = torch.minimum(
                best[
                    q0:
                    q0 + len(q)
                ],
                current,
            )

    return (
        0.5
        * best
    )


# ============================================================
# Exact per-category Normal PCA rank-16
#
# This is deliberately NOT based on acquisition deltas.
# It only captures the dominant normal feature variation.
# ============================================================

normal_pca = {}


print()
print("[BUILD NORMAL-PCA16 CONTROLS]")


for category in CATEGORIES:

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

    mean = memory.mean(
        dim=0,
        keepdim=True,
    )

    centered = (
        memory - mean
    )

    # Exact 768 x 768 covariance.
    covariance = (
        centered.T
        @ centered
    ) / max(
        centered.shape[0] - 1,
        1,
    )

    eigenvalues, eigenvectors = (
        torch.linalg.eigh(
            covariance
        )
    )

    U = eigenvectors[
        :,
        -RANK:
    ].flip(
        dims=[1]
    ).contiguous()

    normal_pca[
        category
    ] = U

    top_energy = float(
        eigenvalues[
            -RANK:
        ].sum().item()
    )

    total_energy = float(
        eigenvalues.sum().item()
    )

    captured = (
        top_energy
        /
        max(
            total_energy,
            1e-12,
        )
    )

    np.save(
        OUT_ROOT
        / f"{category}_normal_pca_rank16.npy",
        U.cpu()
        .numpy()
        .astype(
            np.float32
        ),
    )

    print(
        f"{category:15s} "
        f"PCA16 variance="
        f"{captured:.4f}"
    )

    del memory
    del centered
    del covariance
    del eigenvalues
    del eigenvectors

    torch.cuda.empty_cache()


# ============================================================
# Load learned acquisition tangents
# ============================================================

tangents = {}


for category in CATEGORIES:

    path = (
        TANGENT_ROOT
        / f"{category}_tangent_rank16.npy"
    )

    x = np.load(
        path
    )

    if x.shape != (
        FEATURE_DIM,
        RANK,
    ):
        raise RuntimeError(
            f"Bad tangent shape for "
            f"{category}: {x.shape}"
        )

    U = torch.from_numpy(
        x
    ).to(
        device=device,
        dtype=torch.float32,
    )

    # Sanity: tangent should already be orthonormal.
    gram = (
        U.T
        @ U
    )

    error = float(
        (
            gram
            -
            torch.eye(
                RANK,
                device=device,
            )
        ).abs()
        .max()
        .item()
    )

    if error > 1e-3:

        raise RuntimeError(
            f"Tangent not orthonormal: "
            f"{category}, err={error}"
        )

    tangents[
        category
    ] = U


# ============================================================
# Generate maps
# ============================================================

def process_split(
    split_name,
    rows,
):

    print()
    print("#" * 110)
    print(
        "SPLIT:",
        split_name
    )
    print("#" * 110)

    outputs = {
        method:
            np.lib.format.open_memmap(
                OUT_ROOT
                / f"{split_name}_{method}_maps.npy",
                mode="w+",
                dtype=np.float16,
                shape=(
                    len(rows),
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
            in enumerate(rows)
            if row[
                "category"
            ] == category
        ]

        category_rows = [
            rows[i]
            for i in global_ids
        ]


        # ----------------------------------------
        # Current-category normal memory
        # ----------------------------------------

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


        # ----------------------------------------
        # Control 1: Normal PCA16
        # ----------------------------------------

        U_pca = normal_pca[
            category
        ]

        memory_pca = project_perp(
            memory,
            U_pca,
        )


        # ----------------------------------------
        # Control 2: Wrong-category acquisition tangent
        # ----------------------------------------

        wrong_category = WRONG_CATEGORY[
            category
        ]

        U_wrong = tangents[
            wrong_category
        ]

        memory_wrong = project_perp(
            memory,
            U_wrong,
        )


        loader = DataLoader(
            ImageDS(
                category_rows
            ),
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            persistent_workers=(
                NUM_WORKERS > 0
            ),
        )


        seen = 0


        with torch.inference_mode():

            for images, local_ids in loader:

                images = images.to(
                    device,
                    non_blocking=True,
                )

                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.bfloat16,
                ):

                    features = extract_tokens(
                        images
                    )


                for b in range(
                    features.shape[0]
                ):

                    query = features[
                        b
                    ].float()


                    # Normal PCA control
                    q_pca = project_perp(
                        query,
                        U_pca,
                    )

                    score_pca = nn_scores(
                        q_pca,
                        memory_pca,
                    )


                    # Wrong-category tangent control
                    q_wrong = project_perp(
                        query,
                        U_wrong,
                    )

                    score_wrong = nn_scores(
                        q_wrong,
                        memory_wrong,
                    )


                    local_idx = int(
                        local_ids[b]
                    )

                    global_idx = global_ids[
                        local_idx
                    ]


                    outputs[
                        "normalpca16"
                    ][
                        global_idx
                    ] = (
                        score_pca
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
                        "wrongcat"
                    ][
                        global_idx
                    ] = (
                        score_wrong
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
            f"{seen:4d} "
            f"(wrong tangent="
            f"{wrong_category})"
        )


        del memory
        del memory_pca
        del memory_wrong

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
# Final audit
# ============================================================

print()
print("=" * 110)

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
            len(rows),
            32,
            32,
        )

        if x.shape != expected:

            raise RuntimeError(
                f"{path}: "
                f"{x.shape} != "
                f"{expected}"
            )

        print(
            f"{split_name:10s} "
            f"{method:12s}: "
            f"{x.shape}"
        )


print()
print("WRONG-CATEGORY MAPPING")

for category in CATEGORIES:

    print(
        f"{category:15s} -> "
        f"{WRONG_CATEGORY[category]}"
    )


print()
print("STATUS: PASS")
print("=" * 110)
