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
    "orbitad/results/a5_orthogonal_scoring"
)

OUT_ROOT.mkdir(
    parents=True,
    exist_ok=True
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


MODEL_NAME = (
    "vit_base_patch16_dinov3.lvd1689m"
)

INPUT_SIZE = 512
FEATURE_DIM = 768
RANK = 16

SEED = 20260913

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


METHODS = [
    "raw",
    "ant",
    "random16",
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


# ============================================================
# Model
# ============================================================

print("=" * 110)
print(
    "A5.1-1 — GENERATE "
    "NUISANCE-ORTHOGONAL ANOMALY MAPS"
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


transform = transforms.Compose([
    transforms.Resize(
        (
            INPUT_SIZE,
            INPUT_SIZE,
        ),
        interpolation=(
            InterpolationMode.BICUBIC
        ),
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

        with Image.open(
            path
        ) as im:

            x = transform(
                im.convert(
                    "RGB"
                )
            )

        return x, idx


def extract_tokens(
    images
):

    out = model.forward_features(
        images
    )

    if isinstance(
        out,
        dict
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
            str(
                tuple(
                    p.shape
                )
            )
        )


    return p


# ============================================================
# Random control — deterministic orthonormal rank-16
# ============================================================

random_subspaces = {}


for category in CATEGORIES:

    g = torch.Generator(
        device=device
    )

    g.manual_seed(
        SEED
        +
        sum(
            ord(c)
            for c in category
        )
    )

    A = torch.randn(
        FEATURE_DIM,
        RANK,
        generator=g,
        device=device,
        dtype=torch.float32,
    )

    Q, _ = torch.linalg.qr(
        A,
        mode="reduced",
    )

    random_subspaces[
        category
    ] = Q


# ============================================================
# Projected Euclidean NN
#
# raw:
#   0.5 ||q-m||^2
#
# projected:
#   0.5 ||P_perp(q-m)||^2
#
# Since P_perp is linear:
#   project q and m first, then NN.
# ============================================================

def projected_features(
    x,
    U
):

    # x [...,768]
    #
    # P_perp x =
    # x - U(U^T x)

    return (
        x
        -
        (
            x
            @ U
        )
        @ U.T
    )


def nn_euclidean_scores(
    query,
    memory,
    query_chunk=256,
    memory_chunk=4096,
):

    # returns 0.5 * min ||q-m||^2

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
        device=device,
        dtype=torch.float32,
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


def process_split(
    name,
    rows
):

    print()
    print(
        "#" * 110
    )
    print(
        "SPLIT:",
        name
    )
    print(
        "#" * 110
    )


    output = {
        method:
            np.lib.format.open_memmap(
                OUT_ROOT
                / f"{name}_{method}_maps.npy",
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

        ids = [
            i
            for i, r in enumerate(
                rows
            )
            if r[
                "category"
            ] == category
        ]

        category_rows = [
            rows[i]
            for i in ids
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
            device
        )


        # Verify memory is normalized enough
        memory = F.normalize(
            memory.float(),
            dim=-1,
        )


        tangent_np = np.load(
            TANGENT_ROOT
            / f"{category}_tangent_rank16.npy"
        )


        tangent = torch.from_numpy(
            tangent_np
        ).to(
            device=device,
            dtype=torch.float32,
        )


        random_u = random_subspaces[
            category
        ]


        memory_ant = projected_features(
            memory,
            tangent,
        )

        memory_random = projected_features(
            memory,
            random_u,
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


                    # ----------------------------
                    # RAW
                    # ----------------------------

                    raw_scores = (
                        nn_euclidean_scores(
                            query,
                            memory,
                        )
                    )


                    # ----------------------------
                    # ANT
                    # ----------------------------

                    query_ant = (
                        projected_features(
                            query,
                            tangent,
                        )
                    )

                    ant_scores = (
                        nn_euclidean_scores(
                            query_ant,
                            memory_ant,
                        )
                    )


                    # ----------------------------
                    # Random rank16
                    # ----------------------------

                    query_random = (
                        projected_features(
                            query,
                            random_u,
                        )
                    )

                    random_scores = (
                        nn_euclidean_scores(
                            query_random,
                            memory_random,
                        )
                    )


                    local_idx = int(
                        local_ids[b]
                    )

                    global_idx = ids[
                        local_idx
                    ]


                    output["raw"][
                        global_idx
                    ] = (
                        raw_scores
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


                    output["ant"][
                        global_idx
                    ] = (
                        ant_scores
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


                    output["random16"][
                        global_idx
                    ] = (
                        random_scores
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
        del memory_ant
        del memory_random
        del tangent

        torch.cuda.empty_cache()


    for mmap in output.values():
        mmap.flush()


process_split(
    "public",
    public_rows,
)

process_split(
    "validation",
    val_rows,
)


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

        print(
            f"{split_name:10s} "
            f"{method:10s}: "
            f"{x.shape}"
        )


print()
print("STATUS: PASS")
print("=" * 110)
