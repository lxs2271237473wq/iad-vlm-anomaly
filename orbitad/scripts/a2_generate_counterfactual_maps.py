from pathlib import Path
import sys
import csv
import math

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import (
    to_tensor,
    resize,
)
from torchvision.transforms import InterpolationMode

import timm
from timm.data import resolve_model_data_config


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


DATA_ROOT = Path(
    "/root/private_data/iad-vlm-anomaly/datasets/MVTec_AD_2"
)

A1_ROOT = Path(
    "orbitad/results/a1_anomaly_score_sensitivity"
)

MEMORY_ROOT = A1_ROOT / "memory"

PUBLIC_META_IN = A1_ROOT / "public_patch_maps.csv"
VAL_META_IN = A1_ROOT / "validation_patch_maps.csv"

OUT_ROOT = Path(
    "orbitad/results/a2_counterfactual_persistence"
)

OUT_ROOT.mkdir(
    parents=True,
    exist_ok=True
)

PUBLIC_OUT = OUT_ROOT / "public_orbit_maps.npy"
VAL_OUT = OUT_ROOT / "validation_orbit_maps.npy"

PUBLIC_META_OUT = OUT_ROOT / "public_orbit_meta.csv"
VAL_META_OUT = OUT_ROOT / "validation_orbit_meta.csv"


MODEL_NAME = "vit_base_patch16_dinov3.lvd1689m"

INPUT_SIZE = 512
NUM_WORKERS = 6
BATCH_SIZE = 2
SEED = 20260913


ORBIT_NAMES = [
    "original",
    "gradient_45",
    "gradient_135",
    "spotlight",
    "shadow",
    "lowfreq_g8_a10",
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
    raise RuntimeError("CUDA unavailable")

device = torch.device("cuda")


print("=" * 105)
print("OrbitAD A2.1-1 — GENERATE COUNTERFACTUAL ORBIT MAPS")
print("=" * 105)


model = timm.create_model(
    MODEL_NAME,
    pretrained=True,
    num_classes=0,
    img_size=INPUT_SIZE,
).eval().to(device)


cfg = resolve_model_data_config(model)

mean = torch.tensor(
    cfg["mean"],
    device=device,
).view(1, 3, 1, 1)

std = torch.tensor(
    cfg["std"],
    device=device,
).view(1, 3, 1, 1)


def extract_patches(x):

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
                str(list(out.keys()))
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
            f"Unexpected shape {tuple(p.shape)}"
        )

    return p


def read_meta(path):

    rows = []

    with path.open() as f:

        for row in csv.DictReader(f):

            row["map_index"] = int(
                row["map_index"]
            )

            rows.append(row)

    return rows


public_rows = read_meta(
    PUBLIC_META_IN
)

val_rows = read_meta(
    VAL_META_IN
)


class RawDataset(Dataset):

    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):

        path = (
            DATA_ROOT
            / self.rows[idx]["image_path"]
        )

        with Image.open(path) as im:

            im = im.convert("RGB")

            im = resize(
                im,
                [INPUT_SIZE, INPUT_SIZE],
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            )

            x = to_tensor(im)

        return x, idx


def make_orbit(
    raw,
    global_indices,
):

    raw = raw.to(
        device,
        non_blocking=True,
    )

    states = []

    states.append(raw)

    states.append(
        directional_gradient(
            raw,
            0.30,
            45,
        )
    )

    states.append(
        directional_gradient(
            raw,
            0.30,
            135,
        )
    )

    states.append(
        spotlight(
            raw,
            0.30,
            center_x=-0.35,
            center_y=0.20,
            sigma_fraction=0.30,
        )
    )

    states.append(
        shadow(
            raw,
            0.30,
            center_x=0.35,
            center_y=-0.20,
            sigma_fraction=0.30,
        )
    )

    lowfreq = []

    for b in range(
        raw.shape[0]
    ):

        generator = torch.Generator(
            device=device
        )

        generator.manual_seed(
            SEED
            + int(
                global_indices[b]
            ) * 1009
        )

        lf = low_frequency_field(
            raw[b:b+1],
            0.10,
            8,
            generator=generator,
        )

        lowfreq.append(
            lf
        )

    states.append(
        torch.cat(
            lowfreq,
            dim=0,
        )
    )

    # [K,B,C,H,W]
    orbit = torch.stack(
        states,
        dim=0,
    )

    K, B, C, H, W = orbit.shape

    orbit = orbit.reshape(
        K * B,
        C,
        H,
        W,
    )

    orbit = (
        orbit
        - mean
    ) / std

    return orbit, K, B


def score_tokens(
    tokens,
    memory_gpu,
    chunk=384,
):

    flat = tokens.reshape(
        -1,
        tokens.shape[-1],
    )

    result = []

    for start in range(
        0,
        flat.shape[0],
        chunk,
    ):

        q = flat[
            start:start + chunk
        ].to(
            dtype=torch.float16
        )

        sim = (
            q
            @ memory_gpu.T
        )

        best = sim.max(
            dim=1
        ).values

        d = (
            1.0
            - best.float()
        ).clamp_min(0)

        result.append(
            d.cpu()
        )

    return torch.cat(
        result
    )


def process(
    rows,
    output_path,
    metadata_path,
):

    K = len(
        ORBIT_NAMES
    )

    store = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=np.float16,
        shape=(
            len(rows),
            K,
            32,
            32,
        ),
    )

    for category in CATEGORIES:

        global_ids = [
            i
            for i, row in enumerate(rows)
            if row["category"] == category
        ]

        records = [
            rows[i]
            for i in global_ids
        ]

        memory = np.load(
            MEMORY_ROOT
            / f"{category}.npy"
        )

        memory_gpu = torch.from_numpy(
            memory.astype(
                np.float16,
                copy=False,
            )
        ).to(device)

        loader = DataLoader(
            RawDataset(records),
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

            for raw, local_indices in loader:

                true_global = [
                    global_ids[
                        int(x)
                    ]
                    for x in local_indices
                ]

                orbit, K_, B = make_orbit(
                    raw,
                    true_global,
                )

                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.bfloat16,
                ):

                    patches = extract_patches(
                        orbit
                    )

                scores = score_tokens(
                    patches,
                    memory_gpu,
                )

                scores = scores.reshape(
                    K_,
                    B,
                    32,
                    32,
                )

                for b in range(B):

                    global_idx = true_global[
                        b
                    ]

                    store[
                        global_idx
                    ] = (
                        scores[:, b]
                        .numpy()
                        .astype(np.float16)
                    )

                    seen += 1

                if (
                    seen % 20 < B
                    or seen == len(records)
                ):
                    print(
                        f"{category:15s} "
                        f"{seen:4d}/"
                        f"{len(records):4d}"
                    )

        del memory_gpu
        torch.cuda.empty_cache()

    store.flush()

    with metadata_path.open(
        "w",
        newline="",
    ) as f:

        fields = [
            "orbit_index",
            "set_type",
            "label",
            "category",
            "instance_id",
            "condition",
            "image_path",
            "mask_path",
        ]

        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()

        for i, row in enumerate(rows):

            writer.writerow({
                "orbit_index": i,
                "set_type":
                    row["set_type"],
                "label":
                    row["label"],
                "category":
                    row["category"],
                "instance_id":
                    row["instance_id"],
                "condition":
                    row["condition"],
                "image_path":
                    row["image_path"],
                "mask_path":
                    row["mask_path"],
            })


print()
print("[PUBLIC]")
process(
    public_rows,
    PUBLIC_OUT,
    PUBLIC_META_OUT,
)

print()
print("[VALIDATION]")
process(
    val_rows,
    VAL_OUT,
    VAL_META_OUT,
)


print()
print("=" * 105)
print("ORBIT STATES :", ORBIT_NAMES)
print("PUBLIC MAPS :", PUBLIC_OUT)
print("VAL MAPS    :", VAL_OUT)
print("STATUS      : PASS")
print("=" * 105)
