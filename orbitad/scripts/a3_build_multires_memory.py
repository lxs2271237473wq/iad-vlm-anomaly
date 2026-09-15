from pathlib import Path
import gc
import math
import random

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

OUT_ROOT = Path(
    "orbitad/results/a3_resolution_audit/memory"
)

OUT_ROOT.mkdir(
    parents=True,
    exist_ok=True
)


MODEL_NAME = "vit_base_patch16_dinov3.lvd1689m"

RESOLUTIONS = [
    256,
    512,
    1024,
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

PATCHES_PER_IMAGE = 256
MAX_PATCHES_PER_CATEGORY = 32768

SEED = 20260913
NUM_WORKERS = 8


random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


if not torch.cuda.is_available():
    raise RuntimeError("CUDA unavailable")

device = torch.device("cuda")


def train_records(category):

    root = (
        DATA_ROOT
        / category
        / "train"
        / "good"
    )

    exts = {
        ".png", ".jpg", ".jpeg",
        ".bmp", ".tif", ".tiff",
    }

    return sorted(
        p for p in root.iterdir()
        if p.is_file()
        and p.suffix.lower() in exts
    )


class TrainDS(Dataset):

    def __init__(
        self,
        paths,
        transform,
    ):
        self.paths = paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):

        with Image.open(
            self.paths[idx]
        ) as im:

            x = self.transform(
                im.convert("RGB")
            )

        return x


def sampling_indices(grid):

    # Always select 16 x 16 = 256 positions,
    # evenly distributed over the token grid.

    coords = np.rint(
        np.linspace(
            0,
            grid - 1,
            16,
        )
    ).astype(
        np.int64
    )

    indices = []

    for y in coords:
        for x in coords:

            indices.append(
                int(
                    y * grid + x
                )
            )

    indices = np.asarray(
        indices,
        dtype=np.int64,
    )

    if len(indices) != PATCHES_PER_IMAGE:
        raise RuntimeError(
            "Sampling count mismatch"
        )

    if len(np.unique(indices)) != len(indices):
        raise RuntimeError(
            f"Duplicate indices for grid={grid}"
        )

    return indices


def extract_tokens(
    model,
    images,
):

    out = model.forward_features(
        images
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

    return F.normalize(
        p.float(),
        dim=-1,
    )


print("=" * 105)
print("OrbitAD A3.0-1 — EQUAL-CAPACITY MULTI-RESOLUTION MEMORY")
print("=" * 105)


for resolution in RESOLUTIONS:

    print()
    print("#" * 105)
    print("RESOLUTION:", resolution)
    print("#" * 105)

    model = timm.create_model(
        MODEL_NAME,
        pretrained=True,
        num_classes=0,
        img_size=resolution,
    ).eval().to(device)

    cfg = resolve_model_data_config(
        model
    )

    transform = transforms.Compose([
        transforms.Resize(
            (
                resolution,
                resolution,
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

    grid = resolution // 16
    sample_idx = sampling_indices(
        grid
    )

    # Keep 1024 conservative.
    batch_size = (
        8 if resolution == 256
        else
        4 if resolution == 512
        else
        1
    )

    for category in CATEGORIES:

        paths = train_records(
            category
        )

        loader = DataLoader(
            TrainDS(
                paths,
                transform,
            ),
            batch_size=batch_size,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            persistent_workers=(
                NUM_WORKERS > 0
            ),
        )

        chunks = []

        with torch.inference_mode():

            for images in loader:

                images = images.to(
                    device,
                    non_blocking=True,
                )

                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.bfloat16,
                ):

                    patches = extract_tokens(
                        model,
                        images,
                    )

                expected = grid * grid

                if patches.shape[1] != expected:
                    raise RuntimeError(
                        f"{resolution}/{category}: "
                        f"expected {expected}, "
                        f"got {patches.shape[1]}"
                    )

                selected = patches[
                    :,
                    sample_idx,
                    :
                ]

                chunks.append(
                    selected
                    .reshape(
                        -1,
                        selected.shape[-1],
                    )
                    .cpu()
                    .numpy()
                    .astype(
                        np.float16
                    )
                )

        memory = np.concatenate(
            chunks,
            axis=0,
        )

        before_cap = len(
            memory
        )

        if (
            before_cap
            >
            MAX_PATCHES_PER_CATEGORY
        ):

            rng = np.random.default_rng(
                SEED
                + resolution
                + sum(
                    ord(c)
                    for c in category
                )
            )

            chosen = rng.choice(
                before_cap,
                size=MAX_PATCHES_PER_CATEGORY,
                replace=False,
            )

            chosen.sort()

            memory = memory[
                chosen
            ]

        out = (
            OUT_ROOT
            / f"r{resolution}_{category}.npy"
        )

        np.save(
            out,
            memory,
        )

        print(
            f"{category:15s} "
            f"train={len(paths):4d} "
            f"sampled={before_cap:7d} "
            f"memory={len(memory):6d}"
        )

    del model

    gc.collect()
    torch.cuda.empty_cache()


print()
print("=" * 105)
print("EXPECTED FINAL CAPACITY")
print("=" * 105)

for resolution in RESOLUTIONS:

    for category in CATEGORIES:

        path = (
            OUT_ROOT
            / f"r{resolution}_{category}.npy"
        )

        x = np.load(
            path,
            mmap_mode="r",
        )

        print(
            f"r{resolution:<5d} "
            f"{category:15s} "
            f"{len(x):6d}"
        )


print()
print("STATUS: PASS")
print("=" * 105)
