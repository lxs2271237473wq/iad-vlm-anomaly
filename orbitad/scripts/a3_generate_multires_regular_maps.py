from pathlib import Path
import csv
import gc
import math

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

GOOD_MANIFEST = Path(
    "orbitad/data/manifests/test_public_good.csv"
)

BAD_MANIFEST = Path(
    "orbitad/data/manifests/test_public_bad.csv"
)

ROOT = Path(
    "orbitad/results/a3_resolution_audit"
)

MEMORY_ROOT = ROOT / "memory"

META_OUT = ROOT / "regular_public_meta.csv"

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

NUM_WORKERS = 6


if not torch.cuda.is_available():
    raise RuntimeError("CUDA unavailable")

device = torch.device("cuda")


# ============================================================
# Public REGULAR only
# ============================================================

rows = []


with GOOD_MANIFEST.open() as f:

    for r in csv.DictReader(f):

        if r["condition"] != "regular":
            continue

        rows.append({
            "set_type": "normal",
            "label": 0,
            "category": r["category"],
            "instance_id": r["instance_id"],
            "condition": r["condition"],
            "image_path": r["image_path"],
            "mask_path": "",
        })


with BAD_MANIFEST.open() as f:

    for r in csv.DictReader(f):

        if r["condition"] != "regular":
            continue

        rows.append({
            "set_type": "defect",
            "label": 1,
            "category": r["category"],
            "instance_id": r["instance_id"],
            "condition": r["condition"],
            "image_path": r["image_path"],
            "mask_path": r["mask_path"],
        })


normal_count = sum(
    r["label"] == 0
    for r in rows
)

defect_count = sum(
    r["label"] == 1
    for r in rows
)


print("=" * 105)
print("OrbitAD A3.0-2 — MULTI-RESOLUTION REGULAR MAPS")
print("=" * 105)

print(
    "normal regular :",
    normal_count
)

print(
    "defect regular :",
    defect_count
)

print(
    "total          :",
    len(rows)
)


if normal_count != 64:
    raise RuntimeError(
        f"Expected 64 normal regular, got {normal_count}"
    )

if defect_count != 120:
    raise RuntimeError(
        f"Expected 120 defect regular, got {defect_count}"
    )


with META_OUT.open(
    "w",
    newline="",
) as f:

    fields = [
        "map_index",
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

    for i, r in enumerate(rows):

        writer.writerow({
            "map_index": i,
            **r,
        })


class TestDS(Dataset):

    def __init__(
        self,
        records,
        transform,
    ):

        self.records = records
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):

        path = (
            DATA_ROOT
            / self.records[idx][
                "image_path"
            ]
        )

        with Image.open(path) as im:

            x = self.transform(
                im.convert("RGB")
            )

        return x, idx


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


def nearest_scores(
    query,
    memory_gpu,
    chunk=256,
):

    outputs = []

    for start in range(
        0,
        query.shape[0],
        chunk,
    ):

        q = query[
            start:start + chunk
        ].to(
            dtype=torch.float16
        )

        similarities = (
            q
            @ memory_gpu.T
        )

        best = similarities.max(
            dim=1
        ).values

        dist = (
            1.0
            -
            best.float()
        ).clamp_min(
            0.0
        )

        outputs.append(
            dist.cpu()
        )

    return torch.cat(
        outputs,
        dim=0,
    )


for resolution in RESOLUTIONS:

    print()
    print("#" * 105)
    print("RESOLUTION:", resolution)
    print("#" * 105)

    grid = (
        resolution // 16
    )

    out_path = (
        ROOT
        / f"regular_maps_r{resolution}.npy"
    )

    store = np.lib.format.open_memmap(
        out_path,
        mode="w+",
        dtype=np.float16,
        shape=(
            len(rows),
            grid,
            grid,
        ),
    )

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

    # Conservative test batches.
    batch_size = (
        8 if resolution == 256
        else
        4 if resolution == 512
        else
        1
    )

    for category in CATEGORIES:

        global_ids = [
            i
            for i, r in enumerate(rows)
            if r["category"] == category
        ]

        records = [
            rows[i]
            for i in global_ids
        ]

        memory = np.load(
            MEMORY_ROOT
            / f"r{resolution}_{category}.npy"
        )

        if len(memory) != 32768:
            raise RuntimeError(
                f"Memory capacity error: "
                f"r{resolution}/{category} "
                f"has {len(memory)}"
            )

        memory_gpu = torch.from_numpy(
            memory.astype(
                np.float16,
                copy=False,
            )
        ).to(device)

        loader = DataLoader(
            TestDS(
                records,
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

        seen = 0

        with torch.inference_mode():

            for images, local_indices in loader:

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

                if (
                    patches.shape[1]
                    != grid * grid
                ):
                    raise RuntimeError(
                        "Patch grid mismatch."
                    )

                for b in range(
                    patches.shape[0]
                ):

                    scores = nearest_scores(
                        patches[b],
                        memory_gpu,
                    )

                    local_idx = int(
                        local_indices[b]
                    )

                    global_idx = global_ids[
                        local_idx
                    ]

                    store[
                        global_idx
                    ] = (
                        scores
                        .reshape(
                            grid,
                            grid,
                        )
                        .numpy()
                        .astype(
                            np.float16
                        )
                    )

                    seen += 1

        print(
            f"{category:15s}: "
            f"{seen:3d} images"
        )

        del memory_gpu
        torch.cuda.empty_cache()

    store.flush()

    print(
        "saved:",
        out_path
    )

    del model
    gc.collect()
    torch.cuda.empty_cache()


print()
print("=" * 105)
print("META   :", META_OUT)

for resolution in RESOLUTIONS:

    path = (
        ROOT
        / f"regular_maps_r{resolution}.npy"
    )

    x = np.load(
        path,
        mmap_mode="r",
    )

    print(
        f"r{resolution:<5d}:",
        x.shape
    )

print()
print("STATUS : PASS")
print("=" * 105)
