from pathlib import Path
import csv
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

PUBLIC_GOOD = Path(
    "orbitad/data/manifests/test_public_good.csv"
)

PUBLIC_BAD = Path(
    "orbitad/data/manifests/test_public_bad.csv"
)

ROOT = Path(
    "orbitad/results/a1_anomaly_score_sensitivity"
)

MEMORY_ROOT = ROOT / "memory"

PUBLIC_MAPS = ROOT / "public_patch_maps.npy"
PUBLIC_META = ROOT / "public_patch_maps.csv"

VAL_MAPS = ROOT / "validation_patch_maps.npy"
VAL_META = ROOT / "validation_patch_maps.csv"


MODEL_NAME = "vit_base_patch16_dinov3.lvd1689m"

INPUT_SIZE = 512
BATCH_SIZE = 4
NUM_WORKERS = 8


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


print("=" * 100)
print("OrbitAD A1.4-1 — GENERATE PIXEL DIAGNOSTIC MAPS")
print("=" * 100)


model = timm.create_model(
    MODEL_NAME,
    pretrained=True,
    num_classes=0,
    img_size=INPUT_SIZE,
).eval().to(device)


data_config = resolve_model_data_config(model)

transform = transforms.Compose([
    transforms.Resize(
        (INPUT_SIZE, INPUT_SIZE),
        interpolation=InterpolationMode.BICUBIC,
        antialias=True,
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=data_config["mean"],
        std=data_config["std"],
    ),
])


def extract_patch_tokens(images):

    out = model.forward_features(images)

    if isinstance(out, dict):

        if "x_norm_patchtokens" in out:
            patches = out["x_norm_patchtokens"]

        elif "x" in out:
            prefix = getattr(
                model,
                "num_prefix_tokens",
                1
            )
            patches = out["x"][:, prefix:, :]

        else:
            raise RuntimeError(
                f"Unsupported keys: {list(out.keys())}"
            )

    else:

        prefix = getattr(
            model,
            "num_prefix_tokens",
            1
        )

        patches = out[:, prefix:, :]

    patches = F.normalize(
        patches.float(),
        dim=-1
    )

    B, N, D = patches.shape

    grid = int(round(math.sqrt(N)))

    if (grid, N, D) != (32, 1024, 768):
        raise RuntimeError(
            f"Unexpected token shape: "
            f"{tuple(patches.shape)}"
        )

    return patches


class DS(Dataset):

    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):

        p = DATA_ROOT / self.rows[idx]["image_path"]

        with Image.open(p) as im:
            x = transform(im.convert("RGB"))

        return x, idx


def load_public():

    rows = []

    with PUBLIC_GOOD.open() as f:

        for r in csv.DictReader(f):

            rows.append({
                "set_type": "normal",
                "label": "0",
                "category": r["category"],
                "instance_id": r["instance_id"],
                "condition": r["condition"],
                "image_path": r["image_path"],
                "mask_path": "",
            })

    with PUBLIC_BAD.open() as f:

        for r in csv.DictReader(f):

            rows.append({
                "set_type": "defect",
                "label": "1",
                "category": r["category"],
                "instance_id": r["instance_id"],
                "condition": r["condition"],
                "image_path": r["image_path"],
                "mask_path": r["mask_path"],
            })

    if len(rows) != 1084:
        raise RuntimeError(
            f"Expected 1084 public images, got {len(rows)}"
        )

    return rows


def load_validation():

    rows = []

    exts = {
        ".png", ".jpg", ".jpeg",
        ".bmp", ".tif", ".tiff",
    }

    for cat in CATEGORIES:

        root = (
            DATA_ROOT
            / cat
            / "validation"
            / "good"
        )

        for p in sorted(root.iterdir()):

            if (
                p.is_file()
                and p.suffix.lower() in exts
            ):

                rows.append({
                    "set_type": "validation_normal",
                    "label": "0",
                    "category": cat,
                    "instance_id": p.stem.split("_")[0],
                    "condition": "regular",
                    "image_path": str(
                        p.relative_to(DATA_ROOT)
                    ),
                    "mask_path": "",
                })

    if len(rows) != 302:
        raise RuntimeError(
            f"Expected 302 validation images, got {len(rows)}"
        )

    return rows


def nearest_scores(query, memory, chunk=256):

    mem = torch.from_numpy(
        memory.astype(np.float16, copy=False)
    ).to(device)

    output = []

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

        sim = q @ mem.T

        best = sim.max(
            dim=1
        ).values

        scores = (
            1.0 - best.float()
        ).clamp_min(0.0)

        output.append(
            scores.cpu()
        )

    del mem

    return torch.cat(
        output
    )


def process(rows, output_maps, output_meta):

    store = np.lib.format.open_memmap(
        output_maps,
        mode="w+",
        dtype=np.float16,
        shape=(
            len(rows),
            32,
            32,
        ),
    )

    # Process category-wise so each memory bank
    # is transferred to GPU only for its own category.
    for cat in CATEGORIES:

        ids = [
            i for i, r in enumerate(rows)
            if r["category"] == cat
        ]

        cat_rows = [
            rows[i] for i in ids
        ]

        memory_path = (
            MEMORY_ROOT / f"{cat}.npy"
        )

        if not memory_path.exists():
            raise RuntimeError(
                f"Missing memory: {memory_path}"
            )

        memory = np.load(
            memory_path
        )

        loader = DataLoader(
            DS(cat_rows),
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            persistent_workers=(NUM_WORKERS > 0),
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
                    patches = extract_patch_tokens(
                        images
                    )

                for b in range(
                    patches.shape[0]
                ):

                    scores = nearest_scores(
                        patches[b],
                        memory,
                    )

                    local_idx = int(
                        local_indices[b]
                    )

                    global_idx = ids[
                        local_idx
                    ]

                    store[
                        global_idx
                    ] = (
                        scores
                        .reshape(32, 32)
                        .numpy()
                        .astype(np.float16)
                    )

                    seen += 1

        print(
            f"{cat:15s}: "
            f"{seen:4d} maps"
        )

        del memory
        torch.cuda.empty_cache()

    store.flush()

    with output_meta.open(
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
            fieldnames=fields
        )

        writer.writeheader()

        for i, r in enumerate(rows):

            writer.writerow({
                "map_index": i,
                **r,
            })


public_rows = load_public()
validation_rows = load_validation()


print()
print("[PUBLIC]")
process(
    public_rows,
    PUBLIC_MAPS,
    PUBLIC_META,
)

print()
print("[VALIDATION]")
process(
    validation_rows,
    VAL_MAPS,
    VAL_META,
)


print()
print("=" * 100)
print("PUBLIC MAPS :", PUBLIC_MAPS)
print("PUBLIC META :", PUBLIC_META)
print("VAL MAPS    :", VAL_MAPS)
print("VAL META    :", VAL_META)
print("STATUS      : PASS")
print("=" * 100)
