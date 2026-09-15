from pathlib import Path
from collections import defaultdict
import csv
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


# ============================================================
# Locked A1.3 configuration
# ============================================================

DATA_ROOT = Path(
    "/root/private_data/iad-vlm-anomaly/datasets/MVTec_AD_2"
)

GOOD_MANIFEST = Path(
    "orbitad/data/manifests/test_public_good.csv"
)

BAD_MANIFEST = Path(
    "orbitad/data/manifests/test_public_bad.csv"
)

OUT_ROOT = Path(
    "orbitad/results/a1_anomaly_score_sensitivity"
)

MEMORY_ROOT = OUT_ROOT / "memory"

OUT_ROOT.mkdir(parents=True, exist_ok=True)
MEMORY_ROOT.mkdir(parents=True, exist_ok=True)

SCORES_CSV = OUT_ROOT / "test_scores.csv"

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

MODEL_NAME = "vit_base_patch16_dinov3.lvd1689m"

INPUT_SIZE = 512
PATCH_SIZE = 16

PATCHES_PER_TRAIN_IMAGE = 64
MAX_PATCHES_PER_CATEGORY = 32768

TEST_BATCH_SIZE = 4
TRAIN_BATCH_SIZE = 4
NUM_WORKERS = 8

TOPK = 10
SEED = 20260913

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# Model
# ============================================================

if not torch.cuda.is_available():
    raise RuntimeError("CUDA unavailable.")

device = torch.device("cuda")

print("=" * 110)
print("OrbitAD A1.3 — DINOv3-512 NORMAL MEMORY BASELINE")
print("=" * 110)

print("Loading model:", MODEL_NAME)

model = timm.create_model(
    MODEL_NAME,
    pretrained=True,
    num_classes=0,
    img_size=INPUT_SIZE,
)

model = model.eval().to(device)

data_config = resolve_model_data_config(model)

mean = data_config["mean"]
std = data_config["std"]

# Full-frame resize rather than center crop.
# Important for AD2 because border content must not disappear.
transform = transforms.Compose([
    transforms.Resize(
        (INPUT_SIZE, INPUT_SIZE),
        interpolation=InterpolationMode.BICUBIC,
        antialias=True,
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=mean,
        std=std,
    ),
])

print("input size        :", INPUT_SIZE)
print("patch size        :", PATCH_SIZE)
print("num prefix tokens :", getattr(model, "num_prefix_tokens", None))
print("normalization mean:", mean)
print("normalization std :", std)


# ============================================================
# Patch extraction
# ============================================================

def extract_patch_tokens(images):

    out = model.forward_features(images)

    if isinstance(out, dict):

        if "x_norm_patchtokens" in out:

            patches = out["x_norm_patchtokens"]

        elif "x" in out:

            tokens = out["x"]

            prefix = getattr(
                model,
                "num_prefix_tokens",
                1,
            )

            patches = tokens[:, prefix:, :]

        else:

            raise RuntimeError(
                f"Unsupported forward_features keys: "
                f"{list(out.keys())}"
            )

    else:

        tokens = out

        if tokens.ndim != 3:
            raise RuntimeError(
                f"Expected [B,N,D], got {tuple(tokens.shape)}"
            )

        prefix = getattr(
            model,
            "num_prefix_tokens",
            1,
        )

        patches = tokens[:, prefix:, :]

    patches = F.normalize(
        patches.float(),
        dim=-1,
    )

    B, N, D = patches.shape

    grid = int(round(math.sqrt(N)))

    if grid * grid != N:
        raise RuntimeError(
            f"Patch count {N} is not square."
        )

    if grid != 32 or N != 1024 or D != 768:
        raise RuntimeError(
            f"Unexpected token shape: "
            f"B={B}, N={N}, D={D}, grid={grid}"
        )

    return patches


# ============================================================
# Dataset helpers
# ============================================================

class ImageDataset(Dataset):

    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):

        row = self.records[idx]

        path = DATA_ROOT / row["image_path"]

        with Image.open(path) as im:
            image = transform(
                im.convert("RGB")
            )

        return image, idx


def load_public_records():

    rows = []

    with GOOD_MANIFEST.open() as f:

        for row in csv.DictReader(f):

            rows.append({
                "set_type": "normal",
                "label": 0,
                "category": row["category"],
                "instance_id": row["instance_id"],
                "condition": row["condition"],
                "image_path": row["image_path"],
            })

    with BAD_MANIFEST.open() as f:

        for row in csv.DictReader(f):

            rows.append({
                "set_type": "defect",
                "label": 1,
                "category": row["category"],
                "instance_id": row["instance_id"],
                "condition": row["condition"],
                "image_path": row["image_path"],
            })

    if len(rows) != 1084:
        raise RuntimeError(
            f"Expected 1084 public images, found {len(rows)}"
        )

    return rows


def train_records(category):

    root = (
        DATA_ROOT
        / category
        / "train"
        / "good"
    )

    image_exts = {
        ".png", ".jpg", ".jpeg",
        ".bmp", ".tif", ".tiff"
    }

    paths = sorted(
        p for p in root.iterdir()
        if p.is_file()
        and p.suffix.lower() in image_exts
    )

    rows = [
        {
            "image_path": str(
                p.relative_to(DATA_ROOT)
            )
        }
        for p in paths
    ]

    return rows


# ============================================================
# Deterministic spatial sampling
# ============================================================

def fixed_grid_patch_indices():

    # 8 x 8 evenly distributed points from a 32 x 32 token grid.
    coords = np.rint(
        np.linspace(
            0,
            31,
            8,
        )
    ).astype(int)

    indices = []

    for y in coords:
        for x in coords:
            indices.append(
                int(y * 32 + x)
            )

    indices = np.asarray(
        indices,
        dtype=np.int64,
    )

    if len(indices) != PATCHES_PER_TRAIN_IMAGE:
        raise RuntimeError(
            "Patch sampling count mismatch."
        )

    if len(np.unique(indices)) != len(indices):
        raise RuntimeError(
            "Duplicate fixed patch indices."
        )

    return indices


SAMPLE_INDICES = fixed_grid_patch_indices()

print()
print(
    "memory patches / train image:",
    len(SAMPLE_INDICES),
)


# ============================================================
# Build one category memory
# ============================================================

def build_memory(category):

    records = train_records(category)

    loader = DataLoader(
        ImageDataset(records),
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=(NUM_WORKERS > 0),
    )

    chunks = []

    seen = 0

    with torch.inference_mode():

        for images, _ in loader:

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

            selected = patches[
                :,
                SAMPLE_INDICES,
                :
            ]

            selected = (
                selected
                .reshape(-1, selected.shape[-1])
                .cpu()
                .numpy()
                .astype(np.float16)
            )

            chunks.append(selected)

            seen += images.shape[0]

    memory = np.concatenate(
        chunks,
        axis=0,
    )

    if len(memory) > MAX_PATCHES_PER_CATEGORY:

        rng = np.random.default_rng(
            SEED
        )

        choose = rng.choice(
            len(memory),
            size=MAX_PATCHES_PER_CATEGORY,
            replace=False,
        )

        choose.sort()

        memory = memory[choose]

    path = MEMORY_ROOT / f"{category}.npy"

    np.save(
        path,
        memory,
    )

    print(
        f"{category:15s} | "
        f"train={seen:4d} | "
        f"memory={len(memory):6d} | "
        f"saved={path}"
    )

    return memory


# ============================================================
# Nearest-normal scoring
# ============================================================

def nearest_normal_scores(
    query,
    memory,
    query_chunk=256,
):

    # Both are normalized vectors.
    # cosine distance = 1 - max cosine similarity.

    output = []

    memory_gpu = torch.from_numpy(
        np.asarray(
            memory,
            dtype=np.float16,
        )
    ).to(device)

    for start in range(
        0,
        query.shape[0],
        query_chunk,
    ):

        q = query[
            start:start + query_chunk
        ]

        q = q.to(
            device=device,
            dtype=torch.float16,
        )

        similarity = torch.matmul(
            q,
            memory_gpu.T,
        )

        max_similarity = similarity.max(
            dim=1
        ).values

        distance = (
            1.0 - max_similarity.float()
        ).clamp_min(0.0)

        output.append(
            distance.cpu()
        )

    return torch.cat(
        output,
        dim=0,
    )


# ============================================================
# Score public dataset
# ============================================================

public_rows = load_public_records()

by_category = defaultdict(list)

for row in public_rows:
    by_category[row["category"]].append(row)


score_rows = []


for category in CATEGORIES:

    print()
    print("=" * 110)
    print("CATEGORY:", category)
    print("=" * 110)

    # --------------------------------------------------------
    # Build memory strictly from train/good regular
    # --------------------------------------------------------

    memory = build_memory(
        category
    )

    records = by_category[category]

    loader = DataLoader(
        ImageDataset(records),
        batch_size=TEST_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=(NUM_WORKERS > 0),
    )

    seen = 0

    with torch.inference_mode():

        for images, indices in loader:

            images = images.to(
                device,
                non_blocking=True,
            )

            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
            ):

                patch_features = (
                    extract_patch_tokens(
                        images
                    )
                )

            B = patch_features.shape[0]

            for b in range(B):

                patch_scores = nearest_normal_scores(
                    patch_features[b],
                    memory,
                )

                sorted_scores = torch.sort(
                    patch_scores,
                    descending=True,
                ).values

                topk = min(
                    TOPK,
                    len(sorted_scores),
                )

                image_score = float(
                    sorted_scores[:topk]
                    .mean()
                    .item()
                )

                patch_mean = float(
                    patch_scores.mean().item()
                )

                patch_p95 = float(
                    torch.quantile(
                        patch_scores,
                        0.95,
                    ).item()
                )

                patch_p99 = float(
                    torch.quantile(
                        patch_scores,
                        0.99,
                    ).item()
                )

                patch_max = float(
                    patch_scores.max().item()
                )

                local_idx = int(
                    indices[b]
                )

                row = records[
                    local_idx
                ]

                score_rows.append({
                    **row,
                    "image_score_top10": image_score,
                    "patch_mean": patch_mean,
                    "patch_p95": patch_p95,
                    "patch_p99": patch_p99,
                    "patch_max": patch_max,
                })

                seen += 1

            if (
                seen % 40 < TEST_BATCH_SIZE
                or seen == len(records)
            ):

                print(
                    f"scored {seen:4d}/"
                    f"{len(records):4d}"
                )

    del memory
    torch.cuda.empty_cache()


# ============================================================
# Save scores
# ============================================================

score_rows = sorted(
    score_rows,
    key=lambda x: (
        x["category"],
        x["set_type"],
        int(x["instance_id"]),
        x["condition"],
    ),
)

with SCORES_CSV.open(
    "w",
    newline="",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=list(
            score_rows[0].keys()
        )
    )

    writer.writeheader()
    writer.writerows(
        score_rows
    )


if len(score_rows) != 1084:
    raise RuntimeError(
        f"Expected 1084 scores, got {len(score_rows)}"
    )


print()
print("=" * 110)
print("A1.3 SCORING COMPLETE")
print("=" * 110)
print("public images :", len(score_rows))
print("score CSV     :", SCORES_CSV)
print("STATUS        : PASS")
print("=" * 110)
