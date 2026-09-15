from pathlib import Path
from PIL import Image
import numpy as np
import csv
import json
from collections import defaultdict, Counter
from itertools import combinations

DATA_ROOT = Path(
    "/root/private_data/iad-vlm-anomaly/datasets/MVTec_AD_2"
)

MANIFEST = Path(
    "orbitad/data/manifests/test_public_bad.csv"
)

OUT_ROOT = Path(
    "orbitad/results/a0_dataset_audit"
)

OUT_ROOT.mkdir(parents=True, exist_ok=True)

GROUP_CSV = OUT_ROOT / "public_condition_group_summary.csv"
PAIR_CSV = OUT_ROOT / "public_condition_mask_iou.csv"
SUMMARY_JSON = OUT_ROOT / "public_condition_pair_summary.json"


def load_binary_mask(rel_path):

    path = DATA_ROOT / rel_path

    with Image.open(path) as im:
        arr = np.asarray(im)

    if arr.ndim == 3:
        return np.any(arr > 0, axis=2)

    return arr > 0


def mask_iou(a, b):

    if a.shape != b.shape:
        return None

    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()

    if union == 0:
        return 1.0

    return float(inter / union)


# ============================================================
# Load manifest
# ============================================================

groups = defaultdict(list)

with MANIFEST.open() as f:

    reader = csv.DictReader(f)

    for row in reader:

        key = (
            row["category"],
            row["instance_id"],
        )

        groups[key].append(row)


print("=" * 110)
print("OrbitAD A0.5 — PUBLIC MULTI-CONDITION INSTANCE AUDIT")
print("=" * 110)

group_rows = []
pair_rows = []

global_condition_counter = Counter()
category_group_counter = Counter()

complete_groups = 0
multi_condition_groups = 0
single_condition_groups = 0

all_pair_ious = []
regular_pair_ious = []


# ============================================================
# Analyze groups
# ============================================================

for (category, instance_id), rows in sorted(groups.items()):

    rows = sorted(
        rows,
        key=lambda x: x["condition"]
    )

    conditions = [
        r["condition"]
        for r in rows
    ]

    for c in conditions:
        global_condition_counter[c] += 1

    category_group_counter[category] += 1

    if len(rows) == 1:
        single_condition_groups += 1
    else:
        multi_condition_groups += 1

    masks = {}

    shapes = {}

    for row in rows:

        mask = load_binary_mask(
            row["mask_path"]
        )

        masks[row["condition"]] = mask
        shapes[row["condition"]] = mask.shape

    pairwise_ious = []
    regular_ious = []

    # --------------------------------------------------------
    # Every condition pair
    # --------------------------------------------------------

    for a, b in combinations(rows, 2):

        ca = a["condition"]
        cb = b["condition"]

        ma = masks[ca]
        mb = masks[cb]

        iou = mask_iou(ma, mb)

        pair_rows.append({
            "category": category,
            "instance_id": instance_id,
            "condition_a": ca,
            "condition_b": cb,
            "mask_iou": (
                "" if iou is None else iou
            ),
            "shape_match": ma.shape == mb.shape,
        })

        if iou is not None:

            pairwise_ious.append(iou)
            all_pair_ious.append(iou)

            if ca == "regular" or cb == "regular":
                regular_ious.append(iou)
                regular_pair_ious.append(iou)

    if pairwise_ious:

        min_iou = float(
            min(pairwise_ious)
        )

        mean_iou = float(
            np.mean(pairwise_ious)
        )

        median_iou = float(
            np.median(pairwise_ious)
        )

    else:

        min_iou = None
        mean_iou = None
        median_iou = None

    if regular_ious:

        regular_min_iou = float(
            min(regular_ious)
        )

        regular_mean_iou = float(
            np.mean(regular_ious)
        )

    else:

        regular_min_iou = None
        regular_mean_iou = None

    group_rows.append({
        "category": category,
        "instance_id": instance_id,
        "num_conditions": len(rows),
        "conditions": "|".join(conditions),
        "has_regular": "regular" in conditions,
        "all_shapes_equal":
            len(set(shapes.values())) == 1,
        "pairwise_min_mask_iou":
            "" if min_iou is None else min_iou,
        "pairwise_mean_mask_iou":
            "" if mean_iou is None else mean_iou,
        "pairwise_median_mask_iou":
            "" if median_iou is None else median_iou,
        "regular_min_mask_iou":
            "" if regular_min_iou is None
            else regular_min_iou,
        "regular_mean_mask_iou":
            "" if regular_mean_iou is None
            else regular_mean_iou,
    })


# ============================================================
# Write CSVs
# ============================================================

with GROUP_CSV.open("w", newline="") as f:

    writer = csv.DictWriter(
        f,
        fieldnames=list(group_rows[0].keys())
    )

    writer.writeheader()
    writer.writerows(group_rows)


with PAIR_CSV.open("w", newline="") as f:

    fields = [
        "category",
        "instance_id",
        "condition_a",
        "condition_b",
        "mask_iou",
        "shape_match",
    ]

    writer = csv.DictWriter(
        f,
        fieldnames=fields
    )

    writer.writeheader()
    writer.writerows(pair_rows)


# ============================================================
# Summary
# ============================================================

num_groups = len(groups)

condition_histogram = Counter(
    len(rows)
    for rows in groups.values()
)

if all_pair_ious:

    all_iou_summary = {
        "count": len(all_pair_ious),
        "min": float(np.min(all_pair_ious)),
        "p05": float(np.percentile(
            all_pair_ious, 5
        )),
        "median": float(np.median(
            all_pair_ious
        )),
        "mean": float(np.mean(
            all_pair_ious
        )),
        "p95": float(np.percentile(
            all_pair_ious, 95
        )),
        "max": float(np.max(all_pair_ious)),
        "fraction_ge_0_90":
            float(np.mean(
                np.array(all_pair_ious) >= 0.90
            )),
        "fraction_ge_0_95":
            float(np.mean(
                np.array(all_pair_ious) >= 0.95
            )),
        "fraction_ge_0_99":
            float(np.mean(
                np.array(all_pair_ious) >= 0.99
            )),
    }

else:

    all_iou_summary = {}


if regular_pair_ious:

    regular_iou_summary = {
        "count":
            len(regular_pair_ious),

        "min":
            float(np.min(
                regular_pair_ious
            )),

        "median":
            float(np.median(
                regular_pair_ious
            )),

        "mean":
            float(np.mean(
                regular_pair_ious
            )),

        "fraction_ge_0_95":
            float(np.mean(
                np.array(
                    regular_pair_ious
                ) >= 0.95
            )),
    }

else:

    regular_iou_summary = {}


payload = {

    "num_unique_instances":
        num_groups,

    "multi_condition_instances":
        multi_condition_groups,

    "single_condition_instances":
        single_condition_groups,

    "condition_count_histogram":
        dict(
            sorted(
                condition_histogram.items()
            )
        ),

    "global_condition_counts":
        dict(
            sorted(
                global_condition_counter.items()
            )
        ),

    "category_instance_counts":
        dict(
            sorted(
                category_group_counter.items()
            )
        ),

    "all_pairwise_mask_iou":
        all_iou_summary,

    "regular_vs_other_mask_iou":
        regular_iou_summary,
}


with SUMMARY_JSON.open("w") as f:

    json.dump(
        payload,
        f,
        indent=2
    )


# ============================================================
# Console report
# ============================================================

print()
print("[INSTANCE STRUCTURE]")
print(
    f"unique defect instances      : {num_groups}"
)

print(
    f"multi-condition instances    : {multi_condition_groups}"
)

print(
    f"single-condition instances   : {single_condition_groups}"
)


print()
print("[NUMBER OF CONDITIONS PER INSTANCE]")

for n, count in sorted(
    condition_histogram.items()
):

    print(
        f"  {n:2d} conditions : {count:4d} instances"
    )


print()
print("[GLOBAL CONDITION COUNTS]")

for condition, count in sorted(
    global_condition_counter.items()
):

    print(
        f"  {condition:20s}: {count}"
    )


print()
print("[INSTANCES PER CATEGORY]")

for category, count in sorted(
    category_group_counter.items()
):

    print(
        f"  {category:15s}: {count}"
    )


print()
print("[ALL CONDITION-PAIR MASK IoU]")

if all_iou_summary:

    for k, v in all_iou_summary.items():

        if isinstance(v, float):
            print(
                f"  {k:20s}: {v:.8f}"
            )

        else:
            print(
                f"  {k:20s}: {v}"
            )


print()
print("[REGULAR VS OTHER MASK IoU]")

if regular_iou_summary:

    for k, v in regular_iou_summary.items():

        if isinstance(v, float):
            print(
                f"  {k:20s}: {v:.8f}"
            )

        else:
            print(
                f"  {k:20s}: {v}"
            )


print()
print(f"GROUP CSV   : {GROUP_CSV}")
print(f"PAIR CSV    : {PAIR_CSV}")
print(f"SUMMARY     : {SUMMARY_JSON}")

print("=" * 110)
