from pathlib import Path
from PIL import Image
import numpy as np
import csv
import json
import re
from collections import Counter, defaultdict

DATA_ROOT = Path("/root/private_data/iad-vlm-anomaly/datasets/MVTec_AD_2")
OUT_ROOT = Path("orbitad/results/a0_dataset_audit")
MANIFEST_ROOT = Path("orbitad/data/manifests")

OUT_ROOT.mkdir(parents=True, exist_ok=True)
MANIFEST_ROOT.mkdir(parents=True, exist_ok=True)

DETAIL_CSV = OUT_ROOT / "public_mask_audit.csv"
SUMMARY_CSV = OUT_ROOT / "public_mask_summary.csv"
CONDITION_CSV = OUT_ROOT / "public_condition_summary.csv"
MANIFEST_CSV = MANIFEST_ROOT / "test_public_bad.csv"
SUMMARY_JSON = OUT_ROOT / "public_mask_summary.json"

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

IMAGE_EXTS = {
    ".png", ".jpg", ".jpeg",
    ".bmp", ".tif", ".tiff"
}


def image_files(path):
    """Recursively return image files."""
    return sorted(
        p for p in path.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def gt_key(path):
    """
    004_shift_1_mask.png -> 004_shift_1
    """
    stem = path.stem.lower()

    if stem.endswith("_mask"):
        stem = stem[:-5]

    return stem


def bad_key(path):
    """
    004_shift_1.png -> 004_shift_1
    """
    return path.stem.lower()


def parse_instance_condition(stem):
    """
    Examples:

    004_regular      -> instance_id=004, condition=regular
    004_overexposed  -> instance_id=004, condition=overexposed
    004_shift_1      -> instance_id=004, condition=shift_1
    """

    m = re.match(r"^(\d+)_(.+)$", stem)

    if not m:
        return None, "unknown"

    return m.group(1), m.group(2)


def load_mask(path):
    with Image.open(path) as im:
        arr = np.asarray(im)

    if arr.ndim == 3:
        binary = np.any(arr > 0, axis=2)
    else:
        binary = arr > 0

    return arr, binary


detail_rows = []
summary_rows = []
condition_counter = Counter()

global_failures = []

total_good = 0
total_bad = 0
total_gt = 0
total_pairs = 0


print("=" * 110)
print("OrbitAD A0.4 — TEST_PUBLIC IMAGE / MASK / CONDITION AUDIT")
print("=" * 110)


for category in CATEGORIES:

    root = DATA_ROOT / category / "test_public"

    good_files = image_files(root / "good")
    bad_files = image_files(root / "bad")
    gt_files = image_files(root / "ground_truth")

    bad_index = {}
    gt_index = {}

    duplicate_bad = []
    duplicate_gt = []

    # ---------------------------------------------------------
    # Index BAD images
    # ---------------------------------------------------------

    for p in bad_files:

        key = bad_key(p)

        if key in bad_index:
            duplicate_bad.append(key)

        bad_index[key] = p

    # ---------------------------------------------------------
    # Index GT masks
    # ---------------------------------------------------------

    for p in gt_files:

        key = gt_key(p)

        if key in gt_index:
            duplicate_gt.append(key)

        gt_index[key] = p

    bad_keys = set(bad_index)
    gt_keys = set(gt_index)

    missing_gt = sorted(bad_keys - gt_keys)
    unused_gt = sorted(gt_keys - bad_keys)

    common = sorted(bad_keys & gt_keys)

    category_failures = []

    if duplicate_bad:
        category_failures.append(
            f"duplicate bad keys={len(duplicate_bad)}"
        )

    if duplicate_gt:
        category_failures.append(
            f"duplicate GT keys={len(duplicate_gt)}"
        )

    if missing_gt:
        category_failures.append(
            f"missing GT={len(missing_gt)}"
        )

    if unused_gt:
        category_failures.append(
            f"unused GT={len(unused_gt)}"
        )

    ratios = []

    empty_masks = 0
    resolution_mismatches = 0

    touch_border = 0
    within32 = 0
    within64 = 0
    within128 = 0

    cat_conditions = Counter()

    # ---------------------------------------------------------
    # Strict image-mask pairs
    # ---------------------------------------------------------

    for key in common:

        image_path = bad_index[key]
        mask_path = gt_index[key]

        instance_id, condition = parse_instance_condition(key)

        condition_counter[(category, condition)] += 1
        cat_conditions[condition] += 1

        with Image.open(image_path) as im:
            width, height = im.size

        with Image.open(mask_path) as im:
            mask_width, mask_height = im.size

        resolution_match = (
            width == mask_width and
            height == mask_height
        )

        if not resolution_match:
            resolution_mismatches += 1
            category_failures.append(
                f"resolution mismatch: {key}"
            )

        raw_mask, mask = load_mask(mask_path)

        positive_pixels = int(mask.sum())
        total_pixels = int(mask.size)

        defect_ratio = (
            positive_pixels / total_pixels
            if total_pixels > 0
            else 0.0
        )

        ratios.append(defect_ratio)

        if positive_pixels == 0:

            empty_masks += 1
            min_border_distance = -1

            category_failures.append(
                f"empty mask: {mask_path.name}"
            )

        else:

            ys, xs = np.where(mask)

            min_y = int(ys.min())
            max_y = int(ys.max())

            min_x = int(xs.min())
            max_x = int(xs.max())

            mask_h, mask_w = mask.shape[:2]

            min_border_distance = int(
                min(
                    min_y,
                    min_x,
                    mask_h - 1 - max_y,
                    mask_w - 1 - max_x,
                )
            )

            touch_border += int(
                min_border_distance == 0
            )

            within32 += int(
                min_border_distance < 32
            )

            within64 += int(
                min_border_distance < 64
            )

            within128 += int(
                min_border_distance < 128
            )

        unique_values = np.unique(raw_mask)

        row = {
            "category": category,
            "instance_id": instance_id,
            "condition": condition,
            "pair_key": key,

            "image_path":
                str(image_path.relative_to(DATA_ROOT)),

            "mask_path":
                str(mask_path.relative_to(DATA_ROOT)),

            "width": width,
            "height": height,

            "resolution_match": resolution_match,

            "mask_positive_pixels":
                positive_pixels,

            "defect_area_ratio":
                defect_ratio,

            "min_border_distance_px":
                min_border_distance,

            "touches_border":
                min_border_distance == 0,

            "within_32px_border":
                0 <= min_border_distance < 32,

            "within_64px_border":
                0 <= min_border_distance < 64,

            "within_128px_border":
                0 <= min_border_distance < 128,

            "mask_unique_value_count":
                len(unique_values),

            "mask_min_value":
                float(unique_values.min()),

            "mask_max_value":
                float(unique_values.max()),
        }

        detail_rows.append(row)

    # ---------------------------------------------------------
    # Category summary
    # ---------------------------------------------------------

    category_status = (
        len(category_failures) == 0
        and len(bad_files) == len(gt_files)
        and len(common) == len(bad_files)
    )

    if not category_status:
        global_failures.extend(
            f"{category}: {x}"
            for x in category_failures
        )

    summary_rows.append({
        "category":
            category,

        "good_images":
            len(good_files),

        "bad_images":
            len(bad_files),

        "gt_masks":
            len(gt_files),

        "paired_bad_masks":
            len(common),

        "missing_masks":
            len(missing_gt),

        "unused_masks":
            len(unused_gt),

        "resolution_mismatches":
            resolution_mismatches,

        "empty_masks":
            empty_masks,

        "defect_ratio_min":
            min(ratios) if ratios else 0,

        "defect_ratio_median":
            float(np.median(ratios))
            if ratios else 0,

        "defect_ratio_mean":
            float(np.mean(ratios))
            if ratios else 0,

        "defect_ratio_max":
            max(ratios) if ratios else 0,

        "touch_border_count":
            touch_border,

        "within_32px_count":
            within32,

        "within_64px_count":
            within64,

        "within_128px_count":
            within128,

        "status":
            "PASS" if category_status else "FAIL",
    })

    total_good += len(good_files)
    total_bad += len(bad_files)
    total_gt += len(gt_files)
    total_pairs += len(common)

    print(f"\n[{category}]")

    print(f"  good images            : {len(good_files)}")
    print(f"  bad images             : {len(bad_files)}")
    print(f"  GT masks               : {len(gt_files)}")
    print(f"  paired                 : {len(common)}")
    print(f"  missing GT             : {len(missing_gt)}")
    print(f"  unused GT              : {len(unused_gt)}")
    print(f"  resolution mismatch    : {resolution_mismatches}")
    print(f"  empty masks            : {empty_masks}")

    print("  conditions:")

    for condition, n in sorted(cat_conditions.items()):
        print(f"    {condition:20s}: {n}")

    if ratios:

        print(
            f"  defect ratio min       : "
            f"{min(ratios):.8f}"
        )

        print(
            f"  defect ratio median    : "
            f"{np.median(ratios):.8f}"
        )

        print(
            f"  defect ratio max       : "
            f"{max(ratios):.8f}"
        )

    print(f"  touches border         : {touch_border}")
    print(f"  within 32px            : {within32}")
    print(f"  within 64px            : {within64}")
    print(f"  within 128px           : {within128}")

    print(
        f"  STATUS                 : "
        f"{'PASS' if category_status else 'FAIL'}"
    )


# =====================================================================
# Write detailed audit + reusable manifest
# =====================================================================

fields = list(detail_rows[0].keys())

with DETAIL_CSV.open("w", newline="") as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fields
    )

    writer.writeheader()
    writer.writerows(detail_rows)


with MANIFEST_CSV.open("w", newline="") as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fields
    )

    writer.writeheader()
    writer.writerows(detail_rows)


# =====================================================================
# Category summary
# =====================================================================

with SUMMARY_CSV.open("w", newline="") as f:

    writer = csv.DictWriter(
        f,
        fieldnames=list(summary_rows[0].keys())
    )

    writer.writeheader()
    writer.writerows(summary_rows)


# =====================================================================
# Condition summary
# =====================================================================

condition_rows = []

for (category, condition), count in sorted(
    condition_counter.items()
):

    condition_rows.append({
        "category": category,
        "condition": condition,
        "count": count,
    })

with CONDITION_CSV.open("w", newline="") as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "category",
            "condition",
            "count",
        ]
    )

    writer.writeheader()
    writer.writerows(condition_rows)


# =====================================================================
# JSON
# =====================================================================

global_status = (
    "PASS"
    if len(global_failures) == 0
    else "FAIL"
)

payload = {
    "status": global_status,
    "total_good_images": total_good,
    "total_bad_images": total_bad,
    "total_gt_masks": total_gt,
    "total_paired_bad_masks": total_pairs,
    "category_summary": summary_rows,
    "condition_summary": condition_rows,
    "failures": global_failures,
}

with SUMMARY_JSON.open("w") as f:
    json.dump(payload, f, indent=2)


print("\n" + "=" * 110)

print(f"TOTAL GOOD IMAGES        : {total_good}")
print(f"TOTAL BAD IMAGES         : {total_bad}")
print(f"TOTAL GT MASKS           : {total_gt}")
print(f"TOTAL IMAGE/MASK PAIRS   : {total_pairs}")

print()
print(f"DETAIL                   : {DETAIL_CSV}")
print(f"PUBLIC MANIFEST          : {MANIFEST_CSV}")
print(f"CATEGORY SUMMARY         : {SUMMARY_CSV}")
print(f"CONDITION SUMMARY        : {CONDITION_CSV}")
print(f"SUMMARY JSON             : {SUMMARY_JSON}")

print()
print(f"GLOBAL STATUS            : {global_status}")

print("=" * 110)

if global_failures:

    print("\nFAILURES:")

    for x in global_failures[:30]:
        print(" -", x)
