from pathlib import Path
from collections import defaultdict, Counter
import csv
import json

import cv2
import numpy as np
from PIL import Image


DATA_ROOT = Path(
    "/root/private_data/iad-vlm-anomaly/datasets/MVTec_AD_2"
)

ROOT = Path(
    "orbitad/results/a3_resolution_audit"
)

META = ROOT / "regular_public_meta.csv"

RESOLUTIONS = [
    256,
    512,
    1024,
]

SIZE_CSV = ROOT / "resolution_size_metrics.csv"
BORDER_CSV = ROOT / "resolution_border_metrics.csv"
DEFECT_CSV = ROOT / "regular_defect_strata.csv"
SUMMARY_JSON = ROOT / "a3_0_summary.json"


# Histogram metric resolution.
# Cosine distance for normalized features lies in [0,2].
HIST_BINS = 8192
SCORE_MIN = 0.0
SCORE_MAX = 2.0

MAX_FPR_PRO = 0.05

edges = np.linspace(
    SCORE_MIN,
    SCORE_MAX,
    HIST_BINS + 1,
    dtype=np.float64,
)


# ============================================================
# Metadata
# ============================================================

rows = []

with META.open() as f:

    for row in csv.DictReader(f):

        row["map_index"] = int(
            row["map_index"]
        )

        row["label"] = int(
            row["label"]
        )

        rows.append(row)


if len(rows) != 184:
    raise RuntimeError(
        f"Expected 184 regular-condition images, got {len(rows)}"
    )


normal_rows = [
    r for r in rows
    if r["label"] == 0
]

defect_rows = [
    r for r in rows
    if r["label"] == 1
]


if len(normal_rows) != 64:
    raise RuntimeError(
        f"Expected 64 normal instances, got {len(normal_rows)}"
    )

if len(defect_rows) != 120:
    raise RuntimeError(
        f"Expected 120 defect instances, got {len(defect_rows)}"
    )


maps = {
    resolution: np.load(
        ROOT / f"regular_maps_r{resolution}.npy",
        mmap_mode="r",
    )
    for resolution in RESOLUTIONS
}


expected_shapes = {
    256:  (184, 16, 16),
    512:  (184, 32, 32),
    1024: (184, 64, 64),
}


for resolution in RESOLUTIONS:

    if maps[resolution].shape != expected_shapes[resolution]:

        raise RuntimeError(
            f"Unexpected r{resolution} shape: "
            f"{maps[resolution].shape}"
        )


# ============================================================
# Native-resolution GT audit for 120 defect instances
# ============================================================

def load_native_mask(row):

    path = (
        DATA_ROOT
        / row["mask_path"]
    )

    mask = cv2.imread(
        str(path),
        cv2.IMREAD_GRAYSCALE,
    )

    if mask is None:
        raise RuntimeError(
            f"Cannot read GT mask: {path}"
        )

    return (
        mask > 0
    ).astype(
        np.uint8
    )


def native_image_shape(row):

    path = (
        DATA_ROOT
        / row["image_path"]
    )

    with Image.open(path) as im:

        width, height = im.size

    return height, width


def border_distance(mask):

    ys, xs = np.where(
        mask > 0
    )

    if len(xs) == 0:
        raise RuntimeError(
            "Encountered empty GT mask"
        )

    h, w = mask.shape

    return int(
        min(
            ys.min(),
            xs.min(),
            h - 1 - ys.max(),
            w - 1 - xs.max(),
        )
    )


def border_bin(distance):

    if distance == 0:
        return "touch"

    if distance < 32:
        return "near_1_31"

    if distance < 64:
        return "near_32_63"

    return "far_ge_64"


defect_info = []


for row in defect_rows:

    mask = load_native_mask(
        row
    )

    positive = int(
        mask.sum()
    )

    total = int(
        mask.size
    )

    ratio = (
        positive
        / total
    )

    d_border = border_distance(
        mask
    )

    defect_info.append({
        "map_index":
            row["map_index"],

        "category":
            row["category"],

        "instance_id":
            row["instance_id"],

        "image_path":
            row["image_path"],

        "mask_path":
            row["mask_path"],

        "native_width":
            mask.shape[1],

        "native_height":
            mask.shape[0],

        "positive_pixels":
            positive,

        "defect_area_ratio":
            ratio,

        "border_distance_px":
            d_border,

        "border_bin":
            border_bin(
                d_border
            ),
    })


# ============================================================
# EXACT rank quartiles:
# 30 physical defect instances per quartile.
# ============================================================

sorted_defects = sorted(
    defect_info,
    key=lambda x: (
        x["defect_area_ratio"],
        x["category"],
        int(x["instance_id"]),
    ),
)


if len(sorted_defects) != 120:
    raise RuntimeError(
        "Expected exactly 120 defects"
    )


for rank, info in enumerate(
    sorted_defects
):

    quartile = (
        rank // 30
    ) + 1

    info["size_rank"] = (
        rank + 1
    )

    info["size_quartile"] = (
        f"Q{quartile}"
    )


info_by_map_index = {
    x["map_index"]: x
    for x in sorted_defects
}


# ============================================================
# Save immutable strata definition
# ============================================================

with DEFECT_CSV.open(
    "w",
    newline="",
) as f:

    fields = list(
        sorted_defects[0].keys()
    )

    writer = csv.DictWriter(
        f,
        fieldnames=fields,
    )

    writer.writeheader()
    writer.writerows(
        sorted_defects
    )


# ============================================================
# Report quartile boundaries/composition
# ============================================================

print("=" * 110)
print("OrbitAD A3.0 — RESOLUTION-LOSS AUDIT")
print("=" * 110)

print()
print("[DEFECT SIZE STRATA]")


for q in [
    "Q1",
    "Q2",
    "Q3",
    "Q4",
]:

    members = [
        x for x in sorted_defects
        if x["size_quartile"] == q
    ]

    ratios = np.asarray(
        [
            x["defect_area_ratio"]
            for x in members
        ],
        dtype=np.float64,
    )

    print(
        f"{q}: "
        f"N={len(members):2d} | "
        f"ratio min={ratios.min():.8f} | "
        f"median={np.median(ratios):.8f} | "
        f"max={ratios.max():.8f}"
    )


print()
print("[Q1 CATEGORY COMPOSITION]")

q1_members = [
    x for x in sorted_defects
    if x["size_quartile"] == "Q1"
]

q1_comp = Counter(
    x["category"]
    for x in q1_members
)

for category, count in sorted(
    q1_comp.items()
):
    print(
        f"{category:15s}: {count}"
    )


print()
print("[BORDER STRATA]")

border_counts = Counter(
    x["border_bin"]
    for x in sorted_defects
)

for key in [
    "touch",
    "near_1_31",
    "near_32_63",
    "far_ge_64",
]:

    print(
        f"{key:15s}: "
        f"{border_counts.get(key, 0)}"
    )


# ============================================================
# Histogram accumulators
# ============================================================

class GroupStats:

    def __init__(self):

        self.pos_hist = np.zeros(
            HIST_BINS,
            dtype=np.int64,
        )

        # normalized histogram summed over connected GT regions
        self.region_hist_sum = np.zeros(
            HIST_BINS,
            dtype=np.float64,
        )

        self.num_regions = 0
        self.num_defects = 0


# Shared negative pool PER RESOLUTION.
#
# This is deliberate:
# every size/border stratum is evaluated against exactly
# the same negative-pixel distribution.
#
# Negatives contain:
# - all pixels from the 64 normal regular images
# - background pixels from all 120 defect regular images
negative_hist = {
    resolution: np.zeros(
        HIST_BINS,
        dtype=np.int64,
    )
    for resolution in RESOLUTIONS
}


size_stats = {
    resolution: defaultdict(
        GroupStats
    )
    for resolution in RESOLUTIONS
}


border_stats = {
    resolution: defaultdict(
        GroupStats
    )
    for resolution in RESOLUTIONS
}


# ============================================================
# Utilities
# ============================================================

def resize_score_to_shape(
    score_map,
    height,
    width,
):

    return cv2.resize(
        np.asarray(
            score_map,
            dtype=np.float32,
        ),
        (
            width,
            height,
        ),
        interpolation=cv2.INTER_LINEAR,
    )


def add_hist(
    histogram,
    values,
):

    h, _ = np.histogram(
        values,
        bins=edges,
    )

    histogram += h


def add_positive_region_stats(
    group,
    score,
    mask,
):

    positive = (
        mask > 0
    )

    values = score[
        positive
    ]

    if len(values) > 0:

        add_hist(
            group.pos_hist,
            values,
        )

    num_labels, labels = cv2.connectedComponents(
        mask.astype(
            np.uint8
        ),
        connectivity=8,
    )

    for region_id in range(
        1,
        num_labels,
    ):

        region = (
            labels == region_id
        )

        region_scores = score[
            region
        ]

        if len(region_scores) == 0:
            continue

        h, _ = np.histogram(
            region_scores,
            bins=edges,
        )

        group.region_hist_sum += (
            h.astype(
                np.float64
            )
            /
            len(region_scores)
        )

        group.num_regions += 1

    group.num_defects += 1


# ============================================================
# Accumulate all 184 images, all 3 resolutions
# ============================================================

for resolution in RESOLUTIONS:

    print()
    print(
        f"[ACCUMULATING r{resolution}]"
    )

    current_maps = maps[
        resolution
    ]

    for n, row in enumerate(
        rows,
        start=1,
    ):

        map_index = row[
            "map_index"
        ]

        if row["label"] == 1:

            mask = load_native_mask(
                row
            )

            height, width = mask.shape

        else:

            height, width = native_image_shape(
                row
            )

            mask = np.zeros(
                (
                    height,
                    width,
                ),
                dtype=np.uint8,
            )

        score = resize_score_to_shape(
            current_maps[
                map_index
            ],
            height,
            width,
        )

        positive = (
            mask > 0
        )

        negative = ~positive

        # fixed negative population for all strata
        add_hist(
            negative_hist[
                resolution
            ],
            score[
                negative
            ],
        )

        if row["label"] == 1:

            info = info_by_map_index[
                map_index
            ]

            q = info[
                "size_quartile"
            ]

            b = info[
                "border_bin"
            ]

            # ALL
            add_positive_region_stats(
                size_stats[
                    resolution
                ]["ALL"],
                score,
                mask,
            )

            # size quartile
            add_positive_region_stats(
                size_stats[
                    resolution
                ][q],
                score,
                mask,
            )

            # border
            add_positive_region_stats(
                border_stats[
                    resolution
                ][b],
                score,
                mask,
            )

        if (
            n % 50 == 0
            or n == len(rows)
        ):
            print(
                f"  {n:3d}/{len(rows)}"
            )


# ============================================================
# Metrics
# ============================================================

def pixel_auc(
    pos_hist,
    neg_hist,
):

    P = int(
        pos_hist.sum()
    )

    N = int(
        neg_hist.sum()
    )

    if P == 0 or N == 0:
        return np.nan

    tp = np.concatenate([
        [0.0],
        np.cumsum(
            pos_hist[::-1],
            dtype=np.float64,
        ),
    ])

    fp = np.concatenate([
        [0.0],
        np.cumsum(
            neg_hist[::-1],
            dtype=np.float64,
        ),
    ])

    tpr = tp / P
    fpr = fp / N

    return float(
        np.trapezoid(
            tpr,
            fpr,
        )
    )


def aupro_005(
    group,
    neg_hist,
):

    N = int(
        neg_hist.sum()
    )

    if (
        N == 0
        or group.num_regions == 0
    ):
        return np.nan

    fp = np.concatenate([
        [0.0],
        np.cumsum(
            neg_hist[::-1],
            dtype=np.float64,
        ),
    ])

    pro = np.concatenate([
        [0.0],
        np.cumsum(
            group.region_hist_sum[::-1],
            dtype=np.float64,
        )
        /
        group.num_regions,
    ])

    fpr = (
        fp / N
    )

    keep = (
        fpr <= MAX_FPR_PRO
    )

    x = fpr[
        keep
    ]

    y = pro[
        keep
    ]

    if len(x) == 0:

        x = np.asarray(
            [0.0]
        )

        y = np.asarray(
            [0.0]
        )

    if x[-1] < MAX_FPR_PRO:

        idx = np.searchsorted(
            fpr,
            MAX_FPR_PRO,
            side="right",
        )

        if idx < len(fpr):

            x0 = fpr[
                idx - 1
            ]

            x1 = fpr[
                idx
            ]

            y0 = pro[
                idx - 1
            ]

            y1 = pro[
                idx
            ]

            if x1 > x0:

                alpha = (
                    MAX_FPR_PRO - x0
                ) / (
                    x1 - x0
                )

                y_at = (
                    y0
                    +
                    alpha
                    * (
                        y1 - y0
                    )
                )

            else:

                y_at = y1

            x = np.append(
                x,
                MAX_FPR_PRO,
            )

            y = np.append(
                y,
                y_at,
            )

    return float(
        np.trapezoid(
            y,
            x,
        )
        /
        MAX_FPR_PRO
    )


# ============================================================
# Size table
# ============================================================

size_rows = []


for resolution in RESOLUTIONS:

    for group_name in [
        "ALL",
        "Q1",
        "Q2",
        "Q3",
        "Q4",
    ]:

        g = size_stats[
            resolution
        ][group_name]

        size_rows.append({
            "resolution":
                resolution,

            "size_group":
                group_name,

            "n_defects":
                g.num_defects,

            "n_regions":
                g.num_regions,

            "pixel_auroc":
                pixel_auc(
                    g.pos_hist,
                    negative_hist[
                        resolution
                    ],
                ),

            "aupro_0_05":
                aupro_005(
                    g,
                    negative_hist[
                        resolution
                    ],
                ),
        })


with SIZE_CSV.open(
    "w",
    newline="",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=list(
            size_rows[0].keys()
        ),
    )

    writer.writeheader()
    writer.writerows(
        size_rows
    )


# ============================================================
# Border table
# ============================================================

border_rows = []


for resolution in RESOLUTIONS:

    for group_name in [
        "touch",
        "near_1_31",
        "near_32_63",
        "far_ge_64",
    ]:

        g = border_stats[
            resolution
        ][group_name]

        border_rows.append({
            "resolution":
                resolution,

            "border_group":
                group_name,

            "n_defects":
                g.num_defects,

            "n_regions":
                g.num_regions,

            "pixel_auroc":
                pixel_auc(
                    g.pos_hist,
                    negative_hist[
                        resolution
                    ],
                ),

            "aupro_0_05":
                aupro_005(
                    g,
                    negative_hist[
                        resolution
                    ],
                ),
        })


with BORDER_CSV.open(
    "w",
    newline="",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=list(
            border_rows[0].keys()
        ),
    )

    writer.writeheader()
    writer.writerows(
        border_rows
    )


# ============================================================
# Lookup
# ============================================================

size_lookup = {
    (
        r["resolution"],
        r["size_group"],
    ): r
    for r in size_rows
}


tiny_256 = size_lookup[
    (
        256,
        "Q1",
    )
]["aupro_0_05"]

tiny_512 = size_lookup[
    (
        512,
        "Q1",
    )
]["aupro_0_05"]

tiny_1024 = size_lookup[
    (
        1024,
        "Q1",
    )
]["aupro_0_05"]


large_256 = size_lookup[
    (
        256,
        "Q4",
    )
]["aupro_0_05"]

large_1024 = size_lookup[
    (
        1024,
        "Q4",
    )
]["aupro_0_05"]


delta_tiny = (
    tiny_1024
    -
    tiny_256
)

delta_large = (
    large_1024
    -
    large_256
)

selective_gap = (
    delta_tiny
    -
    delta_large
)


# ============================================================
# PRE-REGISTERED GO / NO-GO
# ============================================================

tiny_recovery_signal = (
    delta_tiny >= 0.05
)

size_specific_signal = (
    selective_gap >= 0.03
)

decision = (
    "GO"
    if (
        tiny_recovery_signal
        and
        size_specific_signal
    )
    else "NO_GO"
)


payload = {
    "protocol": {
        "condition":
            "regular only",

        "normal_instances":
            64,

        "defect_instances":
            120,

        "size_strata":
            "exact rank quartiles, 30 defect instances each",

        "negative_pool":
            "shared within each resolution across all size/border strata",

        "evaluation_grid":
            "native GT resolution",

        "test_gt_used_for_model_selection":
            False,
    },

    "pre_registered_thresholds": {
        "tiny_q1_1024_minus_256_aupro_min":
            0.05,

        "tiny_minus_large_recovery_gap_min":
            0.03,
    },

    "observed": {
        "q1_pro_r256":
            float(tiny_256),

        "q1_pro_r512":
            float(tiny_512),

        "q1_pro_r1024":
            float(tiny_1024),

        "q4_pro_r256":
            float(large_256),

        "q4_pro_r1024":
            float(large_1024),

        "tiny_recovery_1024_minus_256":
            float(delta_tiny),

        "large_recovery_1024_minus_256":
            float(delta_large),

        "tiny_minus_large_recovery_gap":
            float(selective_gap),
    },

    "signals": {
        "tiny_recovery_signal":
            bool(
                tiny_recovery_signal
            ),

        "size_specific_signal":
            bool(
                size_specific_signal
            ),
    },

    "decision":
        decision,
}


with SUMMARY_JSON.open("w") as f:

    json.dump(
        payload,
        f,
        indent=2,
    )


# ============================================================
# Console results
# ============================================================

print()
print("=" * 110)
print("A3.0 SIZE-STRATIFIED RESOLUTION RESULTS")
print("=" * 110)

print(
    f"{'resolution':>10s}"
    f"{'ALL':>11s}"
    f"{'Q1 tiny':>11s}"
    f"{'Q2':>11s}"
    f"{'Q3':>11s}"
    f"{'Q4 large':>11s}"
)

print("-" * 65)

for resolution in RESOLUTIONS:

    values = []

    for group in [
        "ALL",
        "Q1",
        "Q2",
        "Q3",
        "Q4",
    ]:

        values.append(
            size_lookup[
                (
                    resolution,
                    group,
                )
            ][
                "aupro_0_05"
            ]
        )

    print(
        f"{resolution:10d}"
        +
        "".join(
            f"{x:11.4f}"
            for x in values
        )
    )


print()
print("[256 -> 1024 RECOVERY]")

print(
    "Q1 tiny  AU-PRO gain : "
    f"{delta_tiny:+.4f}"
)

print(
    "Q4 large AU-PRO gain : "
    f"{delta_large:+.4f}"
)

print(
    "tiny - large gap     : "
    f"{selective_gap:+.4f}"
)


print()
print("[BORDER-STRATIFIED AU-PRO]")

print(
    f"{'resolution':>10s}"
    f"{'touch':>12s}"
    f"{'1-31px':>12s}"
    f"{'32-63px':>12s}"
    f"{'>=64px':>12s}"
)

print("-" * 58)


border_lookup = {
    (
        r["resolution"],
        r["border_group"],
    ): r
    for r in border_rows
}


for resolution in RESOLUTIONS:

    vals = []

    for group in [
        "touch",
        "near_1_31",
        "near_32_63",
        "far_ge_64",
    ]:

        vals.append(
            border_lookup[
                (
                    resolution,
                    group,
                )
            ][
                "aupro_0_05"
            ]
        )

    print(
        f"{resolution:10d}"
        +
        "".join(
            f"{x:12.4f}"
            if np.isfinite(x)
            else
            f"{'NA':>12s}"
            for x in vals
        )
    )


print()
print("[PRE-REGISTERED A3.0 GO / NO-GO]")

print(
    "required Q1 recovery >= +0.0500 :",
    tiny_recovery_signal,
)

print(
    "required Q1-Q4 gap >= +0.0300   :",
    size_specific_signal,
)

print()
print("DECISION :", decision)

print()
print("DEFECT STRATA :", DEFECT_CSV)
print("SIZE METRICS  :", SIZE_CSV)
print("BORDER METRICS:", BORDER_CSV)
print("SUMMARY       :", SUMMARY_JSON)

print("=" * 110)
