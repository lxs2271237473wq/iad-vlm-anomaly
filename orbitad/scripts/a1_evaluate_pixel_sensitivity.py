from pathlib import Path
from collections import defaultdict
import csv
import json

import cv2
import numpy as np


ROOT = Path(
    "orbitad/results/a1_anomaly_score_sensitivity"
)

DATA_ROOT = Path(
    "/root/private_data/iad-vlm-anomaly/datasets/MVTec_AD_2"
)

PUBLIC_MAPS = ROOT / "public_patch_maps.npy"
PUBLIC_META = ROOT / "public_patch_maps.csv"

VAL_MAPS = ROOT / "validation_patch_maps.npy"
VAL_META = ROOT / "validation_patch_maps.csv"

METRICS_CSV = ROOT / "pixel_category_condition_metrics.csv"
CONDITION_CSV = ROOT / "pixel_condition_metrics.csv"
THRESHOLD_JSON = ROOT / "validation_seg_thresholds.json"
SUMMARY_JSON = ROOT / "a1_4_summary.json"


EVAL_SIZE = 512

HIST_BINS = 4096
SCORE_MIN = 0.0
SCORE_MAX = 2.0

MAX_FPR_PRO = 0.05
VAL_NORMAL_QUANTILE = 0.995


edges = np.linspace(
    SCORE_MIN,
    SCORE_MAX,
    HIST_BINS + 1,
    dtype=np.float64,
)


def read_rows(path):

    rows = []

    with path.open() as f:

        for r in csv.DictReader(f):
            r["map_index"] = int(
                r["map_index"]
            )
            r["label"] = int(
                r["label"]
            )
            rows.append(r)

    return rows


public_rows = read_rows(
    PUBLIC_META
)

val_rows = read_rows(
    VAL_META
)

public_maps = np.load(
    PUBLIC_MAPS,
    mmap_mode="r",
)

val_maps = np.load(
    VAL_MAPS,
    mmap_mode="r",
)


# ============================================================
# Resize low-res score map to diagnostic 512x512 grid
# ============================================================

def upsample_map(m):

    return cv2.resize(
        np.asarray(
            m,
            dtype=np.float32,
        ),
        (EVAL_SIZE, EVAL_SIZE),
        interpolation=cv2.INTER_LINEAR,
    )


def load_mask(row):

    if row["label"] == 0:

        return np.zeros(
            (EVAL_SIZE, EVAL_SIZE),
            dtype=np.uint8,
        )

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
            f"Cannot read mask: {path}"
        )

    mask = cv2.resize(
        mask,
        (EVAL_SIZE, EVAL_SIZE),
        interpolation=cv2.INTER_NEAREST,
    )

    return (
        mask > 0
    ).astype(np.uint8)


# ============================================================
# Validation-only segmentation thresholds
# ============================================================

val_hist = defaultdict(
    lambda: np.zeros(
        HIST_BINS,
        dtype=np.int64,
    )
)


for row in val_rows:

    score = upsample_map(
        val_maps[
            row["map_index"]
        ]
    )

    h, _ = np.histogram(
        score,
        bins=edges,
    )

    val_hist[
        row["category"]
    ] += h


def hist_quantile(
    hist,
    q,
):

    total = hist.sum()

    if total == 0:
        raise RuntimeError(
            "Empty validation histogram."
        )

    cdf = np.cumsum(
        hist
    )

    target = q * total

    idx = int(
        np.searchsorted(
            cdf,
            target,
            side="left",
        )
    )

    idx = min(
        max(idx, 0),
        HIST_BINS - 1,
    )

    return float(
        edges[idx]
    )


seg_thresholds = {
    cat: hist_quantile(
        hist,
        VAL_NORMAL_QUANTILE,
    )
    for cat, hist in val_hist.items()
}


with THRESHOLD_JSON.open("w") as f:

    json.dump(
        {
            "quantile":
                VAL_NORMAL_QUANTILE,

            "thresholds":
                seg_thresholds,
        },
        f,
        indent=2,
    )


print("=" * 110)
print("OrbitAD A1.4 — PIXEL-LEVEL ACQUISITION SENSITIVITY")
print("=" * 110)

print()
print("[VALIDATION THRESHOLDS]")

for cat in sorted(
    seg_thresholds
):

    print(
        f"{cat:15s}: "
        f"{seg_thresholds[cat]:.8f}"
    )


# ============================================================
# Group accumulators
# ============================================================

class GroupStats:

    def __init__(self):

        self.pos_hist = np.zeros(
            HIST_BINS,
            dtype=np.int64,
        )

        self.neg_hist = np.zeros(
            HIST_BINS,
            dtype=np.int64,
        )

        # Sum of normalized per-region score histograms.
        # Cumulative version gives mean per-region overlap.
        self.region_hist_sum = np.zeros(
            HIST_BINS,
            dtype=np.float64,
        )

        self.num_regions = 0

        self.tp = 0
        self.fp = 0
        self.fn = 0

        self.n_images = 0
        self.n_normal = 0
        self.n_defect = 0


groups = defaultdict(
    GroupStats
)


for n, row in enumerate(
    public_rows,
    start=1,
):

    category = row["category"]
    condition = row["condition"]

    g = groups[
        (category, condition)
    ]

    score = upsample_map(
        public_maps[
            row["map_index"]
        ]
    )

    mask = load_mask(row)

    positive = mask.astype(bool)
    negative = ~positive

    pos_values = score[
        positive
    ]

    neg_values = score[
        negative
    ]

    if len(pos_values):

        h, _ = np.histogram(
            pos_values,
            bins=edges,
        )

        g.pos_hist += h

    if len(neg_values):

        h, _ = np.histogram(
            neg_values,
            bins=edges,
        )

        g.neg_hist += h

    # --------------------------------------------------------
    # PRO regions
    # --------------------------------------------------------

    if row["label"] == 1:

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

            region_size = len(
                region_scores
            )

            if region_size == 0:
                continue

            h, _ = np.histogram(
                region_scores,
                bins=edges,
            )

            g.region_hist_sum += (
                h.astype(
                    np.float64
                )
                / region_size
            )

            g.num_regions += 1

    # --------------------------------------------------------
    # Validation-calibrated SegF1
    # --------------------------------------------------------

    threshold = seg_thresholds[
        category
    ]

    pred = (
        score >= threshold
    )

    g.tp += int(
        np.logical_and(
            pred,
            positive,
        ).sum()
    )

    g.fp += int(
        np.logical_and(
            pred,
            negative,
        ).sum()
    )

    g.fn += int(
        np.logical_and(
            ~pred,
            positive,
        ).sum()
    )

    g.n_images += 1

    if row["label"] == 0:
        g.n_normal += 1
    else:
        g.n_defect += 1

    if n % 100 == 0:
        print(
            f"processed "
            f"{n}/{len(public_rows)}"
        )


# ============================================================
# Histogram metric functions
# ============================================================

def pixel_auc(
    pos_hist,
    neg_hist,
):

    P = pos_hist.sum()
    N = neg_hist.sum()

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
        np.trapz(
            tpr,
            fpr,
        )
    )


def au_pro(
    neg_hist,
    region_hist_sum,
    num_regions,
    max_fpr=0.05,
):

    N = neg_hist.sum()

    if (
        N == 0
        or num_regions == 0
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
            region_hist_sum[::-1],
            dtype=np.float64,
        ) / num_regions,
    ])

    fpr = fp / N

    # Keep curve through requested FPR.
    keep = (
        fpr <= max_fpr
    )

    x = fpr[keep]
    y = pro[keep]

    # Explicit interpolation at exactly max_fpr.
    if x[-1] < max_fpr:

        idx = np.searchsorted(
            fpr,
            max_fpr,
            side="right",
        )

        if idx < len(fpr):

            x0 = fpr[idx - 1]
            x1 = fpr[idx]

            y0 = pro[idx - 1]
            y1 = pro[idx]

            if x1 > x0:

                alpha = (
                    max_fpr - x0
                ) / (
                    x1 - x0
                )

                y_at = (
                    y0
                    + alpha
                    * (y1 - y0)
                )

            else:
                y_at = y1

            x = np.append(
                x,
                max_fpr,
            )

            y = np.append(
                y,
                y_at,
            )

    area = np.trapz(
        y,
        x,
    )

    # Normalize to [0,1] over [0,max_fpr]
    return float(
        area / max_fpr
    )


def seg_f1(
    tp,
    fp,
    fn,
):

    denom = (
        2 * tp
        + fp
        + fn
    )

    if denom == 0:
        return np.nan

    return float(
        2 * tp / denom
    )


# ============================================================
# Category-condition metrics
# ============================================================

metric_rows = []


for (
    category,
    condition,
), g in sorted(
    groups.items()
):

    metric_rows.append({
        "category": category,
        "condition": condition,
        "n_images": g.n_images,
        "n_normal": g.n_normal,
        "n_defect": g.n_defect,
        "n_regions": g.num_regions,

        "pixel_auroc":
            pixel_auc(
                g.pos_hist,
                g.neg_hist,
            ),

        "aupro_0_05":
            au_pro(
                g.neg_hist,
                g.region_hist_sum,
                g.num_regions,
                MAX_FPR_PRO,
            ),

        "segf1_val_q995":
            seg_f1(
                g.tp,
                g.fp,
                g.fn,
            ),

        "seg_threshold":
            seg_thresholds[
                category
            ],
    })


# regular reference within each category
regular_ref = {
    row["category"]: row
    for row in metric_rows
    if row["condition"] == "regular"
}


for row in metric_rows:

    ref = regular_ref.get(
        row["category"]
    )

    if ref is None:

        row["delta_pixel_auroc"] = np.nan
        row["delta_aupro_0_05"] = np.nan
        row["delta_segf1"] = np.nan

    else:

        row["delta_pixel_auroc"] = (
            row["pixel_auroc"]
            -
            ref["pixel_auroc"]
        )

        row["delta_aupro_0_05"] = (
            row["aupro_0_05"]
            -
            ref["aupro_0_05"]
        )

        row["delta_segf1"] = (
            row["segf1_val_q995"]
            -
            ref["segf1_val_q995"]
        )


with METRICS_CSV.open(
    "w",
    newline="",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=list(
            metric_rows[0].keys()
        )
    )

    writer.writeheader()
    writer.writerows(
        metric_rows
    )


# ============================================================
# Macro condition metrics
# ============================================================

by_condition = defaultdict(
    list
)

for row in metric_rows:

    by_condition[
        row["condition"]
    ].append(row)


condition_rows = []


def mean_finite(values):

    x = np.asarray(
        values,
        dtype=np.float64,
    )

    x = x[
        np.isfinite(x)
    ]

    if len(x) == 0:
        return np.nan

    return float(
        x.mean()
    )


for condition, members in sorted(
    by_condition.items()
):

    condition_rows.append({
        "condition": condition,
        "categories": len(members),

        "macro_pixel_auroc":
            mean_finite([
                x["pixel_auroc"]
                for x in members
            ]),

        "macro_aupro_0_05":
            mean_finite([
                x["aupro_0_05"]
                for x in members
            ]),

        "macro_segf1":
            mean_finite([
                x["segf1_val_q995"]
                for x in members
            ]),

        "macro_delta_pixel_auroc":
            mean_finite([
                x["delta_pixel_auroc"]
                for x in members
            ]),

        "macro_delta_aupro_0_05":
            mean_finite([
                x["delta_aupro_0_05"]
                for x in members
            ]),

        "macro_delta_segf1":
            mean_finite([
                x["delta_segf1"]
                for x in members
            ]),
    })


with CONDITION_CSV.open(
    "w",
    newline="",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=list(
            condition_rows[0].keys()
        )
    )

    writer.writeheader()
    writer.writerows(
        condition_rows
    )


# ============================================================
# Pre-registered GO / NO-GO
# ============================================================

shift_rows = [
    r for r in condition_rows
    if r["condition"] in {
        "shift_1",
        "shift_2",
        "shift_3",
    }
]


worst_pro_delta = min(
    r["macro_delta_aupro_0_05"]
    for r in shift_rows
)

worst_f1_delta = min(
    r["macro_delta_segf1"]
    for r in shift_rows
)


pro_signal = (
    worst_pro_delta <= -0.03
)

f1_signal = (
    worst_f1_delta <= -0.03
)

decision = (
    "GO"
    if (
        pro_signal
        or f1_signal
    )
    else "NO_GO"
)


payload = {
    "protocol": {
        "evaluation_resolution":
            "512x512 diagnostic grid",

        "aupro_max_fpr":
            MAX_FPR_PRO,

        "segmentation_threshold":
            "per-category validation-normal "
            "99.5th percentile",

        "test_gt_used_for_thresholding":
            False,
    },

    "pre_registered_thresholds": {
        "aupro_drop":
            -0.03,

        "segf1_drop":
            -0.03,
    },

    "observed": {
        "worst_shift_macro_aupro_delta":
            float(worst_pro_delta),

        "worst_shift_macro_segf1_delta":
            float(worst_f1_delta),
    },

    "signals": {
        "aupro_signal":
            bool(pro_signal),

        "segf1_signal":
            bool(f1_signal),
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
# Console
# ============================================================

print()
print("[CONDITION PIXEL METRICS]")

print(
    f"{'condition':16s}"
    f"{'cats':>6s}"
    f"{'Pix-AUC':>10s}"
    f"{'PRO05':>10s}"
    f"{'SegF1':>10s}"
    f"{'dPRO':>10s}"
    f"{'dF1':>10s}"
)

print("-" * 72)

for r in condition_rows:

    print(
        f"{r['condition']:16s}"
        f"{r['categories']:6d}"
        f"{r['macro_pixel_auroc']:10.4f}"
        f"{r['macro_aupro_0_05']:10.4f}"
        f"{r['macro_segf1']:10.4f}"
        f"{r['macro_delta_aupro_0_05']:10.4f}"
        f"{r['macro_delta_segf1']:10.4f}"
    )


print()
print("[PRE-REGISTERED GO / NO-GO]")

print(
    "worst shift macro AU-PRO0.05 delta : "
    f"{worst_pro_delta:.4f}"
)

print(
    "worst shift macro SegF1 delta       : "
    f"{worst_f1_delta:.4f}"
)

print(
    "AU-PRO signal                       :",
    pro_signal,
)

print(
    "SegF1 signal                        :",
    f1_signal,
)

print()
print("DECISION :", decision)

print()
print("CATEGORY METRICS :", METRICS_CSV)
print("CONDITION METRICS:", CONDITION_CSV)
print("THRESHOLDS       :", THRESHOLD_JSON)
print("SUMMARY          :", SUMMARY_JSON)

print("=" * 110)
