from pathlib import Path
from collections import defaultdict
import csv
import json
import math

import cv2
import numpy as np


# ============================================================
# Paths
# ============================================================

DATA_ROOT = Path(
    "/root/private_data/iad-vlm-anomaly/datasets/MVTec_AD_2"
)

A2_ROOT = Path(
    "orbitad/results/a2_counterfactual_persistence"
)

OUT_ROOT = Path(
    "orbitad/results/a4_occ"
)

OUT_ROOT.mkdir(
    parents=True,
    exist_ok=True
)

PUBLIC_MAPS_PATH = (
    A2_ROOT / "public_orbit_maps.npy"
)

VAL_MAPS_PATH = (
    A2_ROOT / "validation_orbit_maps.npy"
)

PUBLIC_META_PATH = (
    A2_ROOT / "public_orbit_meta.csv"
)

VAL_META_PATH = (
    A2_ROOT / "validation_orbit_meta.csv"
)

DETAIL_CSV = (
    OUT_ROOT / "occ_category_condition_metrics.csv"
)

CONDITION_CSV = (
    OUT_ROOT / "occ_condition_metrics.csv"
)

THRESHOLD_JSON = (
    OUT_ROOT / "occ_validation_thresholds.json"
)

SUMMARY_JSON = (
    OUT_ROOT / "a4_0_summary.json"
)


# ============================================================
# Locked protocol
# ============================================================

METHODS = [
    "raw_regular_cal",
    "raw_orbit_cal",
    "spatial_occ",
]

SHIFT_CONDITIONS = [
    "shift_1",
    "shift_2",
    "shift_3",
]

EVAL_SIZE = 512

VAL_Q = 0.995

MAX_FPR = 0.05

RAW_BINS = 4096
OCC_BINS = 4096

RAW_EDGES = np.linspace(
    0.0,
    2.0,
    RAW_BINS + 1,
)

OCC_EDGES = np.linspace(
    0.0,
    8.0,
    OCC_BINS + 1,
)


# ============================================================
# Load
# ============================================================

public_maps = np.load(
    PUBLIC_MAPS_PATH,
    mmap_mode="r",
)

val_maps = np.load(
    VAL_MAPS_PATH,
    mmap_mode="r",
)


def read_meta(path):

    rows = []

    with path.open() as f:

        for row in csv.DictReader(f):

            row["orbit_index"] = int(
                row["orbit_index"]
            )

            row["label"] = int(
                row["label"]
            )

            rows.append(row)

    return rows


public_rows = read_meta(
    PUBLIC_META_PATH
)

val_rows = read_meta(
    VAL_META_PATH
)


print("=" * 110)
print("OrbitAD A4.0 — ORBIT-CONDITIONED CONFORMAL CALIBRATION")
print("=" * 110)

print("public maps    :", public_maps.shape)
print("validation maps:", val_maps.shape)


if public_maps.shape != (
    1084,
    6,
    32,
    32,
):
    raise RuntimeError(
        f"Unexpected public shape: {public_maps.shape}"
    )


if val_maps.shape != (
    302,
    6,
    32,
    32,
):
    raise RuntimeError(
        f"Unexpected val shape: {val_maps.shape}"
    )


# ============================================================
# Deterministic validation split
#
# Half:
#   conformal null calibration
#
# Half:
#   segmentation threshold calibration
#
# No public GT participates.
# ============================================================

val_by_category = defaultdict(
    list
)

for row in val_rows:

    val_by_category[
        row["category"]
    ].append(row)


calibration_rows = defaultdict(
    list
)

threshold_rows = defaultdict(
    list
)


for category, members in sorted(
    val_by_category.items()
):

    members = sorted(
        members,
        key=lambda r: (
            int(r["instance_id"]),
            r["image_path"],
        )
    )

    for i, row in enumerate(members):

        if i % 2 == 0:

            calibration_rows[
                category
            ].append(row)

        else:

            threshold_rows[
                category
            ].append(row)


print()
print("[VALIDATION SPLIT]")

for category in sorted(
    val_by_category
):

    print(
        f"{category:15s} "
        f"null={len(calibration_rows[category]):3d} "
        f"threshold={len(threshold_rows[category]):3d}"
    )


# ============================================================
# Spatial conformal null
#
# Each spatial patch gets its own synthetic-normal null.
# Shape per category:
#
# [Ncal * 6 states, 1024 patch positions]
# ============================================================

sorted_null = {}


for category in sorted(
    calibration_rows
):

    indices = [
        r["orbit_index"]
        for r in calibration_rows[
            category
        ]
    ]

    x = np.asarray(
        val_maps[
            indices
        ],
        dtype=np.float32,
    )

    # [N,6,32,32]
    # ->
    # [N*6,1024]
    x = x.reshape(
        -1,
        1024,
    )

    x.sort(
        axis=0
    )

    sorted_null[
        category
    ] = x


# ============================================================
# OCC transform
# ============================================================

def occ_transform(
    raw32,
    category,
):

    null = sorted_null[
        category
    ]

    n = null.shape[0]

    raw = np.asarray(
        raw32,
        dtype=np.float32,
    ).reshape(
        -1
    )

    score = np.zeros(
        1024,
        dtype=np.float32,
    )

    # Only 1024 positions/image.
    for p in range(
        1024
    ):

        # Number of calibration-null scores >= observed score.
        left = np.searchsorted(
            null[:, p],
            raw[p],
            side="left",
        )

        tail_count = (
            n - left
        )

        conformal_p = (
            1.0
            + tail_count
        ) / (
            n + 1.0
        )

        score[p] = (
            -math.log(
                conformal_p
            )
        )

    return score.reshape(
        32,
        32,
    )


# ============================================================
# Threshold calibration
#
# Controls:
#
# raw_regular_cal:
#     raw map threshold from regular validation only
#
# raw_orbit_cal:
#     same raw test map, but threshold trained on all synthetic
#     validation-normal orbit states
#
# spatial_occ:
#     position-conditioned conformal map
# ============================================================

thresholds = {}


for category in sorted(
    threshold_rows
):

    indices = [
        r["orbit_index"]
        for r in threshold_rows[
            category
        ]
    ]

    # ----------------------------------------
    # Raw regular-only calibration
    # ----------------------------------------

    raw_regular = np.asarray(
        val_maps[
            indices,
            0,
        ],
        dtype=np.float32,
    ).reshape(
        -1
    )

    thresholds[
        (
            "raw_regular_cal",
            category,
        )
    ] = float(
        np.quantile(
            raw_regular,
            VAL_Q,
        )
    )

    # ----------------------------------------
    # Raw acquisition-orbit calibration
    # ----------------------------------------

    raw_orbit = np.asarray(
        val_maps[
            indices
        ],
        dtype=np.float32,
    ).reshape(
        -1
    )

    thresholds[
        (
            "raw_orbit_cal",
            category,
        )
    ] = float(
        np.quantile(
            raw_orbit,
            VAL_Q,
        )
    )

    # ----------------------------------------
    # OCC threshold
    # ----------------------------------------

    occ_values = []

    for row in threshold_rows[
        category
    ]:

        orbit = np.asarray(
            val_maps[
                row["orbit_index"]
            ],
            dtype=np.float32,
        )

        for state in range(
            orbit.shape[0]
        ):

            occ = occ_transform(
                orbit[
                    state
                ],
                category,
            )

            occ_values.append(
                occ.reshape(
                    -1
                )
            )

    occ_values = np.concatenate(
        occ_values
    )

    thresholds[
        (
            "spatial_occ",
            category,
        )
    ] = float(
        np.quantile(
            occ_values,
            VAL_Q,
        )
    )


with THRESHOLD_JSON.open(
    "w"
) as f:

    json.dump(
        {
            f"{method}/{category}":
                float(value)

            for (
                method,
                category,
            ), value in thresholds.items()
        },
        f,
        indent=2,
    )


print()
print("[THRESHOLDS]")

for category in sorted(
    val_by_category
):

    print(
        f"{category:15s} "
        f"raw-reg="
        f"{thresholds[('raw_regular_cal', category)]:.6f} "
        f"raw-orbit="
        f"{thresholds[('raw_orbit_cal', category)]:.6f} "
        f"OCC="
        f"{thresholds[('spatial_occ', category)]:.6f}"
    )


# ============================================================
# Public evaluation
# ============================================================

def edges_for(
    method,
):

    if method == "spatial_occ":
        return OCC_EDGES

    return RAW_EDGES


class Stats:

    def __init__(
        self,
        edges,
    ):

        bins = (
            len(edges) - 1
        )

        self.edges = edges

        self.pos = np.zeros(
            bins,
            dtype=np.int64,
        )

        self.neg = np.zeros(
            bins,
            dtype=np.int64,
        )

        self.region = np.zeros(
            bins,
            dtype=np.float64,
        )

        self.regions = 0

        self.tp = 0
        self.fp = 0
        self.fn = 0


stats = {}


def get_stats(
    method,
    category,
    condition,
):

    key = (
        method,
        category,
        condition,
    )

    if key not in stats:

        stats[key] = Stats(
            edges_for(
                method
            )
        )

    return stats[
        key
    ]


def upsample(
    score32,
):

    return cv2.resize(
        np.asarray(
            score32,
            dtype=np.float32,
        ),
        (
            EVAL_SIZE,
            EVAL_SIZE,
        ),
        interpolation=cv2.INTER_LINEAR,
    )


def load_mask(
    row,
):

    if row["label"] == 0:

        return np.zeros(
            (
                EVAL_SIZE,
                EVAL_SIZE,
            ),
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
        (
            EVAL_SIZE,
            EVAL_SIZE,
        ),
        interpolation=cv2.INTER_NEAREST,
    )

    return (
        mask > 0
    ).astype(
        np.uint8
    )


def add_hist(
    hist,
    values,
    edges,
):

    h, _ = np.histogram(
        values,
        bins=edges,
    )

    hist += h


for n, row in enumerate(
    public_rows,
    1,
):

    category = row[
        "category"
    ]

    condition = row[
        "condition"
    ]

    # IMPORTANT:
    #
    # Only the REAL observed public image is scored.
    # No synthetic test-time orbit aggregation.
    raw32 = np.asarray(
        public_maps[
            row["orbit_index"],
            0,
        ],
        dtype=np.float32,
    )

    occ32 = occ_transform(
        raw32,
        category,
    )

    maps = {
        "raw_regular_cal":
            raw32,

        "raw_orbit_cal":
            raw32,

        "spatial_occ":
            occ32,
    }

    mask = load_mask(
        row
    )

    positive = (
        mask > 0
    )

    negative = ~positive


    for method, map32 in maps.items():

        g = get_stats(
            method,
            category,
            condition,
        )

        score = upsample(
            map32
        )

        add_hist(
            g.pos,
            score[
                positive
            ],
            g.edges,
        )

        add_hist(
            g.neg,
            score[
                negative
            ],
            g.edges,
        )

        # ----------------------------------------
        # AU-PRO regions
        # ----------------------------------------

        if row["label"] == 1:

            num, labels = (
                cv2.connectedComponents(
                    mask,
                    connectivity=8,
                )
            )

            for rid in range(
                1,
                num,
            ):

                values = score[
                    labels == rid
                ]

                if len(
                    values
                ) == 0:
                    continue

                h, _ = np.histogram(
                    values,
                    bins=g.edges,
                )

                g.region += (
                    h.astype(
                        np.float64
                    )
                    /
                    len(values)
                )

                g.regions += 1

        # ----------------------------------------
        # SegF1
        # ----------------------------------------

        threshold = thresholds[
            (
                method,
                category,
            )
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

    if (
        n % 100 == 0
        or n == len(
            public_rows
        )
    ):

        print(
            f"processed "
            f"{n}/{len(public_rows)}"
        )


# ============================================================
# Metric functions
# ============================================================

def pixel_auc(
    g,
):

    P = int(
        g.pos.sum()
    )

    N = int(
        g.neg.sum()
    )

    if (
        P == 0
        or N == 0
    ):
        return np.nan

    tp = np.concatenate([
        [0.0],
        np.cumsum(
            g.pos[::-1],
            dtype=np.float64,
        ),
    ])

    fp = np.concatenate([
        [0.0],
        np.cumsum(
            g.neg[::-1],
            dtype=np.float64,
        ),
    ])

    return float(
        np.trapezoid(
            tp / P,
            fp / N,
        )
    )


def aupro(
    g,
):

    N = int(
        g.neg.sum()
    )

    if (
        N == 0
        or g.regions == 0
    ):
        return np.nan

    fp = np.concatenate([
        [0.0],
        np.cumsum(
            g.neg[::-1],
            dtype=np.float64,
        ),
    ])

    pro = np.concatenate([
        [0.0],
        np.cumsum(
            g.region[::-1],
            dtype=np.float64,
        )
        /
        g.regions,
    ])

    fpr = (
        fp / N
    )

    keep = (
        fpr <= MAX_FPR
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

    if x[-1] < MAX_FPR:

        idx = np.searchsorted(
            fpr,
            MAX_FPR,
            side="right",
        )

        if idx < len(
            fpr
        ):

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

            alpha = (
                MAX_FPR
                - x0
            ) / max(
                x1 - x0,
                1e-12,
            )

            y_at = (
                y0
                + alpha
                * (
                    y1 - y0
                )
            )

            x = np.append(
                x,
                MAX_FPR,
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
        MAX_FPR
    )


def segf1(
    g,
):

    denom = (
        2 * g.tp
        + g.fp
        + g.fn
    )

    if denom == 0:
        return np.nan

    return float(
        2 * g.tp
        /
        denom
    )


# ============================================================
# Category-condition table
# ============================================================

detail_rows = []


for (
    method,
    category,
    condition,
), g in sorted(
    stats.items()
):

    detail_rows.append({
        "method":
            method,

        "category":
            category,

        "condition":
            condition,

        "pixel_auroc":
            pixel_auc(
                g
            ),

        "aupro_0_05":
            aupro(
                g
            ),

        "segf1":
            segf1(
                g
            ),
    })


with DETAIL_CSV.open(
    "w",
    newline="",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=list(
            detail_rows[0].keys()
        ),
    )

    writer.writeheader()
    writer.writerows(
        detail_rows
    )


# ============================================================
# Macro by method × condition
# ============================================================

macro_groups = defaultdict(
    list
)


for row in detail_rows:

    macro_groups[
        (
            row["method"],
            row["condition"],
        )
    ].append(row)


def avg_finite(
    values,
):

    x = np.asarray(
        values,
        dtype=np.float64,
    )

    x = x[
        np.isfinite(
            x
        )
    ]

    if len(x) == 0:
        return np.nan

    return float(
        x.mean()
    )


condition_rows = []


for (
    method,
    condition,
), members in sorted(
    macro_groups.items()
):

    condition_rows.append({
        "method":
            method,

        "condition":
            condition,

        "categories":
            len(members),

        "pixel_auroc":
            avg_finite([
                x["pixel_auroc"]
                for x in members
            ]),

        "aupro_0_05":
            avg_finite([
                x["aupro_0_05"]
                for x in members
            ]),

        "segf1":
            avg_finite([
                x["segf1"]
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
        ),
    )

    writer.writeheader()
    writer.writerows(
        condition_rows
    )


lookup = {
    (
        r["method"],
        r["condition"],
    ): r
    for r in condition_rows
}


# ============================================================
# Pre-registered A4.0 test
# ============================================================

def shift_mean(
    method,
    metric,
):

    values = []

    for condition in SHIFT_CONDITIONS:

        key = (
            method,
            condition,
        )

        if key in lookup:

            values.append(
                lookup[key][
                    metric
                ]
            )

    return float(
        np.mean(
            values
        )
    )


BASELINE = (
    "raw_regular_cal"
)

PRIMARY = (
    "spatial_occ"
)


baseline_shift_f1 = shift_mean(
    BASELINE,
    "segf1",
)

occ_shift_f1 = shift_mean(
    PRIMARY,
    "segf1",
)

baseline_shift_pro = shift_mean(
    BASELINE,
    "aupro_0_05",
)

occ_shift_pro = shift_mean(
    PRIMARY,
    "aupro_0_05",
)


baseline_reg_f1 = lookup[
    (
        BASELINE,
        "regular",
    )
]["segf1"]

occ_reg_f1 = lookup[
    (
        PRIMARY,
        "regular",
    )
]["segf1"]

baseline_reg_pro = lookup[
    (
        BASELINE,
        "regular",
    )
]["aupro_0_05"]

occ_reg_pro = lookup[
    (
        PRIMARY,
        "regular",
    )
]["aupro_0_05"]


shift_f1_gain = (
    occ_shift_f1
    -
    baseline_shift_f1
)

regular_f1_delta = (
    occ_reg_f1
    -
    baseline_reg_f1
)

shift_pro_gain = (
    occ_shift_pro
    -
    baseline_shift_pro
)

regular_pro_delta = (
    occ_reg_pro
    -
    baseline_reg_pro
)


seg_signal = (
    shift_f1_gain >= 0.02
    and
    regular_f1_delta >= -0.015
)

pro_signal = (
    shift_pro_gain >= 0.01
    and
    regular_pro_delta >= -0.015
)


decision = (
    "GO"
    if (
        seg_signal
        or
        pro_signal
    )
    else "NO_GO"
)


payload = {
    "method":
        "Orbit-Conditioned Conformal Calibration",

    "protocol": {
        "validation_split":
            "deterministic half/null + half/threshold",

        "null_states":
            6,

        "test_time_augmentation":
            False,

        "public_gt_used_for_calibration":
            False,

        "spatial_conditioning":
            "category + 32x32 patch position",
    },

    "pre_registered_thresholds": {
        "shift_segf1_gain":
            0.02,

        "shift_aupro_gain":
            0.01,

        "max_regular_loss":
            0.015,
    },

    "observed": {
        "baseline_shift123_mean_segf1":
            float(
                baseline_shift_f1
            ),

        "occ_shift123_mean_segf1":
            float(
                occ_shift_f1
            ),

        "shift_segf1_gain":
            float(
                shift_f1_gain
            ),

        "regular_segf1_delta":
            float(
                regular_f1_delta
            ),

        "baseline_shift123_mean_aupro":
            float(
                baseline_shift_pro
            ),

        "occ_shift123_mean_aupro":
            float(
                occ_shift_pro
            ),

        "shift_aupro_gain":
            float(
                shift_pro_gain
            ),

        "regular_aupro_delta":
            float(
                regular_pro_delta
            ),
    },

    "signals": {
        "segf1_signal":
            bool(
                seg_signal
            ),

        "aupro_signal":
            bool(
                pro_signal
            ),
    },

    "decision":
        decision,
}


with SUMMARY_JSON.open(
    "w"
) as f:

    json.dump(
        payload,
        f,
        indent=2,
    )


# ============================================================
# Console
# ============================================================

print()
print("=" * 110)
print("A4.0 OCC CONDITION RESULTS")
print("=" * 110)

print(
    f"{'method':18s}"
    f"{'condition':16s}"
    f"{'Pix-AUC':>10s}"
    f"{'PRO05':>10s}"
    f"{'SegF1':>10s}"
)

print("-" * 64)


for row in condition_rows:

    print(
        f"{row['method']:18s}"
        f"{row['condition']:16s}"
        f"{row['pixel_auroc']:10.4f}"
        f"{row['aupro_0_05']:10.4f}"
        f"{row['segf1']:10.4f}"
    )


print()
print("[PRIMARY OCC GO / NO-GO]")

print(
    "shift123 SegF1 baseline : "
    f"{baseline_shift_f1:.4f}"
)

print(
    "shift123 SegF1 OCC      : "
    f"{occ_shift_f1:.4f}"
)

print(
    "shift SegF1 gain        : "
    f"{shift_f1_gain:+.4f}"
)

print(
    "regular SegF1 delta     : "
    f"{regular_f1_delta:+.4f}"
)

print()
print(
    "shift123 PRO baseline   : "
    f"{baseline_shift_pro:.4f}"
)

print(
    "shift123 PRO OCC        : "
    f"{occ_shift_pro:.4f}"
)

print(
    "shift PRO gain          : "
    f"{shift_pro_gain:+.4f}"
)

print(
    "regular PRO delta       : "
    f"{regular_pro_delta:+.4f}"
)

print()
print(
    "SegF1 signal            :",
    seg_signal
)

print(
    "AU-PRO signal           :",
    pro_signal
)

print()
print("DECISION :", decision)

print()
print("DETAIL     :", DETAIL_CSV)
print("CONDITIONS :", CONDITION_CSV)
print("THRESHOLDS :", THRESHOLD_JSON)
print("SUMMARY    :", SUMMARY_JSON)

print("=" * 110)
