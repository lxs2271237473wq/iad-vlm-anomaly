from pathlib import Path
from collections import defaultdict
import csv
import json

import cv2
import numpy as np


DATA_ROOT = Path(
    "/root/private_data/iad-vlm-anomaly/datasets/MVTec_AD_2"
)

A1_ROOT = Path(
    "orbitad/results/a1_anomaly_score_sensitivity"
)

ROOT = Path(
    "orbitad/results/a5_orthogonal_scoring"
)

PUBLIC_META = (
    A1_ROOT / "public_patch_maps.csv"
)

VAL_META = (
    A1_ROOT / "validation_patch_maps.csv"
)


METHODS = [
    "raw",
    "random16",
    "normalpca16",
    "wrongcat",
    "ant",
]

PRIMARY_SHIFTS = [
    "shift_1",
    "shift_2",
    "shift_3",
]

EVAL_SIZE = 512
VAL_Q = 0.995
MAX_FPR = 0.05

HIST_BINS = 65536

# All three distances are 0.5 * squared Euclidean
# distances between normalized/projected features.
# Orthogonal projection cannot increase pair distance.
EDGES = np.linspace(
    0.0,
    2.0,
    HIST_BINS + 1,
    dtype=np.float64,
)


DETAIL_CSV = (
    ROOT /
    "a5_2_category_condition_metrics.csv"
)

CONDITION_CSV = (
    ROOT /
    "a5_2_condition_metrics.csv"
)

SUMMARY_CSV = (
    ROOT /
    "a5_2_method_summary.csv"
)

CATEGORY_CSV = (
    ROOT /
    "a5_2_category_shift_gains.csv"
)

THRESHOLD_JSON = (
    ROOT /
    "a5_2_validation_thresholds.json"
)

SUMMARY_JSON = (
    ROOT /
    "a5_2_intermediate_summary.json"
)


# ============================================================
# Metadata
# ============================================================

def read_meta(path):

    rows = []

    with path.open() as f:

        for r in csv.DictReader(f):

            r["map_index"] = int(
                r["map_index"]
            )

            if "label" in r:
                r["label"] = int(
                    r["label"]
                )

            rows.append(r)

    return rows


public_rows = read_meta(
    PUBLIC_META
)

val_rows = read_meta(
    VAL_META
)


if len(public_rows) != 1084:
    raise RuntimeError(
        f"Expected 1084 public rows, "
        f"got {len(public_rows)}"
    )

if len(val_rows) != 302:
    raise RuntimeError(
        f"Expected 302 validation rows, "
        f"got {len(val_rows)}"
    )


# ============================================================
# Maps
# ============================================================

public_maps = {
    method: np.load(
        ROOT /
        f"public_{method}_maps.npy",
        mmap_mode="r",
    )
    for method in METHODS
}

val_maps = {
    method: np.load(
        ROOT /
        f"validation_{method}_maps.npy",
        mmap_mode="r",
    )
    for method in METHODS
}


for method in METHODS:

    if public_maps[
        method
    ].shape != (
        1084,
        32,
        32,
    ):

        raise RuntimeError(
            f"Bad public/{method}: "
            f"{public_maps[method].shape}"
        )

    if val_maps[
        method
    ].shape != (
        302,
        32,
        32,
    ):

        raise RuntimeError(
            f"Bad validation/{method}: "
            f"{val_maps[method].shape}"
        )


print("=" * 115)
print(
    "A5.1 — CANONICAL "
    "NUISANCE-ORTHOGONAL ANOMALY SCORING"
)
print("=" * 115)


# ============================================================
# Utilities
# ============================================================

def upsample(
    score32
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
    row
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


# ============================================================
# Validation-normal-only thresholds
#
# Use EXACT SAME evaluator support and full validation/good.
# Public GT never participates.
# ============================================================

val_hist = defaultdict(
    lambda: np.zeros(
        HIST_BINS,
        dtype=np.int64,
    )
)


for n, row in enumerate(
    val_rows,
    1,
):

    category = row[
        "category"
    ]

    for method in METHODS:

        score = upsample(
            val_maps[
                method
            ][
                row[
                    "map_index"
                ]
            ]
        )

        h, _ = np.histogram(
            score,
            bins=EDGES,
        )

        val_hist[
            (
                method,
                category,
            )
        ] += h


def hist_quantile(
    hist,
    q
):

    total = hist.sum()

    if total <= 0:
        raise RuntimeError(
            "Empty validation histogram"
        )

    target = (
        q * total
    )

    cdf = np.cumsum(
        hist
    )

    idx = int(
        np.searchsorted(
            cdf,
            target,
            side="left",
        )
    )

    idx = min(
        max(
            idx,
            0,
        ),
        HIST_BINS - 1,
    )

    # Bin midpoint.
    return float(
        0.5
        * (
            EDGES[idx]
            +
            EDGES[idx + 1]
        )
    )


thresholds = {
    key: hist_quantile(
        hist,
        VAL_Q,
    )
    for key, hist
    in val_hist.items()
}


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
            ), value
            in thresholds.items()
        },
        f,
        indent=2,
    )


print()
print("[VALIDATION NORMAL THRESHOLDS]")

for category in sorted({
    r["category"]
    for r in val_rows
}):

    print(
        f"{category:15s} "
        f"raw={thresholds[('raw', category)]:.6f} "
        f"random16={thresholds[('random16', category)]:.6f} "
        f"ant={thresholds[('ant', category)]:.6f}"
    )


# ============================================================
# Public statistics
# ============================================================

class Stats:

    def __init__(self):

        self.pos = np.zeros(
            HIST_BINS,
            dtype=np.int64,
        )

        self.neg = np.zeros(
            HIST_BINS,
            dtype=np.int64,
        )

        self.region = np.zeros(
            HIST_BINS,
            dtype=np.float64,
        )

        self.regions = 0

        self.tp = 0
        self.fp = 0
        self.fn = 0


stats = defaultdict(
    Stats
)


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

    mask = load_mask(
        row
    )

    positive = (
        mask > 0
    )

    negative = ~positive


    for method in METHODS:

        score = upsample(
            public_maps[
                method
            ][
                row[
                    "map_index"
                ]
            ]
        )

        g = stats[
            (
                method,
                category,
                condition,
            )
        ]


        # ----------------------------------------
        # Pixel histograms
        # ----------------------------------------

        hp, _ = np.histogram(
            score[
                positive
            ],
            bins=EDGES,
        )

        hn, _ = np.histogram(
            score[
                negative
            ],
            bins=EDGES,
        )

        g.pos += hp
        g.neg += hn


        # ----------------------------------------
        # Region PRO
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

                if len(values) == 0:
                    continue

                h, _ = np.histogram(
                    values,
                    bins=EDGES,
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
        # Segmentation F1
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
# Metrics
# ============================================================

def pixel_auc(
    g
):

    P = int(
        g.pos.sum()
    )

    N = int(
        g.neg.sum()
    )

    if (
        P == 0
        or
        N == 0
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
    g
):

    N = int(
        g.neg.sum()
    )

    if (
        N == 0
        or
        g.regions == 0
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
                +
                alpha
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
    g
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
# Category × condition
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
# Macro method × condition
# ============================================================

groups = defaultdict(
    list
)


for row in detail_rows:

    groups[
        (
            row["method"],
            row["condition"],
        )
    ].append(
        row
    )


def mean_finite(
    values
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
    groups.items()
):

    condition_rows.append({
        "method":
            method,

        "condition":
            condition,

        "categories":
            len(
                members
            ),

        "pixel_auroc":
            mean_finite([
                x["pixel_auroc"]
                for x in members
            ]),

        "aupro_0_05":
            mean_finite([
                x["aupro_0_05"]
                for x in members
            ]),

        "segf1":
            mean_finite([
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


def shift_mean(
    method,
    metric
):

    values = []

    for condition in PRIMARY_SHIFTS:

        key = (
            method,
            condition,
        )

        if key not in lookup:
            continue

        value = lookup[
            key
        ][
            metric
        ]

        if np.isfinite(
            value
        ):

            values.append(
                value
            )

    return float(
        np.mean(
            values
        )
    )


# ============================================================
# Method summary
# ============================================================

summary_rows = []


for method in METHODS:

    regular = lookup[
        (
            method,
            "regular",
        )
    ]

    summary_rows.append({
        "method":
            method,

        "regular_pixel_auc":
            regular[
                "pixel_auroc"
            ],

        "shift123_pixel_auc":
            shift_mean(
                method,
                "pixel_auroc",
            ),

        "regular_aupro":
            regular[
                "aupro_0_05"
            ],

        "shift123_aupro":
            shift_mean(
                method,
                "aupro_0_05",
            ),

        "regular_segf1":
            regular[
                "segf1"
            ],

        "shift123_segf1":
            shift_mean(
                method,
                "segf1",
            ),
    })


summary = {
    r["method"]: r
    for r in summary_rows
}


# ============================================================
# RAW SANITY CHECK
#
# Canonical raw anchors from previous evaluator.
# Small fp16/histogram tolerance is allowed.
# ============================================================

RAW_ANCHOR = {
    "regular_pixel_auc":
        0.8939,

    "shift123_pixel_auc":
        0.8748,

    "regular_aupro":
        0.2257,

    "shift123_aupro":
        0.2161,
}

RAW_TOL = 0.0020


raw_sanity = {}


for key, expected in (
    RAW_ANCHOR.items()
):

    observed = summary[
        "raw"
    ][
        key
    ]

    error = abs(
        observed
        -
        expected
    )

    raw_sanity[
        key
    ] = {
        "expected":
            expected,

        "observed":
            observed,

        "absolute_error":
            error,

        "pass":
            bool(
                error <= RAW_TOL
            ),
    }


raw_sanity_pass = all(
    x["pass"]
    for x in raw_sanity.values()
)


if not raw_sanity_pass:

    print()
    print(
        "RAW SANITY FAILURE"
    )

    for key, info in (
        raw_sanity.items()
    ):

        print(
            key,
            info
        )

    raise RuntimeError(
        "Generated RAW maps do not reproduce "
        "the canonical raw detector. "
        "Do not interpret ANT results."
    )


# ============================================================
# Gains
# ============================================================

raw = summary[
    "raw"
]

ant = summary[
    "ant"
]

rnd = summary[
    "random16"
]


ant_shift_pro_gain = (
    ant[
        "shift123_aupro"
    ]
    -
    raw[
        "shift123_aupro"
    ]
)

ant_regular_pro_delta = (
    ant[
        "regular_aupro"
    ]
    -
    raw[
        "regular_aupro"
    ]
)


random_shift_pro_gain = (
    rnd[
        "shift123_aupro"
    ]
    -
    raw[
        "shift123_aupro"
    ]
)


ant_shift_f1_gain = (
    ant[
        "shift123_segf1"
    ]
    -
    raw[
        "shift123_segf1"
    ]
)

ant_regular_f1_delta = (
    ant[
        "regular_segf1"
    ]
    -
    raw[
        "regular_segf1"
    ]
)


random_shift_f1_gain = (
    rnd[
        "shift123_segf1"
    ]
    -
    raw[
        "shift123_segf1"
    ]
)


# ============================================================
# PRE-REGISTERED A5.1 DECISION
# ============================================================

pro_effect = (
    ant_shift_pro_gain
    >= 0.015
)

pro_regular_safe = (
    ant_regular_pro_delta
    >= -0.015
)

pro_specific = (
    ant_shift_pro_gain
    >=
    random_shift_pro_gain
    + 0.005
)


seg_effect = (
    ant_shift_f1_gain
    >= 0.020
)

seg_regular_safe = (
    ant_regular_f1_delta
    >= -0.015
)

seg_specific = (
    ant_shift_f1_gain
    >=
    random_shift_f1_gain
    + 0.005
)


pro_signal = (
    pro_effect
    and
    pro_regular_safe
    and
    pro_specific
)

seg_signal = (
    seg_effect
    and
    seg_regular_safe
    and
    seg_specific
)


decision = (
    "GO"
    if (
        pro_signal
        or
        seg_signal
    )
    else "NO_GO"
)


# ============================================================
# Category shift gains
# ============================================================

detail_lookup = {
    (
        r["method"],
        r["category"],
        r["condition"],
    ): r
    for r in detail_rows
}


categories = sorted({
    r["category"]
    for r in detail_rows
})


category_rows = []


for category in categories:

    row = {
        "category":
            category,
    }

    for metric in [
        "aupro_0_05",
        "segf1",
    ]:

        raw_values = []
        ant_values = []
        rnd_values = []

        for condition in PRIMARY_SHIFTS:

            raw_key = (
                "raw",
                category,
                condition,
            )

            ant_key = (
                "ant",
                category,
                condition,
            )

            rnd_key = (
                "random16",
                category,
                condition,
            )

            if (
                raw_key
                not in detail_lookup
            ):
                continue


            rv = detail_lookup[
                raw_key
            ][
                metric
            ]

            av = detail_lookup[
                ant_key
            ][
                metric
            ]

            nv = detail_lookup[
                rnd_key
            ][
                metric
            ]


            if (
                np.isfinite(rv)
                and
                np.isfinite(av)
                and
                np.isfinite(nv)
            ):

                raw_values.append(
                    rv
                )

                ant_values.append(
                    av
                )

                rnd_values.append(
                    nv
                )


        if raw_values:

            raw_mean = float(
                np.mean(
                    raw_values
                )
            )

            ant_mean = float(
                np.mean(
                    ant_values
                )
            )

            rnd_mean = float(
                np.mean(
                    rnd_values
                )
            )

            row[
                f"raw_shift_{metric}"
            ] = raw_mean

            row[
                f"ant_shift_{metric}"
            ] = ant_mean

            row[
                f"random_shift_{metric}"
            ] = rnd_mean

            row[
                f"ant_gain_{metric}"
            ] = (
                ant_mean
                -
                raw_mean
            )

            row[
                f"random_gain_{metric}"
            ] = (
                rnd_mean
                -
                raw_mean
            )

    category_rows.append(
        row
    )


with CATEGORY_CSV.open(
    "w",
    newline="",
) as f:

    fields = sorted({
        key
        for row in category_rows
        for key in row.keys()
    })

    # category first
    fields.remove(
        "category"
    )

    fields = [
        "category"
    ] + fields

    writer = csv.DictWriter(
        f,
        fieldnames=fields,
    )

    writer.writeheader()

    writer.writerows(
        category_rows
    )


# ============================================================
# Save summary
# ============================================================

for r in summary_rows:

    r[
        "shift_aupro_gain_vs_raw"
    ] = (
        r[
            "shift123_aupro"
        ]
        -
        raw[
            "shift123_aupro"
        ]
    )

    r[
        "regular_aupro_delta_vs_raw"
    ] = (
        r[
            "regular_aupro"
        ]
        -
        raw[
            "regular_aupro"
        ]
    )

    r[
        "shift_segf1_gain_vs_raw"
    ] = (
        r[
            "shift123_segf1"
        ]
        -
        raw[
            "shift123_segf1"
        ]
    )

    r[
        "regular_segf1_delta_vs_raw"
    ] = (
        r[
            "regular_segf1"
        ]
        -
        raw[
            "regular_segf1"
        ]
    )


with SUMMARY_CSV.open(
    "w",
    newline="",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=list(
            summary_rows[0].keys()
        ),
    )

    writer.writeheader()

    writer.writerows(
        summary_rows
    )


payload = {
    "method":
        "Acquisition Nuisance Tangent",

    "rank":
        16,

    "raw_sanity":
        raw_sanity,

    "raw_sanity_pass":
        raw_sanity_pass,

    "pre_registered": {
        "minimum_shift_aupro_gain":
            0.015,

        "maximum_regular_aupro_loss":
            0.015,

        "minimum_shift_segf1_gain":
            0.020,

        "maximum_regular_segf1_loss":
            0.015,

        "minimum_gain_over_random16":
            0.005,
    },

    "observed": {
        "ant_shift_aupro_gain":
            ant_shift_pro_gain,

        "ant_regular_aupro_delta":
            ant_regular_pro_delta,

        "random16_shift_aupro_gain":
            random_shift_pro_gain,

        "ant_minus_random_aupro_gain":
            (
                ant_shift_pro_gain
                -
                random_shift_pro_gain
            ),

        "ant_shift_segf1_gain":
            ant_shift_f1_gain,

        "ant_regular_segf1_delta":
            ant_regular_f1_delta,

        "random16_shift_segf1_gain":
            random_shift_f1_gain,

        "ant_minus_random_segf1_gain":
            (
                ant_shift_f1_gain
                -
                random_shift_f1_gain
            ),
    },

    "signals": {
        "aupro_effect":
            bool(
                pro_effect
            ),

        "aupro_regular_safe":
            bool(
                pro_regular_safe
            ),

        "aupro_specificity":
            bool(
                pro_specific
            ),

        "aupro_primary_signal":
            bool(
                pro_signal
            ),

        "segf1_effect":
            bool(
                seg_effect
            ),

        "segf1_regular_safe":
            bool(
                seg_regular_safe
            ),

        "segf1_specificity":
            bool(
                seg_specific
            ),

        "segf1_primary_signal":
            bool(
                seg_signal
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
print("=" * 115)
print("A5.2 FIVE-WAY SPECIFICITY RESULTS")
print("=" * 115)

print(
    f"{'method':12s}"
    f"{'Reg-PixAUC':>12s}"
    f"{'Shift-PixAUC':>14s}"
    f"{'Reg-PRO':>11s}"
    f"{'Shift-PRO':>12s}"
    f"{'Reg-F1':>11s}"
    f"{'Shift-F1':>11s}"
)

print("-" * 83)


for r in summary_rows:

    print(
        f"{r['method']:12s}"
        f"{r['regular_pixel_auc']:12.4f}"
        f"{r['shift123_pixel_auc']:14.4f}"
        f"{r['regular_aupro']:11.4f}"
        f"{r['shift123_aupro']:12.4f}"
        f"{r['regular_segf1']:11.4f}"
        f"{r['shift123_segf1']:11.4f}"
    )


print()
print("[RAW REPRODUCIBILITY]")

for key, info in (
    raw_sanity.items()
):

    print(
        f"{key:24s}: "
        f"expected={info['expected']:.4f} "
        f"observed={info['observed']:.4f} "
        f"error={info['absolute_error']:.6f} "
        f"PASS={info['pass']}"
    )


print()
print("[ANT VS RAW]")

print(
    "shift AU-PRO gain       :",
    f"{ant_shift_pro_gain:+.4f}"
)

print(
    "regular AU-PRO delta    :",
    f"{ant_regular_pro_delta:+.4f}"
)

print(
    "random16 shift PRO gain :",
    f"{random_shift_pro_gain:+.4f}"
)

print(
    "ANT - random PRO gain   :",
    f"{ant_shift_pro_gain-random_shift_pro_gain:+.4f}"
)


print()

print(
    "shift SegF1 gain        :",
    f"{ant_shift_f1_gain:+.4f}"
)

print(
    "regular SegF1 delta     :",
    f"{ant_regular_f1_delta:+.4f}"
)

print(
    "random16 shift F1 gain  :",
    f"{random_shift_f1_gain:+.4f}"
)

print(
    "ANT - random F1 gain    :",
    f"{ant_shift_f1_gain-random_shift_f1_gain:+.4f}"
)


print()
print("[CATEGORY SHIFT AU-PRO GAINS]")

for r in category_rows:

    ant_gain = r.get(
        "ant_gain_aupro_0_05",
        np.nan,
    )

    rnd_gain = r.get(
        "random_gain_aupro_0_05",
        np.nan,
    )

    print(
        f"{r['category']:15s} "
        f"ANT={ant_gain:+.4f} "
        f"Random16={rnd_gain:+.4f}"
    )


print()
print("[PRE-REGISTERED A5.1 GO / NO-GO]")

print(
    "AU-PRO effect >= +0.015 :",
    pro_effect
)

print(
    "AU-PRO regular safe     :",
    pro_regular_safe
)

print(
    "AU-PRO > random +.005   :",
    pro_specific
)

print(
    "AU-PRO PRIMARY SIGNAL   :",
    pro_signal
)

print()

print(
    "SegF1 effect >= +0.020  :",
    seg_effect
)

print(
    "SegF1 regular safe      :",
    seg_regular_safe
)

print(
    "SegF1 > random +.005    :",
    seg_specific
)

print(
    "SegF1 PRIMARY SIGNAL    :",
    seg_signal
)


print()
print(
    "DECISION :",
    decision
)

print()
print(
    "DETAIL     :",
    DETAIL_CSV
)

print(
    "CONDITIONS :",
    CONDITION_CSV
)

print(
    "SUMMARY    :",
    SUMMARY_CSV
)

print(
    "CATEGORY   :",
    CATEGORY_CSV
)

print(
    "THRESHOLDS :",
    THRESHOLD_JSON
)

print(
    "JSON       :",
    SUMMARY_JSON
)

print("=" * 115)
