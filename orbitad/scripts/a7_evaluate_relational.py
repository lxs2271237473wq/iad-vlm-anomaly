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

A5_SCORE_ROOT = Path(
    "orbitad/results/a5_orthogonal_scoring"
)

A5_CONTROL_ROOT = Path(
    "orbitad/results/a5_tangent_specificity"
)

A7_ROOT = Path(
    "orbitad/results/a7_relational"
)

PUBLIC_META = (
    A1_ROOT / "public_patch_maps.csv"
)


METHOD_PATHS = {
    "raw":
        A5_SCORE_ROOT
        / "public_raw_maps.npy",

    "pca16":
        A5_CONTROL_ROOT
        / "public_normalpca16_maps.npy",

    "relation":
        A7_ROOT
        / "public_relation_maps.npy",
}


METHODS = [
    "raw",
    "pca16",
    "relation",
]


PRIMARY_SHIFTS = [
    "shift_1",
    "shift_2",
    "shift_3",
]


EVAL_SIZE = 512
MAX_FPR = 0.05
HIST_BINS = 65536


DETAIL_CSV = (
    A7_ROOT
    / "a7_1_category_condition_metrics.csv"
)

CATEGORY_CSV = (
    A7_ROOT
    / "a7_1_category_shift_metrics.csv"
)

SUMMARY_CSV = (
    A7_ROOT
    / "a7_1_method_summary.csv"
)

SUMMARY_JSON = (
    A7_ROOT
    / "a7_1_decision.json"
)


# ============================================================
# Metadata
# ============================================================

rows = []

with PUBLIC_META.open() as f:

    for r in csv.DictReader(f):

        r["map_index"] = int(
            r["map_index"]
        )

        r["label"] = int(
            r["label"]
        )

        rows.append(r)


if len(rows) != 1084:

    raise RuntimeError(
        f"Expected 1084 rows, "
        f"got {len(rows)}"
    )


CATEGORIES = sorted({
    r["category"]
    for r in rows
})


# ============================================================
# Maps
# ============================================================

maps = {}


for method, path in METHOD_PATHS.items():

    x = np.load(
        path,
        mmap_mode="r",
    )

    if x.shape != (
        1084,
        32,
        32,
    ):

        raise RuntimeError(
            f"{method}: bad shape "
            f"{x.shape}"
        )

    if not np.isfinite(
        x
    ).all():

        raise RuntimeError(
            f"{method}: non-finite"
        )

    maps[
        method
    ] = x


# ============================================================
# Method-specific histogram support
# ============================================================

EDGES = {}


print("=" * 112)
print("A7.1 — RELATIONAL MEMORY CANONICAL EVALUATION")
print("=" * 112)

print()
print("[HISTOGRAM SUPPORT]")


for method in METHODS:

    observed_max = float(
        np.max(
            maps[
                method
            ]
        )
    )

    upper = max(
        observed_max
        * 1.001,
        observed_max
        + 1e-6,
        1e-5,
    )

    EDGES[
        method
    ] = np.linspace(
        0.0,
        upper,
        HIST_BINS + 1,
        dtype=np.float64,
    )

    print(
        f"{method:10s}: "
        f"max={observed_max:.6f} "
        f"support={upper:.6f}"
    )


# ============================================================
# Utilities
# ============================================================

def upsample(
    x
):

    return cv2.resize(
        np.asarray(
            x,
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
        / row[
            "mask_path"
        ]
    )


    mask = cv2.imread(
        str(path),
        cv2.IMREAD_GRAYSCALE,
    )


    if mask is None:

        raise RuntimeError(
            f"Cannot read mask: "
            f"{path}"
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
# AU-PRO statistics
# ============================================================

class Stats:

    def __init__(
        self,
        edges,
    ):

        bins = (
            len(edges)
            - 1
        )

        self.edges = edges

        self.neg = np.zeros(
            bins,
            dtype=np.int64,
        )

        self.region = np.zeros(
            bins,
            dtype=np.float64,
        )

        self.regions = 0


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

        stats[
            key
        ] = Stats(
            EDGES[
                method
            ]
        )

    return stats[
        key
    ]


# ============================================================
# Public evaluation
# ============================================================

for n, row in enumerate(
    rows,
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
            maps[
                method
            ][
                row[
                    "map_index"
                ]
            ]
        )


        g = get_stats(
            method,
            category,
            condition,
        )


        # False-positive distribution
        hn, _ = np.histogram(
            score[
                negative
            ],
            bins=g.edges,
        )

        g.neg += hn


        # Region overlap distributions
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
                    len(
                        values
                    )
                )


                g.regions += 1


    if (
        n % 100 == 0
        or
        n == len(rows)
    ):

        print(
            f"processed "
            f"{n}/{len(rows)}"
        )


# ============================================================
# AU-PRO @ FPR <= 0.05
# ============================================================

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
                -
                x0
            ) / max(
                x1
                -
                x0,
                1e-12,
            )


            y_at = (
                y0
                +
                alpha
                *
                (
                    y1
                    -
                    y0
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

        "aupro_0_05":
            aupro(
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
            detail_rows[
                0
            ].keys()
        ),
    )

    writer.writeheader()

    writer.writerows(
        detail_rows
    )


lookup = {
    (
        r["method"],
        r["category"],
        r["condition"],
    ):
        r[
            "aupro_0_05"
        ]

    for r in detail_rows
}


# ============================================================
# Canonical macro:
#
# regular:
#   mean over categories
#
# shift:
#   first mean over categories within each
#   condition, then mean shift_1/2/3.
#
# This is exactly the canonical A5/A6 weighting.
# ============================================================

def macro_regular(
    method
):

    values = []


    for category in CATEGORIES:

        key = (
            method,
            category,
            "regular",
        )


        if key not in lookup:
            continue


        value = lookup[
            key
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


def macro_shift(
    method
):

    condition_values = []


    for condition in PRIMARY_SHIFTS:

        category_values = []


        for category in CATEGORIES:

            key = (
                method,
                category,
                condition,
            )


            if key not in lookup:
                continue


            value = lookup[
                key
            ]


            if np.isfinite(
                value
            ):

                category_values.append(
                    value
                )


        if category_values:

            condition_values.append(
                float(
                    np.mean(
                        category_values
                    )
                )
            )


    return float(
        np.mean(
            condition_values
        )
    )


# ============================================================
# Category-level shift mean
# ============================================================

category_rows = []


for category in CATEGORIES:

    row = {
        "category":
            category,
    }


    for method in METHODS:

        values = []


        for condition in PRIMARY_SHIFTS:

            key = (
                method,
                category,
                condition,
            )


            if key not in lookup:
                continue


            value = lookup[
                key
            ]


            if np.isfinite(
                value
            ):

                values.append(
                    value
                )


        row[
            f"{method}_shift_aupro"
        ] = (
            float(
                np.mean(
                    values
                )
            )
            if values
            else np.nan
        )


    row[
        "relation_minus_raw"
    ] = (
        row[
            "relation_shift_aupro"
        ]
        -
        row[
            "raw_shift_aupro"
        ]
    )


    row[
        "relation_minus_pca"
    ] = (
        row[
            "relation_shift_aupro"
        ]
        -
        row[
            "pca16_shift_aupro"
        ]
    )


    row[
        "relation_beats_raw"
    ] = bool(
        row[
            "relation_shift_aupro"
        ]
        >
        row[
            "raw_shift_aupro"
        ]
    )


    category_rows.append(
        row
    )


with CATEGORY_CSV.open(
    "w",
    newline="",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=list(
            category_rows[
                0
            ].keys()
        ),
    )

    writer.writeheader()

    writer.writerows(
        category_rows
    )


# ============================================================
# Summary
# ============================================================

summary_rows = []


for method in METHODS:

    summary_rows.append({
        "method":
            method,

        "regular_aupro":
            macro_regular(
                method
            ),

        "shift123_aupro":
            macro_shift(
                method
            ),
    })


with SUMMARY_CSV.open(
    "w",
    newline="",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=list(
            summary_rows[
                0
            ].keys()
        ),
    )

    writer.writeheader()

    writer.writerows(
        summary_rows
    )


summary = {
    r["method"]: r
    for r in summary_rows
}


# ============================================================
# Reproducibility anchors
# ============================================================

ANCHORS = {
    "raw": {
        "regular_aupro":
            0.2256,

        "shift123_aupro":
            0.2161,
    },

    "pca16": {
        "regular_aupro":
            0.2905,

        "shift123_aupro":
            0.2774,
    },
}


TOL = 0.002


sanity = {}


for method, expected in (
    ANCHORS.items()
):

    sanity[
        method
    ] = {}


    for metric, target in (
        expected.items()
    ):

        observed = summary[
            method
        ][
            metric
        ]


        error = abs(
            observed
            -
            target
        )


        sanity[
            method
        ][
            metric
        ] = {
            "expected":
                target,

            "observed":
                observed,

            "error":
                error,

            "pass":
                bool(
                    error <= TOL
                ),
        }


sanity_pass = all(
    item[
        "pass"
    ]

    for method in sanity.values()

    for item in method.values()
)


if not sanity_pass:

    print()
    print("[SANITY FAILURE]")

    print(
        json.dumps(
            sanity,
            indent=2,
        )
    )

    raise RuntimeError(
        "Raw/PCA16 reproduction failed. "
        "Do not interpret relational results."
    )


# ============================================================
# Pre-registered A7.1 decision
# ============================================================

raw = summary[
    "raw"
]

pca = summary[
    "pca16"
]

relation = summary[
    "relation"
]


shift_gain_raw = (
    relation[
        "shift123_aupro"
    ]
    -
    raw[
        "shift123_aupro"
    ]
)


regular_delta_raw = (
    relation[
        "regular_aupro"
    ]
    -
    raw[
        "regular_aupro"
    ]
)


pca_gap = (
    relation[
        "shift123_aupro"
    ]
    -
    pca[
        "shift123_aupro"
    ]
)


category_wins = sum(
    r[
        "relation_beats_raw"
    ]
    for r in category_rows
)


signal_shift_gain = (
    shift_gain_raw
    >= 0.030
)


signal_regular_safe = (
    regular_delta_raw
    >= -0.015
)


signal_category_consistency = (
    category_wins
    >= 6
)


signal_pca_competitive = (
    pca_gap
    >= -0.010
)


decision = (
    "GO"
    if (
        signal_shift_gain
        and
        signal_regular_safe
        and
        signal_category_consistency
        and
        signal_pca_competitive
    )
    else "NO_GO"
)


payload = {
    "pre_registered": {
        "minimum_shift_gain_vs_raw":
            0.030,

        "maximum_regular_loss_vs_raw":
            0.015,

        "minimum_categories_beating_raw":
            6,

        "maximum_shift_gap_vs_pca16":
            0.010,
    },

    "observed": {
        "raw_regular_aupro":
            raw[
                "regular_aupro"
            ],

        "raw_shift_aupro":
            raw[
                "shift123_aupro"
            ],

        "pca16_regular_aupro":
            pca[
                "regular_aupro"
            ],

        "pca16_shift_aupro":
            pca[
                "shift123_aupro"
            ],

        "relation_regular_aupro":
            relation[
                "regular_aupro"
            ],

        "relation_shift_aupro":
            relation[
                "shift123_aupro"
            ],

        "relation_minus_raw_shift":
            shift_gain_raw,

        "relation_minus_raw_regular":
            regular_delta_raw,

        "relation_minus_pca_shift":
            pca_gap,

        "categories_beating_raw":
            int(
                category_wins
            ),
    },

    "signals": {
        "shift_gain":
            bool(
                signal_shift_gain
            ),

        "regular_safe":
            bool(
                signal_regular_safe
            ),

        "category_consistency":
            bool(
                signal_category_consistency
            ),

        "pca_competitive":
            bool(
                signal_pca_competitive
            ),
    },

    "sanity":
        sanity,

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
print("=" * 112)
print("A7.1 CANONICAL RESULTS")
print("=" * 112)

print(
    f"{'method':12s}"
    f"{'Reg-PRO':>12s}"
    f"{'Shift-PRO':>12s}"
    f"{'vs-Raw':>12s}"
    f"{'vs-PCA':>12s}"
)

print("-" * 60)


for method in METHODS:

    r = summary[
        method
    ]


    vs_raw = (
        r[
            "shift123_aupro"
        ]
        -
        raw[
            "shift123_aupro"
        ]
    )


    vs_pca = (
        r[
            "shift123_aupro"
        ]
        -
        pca[
            "shift123_aupro"
        ]
    )


    print(
        f"{method:12s}"
        f"{r['regular_aupro']:12.4f}"
        f"{r['shift123_aupro']:12.4f}"
        f"{vs_raw:+12.4f}"
        f"{vs_pca:+12.4f}"
    )


print()
print("[CATEGORY SHIFT AU-PRO]")

print(
    f"{'category':15s}"
    f"{'Raw':>10s}"
    f"{'PCA16':>10s}"
    f"{'Relation':>11s}"
    f"{'Rel-Raw':>10s}"
    f"{'Rel-PCA':>10s}"
)

print("-" * 66)


for r in category_rows:

    print(
        f"{r['category']:15s}"
        f"{r['raw_shift_aupro']:10.4f}"
        f"{r['pca16_shift_aupro']:10.4f}"
        f"{r['relation_shift_aupro']:11.4f}"
        f"{r['relation_minus_raw']:+10.4f}"
        f"{r['relation_minus_pca']:+10.4f}"
    )


print()
print("[PRE-REGISTERED A7.1 GO / NO-GO]")

print(
    "Relation > Raw shift by >= .030 :",
    signal_shift_gain
)

print(
    "Relation regular within -.015   :",
    signal_regular_safe
)

print(
    "Relation beats Raw in >=6/8     :",
    signal_category_consistency,
    f"({category_wins}/8)"
)

print(
    "Relation within .010 of PCA16   :",
    signal_pca_competitive
)

print()
print(
    "DECISION :",
    decision
)

print()
print(
    "SUMMARY :",
    SUMMARY_JSON
)

print("=" * 112)
