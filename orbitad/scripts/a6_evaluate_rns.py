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

A6_ROOT = Path(
    "orbitad/results/a6_rns"
)

PUBLIC_META = (
    A1_ROOT / "public_patch_maps.csv"
)

OUT_DETAIL = (
    A6_ROOT / "a6_0_category_condition_metrics.csv"
)

OUT_CATEGORY = (
    A6_ROOT / "a6_0_category_shift_metrics.csv"
)

OUT_SUMMARY = (
    A6_ROOT / "a6_0_method_summary.csv"
)

OUT_JSON = (
    A6_ROOT / "a6_0_decision.json"
)


METHOD_PATHS = {
    "raw":
        A5_SCORE_ROOT
        / "public_raw_maps.npy",

    "pca16":
        A5_CONTROL_ROOT
        / "public_normalpca16_maps.npy",

    "nws32":
        A6_ROOT
        / "public_nws32_maps.npy",

    "rns32":
        A6_ROOT
        / "public_rns32_maps.npy",
}


METHODS = [
    "raw",
    "pca16",
    "nws32",
    "rns32",
]


PRIMARY_SHIFTS = [
    "shift_1",
    "shift_2",
    "shift_3",
]

EVAL_SIZE = 512
MAX_FPR = 0.05
HIST_BINS = 65536


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
        f"Expected 1084 rows, got {len(rows)}"
    )


CATEGORIES = sorted({
    r["category"]
    for r in rows
})


# ============================================================
# Load maps
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
            f"{method}: bad shape {x.shape}"
        )

    if not np.isfinite(x).all():

        raise RuntimeError(
            f"{method}: non-finite values"
        )

    maps[
        method
    ] = x


# ============================================================
# Method-specific histogram support
#
# Avoid assuming transformed metric distances <= 2.
# ============================================================

EDGES = {}


print("=" * 110)
print("A6.0 — RNS CANONICAL EVALUATION")
print("=" * 110)

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
        1e-4,
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
        f"max={observed_max:.6f}, "
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
# AU-PRO statistics
# ============================================================

class Stats:

    def __init__(
        self,
        edges
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
# Evaluate
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


        # Region overlap distribution
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


    # Interpolate exact FPR=0.05 endpoint
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
# Category-condition results
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


with OUT_DETAIL.open(
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
# Category shift means
# ============================================================

category_rows = []


for category in CATEGORIES:

    row = {
        "category":
            category,
    }


    for method in METHODS:

        values = []


        for condition in (
            PRIMARY_SHIFTS
        ):

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


        if values:

            row[
                f"{method}_shift_aupro"
            ] = float(
                np.mean(
                    values
                )
            )

        else:

            row[
                f"{method}_shift_aupro"
            ] = np.nan


    row[
        "rns_minus_pca"
    ] = (
        row[
            "rns32_shift_aupro"
        ]
        -
        row[
            "pca16_shift_aupro"
        ]
    )


    row[
        "rns_ge_pca"
    ] = bool(
        row[
            "rns32_shift_aupro"
        ]
        >=
        row[
            "pca16_shift_aupro"
        ]
    )


    category_rows.append(
        row
    )


with OUT_CATEGORY.open(
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
# Macro summaries
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


        if key in lookup:

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

    # Canonical A5 weighting:
    #
    # 1. Macro-average categories separately
    #    for each acquisition condition.
    #
    # 2. Average the three condition-level
    #    macro scores.
    #
    # This matters because AD2 categories do
    # not all contain the same set/number of
    # shift conditions.

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


with OUT_SUMMARY.open(
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
# Reproducibility check
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
            0.2775,
    },
}


SANITY_TOL = 0.002


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

            "absolute_error":
                error,

            "pass":
                bool(
                    error
                    <=
                    SANITY_TOL
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
    print(
        "SANITY CHECK FAILED"
    )

    print(
        json.dumps(
            sanity,
            indent=2,
        )
    )

    raise RuntimeError(
        "Raw/PCA16 reproduction failed. "
        "Do not interpret RNS."
    )


# ============================================================
# PRE-REGISTERED GO / NO-GO
# ============================================================

pca = summary[
    "pca16"
]

nws = summary[
    "nws32"
]

rns = summary[
    "rns32"
]


shift_margin_pca = (
    rns[
        "shift123_aupro"
    ]
    -
    pca[
        "shift123_aupro"
    ]
)


regular_margin_pca = (
    rns[
        "regular_aupro"
    ]
    -
    pca[
        "regular_aupro"
    ]
)


shift_margin_nws = (
    rns[
        "shift123_aupro"
    ]
    -
    nws[
        "shift123_aupro"
    ]
)


category_wins = sum(
    r[
        "rns_ge_pca"
    ]
    for r in category_rows
)


signal_shift = (
    shift_margin_pca
    >= 0.010
)


signal_regular = (
    regular_margin_pca
    >= -0.010
)


signal_relative_spectrum = (
    shift_margin_nws
    >= 0.005
)


signal_categories = (
    category_wins
    >= 5
)


decision = (
    "GO"
    if (
        signal_shift
        and
        signal_regular
        and
        signal_relative_spectrum
        and
        signal_categories
    )
    else "NO_GO"
)


payload = {
    "protocol": {
        "beta":
            1.0,

        "rank_search":
            False,

        "public_gt_used_for_tuning":
            False,

        "metric":
            "AU-PRO@FPR<=0.05",
    },

    "sanity":
        sanity,

    "pre_registered": {
        "minimum_shift_margin_vs_pca16":
            0.010,

        "minimum_regular_margin_vs_pca16":
            -0.010,

        "minimum_shift_margin_vs_nws32":
            0.005,

        "minimum_categories_not_worse_than_pca":
            5,
    },

    "observed": {
        "raw_regular_aupro":
            summary[
                "raw"
            ][
                "regular_aupro"
            ],

        "raw_shift_aupro":
            summary[
                "raw"
            ][
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

        "nws32_regular_aupro":
            nws[
                "regular_aupro"
            ],

        "nws32_shift_aupro":
            nws[
                "shift123_aupro"
            ],

        "rns32_regular_aupro":
            rns[
                "regular_aupro"
            ],

        "rns32_shift_aupro":
            rns[
                "shift123_aupro"
            ],

        "rns_minus_pca_shift":
            shift_margin_pca,

        "rns_minus_pca_regular":
            regular_margin_pca,

        "rns_minus_nws_shift":
            shift_margin_nws,

        "categories_rns_ge_pca":
            int(
                category_wins
            ),
    },

    "signals": {
        "shift_vs_pca":
            bool(
                signal_shift
            ),

        "regular_safe":
            bool(
                signal_regular
            ),

        "relative_spectrum_value":
            bool(
                signal_relative_spectrum
            ),

        "category_robustness":
            bool(
                signal_categories
            ),
    },

    "decision":
        decision,
}


with OUT_JSON.open(
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
print("A6.0 CANONICAL RESULTS")
print("=" * 110)

print(
    f"{'method':12s}"
    f"{'Reg-PRO':>12s}"
    f"{'Shift-PRO':>12s}"
    f"{'vs-PCA':>12s}"
)

print("-" * 48)


for method in METHODS:

    r = summary[
        method
    ]


    margin = (
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
        f"{margin:+12.4f}"
    )


print()
print("[CATEGORY SHIFT AU-PRO]")

print(
    f"{'category':15s}"
    f"{'PCA16':>10s}"
    f"{'NWS32':>10s}"
    f"{'RNS32':>10s}"
    f"{'RNS-PCA':>11s}"
)

print("-" * 56)


for r in category_rows:

    print(
        f"{r['category']:15s}"
        f"{r['pca16_shift_aupro']:10.4f}"
        f"{r['nws32_shift_aupro']:10.4f}"
        f"{r['rns32_shift_aupro']:10.4f}"
        f"{r['rns_minus_pca']:+11.4f}"
    )


print()
print("[PRE-REGISTERED A6.0 GO / NO-GO]")

print(
    "RNS > PCA shift by >= .010 :",
    signal_shift
)

print(
    "RNS regular within -.010   :",
    signal_regular
)

print(
    "RNS > NWS shift by >= .005 :",
    signal_relative_spectrum
)

print(
    "RNS >= PCA in >=5/8 cats   :",
    signal_categories,
    f"({category_wins}/8)"
)

print()
print(
    "DECISION :",
    decision
)

print()
print(
    "SUMMARY :",
    OUT_JSON
)

print("=" * 110)
