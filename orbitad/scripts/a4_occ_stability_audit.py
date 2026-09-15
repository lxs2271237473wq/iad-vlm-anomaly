from pathlib import Path
from collections import defaultdict
import csv
import json
import math

import cv2
import numpy as np


DATA_ROOT = Path(
    "/root/private_data/iad-vlm-anomaly/datasets/MVTec_AD_2"
)

A2_ROOT = Path(
    "orbitad/results/a2_counterfactual_persistence"
)

OUT_ROOT = Path(
    "orbitad/results/a4_occ/stability"
)

OUT_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)


PUBLIC_MAPS = np.load(
    A2_ROOT / "public_orbit_maps.npy",
    mmap_mode="r",
)

VAL_MAPS = np.load(
    A2_ROOT / "validation_orbit_maps.npy",
    mmap_mode="r",
)

PUBLIC_META = (
    A2_ROOT / "public_orbit_meta.csv"
)

VAL_META = (
    A2_ROOT / "validation_orbit_meta.csv"
)

SPLIT_CSV = (
    OUT_ROOT / "split_stability.csv"
)

CATEGORY_CSV = (
    OUT_ROOT / "category_shift_gain.csv"
)

SUMMARY_JSON = (
    OUT_ROOT / "a4_2_summary.json"
)


SEEDS = [
    20260913,
    20260914,
    20260915,
    20260916,
    20260917,
    20260918,
    20260919,
    20260920,
    20260921,
    20260922,
]

SHIFT_CONDITIONS = {
    "shift_1",
    "shift_2",
    "shift_3",
}

EVAL_SIZE = 512
MAX_FPR = 0.05
HIST_BINS = 65536


def read_meta(path):

    rows = []

    with path.open() as f:

        for r in csv.DictReader(f):

            r["orbit_index"] = int(
                r["orbit_index"]
            )

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


CATEGORIES = sorted({
    r["category"]
    for r in val_rows
})


# ============================================================
# Fixed RAW metrics — independent of calibration split
# ============================================================

RAW_EDGES = np.linspace(
    0.0,
    2.0,
    HIST_BINS + 1,
)


class Stats:

    def __init__(
        self,
        edges,
    ):

        bins = (
            len(edges) - 1
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


def load_mask(row):

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
            str(path)
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


def upsample(x):

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


def add_score(
    stat,
    score32,
    mask,
):

    score = upsample(
        score32
    )

    positive = (
        mask > 0
    )

    negative = ~positive

    hn, _ = np.histogram(
        score[
            negative
        ],
        bins=stat.edges,
    )

    stat.neg += hn

    if positive.any():

        num, labels = cv2.connectedComponents(
            mask,
            connectivity=8,
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
                bins=stat.edges,
            )

            stat.region += (
                h.astype(
                    np.float64
                )
                / len(values)
            )

            stat.regions += 1


def aupro(stat):

    if (
        stat.regions == 0
        or
        stat.neg.sum() == 0
    ):
        return np.nan

    fp = np.concatenate([
        [0.0],
        np.cumsum(
            stat.neg[::-1],
            dtype=np.float64,
        ),
    ])

    pro = np.concatenate([
        [0.0],
        np.cumsum(
            stat.region[::-1],
            dtype=np.float64,
        )
        /
        stat.regions,
    ])

    fpr = (
        fp / stat.neg.sum()
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

        if idx < len(fpr):

            x0 = fpr[idx - 1]
            x1 = fpr[idx]

            y0 = pro[idx - 1]
            y1 = pro[idx]

            alpha = (
                MAX_FPR - x0
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
        / MAX_FPR
    )


raw_stats = defaultdict(
    lambda: Stats(
        RAW_EDGES
    )
)


public_masks = {}


for row in public_rows:

    public_masks[
        row["orbit_index"]
    ] = load_mask(
        row
    )

    raw32 = np.asarray(
        PUBLIC_MAPS[
            row["orbit_index"],
            0,
        ],
        dtype=np.float32,
    )

    add_score(
        raw_stats[
            (
                row["category"],
                row["condition"],
            )
        ],
        raw32,
        public_masks[
            row["orbit_index"]
        ],
    )


raw_category_condition = {
    key: aupro(stat)
    for key, stat
    in raw_stats.items()
}


def macro_condition(
    table,
    conditions,
):

    values = []

    for cat in CATEGORIES:

        cat_values = []

        for condition in conditions:

            key = (
                cat,
                condition,
            )

            if key in table:

                value = table[
                    key
                ]

                if np.isfinite(
                    value
                ):

                    cat_values.append(
                        value
                    )

        if cat_values:

            values.append(
                float(
                    np.mean(
                        cat_values
                    )
                )
            )

    return float(
        np.mean(
            values
        )
    )


RAW_REGULAR = macro_condition(
    raw_category_condition,
    ["regular"],
)

RAW_SHIFT = macro_condition(
    raw_category_condition,
    SHIFT_CONDITIONS,
)


print("=" * 110)
print("A4.2 — OCC CALIBRATION STABILITY AUDIT")
print("=" * 110)

print(
    "Raw regular PRO :",
    f"{RAW_REGULAR:.6f}"
)

print(
    "Raw shift PRO   :",
    f"{RAW_SHIFT:.6f}"
)


# ============================================================
# Validation rows per category
# ============================================================

val_by_cat = defaultdict(
    list
)

for row in val_rows:

    val_by_cat[
        row["category"]
    ].append(row)


for cat in CATEGORIES:

    val_by_cat[
        cat
    ] = sorted(
        val_by_cat[
            cat
        ],
        key=lambda r: (
            int(
                r["instance_id"]
            ),
            r["image_path"],
        )
    )


# ============================================================
# Spatial conformal transform
# ============================================================

def spatial_transform(
    raw32,
    null,
):

    raw = np.asarray(
        raw32,
        dtype=np.float32,
    ).reshape(
        1024
    )

    n = null.shape[0]

    out = np.empty(
        1024,
        dtype=np.float32,
    )

    for p in range(
        1024
    ):

        left = np.searchsorted(
            null[:, p],
            raw[p],
            side="left",
        )

        tail = (
            n - left
        )

        cp = (
            1.0
            + tail
        ) / (
            n + 1.0
        )

        out[p] = (
            -math.log(
                cp
            )
        )

    return out.reshape(
        32,
        32,
    )


# ============================================================
# Run 10 independent deterministic 50% splits
# ============================================================

split_rows = []

category_gain_accumulator = defaultdict(
    list
)


for seed in SEEDS:

    print()
    print(
        "-" * 110
    )
    print(
        "SEED:",
        seed
    )

    rng = np.random.default_rng(
        seed
    )

    nulls = {}

    spatial_max = 0.0


    for cat in CATEGORIES:

        members = val_by_cat[
            cat
        ]

        n_total = len(
            members
        )

        n_cal = (
            (n_total + 1)
            // 2
        )

        chosen = rng.choice(
            n_total,
            size=n_cal,
            replace=False,
        )

        chosen = np.sort(
            chosen
        )

        ids = [
            members[int(i)][
                "orbit_index"
            ]
            for i in chosen
        ]

        orbit = np.asarray(
            VAL_MAPS[
                ids
            ],
            dtype=np.float32,
        )

        null = orbit.reshape(
            -1,
            1024,
        )

        null.sort(
            axis=0
        )

        nulls[
            cat
        ] = null

        spatial_max = max(
            spatial_max,
            math.log(
                null.shape[0]
                + 1
            )
            + 1e-3,
        )


    edges = np.linspace(
        0.0,
        spatial_max,
        HIST_BINS + 1,
    )


    seed_stats = defaultdict(
        lambda: Stats(
            edges
        )
    )


    for row in public_rows:

        cat = row[
            "category"
        ]

        raw32 = np.asarray(
            PUBLIC_MAPS[
                row["orbit_index"],
                0,
            ],
            dtype=np.float32,
        )

        occ32 = spatial_transform(
            raw32,
            nulls[
                cat
            ],
        )

        add_score(
            seed_stats[
                (
                    cat,
                    row[
                        "condition"
                    ],
                )
            ],
            occ32,
            public_masks[
                row["orbit_index"]
            ],
        )


    occ_table = {
        key: aupro(stat)
        for key, stat
        in seed_stats.items()
    }


    occ_regular = macro_condition(
        occ_table,
        ["regular"],
    )

    occ_shift = macro_condition(
        occ_table,
        SHIFT_CONDITIONS,
    )


    shift_gain = (
        occ_shift
        -
        RAW_SHIFT
    )

    regular_gain = (
        occ_regular
        -
        RAW_REGULAR
    )


    split_rows.append({
        "seed":
            seed,

        "raw_regular_pro":
            RAW_REGULAR,

        "occ_regular_pro":
            occ_regular,

        "regular_gain":
            regular_gain,

        "raw_shift_pro":
            RAW_SHIFT,

        "occ_shift_pro":
            occ_shift,

        "shift_gain":
            shift_gain,
    })


    # Category-wise shift gain
    for cat in CATEGORIES:

        raw_values = []
        occ_values = []

        for condition in SHIFT_CONDITIONS:

            key = (
                cat,
                condition,
            )

            if key in (
                raw_category_condition
            ) and key in occ_table:

                rv = (
                    raw_category_condition[
                        key
                    ]
                )

                ov = (
                    occ_table[
                        key
                    ]
                )

                if (
                    np.isfinite(rv)
                    and
                    np.isfinite(ov)
                ):

                    raw_values.append(
                        rv
                    )

                    occ_values.append(
                        ov
                    )

        if raw_values:

            category_gain_accumulator[
                cat
            ].append(
                float(
                    np.mean(
                        occ_values
                    )
                    -
                    np.mean(
                        raw_values
                    )
                )
            )


    print(
        "regular gain :",
        f"{regular_gain:+.6f}"
    )

    print(
        "shift gain   :",
        f"{shift_gain:+.6f}"
    )


# ============================================================
# Save split results
# ============================================================

with SPLIT_CSV.open(
    "w",
    newline="",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=list(
            split_rows[0].keys()
        ),
    )

    writer.writeheader()
    writer.writerows(
        split_rows
    )


shift_gains = np.asarray(
    [
        r["shift_gain"]
        for r in split_rows
    ],
    dtype=np.float64,
)

regular_gains = np.asarray(
    [
        r["regular_gain"]
        for r in split_rows
    ],
    dtype=np.float64,
)


# ============================================================
# Category robustness
# ============================================================

category_rows = []


for cat in CATEGORIES:

    gains = np.asarray(
        category_gain_accumulator[
            cat
        ],
        dtype=np.float64,
    )

    category_rows.append({
        "category":
            cat,

        "mean_shift_gain":
            float(
                gains.mean()
            ),

        "median_shift_gain":
            float(
                np.median(
                    gains
                )
            ),

        "positive_fraction":
            float(
                np.mean(
                    gains > 0
                )
            ),

        "min_shift_gain":
            float(
                gains.min()
            ),

        "max_shift_gain":
            float(
                gains.max()
            ),
    })


with CATEGORY_CSV.open(
    "w",
    newline="",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=list(
            category_rows[0].keys()
        ),
    )

    writer.writeheader()
    writer.writerows(
        category_rows
    )


# ============================================================
# Pre-registered A4.2 decision
# ============================================================

mean_shift_gain = float(
    shift_gains.mean()
)

positive_fraction = float(
    np.mean(
        shift_gains > 0
    )
)

mean_regular_gain = float(
    regular_gains.mean()
)


signal_mean = (
    mean_shift_gain
    >= 0.010
)

signal_consistency = (
    positive_fraction
    >= 0.80
)

signal_regular = (
    mean_regular_gain
    >= 0.0
)


decision = (
    "GO"
    if (
        signal_mean
        and
        signal_consistency
        and
        signal_regular
    )
    else "NO_GO"
)


payload = {
    "protocol": {
        "num_splits":
            len(SEEDS),

        "calibration_fraction":
            0.5,

        "validation_only":
            True,

        "public_gt_used_for_calibration":
            False,

        "metric":
            "AU-PRO@0.05",
    },

    "pre_registered": {
        "minimum_mean_shift_gain":
            0.010,

        "minimum_positive_split_fraction":
            0.80,

        "minimum_mean_regular_gain":
            0.0,
    },

    "observed": {
        "mean_shift_gain":
            mean_shift_gain,

        "std_shift_gain":
            float(
                shift_gains.std(
                    ddof=1
                )
            ),

        "min_shift_gain":
            float(
                shift_gains.min()
            ),

        "max_shift_gain":
            float(
                shift_gains.max()
            ),

        "positive_split_fraction":
            positive_fraction,

        "mean_regular_gain":
            mean_regular_gain,

        "std_regular_gain":
            float(
                regular_gains.std(
                    ddof=1
                )
            ),
    },

    "signals": {
        "mean_shift_gain":
            bool(
                signal_mean
            ),

        "split_consistency":
            bool(
                signal_consistency
            ),

        "regular_non_degradation":
            bool(
                signal_regular
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
print("A4.2 CALIBRATION STABILITY RESULTS")
print("=" * 110)

print(
    f"{'seed':12s}"
    f"{'RegGain':>12s}"
    f"{'ShiftGain':>12s}"
)

print("-" * 36)


for r in split_rows:

    print(
        f"{r['seed']:12d}"
        f"{r['regular_gain']:12.4f}"
        f"{r['shift_gain']:12.4f}"
    )


print()
print("[AGGREGATE]")

print(
    "mean shift gain       :",
    f"{mean_shift_gain:+.4f}"
)

print(
    "std shift gain        :",
    f"{shift_gains.std(ddof=1):.4f}"
)

print(
    "min / max shift gain  :",
    f"{shift_gains.min():+.4f}",
    "/",
    f"{shift_gains.max():+.4f}"
)

print(
    "positive split ratio  :",
    f"{positive_fraction:.2f}"
)

print(
    "mean regular gain     :",
    f"{mean_regular_gain:+.4f}"
)


print()
print("[CATEGORY MEAN SHIFT GAINS]")

for r in category_rows:

    print(
        f"{r['category']:15s} "
        f"{r['mean_shift_gain']:+.4f} "
        f"(positive splits="
        f"{r['positive_fraction']:.2f})"
    )


print()
print("[PRE-REGISTERED A4.2 GO / NO-GO]")

print(
    "mean shift >= +0.0100 :",
    signal_mean
)

print(
    ">=80% splits positive :",
    signal_consistency
)

print(
    "mean regular >= 0      :",
    signal_regular
)

print()
print(
    "DECISION :",
    decision
)

print()
print(
    "SPLITS   :",
    SPLIT_CSV
)

print(
    "CATEGORY :",
    CATEGORY_CSV
)

print(
    "SUMMARY  :",
    SUMMARY_JSON
)

print("=" * 110)
