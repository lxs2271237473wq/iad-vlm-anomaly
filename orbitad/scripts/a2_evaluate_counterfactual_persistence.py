from pathlib import Path
from collections import defaultdict
import csv
import json

import cv2
import numpy as np


ROOT = Path(
    "orbitad/results/a2_counterfactual_persistence"
)

DATA_ROOT = Path(
    "/root/private_data/iad-vlm-anomaly/datasets/MVTec_AD_2"
)

PUBLIC_MAPS = np.load(
    ROOT / "public_orbit_maps.npy",
    mmap_mode="r",
)

VAL_MAPS = np.load(
    ROOT / "validation_orbit_maps.npy",
    mmap_mode="r",
)

PUBLIC_META = ROOT / "public_orbit_meta.csv"
VAL_META = ROOT / "validation_orbit_meta.csv"

DETAIL_CSV = ROOT / "method_condition_metrics.csv"
SUMMARY_CSV = ROOT / "method_summary.csv"
THRESHOLDS_JSON = ROOT / "method_validation_thresholds.json"
SUMMARY_JSON = ROOT / "a2_1_summary.json"


METHODS = [
    "original",
    "mean",
    "median",
    "q25",
    "min",
    "spr",
]

EVAL_SIZE = 512
MAX_FPR = 0.05
Q_VAL = 0.995

HIST_BINS = 4096

edges = np.linspace(
    0.0,
    2.0,
    HIST_BINS + 1,
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
    PUBLIC_META
)

val_rows = read_meta(
    VAL_META
)


def aggregate(
    orbit,
    method,
):

    x = np.asarray(
        orbit,
        dtype=np.float32,
    )

    if method == "original":
        return x[0]

    if method == "mean":
        return np.mean(
            x,
            axis=0,
        )

    if method == "median":
        return np.median(
            x,
            axis=0,
        )

    if method == "q25":
        return np.quantile(
            x,
            0.25,
            axis=0,
        )

    if method == "min":
        return np.min(
            x,
            axis=0,
        )

    if method == "spr":

        q25 = np.quantile(
            x,
            0.25,
            axis=0,
        )

        q75 = np.quantile(
            x,
            0.75,
            axis=0,
        )

        persistence = np.clip(
            q25
            /
            (
                q75
                + 1e-6
            ),
            0.0,
            1.0,
        )

        return (
            x[0]
            * persistence
        )

    raise ValueError(method)


def upsample(x):

    return cv2.resize(
        x.astype(
            np.float32
        ),
        (
            EVAL_SIZE,
            EVAL_SIZE,
        ),
        interpolation=cv2.INTER_LINEAR,
    )


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


# ============================================================
# Validation-only thresholds PER METHOD + CATEGORY
# ============================================================

val_hist = defaultdict(
    lambda: np.zeros(
        HIST_BINS,
        dtype=np.int64,
    )
)


for row in val_rows:

    orbit = VAL_MAPS[
        row["orbit_index"]
    ]

    for method in METHODS:

        score = upsample(
            aggregate(
                orbit,
                method,
            )
        )

        h, _ = np.histogram(
            score,
            bins=edges,
        )

        val_hist[
            (
                method,
                row["category"],
            )
        ] += h


def quantile_from_hist(
    hist,
    q,
):

    cdf = np.cumsum(
        hist
    )

    target = q * hist.sum()

    idx = np.searchsorted(
        cdf,
        target,
    )

    idx = min(
        max(
            int(idx),
            0,
        ),
        HIST_BINS - 1,
    )

    return float(
        edges[idx]
    )


thresholds = {}

for key, hist in val_hist.items():

    thresholds[key] = quantile_from_hist(
        hist,
        Q_VAL,
    )


with THRESHOLDS_JSON.open("w") as f:

    json.dump(
        {
            f"{m}/{c}": t
            for (
                m,
                c,
            ), t in thresholds.items()
        },
        f,
        indent=2,
    )


# ============================================================
# Metrics
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

        self.image_pos = []
        self.image_neg = []


stats = defaultdict(
    Stats
)


def image_score(
    map32,
):

    x = map32.reshape(-1)

    k = 10

    idx = np.argpartition(
        x,
        -k,
    )[-k:]

    return float(
        x[idx].mean()
    )


for n, row in enumerate(
    public_rows,
    1,
):

    orbit = PUBLIC_MAPS[
        row["orbit_index"]
    ]

    mask = load_mask(
        row
    )

    pos_mask = mask.astype(
        bool
    )

    neg_mask = ~pos_mask

    for method in METHODS:

        map32 = aggregate(
            orbit,
            method,
        )

        score = upsample(
            map32
        )

        g = stats[
            (
                method,
                row["category"],
                row["condition"],
            )
        ]

        if row["label"] == 1:
            g.image_pos.append(
                image_score(
                    map32
                )
            )
        else:
            g.image_neg.append(
                image_score(
                    map32
                )
            )

        pos_values = score[
            pos_mask
        ]

        neg_values = score[
            neg_mask
        ]

        if len(pos_values):

            h, _ = np.histogram(
                pos_values,
                bins=edges,
            )

            g.pos += h

        if len(neg_values):

            h, _ = np.histogram(
                neg_values,
                bins=edges,
            )

            g.neg += h

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

                region_scores = score[
                    labels == rid
                ]

                if len(
                    region_scores
                ) == 0:
                    continue

                h, _ = np.histogram(
                    region_scores,
                    bins=edges,
                )

                g.region += (
                    h.astype(
                        np.float64
                    )
                    /
                    len(
                        region_scores
                    )
                )

                g.regions += 1

        threshold = thresholds[
            (
                method,
                row["category"],
            )
        ]

        pred = (
            score >= threshold
        )

        g.tp += int(
            np.logical_and(
                pred,
                pos_mask,
            ).sum()
        )

        g.fp += int(
            np.logical_and(
                pred,
                neg_mask,
            ).sum()
        )

        g.fn += int(
            np.logical_and(
                ~pred,
                pos_mask,
            ).sum()
        )

    if n % 100 == 0:
        print(
            f"processed "
            f"{n}/{len(public_rows)}"
        )


def auc_image(
    pos,
    neg,
):

    if (
        len(pos) == 0
        or len(neg) == 0
    ):
        return np.nan

    p = np.asarray(pos)
    n = np.asarray(neg)

    d = (
        p[:, None]
        - n[None, :]
    )

    return float(
        (
            (d > 0).sum()
            +
            0.5 * (
                d == 0
            ).sum()
        )
        /
        d.size
    )


def pixel_auc(
    pos,
    neg,
):

    P = pos.sum()
    N = neg.sum()

    if P == 0 or N == 0:
        return np.nan

    tp = np.concatenate(
        [
            [0],
            np.cumsum(
                pos[::-1],
                dtype=np.float64,
            ),
        ]
    )

    fp = np.concatenate(
        [
            [0],
            np.cumsum(
                neg[::-1],
                dtype=np.float64,
            ),
        ]
    )

    return float(
        np.trapezoid(
            tp / P,
            fp / N,
        )
    )


def aupro(
    g,
):

    if (
        g.regions == 0
        or g.neg.sum() == 0
    ):
        return np.nan

    fp = np.concatenate(
        [
            [0],
            np.cumsum(
                g.neg[::-1],
                dtype=np.float64,
            ),
        ]
    )

    pro = np.concatenate(
        [
            [0],
            np.cumsum(
                g.region[::-1],
                dtype=np.float64,
            )
            /
            g.regions,
        ]
    )

    fpr = (
        fp
        /
        g.neg.sum()
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

    if x[-1] < MAX_FPR:

        idx = np.searchsorted(
            fpr,
            MAX_FPR,
            side="right",
        )

        if idx < len(fpr):

            x0, x1 = (
                fpr[idx-1],
                fpr[idx],
            )

            y0, y1 = (
                pro[idx-1],
                pro[idx],
            )

            alpha = (
                MAX_FPR - x0
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


def f1(
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
        2 * tp
        /
        denom
    )


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

        "image_auc":
            auc_image(
                g.image_pos,
                g.image_neg,
            ),

        "pixel_auc":
            pixel_auc(
                g.pos,
                g.neg,
            ),

        "aupro_0_05":
            aupro(g),

        "segf1":
            f1(
                g.tp,
                g.fp,
                g.fn,
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
        )
    )

    writer.writeheader()
    writer.writerows(
        detail_rows
    )


# ============================================================
# Macro condition metrics
# ============================================================

macro = defaultdict(
    list
)

for r in detail_rows:

    macro[
        (
            r["method"],
            r["condition"],
        )
    ].append(r)


macro_rows = []

for (
    method,
    condition,
), members in sorted(
    macro.items()
):

    def avg(key):

        x = np.asarray(
            [
                r[key]
                for r in members
            ],
            dtype=float,
        )

        x = x[
            np.isfinite(x)
        ]

        return float(
            x.mean()
        )

    macro_rows.append({
        "method":
            method,

        "condition":
            condition,

        "categories":
            len(members),

        "image_auc":
            avg("image_auc"),

        "pixel_auc":
            avg("pixel_auc"),

        "aupro_0_05":
            avg("aupro_0_05"),

        "segf1":
            avg("segf1"),
    })


lookup = {
    (
        r["method"],
        r["condition"],
    ): r
    for r in macro_rows
}


# ============================================================
# Method summary
# ============================================================

summary_rows = []

shift_conditions = [
    "shift_1",
    "shift_2",
    "shift_3",
]


baseline_regular = lookup[
    (
        "original",
        "regular",
    )
]


def shift_mean(
    method,
    metric,
):

    vals = []

    for condition in shift_conditions:

        key = (
            method,
            condition,
        )

        if key in lookup:

            vals.append(
                lookup[key][
                    metric
                ]
            )

    return float(
        np.mean(vals)
    )


baseline_shift_f1 = shift_mean(
    "original",
    "segf1",
)

baseline_shift_pro = shift_mean(
    "original",
    "aupro_0_05",
)


for method in METHODS:

    regular = lookup[
        (
            method,
            "regular",
        )
    ]

    shift_f1 = shift_mean(
        method,
        "segf1",
    )

    shift_pro = shift_mean(
        method,
        "aupro_0_05",
    )

    summary_rows.append({
        "method":
            method,

        "regular_segf1":
            regular[
                "segf1"
            ],

        "shift123_mean_segf1":
            shift_f1,

        "shift_segf1_gain_vs_original":
            shift_f1
            -
            baseline_shift_f1,

        "regular_segf1_delta":
            regular[
                "segf1"
            ]
            -
            baseline_regular[
                "segf1"
            ],

        "regular_aupro":
            regular[
                "aupro_0_05"
            ],

        "shift123_mean_aupro":
            shift_pro,

        "shift_aupro_gain_vs_original":
            shift_pro
            -
            baseline_shift_pro,

        "regular_aupro_delta":
            regular[
                "aupro_0_05"
            ]
            -
            baseline_regular[
                "aupro_0_05"
            ],
    })


with SUMMARY_CSV.open(
    "w",
    newline="",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=list(
            summary_rows[0].keys()
        )
    )

    writer.writeheader()
    writer.writerows(
        summary_rows
    )


# ============================================================
# Pre-registered method-level decision
# ============================================================

PRIMARY = {
    "q25",
    "spr",
}


eligible = []

for r in summary_rows:

    if r["method"] not in PRIMARY:
        continue

    seg_signal = (
        r[
            "shift_segf1_gain_vs_original"
        ] >= 0.02
        and
        r[
            "regular_segf1_delta"
        ] >= -0.015
    )

    pro_signal = (
        r[
            "shift_aupro_gain_vs_original"
        ] >= 0.01
        and
        r[
            "regular_aupro_delta"
        ] >= -0.015
    )

    if (
        seg_signal
        or pro_signal
    ):

        eligible.append(
            r["method"]
        )


decision = (
    "PASS"
    if eligible
    else "FAIL"
)


payload = {
    "primary_methods":
        sorted(PRIMARY),

    "decision_thresholds": {
        "shift_segf1_gain":
            0.02,

        "shift_aupro_gain":
            0.01,

        "maximum_regular_loss":
            0.015,
    },

    "eligible_primary_methods":
        eligible,

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
print("=" * 110)
print("A2.1 COUNTERFACTUAL PERSISTENCE RESULTS")
print("=" * 110)

print(
    f"{'method':10s}"
    f"{'Reg-F1':>10s}"
    f"{'Shift-F1':>11s}"
    f"{'gain-F1':>10s}"
    f"{'Reg-PRO':>10s}"
    f"{'Shift-PRO':>11s}"
    f"{'gain-PRO':>10s}"
)

print("-" * 72)

for r in summary_rows:

    print(
        f"{r['method']:10s}"
        f"{r['regular_segf1']:10.4f}"
        f"{r['shift123_mean_segf1']:11.4f}"
        f"{r['shift_segf1_gain_vs_original']:10.4f}"
        f"{r['regular_aupro']:10.4f}"
        f"{r['shift123_mean_aupro']:11.4f}"
        f"{r['shift_aupro_gain_vs_original']:10.4f}"
    )


print()
print("[PRIMARY GO / NO-GO]")

print(
    "primary methods :",
    sorted(PRIMARY)
)

print(
    "eligible        :",
    eligible
)

print()
print("DECISION :", decision)

print()
print("DETAIL  :", DETAIL_CSV)
print("SUMMARY :", SUMMARY_CSV)
print("JSON    :", SUMMARY_JSON)

print("=" * 110)
