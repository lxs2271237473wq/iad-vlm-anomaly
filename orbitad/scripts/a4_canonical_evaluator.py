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
    "orbitad/results/a4_occ/canonical"
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


DETAIL_CSV = (
    OUT_ROOT / "category_condition_metrics.csv"
)

SUMMARY_CSV = (
    OUT_ROOT / "summary.csv"
)

AUDIT_JSON = (
    OUT_ROOT / "reproducibility_audit.json"
)


METHODS = [
    "raw",
    "global_regular",
    "global_orbit",
    "spatial_regular",
    "spatial_orbit",
]

SHIFT_CONDITIONS = [
    "shift_1",
    "shift_2",
    "shift_3",
]

EVAL_SIZE = 512
MAX_FPR = 0.05

# Much finer than previous 4096-bin approximation.
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


if PUBLIC_MAPS.shape != (
    1084,
    6,
    32,
    32,
):
    raise RuntimeError(
        f"Bad public shape: {PUBLIC_MAPS.shape}"
    )

if VAL_MAPS.shape != (
    302,
    6,
    32,
    32,
):
    raise RuntimeError(
        f"Bad validation shape: {VAL_MAPS.shape}"
    )


print("=" * 110)
print("A4 CANONICAL REPRODUCIBILITY EVALUATOR")
print("=" * 110)


# ============================================================
# EXACT SAME deterministic calibration half
# ============================================================

by_category = defaultdict(list)

for r in val_rows:

    by_category[
        r["category"]
    ].append(r)


cal_rows = defaultdict(list)


for category, members in sorted(
    by_category.items()
):

    members = sorted(
        members,
        key=lambda r: (
            int(r["instance_id"]),
            r["image_path"],
        )
    )

    for i, r in enumerate(members):

        if i % 2 == 0:

            cal_rows[
                category
            ].append(r)


print()
print("[CALIBRATION SPLIT]")

for cat in sorted(cal_rows):

    print(
        f"{cat:15s}: "
        f"{len(cal_rows[cat])}"
    )


# ============================================================
# Null construction
# ============================================================

GR = {}
GO = {}
SR = {}
SO = {}


for cat, members in sorted(
    cal_rows.items()
):

    ids = [
        r["orbit_index"]
        for r in members
    ]

    orbit = np.asarray(
        VAL_MAPS[ids],
        dtype=np.float32,
    )

    # [N,6,32,32]

    gr = (
        orbit[:, 0]
        .reshape(-1)
        .copy()
    )

    gr.sort()

    GR[cat] = gr


    go = (
        orbit
        .reshape(-1)
        .copy()
    )

    go.sort()

    GO[cat] = go


    sr = (
        orbit[:, 0]
        .reshape(
            len(ids),
            1024,
        )
        .copy()
    )

    sr.sort(
        axis=0
    )

    SR[cat] = sr


    so = (
        orbit
        .reshape(
            -1,
            1024,
        )
        .copy()
    )

    so.sort(
        axis=0
    )

    SO[cat] = so


# ============================================================
# IMPORTANT identity audit
#
# Reconstruct spatial-orbit twice using two separately
# written but mathematically equivalent functions.
# They MUST be identical.
# ============================================================

def spatial_transform_A(
    raw32,
    null,
):

    raw = np.asarray(
        raw32,
        dtype=np.float32,
    ).reshape(1024)

    n = null.shape[0]

    out = np.empty(
        1024,
        dtype=np.float32,
    )

    for p in range(1024):

        left = np.searchsorted(
            null[:, p],
            raw[p],
            side="left",
        )

        tail = n - left

        cp = (
            1.0 + tail
        ) / (
            n + 1.0
        )

        out[p] = -math.log(
            cp
        )

    return out.reshape(
        32,
        32,
    )


def spatial_transform_B(
    raw32,
    null,
):

    raw = np.asarray(
        raw32,
        dtype=np.float32,
    ).reshape(-1)

    n = null.shape[0]

    values = []

    for p, value in enumerate(raw):

        idx = np.searchsorted(
            null[:, p],
            value,
            side="left",
        )

        p_tail = (
            1
            +
            (
                n - idx
            )
        ) / (
            n + 1
        )

        values.append(
            -np.log(
                p_tail
            )
        )

    return np.asarray(
        values,
        dtype=np.float32,
    ).reshape(
        32,
        32,
    )


identity_max_error = 0.0


for row in public_rows:

    cat = row["category"]

    raw = np.asarray(
        PUBLIC_MAPS[
            row["orbit_index"],
            0,
        ],
        dtype=np.float32,
    )

    a = spatial_transform_A(
        raw,
        SO[cat],
    )

    b = spatial_transform_B(
        raw,
        SO[cat],
    )

    error = float(
        np.max(
            np.abs(
                a - b
            )
        )
    )

    identity_max_error = max(
        identity_max_error,
        error,
    )


print()
print("[SPATIAL-ORBIT IDENTITY TEST]")
print(
    "maximum absolute map error:",
    f"{identity_max_error:.12g}"
)

if identity_max_error > 1e-7:

    raise RuntimeError(
        "Spatial OCC identity test FAILED"
    )

print("IDENTITY STATUS: PASS")


# ============================================================
# Conformal transforms
# ============================================================

def conformal_global(
    raw32,
    null,
):

    raw = np.asarray(
        raw32,
        dtype=np.float32,
    )

    flat = raw.reshape(-1)

    n = len(null)

    left = np.searchsorted(
        null,
        flat,
        side="left",
    )

    tail = (
        n - left
    )

    cp = (
        1.0 + tail
    ) / (
        n + 1.0
    )

    return (
        -np.log(cp)
    ).reshape(
        32,
        32,
    ).astype(
        np.float32
    )


# ============================================================
# Dynamic histogram ranges
#
# Do not truncate conformal scores.
# ============================================================

raw_max = 2.0

max_global_n = max(
    max(
        len(x)
        for x in GR.values()
    ),
    max(
        len(x)
        for x in GO.values()
    ),
)

max_spatial_n = max(
    max(
        x.shape[0]
        for x in SR.values()
    ),
    max(
        x.shape[0]
        for x in SO.values()
    ),
)

global_max = (
    math.log(
        max_global_n + 1
    )
    + 1e-3
)

spatial_max = (
    math.log(
        max_spatial_n + 1
    )
    + 1e-3
)


RAW_EDGES = np.linspace(
    0.0,
    raw_max,
    HIST_BINS + 1,
)

GLOBAL_EDGES = np.linspace(
    0.0,
    global_max,
    HIST_BINS + 1,
)

SPATIAL_EDGES = np.linspace(
    0.0,
    spatial_max,
    HIST_BINS + 1,
)


print()
print("[HISTOGRAM SUPPORT]")

print(
    "raw max     :",
    raw_max
)

print(
    "global max  :",
    global_max
)

print(
    "spatial max :",
    spatial_max
)


def method_edges(method):

    if method == "raw":
        return RAW_EDGES

    if method.startswith(
        "global_"
    ):
        return GLOBAL_EDGES

    return SPATIAL_EDGES


# ============================================================
# Evaluation
# ============================================================

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
            method_edges(
                method
            )
        )

    return stats[key]


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


def upsample(
    x,
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


for n, row in enumerate(
    public_rows,
    1,
):

    cat = row["category"]

    condition = row[
        "condition"
    ]

    raw = np.asarray(
        PUBLIC_MAPS[
            row["orbit_index"],
            0,
        ],
        dtype=np.float32,
    )

    candidate = {
        "raw":
            raw,

        "global_regular":
            conformal_global(
                raw,
                GR[cat],
            ),

        "global_orbit":
            conformal_global(
                raw,
                GO[cat],
            ),

        "spatial_regular":
            spatial_transform_A(
                raw,
                SR[cat],
            ),

        "spatial_orbit":
            spatial_transform_A(
                raw,
                SO[cat],
            ),
    }

    mask = load_mask(
        row
    )

    positive = (
        mask > 0
    )

    negative = ~positive


    for method, map32 in (
        candidate.items()
    ):

        g = get_stats(
            method,
            cat,
            condition,
        )

        score = upsample(
            map32
        )

        hp, _ = np.histogram(
            score[
                positive
            ],
            bins=g.edges,
        )

        hn, _ = np.histogram(
            score[
                negative
            ],
            bins=g.edges,
        )

        g.pos += hp
        g.neg += hn


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

def pixel_auc(g):

    P = g.pos.sum()
    N = g.neg.sum()

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


def aupro(g):

    if (
        g.regions == 0
        or
        g.neg.sum() == 0
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
        fp / g.neg.sum()
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
        / MAX_FPR
    )


# ============================================================
# Category-condition metrics
# ============================================================

detail = []


for (
    method,
    category,
    condition,
), g in sorted(
    stats.items()
):

    detail.append({
        "method":
            method,

        "category":
            category,

        "condition":
            condition,

        "pixel_auc":
            pixel_auc(g),

        "aupro_0_05":
            aupro(g),
    })


with DETAIL_CSV.open(
    "w",
    newline="",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=list(
            detail[0].keys()
        ),
    )

    writer.writeheader()
    writer.writerows(
        detail
    )


# ============================================================
# Macro
# ============================================================

groups = defaultdict(list)

for r in detail:

    groups[
        (
            r["method"],
            r["condition"],
        )
    ].append(r)


macro = []


def mean_finite(values):

    x = np.asarray(
        values,
        dtype=np.float64,
    )

    x = x[
        np.isfinite(x)
    ]

    return float(
        x.mean()
    )


for (
    method,
    condition,
), members in sorted(
    groups.items()
):

    macro.append({
        "method":
            method,

        "condition":
            condition,

        "pixel_auc":
            mean_finite([
                r["pixel_auc"]
                for r in members
            ]),

        "aupro_0_05":
            mean_finite([
                r["aupro_0_05"]
                for r in members
            ]),
    })


lookup = {
    (
        r["method"],
        r["condition"],
    ): r
    for r in macro
}


def shift_mean(
    method,
    metric,
):

    return float(
        np.mean([
            lookup[
                (
                    method,
                    condition,
                )
            ][metric]

            for condition
            in SHIFT_CONDITIONS
        ])
    )


summary = []


for method in METHODS:

    summary.append({
        "method":
            method,

        "regular_pixel_auc":
            lookup[
                (
                    method,
                    "regular",
                )
            ][
                "pixel_auc"
            ],

        "shift123_pixel_auc":
            shift_mean(
                method,
                "pixel_auc",
            ),

        "regular_aupro":
            lookup[
                (
                    method,
                    "regular",
                )
            ][
                "aupro_0_05"
            ],

        "shift123_aupro":
            shift_mean(
                method,
                "aupro_0_05",
            ),
    })


with SUMMARY_CSV.open(
    "w",
    newline="",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=list(
            summary[0].keys()
        ),
    )

    writer.writeheader()
    writer.writerows(
        summary
    )


s = {
    r["method"]: r
    for r in summary
}


raw_shift = (
    s["raw"][
        "shift123_aupro"
    ]
)

so_shift = (
    s["spatial_orbit"][
        "shift123_aupro"
    ]
)

gain = (
    so_shift
    -
    raw_shift
)


print()
print("=" * 110)
print("CANONICAL A4 RESULTS")
print("=" * 110)

print(
    f"{'method':20s}"
    f"{'Reg-PixAUC':>12s}"
    f"{'Shift-PixAUC':>14s}"
    f"{'Reg-PRO':>11s}"
    f"{'Shift-PRO':>12s}"
)

print("-" * 70)


for r in summary:

    print(
        f"{r['method']:20s}"
        f"{r['regular_pixel_auc']:12.4f}"
        f"{r['shift123_pixel_auc']:14.4f}"
        f"{r['regular_aupro']:11.4f}"
        f"{r['shift123_aupro']:12.4f}"
    )


print()
print("[CANONICAL OCC CHECK]")

print(
    "Spatial-Orbit gain vs Raw:",
    f"{gain:+.4f}"
)

print(
    "Spatial-Orbit vs Global-Orbit:",
    f"{so_shift - s['global_orbit']['shift123_aupro']:+.4f}"
)

print(
    "Spatial-Orbit vs Spatial-Regular:",
    f"{so_shift - s['spatial_regular']['shift123_aupro']:+.4f}"
)


payload = {
    "hist_bins":
        HIST_BINS,

    "histogram_support": {
        "raw_max":
            raw_max,

        "global_max":
            global_max,

        "spatial_max":
            spatial_max,
    },

    "identity_max_error":
        identity_max_error,

    "summary":
        summary,

    "spatial_orbit_gain_vs_raw":
        gain,
}


with AUDIT_JSON.open(
    "w"
) as f:

    json.dump(
        payload,
        f,
        indent=2,
    )


print()
print("AUDIT :", AUDIT_JSON)

print("=" * 110)
