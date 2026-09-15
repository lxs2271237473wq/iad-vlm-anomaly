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
    "orbitad/results/a4_occ"
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
    OUT_ROOT /
    "a4_1_mechanism_category_condition.csv"
)

SUMMARY_CSV = (
    OUT_ROOT /
    "a4_1_mechanism_summary.csv"
)

SUMMARY_JSON = (
    OUT_ROOT /
    "a4_1_summary.json"
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

HIST_BINS = 4096

RAW_EDGES = np.linspace(
    0.0,
    2.0,
    HIST_BINS + 1,
)

CONF_EDGES = np.linspace(
    0.0,
    12.0,
    HIST_BINS + 1,
)


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


# ============================================================
# Same deterministic half split as A4.0
# ============================================================

by_cat = defaultdict(list)

for r in val_rows:

    by_cat[
        r["category"]
    ].append(r)


cal_rows = defaultdict(list)


for cat, members in sorted(
    by_cat.items()
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
                cat
            ].append(r)


# ============================================================
# Four calibration nulls
# ============================================================

global_regular_null = {}
global_orbit_null = {}

spatial_regular_null = {}
spatial_orbit_null = {}


for cat, members in sorted(
    cal_rows.items()
):

    idx = [
        r["orbit_index"]
        for r in members
    ]

    # [N,6,32,32]
    orbit = np.asarray(
        VAL_MAPS[
            idx
        ],
        dtype=np.float32,
    )

    # ----------------------------------------
    # Global regular
    # ----------------------------------------

    x = orbit[
        :,
        0,
        :,
        :
    ].reshape(-1)

    x.sort()

    global_regular_null[
        cat
    ] = x


    # ----------------------------------------
    # Global orbit
    # ----------------------------------------

    x = orbit.reshape(
        -1
    )

    x.sort()

    global_orbit_null[
        cat
    ] = x


    # ----------------------------------------
    # Spatial regular
    # [N,1024]
    # ----------------------------------------

    x = orbit[
        :,
        0,
        :,
        :
    ].reshape(
        len(idx),
        1024,
    )

    x.sort(
        axis=0
    )

    spatial_regular_null[
        cat
    ] = x


    # ----------------------------------------
    # Spatial orbit
    # [N*6,1024]
    # ----------------------------------------

    x = orbit.reshape(
        -1,
        1024,
    )

    x.sort(
        axis=0
    )

    spatial_orbit_null[
        cat
    ] = x


# ============================================================
# Conformal transforms
# ============================================================

def conformal_global(
    raw32,
    sorted_null,
):

    raw = np.asarray(
        raw32,
        dtype=np.float32,
    )

    flat = raw.reshape(
        -1
    )

    n = len(
        sorted_null
    )

    left = np.searchsorted(
        sorted_null,
        flat,
        side="left",
    )

    tail = (
        n - left
    )

    p = (
        1.0
        + tail
    ) / (
        n + 1.0
    )

    out = (
        -np.log(p)
    )

    return out.reshape(
        32,
        32,
    ).astype(
        np.float32
    )


def conformal_spatial(
    raw32,
    sorted_null,
):

    raw = np.asarray(
        raw32,
        dtype=np.float32,
    ).reshape(
        1024
    )

    n = sorted_null.shape[
        0
    ]

    out = np.empty(
        1024,
        dtype=np.float32,
    )

    for p in range(
        1024
    ):

        left = np.searchsorted(
            sorted_null[
                :,
                p
            ],
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
# Metrics
# ============================================================

class Stats:

    def __init__(
        self,
        edges,
    ):

        bins = len(
            edges
        ) - 1

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

        edges = (
            RAW_EDGES
            if method == "raw"
            else CONF_EDGES
        )

        stats[
            key
        ] = Stats(
            edges
        )

    return stats[
        key
    ]


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


for n, row in enumerate(
    public_rows,
    1,
):

    cat = row[
        "category"
    ]

    condition = row[
        "condition"
    ]

    # REAL observed image only.
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
                global_regular_null[
                    cat
                ],
            ),

        "global_orbit":
            conformal_global(
                raw,
                global_orbit_null[
                    cat
                ],
            ),

        "spatial_regular":
            conformal_spatial(
                raw,
                spatial_regular_null[
                    cat
                ],
            ),

        "spatial_orbit":
            conformal_spatial(
                raw,
                spatial_orbit_null[
                    cat
                ],
            ),
    }

    mask = load_mask(
        row
    )

    pos = mask.astype(
        bool
    )

    neg = ~pos


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
                pos
            ],
            bins=g.edges,
        )

        hn, _ = np.histogram(
            score[
                neg
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


def pixel_auc(
    g,
):

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


def aupro(
    g,
):

    if (
        g.regions == 0
        or g.neg.sum() == 0
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
        /
        MAX_FPR
    )


# ============================================================
# Category-condition results
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
            pixel_auc(
                g
            ),

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
            detail[0].keys()
        ),
    )

    writer.writeheader()
    writer.writerows(
        detail
    )


# ============================================================
# Macro summary
# ============================================================

groups = defaultdict(
    list
)


for r in detail:

    groups[
        (
            r["method"],
            r["condition"],
        )
    ].append(r)


macro = []


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

        "categories":
            len(members),

        "pixel_auc":
            float(
                np.mean([
                    r[
                        "pixel_auc"
                    ]
                    for r in members
                    if np.isfinite(
                        r[
                            "pixel_auc"
                        ]
                    )
                ])
            ),

        "aupro_0_05":
            float(
                np.mean([
                    r[
                        "aupro_0_05"
                    ]
                    for r in members
                    if np.isfinite(
                        r[
                            "aupro_0_05"
                        ]
                    )
                ])
            ),
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

    vals = [
        lookup[
            (
                method,
                c,
            )
        ][
            metric
        ]
        for c in SHIFT_CONDITIONS
    ]

    return float(
        np.mean(
            vals
        )
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


# ============================================================
# Pre-registered mechanism decision
# ============================================================

s = {
    r["method"]: r
    for r in summary
}


raw_shift = s[
    "raw"
]["shift123_aupro"]

so_shift = s[
    "spatial_orbit"
]["shift123_aupro"]

go_shift = s[
    "global_orbit"
]["shift123_aupro"]

sr_shift = s[
    "spatial_regular"
]["shift123_aupro"]


primary_gain = (
    so_shift - raw_shift
)

beats_global_orbit = (
    so_shift >= go_shift
)

beats_spatial_regular = (
    so_shift >= sr_shift
)


decision = (
    "GO"
    if (
        primary_gain >= 0.01
        and
        beats_global_orbit
        and
        beats_spatial_regular
    )
    else "NO_GO"
)


payload = {
    "primary":
        "spatial_orbit",

    "pre_registered": {
        "minimum_shift_aupro_gain_vs_raw":
            0.01,

        "must_match_or_beat_global_orbit":
            True,

        "must_match_or_beat_spatial_regular":
            True,
    },

    "observed": {
        "raw_shift123_aupro":
            float(
                raw_shift
            ),

        "spatial_orbit_shift123_aupro":
            float(
                so_shift
            ),

        "gain_vs_raw":
            float(
                primary_gain
            ),

        "global_orbit_shift123_aupro":
            float(
                go_shift
            ),

        "spatial_regular_shift123_aupro":
            float(
                sr_shift
            ),
    },

    "signals": {
        "gain_signal":
            bool(
                primary_gain >= 0.01
            ),

        "beats_global_orbit":
            bool(
                beats_global_orbit
            ),

        "beats_spatial_regular":
            bool(
                beats_spatial_regular
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
print("=" * 100)
print("A4.1 OCC MECHANISM ABLATION")
print("=" * 100)

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
print("[MECHANISM TEST]")

print(
    "Spatial-Orbit gain vs Raw      : "
    f"{primary_gain:+.4f}"
)

print(
    "Spatial-Orbit >= Global-Orbit  :",
    beats_global_orbit
)

print(
    "Spatial-Orbit >= Spatial-Reg   :",
    beats_spatial_regular
)

print()
print("DECISION :", decision)

print()
print("DETAIL :", DETAIL_CSV)
print("SUMMARY:", SUMMARY_CSV)
print("JSON   :", SUMMARY_JSON)

print("=" * 100)
