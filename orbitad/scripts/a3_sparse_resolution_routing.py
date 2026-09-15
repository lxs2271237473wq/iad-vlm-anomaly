from pathlib import Path
from collections import defaultdict
import csv
import json

import cv2
import numpy as np


DATA_ROOT = Path(
    "/root/private_data/iad-vlm-anomaly/datasets/MVTec_AD_2"
)

ROOT = Path(
    "orbitad/results/a3_resolution_audit"
)

META = ROOT / "regular_public_meta.csv"

DEFECT_STRATA = ROOT / "regular_defect_strata.csv"

LOW_PATH = ROOT / "regular_maps_r256.npy"
HIGH_PATH = ROOT / "regular_maps_r1024.npy"

DETAIL_CSV = ROOT / "sparse_routing_metrics.csv"
SUMMARY_JSON = ROOT / "a3_1_summary.json"


BUDGETS = [
    0.0625,
    0.125,
    0.25,
    0.50,
]

MAX_FPR = 0.05

HIST_BINS = 8192

edges = np.linspace(
    0.0,
    2.0,
    HIST_BINS + 1,
    dtype=np.float64,
)


# ============================================================
# Load
# ============================================================

rows = []

with META.open() as f:

    for r in csv.DictReader(f):

        r["map_index"] = int(
            r["map_index"]
        )

        r["label"] = int(
            r["label"]
        )

        rows.append(r)


low_maps = np.load(
    LOW_PATH,
    mmap_mode="r",
)

high_maps = np.load(
    HIGH_PATH,
    mmap_mode="r",
)


if low_maps.shape != (
    184,
    16,
    16,
):
    raise RuntimeError(
        str(low_maps.shape)
    )


if high_maps.shape != (
    184,
    64,
    64,
):
    raise RuntimeError(
        str(high_maps.shape)
    )


# ============================================================
# Defect quartiles from immutable A3.0 strata
# ============================================================

quartile = {}

with DEFECT_STRATA.open() as f:

    for r in csv.DictReader(f):

        key = (
            r["category"],
            r["instance_id"],
        )

        quartile[key] = (
            r["size_quartile"]
        )


# ============================================================
# Router
# ============================================================

def low_to_high(
    low,
):

    # Each 16x16 low cell corresponds exactly
    # to a 4x4 region on the 64x64 map.
    return np.repeat(
        np.repeat(
            low,
            4,
            axis=0,
        ),
        4,
        axis=1,
    )


def routed_map(
    low,
    high,
    budget,
):

    if not (
        0 < budget <= 1
    ):
        raise ValueError(
            budget
        )

    low_flat = low.reshape(
        -1
    )

    num_cells = len(
        low_flat
    )

    k = max(
        1,
        int(
            round(
                budget
                * num_cells
            )
        ),
    )

    # IMPORTANT:
    # routing sees ONLY low-resolution anomaly scores.
    selected = np.argpartition(
        low_flat,
        -k,
    )[-k:]

    route_mask = np.zeros(
        num_cells,
        dtype=bool,
    )

    route_mask[
        selected
    ] = True

    route_mask = route_mask.reshape(
        16,
        16,
    )

    hybrid = low_to_high(
        low
    ).astype(
        np.float32
    )

    # Replace selected 4x4 high-resolution blocks.
    ys, xs = np.where(
        route_mask
    )

    for y, x in zip(
        ys,
        xs,
    ):

        y0 = 4 * y
        x0 = 4 * x

        hybrid[
            y0:y0 + 4,
            x0:x0 + 4,
        ] = high[
            y0:y0 + 4,
            x0:x0 + 4,
        ]

    return hybrid


# ============================================================
# Metrics
# ============================================================

class Stats:

    def __init__(self):

        self.neg = np.zeros(
            HIST_BINS,
            dtype=np.int64,
        )

        self.region = np.zeros(
            HIST_BINS,
            dtype=np.float64,
        )

        self.regions = 0


def load_mask(row):

    if row["label"] == 0:

        path = (
            DATA_ROOT
            / row["image_path"]
        )

        image = cv2.imread(
            str(path),
            cv2.IMREAD_GRAYSCALE,
        )

        if image is None:
            raise RuntimeError(
                str(path)
            )

        return np.zeros_like(
            image,
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

    return (
        mask > 0
    ).astype(
        np.uint8
    )


def add_image(
    stat,
    map64,
    mask,
):

    h, w = mask.shape

    score = cv2.resize(
        map64.astype(
            np.float32
        ),
        (
            w,
            h,
        ),
        interpolation=cv2.INTER_LINEAR,
    )

    positive = (
        mask > 0
    )

    negative = ~positive

    hn, _ = np.histogram(
        score[
            negative
        ],
        bins=edges,
    )

    stat.neg += hn

    if positive.any():

        num, labels = cv2.connectedComponents(
            mask.astype(
                np.uint8
            ),
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

            hr, _ = np.histogram(
                values,
                bins=edges,
            )

            stat.region += (
                hr.astype(
                    np.float64
                )
                / len(values)
            )

            stat.regions += 1


def compute_pro(
    stat,
):

    if (
        stat.regions == 0
        or stat.neg.sum() == 0
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
        / stat.regions,
    ])

    fpr = (
        fp
        / stat.neg.sum()
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
        / MAX_FPR
    )


# ============================================================
# Baselines + routed budgets
# ============================================================

METHODS = [
    "r256",
    "r1024",
] + [
    f"route_{int(b*10000):04d}"
    for b in BUDGETS
]


stats_all = {
    m: Stats()
    for m in METHODS
}

stats_q1 = {
    m: Stats()
    for m in METHODS
}


for n, row in enumerate(
    rows,
    1,
):

    idx = row[
        "map_index"
    ]

    low = np.asarray(
        low_maps[
            idx
        ],
        dtype=np.float32,
    )

    high = np.asarray(
        high_maps[
            idx
        ],
        dtype=np.float32,
    )

    candidate_maps = {
        "r256":
            low_to_high(
                low
            ),

        "r1024":
            high,
    }

    for budget in BUDGETS:

        key = (
            f"route_"
            f"{int(budget*10000):04d}"
        )

        candidate_maps[
            key
        ] = routed_map(
            low,
            high,
            budget,
        )

    mask = load_mask(
        row
    )

    is_q1 = False

    if row["label"] == 1:

        q = quartile[
            (
                row["category"],
                row["instance_id"],
            )
        ]

        is_q1 = (
            q == "Q1"
        )

    for method, score_map in (
        candidate_maps.items()
    ):

        add_image(
            stats_all[
                method
            ],
            score_map,
            mask,
        )

        # Q1 uses all normal/background negatives,
        # but only Q1 defect regions.
        if row["label"] == 0:

            add_image(
                stats_q1[
                    method
                ],
                score_map,
                mask,
            )

        elif is_q1:

            add_image(
                stats_q1[
                    method
                ],
                score_map,
                mask,
            )

        else:

            # Still include defect-image background
            # in the shared negative population for Q1.
            zero_mask = np.zeros_like(
                mask,
                dtype=np.uint8,
            )

            add_image(
                stats_q1[
                    method
                ],
                score_map,
                zero_mask,
            )

    if (
        n % 50 == 0
        or n == len(rows)
    ):
        print(
            f"{n:3d}/{len(rows)}"
        )


results = []

for method in METHODS:

    results.append({
        "method":
            method,

        "all_aupro":
            compute_pro(
                stats_all[
                    method
                ]
            ),

        "q1_aupro":
            compute_pro(
                stats_q1[
                    method
                ]
            ),
    })


with DETAIL_CSV.open(
    "w",
    newline="",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=list(
            results[0].keys()
        ),
    )

    writer.writeheader()
    writer.writerows(
        results
    )


lookup = {
    x["method"]: x
    for x in results
}


pro256 = lookup[
    "r256"
]["all_aupro"]

pro1024 = lookup[
    "r1024"
]["all_aupro"]

full_gain = (
    pro1024
    -
    pro256
)


route25 = lookup[
    "route_2500"
]["all_aupro"]


if full_gain > 0:

    recovery25 = (
        route25
        -
        pro256
    ) / full_gain

else:

    recovery25 = np.nan


q1_improved = (
    lookup[
        "route_2500"
    ][
        "q1_aupro"
    ]
    >
    lookup[
        "r256"
    ][
        "q1_aupro"
    ]
)


recovery_signal = (
    np.isfinite(
        recovery25
    )
    and
    recovery25 >= 0.70
)

decision = (
    "GO"
    if (
        recovery_signal
        and
        q1_improved
    )
    else "NO_GO"
)


payload = {
    "protocol": {
        "router_input":
            "256-resolution anomaly map only",

        "router":
            "top anomaly-score coarse cells",

        "high_resolution_source":
            "precomputed 1024 map; map-level feasibility audit",

        "gt_used_for_routing":
            False,
    },

    "pre_registered_thresholds": {
        "budget":
            0.25,

        "minimum_full_gain_recovery":
            0.70,

        "q1_must_improve_over_256":
            True,
    },

    "observed": {
        "r256_all_aupro":
            float(
                pro256
            ),

        "r1024_all_aupro":
            float(
                pro1024
            ),

        "full_1024_gain":
            float(
                full_gain
            ),

        "route25_all_aupro":
            float(
                route25
            ),

        "route25_gain_recovery":
            float(
                recovery25
            ),

        "r256_q1_aupro":
            float(
                lookup[
                    "r256"
                ][
                    "q1_aupro"
                ]
            ),

        "route25_q1_aupro":
            float(
                lookup[
                    "route_2500"
                ][
                    "q1_aupro"
                ]
            ),
    },

    "signals": {
        "recovery_signal":
            bool(
                recovery_signal
            ),

        "q1_improved":
            bool(
                q1_improved
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


print()
print("=" * 95)
print("A3.1 SPARSE HIGH-RESOLUTION ROUTING")
print("=" * 95)

print(
    f"{'method':16s}"
    f"{'ALL-PRO':>12s}"
    f"{'Q1-PRO':>12s}"
)

print("-" * 40)

for r in results:

    print(
        f"{r['method']:16s}"
        f"{r['all_aupro']:12.4f}"
        f"{r['q1_aupro']:12.4f}"
    )


print()
print("[25% BUDGET]")

print(
    "full 1024 gain       :",
    f"{full_gain:+.4f}"
)

print(
    "25% routed PRO       :",
    f"{route25:.4f}"
)

print(
    "gain recovery        :",
    f"{recovery25:.4f}"
)

print(
    "Q1 improves over 256 :",
    q1_improved
)

print()
print("DECISION :", decision)

print()
print("DETAIL  :", DETAIL_CSV)
print("SUMMARY :", SUMMARY_JSON)

print("=" * 95)
