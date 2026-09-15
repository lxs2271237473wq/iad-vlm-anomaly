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

A6_ROOT = Path("orbitad/results/a10_multilayer_v1")
assert (A6_ROOT / "MAPS_COMPLETE.json").exists()
PUBLIC_META = A6_ROOT / "public_meta.csv"
OUT_DETAIL = A6_ROOT / "native_category_condition.csv"
METHODS = ["pca16", "full512", "patch640"]
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

from PIL import Image
class MapReader:
    def __init__(self, method):
        self.method=method
        self.cache=np.load("orbitad/results/a9_context_reference_v1/public_pca16.npy",mmap_mode="r") if method=="pca16" else None
    def __getitem__(self,i):
        if self.cache is not None: return self.cache[i]
        row=next(r for r in rows if r["map_index"]==i)
        return np.load(A6_ROOT/self.method/row["category"]/"public"/f"{i:05d}.npy")
maps={m:MapReader(m) for m in METHODS}
EDGES={}
for method in METHODS:
    maximum=0.
    for row in rows:
        a=maps[method][row["map_index"]]
        assert np.isfinite(a).all()
        maximum=max(maximum,float(a.max()))
    EDGES[method]=np.linspace(0,max(maximum*1.001,maximum+1e-6),HIST_BINS+1)
def upsample(x):
    return cv2.resize(np.asarray(x,dtype=np.float32),(mask.shape[1],mask.shape[0]),interpolation=cv2.INTER_LINEAR)
def load_mask(row):
    if row["label"]==0:
        with Image.open(DATA_ROOT/row["image_path"]) as im: w,h=im.size
        return np.zeros((h,w),dtype=np.uint8)
    m=cv2.imread(str(DATA_ROOT/row["mask_path"]),cv2.IMREAD_GRAYSCALE)
    assert m is not None
    return (m>0).astype(np.uint8)
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



summary = {}
for method in METHODS:
    selected = [r for r in detail_rows if r["method"] == method]
    bycat = {}
    for r in selected:
        bycat.setdefault(r["category"], []).append(r["aupro_0_05"])
    summary[method] = {"all_condition_category_macro": float(np.mean([np.mean(v) for v in bycat.values()]))}
(A6_ROOT / "native_summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))

(A6_ROOT / "EVALUATION_COMPLETE.json").write_text(json.dumps({"status":"complete","metric":"native GT AU-PRO@0.05","histogram_bins":HIST_BINS}))
