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

A10 = Path("orbitad/results/a10_multilayer_v1")
A9 = Path("orbitad/results/a9_context_reference_v1")
assert (A10 / "EVALUATION_COMPLETE.json").exists()
A6_ROOT = Path("orbitad/results/a12_cached_multires_pilot_v1")
A6_ROOT.mkdir(parents=True, exist_ok=False)
PUBLIC_META = A10 / "public_meta.csv"
OUT_DETAIL = A6_ROOT / "native_category_condition.csv"
METHODS = [
    "pca16", "full512", "patch640", "pca_patch_equal",
    "full_patch_equal", "all_three_equal", "pca_patch_geom",
]
BASES = ["pca16", "full512", "patch640"]
protocol = {
    "primary":"full_patch_equal", "control":"patch640",
    "sensitivity":["all_three_equal", "pca_patch_geom"],
    "weights":"equal only", "weight_search":False,
    "calibration":"category/method median of per-image q95; each image sampled to 32x32 by bilinear resize",
    "calibration_data":"validation/good only", "fusion_grid":"native image resolution",
    "metric":"native GT AU-PRO@0.05", "public_role":"development",
    "postprocessing":False,
    "architecture":"cached multi-resolution pilot; fixed arithmetic fusion is evidence, not a claimed innovation"
}
(A6_ROOT / "protocol.json").write_text(json.dumps(protocol,indent=2))
# Freeze scales before reading public labels or maps.
validation=list(csv.DictReader(open(A10 / "validation_meta.csv")))
old_validation=list(csv.DictReader(open(A9 / "validation_meta.csv")))
assert [(r["map_index"],r["image_path"]) for r in validation] == [(r["map_index"],r["image_path"]) for r in old_validation]
vpca=np.load(A9 / "validation_pca16.npy",mmap_mode="r")
scale_values=defaultdict(list)
for row in validation:
    assert int(row["label"])==0 and "/validation/good/" in "/"+row["image_path"]
    for method in BASES:
        i=int(row["map_index"])
        a=vpca[i] if method=="pca16" else np.load(A10 / method / row["category"] / "validation" / f"{i:05d}.npy")
        a=cv2.resize(a,(32,32),interpolation=cv2.INTER_LINEAR)
        assert np.isfinite(a).all()
        scale_values[row["category"],method].append(float(np.quantile(a,.95)))
scales={cat:{method:max(float(np.median(scale_values[cat,method])),1e-12) for method in BASES} for cat in sorted({r["category"] for r in validation})}
(A6_ROOT / "frozen_scales.json").write_text(json.dumps(scales,indent=2))
print("CALIBRATION_FROZEN",flush=True)
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
old_public=list(csv.DictReader(open(A9 / "public_meta.csv")))
assert [(str(r["map_index"]),r["image_path"]) for r in rows] == [(r["map_index"],r["image_path"]) for r in old_public]
row_lookup={r["map_index"]:r for r in rows}
pca_cache=np.load(A9 / "public_pca16.npy",mmap_mode="r")
def base_map(method,row):
    i=row["map_index"]
    return pca_cache[i] if method=="pca16" else np.load(A10 / method / row["category"] / "public" / f"{i:05d}.npy")
class MapReader:
    def __init__(self,method): self.method=method
    def __getitem__(self,i):
        row=row_lookup[i]
        if self.method in BASES: return base_map(self.method,row)
        with Image.open(DATA_ROOT / row["image_path"]) as im: size=im.size
        if self.method == "pca_patch_equal":
            components = ("pca16", "patch640")
        elif self.method == "full_patch_equal":
            components = ("full512", "patch640")
        elif self.method == "all_three_equal":
            components = ("pca16", "full512", "patch640")
        elif self.method == "pca_patch_geom":
            components = ("pca16", "patch640")
        else:
            raise KeyError(self.method)
        parts=[cv2.resize(np.asarray(base_map(m,row),dtype=np.float32),size,interpolation=cv2.INTER_LINEAR)/scales[row["category"]][m] for m in components]
        if self.method == "pca_patch_geom":
            result=np.sqrt(np.maximum(parts[0],0.0)*np.maximum(parts[1],0.0))
        else:
            result=sum(parts)/len(parts)
        assert np.isfinite(result).all()
        return result
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


# Reproducibility of the three cached baselines with the same evaluator.
previous=json.loads((A10 / "native_summary.json").read_text())
errors={m:abs(summary[m]["all_condition_category_macro"]-previous[m]["all_condition_category_macro"]) for m in BASES}
(A6_ROOT / "baseline_reproducibility.json").write_text(json.dumps(errors,indent=2))
assert max(errors.values())<1e-6, errors
print("BASELINES_REPRODUCED",errors,flush=True)
(A6_ROOT / "EVALUATION_COMPLETE.json").write_text(json.dumps({"status":"complete","metric":"native GT AU-PRO@0.05","histogram_bins":HIST_BINS}))
