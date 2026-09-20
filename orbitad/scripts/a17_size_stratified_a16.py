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
A12 = Path("orbitad/results/a12_cached_multires_pilot_v1")
A16 = Path("orbitad/results/a16_normal_only_component_soft_router_v1")
assert (A10 / "EVALUATION_COMPLETE.json").exists()
assert (A12 / "EVALUATION_COMPLETE.json").exists()
assert (A16 / "EVALUATION_COMPLETE.json").exists()
A6_ROOT = Path("orbitad/results/a17_size_stratified_a16_v1")
A6_ROOT.mkdir(parents=True, exist_ok=False)
PUBLIC_META = A10 / "public_meta.csv"
OUT_DETAIL = A6_ROOT / "native_category_condition.csv"
METHODS = ["patch640", "full_patch_equal", "normal_only_component_soft_router"]
BASES = ["full512", "patch640"]
SIZE_BANDS = ("all", "tiny_le_0.1pct", "small_0.1_to_1pct", "large_gt_1pct")
protocol = {
    "primary":"normal_only_component_soft_router", "controls":["patch640","full_patch_equal"],
    "weights":[0.5,0.5], "weight_search":False,
    "region_size_bands":{"tiny_le_0.1pct":"area/image_area <= 0.001","small_0.1_to_1pct":"0.001 < area/image_area <= 0.01","large_gt_1pct":"area/image_area > 0.01"},
    "calibration":"category/method median of per-image q95; each image sampled to 32x32 by bilinear resize",
    "calibration_data":"validation/good only", "fusion_grid":"native image resolution",
    "metric":"native GT AU-PRO@0.05", "public_role":"development",
    "postprocessing":False, "analysis":"A16 component soft router stratified by GT connected-component area"
}
(A6_ROOT / "protocol.json").write_text(json.dumps(protocol,indent=2))
# Freeze scales before reading public labels or maps.
scales=json.loads((A12 / "frozen_scales.json").read_text())
route_thresholds=json.loads((A16 / "route_thresholds.json").read_text())
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
old_public=list(csv.DictReader(open(A10 / "public_meta.csv")))
assert [(str(r["map_index"]),r["image_path"]) for r in rows] == [(r["map_index"],r["image_path"]) for r in old_public]
row_lookup={r["map_index"]:r for r in rows}
def base_map(method,row):
    i=row["map_index"]
    return np.load(A10 / method / row["category"] / "public" / f"{i:05d}.npy")
def patch_and_alpha(row):
    patch=np.asarray(base_map("patch640",row),dtype=np.float32)/scales[row["category"]]["patch640"]
    binary=(patch>route_thresholds[row["category"]]).astype(np.uint8)
    num,labels,stats,_=cv2.connectedComponentsWithStats(binary,connectivity=8)
    alpha=np.zeros_like(patch,dtype=np.float32)
    for rid in range(1,num):
        ratio=float(stats[rid,cv2.CC_STAT_AREA]/binary.size)
        alpha[labels==rid]=np.exp(-ratio/0.001)
    return patch,alpha
class MapReader:
    def __init__(self,method): self.method=method
    def __getitem__(self,i):
        row=row_lookup[i]
        if self.method in BASES: return base_map(self.method,row)
        with Image.open(DATA_ROOT / row["image_path"]) as im: size=im.size
        patch,alpha=patch_and_alpha(row)
        full=np.asarray(base_map("full512",row),dtype=np.float32)/scales[row["category"]]["full512"]
        patch_native=cv2.resize(patch,size,interpolation=cv2.INTER_LINEAR)
        fusion=0.5*(cv2.resize(full,size,interpolation=cv2.INTER_LINEAR)+patch_native)
        if self.method=="full_patch_equal":
            result=fusion
        elif self.method=="normal_only_component_soft_router":
            alpha_native=cv2.resize(alpha,size,interpolation=cv2.INTER_LINEAR)
            result=fusion+alpha_native*(patch_native-fusion)
        else:
            raise KeyError(self.method)
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
    size_band,
):

    key = (
        method,
        category,
        condition,
        size_band,
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


def region_size_band(area, image_area):
    ratio = area / image_area
    if ratio <= 0.001:
        return "tiny_le_0.1pct"
    if ratio <= 0.01:
        return "small_0.1_to_1pct"
    return "large_gt_1pct"


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

        # False-positive distribution
        hn, _ = np.histogram(
            score[
                negative
            ],
            bins=EDGES[method],
        )

        for size_band in SIZE_BANDS:
            get_stats(method,category,condition,size_band).neg += hn


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
                    bins=EDGES[method],
                )

                contribution=h.astype(np.float64)/len(values)
                band=region_size_band(len(values),mask.size)
                for size_band in ("all",band):
                    g=get_stats(method,category,condition,size_band)
                    g.region += contribution
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
    size_band,
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

        "size_band":
            size_band,

        "regions":
            g.regions,

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
    summary[method] = {}
    for size_band in SIZE_BANDS:
        selected = [r for r in detail_rows if r["method"] == method and r["size_band"] == size_band and np.isfinite(r["aupro_0_05"])]
        bycat = {}
        for r in selected:
            bycat.setdefault(r["category"], []).append(r["aupro_0_05"])
        summary[method][size_band] = {
            "category_macro_over_available_conditions": float(np.mean([np.mean(v) for v in bycat.values()])),
            "categories": len(bycat),
            "category_conditions": len(selected),
            "regions": int(sum(r["regions"] for r in selected)),
        }
(A6_ROOT / "native_summary.json").write_text(json.dumps(summary, indent=2))
previous=json.loads((A16 / "native_summary.json").read_text())
errors={m:abs(summary[m]["all"]["category_macro_over_available_conditions"]-previous[m]["all_condition_category_macro"]) for m in METHODS}
(A6_ROOT / "baseline_reproducibility.json").write_text(json.dumps(errors,indent=2))
assert max(errors.values())<1e-6, errors
(A6_ROOT / "EVALUATION_COMPLETE.json").write_text(json.dumps({"status":"complete","metric":"native GT AU-PRO@0.05","histogram_bins":HIST_BINS}))
print(json.dumps(summary, indent=2),flush=True)
print("A16_REPRODUCED",errors,flush=True)
