#!/usr/bin/env python3
"""Complete paper-facing metrics from existing cached scores/maps.

This script does NOT retrain or rerun the backbone. It reuses:
1) historical per-image AD2 scores already saved in the repository;
2) AD2 global/tiled component maps used by the validated cross-resolution experiment;
3) MVTec AD-14 global/tiled component maps;
4) MVTec Cable global/tiled component maps.

It computes a unified paper-facing metric set:
- image AUROC, image AP, image F1-max
- pixel AUROC, pixel AP, pixel F1-max
- AU-PRO@0.05
- Tiny / Small / Large AU-PRO@0.05

Metrics are macro-averaged over categories where applicable.
Unsupported combinations (e.g. image-only score methods without pixel maps)
are kept blank in the final CSV instead of being fabricated.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import tifffile
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm


ROOT = Path(os.environ.get("IAD_REPO_ROOT", "/root/private_data/iad-vlm-anomaly"))
OUT = Path(
    os.environ.get(
        "IAD_PAPER_METRIC_OUT",
        str(ROOT / "cross_resolution_fusion/results/paper_metric_completion"),
    )
)

BINS = 65536
MAX_FPR = 0.05
EPS = 1e-12
BANDS = ("all", "tiny_le_0.1pct", "small_0.1_to_1pct", "large_gt_1pct")


def read_csv(path: Path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def f1_max(labels, scores):
    labels = np.asarray(labels, np.int64)
    scores = np.asarray(scores, np.float64)
    if len(np.unique(labels)) < 2:
        return float("nan"), float("nan")
    order = np.argsort(-scores, kind="mergesort")
    y = labels[order]
    s = scores[order]
    positives = int(labels.sum())
    tp = 0
    fp = 0
    best = 0.0
    best_threshold = float(s[0]) if len(s) else float("nan")
    for index in range(len(s)):
        if y[index] == 1:
            tp += 1
        else:
            fp += 1
        is_last_tie = index == len(s) - 1 or s[index + 1] != s[index]
        if not is_last_tie:
            continue
        fn = positives - tp
        denom = 2 * tp + fp + fn
        value = (2 * tp / denom) if denom else 0.0
        if value > best:
            best = float(value)
            best_threshold = float(s[index])
    return best, best_threshold


def safe_image_metrics(labels, scores):
    labels = np.asarray(labels, np.int64)
    scores = np.asarray(scores, np.float64)
    if len(np.unique(labels)) < 2:
        return {"image_auroc": np.nan, "image_ap": np.nan, "image_f1max": np.nan}
    f1, threshold = f1_max(labels, scores)
    return {
        "image_auroc": float(roc_auc_score(labels, scores)),
        "image_ap": float(average_precision_score(labels, scores)),
        "image_f1max": float(f1),
        "image_f1_threshold": float(threshold),
    }


def histogram_auroc(negative, positive):
    if negative.sum() == 0 or positive.sum() == 0:
        return float("nan")
    fp = np.r_[0.0, np.cumsum(negative[::-1], dtype=np.float64)]
    tp = np.r_[0.0, np.cumsum(positive[::-1], dtype=np.float64)]
    fpr = fp / negative.sum()
    tpr = tp / positive.sum()
    return float(np.trapezoid(tpr, fpr))


def histogram_ap(negative, positive):
    if positive.sum() == 0:
        return float("nan")
    fp = np.cumsum(negative[::-1], dtype=np.float64)
    tp = np.cumsum(positive[::-1], dtype=np.float64)
    recall = tp / positive.sum()
    precision = tp / np.maximum(tp + fp, 1.0)
    recall_prev = np.r_[0.0, recall[:-1]]
    return float(np.sum((recall - recall_prev) * precision))


def histogram_f1max(negative, positive):
    if positive.sum() == 0:
        return float("nan")
    fp = np.cumsum(negative[::-1], dtype=np.float64)
    tp = np.cumsum(positive[::-1], dtype=np.float64)
    fn = positive.sum() - tp
    denom = 2 * tp + fp + fn
    f1 = np.divide(2 * tp, denom, out=np.zeros_like(tp), where=denom > 0)
    return float(np.nanmax(f1))


def aupro(negative, region_hist, regions):
    if negative.sum() == 0 or regions == 0:
        return float("nan")
    fpr = np.r_[0.0, np.cumsum(negative[::-1], dtype=np.float64)] / negative.sum()
    pro = np.r_[0.0, np.cumsum(region_hist[::-1], dtype=np.float64) / regions]
    keep = fpr <= MAX_FPR
    x, y = fpr[keep], pro[keep]
    if len(x) == 0:
        return float("nan")
    if x[-1] < MAX_FPR:
        index = np.searchsorted(fpr, MAX_FPR, side="right")
        if index < len(fpr):
            weight = (MAX_FPR - fpr[index - 1]) / max(fpr[index] - fpr[index - 1], EPS)
            x = np.r_[x, MAX_FPR]
            y = np.r_[y, pro[index - 1] + weight * (pro[index] - pro[index - 1])]
    return float(np.trapezoid(y, x) / MAX_FPR)


def band_for(region_pixels, image_pixels):
    ratio = region_pixels / image_pixels
    if ratio <= 0.001:
        return "tiny_le_0.1pct"
    if ratio <= 0.01:
        return "small_0.1_to_1pct"
    return "large_gt_1pct"


class PixelStats:
    def __init__(self):
        self.negative = np.zeros(BINS, np.int64)
        self.positive = np.zeros(BINS, np.int64)
        self.region = {band: np.zeros(BINS, np.float64) for band in BANDS}
        self.regions = {band: 0 for band in BANDS}


def add_map_to_stats(stat: PixelStats, score, mask, scale):
    if score.shape != mask.shape:
        score = cv2.resize(score, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_LINEAR)
    bins = np.minimum((np.maximum(score, 0) * scale).astype(np.int64), BINS - 1)
    stat.negative += np.bincount(bins[mask == 0], minlength=BINS)
    stat.positive += np.bincount(bins[mask == 1], minlength=BINS)

    count, labels = cv2.connectedComponents(mask.astype(np.uint8), 8)
    for region_id in range(1, count):
        region = labels == region_id
        pixels = int(region.sum())
        if pixels <= 0:
            continue
        band = band_for(pixels, mask.size)
        contribution = np.bincount(bins[region], minlength=BINS) / pixels
        for target in ("all", band):
            stat.region[target] += contribution
            stat.regions[target] += 1


def summarize_map_stats(stats_by_category):
    rows = []
    for category, stat in sorted(stats_by_category.items()):
        row = {
            "category": category,
            "pixel_auroc": histogram_auroc(stat.negative, stat.positive),
            "pixel_ap": histogram_ap(stat.negative, stat.positive),
            "pixel_f1max": histogram_f1max(stat.negative, stat.positive),
        }
        for band in BANDS:
            row[band] = aupro(stat.negative, stat.region[band], stat.regions[band])
        rows.append(row)
    macro = {}
    for key in ("pixel_auroc", "pixel_ap", "pixel_f1max", *BANDS):
        values = [r[key] for r in rows if np.isfinite(r[key])]
        macro[key] = float(np.mean(values)) if values else float("nan")
    return macro, rows


def macro_image_metrics(per_category):
    result = {}
    detail = []
    for category, data in sorted(per_category.items()):
        metrics = safe_image_metrics(data["labels"], data["scores"])
        detail.append({"category": category, **metrics})
    for key in ("image_auroc", "image_ap", "image_f1max"):
        vals = [row[key] for row in detail if np.isfinite(row[key])]
        result[key] = float(np.mean(vals)) if vals else float("nan")
    return result, detail


# ---------------------------------------------------------------------------
# Historical AD2 image-score methods
# ---------------------------------------------------------------------------

def evaluate_ad2_image_score_history():
    path_512 = ROOT / "model_comparison/stage24_ad2_highres/evaluation/predictions_with_evaluation_labels.csv"
    path_dino = ROOT / "tiny_defects/results/stage25_tiny_defects/dinov2_ad2_v1/evaluation_predictions.csv"
    if not path_512.exists() or not path_dino.exists():
        return [], {"warning": "historical AD2 score CSVs not found"}

    rows_512 = read_csv(path_512)
    rows_dino = read_csv(path_dino)

    specs = (
        ("PatchCore256", rows_512, "D256"),
        ("PatchCore512", rows_512, "D512"),
        ("DINOv2 top1", rows_dino, "dino_top1"),
        ("Frozen score fusion", rows_dino, "fusion_frozen"),
    )
    output = []
    for method, rows, score_key in specs:
        grouped = defaultdict(lambda: {"labels": [], "scores": []})
        for row in rows:
            grouped[row["category"]]["labels"].append(int(row["label"]))
            grouped[row["category"]]["scores"].append(float(row[score_key]))
        macro, detail = macro_image_metrics(grouped)
        output.append({
            "protocol": "AD2 image-level",
            "method": method,
            **macro,
            "pixel_auroc": np.nan,
            "pixel_ap": np.nan,
            "pixel_f1max": np.nan,
            "aupro_0_05": np.nan,
            "tiny": np.nan,
            "small": np.nan,
            "large": np.nan,
            "note": "image-score method; pixel/region metrics not defined for this saved output",
        })
    return output, {}


# ---------------------------------------------------------------------------
# AD2 cross-resolution maps (paper main result)
# ---------------------------------------------------------------------------

AD2_META = ROOT / "orbitad/results/a10_multilayer_v1/public_meta.csv"
AD2_NORMAL = ROOT / "orbitad/results/a68_normal_only_resolution_router_v1/normal_image_statistics.csv"
AD2_A64 = ROOT / "ad2_model_zoo/results/a64_superad_reg4_ad2_all_true672_token_preserving_v1"
AD2_A45 = ROOT / "ad2_model_zoo/results/a45_superad_reg4_public_v1"
AD2_A60 = ROOT / "ad2_model_zoo/results/a60_superad_reg4_sheetmetal_true672_tiled_query_v1/sheet_metal/component_maps/seed=0/sheet_metal"


def ad2_component_paths(row):
    rel = Path(row["image_path"])
    if row["category"] == "sheet_metal":
        stem = AD2_A60 / rel.parent.name / rel.stem
        return Path(str(stem) + "_global.tiff"), Path(str(stem) + "_tiled.tiff")
    global_path = (AD2_A45 / row["category"] / "anomaly_maps/seed=0" / row["category"] /
                   "test" / rel.parent.name / f"{rel.stem}.tiff")
    tiled_stem = (AD2_A64 / row["category"] / "component_maps/seed=0" / row["category"] /
                  rel.parent.name / rel.stem)
    return global_path, Path(str(tiled_stem) + "_tiled.tiff")


def ad2_calibration():
    rows = read_csv(AD2_NORMAL)
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["category"]].append(row)
    return {
        category: {
            "global_q99": float(np.median([float(r["global_q99"]) for r in values])),
            "tiled_q99": float(np.median([float(r["tiled_q99"]) for r in values])),
        }
        for category, values in grouped.items()
    }


def ad2_maps(row, calibration):
    gp, tp = ad2_component_paths(row)
    g = np.asarray(tifffile.imread(gp), np.float32)
    t = np.asarray(tifffile.imread(tp), np.float32)
    c = calibration[row["category"]]
    return {
        "Global 448": g,
        "Local 672": t,
        "Raw mean": 0.5 * (g + t),
        "Ours: q99 fusion": 0.5 * (
            g / max(c["global_q99"], EPS)
            + t / max(c["tiled_q99"], EPS)
        ),
    }


def ad2_mask(row, sample_shape):
    if int(row["label"]) == 0:
        return np.zeros(sample_shape, np.uint8)
    path = ROOT / "datasets/MVTec_AD_2" / row["mask_path"]
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(path)
    return (mask > 0).astype(np.uint8)


def evaluate_ad2_cross_resolution():
    required = (AD2_META, AD2_NORMAL)
    if not all(path.exists() for path in required):
        return [], {"warning": "AD2 metadata/calibration files not found"}

    rows = read_csv(AD2_META)
    calibration = ad2_calibration()
    methods = ("Global 448", "Local 672", "Raw mean", "Ours: q99 fusion")

    missing = [
        str(path)
        for row in rows
        for path in ad2_component_paths(row)
        if not path.exists()
    ]
    if missing:
        return [], {"warning": f"AD2 component maps missing ({len(missing)}); first={missing[:3]}"}

    maxima = defaultdict(float)
    image_data = {
        method: defaultdict(lambda: {"labels": [], "scores": []})
        for method in methods
    }

    for row in tqdm(rows, desc="AD2 pass 1/2: image scores", unit="img", dynamic_ncols=True):
        maps = ad2_maps(row, calibration)
        label = int(row["label"])
        category = row["category"]
        for method, score in maps.items():
            maxima[(method, category)] = max(maxima[(method, category)], float(np.nanmax(score)))
            image_data[method][category]["labels"].append(label)
            image_data[method][category]["scores"].append(float(np.quantile(score, 0.999)))

    pixel_stats = {
        method: defaultdict(PixelStats)
        for method in methods
    }
    for row in tqdm(rows, desc="AD2 pass 2/2: pixel/region", unit="img", dynamic_ncols=True):
        maps = ad2_maps(row, calibration)
        sample = maps["Global 448"]
        mask = ad2_mask(row, sample.shape)
        category = row["category"]
        for method, score in maps.items():
            maximum = maxima[(method, category)]
            scale = BINS / max(maximum * 1.001, maximum + 1e-6)
            add_map_to_stats(pixel_stats[method][category], score, mask, scale)

    output = []
    detail_payload = {}
    for method in methods:
        image_macro, image_detail = macro_image_metrics(image_data[method])
        pixel_macro, pixel_detail = summarize_map_stats(pixel_stats[method])
        output.append({
            "protocol": "AD2 localization",
            "method": method,
            **image_macro,
            "pixel_auroc": pixel_macro["pixel_auroc"],
            "pixel_ap": pixel_macro["pixel_ap"],
            "pixel_f1max": pixel_macro["pixel_f1max"],
            "aupro_0_05": pixel_macro["all"],
            "tiny": pixel_macro["tiny_le_0.1pct"],
            "small": pixel_macro["small_0.1_to_1pct"],
            "large": pixel_macro["large_gt_1pct"],
            "note": "image metrics: category macro; pixel/region metrics: category macro",
        })
        detail_payload[method] = {"image": image_detail, "pixel": pixel_detail}
    return output, detail_payload


# ---------------------------------------------------------------------------
# MVTec AD-14 cross-resolution maps
# ---------------------------------------------------------------------------

MVTEC_DATA = ROOT / "datasets/MVTecAD"
MVTEC_A73 = ROOT / "ad2_model_zoo/results/a73_mvtec14_dual_resolution_v1"
MVTEC_CATEGORIES = (
    "grid", "capsule", "transistor", "zipper", "carpet", "leather", "tile",
    "wood", "bottle", "hazelnut", "metal_nut", "pill", "screw", "toothbrush",
    "cable",
)

# Cable is one of the 15 MVTec AD categories and belongs in the same external
# evaluation. Its component maps come from the A59 (test) and A71 (train/good)
# legacy runs, whose directory layout differs from A73. A grid-pitch check on the
# stored maps gives 21.17 px for cable tiled and 31.75 px for cable global,
# identical to the certified A73 reference (672 and 448 at 1024 px), so the maps
# are read in place rather than regenerated.
LEGACY_MVTEC_MAP_ROOTS = {
    "cable": {
        "train": ROOT / "ad2_model_zoo/results/a71_mvtec_cable_train_normal_dual_resolution_v1",
        "test": ROOT / "ad2_model_zoo/results/a59_superad_reg4_mvtec_cable_tiled_query_v1",
    },
}


def mvtec_component_paths(split, category, anomaly_type, stem):
    if category in LEGACY_MVTEC_MAP_ROOTS:
        root = LEGACY_MVTEC_MAP_ROOTS[category][split]
        folder = root / category / "component_maps/seed=0" / category / anomaly_type
    else:
        folder = MVTEC_A73 / split / category / "component_maps/seed=0" / category / anomaly_type
    return folder / f"{stem}_global.tiff", folder / f"{stem}_tiled.tiff"


def mvtec_read_pair(split, category, anomaly_type, stem):
    gp, tp = mvtec_component_paths(split, category, anomaly_type, stem)
    return np.asarray(tifffile.imread(gp), np.float32), np.asarray(tifffile.imread(tp), np.float32)


def mvtec_calibration(category):
    values_g, values_t = [], []
    for image_path in sorted((MVTEC_DATA / category / "train/good").glob("*.png")):
        g, t = mvtec_read_pair("train", category, "good", image_path.stem)
        values_g.append(float(np.quantile(g, 0.99)))
        values_t.append(float(np.quantile(t, 0.99)))
    if not values_g:
        raise RuntimeError(f"No MVTec train/good maps for {category}")
    return {"global_q99": float(np.median(values_g)), "tiled_q99": float(np.median(values_t))}


def mvtec_samples(category):
    return [
        (folder.name, path)
        for folder in sorted((MVTEC_DATA / category / "test").iterdir())
        if folder.is_dir()
        for path in sorted(folder.glob("*.png"))
    ]


def mvtec_mask(category, anomaly_type, image_path):
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(image_path)
    if anomaly_type == "good":
        return np.zeros_like(image, np.uint8)
    path = MVTEC_DATA / category / "ground_truth" / anomaly_type / f"{image_path.stem}_mask.png"
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(path)
    return (mask > 0).astype(np.uint8)


def mvtec_methods(g, t, calibration):
    # Every MVTec AD image is square, so the tiler (tile_size = min(h, w)) emits a
    # single tile equal to the whole image. The 672 branch is therefore a
    # higher-resolution whole-image branch, not a local view, and is labelled
    # Res672 instead of Local 672.
    return {
        "Global 448": g,
        "Res672": t,
        "Raw mean": 0.5 * (g + t),
        "q99 fusion": 0.5 * (
            g / max(calibration["global_q99"], EPS)
            + t / max(calibration["tiled_q99"], EPS)
        ),
    }


def evaluate_mvtec15():
    if not MVTEC_A73.exists():
        return [], {"warning": f"MVTec-15 map directory not found: {MVTEC_A73}"}

    methods = ("Global 448", "Res672", "Raw mean", "q99 fusion")
    calibration = {c: mvtec_calibration(c) for c in MVTEC_CATEGORIES}
    maxima = defaultdict(float)
    image_data = {
        method: defaultdict(lambda: {"labels": [], "scores": []})
        for method in methods
    }

    all_samples = [(c, t, p) for c in MVTEC_CATEGORIES for t, p in mvtec_samples(c)]
    for category, anomaly_type, image_path in tqdm(
        all_samples, desc="MVTec15 pass 1/2: image scores", unit="img", dynamic_ncols=True
    ):
        g, t = mvtec_read_pair("test", category, anomaly_type, image_path.stem)
        maps = mvtec_methods(g, t, calibration[category])
        label = int(anomaly_type != "good")
        for method, score in maps.items():
            maxima[(method, category)] = max(maxima[(method, category)], float(np.nanmax(score)))
            image_data[method][category]["labels"].append(label)
            image_data[method][category]["scores"].append(float(np.quantile(score, 0.999)))

    pixel_stats = {
        method: defaultdict(PixelStats)
        for method in methods
    }
    for category, anomaly_type, image_path in tqdm(
        all_samples, desc="MVTec15 pass 2/2: pixel/region", unit="img", dynamic_ncols=True
    ):
        g, t = mvtec_read_pair("test", category, anomaly_type, image_path.stem)
        maps = mvtec_methods(g, t, calibration[category])
        mask = mvtec_mask(category, anomaly_type, image_path)
        for method, score in maps.items():
            maximum = maxima[(method, category)]
            scale = BINS / max(maximum * 1.001, maximum + 1e-6)
            add_map_to_stats(pixel_stats[method][category], score, mask, scale)

    output = []
    details = {}
    for method in methods:
        image_macro, image_detail = macro_image_metrics(image_data[method])
        pixel_macro, pixel_detail = summarize_map_stats(pixel_stats[method])
        output.append({
            "protocol": "MVTec AD-15",
            "method": method,
            **image_macro,
            "pixel_auroc": pixel_macro["pixel_auroc"],
            "pixel_ap": pixel_macro["pixel_ap"],
            "pixel_f1max": pixel_macro["pixel_f1max"],
            "aupro_0_05": pixel_macro["all"],
            "tiny": pixel_macro["tiny_le_0.1pct"],
            "small": pixel_macro["small_0.1_to_1pct"],
            "large": pixel_macro["large_gt_1pct"],
            "note": (
                "all 15 MVTec AD categories; category-macro metrics; "
                "Res672 is the whole image at short edge 672 because all MVTec images are square"
            ),
        })
        details[method] = {"image": image_detail, "pixel": pixel_detail}
    return output, details


def blank(value):
    if value is None:
        return ""
    if isinstance(value, float) and not np.isfinite(value):
        return ""
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--only",
        choices=("all", "history", "ad2", "mvtec15"),
        default="all",
        help="Run only one evaluation block if desired.",
    )
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    details = {}

    jobs = []
    if args.only in ("all", "history"):
        jobs.append(("historical_ad2_image_scores", evaluate_ad2_image_score_history))
    if args.only in ("all", "ad2"):
        jobs.append(("ad2_cross_resolution", evaluate_ad2_cross_resolution))
    if args.only in ("all", "mvtec15"):
        jobs.append(("mvtec15_cross_resolution", evaluate_mvtec15))

    for name, fn in jobs:
        print(f"\n[{name}] start", flush=True)
        block_rows, block_details = fn()
        rows.extend(block_rows)
        details[name] = block_details
        print(f"[{name}] complete: {len(block_rows)} rows", flush=True)

    fields = [
        "protocol", "method",
        "image_auroc", "image_ap", "image_f1max",
        "pixel_auroc", "pixel_ap", "pixel_f1max",
        "aupro_0_05", "tiny", "small", "large", "note",
    ]
    with (OUT / "paper_main_table_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: blank(row.get(key, "")) for key in fields})

    payload = {
        "metric_definitions": {
            "image_score": "99.9th percentile of anomaly map for map-based methods",
            "image_f1max": "maximum F1 over all image-score thresholds within each category, then category macro",
            "pixel_f1max": "maximum F1 over pixel-score thresholds using 65,536-bin histograms",
            "pixel_ap": "histogram approximation of average precision with 65,536 bins",
            "aupro_0_05": "normalized area under PRO for pixel FPR <= 0.05",
            "tiny": "connected-region area <= 0.1% of image area",
            "small": "0.1% < connected-region area <= 1%",
            "large": "connected-region area > 1%",
        },
        "important": [
            "No backbone retraining or new feature extraction is performed.",
            "Blank values mean the saved method output does not support that metric or the required local artifact is absent.",
            "Image-only historical score methods are not assigned fabricated pixel/region metrics.",
        ],
        "details": details,
    }
    (OUT / "paper_main_table_metrics_details.json").write_text(
        json.dumps(payload, indent=2, allow_nan=True)
    )
    (OUT / "PAPER_METRIC_COMPLETION_COMPLETE.json").write_text(
        json.dumps({"status": "complete", "rows": len(rows)}, indent=2)
    )

    print("\nSaved:")
    print(OUT / "paper_main_table_metrics.csv")
    print(OUT / "paper_main_table_metrics_details.json")
    print(OUT / "PAPER_METRIC_COMPLETION_COMPLETE.json")


if __name__ == "__main__":
    main()
