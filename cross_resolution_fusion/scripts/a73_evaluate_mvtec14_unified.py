#!/usr/bin/env python3
"""Unified four-method evaluation of the completed A73 MVTec-14 maps."""

from __future__ import annotations

import csv
import json
import os
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import tifffile
from sklearn.metrics import roc_auc_score


ROOT = Path(os.environ.get("IAD_REPO_ROOT", "/root/private_data/iad-vlm-anomaly"))
DATA = ROOT / "datasets/MVTecAD"
A73 = ROOT / "ad2_model_zoo/results/a73_mvtec14_dual_resolution_v1"
OUT = ROOT / "orbitad/results/a73_mvtec14_unified_eval_v1"
CATEGORIES = (
    "grid", "capsule", "transistor", "zipper", "carpet", "leather", "tile",
    "wood", "bottle", "hazelnut", "metal_nut", "pill", "screw", "toothbrush",
)
METHODS = ("global_448", "tiled_672", "raw_mean", "q99_mean")
METRICS = ("aupro_0_05", "pixel_auroc", "image_auroc")
MAX_FPR = 0.05
BINS = 65536
BOOTSTRAPS = 100000
SEED = 20260921


def component_paths(split: str, category: str, anomaly_type: str, stem: str):
    folder = A73 / split / category / "component_maps/seed=0" / category / anomaly_type
    return folder / f"{stem}_global.tiff", folder / f"{stem}_tiled.tiff"


def read_pair(split: str, category: str, anomaly_type: str, stem: str):
    global_path, tiled_path = component_paths(split, category, anomaly_type, stem)
    return (np.asarray(tifffile.imread(global_path), np.float32),
            np.asarray(tifffile.imread(tiled_path), np.float32))


def calibrate_category(category: str):
    folder = DATA / category / "train/good"
    images = sorted(folder.glob("*.png"))
    if not images:
        raise RuntimeError(f"No train/good images for {category}")
    global_q99, tiled_q99 = [], []
    missing = []
    for image_path in images:
        paths = component_paths("train", category, "good", image_path.stem)
        if not all(path.exists() for path in paths):
            missing.extend(str(path) for path in paths if not path.exists())
            continue
        global_map, tiled_map = read_pair("train", category, "good", image_path.stem)
        global_q99.append(float(np.quantile(global_map, .99)))
        tiled_q99.append(float(np.quantile(tiled_map, .99)))
    if missing:
        raise FileNotFoundError(f"{category}: missing {len(missing)} train maps; first={missing[:3]}")
    return {
        "category": category,
        "normal_images": len(images),
        "global_q99": float(np.median(global_q99)),
        "tiled_q99": float(np.median(tiled_q99)),
        "global_q99_min": float(np.min(global_q99)),
        "global_q99_max": float(np.max(global_q99)),
        "tiled_q99_min": float(np.min(tiled_q99)),
        "tiled_q99_max": float(np.max(tiled_q99)),
    }


def method_maps(global_map, tiled_map, calibration):
    eps = 1e-12
    return {
        "global_448": global_map,
        "tiled_672": tiled_map,
        "raw_mean": .5 * (global_map + tiled_map),
        "q99_mean": .5 * (
            global_map / max(calibration["global_q99"], eps)
            + tiled_map / max(calibration["tiled_q99"], eps)
        ),
    }


def samples_for(category: str):
    return [
        (folder.name, image_path)
        for folder in sorted((DATA / category / "test").iterdir()) if folder.is_dir()
        for image_path in sorted(folder.glob("*.png"))
    ]


def mask_for(category: str, anomaly_type: str, image_path: Path):
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(image_path)
    if anomaly_type == "good":
        return np.zeros_like(image, np.uint8)
    mask_path = DATA / category / "ground_truth" / anomaly_type / f"{image_path.stem}_mask.png"
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(mask_path)
    return (mask > 0).astype(np.uint8)


def aupro(negative_hist, region_hist, regions):
    if negative_hist.sum() == 0 or regions == 0:
        return float("nan")
    fpr = np.r_[0., np.cumsum(negative_hist[::-1], dtype=np.float64)] / negative_hist.sum()
    pro = np.r_[0., np.cumsum(region_hist[::-1], dtype=np.float64) / regions]
    keep = fpr <= MAX_FPR
    x, y = fpr[keep], pro[keep]
    if x[-1] < MAX_FPR:
        index = np.searchsorted(fpr, MAX_FPR, side="right")
        if index < len(fpr):
            weight = (MAX_FPR - fpr[index - 1]) / max(fpr[index] - fpr[index - 1], 1e-12)
            x = np.r_[x, MAX_FPR]
            y = np.r_[y, pro[index - 1] + weight * (pro[index] - pro[index - 1])]
    return float(np.trapezoid(y, x) / MAX_FPR)


def histogram_auc(negative_hist, positive_hist):
    if negative_hist.sum() == 0 or positive_hist.sum() == 0:
        return float("nan")
    fpr = np.r_[0., np.cumsum(negative_hist[::-1], dtype=np.float64)] / negative_hist.sum()
    tpr = np.r_[0., np.cumsum(positive_hist[::-1], dtype=np.float64)] / positive_hist.sum()
    return float(np.trapezoid(tpr, fpr))


def bootstrap_ci(values, rng):
    values = np.asarray(values, np.float64)
    draws = rng.choice(values, (BOOTSTRAPS, len(values)), replace=True).mean(axis=1)
    return [float(x) for x in np.quantile(draws, [.025, .975])]


def evaluate_category(category, calibration):
    samples = samples_for(category)
    missing = [
        str(path)
        for anomaly_type, image_path in samples
        for path in component_paths("test", category, anomaly_type, image_path.stem)
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(f"{category}: missing {len(missing)} test maps; first={missing[:3]}")

    maxima = {method: 0. for method in METHODS}
    labels = []
    image_scores = {method: [] for method in METHODS}
    for anomaly_type, image_path in samples:
        labels.append(int(anomaly_type != "good"))
        global_map, tiled_map = read_pair("test", category, anomaly_type, image_path.stem)
        for method, score in method_maps(global_map, tiled_map, calibration).items():
            maxima[method] = max(maxima[method], float(np.nanmax(score)))
            image_scores[method].append(float(np.quantile(score, .999)))

    scales = {method: BINS / max(value * 1.001, value + 1e-6)
              for method, value in maxima.items()}

    stats = {
        method: {
            "negative": np.zeros(BINS, np.int64),
            "positive": np.zeros(BINS, np.int64),
            "region": np.zeros(BINS, np.float64),
            "regions": 0,
        }
        for method in METHODS
    }
    for anomaly_type, image_path in samples:
        mask = mask_for(category, anomaly_type, image_path)
        _, region_labels = cv2.connectedComponents(mask, 8)
        global_map, tiled_map = read_pair("test", category, anomaly_type, image_path.stem)
        for method, score in method_maps(global_map, tiled_map, calibration).items():
            if score.shape != mask.shape:
                score = cv2.resize(score, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_LINEAR)
            bins = np.minimum((np.maximum(score, 0) * scales[method]).astype(np.int64), BINS - 1)
            item = stats[method]
            item["negative"] += np.bincount(bins[mask == 0], minlength=BINS)
            item["positive"] += np.bincount(bins[mask == 1], minlength=BINS)
            for region_id in range(1, int(region_labels.max()) + 1):
                region = region_labels == region_id
                item["region"] += np.bincount(bins[region], minlength=BINS) / int(region.sum())
                item["regions"] += 1

    rows = []
    for method in METHODS:
        item = stats[method]
        rows.append({
            "category": category,
            "method": method,
            "test_images": len(samples),
            "anomaly_images": int(sum(labels)),
            "regions": item["regions"],
            "aupro_0_05": aupro(item["negative"], item["region"], item["regions"]),
            "pixel_auroc": histogram_auc(item["negative"], item["positive"]),
            "image_auroc": float(roc_auc_score(labels, image_scores[method])),
        })
    return rows


def main():
    if not (A73 / "A73_INFERENCE_COMPLETE.json").exists():
        raise FileNotFoundError(A73 / "A73_INFERENCE_COMPLETE.json")
    OUT.mkdir(parents=True, exist_ok=True)

    calibrations = [calibrate_category(category) for category in CATEGORIES]
    with (OUT / "normal_calibration.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(calibrations[0]))
        writer.writeheader()
        writer.writerows(calibrations)
    calibration_by_category = {row["category"]: row for row in calibrations}

    rows = []
    for index, category in enumerate(CATEGORIES, 1):
        print(f"CATEGORY_EVAL_START {index}/{len(CATEGORIES)} {category}", flush=True)
        rows.extend(evaluate_category(category, calibration_by_category[category]))
        print(f"CATEGORY_EVAL_COMPLETE {index}/{len(CATEGORIES)} {category}", flush=True)

    with (OUT / "category_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    rng = np.random.default_rng(SEED)
    macro = {}
    for method in METHODS:
        method_rows = [row for row in rows if row["method"] == method]
        macro[method] = {}
        for metric in METRICS:
            values = [row[metric] for row in method_rows]
            macro[method][metric] = {
                "mean": float(np.mean(values)),
                "category_bootstrap_95ci": bootstrap_ci(values, rng),
            }

    comparisons = {}
    q99 = {(row["category"], metric): row[metric]
           for row in rows if row["method"] == "q99_mean" for metric in METRICS}
    for baseline in ("global_448", "tiled_672", "raw_mean"):
        baseline_rows = {row["category"]: row for row in rows if row["method"] == baseline}
        comparisons[f"q99_mean_vs_{baseline}"] = {}
        for metric in METRICS:
            differences = {
                category: float(q99[category, metric] - baseline_rows[category][metric])
                for category in CATEGORIES
            }
            values = np.asarray(list(differences.values()), np.float64)
            tolerance = 1e-12
            comparisons[f"q99_mean_vs_{baseline}"][metric] = {
                "mean_delta": float(values.mean()),
                "paired_category_bootstrap_95ci": bootstrap_ci(values, rng),
                "wins": int((values > tolerance).sum()),
                "losses": int((values < -tolerance).sum()),
                "ties": int((np.abs(values) <= tolerance).sum()),
                "by_category": differences,
            }

    payload = {
        "protocol": {
            "dataset": "MVTec AD, 14 categories excluding cable",
            "normal_calibration": "per-image q99 on each category train/good, followed by category median",
            "global_resolution": 448,
            "tiled_resolution": 672,
            "methods": {
                "global_448": "raw global component",
                "tiled_672": "raw tiled component",
                "raw_mean": "0.5 * (G + T), without normal calibration",
                "q99_mean": "0.5 * (G/qG + T/qT)",
            },
            "primary_metric": "category-macro AU-PRO integrated over FPR <= 0.05",
            "bootstrap_unit": "category",
            "bootstrap_replicates": BOOTSTRAPS,
            "calibration_reference_overlap": "train/good is also the source pool for 16-reference coreset selection",
        },
        "macro": macro,
        "comparisons": comparisons,
    }
    (OUT / "summary.json").write_text(json.dumps(payload, indent=2))
    (OUT / "A73_UNIFIED_EVALUATION_COMPLETE.json").write_text(json.dumps({"status": "complete"}, indent=2))
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
