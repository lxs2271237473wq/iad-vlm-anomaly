#!/usr/bin/env python3
"""Complete the A69 module ablation with an uncalibrated raw mean."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import tifffile

ROOT = Path("/root/private_data/iad-vlm-anomaly")
META = ROOT / "orbitad/results/a10_multilayer_v1/public_meta.csv"
A69 = ROOT / "orbitad/results/a69_normal_calibrated_multires_fusion_v1"
A64 = ROOT / "ad2_model_zoo/results/a64_superad_reg4_ad2_all_true672_token_preserving_v1"
A45 = ROOT / "ad2_model_zoo/results/a45_superad_reg4_public_v1"
A60 = ROOT / "ad2_model_zoo/results/a60_superad_reg4_sheetmetal_true672_tiled_query_v1/sheet_metal/component_maps/seed=0/sheet_metal"
OUT = ROOT / "orbitad/results/a75_a69_module_ablation_v1"
MAX_FPR, BINS = 0.05, 65536
BANDS = ("all", "tiny_le_0.1pct", "small_0.1_to_1pct", "large_gt_1pct")
METHODS = ("global", "tiled", "raw_mean", "q99_mean")


def component_paths(row):
    rel = Path(row["image_path"])
    if row["category"] == "sheet_metal":
        stem = A60 / rel.parent.name / rel.stem
        return Path(str(stem) + "_global.tiff"), Path(str(stem) + "_tiled.tiff")
    global_path = (A45 / row["category"] / "anomaly_maps/seed=0" / row["category"] /
                   "test" / rel.parent.name / f"{rel.stem}.tiff")
    tiled_stem = (A64 / row["category"] / "component_maps/seed=0" / row["category"] /
                  rel.parent.name / rel.stem)
    return global_path, Path(str(tiled_stem) + "_tiled.tiff")


class Stats:
    def __init__(self):
        self.neg = np.zeros(BINS, np.int64)
        self.region = np.zeros(BINS, np.float64)
        self.regions = 0


def aupro(stats):
    if stats.neg.sum() == 0 or stats.regions == 0:
        return float("nan")
    fp = np.r_[0.0, np.cumsum(stats.neg[::-1], dtype=np.float64)] / stats.neg.sum()
    pro = np.r_[0.0, np.cumsum(stats.region[::-1], dtype=np.float64) / stats.regions]
    keep = fp <= MAX_FPR
    x, y = fp[keep], pro[keep]
    if x[-1] < MAX_FPR:
        i = np.searchsorted(fp, MAX_FPR, side="right")
        if i < len(fp):
            weight = (MAX_FPR - fp[i - 1]) / max(fp[i] - fp[i - 1], 1e-12)
            x = np.r_[x, MAX_FPR]
            y = np.r_[y, pro[i - 1] + weight * (pro[i] - pro[i - 1])]
    return float(np.trapezoid(y, x) / MAX_FPR)


def bootstrap_ci(values, seed):
    values = np.asarray(values, np.float64)
    rng = np.random.default_rng(seed)
    boot = rng.choice(values, (100000, len(values)), replace=True).mean(axis=1)
    return [float(v) for v in np.quantile(boot, [0.025, 0.975])]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(META.open()))
    source_rows = list(csv.DictReader((A69 / "method_category_condition_size.csv").open()))
    source = {
        (r["method"], r["category"], r["condition"], r["size_band"]): r
        for r in source_rows if r["method"] in {"global", "tiled", "q99_mean"}
    }
    missing = [str(path) for row in rows for path in component_paths(row) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} maps; first={missing[:3]}")

    maximum_cache = OUT / "raw_mean_maximum.json"
    if maximum_cache.exists():
        maximum = float(json.loads(maximum_cache.read_text())["maximum"])
        print("max pass cache", maximum, flush=True)
    else:
        maximum = 0.0
        for index, row in enumerate(rows, 1):
            gp, tp = component_paths(row)
            g = np.asarray(tifffile.imread(gp), np.float32)
            t = np.asarray(tifffile.imread(tp), np.float32)
            if g.shape != t.shape:
                raise ValueError(f"Component shape mismatch: {gp} {g.shape} vs {tp} {t.shape}")
            maximum = max(maximum, float(np.nanmax((g + t) * 0.5)))
            if index % 50 == 0:
                print("max pass", index, len(rows), flush=True)
        maximum_cache.write_text(json.dumps({"maximum": maximum}, indent=2))
    scale = BINS / max(maximum * 1.001, maximum + 1e-6)

    stats = {}
    def get(category, condition, band):
        key = category, condition, band
        if key not in stats:
            stats[key] = Stats()
        return stats[key]

    for index, row in enumerate(rows, 1):
        gp, tp = component_paths(row)
        score = (np.asarray(tifffile.imread(gp), np.float32) +
                 np.asarray(tifffile.imread(tp), np.float32)) * 0.5
        if int(row["label"]) == 0:
            mask = np.zeros(score.shape, np.uint8)
        else:
            mask_path = ROOT / "datasets/MVTec_AD_2" / row["mask_path"]
            mask = (cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) > 0).astype(np.uint8)
        if score.shape != mask.shape:
            score = cv2.resize(score, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_LINEAR)
        bins = np.minimum((np.maximum(score, 0) * scale).astype(np.int64), BINS - 1)
        negative_hist = np.bincount(bins[mask == 0], minlength=BINS)
        for band in BANDS:
            get(row["category"], row["condition"], band).neg += negative_hist
        count, labels = cv2.connectedComponents(mask, 8)
        for region_id in range(1, count):
            region = labels == region_id
            ratio = int(region.sum()) / mask.size
            band = "tiny_le_0.1pct" if ratio <= 0.001 else ("small_0.1_to_1pct" if ratio <= 0.01 else "large_gt_1pct")
            contribution = np.bincount(bins[region], minlength=BINS) / int(region.sum())
            for target in ("all", band):
                target_stats = get(row["category"], row["condition"], target)
                target_stats.region += contribution
                target_stats.regions += 1
        if index % 25 == 0:
            print("metric pass", index, len(rows), flush=True)

    raw_rows = []
    for (category, condition, band), value in sorted(stats.items()):
        raw_rows.append({"method": "raw_mean", "category": category, "condition": condition,
                         "size_band": band, "regions": value.regions,
                         "aupro_0_05": aupro(value)})
    all_rows = [r for r in source_rows if r["method"] in {"global", "tiled", "q99_mean"}] + raw_rows
    fields = ["method", "category", "condition", "size_band", "regions", "aupro_0_05"]
    with (OUT / "method_category_condition_size.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(all_rows)

    values = {}
    for row in all_rows:
        value = float(row["aupro_0_05"])
        if np.isfinite(value):
            values[row["method"], row["category"], row["condition"], row["size_band"]] = value
    categories = sorted({row["category"] for row in all_rows})
    summary, comparisons, category_rows = {}, {}, []
    pairs = (("raw_mean", "global"), ("raw_mean", "tiled"), ("q99_mean", "raw_mean"))
    for band_index, band in enumerate(BANDS):
        category_means = {}
        for category in categories:
            conditions = sorted({key[2] for key in values if key[0] == "q99_mean" and key[1] == category and key[3] == band})
            if not conditions:
                continue
            means = {method: float(np.mean([values[method, category, condition, band] for condition in conditions]))
                     for method in METHODS}
            category_means[category] = means
            category_rows.append({"size_band": band, "category": category, **means})
        summary[band] = {method: float(np.mean([v[method] for v in category_means.values()])) for method in METHODS}
        comparisons[band] = {}
        for pair_index, (left, right) in enumerate(pairs):
            deltas = [v[left] - v[right] for v in category_means.values()]
            comparisons[band][f"{left}_minus_{right}"] = {
                "mean_delta": float(np.mean(deltas)),
                "bootstrap_95ci": bootstrap_ci(deltas, 20260921 + band_index * 10 + pair_index),
                "category_wins": int(sum(delta > 0 for delta in deltas)),
                "category_losses": int(sum(delta < 0 for delta in deltas)),
                "categories": len(deltas),
                "by_category": {category: category_means[category][left] - category_means[category][right]
                                for category in category_means},
            }

    with (OUT / "category_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(category_rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(category_rows)
    ablation = [
        {"configuration": "Global only", "global_branch": 1, "tiled_branch": 0, "q99_calibration": 0,
         **{band: summary[band]["global"] for band in BANDS}},
        {"configuration": "Tiled only", "global_branch": 0, "tiled_branch": 1, "q99_calibration": 0,
         **{band: summary[band]["tiled"] for band in BANDS}},
        {"configuration": "Raw mean", "global_branch": 1, "tiled_branch": 1, "q99_calibration": 0,
         **{band: summary[band]["raw_mean"] for band in BANDS}},
        {"configuration": "A69 q99 mean", "global_branch": 1, "tiled_branch": 1, "q99_calibration": 1,
         **{band: summary[band]["q99_mean"] for band in BANDS}},
    ]
    with (OUT / "ablation_table.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(ablation[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(ablation)
    payload = {
        "status": "complete",
        "scope": "A69-identical component maps and evaluator; adds raw_mean only",
        "formula": {"raw_mean": "0.5 * (G + T)", "q99_mean": "0.5 * (G/qG + T/qT)"},
        "images": len(rows),
        "summary": summary,
        "comparisons": comparisons,
    }
    (OUT / "summary.json").write_text(json.dumps(payload, indent=2))
    (OUT / "A75_MODULE_ABLATION_COMPLETE.json").write_text(json.dumps({"status": "complete"}, indent=2))
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
