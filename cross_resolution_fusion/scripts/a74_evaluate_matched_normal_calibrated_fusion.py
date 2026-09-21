#!/usr/bin/env python3
"""Re-evaluate normal-calibrated fusion with matched 448/672 test maps."""

from __future__ import annotations

import csv
import json
import os
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import tifffile

ROOT = Path(os.environ.get("IAD_REPO_ROOT", "/root/private_data/iad-vlm-anomaly"))
META = ROOT / "orbitad/results/a10_multilayer_v1/public_meta.csv"
NORMAL = ROOT / "orbitad/results/a68_normal_only_resolution_router_v1/normal_image_statistics.csv"
A74 = ROOT / "ad2_model_zoo/results/a74_ad2_matched_dual_resolution_v1"
OUT = ROOT / "orbitad/results/a74_matched_normal_calibrated_fusion_v1"
MAX_FPR, BINS = 0.05, 65536
BANDS = ("all", "tiny_le_0.1pct", "small_0.1_to_1pct", "large_gt_1pct")
METHODS = (
    "global", "tiled", "raw_mean", "q99_mean", "q99_geomean", "q99_harmonic",
    "q99_positive_025", "q99_positive_050", "tail_mean", "tail_geomean",
)


def component_paths(row):
    rel = Path(row["image_path"])
    stem = (A74 / row["category"] / "component_maps/seed=0" / row["category"] /
            rel.parent.name / rel.stem)
    return Path(str(stem) + "_global.tiff"), Path(str(stem) + "_tiled.tiff")


def load_calibration():
    rows = list(csv.DictReader(NORMAL.open()))
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["category"]].append(row)
    result = {}
    for category, values in grouped.items():
        result[category] = {
            key: float(np.median([float(v[key]) for v in values]))
            for key in ("global_q95", "global_q99", "tiled_q95", "tiled_q99")
        }
    return result


def candidates(g, t, c):
    eps = 1e-12
    gn = g / max(c["global_q99"], eps)
    tn = t / max(c["tiled_q99"], eps)
    ga = np.maximum((g - c["global_q95"]) / max(c["global_q99"] - c["global_q95"], eps), 0)
    ta = np.maximum((t - c["tiled_q95"]) / max(c["tiled_q99"] - c["tiled_q95"], eps), 0)
    return {
        "global": g,
        "tiled": t,
        "raw_mean": (g + t) * .5,
        "q99_mean": (gn + tn) * .5,
        "q99_geomean": np.sqrt(np.maximum(gn * tn, 0)),
        "q99_harmonic": 2 * gn * tn / np.maximum(gn + tn, eps),
        "q99_positive_025": gn + .25 * np.maximum(tn - gn, 0),
        "q99_positive_050": gn + .50 * np.maximum(tn - gn, 0),
        "tail_mean": (ga + ta) * .5,
        "tail_geomean": np.sqrt(np.maximum(ga * ta, 0)),
    }


class Stats:
    def __init__(self):
        self.neg = np.zeros(BINS, np.int64)
        self.region = np.zeros(BINS, np.float64)
        self.regions = 0


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    calibration = load_calibration()
    rows = list(csv.DictReader(META.open()))
    missing = [str(p) for r in rows for p in component_paths(r) if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} maps; first={missing[:3]}")

    maxima = {method: 0.0 for method in METHODS}
    for n, row in enumerate(rows, 1):
        gp, tp = component_paths(row)
        maps = candidates(np.asarray(tifffile.imread(gp), np.float32),
                          np.asarray(tifffile.imread(tp), np.float32), calibration[row["category"]])
        for method, score in maps.items():
            maxima[method] = max(maxima[method], float(np.nanmax(score)))
        if n % 50 == 0:
            print("max pass", n, len(rows), flush=True)
    scales = {m: BINS / max(v * 1.001, v + 1e-6) for m, v in maxima.items()}

    stats = {}
    def get(method, category, condition, band):
        key = method, category, condition, band
        if key not in stats:
            stats[key] = Stats()
        return stats[key]

    for n, row in enumerate(rows, 1):
        gp, tp = component_paths(row)
        maps = candidates(np.asarray(tifffile.imread(gp), np.float32),
                          np.asarray(tifffile.imread(tp), np.float32), calibration[row["category"]])
        sample = maps["global"]
        if int(row["label"]) == 0:
            mask = np.zeros(sample.shape, np.uint8)
        else:
            mask = (cv2.imread(str(ROOT / "datasets/MVTec_AD_2" / row["mask_path"]), cv2.IMREAD_GRAYSCALE) > 0).astype(np.uint8)
        count, labels = cv2.connectedComponents(mask, 8)
        regions = []
        for rid in range(1, count):
            region = labels == rid
            ratio = int(region.sum()) / mask.size
            band = "tiny_le_0.1pct" if ratio <= .001 else ("small_0.1_to_1pct" if ratio <= .01 else "large_gt_1pct")
            regions.append((region, band))
        for method, score in maps.items():
            if score.shape != mask.shape:
                score = cv2.resize(score, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_LINEAR)
            bins = np.minimum((np.maximum(score, 0) * scales[method]).astype(np.int64), BINS - 1)
            neg_hist = np.bincount(bins[mask == 0], minlength=BINS)
            for band in BANDS:
                get(method, row["category"], row["condition"], band).neg += neg_hist
            for region, band in regions:
                contribution = np.bincount(bins[region], minlength=BINS) / int(region.sum())
                for target in ("all", band):
                    s = get(method, row["category"], row["condition"], target)
                    s.region += contribution
                    s.regions += 1
        if n % 25 == 0:
            print("metric pass", n, len(rows), flush=True)

    def aupro(s):
        if s.neg.sum() == 0 or s.regions == 0:
            return float("nan")
        fp = np.r_[0., np.cumsum(s.neg[::-1], dtype=np.float64)] / s.neg.sum()
        pro = np.r_[0., np.cumsum(s.region[::-1], dtype=np.float64) / s.regions]
        keep = fp <= MAX_FPR
        x, y = fp[keep], pro[keep]
        if x[-1] < MAX_FPR:
            i = np.searchsorted(fp, MAX_FPR, side="right")
            if i < len(fp):
                w = (MAX_FPR - fp[i-1]) / max(fp[i] - fp[i-1], 1e-12)
                x, y = np.r_[x, MAX_FPR], np.r_[y, pro[i-1] + w * (pro[i] - pro[i-1])]
        return float(np.trapezoid(y, x) / MAX_FPR)

    detail = []
    for (method, category, condition, band), s in sorted(stats.items()):
        detail.append({"method": method, "category": category, "condition": condition,
                       "size_band": band, "regions": s.regions, "aupro_0_05": aupro(s)})
    with (OUT / "method_category_condition_size.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(detail[0])); writer.writeheader(); writer.writerows(detail)

    summary = {}
    for method in METHODS:
        summary[method] = {}
        for band in BANDS:
            selected = [r for r in detail if r["method"] == method and r["size_band"] == band and np.isfinite(r["aupro_0_05"])]
            grouped = defaultdict(list)
            for row in selected:
                grouped[row["category"]].append(row["aupro_0_05"])
            summary[method][band] = float(np.mean([np.mean(v) for v in grouped.values()]))
    baseline = summary["global"]["all"]
    ranking = sorted(((summary[m]["all"], m) for m in METHODS), reverse=True)
    category_scores = {}
    for method in METHODS:
        category_scores[method] = {}
        for category in sorted({row["category"] for row in detail}):
            values = [row["aupro_0_05"] for row in detail
                      if row["method"] == method and row["category"] == category
                      and row["size_band"] == "all" and np.isfinite(row["aupro_0_05"])]
            category_scores[method][category] = float(np.mean(values))

    comparisons = {}
    pairs = (
        ("raw_mean", "global"), ("raw_mean", "tiled"),
        ("q99_mean", "global"), ("q99_mean", "tiled"), ("q99_mean", "raw_mean"),
        ("tail_mean", "raw_mean"),
    )
    rng = np.random.default_rng(20260921)
    for method, reference in pairs:
        categories = sorted(category_scores[method])
        differences = np.asarray([
            category_scores[method][category] - category_scores[reference][category]
            for category in categories
        ], np.float64)
        bootstrap = rng.choice(differences, (100000, len(differences)), replace=True).mean(axis=1)
        comparisons[f"{method}_vs_{reference}"] = {
            "mean_delta": float(differences.mean()),
            "paired_category_bootstrap_95ci": [float(x) for x in np.quantile(bootstrap, [.025, .975])],
            "wins": int((differences > 1e-12).sum()),
            "losses": int((differences < -1e-12).sum()),
            "ties": int((np.abs(differences) <= 1e-12).sum()),
            "by_category": dict(zip(categories, map(float, differences))),
        }
    payload = {
        "summary": summary,
        "delta_vs_global": {m: summary[m]["all"] - baseline for m in METHODS},
        "ranking": [{"method": m, "aupro_0_05": v} for v, m in ranking],
        "comparisons": comparisons,
        "selection_rule": "Protocol repair of A69; report all predeclared candidates without overwriting A69",
        "normal_calibration_source": "A67 normal validation maps only",
        "test_component_source": "A74 matched reference/global 448 and tiled 672 component maps",
        "protocol_repair": "Replaces seven A45 global-672 inputs used by A69 with matched global-448 inputs",
    }
    (OUT / "summary.json").write_text(json.dumps(payload, indent=2))
    (OUT / "A74_REEVALUATION_WITH_RAW_MEAN_COMPLETE.json").write_text(
        json.dumps({"status": "complete", "includes_raw_mean": True,
                    "includes_paired_category_bootstrap": True}, indent=2))
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
