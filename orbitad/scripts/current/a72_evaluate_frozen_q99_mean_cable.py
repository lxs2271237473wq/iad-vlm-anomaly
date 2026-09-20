#!/usr/bin/env python3
"""External validation of A69's frozen q99-mean fusion on MVTec Cable."""

import csv
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import tifffile
from sklearn.metrics import roc_auc_score

ROOT = Path("/root/private_data/iad-vlm-anomaly")
DATA = ROOT / "datasets/MVTecAD/cable"
NORMAL = ROOT / "ad2_model_zoo/results/a71_mvtec_cable_train_normal_dual_resolution_v1"
TEST = ROOT / "ad2_model_zoo/results/a59_superad_reg4_mvtec_cable_tiled_query_v1/cable"
OUT = ROOT / "orbitad/results/a72_frozen_q99_mean_cable_v1"
METHODS = ("global_448", "tiled_672", "q99_mean_frozen")
BANDS = ("all", "tiny_le_0.1pct", "small_0.1_to_1pct", "large_gt_1pct")
MAX_FPR, BINS = .05, 65536


def component_paths(base, anomaly_type, stem):
    folder = base / "component_maps/seed=0/cable" / anomaly_type
    return folder / f"{stem}_global.tiff", folder / f"{stem}_tiled.tiff"


def normal_calibration():
    marker = NORMAL / "A71_INFERENCE_COMPLETE.json"
    if not marker.exists():
        raise FileNotFoundError(marker)
    folder = NORMAL / "cable/component_maps/seed=0/cable/good"
    values = {"global": [], "tiled": []}
    for global_path in sorted(folder.glob("*_global.tiff")):
        tiled_path = global_path.with_name(global_path.name.replace("_global.tiff", "_tiled.tiff"))
        values["global"].append(float(np.quantile(tifffile.imread(global_path), .99)))
        values["tiled"].append(float(np.quantile(tifffile.imread(tiled_path), .99)))
    if not values["global"] or len(values["global"]) != len(values["tiled"]):
        raise RuntimeError("Incomplete A71 normal component maps")
    return {"global_q99": float(np.median(values["global"])),
            "tiled_q99": float(np.median(values["tiled"])),
            "normal_images": len(values["global"])}


def maps_for(anomaly_type, stem, calibration):
    gp, tp = component_paths(TEST, anomaly_type, stem)
    g = np.asarray(tifffile.imread(gp), np.float32)
    t = np.asarray(tifffile.imread(tp), np.float32)
    return {"global_448": g, "tiled_672": t,
            "q99_mean_frozen": .5 * (g / calibration["global_q99"] + t / calibration["tiled_q99"])}


def aupro(stat):
    if stat["neg"].sum() == 0 or stat["regions"] == 0:
        return float("nan")
    fp = np.r_[0., np.cumsum(stat["neg"][::-1], dtype=np.float64)] / stat["neg"].sum()
    pro = np.r_[0., np.cumsum(stat["region"][::-1], dtype=np.float64) / stat["regions"]]
    keep = fp <= MAX_FPR
    x, y = fp[keep], pro[keep]
    if x[-1] < MAX_FPR:
        i = np.searchsorted(fp, MAX_FPR, side="right")
        if i < len(fp):
            w = (MAX_FPR - fp[i-1]) / max(fp[i] - fp[i-1], 1e-12)
            x, y = np.r_[x, MAX_FPR], np.r_[y, pro[i-1] + w * (pro[i] - pro[i-1])]
    return float(np.trapezoid(y, x) / MAX_FPR)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    calibration = normal_calibration()
    samples = [(d.name, p) for d in sorted((DATA / "test").iterdir()) if d.is_dir()
               for p in sorted(d.glob("*.png"))]
    missing = [str(p) for t, image in samples for p in component_paths(TEST, t, image.stem) if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} maps; first={missing[:3]}")

    maxima = {m: 0. for m in METHODS}
    image_scores = {m: [] for m in METHODS}
    labels = []
    for n, (anomaly_type, image_path) in enumerate(samples, 1):
        labels.append(int(anomaly_type != "good"))
        for method, score in maps_for(anomaly_type, image_path.stem, calibration).items():
            maxima[method] = max(maxima[method], float(score.max()))
            image_scores[method].append(float(np.quantile(score, .999)))
        if n % 25 == 0:
            print("max pass", n, len(samples), flush=True)
    scales = {m: BINS / max(v * 1.001, v + 1e-6) for m, v in maxima.items()}

    def fresh():
        return {"neg": np.zeros(BINS, np.int64), "region": np.zeros(BINS, np.float64), "regions": 0}
    stats = defaultdict(fresh)
    for n, (anomaly_type, image_path) in enumerate(samples, 1):
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if anomaly_type == "good":
            mask = np.zeros_like(image, np.uint8)
        else:
            mask = (cv2.imread(str(DATA / "ground_truth" / anomaly_type / f"{image_path.stem}_mask.png"),
                               cv2.IMREAD_GRAYSCALE) > 0).astype(np.uint8)
        count, region_labels = cv2.connectedComponents(mask, 8)
        for method, score in maps_for(anomaly_type, image_path.stem, calibration).items():
            if score.shape != mask.shape:
                score = cv2.resize(score, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_LINEAR)
            bins = np.minimum((np.maximum(score, 0) * scales[method]).astype(np.int64), BINS - 1)
            neg_hist = np.bincount(bins[mask == 0], minlength=BINS)
            for defect_type in ("__all__", anomaly_type):
                for band in BANDS:
                    stats[method, defect_type, band]["neg"] += neg_hist
            for rid in range(1, count):
                region = region_labels == rid
                ratio = int(region.sum()) / mask.size
                band = "tiny_le_0.1pct" if ratio <= .001 else ("small_0.1_to_1pct" if ratio <= .01 else "large_gt_1pct")
                contribution = np.bincount(bins[region], minlength=BINS) / int(region.sum())
                for defect_type in ("__all__", anomaly_type):
                    for selected in ("all", band):
                        item = stats[method, defect_type, selected]
                        item["region"] += contribution; item["regions"] += 1
        if n % 25 == 0:
            print("metric pass", n, len(samples), flush=True)

    detail = []
    for (method, defect_type, band), stat in sorted(stats.items()):
        if defect_type != "good":
            detail.append({"method": method, "defect_type": defect_type, "size_band": band,
                           "regions": stat["regions"], "aupro_0_05": aupro(stat)})
    with (OUT / "method_defect_type_size.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(detail[0])); writer.writeheader(); writer.writerows(detail)

    summary = {}
    for method in METHODS:
        summary[method] = {"image_auroc": float(roc_auc_score(labels, image_scores[method]))}
        for band in BANDS:
            row = next(r for r in detail if r["method"] == method and r["defect_type"] == "__all__" and r["size_band"] == band)
            summary[method][band] = {"aupro_0_05": row["aupro_0_05"], "regions": row["regions"]}

    comparisons = {}
    for baseline in ("global_448", "tiled_672"):
        by_type = {}
        defect_types = sorted({r["defect_type"] for r in detail if r["defect_type"] != "__all__"})
        for defect_type in defect_types:
            b = next(r["aupro_0_05"] for r in detail if r["method"] == baseline and r["defect_type"] == defect_type and r["size_band"] == "all")
            f = next(r["aupro_0_05"] for r in detail if r["method"] == "q99_mean_frozen" and r["defect_type"] == defect_type and r["size_band"] == "all")
            if np.isfinite(b) and np.isfinite(f):
                by_type[defect_type] = float(f - b)
        vals = np.asarray(list(by_type.values()))
        rng = np.random.default_rng(20260920 + len(comparisons))
        boot = rng.choice(vals, (100000, len(vals)), replace=True).mean(1)
        comparisons[f"vs_{baseline}"] = {"by_defect_type": by_type, "mean_delta": float(vals.mean()),
                                           "wins": int((vals > 0).sum()), "types": len(vals),
                                           "bootstrap_95ci": [float(x) for x in np.quantile(boot, [.025, .975])]}
    payload = {
        "summary": summary, "comparison": comparisons, "normal_calibration": calibration,
        "protocol": "q99-mean formula selected on AD2 A69 and frozen before MVTec Cable; q99 values estimated from MVTec train/good only",
    }
    (OUT / "summary.json").write_text(json.dumps(payload, indent=2))
    (OUT / "A72_FROZEN_EXTERNAL_VALIDATION_COMPLETE.json").write_text(json.dumps({"status": "complete"}, indent=2))
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
