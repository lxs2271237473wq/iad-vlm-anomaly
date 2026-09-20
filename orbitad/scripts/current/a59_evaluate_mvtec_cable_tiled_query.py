"""Frozen A57 transfer evaluation on MVTec AD cable."""
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
MAP_ROOT = ROOT / "ad2_model_zoo/results/a59_superad_reg4_mvtec_cable_tiled_query_v1/cable"
OUT = ROOT / "orbitad/results/a59_superad_reg4_mvtec_cable_tiled_query_eval_v1"
METHODS = ("global_448", "tiled_672", "global_tiled_residual_010")
BANDS = ("all", "tiny_le_0.1pct", "small_0.1_to_1pct", "large_gt_1pct")
MAX_FPR, BINS = .05, 65536


def paths(anomaly_type, stem):
    comp = MAP_ROOT / "component_maps/seed=0/cable" / anomaly_type
    return {
        "global_448": comp / f"{stem}_global.tiff",
        "tiled_672": comp / f"{stem}_tiled.tiff",
        "global_tiled_residual_010": MAP_ROOT / "anomaly_maps/seed=0/cable/test" / anomaly_type / f"{stem}.tiff",
    }


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
            w = (MAX_FPR - fp[i - 1]) / max(fp[i] - fp[i - 1], 1e-12)
            x = np.r_[x, MAX_FPR]
            y = np.r_[y, pro[i - 1] + w * (pro[i] - pro[i - 1])]
    return float(np.trapezoid(y, x) / MAX_FPR)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    samples = []
    for anomaly_dir in sorted((DATA / "test").iterdir()):
        if not anomaly_dir.is_dir():
            continue
        for image_path in sorted(anomaly_dir.glob("*.png")):
            samples.append((anomaly_dir.name, image_path))
    missing = [str(p) for t, image in samples for p in paths(t, image.stem).values() if not p.exists()]
    if missing:
        raise FileNotFoundError(f"missing {len(missing)} maps, first={missing[:3]}")

    maxima = {m: 0. for m in METHODS}
    image_scores = {m: [] for m in METHODS}
    image_labels = []
    for anomaly_type, image_path in samples:
        image_labels.append(int(anomaly_type != "good"))
        for method, map_path in paths(anomaly_type, image_path.stem).items():
            score = np.asarray(tifffile.imread(map_path), np.float32)
            maxima[method] = max(maxima[method], float(score.max()))
            image_scores[method].append(float(np.quantile(score, .999)))
    edges = {m: np.linspace(0, max(v * 1.001, v + 1e-6), BINS + 1) for m, v in maxima.items()}

    def fresh():
        return {"neg": np.zeros(BINS, np.int64), "region": np.zeros(BINS, np.float64), "regions": 0}
    stats = defaultdict(fresh)
    for n, (anomaly_type, image_path) in enumerate(samples, 1):
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if anomaly_type == "good":
            mask = np.zeros_like(image, np.uint8)
        else:
            mask_path = DATA / "ground_truth" / anomaly_type / f"{image_path.stem}_mask.png"
            mask = (cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) > 0).astype(np.uint8)
        region_count, region_labels = cv2.connectedComponents(mask, 8)
        for method, map_path in paths(anomaly_type, image_path.stem).items():
            score = np.asarray(tifffile.imread(map_path), np.float32)
            if score.shape != mask.shape:
                score = cv2.resize(score, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_LINEAR)
            neg_hist = np.histogram(score[mask == 0], bins=edges[method])[0]
            targets = ((method, "__all__"), (method, anomaly_type))
            for target in targets:
                for band in BANDS:
                    stats[target + (band,)]["neg"] += neg_hist
            for rid in range(1, region_count):
                vals = score[region_labels == rid]
                ratio = len(vals) / mask.size
                band = "tiny_le_0.1pct" if ratio <= .001 else ("small_0.1_to_1pct" if ratio <= .01 else "large_gt_1pct")
                contribution = np.histogram(vals, bins=edges[method])[0] / len(vals)
                for target in targets:
                    for selected_band in ("all", band):
                        item = stats[target + (selected_band,)]
                        item["region"] += contribution
                        item["regions"] += 1
        if n % 25 == 0:
            print("evaluated", n, len(samples), flush=True)

    detail = []
    for (method, defect_type, band), stat in sorted(stats.items()):
        if defect_type == "good":
            continue
        detail.append({"method": method, "defect_type": defect_type, "size_band": band,
                       "regions": stat["regions"], "aupro_0_05": aupro(stat)})
    with (OUT / "method_defect_type_size.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(detail[0])); writer.writeheader(); writer.writerows(detail)

    summary = {}
    for method in METHODS:
        summary[method] = {"image_auroc": float(roc_auc_score(image_labels, image_scores[method]))}
        for band in BANDS:
            row = next(x for x in detail if x["method"] == method and x["defect_type"] == "__all__" and x["size_band"] == band)
            summary[method][band] = {"aupro_0_05": row["aupro_0_05"], "regions": row["regions"]}

    by_type = {}
    for defect_type in sorted({x["defect_type"] for x in detail if x["defect_type"] != "__all__"}):
        base = next(x["aupro_0_05"] for x in detail if x["method"] == "global_448" and x["defect_type"] == defect_type and x["size_band"] == "all")
        fused = next(x["aupro_0_05"] for x in detail if x["method"] == "global_tiled_residual_010" and x["defect_type"] == defect_type and x["size_band"] == "all")
        if np.isfinite(base) and np.isfinite(fused):
            by_type[defect_type] = float(fused - base)
    values = np.asarray(list(by_type.values()))
    rng = np.random.default_rng(20260918)
    boot = rng.choice(values, (100000, len(values)), replace=True).mean(1)
    comparison = {"fused_minus_global_by_defect_type": by_type, "mean_delta": float(values.mean()),
                  "wins": int((values > 0).sum()), "types": len(values),
                  "bootstrap_95ci": [float(x) for x in np.quantile(boot, [.025, .975])]}
    payload = {"summary": summary, "comparison": comparison,
               "protocol": "A57 parameters frozen from AD2; MVTec AD cable native AU-PRO@FPR<=0.05"}
    (OUT / "summary.json").write_text(json.dumps(payload, indent=2))
    (OUT / "A59_EVALUATION_COMPLETE.json").write_text(json.dumps({"status": "complete"}, indent=2))
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
