"""Discover a one-sided local-evidence residual on A57 AD2 component maps.

This is a development-set mechanism search. The selected formula must be frozen
before it is evaluated on the A59 MVTec Cable component maps.
"""
import csv
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import tifffile

ROOT = Path("/root/private_data/iad-vlm-anomaly")
META = ROOT / "orbitad/results/a10_multilayer_v1/public_meta.csv"
COMP = ROOT / "ad2_model_zoo/results/a57_superad_reg4_sheetmetal_tiled_query_v1/sheet_metal/component_maps/seed=0/sheet_metal"
OUT = ROOT / "orbitad/results/a62_one_sided_residual_discovery_v1"
MAX_FPR, BINS = 0.05, 65536
BANDS = ("all", "tiny_le_0.1pct", "small_0.1_to_1pct", "large_gt_1pct")
METHODS = ("global", "linear_010", "positive_010", "positive_025", "positive_050", "positive_100")


def paths(row):
    rel = Path(row["image_path"])
    base = COMP / rel.parent.name / rel.stem
    return Path(str(base) + "_global.tiff"), Path(str(base) + "_tiled.tiff")


def candidates(global_map, tiled_map):
    gq = float(np.quantile(global_map, 0.995))
    tq = float(np.quantile(tiled_map, 0.995))
    local = tiled_map * (gq / max(tq, 1e-12))
    excess = np.maximum(local - global_map, 0.0)
    return {
        "global": global_map,
        "linear_010": global_map + 0.10 * (local - global_map),
        "positive_010": global_map + 0.10 * excess,
        "positive_025": global_map + 0.25 * excess,
        "positive_050": global_map + 0.50 * excess,
        "positive_100": global_map + excess,
    }


class Stats:
    def __init__(self):
        self.neg = np.zeros(BINS, np.int64)
        self.region = np.zeros(BINS, np.float64)
        self.regions = 0


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [r for r in csv.DictReader(open(META)) if r["category"] == "sheet_metal"]
    missing = [str(p) for r in rows for p in paths(r) if not p.exists()]
    if missing:
        raise FileNotFoundError(missing[:3])

    maxima = {m: 0.0 for m in METHODS}
    for n, row in enumerate(rows, 1):
        gp, tp = paths(row)
        maps = candidates(np.asarray(tifffile.imread(gp), np.float32), np.asarray(tifffile.imread(tp), np.float32))
        for method, score in maps.items():
            maxima[method] = max(maxima[method], float(np.nanmax(score)))
        if n % 25 == 0:
            print("max pass", n, len(rows), flush=True)

    scales = {m: BINS / max(v * 1.001, v + 1e-6) for m, v in maxima.items()}
    stats = {}

    def get(method, condition, band):
        key = method, condition, band
        if key not in stats:
            stats[key] = Stats()
        return stats[key]

    for n, row in enumerate(rows, 1):
        gp, tp = paths(row)
        maps = candidates(np.asarray(tifffile.imread(gp), np.float32), np.asarray(tifffile.imread(tp), np.float32))
        sample = maps["global"]
        if int(row["label"]) == 0:
            mask = np.zeros(sample.shape, np.uint8)
        else:
            mask = (cv2.imread(str(ROOT / "datasets/MVTec_AD_2" / row["mask_path"]), cv2.IMREAD_GRAYSCALE) > 0).astype(np.uint8)
        count, labels = cv2.connectedComponents(mask, 8)
        region_info = []
        for rid in range(1, count):
            region = labels == rid
            ratio = int(region.sum()) / mask.size
            band = "tiny_le_0.1pct" if ratio <= 0.001 else ("small_0.1_to_1pct" if ratio <= 0.01 else "large_gt_1pct")
            region_info.append((region, band))

        for method, score in maps.items():
            if score.shape != mask.shape:
                score = cv2.resize(score, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_LINEAR)
            bins = np.minimum((np.maximum(score, 0) * scales[method]).astype(np.int64), BINS - 1)
            neg_hist = np.bincount(bins[mask == 0], minlength=BINS)
            for band in BANDS:
                get(method, row["condition"], band).neg += neg_hist
            for region, band in region_info:
                vals = bins[region]
                contribution = np.bincount(vals, minlength=BINS) / len(vals)
                for target in ("all", band):
                    s = get(method, row["condition"], target)
                    s.region += contribution
                    s.regions += 1
        if n % 10 == 0:
            print("metric pass", n, len(rows), flush=True)

    def aupro(s):
        if s.neg.sum() == 0 or s.regions == 0:
            return float("nan")
        fp = np.r_[0.0, np.cumsum(s.neg[::-1], dtype=np.float64)] / s.neg.sum()
        pro = np.r_[0.0, np.cumsum(s.region[::-1], dtype=np.float64) / s.regions]
        keep = fp <= MAX_FPR
        x, y = fp[keep], pro[keep]
        if x[-1] < MAX_FPR:
            i = np.searchsorted(fp, MAX_FPR, side="right")
            if i < len(fp):
                w = (MAX_FPR - fp[i - 1]) / max(fp[i] - fp[i - 1], 1e-12)
                x = np.r_[x, MAX_FPR]
                y = np.r_[y, pro[i - 1] + w * (pro[i] - pro[i - 1])]
        return float(np.trapezoid(y, x) / MAX_FPR)

    detail = []
    for (method, condition, band), s in sorted(stats.items()):
        detail.append({"method": method, "condition": condition, "size_band": band,
                       "regions": s.regions, "aupro_0_05": aupro(s)})
    with (OUT / "method_condition_size.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(detail[0]))
        writer.writeheader()
        writer.writerows(detail)

    summary = {}
    for method in METHODS:
        summary[method] = {}
        for band in BANDS:
            vals = [r["aupro_0_05"] for r in detail if r["method"] == method and r["size_band"] == band and np.isfinite(r["aupro_0_05"])]
            summary[method][band] = float(np.mean(vals))
    baseline = summary["global"]["all"]
    ranking = sorted(((summary[m]["all"], m) for m in METHODS), reverse=True)
    payload = {
        "summary": summary,
        "delta_vs_global": {m: summary[m]["all"] - baseline for m in METHODS},
        "ranking": [{"method": m, "aupro_0_05": v} for v, m in ranking],
        "selection_rule": "highest six-condition macro AU-PRO on AD2 Sheet Metal development data; freeze before MVTec Cable evaluation",
    }
    (OUT / "summary.json").write_text(json.dumps(payload, indent=2))
    (OUT / "A62_DISCOVERY_COMPLETE.json").write_text(json.dumps({"status": "complete"}, indent=2))
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
