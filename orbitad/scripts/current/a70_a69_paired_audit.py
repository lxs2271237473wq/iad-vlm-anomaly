"""Paired robustness audit for the A69 frozen q99-mean candidate."""
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path("/root/private_data/iad-vlm-anomaly/orbitad/results")
SOURCE = ROOT / "a69_normal_calibrated_multires_fusion_v1/method_category_condition_size.csv"
OUT = ROOT / "a70_a69_paired_audit_v1"
BANDS = ("all", "tiny_le_0.1pct", "small_0.1_to_1pct", "large_gt_1pct")


def bootstrap_ci(values, seed):
    values = np.asarray(values, np.float64)
    rng = np.random.default_rng(seed)
    boot = rng.choice(values, (100000, len(values)), replace=True).mean(1)
    return [float(x) for x in np.quantile(boot, [.025, .975])]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(SOURCE.open()))
    values = {(r["method"], r["category"], r["condition"], r["size_band"]): float(r["aupro_0_05"])
              for r in rows if r["aupro_0_05"].lower() != "nan"}
    categories = sorted({r["category"] for r in rows})
    payload = {}
    detail = []
    for band_i, band in enumerate(BANDS):
        by_category = {}
        for category in categories:
            conditions = sorted({k[2] for k in values if k[1] == category and k[3] == band and k[0] == "q99_mean"})
            if not conditions:
                continue
            means = {m: float(np.mean([values[m, category, c, band] for c in conditions]))
                     for m in ("global", "tiled", "q99_mean")}
            dg, dt = means["q99_mean"] - means["global"], means["q99_mean"] - means["tiled"]
            wins_g = sum(values["q99_mean", category, c, band] > values["global", category, c, band] for c in conditions)
            wins_t = sum(values["q99_mean", category, c, band] > values["tiled", category, c, band] for c in conditions)
            by_category[category] = {**means, "delta_vs_global": dg, "delta_vs_tiled": dt,
                                     "condition_wins_vs_global": wins_g,
                                     "condition_wins_vs_tiled": wins_t, "conditions": len(conditions)}
            for c in conditions:
                detail.append({"category": category, "condition": c, "size_band": band,
                               "global": values["global", category, c, band],
                               "tiled": values["tiled", category, c, band],
                               "q99_mean": values["q99_mean", category, c, band],
                               "delta_vs_global": values["q99_mean", category, c, band] - values["global", category, c, band],
                               "delta_vs_tiled": values["q99_mean", category, c, band] - values["tiled", category, c, band]})
        dg = [v["delta_vs_global"] for v in by_category.values()]
        dt = [v["delta_vs_tiled"] for v in by_category.values()]
        payload[band] = {
            "category_macro": {m: float(np.mean([v[m] for v in by_category.values()])) for m in ("global", "tiled", "q99_mean")},
            "mean_delta_vs_global": float(np.mean(dg)),
            "mean_delta_vs_tiled": float(np.mean(dt)),
            "category_wins_vs_global": int(sum(x > 0 for x in dg)),
            "category_wins_vs_tiled": int(sum(x > 0 for x in dt)),
            "categories": len(by_category),
            "bootstrap_95ci_vs_global": bootstrap_ci(dg, 20260920 + 2 * band_i),
            "bootstrap_95ci_vs_tiled": bootstrap_ci(dt, 20260921 + 2 * band_i),
            "condition_wins_vs_global": int(sum(v["condition_wins_vs_global"] for v in by_category.values())),
            "condition_wins_vs_tiled": int(sum(v["condition_wins_vs_tiled"] for v in by_category.values())),
            "category_condition_pairs": int(sum(v["conditions"] for v in by_category.values())),
            "by_category": by_category,
        }
    with (OUT / "paired_detail.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(detail[0])); writer.writeheader(); writer.writerows(detail)
    result = {"candidate": "q99_mean", "normal_calibration": "A67 normal validation only", "summary": payload}
    (OUT / "summary.json").write_text(json.dumps(result, indent=2))
    (OUT / "A70_PAIRED_AUDIT_COMPLETE.json").write_text(json.dumps({"status": "complete"}, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
