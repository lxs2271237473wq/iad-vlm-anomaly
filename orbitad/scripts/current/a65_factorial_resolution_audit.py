"""Factorial audit for windowing and true query-resolution effects on Sheet Metal."""
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path("/root/private_data/iad-vlm-anomaly/orbitad/results")
OUT = ROOT / "a65_factorial_resolution_audit_v1"
FILES = {
    "global_448": ROOT / "a58_superad_reg4_sheetmetal_global_eval_v1/native_category_condition_size.csv",
    "window_448": ROOT / "a58_superad_reg4_sheetmetal_tiled_eval_v1/native_category_condition_size.csv",
    "window_672": ROOT / "a61_superad_reg4_sheetmetal_true672_tiled_eval_v1/native_category_condition_size.csv",
    "fused_672_010": ROOT / "a60_superad_reg4_sheetmetal_true672_tiled_query_eval_v1/native_category_condition_size.csv",
}


def load(path):
    rows = list(csv.DictReader(open(path)))
    return {(r["condition"], r["size_band"]): float(r["aupro_0_05"]) for r in rows}


def bootstrap(values, seed):
    values = np.asarray(values, np.float64)
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, (100000, len(values)), replace=True).mean(axis=1)
    return [float(x) for x in np.quantile(draws, [0.025, 0.975])]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    scores = {name: load(path) for name, path in FILES.items()}
    conditions = sorted({condition for condition, band in scores["global_448"] if band == "all"})
    comparisons = {}
    for target in ("window_448", "window_672", "fused_672_010"):
        by_condition = {c: scores[target][(c, "all")] - scores["global_448"][(c, "all")] for c in conditions}
        values = list(by_condition.values())
        comparisons[f"{target}_minus_global_448"] = {
            "by_condition": by_condition,
            "mean_delta": float(np.mean(values)),
            "wins": int(sum(v > 0 for v in values)),
            "conditions": len(values),
            "bootstrap_95ci": bootstrap(values, 20260919),
        }
    resolution_delta = {
        c: scores["window_672"][(c, "all")] - scores["window_448"][(c, "all")] for c in conditions
    }
    comparisons["window_672_minus_window_448"] = {
        "by_condition": resolution_delta,
        "mean_delta": float(np.mean(list(resolution_delta.values()))),
        "wins": int(sum(v > 0 for v in resolution_delta.values())),
        "conditions": len(resolution_delta),
        "bootstrap_95ci": bootstrap(list(resolution_delta.values()), 20260920),
    }
    macro = {
        method: {band: float(np.mean([value for (condition, b), value in table.items() if b == band]))
                 for band in ("all", "tiny_le_0.1pct", "small_0.1_to_1pct", "large_gt_1pct")}
        for method, table in scores.items()
    }
    payload = {
        "macro": macro,
        "comparisons": comparisons,
        "interpretation": "448 windowing alone loses global context; true 672 windows reverse the loss and outperform global 448. Full-image 672 was infeasible on the 24GB GPU, so square windows provide the memory-bounded route to higher token density.",
    }
    (OUT / "summary.json").write_text(json.dumps(payload, indent=2))
    (OUT / "A65_FACTORIAL_AUDIT_COMPLETE.json").write_text(json.dumps({"status": "complete"}, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
