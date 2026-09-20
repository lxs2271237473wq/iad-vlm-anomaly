"""Paired category/condition audit of A64 against the 448 SuperAD-Reg4 baseline."""
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("/root/private_data/iad-vlm-anomaly/orbitad/results")
BASE = ROOT / "a47_superad_reg4_unified_eval_v1/native_category_condition_size.csv"
A64 = ROOT / "a64_superad_reg4_ad2_all_true672_token_preserving_eval_v1/native_category_condition_size.csv"
OUT = ROOT / "a66_a64_paired_audit_v1"
BANDS = ("all", "tiny_le_0.1pct", "small_0.1_to_1pct", "large_gt_1pct")


def read(path):
    rows = list(csv.DictReader(open(path)))
    return {(r["category"], r["condition"], r["size_band"]): float(r["aupro_0_05"])
            for r in rows if r["aupro_0_05"].lower() != "nan"}


def ci(values, seed):
    values = np.asarray(values, np.float64)
    rng = np.random.default_rng(seed)
    boot = rng.choice(values, (100000, len(values)), replace=True).mean(1)
    return [float(x) for x in np.quantile(boot, [0.025, 0.975])]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    base, method = read(BASE), read(A64)
    common = sorted(set(base) & set(method))
    categories = sorted({k[0] for k in common})
    detail = []
    summary = {}
    for band_index, band in enumerate(BANDS):
        by_category = {}
        condition_deltas = []
        condition_wins = 0
        for category in categories:
            keys = [k for k in common if k[0] == category and k[2] == band]
            if not keys:
                continue
            b = float(np.mean([base[k] for k in keys]))
            m = float(np.mean([method[k] for k in keys]))
            deltas = [method[k] - base[k] for k in keys]
            by_category[category] = {
                "global_448": b,
                "a64_query_672": m,
                "delta": m - b,
                "condition_wins": int(sum(d > 0 for d in deltas)),
                "conditions": len(deltas),
            }
            for k, d in zip(keys, deltas):
                detail.append({"category": category, "condition": k[1], "size_band": band,
                               "global_448": base[k], "a64_query_672": method[k], "delta": d})
            condition_deltas.extend(deltas)
            condition_wins += sum(d > 0 for d in deltas)
        category_deltas = [v["delta"] for v in by_category.values()]
        summary[band] = {
            "global_448_category_macro": float(np.mean([v["global_448"] for v in by_category.values()])),
            "a64_query_672_category_macro": float(np.mean([v["a64_query_672"] for v in by_category.values()])),
            "mean_delta": float(np.mean(category_deltas)),
            "category_wins": int(sum(d > 0 for d in category_deltas)),
            "categories": len(category_deltas),
            "category_bootstrap_95ci": ci(category_deltas, 20260919 + band_index),
            "condition_wins": int(condition_wins),
            "category_condition_pairs": len(condition_deltas),
            "by_category": by_category,
        }
    with (OUT / "paired_category_condition_size.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(detail[0]))
        w.writeheader(); w.writerows(detail)
    payload = {
        "summary": summary,
        "decision": "A64 is accepted as a positive overall high-resolution query result only if the all-band category CI excludes zero; category-universal improvement requires 8/8 wins and is assessed separately.",
    }
    (OUT / "summary.json").write_text(json.dumps(payload, indent=2))
    (OUT / "A66_PAIRED_AUDIT_COMPLETE.json").write_text(json.dumps({"status":"complete"}, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
