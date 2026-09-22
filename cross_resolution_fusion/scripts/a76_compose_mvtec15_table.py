#!/usr/bin/env python3
"""Compose the 15-class MVTec AD table (A76) from existing per-category metrics.

The 14 non-cable categories were already evaluated by the A73 evaluator, and the
A76 evaluator is that same code with cable added. Re-reading every train/test map
for all 15 classes would repeat tens of GB of I/O to recompute deterministic
numbers that are already on disk. This script therefore:

  1. reuses A73's per-(category, method) rows for the 14 non-cable categories;
  2. evaluates cable with the A76 module's own functions (identical code path,
     reading cable's maps in place via LEGACY_MAP_ROOTS);
  3. recomputes the category macro and the paired category bootstrap over all
     15 classes with the same seed and iteration order as the evaluator.

Output goes to orbitad/results/a76_mvtec15_unified_eval_v1/.
"""

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path("/root/private_data/iad-vlm-anomaly")
A73_OUT = ROOT / "orbitad/results/a73_mvtec14_unified_eval_v1"
A76_OUT = ROOT / "orbitad/results/a76_mvtec15_unified_eval_v1"
EVALUATOR = ROOT / "cross_resolution_fusion/scripts/a76_evaluate_mvtec15_unified.py"

INT_FIELDS = ("test_images", "anomaly_images", "regions")
FLOAT_FIELDS = ("aupro_0_05", "pixel_auroc", "image_auroc")


def load_evaluator():
    spec = importlib.util.spec_from_file_location("a76_evaluator", str(EVALUATOR))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def typed(row):
    out = dict(row)
    for key in INT_FIELDS:
        out[key] = int(out[key])
    for key in FLOAT_FIELDS:
        out[key] = float(out[key])
    return out


def main():
    mod = load_evaluator()
    A76_OUT.mkdir(parents=True, exist_ok=True)

    rows14 = [typed(r) for r in csv.DictReader((A73_OUT / "category_metrics.csv").open())]
    cables = [r["category"] for r in rows14]
    assert "cable" not in cables, "A73 metrics must not already contain cable"

    print("CALIBRATING cable ...", flush=True)
    cable_calibration = mod.calibrate_category("cable")
    print("EVALUATING cable ...", flush=True)
    cable_rows = mod.evaluate_category("cable", cable_calibration)

    key_sets = {tuple(sorted(r)) for r in rows14 + cable_rows}
    assert len(key_sets) == 1, "field mismatch between A73 rows and cable rows"

    order = {category: index for index, category in enumerate(mod.CATEGORIES)}
    rows = sorted(rows14 + cable_rows, key=lambda r: (order[r["category"]], mod.METHODS.index(r["method"])))

    with (A76_OUT / "category_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    calibration_rows = [
        typed_cal(r) for r in csv.DictReader((A73_OUT / "normal_calibration.csv").open())
    ] + [cable_calibration]
    calibration_rows.sort(key=lambda r: order[r["category"]])
    with (A76_OUT / "normal_calibration.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(calibration_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(calibration_rows)

    rng = np.random.default_rng(mod.SEED)
    macro = {}
    for method in mod.METHODS:
        method_rows = [r for r in rows if r["method"] == method]
        macro[method] = {}
        for metric in mod.METRICS:
            values = [r[metric] for r in method_rows]
            macro[method][metric] = {
                "mean": float(np.mean(values)),
                "category_bootstrap_95ci": mod.bootstrap_ci(values, rng),
            }

    indexed = {(r["category"], r["method"]): r for r in rows}
    comparisons = {}
    pairs = (
        ("q99_mean", "global_448"), ("q99_mean", "tiled_672"), ("q99_mean", "raw_mean"),
        ("raw_mean", "global_448"), ("raw_mean", "tiled_672"),
    )
    for candidate, baseline in pairs:
        entry = {}
        for metric in mod.METRICS:
            differences = {
                category: float(indexed[(category, candidate)][metric] - indexed[(category, baseline)][metric])
                for category in mod.CATEGORIES
            }
            values = np.asarray(list(differences.values()), np.float64)
            tolerance = 1e-12
            entry[metric] = {
                "mean_delta": float(values.mean()),
                "paired_category_bootstrap_95ci": mod.bootstrap_ci(values, rng),
                "wins": int((values > tolerance).sum()),
                "losses": int((values < -tolerance).sum()),
                "ties": int((np.abs(values) <= tolerance).sum()),
                "by_category": differences,
            }
        comparisons[f"{candidate}_vs_{baseline}"] = entry

    payload = {
        "protocol": {
            "dataset": "MVTec AD, all 15 categories (cable included)",
            "normal_calibration": "per-image q99 on each category train/good, followed by category median",
            "global_resolution": 448,
            "tiled_resolution": 672,
            "methods": {
                "global_448": "raw global component",
                "tiled_672": "raw 672 component; labelled Res672 in paper tables because every "
                              "MVTec AD image is square, so the tiler (tile_size = min(h, w)) emits a "
                              "single tile equal to the whole image",
                "raw_mean": "0.5 * (G + T), without normal calibration",
                "q99_mean": "0.5 * (G/qG + T/qT)",
            },
            "primary_metric": "category-macro AU-PRO integrated over FPR <= 0.05",
            "bootstrap_unit": "category",
            "bootstrap_replicates": mod.BOOTSTRAPS,
            "calibration_reference_overlap": "train/good is also the source pool for 16-reference coreset selection",
            "eval_scope": "A73 keyed the 672 method as tiled_672 for historical continuity; the paper label is Res672",
        },
        "provenance": {
            "category_metrics_source": "14 non-cable categories reused from A73 per-category metrics "
                                       "(identical evaluator code); cable computed here with the A76 "
                                       "evaluator functions",
            "cable_maps": {
                "test": "ad2_model_zoo/results/a59_superad_reg4_mvtec_cable_tiled_query_v1",
                "train": "ad2_model_zoo/results/a71_mvtec_cable_train_normal_dual_resolution_v1",
                "verification": "grid-pitch check: cable tiled 21.17 px and cable global 31.75 px, "
                                "identical to the certified A73 reference (672 and 448 at 1024 px)",
            },
            "why_a76": "MVTec AD has 15 categories and cable is one of them; cable is therefore "
                       "evaluated inside the same protocol instead of being reported as a separate "
                       "single-category transfer experiment",
        },
        "macro": macro,
        "comparisons": comparisons,
    }
    (A76_OUT / "summary.json").write_text(json.dumps(payload, indent=2))
    (A76_OUT / "A76_UNIFIED_EVALUATION_COMPLETE.json").write_text(
        json.dumps({"status": "complete", "categories": len(mod.CATEGORIES), "methods": len(mod.METHODS)}, indent=2)
    )

    print("\n=== MVTec AD 15-class (A76) ===")
    print(f"{'method':<12}{'AU-PRO@0.05':>22}{'Pixel AUROC':>22}{'Image AUROC':>22}")
    for method in mod.METHODS:
        cells = []
        for metric in mod.METRICS:
            m = macro[method][metric]
            cells.append(f"{m['mean']:.6f} [{m['category_bootstrap_95ci'][0]:.4f},{m['category_bootstrap_95ci'][1]:.4f}]")
        print(f"{method:<12}{cells[0]:>22}{cells[1]:>22}{cells[2]:>22}")
    print("\n=== AU-PRO@0.05 comparisons ===")
    for name, entry in comparisons.items():
        e = entry["aupro_0_05"]
        print(f"{name:<26} delta={e['mean_delta']:+.6f} CI=[{e['paired_category_bootstrap_95ci'][0]:+.6f},"
              f"{e['paired_category_bootstrap_95ci'][1]:+.6f}] wins={e['wins']}/{e['wins']+e['losses']+e['ties']}")
    print("\nWROTE", A76_OUT)


def typed_cal(row):
    out = dict(row)
    out["normal_images"] = int(out["normal_images"])
    for key in ("global_q99", "tiled_q99", "global_q99_min", "global_q99_max",
                "tiled_q99_min", "tiled_q99_max"):
        out[key] = float(out[key])
    return out


if __name__ == "__main__":
    main()
