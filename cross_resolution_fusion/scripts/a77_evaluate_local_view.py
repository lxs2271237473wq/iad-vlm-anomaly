#!/usr/bin/env python3
"""A77: go/no-go evaluation of a true 2D local view against a whole-image view.

Five configurations are formed from four component maps produced by one shared
run per category (identical bank, references, preprocessing and coordinates):

    G448          whole image at 448                     -> _global.tiff
    G672          whole image at 672                     -> _global672.tiff
    legacy_mean   0.5 * (global + legacy tiled)          -> _global.tiff + _tiled.tiff
    T_local       true 2D sliding windows at 672         -> _local672.tiff
    G448+T_local  0.5 * (global + local)                 -> _global.tiff + _local672.tiff

Primary metric: pooled AU-PRO@0.05 (all test images of a category pooled into one
PRO curve, then category macro). This is the protocol the audit asked for and it
is the harder of the two protocols used in the archive.

Decision rule (pre-registered for this screening round):
  Q1  T_local > G672                      is a genuine local view better than
                                          simply raising whole-image resolution?
  Q2  G448+T_local > legacy_mean           does the local view beat the existing
                                          tiled component inside the fusion?
  Q3  G448+T_local > T_local               does global context still help when
                                          the local branch is genuinely local?
If Q1 and Q3 hold and Q2 holds, the multi-view hypothesis survives; if the local
view only beats the weaker baseline, it does not.
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import tifffile

ROOT = Path(os.environ.get("IAD_REPO_ROOT", "/root/private_data/iad-vlm-anomaly"))
MAPS = ROOT / "ad2_model_zoo/results/a77_local_view_dual_resolution_v1/test"
OUT = ROOT / "orbitad/results/a77_local_view_go_no_go_v1"
OUT.mkdir(parents=True, exist_ok=True)

BINS, MAX_FPR = 65536, 0.05
SEED, REPS = 20260921, 100_000
BANDS = ("all", "tiny_le_0.1pct", "small_0.1_to_1pct", "large_gt_1pct")
CONFIGS = ("G448", "G672", "legacy_mean", "T_local", "G448+T_local")

DATASETS = {
    "mvtec": {
        "root": ROOT / "datasets/MVTecAD",
        "test": "{c}/test",
        "mask": "{c}/ground_truth/{t}/{s}_mask.png",
        "categories": ("cable", "grid"),
    },
    "ad2": {
        "root": ROOT / "datasets/MVTec_AD_2",
        "test": "{c}/test_public",
        "mask": "{c}/test_public/ground_truth/{t}/{s}_mask.png",
        "categories": ("can", "sheet_metal"),
    },
}


def samples(ds, category):
    spec = DATASETS[ds]
    base = spec["root"] / spec["test"].format(c=category)
    out = []
    for folder in sorted(p for p in base.iterdir() if p.is_dir() and p.name != "ground_truth"):
        for img in sorted(folder.glob("*.png")):
            out.append((folder.name, img))
    return out


def mask_for(ds, category, anomaly_type, image_path):
    spec = DATASETS[ds]
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(image_path)
    if anomaly_type == "good":
        return np.zeros_like(image, np.uint8)
    path = spec["root"] / spec["mask"].format(c=category, t=anomaly_type, s=image_path.stem)
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(path)
    return (mask > 0).astype(np.uint8)


def component(category, anomaly_type, stem, suffix):
    return (MAPS / category / "component_maps/seed=0" / category / anomaly_type
            / f"{stem}_{suffix}.tiff")


def load_maps(category, anomaly_type, stem):
    g = np.asarray(tifffile.imread(component(category, anomaly_type, stem, "global")), np.float32)
    g672 = np.asarray(tifffile.imread(component(category, anomaly_type, stem, "global672")), np.float32)
    t = np.asarray(tifffile.imread(component(category, anomaly_type, stem, "tiled")), np.float32)
    local = np.asarray(tifffile.imread(component(category, anomaly_type, stem, "local672")), np.float32)
    return {
        "G448": g,
        "G672": g672,
        "legacy_mean": 0.5 * (g + t),
        "T_local": local,
        "G448+T_local": 0.5 * (g + local),
    }


def aupro(negative, region, regions):
    if negative.sum() == 0 or regions == 0:
        return float("nan")
    fp = np.r_[0., np.cumsum(negative[::-1], dtype=np.float64)] / negative.sum()
    pro = np.r_[0., np.cumsum(region[::-1], dtype=np.float64) / regions]
    keep = fp <= MAX_FPR
    x, y = fp[keep], pro[keep]
    if x[-1] < MAX_FPR:
        i = np.searchsorted(fp, MAX_FPR, side="right")
        if i < len(fp):
            w = (MAX_FPR - fp[i - 1]) / max(fp[i] - fp[i - 1], 1e-12)
            x, y = np.r_[x, MAX_FPR], np.r_[y, pro[i - 1] + w * (pro[i] - pro[i - 1])]
    return float(np.trapezoid(y, x) / MAX_FPR)


def evaluate_category(ds, category):
    sample_list = samples(ds, category)
    missing = [str(component(category, t, p.stem, s))
               for t, p in sample_list
               for s in ("global", "global672", "tiled", "local672")
               if not component(category, t, p.stem, s).exists()]
    if missing:
        return None, f"{category}: {len(missing)} component maps missing; first={missing[:1]}"

    maxima = {cfg: 0.0 for cfg in CONFIGS}
    for anomaly_type, image_path in sample_list:
        for cfg, score in load_maps(category, anomaly_type, image_path.stem).items():
            maxima[cfg] = max(maxima[cfg], float(np.nanmax(score)))
    scales = {cfg: BINS / max(v * 1.001, v + 1e-6) for cfg, v in maxima.items()}

    stats = {cfg: {"neg": np.zeros(BINS, np.int64),
                   "region": {b: np.zeros(BINS, np.float64) for b in BANDS},
                   "regions": {b: 0 for b in BANDS}} for cfg in CONFIGS}

    for anomaly_type, image_path in sample_list:
        mask = mask_for(ds, category, anomaly_type, image_path)
        count, labels = cv2.connectedComponents(mask, 8)
        regions = []
        for rid in range(1, count):
            region = labels == rid
            ratio = int(region.sum()) / mask.size
            band = ("tiny_le_0.1pct" if ratio <= 0.001
                    else "small_0.1_to_1pct" if ratio <= 0.01 else "large_gt_1pct")
            regions.append((region, band))
        for cfg, score in load_maps(category, anomaly_type, image_path.stem).items():
            if score.shape != mask.shape:
                score = cv2.resize(score, (mask.shape[1], mask.shape[0]),
                                   interpolation=cv2.INTER_LINEAR)
            bins = np.minimum((np.maximum(score, 0) * scales[cfg]).astype(np.int64), BINS - 1)
            item = stats[cfg]
            item["neg"] += np.bincount(bins[mask == 0], minlength=BINS)
            for region, band in regions:
                contribution = np.bincount(bins[region], minlength=BINS) / int(region.sum())
                for target in ("all", band):
                    item["region"][target] += contribution
                    item["regions"][target] += 1

    rows = []
    for cfg in CONFIGS:
        item = stats[cfg]
        row = {"category": category, "config": cfg, "test_images": len(sample_list)}
        for band in BANDS:
            row[band] = aupro(item["neg"], item["region"][band], item["regions"][band])
        rows.append(row)
    return rows, None


def bootstrap_ci(values, rng):
    v = np.asarray(values, np.float64)
    draws = rng.choice(v, (REPS, len(v)), replace=True).mean(axis=1)
    return [float(x) for x in np.quantile(draws, [.025, .975])]


def main():
    wanted = sys.argv[1:] or [c for spec in DATASETS.values() for c in spec["categories"]]
    ds_of = {c: ds for ds, spec in DATASETS.items() for c in spec["categories"]}

    all_rows, notes = [], []
    for category in wanted:
        rows, err = evaluate_category(ds_of[category], category)
        if err:
            notes.append(err)
            print(f"SKIP {err}", flush=True)
            continue
        all_rows.extend(rows)
        print(f"EVAL_COMPLETE {category}", flush=True)

    if not all_rows:
        print(json.dumps({"status": "incomplete", "notes": notes}, indent=2))
        return

    cats = sorted({r["category"] for r in all_rows})
    macro = {}
    for cfg in CONFIGS:
        macro[cfg] = {}
        for band in BANDS:
            vals = [r[band] for r in all_rows if r["config"] == cfg and np.isfinite(r[band])]
            macro[cfg][band] = {"mean": float(np.mean(vals)), "n": len(vals)}

    rng = np.random.default_rng(SEED)
    indexed = {(r["category"], r["config"]): r for r in all_rows}
    comparisons = {}
    for cand, base in (("T_local", "G672"), ("G448+T_local", "legacy_mean"),
                       ("G448+T_local", "T_local"), ("T_local", "G448"),
                       ("G672", "G448")):
        entry = {}
        for band in BANDS:
            diffs = {c: float(indexed[(c, cand)][band] - indexed[(c, base)][band])
                     for c in cats
                     if np.isfinite(indexed[(c, cand)][band]) and np.isfinite(indexed[(c, base)][band])}
            v = np.asarray(list(diffs.values()), np.float64)
            entry[band] = {"mean_delta": float(v.mean()), "ci": bootstrap_ci(v, rng),
                           "wins": int((v > 1e-12).sum()), "n": len(v), "by_category": diffs}
        comparisons[f"{cand}_minus_{base}"] = entry

    payload = {
        "protocol": "pooled AU-PRO@0.05 per category (all test images merged), then category macro",
        "configurations": list(CONFIGS),
        "categories_evaluated": cats,
        "categories_pending": notes,
        "macro": macro,
        "comparisons": comparisons,
        "decision_rule": {
            "Q1_local_beats_G672": "T_local > G672",
            "Q2_local_beats_legacy_in_fusion": "G448+T_local > legacy_mean",
            "Q3_context_still_helps": "G448+T_local > T_local",
        },
    }
    (OUT / "summary.json").write_text(json.dumps(payload, indent=2, default=float))
    with (OUT / "config_category_metrics.csv").open("w", newline="") as handle:
        import csv as _csv
        w = _csv.DictWriter(handle, fieldnames=["category", "config", "test_images", *BANDS],
                            lineterminator="\n")
        w.writeheader()
        w.writerows(all_rows)

    print("\n=== pooled AU-PRO@0.05, category macro ===")
    print(f"{'config':<14}" + "".join(f"{b[:16]:>18}" for b in BANDS))
    for cfg in CONFIGS:
        print(f"{cfg:<14}" + "".join(
            f"{macro[cfg][b]['mean']:>10.4f}(n={macro[cfg][b]['n']})" for b in BANDS))
    print("\n=== decision comparisons (all band) ===")
    for name, entry in comparisons.items():
        e = entry["all"]
        sig = "SIG" if (e["ci"][0] > 0) == (e["ci"][1] > 0) else "n.s."
        print(f"{name:<30} {e['mean_delta']:+.4f} CI[{e['ci'][0]:+.4f},{e['ci'][1]:+.4f}]"
              f" {e['wins']}/{e['n']} {sig}")
    print(f"\nWROTE {OUT}")


if __name__ == "__main__":
    main()
