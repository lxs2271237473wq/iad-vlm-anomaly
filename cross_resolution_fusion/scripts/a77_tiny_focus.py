#!/usr/bin/env python3
"""A77 tiny-defect focus: centre assignment vs overlap averaging, plus negative
controls borrowed from the current SOTA pipeline.

Primary metric: pooled AU-PRO@0.05 restricted to TINY defect regions
(area <= 0.1% of the image), which is the objective this study targets.
Overall / small / large bands are reported for completeness.

Variants
    G448            whole image at 448                       (weak baseline)
    G672            whole image at 672
    legacy_mean     0.5 * (global + legacy tiled)
    T_mean          local windows, overlap AVERAGED          (current)
    T_center        local windows, CENTRE assignment         (SuperADD style)
    G+T_mean        0.5 * (global + T_mean)
    G+T_center      0.5 * (global + T_center)

Negative controls applied to the best fused map:
    closing r        grayscale morphological closing, radius r at full resolution
    downscale f      map reduced by f and evaluated back at full resolution
Both are used by the current SOTA for its own objective; the question here is
whether they help or hurt TINY defects.
"""
from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import tifffile

ROOT = Path(os.environ.get("IAD_REPO_ROOT", "/root/private_data/iad-vlm-anomaly"))
MAPS = ROOT / "ad2_model_zoo/results/a77_local_view_dual_resolution_v1/test"
OUT = ROOT / "orbitad/results/a77_tiny_focus_v1"
OUT.mkdir(parents=True, exist_ok=True)

BINS, MAX_FPR = 65536, 0.05
SEED, REPS = 20260921, 100_000
BANDS = ("all", "tiny_le_0.1pct", "small_0.1_to_1pct", "large_gt_1pct")
BASE = ("G448", "G672", "legacy_mean", "T_mean", "T_center", "G+T_mean", "G+T_center")
CLOSING_RADII = (5, 15, 30, 60)
DOWNSCALES = (2, 4)

DATASETS = {
    "mvtec": {"root": ROOT / "datasets/MVTecAD", "test": "{c}/test",
              "mask": "{c}/ground_truth/{t}/{s}_mask.png", "cats": ("cable", "grid")},
    "ad2": {"root": ROOT / "datasets/MVTec_AD_2", "test": "{c}/test_public",
            "mask": "{c}/test_public/ground_truth/{t}/{s}_mask.png",
            "cats": ("can", "sheet_metal")},
}
DS_OF = {c: ds for ds, spec in DATASETS.items() for c in spec["cats"]}


def samples(ds, category):
    base = DATASETS[ds]["root"] / DATASETS[ds]["test"].format(c=category)
    return [(f.name, p) for f in sorted(x for x in base.iterdir()
                                        if x.is_dir() and x.name != "ground_truth")
            for p in sorted(f.glob("*.png"))]


def mask_for(ds, category, anomaly_type, image_path):
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(image_path)
    if anomaly_type == "good":
        return np.zeros_like(image, np.uint8)
    path = DATASETS[ds]["root"] / DATASETS[ds]["mask"].format(
        c=category, t=anomaly_type, s=image_path.stem)
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(path)
    return (mask > 0).astype(np.uint8)


def _p(category, anomaly_type, stem, suffix):
    return (MAPS / category / "component_maps/seed=0" / category / anomaly_type
            / f"{stem}_{suffix}.tiff")


def load(category, anomaly_type, stem, suffix):
    path = _p(category, anomaly_type, stem, suffix)
    if not path.exists():
        return None
    return np.asarray(tifffile.imread(path), np.float32)


def variants(category, anomaly_type, stem):
    g = load(category, anomaly_type, stem, "global")
    g672 = load(category, anomaly_type, stem, "global672")
    tiled = load(category, anomaly_type, stem, "tiled")
    tmean = load(category, anomaly_type, stem, "local672")
    tcenter = load(category, anomaly_type, stem, "localcenter672")
    out = {"G448": g, "G672": g672, "legacy_mean": 0.5 * (g + tiled),
           "T_mean": tmean, "G+T_mean": 0.5 * (g + tmean)}
    if tcenter is not None:
        out["T_center"] = tcenter
        out["G+T_center"] = 0.5 * (g + tcenter)
    return out


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


def evaluate_category(ds, category, controls):
    sl = samples(ds, category)
    probe = variants(category, sl[0][0], sl[0][1].stem)
    names = [n for n in BASE if n in probe]
    names = list(names)
    if controls:
        ref = "G+T_center" if "G+T_center" in names else "G+T_mean"
        names += [f"{ref}|close{r}" for r in CLOSING_RADII]
        names += [f"{ref}|down{f}" for f in DOWNSCALED] if False else []
        names += [f"{ref}|down{f}" for f in DOWNSCALES]
    focus = ("G+T_center" if "G+T_center" in names else "G+T_mean")
    out = {n: {"neg": np.zeros(BINS, np.int64),
               "region": {b: np.zeros(BINS, np.float64) for b in BANDS},
               "regions": {b: 0 for b in BANDS}} for n in names}
    maxima = {n: 0.0 for n in names}

    per_image = []
    for at, p in sl:
        v = variants(category, at, p.stem)
        if focus not in v:
            continue
        item = {n: v[n] for n in names if n in v}
        if controls:
            base_map = v[focus]
            for r in CLOSING_RADII:
                k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
                item[f"{focus}|close{r}"] = cv2.morphologyEx(base_map, cv2.MORPH_CLOSE, k)
            for f in DOWNSCALES:
                small = cv2.resize(base_map, (max(1, base_map.shape[1] // f),
                                              max(1, base_map.shape[0] // f)),
                                   interpolation=cv2.INTER_AREA)
                item[f"{focus}|down{f}"] = small
        per_image.append((at, p, item))

    for n in names:
        for at, p, item in per_image:
            maxima[n] = max(maxima[n], float(np.nanmax(item[n])))
    scales = {n: BINS / max(v * 1.001, v + 1e-6) for n, v in maxima.items()}

    for at, p, item in per_image:
        mask = mask_for(ds, category, at, p)
        count, labels = cv2.connectedComponents(mask, 8)
        regions = []
        for rid in range(1, count):
            rr = labels == rid
            ratio = int(rr.sum()) / mask.size
            regions.append((rr, "tiny_le_0.1pct" if ratio <= 0.001
                            else "small_0.1_to_1pct" if ratio <= 0.01 else "large_gt_1pct"))
        for n in names:
            s = item[n]
            if s.shape != mask.shape:
                s = cv2.resize(s, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_LINEAR)
            bins = np.minimum((np.maximum(s, 0) * scales[n]).astype(np.int64), BINS - 1)
            st = out[n]
            st["neg"] += np.bincount(bins[mask == 0], minlength=BINS)
            for rr, band in regions:
                c = np.bincount(bins[rr], minlength=BINS) / int(rr.sum())
                for target in ("all", band):
                    st["region"][target] += c
                    st["regions"][target] += 1

    return [{"category": category, "variant": n, "test_images": len(per_image),
             **{b: aupro(out[n]["neg"], out[n]["region"][b], out[n]["regions"][b])
                for b in BANDS}} for n in names]


def boot(v, rng):
    draws = rng.choice(v, (REPS, len(v)), replace=True).mean(axis=1)
    return [float(x) for x in np.quantile(draws, [.025, .975])]


def main():
    wanted = [a for a in sys.argv[1:] if not a.startswith("-")]
    controls = "--controls" in sys.argv
    wanted = wanted or [c for spec in DATASETS.values() for c in spec["cats"]]
    rows = []
    for c in wanted:
        try:
            rows.extend(evaluate_category(DS_OF[c], c, controls))
            print(f"EVAL_COMPLETE {c}", flush=True)
        except FileNotFoundError as exc:
            print(f"SKIP {c}: {exc}", flush=True)
    if not rows:
        print("nothing evaluated"); return
    cats = sorted({r["category"] for r in rows})
    names = [n for n in dict.fromkeys(r["variant"] for r in rows)]
    macro = {n: {b: float(np.mean([r[b] for r in rows
                                   if r["variant"] == n and np.isfinite(r[b])]))
                 for b in BANDS} for n in names}
    rng = np.random.default_rng(SEED)
    idx = {(r["category"], r["variant"]): r for r in rows}

    def cmp(cand, base, band):
        d = {c: float(idx[(c, cand)][band] - idx[(c, base)][band]) for c in cats
             if (c, cand) in idx and (c, base) in idx
             and np.isfinite(idx[(c, cand)][band]) and np.isfinite(idx[(c, base)][band])}
        v = np.asarray(list(d.values()), np.float64)
        return {"mean_delta": float(v.mean()), "ci": boot(v, rng),
                "wins": int((v > 1e-12).sum()), "n": len(v), "by_category": d}

    key = ("T_center", "T_mean") if ("T_center", "T_mean") else None
    comparisons = {}
    if "T_center" in names:
        for base in ("T_mean", "G672", "G448"):
            comparisons[f"T_center_minus_{base}"] = {b: cmp("T_center", base, b) for b in BANDS}
        for base in ("G+T_mean", "legacy_mean"):
            comparisons[f"G+T_center_minus_{base}"] = {b: cmp("G+T_center", base, b) for b in BANDS}
    comparisons["T_mean_minus_G672"] = {b: cmp("T_mean", "G672", b) for b in BANDS}

    payload = {"categories": cats, "macro": macro, "comparisons": comparisons,
               "primary_band": "tiny_le_0.1pct", "controls": controls}
    (OUT / "summary.json").write_text(json.dumps(payload, indent=2, default=float))
    with (OUT / "category_metrics.csv").open("w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=["category", "variant", "test_images", *BANDS],
                           lineterminator="\n")
        w.writeheader(); w.writerows(rows)

    print(f"\n=== pooled AU-PRO@0.05, category macro (n={len(cats)}: {cats}) ===")
    print(f"{'variant':<22}{'TINY':>10}{'All':>10}{'Small':>10}{'Large':>10}")
    for n in names:
        print(f"{n:<22}{macro[n]['tiny_le_0.1pct']:>10.4f}{macro[n]['all']:>10.4f}"
              f"{macro[n]['small_0.1_to_1pct']:>10.4f}{macro[n]['large_gt_1pct']:>10.4f}")
    print("\n=== TINY-band paired bootstrap ===")
    for name, bands in comparisons.items():
        e = bands["tiny_le_0.1pct"]
        sig = "SIG" if (e["ci"][0] > 0) == (e["ci"][1] > 0) else "n.s."
        print(f"{name:<30} {e['mean_delta']:+.4f} CI[{e['ci'][0]:+.4f},{e['ci'][1]:+.4f}]"
              f" {e['wins']}/{e['n']} {sig}")
    print("\n=== all-band paired bootstrap ===")
    for name, bands in comparisons.items():
        e = bands["all"]
        sig = "SIG" if (e["ci"][0] > 0) == (e["ci"][1] > 0) else "n.s."
        print(f"{name:<30} {e['mean_delta']:+.4f} CI[{e['ci'][0]:+.4f},{e['ci'][1]:+.4f}]"
              f" {e['wins']}/{e['n']} {sig}")
    print(f"\nWROTE {OUT}")


if __name__ == "__main__":
    main()
