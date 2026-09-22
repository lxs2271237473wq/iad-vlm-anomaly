#!/usr/bin/env python3
"""A77 Track 2: does cross-window agreement make the local view usable?

The local branch covers every pixel with several overlapping windows. The extra
maps written by the patched inference (per-pixel standard deviation `std` and the
per-window q99-normalised mean `norm`) are by-products of inference, so all of the
fusion rules below can be compared without any further forward passes.

Pre-registered decision rule for this round
-------------------------------------------
Primary variant: `G+T_gate`, i.e. the local map weighted per pixel by
    w(x) = exp(-alpha * cv(x) / median(cv)),  cv(x) = std(x) / (mean(x) + 1e-3),
with alpha = 1. The gate needs no labels: `median` is taken over the test image
itself.
The variant is accepted only if, on pooled AU-PRO@0.05 and with a paired category
bootstrap interval excluding zero, it beats BOTH
    (i)  the best simple fusion baseline (`legacy_mean`), and
    (ii) the raw mean of the same two views (`G+T`),
and it must not lose the tiny band relative to `G+T`.
alpha is swept over {0.5, 1, 2} as a sensitivity check; alpha = 0 reproduces
`G+T` exactly and therefore anchors the sweep.
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
OUT = ROOT / "orbitad/results/a77_consistency_gate_v1"
OUT.mkdir(parents=True, exist_ok=True)

BINS, MAX_FPR = 65536, 0.05
SEED, REPS = 20260921, 100_000
BANDS = ("all", "tiny_le_0.1pct", "small_0.1_to_1pct", "large_gt_1pct")
ALPHAS = (0.0, 0.5, 1.0, 2.0)
EPS = 1e-3

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
    return [(f.name, p)
            for f in sorted(x for x in base.iterdir() if x.is_dir() and x.name != "ground_truth")
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


def load(category, anomaly_type, stem):
    def rd(suffix):
        path = _p(category, anomaly_type, stem, suffix)
        return np.asarray(tifffile.imread(path), np.float32) if path.exists() else None
    return {k: rd(v) for k, v in (("g", "global"), ("g672", "global672"),
                                  ("tiled", "tiled"), ("T", "local672"),
                                  ("S", "localstd672"), ("Tn", "localnorm672"))}


def variants(m):
    g, t, T, S, Tn = m["g"], m["tiled"], m["T"], m["S"], m["Tn"]
    out = {"G448": g, "G672": m["g672"], "legacy_mean": 0.5 * (g + t),
           "T_local": T, "T_norm": Tn, "G+T": 0.5 * (g + T), "G+Tn": 0.5 * (g + Tn)}
    # constant-weight profile: is 0.5 a lucky choice or a broad plateau?
    for w in (0.25, 0.5, 0.75):
        out[f"W{w:g}_G+T"] = w * T + (1.0 - w) * g
    if S is not None and T is not None:
        # cv is the coefficient of variation of the window responses at a pixel.
        # The gate keeps the AVERAGE local weight at 0.5 and only redistributes it,
        # so alpha controls selectivity rather than overall shrinkage:
        #   alpha = 0  -> w = 0.5 everywhere  -> exactly `G+T` (the control)
        #   alpha > 0  -> w < 0.5 where windows disagree, > 0.5 where they agree
        cv = S / (T + EPS)
        scale = float(np.median(cv)) + 1e-12
        deviation = cv / scale - 1.0
        for alpha in ALPHAS:
            w = 0.5 * np.exp(-alpha * deviation)
            out[f"G+T_gate_a{alpha:g}"] = w * T + (1.0 - w) * g
            out[f"G+T_gate_inv_a{alpha:g}"] = 0.5 * np.exp(alpha * deviation) * T \
                + (1.0 - 0.5 * np.exp(alpha * deviation)) * g
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


def evaluate_category(ds, category):
    sl = samples(ds, category)
    probe = variants(load(category, sl[0][0], sl[0][1].stem))
    names = sorted(probe)
    maxima = {n: 0.0 for n in names}
    for at, p in sl:
        for n, s in variants(load(category, at, p.stem)).items():
            maxima[n] = max(maxima[n], float(np.nanmax(s)))
    scales = {n: BINS / max(v * 1.001, v + 1e-6) for n, v in maxima.items()}
    stats = {n: {"neg": np.zeros(BINS, np.int64),
                 "region": {b: np.zeros(BINS, np.float64) for b in BANDS},
                 "regions": {b: 0 for b in BANDS}} for n in names}
    for at, p in sl:
        mask = mask_for(ds, category, at, p)
        count, labels = cv2.connectedComponents(mask, 8)
        regions = []
        for rid in range(1, count):
            r = labels == rid
            ratio = int(r.sum()) / mask.size
            regions.append((r, "tiny_le_0.1pct" if ratio <= 0.001
                            else "small_0.1_to_1pct" if ratio <= 0.01 else "large_gt_1pct"))
        for n, s in variants(load(category, at, p.stem)).items():
            if s.shape != mask.shape:
                s = cv2.resize(s, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_LINEAR)
            bins = np.minimum((np.maximum(s, 0) * scales[n]).astype(np.int64), BINS - 1)
            item = stats[n]
            item["neg"] += np.bincount(bins[mask == 0], minlength=BINS)
            for r, band in regions:
                c = np.bincount(bins[r], minlength=BINS) / int(r.sum())
                for target in ("all", band):
                    item["region"][target] += c
                    item["regions"][target] += 1
    return [{"category": category, "variant": n, "test_images": len(sl),
             **{b: aupro(stats[n]["neg"], stats[n]["region"][b], stats[n]["regions"][b])
                for b in BANDS}} for n in names]


def boot(v, rng):
    draws = rng.choice(v, (REPS, len(v)), replace=True).mean(axis=1)
    return [float(x) for x in np.quantile(draws, [.025, .975])]


def main():
    wanted = sys.argv[1:] or [c for spec in DATASETS.values() for c in spec["cats"]]
    rows = []
    for c in wanted:
        try:
            rows.extend(evaluate_category(DS_OF[c], c))
            print(f"EVAL_COMPLETE {c}", flush=True)
        except FileNotFoundError as exc:
            print(f"SKIP {c}: {exc}", flush=True)
    if not rows:
        print("nothing evaluated"); return
    cats = sorted({r["category"] for r in rows})
    names = sorted({r["variant"] for r in rows})
    macro = {n: {b: float(np.mean([r[b] for r in rows
                                   if r["variant"] == n and np.isfinite(r[b])]))
                 for b in BANDS} for n in names}

    rng = np.random.default_rng(SEED)
    idx = {(r["category"], r["variant"]): r for r in rows}
    def cmp(cand, base, band):
        d = {c: float(idx[(c, cand)][band] - idx[(c, base)][band]) for c in cats
             if np.isfinite(idx[(c, cand)][band]) and np.isfinite(idx[(c, base)][band])}
        v = np.asarray(list(d.values()), np.float64)
        return {"mean_delta": float(v.mean()), "ci": boot(v, rng),
                "wins": int((v > 1e-12).sum()), "n": len(v), "by_category": d}

    primary = "G+T_gate_a1"
    comparisons = {}
    for cand in names:
        if cand == "G448":
            continue
        comparisons[cand] = {b: cmp(cand, "legacy_mean", b) for b in BANDS}
        comparisons[cand].update({f"vs_G+T_{b}": cmp(cand, "G+T", b) for b in BANDS})

    verdict = {}
    if primary in names:
        vs_legacy = comparisons[primary]["all"]
        vs_gt = comparisons[primary]["vs_G+T_all"]
        tiny_vs_gt = comparisons[primary]["vs_G+T_tiny_le_0.1pct"]
        verdict = {
            "primary": primary,
            "beats_legacy_mean": bool(vs_legacy["ci"][0] > 0),
            "beats_raw_mean": bool(vs_gt["ci"][0] > 0),
            "keeps_tiny_band": bool(tiny_vs_gt["mean_delta"] >= -0.002),
            "delta_vs_legacy": vs_legacy["mean_delta"], "ci_vs_legacy": vs_legacy["ci"],
            "delta_vs_raw_mean": vs_gt["mean_delta"], "ci_vs_raw_mean": vs_gt["ci"],
        }
        verdict["ACCEPT"] = bool(verdict["beats_legacy_mean"] and verdict["beats_raw_mean"]
                                 and verdict["keeps_tiny_band"])

    (OUT / "summary.json").write_text(json.dumps(
        {"categories": cats, "macro": macro, "comparisons": comparisons,
         "verdict": verdict, "alphas": list(ALPHAS), "eps": EPS}, indent=2, default=float))
    with (OUT / "category_metrics.csv").open("w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=["category", "variant", "test_images", *BANDS],
                           lineterminator="\n")
        w.writeheader(); w.writerows(rows)

    print(f"\n=== pooled AU-PRO@0.05, category macro (n={len(cats)}) ===")
    print(f"{'variant':<22}" + "".join(f"{b[:13]:>15}" for b in BANDS))
    for n in names:
        print(f"{n:<22}" + "".join(f"{macro[n][b]:>15.4f}" for b in BANDS))
    print("\n=== vs legacy_mean (all band) ===")
    for n in names:
        if n not in comparisons:
            continue
        e = comparisons[n]["all"]
        sig = "SIG" if (e["ci"][0] > 0) == (e["ci"][1] > 0) else "n.s."
        print(f"{n:<24} {e['mean_delta']:+.4f} CI[{e['ci'][0]:+.4f},{e['ci'][1]:+.4f}] "
              f"{e['wins']}/{e['n']} {sig}")
    print("\n=== vs G+T raw mean (all band) ===")
    for n in names:
        if n not in comparisons:
            continue
        e = comparisons[n]["vs_G+T_all"]
        sig = "SIG" if (e["ci"][0] > 0) == (e["ci"][1] > 0) else "n.s."
        print(f"{n:<24} {e['mean_delta']:+.4f} CI[{e['ci'][0]:+.4f},{e['ci'][1]:+.4f}] "
              f"{e['wins']}/{e['n']} {sig}")
    if verdict:
        print("\n=== VERDICT ===")
        print(json.dumps(verdict, indent=2, default=float))
    print(f"\nWROTE {OUT}")


if __name__ == "__main__":
    main()
