#!/usr/bin/env python3

import csv
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import tifffile
from sklearn.metrics import roc_auc_score

ROOT = Path("/root/private_data/iad-vlm-anomaly")

META = ROOT / "orbitad/results/a10_multilayer_v1/public_meta.csv"
NORMAL = ROOT / "orbitad/results/a68_normal_only_resolution_router_v1/normal_image_statistics.csv"

A64 = ROOT / "ad2_model_zoo/results/a64_superad_reg4_ad2_all_true672_token_preserving_v1"
A45 = ROOT / "ad2_model_zoo/results/a45_superad_reg4_public_v1"
A60 = ROOT / (
    "ad2_model_zoo/results/"
    "a60_superad_reg4_sheetmetal_true672_tiled_query_v1/"
    "sheet_metal/component_maps/seed=0/sheet_metal"
)

CATEGORIES = [
    "can",
    "fabric",
    "fruit_jelly",
    "rice",
    "sheet_metal",
    "vial",
    "wallplugs",
    "walnuts",
]


def component_paths(row):
    rel = Path(row["image_path"])
    category = row["category"]

    if category == "sheet_metal":
        stem = A60 / rel.parent.name / rel.stem
        return (
            Path(str(stem) + "_global.tiff"),
            Path(str(stem) + "_tiled.tiff"),
        )

    global_path = (
        A45
        / category
        / "anomaly_maps/seed=0"
        / category
        / "test"
        / rel.parent.name
        / f"{rel.stem}.tiff"
    )

    tiled_stem = (
        A64
        / category
        / "component_maps/seed=0"
        / category
        / rel.parent.name
        / rel.stem
    )

    tiled_path = Path(str(tiled_stem) + "_tiled.tiff")

    return global_path, tiled_path


def load_calibration():
    rows = list(csv.DictReader(NORMAL.open()))
    grouped = defaultdict(list)

    for row in rows:
        grouped[row["category"]].append(row)

    result = {}

    for category, values in grouped.items():
        result[category] = {
            "global_q99": float(
                np.median([float(v["global_q99"]) for v in values])
            ),
            "tiled_q99": float(
                np.median([float(v["tiled_q99"]) for v in values])
            ),
        }

    return result


def make_scores(g, t, cal):
    eps = 1e-12

    gn = g / max(cal["global_q99"], eps)
    tn = t / max(cal["tiled_q99"], eps)

    # A69 正式 q99_mean 公式
    q99_mean = 0.5 * (gn + tn)

    return {
        "global": g,
        "tiled": t,
        "q99_mean": q99_mean,
    }


def main():
    calibration = load_calibration()
    rows = list(csv.DictReader(META.open()))

    rows_by_category = defaultdict(list)
    for row in rows:
        rows_by_category[row["category"]].append(row)

    results = {
        "global": {},
        "tiled": {},
        "q99_mean": {},
    }

    for category in CATEGORIES:

        print(f"\n[{category}]")

        gt_all = []
        pred_all = {
            "global": [],
            "tiled": [],
            "q99_mean": [],
        }

        category_rows = rows_by_category[category]

        for idx, row in enumerate(category_rows, 1):

            gp, tp = component_paths(row)

            if not gp.exists():
                raise FileNotFoundError(gp)
            if not tp.exists():
                raise FileNotFoundError(tp)

            g = np.asarray(tifffile.imread(gp), dtype=np.float32)
            t = np.asarray(tifffile.imread(tp), dtype=np.float32)

            if g.shape != t.shape:
                t = cv2.resize(
                    t,
                    (g.shape[1], g.shape[0]),
                    interpolation=cv2.INTER_LINEAR,
                )

            scores = make_scores(
                g,
                t,
                calibration[category],
            )

            # 与 SuperADD 的 AU-ROC 计算方式保持一致：
            # GT resize 到 prediction map 分辨率
            if int(row["label"]) == 0:
                mask = np.zeros(g.shape, dtype=np.uint8)
            else:
                mask_path = (
                    ROOT
                    / "datasets/MVTec_AD_2"
                    / row["mask_path"]
                )

                mask = cv2.imread(
                    str(mask_path),
                    cv2.IMREAD_GRAYSCALE,
                )

                if mask is None:
                    raise FileNotFoundError(mask_path)

                mask = cv2.resize(
                    mask,
                    (g.shape[1], g.shape[0]),
                    interpolation=cv2.INTER_LINEAR,
                )

                mask = (mask > 0).astype(np.uint8)

            gt_all.append(mask.reshape(-1))

            for method in pred_all:
                pred_all[method].append(
                    scores[method].reshape(-1)
                )

            if idx % 25 == 0 or idx == len(category_rows):
                print(
                    f"\r  {idx}/{len(category_rows)}",
                    end="",
                    flush=True,
                )

        print()

        gt = np.concatenate(gt_all)

        for method in ["global", "tiled", "q99_mean"]:
            pred = np.concatenate(pred_all[method])

            score = roc_auc_score(
                gt,
                pred,
                max_fpr=0.05,
            )

            results[method][category] = float(score)

            print(
                f"  {method:<10s}: "
                f"{score:.6f} "
                f"({score * 100:.2f}%)"
            )

    print("\n" + "=" * 58)
    print("MVTec AD 2 TEST_PUBLIC — AU-ROC@0.05")
    print("=" * 58)

    for method in ["global", "tiled", "q99_mean"]:

        vals = [
            results[method][c]
            for c in CATEGORIES
        ]

        macro = float(np.mean(vals))

        print(
            f"{method:<10s}: "
            f"{macro:.6f} "
            f"({macro * 100:.2f}%)"
        )

    print("=" * 58)
    print("A69 = q99_mean")


if __name__ == "__main__":
    main()
