import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.analysis.patchcore_candidate_regions import (
    collect_predictions,
    get_field,
    take_item,
    mask_to_2d,
    normalize_map,
)


def compute_iou_f1(pred_mask, gt_mask):
    pred = pred_mask > 0
    gt = gt_mask > 0

    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()
    union = np.logical_or(pred, gt).sum()

    iou = float(tp / union) if union > 0 else 0.0

    denom = 2 * tp + fp + fn
    f1 = float(2 * tp / denom) if denom > 0 else 0.0

    return iou, f1


def largest_component(mask):
    mask = mask.astype(bool)
    h, w = mask.shape

    visited = np.zeros_like(mask, dtype=bool)
    best_pixels = []

    for y in range(h):
        for x in range(w):
            if visited[y, x] or not mask[y, x]:
                continue

            stack = [(y, x)]
            visited[y, x] = True
            pixels = []

            while stack:
                cy, cx = stack.pop()
                pixels.append((cy, cx))

                for ny, nx in [(cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)]:
                    if ny < 0 or ny >= h or nx < 0 or nx >= w:
                        continue
                    if visited[ny, nx] or not mask[ny, nx]:
                        continue
                    visited[ny, nx] = True
                    stack.append((ny, nx))

            if len(pixels) > len(best_pixels):
                best_pixels = pixels

    out = np.zeros_like(mask, dtype=bool)
    for y, x in best_pixels:
        out[y, x] = True

    return out


def build_mask(anomaly_map, method, value, keep_largest=False):
    if method == "fixed":
        threshold = float(value)
    elif method == "percentile":
        threshold = float(np.percentile(anomaly_map, value))
    elif method == "mean_std":
        threshold = float(anomaly_map.mean() + value * anomaly_map.std())
    else:
        raise ValueError(f"Unknown method: {method}")

    mask = anomaly_map >= threshold

    if keep_largest and mask.sum() > 0:
        mask = largest_component(mask)

    return mask, threshold


def collect_maps(args, category):
    predictions = collect_predictions(args, category)

    samples = []

    for batch in predictions:
        image_paths = get_field(batch, "image_path")
        gt_masks = get_field(batch, "gt_mask")
        anomaly_maps = get_field(batch, "anomaly_map")

        batch_size = len(image_paths) if isinstance(image_paths, (list, tuple)) else 1

        for i in range(batch_size):
            image_path = str(take_item(image_paths, i))
            gt_mask = mask_to_2d(take_item(gt_masks, i))
            anomaly_map = normalize_map(take_item(anomaly_maps, i))

            if gt_mask is None or anomaly_map is None:
                continue

            gt_mask = gt_mask > 0

            if gt_mask.sum() == 0:
                continue

            samples.append(
                {
                    "image_path": image_path,
                    "gt_mask": gt_mask,
                    "anomaly_map": anomaly_map,
                }
            )

    return samples


def evaluate_category(args, category):
    samples = collect_maps(args, category)

    configs = []

    for t in np.linspace(0.05, 0.95, 19):
        configs.append(("fixed", float(t), False))
        configs.append(("fixed", float(t), True))

    for p in [90, 92, 94, 95, 96, 97, 98, 99]:
        configs.append(("percentile", float(p), False))
        configs.append(("percentile", float(p), True))

    for k in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]:
        configs.append(("mean_std", float(k), False))
        configs.append(("mean_std", float(k), True))

    rows = []

    for method, value, keep_largest in configs:
        ious = []
        f1s = []
        areas = []
        thresholds = []

        for sample in samples:
            pred_mask, threshold = build_mask(
                sample["anomaly_map"],
                method=method,
                value=value,
                keep_largest=keep_largest,
            )

            iou, f1 = compute_iou_f1(pred_mask, sample["gt_mask"])

            ious.append(iou)
            f1s.append(f1)
            areas.append(int(pred_mask.sum()))
            thresholds.append(threshold)

        rows.append(
            {
                "category": category,
                "method": method,
                "value": value,
                "keep_largest": keep_largest,
                "num_samples": len(samples),
                "mean_iou": float(np.mean(ious)) if ious else 0.0,
                "mean_f1": float(np.mean(f1s)) if f1s else 0.0,
                "mean_pred_area": float(np.mean(areas)) if areas else 0.0,
                "mean_threshold": float(np.mean(thresholds)) if thresholds else 0.0,
            }
        )

    out_dir = Path(args.output_root) / "MVTecAD" / category
    out_dir.mkdir(parents=True, exist_ok=True)

    detail_df = pd.DataFrame(rows)
    detail_csv = out_dir / "anomaly_map_calibration_detail.csv"
    detail_df.to_csv(detail_csv, index=False)

    best = detail_df.sort_values("mean_f1", ascending=False).iloc[0]

    summary = {
        "category": category,
        "num_samples": int(best["num_samples"]),
        "best_method": best["method"],
        "best_value": float(best["value"]),
        "best_keep_largest": bool(best["keep_largest"]),
        "best_mean_iou": float(best["mean_iou"]),
        "best_mean_f1": float(best["mean_f1"]),
        "best_mean_pred_area": float(best["mean_pred_area"]),
        "best_mean_threshold": float(best["mean_threshold"]),
        "detail_csv": str(detail_csv),
    }

    print(
        f"[DONE] {category}: "
        f"best={summary['best_method']} value={summary['best_value']} "
        f"keep_largest={summary['best_keep_largest']} "
        f"F1={summary['best_mean_f1']:.4f}"
    )

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="datasets/MVTecAD")
    parser.add_argument("--categories", nargs="+", default=["grid", "screw", "leather", "wood"])
    parser.add_argument("--output_root", type=str, default="results/analysis/anomaly_map_calibration")
    parser.add_argument("--work_root", type=str, default="runs/analysis/anomaly_map_calibration")
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    summaries = []

    for category in args.categories:
        summaries.append(evaluate_category(args, category))

    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    summary_df = pd.DataFrame(summaries)

    mean_row = {
        "category": "MEAN",
        "num_samples": summary_df["num_samples"].sum(),
        "best_method": "",
        "best_value": "",
        "best_keep_largest": "",
        "best_mean_iou": summary_df["best_mean_iou"].mean(),
        "best_mean_f1": summary_df["best_mean_f1"].mean(),
        "best_mean_pred_area": summary_df["best_mean_pred_area"].mean(),
        "best_mean_threshold": summary_df["best_mean_threshold"].mean(),
        "detail_csv": "",
    }

    summary_df = pd.concat([summary_df, pd.DataFrame([mean_row])], ignore_index=True)

    summary_csv = out_root / "anomaly_map_calibration_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    print(summary_df.to_string(index=False))
    print(f"[DONE] Saved summary to: {summary_csv}")


if __name__ == "__main__":
    main()
