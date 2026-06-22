import argparse
import csv
from pathlib import Path
import sys

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from anomalib.engine import Engine
from anomalib.models import Patchcore

from experiments.baselines.patchcore_mvtec import build_mvtec_folder_datamodule


def get_field(batch, name):
    if isinstance(batch, dict):
        return batch.get(name, None)
    return getattr(batch, name, None)


def to_numpy(x):
    if x is None:
        return None
    if hasattr(x, "detach"):
        x = x.detach().cpu()
    if hasattr(x, "numpy"):
        x = x.numpy()
    return x


def take_item(x, index):
    if x is None:
        return None
    if isinstance(x, (list, tuple)):
        return x[index]
    try:
        return x[index]
    except Exception:
        return x


def mask_to_2d(mask):
    mask = to_numpy(mask)
    if mask is None:
        return None

    mask = np.squeeze(mask)

    if mask.ndim > 2:
        mask = mask[0]

    return mask


def normalize_map(anomaly_map):
    anomaly_map = mask_to_2d(anomaly_map)
    if anomaly_map is None:
        return None

    anomaly_map = anomaly_map.astype(np.float32)
    amin = float(np.nanmin(anomaly_map))
    amax = float(np.nanmax(anomaly_map))

    if amax - amin < 1e-8:
        return np.zeros_like(anomaly_map, dtype=np.float32)

    return (anomaly_map - amin) / (amax - amin)


def compute_pixel_f1(pred, gt):
    pred = pred > 0
    gt = gt > 0

    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()

    denom = 2 * tp + fp + fn
    if denom == 0:
        return None

    return float(2 * tp / denom)


def collect_prediction_maps(args, category):
    run_args = argparse.Namespace(
        data_root=args.data_root,
        category=category,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    datamodule = build_mvtec_folder_datamodule(run_args)

    pre_processor = Patchcore.configure_pre_processor(
        image_size=(256, 256),
        center_crop_size=(224, 224),
    )

    model = Patchcore(
        backbone="wide_resnet50_2",
        layers=["layer2", "layer3"],
        pre_trained=True,
        coreset_sampling_ratio=0.1,
        num_neighbors=9,
        pre_processor=pre_processor,
    )

    work_dir = Path(args.work_root) / "MVTecAD" / category
    work_dir.mkdir(parents=True, exist_ok=True)

    engine = Engine(
        default_root_dir=str(work_dir),
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        logger=False,
    )

    print(f"[INFO] Fitting PatchCore: {category}")
    engine.fit(model=model, datamodule=datamodule)

    print(f"[INFO] Predicting anomaly maps: {category}")
    predictions = engine.predict(model=model, datamodule=datamodule)

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


def diagnose_category(args, category):
    samples = collect_prediction_maps(args, category)

    thresholds = np.linspace(args.min_threshold, args.max_threshold, args.num_thresholds)

    rows = []
    best = {
        "category": category,
        "best_threshold": None,
        "best_mean_pixel_f1": -1.0,
        "num_samples": len(samples),
    }

    for threshold in thresholds:
        f1_values = []

        for sample in samples:
            pred_mask = sample["anomaly_map"] >= threshold
            f1 = compute_pixel_f1(pred_mask, sample["gt_mask"])
            if f1 is not None:
                f1_values.append(f1)

        mean_f1 = float(np.mean(f1_values)) if f1_values else 0.0

        rows.append(
            {
                "category": category,
                "threshold": float(threshold),
                "mean_pixel_f1": mean_f1,
                "num_samples": len(f1_values),
            }
        )

        if mean_f1 > best["best_mean_pixel_f1"]:
            best["best_threshold"] = float(threshold)
            best["best_mean_pixel_f1"] = mean_f1

    out_dir = Path(args.output_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    curve_path = out_dir / f"{category}_threshold_curve.csv"
    with curve_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["category", "threshold", "mean_pixel_f1", "num_samples"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(
        f"[DONE] {category}: best_threshold={best['best_threshold']:.4f}, "
        f"best_mean_pixel_f1={best['best_mean_pixel_f1']:.4f}, "
        f"samples={best['num_samples']}"
    )

    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="datasets/MVTecAD")
    parser.add_argument("--categories", nargs="+", default=["grid", "screw", "leather", "wood"])
    parser.add_argument("--output_root", type=str, default="results/analysis/patchcore_threshold_diagnosis")
    parser.add_argument("--work_root", type=str, default="runs/analysis/patchcore_threshold_diagnosis")
    parser.add_argument("--min_threshold", type=float, default=0.05)
    parser.add_argument("--max_threshold", type=float, default=0.95)
    parser.add_argument("--num_thresholds", type=int, default=19)
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    summary_rows = []

    for category in args.categories:
        best = diagnose_category(args, category)
        summary_rows.append(best)

    summary_path = Path(args.output_root) / "threshold_diagnosis_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "category",
                "best_threshold",
                "best_mean_pixel_f1",
                "num_samples",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"[DONE] Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()
