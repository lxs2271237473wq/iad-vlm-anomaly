import argparse
import csv
from pathlib import Path

import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


import matplotlib.pyplot as plt
import numpy as np
import torch

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
    if hasattr(x, "__len__") and not isinstance(x, (str, bytes)):
        try:
            return x[index]
        except Exception:
            return x
    return x


def image_tensor_to_uint8(img):
    img = to_numpy(img)

    if img is None:
        return None

    if img.ndim == 3 and img.shape[0] in [1, 3]:
        img = np.transpose(img, (1, 2, 0))

    img = img.astype(np.float32)

    if img.max() <= 1.5:
        img = img * 255.0

    img = np.clip(img, 0, 255).astype(np.uint8)

    if img.ndim == 3 and img.shape[-1] == 1:
        img = img[..., 0]

    return img


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


def compute_pixel_f1(pred_mask, gt_mask):
    pred = mask_to_2d(pred_mask)
    gt = mask_to_2d(gt_mask)

    if pred is None or gt is None:
        return None

    pred = pred > 0
    gt = gt > 0

    if gt.sum() == 0:
        return None

    inter = np.logical_and(pred, gt).sum()
    denom = pred.sum() + gt.sum()

    if denom == 0:
        return 0.0

    return float(2.0 * inter / denom)


def get_scalar(x):
    x = to_numpy(x)
    if x is None:
        return None
    try:
        return float(np.asarray(x).reshape(-1)[0])
    except Exception:
        return None


def save_visual(sample, out_path, title):
    image = sample["image"]
    gt_mask = sample["gt_mask"]
    pred_mask = sample["pred_mask"]
    anomaly_map = sample["anomaly_map"]

    fig = plt.figure(figsize=(14, 4))

    ax1 = fig.add_subplot(1, 4, 1)
    ax1.imshow(image)
    ax1.set_title("Image")
    ax1.axis("off")

    ax2 = fig.add_subplot(1, 4, 2)
    if gt_mask is not None:
        ax2.imshow(gt_mask, cmap="gray")
    ax2.set_title("GT Mask")
    ax2.axis("off")

    ax3 = fig.add_subplot(1, 4, 3)
    if anomaly_map is not None:
        ax3.imshow(anomaly_map)
    ax3.set_title("Anomaly Map")
    ax3.axis("off")

    ax4 = fig.add_subplot(1, 4, 4)
    ax4.imshow(image)
    if pred_mask is not None:
        ax4.imshow(pred_mask > 0, alpha=0.45)
    ax4.set_title("Pred Mask Overlay")
    ax4.axis("off")

    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run_category(args, category):
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

    print(f"[INFO] Fitting PatchCore for category: {category}")
    engine.fit(model=model, datamodule=datamodule)

    print(f"[INFO] Predicting samples for category: {category}")
    predictions = engine.predict(model=model, datamodule=datamodule)

    if predictions is None:
        raise RuntimeError("engine.predict returned None. Cannot export failure visualizations.")

    samples = []

    for batch in predictions:
        images = get_field(batch, "image")
        image_paths = get_field(batch, "image_path")
        gt_masks = get_field(batch, "gt_mask")
        pred_masks = get_field(batch, "pred_mask")
        anomaly_maps = get_field(batch, "anomaly_map")
        pred_scores = get_field(batch, "pred_score")
        gt_labels = get_field(batch, "gt_label")

        batch_size = len(image_paths) if isinstance(image_paths, (list, tuple)) else 1

        for i in range(batch_size):
            image = image_tensor_to_uint8(take_item(images, i))
            gt_mask = mask_to_2d(take_item(gt_masks, i))
            pred_mask = mask_to_2d(take_item(pred_masks, i))
            anomaly_map = normalize_map(take_item(anomaly_maps, i))
            pred_score = get_scalar(take_item(pred_scores, i))
            gt_label = get_scalar(take_item(gt_labels, i))
            image_path = str(take_item(image_paths, i))

            if image is None:
                continue

            gt_area = 0 if gt_mask is None else int((gt_mask > 0).sum())
            pred_area = 0 if pred_mask is None else int((pred_mask > 0).sum())
            pixel_f1 = compute_pixel_f1(pred_mask, gt_mask)

            if gt_area == 0:
                continue

            samples.append(
                {
                    "category": category,
                    "image_path": image_path,
                    "image": image,
                    "gt_mask": gt_mask,
                    "pred_mask": pred_mask,
                    "anomaly_map": anomaly_map,
                    "pred_score": pred_score,
                    "gt_label": gt_label,
                    "gt_area": gt_area,
                    "pred_area": pred_area,
                    "pixel_f1": pixel_f1,
                }
            )

    samples = [s for s in samples if s["pixel_f1"] is not None]
    samples = sorted(samples, key=lambda x: x["pixel_f1"])

    out_dir = Path(args.output_root) / "MVTecAD" / category / "failures"
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "failure_cases.csv"

    rows = []
    for rank, sample in enumerate(samples[: args.topk], start=1):
        save_name = f"rank_{rank:02d}_f1_{sample['pixel_f1']:.4f}.png"
        save_path = out_dir / save_name

        title = (
            f"{category} | rank={rank} | "
            f"pixel_f1={sample['pixel_f1']:.4f} | "
            f"score={sample['pred_score']}"
        )

        save_visual(sample, save_path, title)

        rows.append(
            {
                "rank": rank,
                "category": category,
                "image_path": sample["image_path"],
                "pred_score": sample["pred_score"],
                "gt_label": sample["gt_label"],
                "gt_area": sample["gt_area"],
                "pred_area": sample["pred_area"],
                "pixel_f1": sample["pixel_f1"],
                "visual_path": str(save_path),
            }
        )

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "rank",
                "category",
                "image_path",
                "pred_score",
                "gt_label",
                "gt_area",
                "pred_area",
                "pixel_f1",
                "visual_path",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"[DONE] Category: {category}")
    print(f"[DONE] Saved visuals to: {out_dir}")
    print(f"[DONE] Saved case table to: {csv_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="datasets/MVTecAD")
    parser.add_argument("--categories", nargs="+", default=["grid"])
    parser.add_argument("--output_root", type=str, default="results/visualizations/patchcore")
    parser.add_argument("--work_root", type=str, default="runs/analysis/patchcore_failure_export")
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    for category in args.categories:
        run_category(args, category)


if __name__ == "__main__":
    main()
