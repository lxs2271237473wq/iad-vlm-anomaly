import argparse
import csv
from collections import deque
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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


def connected_components(binary_mask):
    binary_mask = binary_mask.astype(bool)
    h, w = binary_mask.shape

    visited = np.zeros_like(binary_mask, dtype=bool)
    components = []

    for y in range(h):
        for x in range(w):
            if not binary_mask[y, x] or visited[y, x]:
                continue

            q = deque([(y, x)])
            visited[y, x] = True
            pixels = []

            while q:
                cy, cx = q.popleft()
                pixels.append((cy, cx))

                for ny, nx in [
                    (cy - 1, cx),
                    (cy + 1, cx),
                    (cy, cx - 1),
                    (cy, cx + 1),
                ]:
                    if ny < 0 or ny >= h or nx < 0 or nx >= w:
                        continue
                    if visited[ny, nx] or not binary_mask[ny, nx]:
                        continue

                    visited[ny, nx] = True
                    q.append((ny, nx))

            components.append(pixels)

    return components


def component_to_record(component, anomaly_map, gt_mask=None):
    ys = np.array([p[0] for p in component])
    xs = np.array([p[1] for p in component])

    x1 = int(xs.min())
    y1 = int(ys.min())
    x2 = int(xs.max())
    y2 = int(ys.max())

    area = int(len(component))
    cx = float(xs.mean())
    cy = float(ys.mean())

    comp_mask = np.zeros_like(anomaly_map, dtype=bool)
    comp_mask[ys, xs] = True

    mean_score = float(anomaly_map[comp_mask].mean())
    max_score = float(anomaly_map[comp_mask].max())

    gt_iou = None
    gt_f1 = None

    if gt_mask is not None and gt_mask.sum() > 0:
        gt = gt_mask.astype(bool)
        inter = np.logical_and(comp_mask, gt).sum()
        union = np.logical_or(comp_mask, gt).sum()
        denom = comp_mask.sum() + gt.sum()

        gt_iou = float(inter / union) if union > 0 else 0.0
        gt_f1 = float(2 * inter / denom) if denom > 0 else 0.0

    return {
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "cx": cx,
        "cy": cy,
        "area": area,
        "mean_score": mean_score,
        "max_score": max_score,
        "gt_iou": gt_iou,
        "gt_f1": gt_f1,
        "mask": comp_mask,
    }


def save_visual(image, gt_mask, anomaly_map, records, out_path, title):
    fig = plt.figure(figsize=(16, 4))

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
    ax3.imshow(anomaly_map)
    ax3.set_title("Anomaly Map")
    ax3.axis("off")

    ax4 = fig.add_subplot(1, 4, 4)
    ax4.imshow(image)
    for idx, rec in enumerate(records, start=1):
        x1, y1, x2, y2 = rec["x1"], rec["y1"], rec["x2"], rec["y2"]
        ax4.add_patch(
            plt.Rectangle(
                (x1, y1),
                x2 - x1 + 1,
                y2 - y1 + 1,
                fill=False,
                linewidth=2,
            )
        )
        ax4.scatter([rec["cx"]], [rec["cy"]], s=20)
        ax4.text(x1, max(0, y1 - 3), str(idx), fontsize=8)

    ax4.set_title("Candidate Regions")
    ax4.axis("off")

    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def load_thresholds(path):
    df = pd.read_csv(path)
    thresholds = {}
    for _, row in df.iterrows():
        thresholds[str(row["category"])] = float(row["best_threshold"])
    return thresholds


def collect_predictions(args, category):
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
    return engine.predict(model=model, datamodule=datamodule)


def run_category(args, category, threshold):
    predictions = collect_predictions(args, category)

    out_dir = Path(args.output_root) / "MVTecAD" / category / "candidate_regions"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    image_counter = 0
    no_candidate_count = 0

    for batch in predictions:
        images = get_field(batch, "image")
        image_paths = get_field(batch, "image_path")
        gt_masks = get_field(batch, "gt_mask")
        anomaly_maps = get_field(batch, "anomaly_map")

        batch_size = len(image_paths) if isinstance(image_paths, (list, tuple)) else 1

        for i in range(batch_size):
            image = image_tensor_to_uint8(take_item(images, i))
            image_path = str(take_item(image_paths, i))
            gt_mask = mask_to_2d(take_item(gt_masks, i))
            anomaly_map = normalize_map(take_item(anomaly_maps, i))

            if image is None or anomaly_map is None:
                continue

            if gt_mask is not None:
                gt_mask = gt_mask > 0
                if gt_mask.sum() == 0:
                    continue

            binary_mask = anomaly_map >= threshold
            components = connected_components(binary_mask)

            records = []
            for comp in components:
                if len(comp) < args.min_area:
                    continue
                rec = component_to_record(comp, anomaly_map, gt_mask)
                records.append(rec)

            records = sorted(records, key=lambda r: (r["area"], r["mean_score"]), reverse=True)
            records = records[: args.top_components]

            image_counter += 1

            if len(records) == 0:
                no_candidate_count += 1

            visual_name = f"{category}_{image_counter:04d}_candidates.png"
            visual_path = out_dir / visual_name

            title = f"{category} | threshold={threshold:.2f} | candidates={len(records)}"
            save_visual(image, gt_mask, anomaly_map, records, visual_path, title)

            if len(records) == 0:
                rows.append(
                    {
                        "category": category,
                        "image_path": image_path,
                        "component_rank": 0,
                        "threshold": threshold,
                        "x1": "",
                        "y1": "",
                        "x2": "",
                        "y2": "",
                        "cx": "",
                        "cy": "",
                        "area": 0,
                        "mean_score": "",
                        "max_score": "",
                        "gt_iou": "",
                        "gt_f1": "",
                        "visual_path": str(visual_path),
                    }
                )

            for rank, rec in enumerate(records, start=1):
                rows.append(
                    {
                        "category": category,
                        "image_path": image_path,
                        "component_rank": rank,
                        "threshold": threshold,
                        "x1": rec["x1"],
                        "y1": rec["y1"],
                        "x2": rec["x2"],
                        "y2": rec["y2"],
                        "cx": f"{rec['cx']:.2f}",
                        "cy": f"{rec['cy']:.2f}",
                        "area": rec["area"],
                        "mean_score": f"{rec['mean_score']:.6f}",
                        "max_score": f"{rec['max_score']:.6f}",
                        "gt_iou": "" if rec["gt_iou"] is None else f"{rec['gt_iou']:.6f}",
                        "gt_f1": "" if rec["gt_f1"] is None else f"{rec['gt_f1']:.6f}",
                        "visual_path": str(visual_path),
                    }
                )

    csv_path = out_dir / "candidate_regions.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "category",
            "image_path",
            "component_rank",
            "threshold",
            "x1",
            "y1",
            "x2",
            "y2",
            "cx",
            "cy",
            "area",
            "mean_score",
            "max_score",
            "gt_iou",
            "gt_f1",
            "visual_path",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[DONE] {category}: images={image_counter}, no_candidate={no_candidate_count}")
    print(f"[DONE] Saved candidates to: {csv_path}")

    return {
        "category": category,
        "threshold": threshold,
        "num_images": image_counter,
        "no_candidate_images": no_candidate_count,
        "candidate_rows": len(rows),
        "candidate_csv": str(csv_path),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="datasets/MVTecAD")
    parser.add_argument("--categories", nargs="+", default=["grid", "screw", "leather", "wood"])
    parser.add_argument("--threshold_csv", type=str, default="results/analysis/patchcore_threshold_diagnosis/threshold_diagnosis_summary.csv")
    parser.add_argument("--output_root", type=str, default="results/analysis/patchcore_candidate_regions")
    parser.add_argument("--work_root", type=str, default="runs/analysis/patchcore_candidate_regions")
    parser.add_argument("--top_components", type=int, default=3)
    parser.add_argument("--min_area", type=int, default=20)
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    thresholds = load_thresholds(args.threshold_csv)

    summary = []
    for category in args.categories:
        if category not in thresholds:
            raise KeyError(f"Category {category} not found in threshold csv: {args.threshold_csv}")

        result = run_category(args, category, thresholds[category])
        summary.append(result)

    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    summary_path = out_root / "candidate_region_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "category",
            "threshold",
            "num_images",
            "no_candidate_images",
            "candidate_rows",
            "candidate_csv",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary)

    print(f"[DONE] Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()
