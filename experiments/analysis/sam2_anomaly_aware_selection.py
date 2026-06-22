import argparse
import csv
import sys
from collections import deque
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from anomalib.engine import Engine
from anomalib.models import Patchcore
from experiments.baselines.patchcore_mvtec import build_mvtec_folder_datamodule


def add_sam2_to_path(sam2_root):
    sam2_root = Path(sam2_root).resolve()
    if str(sam2_root) not in sys.path:
        sys.path.insert(0, str(sam2_root))


def load_sam2_predictor(args):
    add_sam2_to_path(args.sam2_root)

    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_sam2(args.model_cfg, args.checkpoint, device=device)
    predictor = SAM2ImagePredictor(model)

    print(f"[INFO] Loaded SAM2 on device: {device}")
    return predictor


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

    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)

    return img


def mask_to_2d(mask):
    mask = to_numpy(mask)
    if mask is None:
        return None

    mask = np.squeeze(mask)
    if mask.ndim > 2:
        mask = mask[0]

    return mask


def resize_float_map(arr, size_hw):
    h, w = size_hw
    img = Image.fromarray(arr.astype(np.float32))
    img = img.resize((w, h), Image.BILINEAR)
    return np.array(img).astype(np.float32)


def resize_bool_mask(mask, size_hw):
    h, w = size_hw
    img = Image.fromarray(mask.astype(np.uint8) * 255)
    img = img.resize((w, h), Image.NEAREST)
    return np.array(img) > 0


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

                for ny, nx in [(cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)]:
                    if ny < 0 or ny >= h or nx < 0 or nx >= w:
                        continue
                    if visited[ny, nx] or not binary_mask[ny, nx]:
                        continue
                    visited[ny, nx] = True
                    q.append((ny, nx))

            components.append(pixels)

    return components


def component_to_mask(component, shape):
    mask = np.zeros(shape, dtype=bool)
    ys = np.array([p[0] for p in component])
    xs = np.array([p[1] for p in component])
    mask[ys, xs] = True
    return mask


def component_box(component, shape):
    h, w = shape
    ys = np.array([p[0] for p in component])
    xs = np.array([p[1] for p in component])

    x1 = max(0, int(xs.min()))
    y1 = max(0, int(ys.min()))
    x2 = min(w - 1, int(xs.max()))
    y2 = min(h - 1, int(ys.max()))

    if x2 <= x1:
        x2 = min(w - 1, x1 + 1)
    if y2 <= y1:
        y2 = min(h - 1, y1 + 1)

    return np.array([x1, y1, x2, y2], dtype=np.float32)


def choose_best_component(anomaly_map, threshold, min_area):
    binary = anomaly_map >= threshold
    components = connected_components(binary)

    records = []
    for comp in components:
        if len(comp) < min_area:
            continue

        mask = component_to_mask(comp, anomaly_map.shape)
        area = int(mask.sum())
        mean_score = float(anomaly_map[mask].mean())
        max_score = float(anomaly_map[mask].max())

        records.append(
            {
                "component": comp,
                "mask": mask,
                "area": area,
                "mean_score": mean_score,
                "max_score": max_score,
                "rank_score": mean_score * np.sqrt(area),
            }
        )

    if not records:
        return None

    records = sorted(records, key=lambda r: r["rank_score"], reverse=True)
    return records[0]


def get_prompt_points(anomaly_map, component_mask, num_negative=4):
    h, w = anomaly_map.shape

    masked_scores = np.where(component_mask, anomaly_map, -1)
    pos_y, pos_x = np.unravel_index(np.argmax(masked_scores), masked_scores.shape)

    point_coords = [[float(pos_x), float(pos_y)]]
    point_labels = [1]

    low_mask = (anomaly_map <= np.percentile(anomaly_map, 20)) & (~component_mask)

    quadrants = [
        (slice(0, h // 2), slice(0, w // 2)),
        (slice(0, h // 2), slice(w // 2, w)),
        (slice(h // 2, h), slice(0, w // 2)),
        (slice(h // 2, h), slice(w // 2, w)),
    ]

    for ys, xs in quadrants:
        sub_mask = low_mask[ys, xs]
        if sub_mask.sum() == 0:
            continue

        sub_scores = np.where(sub_mask, anomaly_map[ys, xs], 999)
        ny, nx = np.unravel_index(np.argmin(sub_scores), sub_scores.shape)

        global_y = ny + (0 if ys.start is None else ys.start)
        global_x = nx + (0 if xs.start is None else xs.start)

        point_coords.append([float(global_x), float(global_y)])
        point_labels.append(0)

        if len(point_labels) >= num_negative + 1:
            break

    return np.array(point_coords, dtype=np.float32), np.array(point_labels, dtype=np.int32)


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


def anomaly_consistency_score(mask, anomaly_map, coarse_mask):
    mask = mask > 0
    if mask.sum() == 0:
        return -999.0

    inside_mean = float(anomaly_map[mask].mean())
    outside_mean = float(anomaly_map[~mask].mean()) if (~mask).sum() > 0 else 0.0

    inter = np.logical_and(mask, coarse_mask).sum()
    coarse_area = coarse_mask.sum()
    mask_area = mask.sum()

    coarse_recall = float(inter / coarse_area) if coarse_area > 0 else 0.0
    area_ratio = float(mask_area / mask.size)
    coarse_ratio = float(coarse_area / coarse_mask.size)

    score = inside_mean - outside_mean
    score += 0.30 * coarse_recall
    score -= 0.20 * abs(area_ratio - coarse_ratio)

    return float(score)


def pick_by_sam_score(masks, scores):
    idx = int(np.argmax(scores))
    return masks[idx] > 0, float(scores[idx]), idx


def pick_by_anomaly_consistency(masks, scores, anomaly_map, coarse_mask):
    best_idx = 0
    best_score = -999.0

    for idx, mask in enumerate(masks):
        score = anomaly_consistency_score(mask, anomaly_map, coarse_mask)
        if score > best_score:
            best_score = score
            best_idx = idx

    return masks[best_idx] > 0, float(scores[best_idx]), best_idx, float(best_score)


def save_visual(image, gt_mask, anomaly_map, coarse_mask, sam_mask, box, points, labels, out_path, title):
    fig = plt.figure(figsize=(18, 4))

    ax1 = fig.add_subplot(1, 5, 1)
    ax1.imshow(image)
    ax1.set_title("Image")
    ax1.axis("off")

    ax2 = fig.add_subplot(1, 5, 2)
    ax2.imshow(gt_mask, cmap="gray")
    ax2.set_title("GT Mask")
    ax2.axis("off")

    ax3 = fig.add_subplot(1, 5, 3)
    ax3.imshow(anomaly_map)
    x1, y1, x2, y2 = box
    ax3.add_patch(plt.Rectangle((x1, y1), x2 - x1 + 1, y2 - y1 + 1, fill=False, linewidth=2))
    ax3.set_title("Anomaly Map + Box")
    ax3.axis("off")

    ax4 = fig.add_subplot(1, 5, 4)
    ax4.imshow(image)
    ax4.imshow(coarse_mask, alpha=0.45)
    ax4.set_title("PatchCore Component")
    ax4.axis("off")

    ax5 = fig.add_subplot(1, 5, 5)
    ax5.imshow(image)
    ax5.imshow(sam_mask, alpha=0.45)

    if points is not None:
        for (x, y), label in zip(points, labels):
            marker = "o" if label == 1 else "x"
            ax5.scatter([x], [y], marker=marker, s=35)

    ax5.set_title("SAM2 Selected Mask")
    ax5.axis("off")

    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def load_thresholds(path):
    df = pd.read_csv(path)
    return {str(row["category"]): float(row["best_threshold"]) for _, row in df.iterrows()}


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


def run_category(args, predictor, category, threshold):
    predictions = collect_predictions(args, category)

    out_dir = Path(args.output_root) / "MVTecAD" / category / "anomaly_aware_sam2"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    sample_idx = 0

    for batch in predictions:
        images = get_field(batch, "image")
        image_paths = get_field(batch, "image_path")
        gt_masks = get_field(batch, "gt_mask")
        anomaly_maps = get_field(batch, "anomaly_map")

        batch_size = len(image_paths) if isinstance(image_paths, (list, tuple)) else 1

        for i in range(batch_size):
            image = image_tensor_to_uint8(take_item(images, i))
            gt_mask = mask_to_2d(take_item(gt_masks, i))
            anomaly_map = normalize_map(take_item(anomaly_maps, i))
            image_path = str(take_item(image_paths, i))

            if image is None or gt_mask is None or anomaly_map is None:
                continue

            h, w = image.shape[:2]

            gt_mask = gt_mask > 0
            if gt_mask.shape != (h, w):
                gt_mask = resize_bool_mask(gt_mask, (h, w))

            if anomaly_map.shape != (h, w):
                anomaly_map = resize_float_map(anomaly_map, (h, w))
                anomaly_map = normalize_map(anomaly_map)

            if gt_mask.sum() == 0:
                continue

            best_component = choose_best_component(anomaly_map, threshold, args.min_area)
            if best_component is None:
                continue

            coarse_mask = best_component["mask"]
            box = component_box(best_component["component"], anomaly_map.shape)
            point_coords, point_labels = get_prompt_points(anomaly_map, coarse_mask, num_negative=args.num_negative_points)

            predictor.set_image(image)

            box_masks, box_scores, _ = predictor.predict(
                point_coords=None,
                point_labels=None,
                box=box,
                multimask_output=True,
            )

            bp_masks, bp_scores, _ = predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                box=box,
                multimask_output=True,
            )

            sam_score_mask, sam_score_value, sam_score_idx = pick_by_sam_score(box_masks, box_scores)

            anomaly_box_mask, anomaly_box_sam_score, anomaly_box_idx, anomaly_box_consistency = pick_by_anomaly_consistency(
                box_masks, box_scores, anomaly_map, coarse_mask
            )

            anomaly_bp_mask, anomaly_bp_sam_score, anomaly_bp_idx, anomaly_bp_consistency = pick_by_anomaly_consistency(
                bp_masks, bp_scores, anomaly_map, coarse_mask
            )

            patchcore_iou, patchcore_f1 = compute_iou_f1(coarse_mask, gt_mask)
            sam_score_iou, sam_score_f1 = compute_iou_f1(sam_score_mask, gt_mask)
            anomaly_box_iou, anomaly_box_f1 = compute_iou_f1(anomaly_box_mask, gt_mask)
            anomaly_bp_iou, anomaly_bp_f1 = compute_iou_f1(anomaly_bp_mask, gt_mask)

            sample_idx += 1
            visual_name = f"{category}_{sample_idx:04d}_anomaly_aware_sam2.png"
            visual_path = out_dir / visual_name

            title = (
                f"{category} | PatchCore F1={patchcore_f1:.4f} | "
                f"AA-SAM2 F1={anomaly_bp_f1:.4f}"
            )

            save_visual(
                image=image,
                gt_mask=gt_mask,
                anomaly_map=anomaly_map,
                coarse_mask=coarse_mask,
                sam_mask=anomaly_bp_mask,
                box=box,
                points=point_coords,
                labels=point_labels,
                out_path=visual_path,
                title=title,
            )

            rows.append(
                {
                    "category": category,
                    "image_path": image_path,
                    "threshold": threshold,
                    "patchcore_component_iou": patchcore_iou,
                    "patchcore_component_f1": patchcore_f1,
                    "sam2_box_sam_score_iou": sam_score_iou,
                    "sam2_box_sam_score_f1": sam_score_f1,
                    "sam2_box_anomaly_select_iou": anomaly_box_iou,
                    "sam2_box_anomaly_select_f1": anomaly_box_f1,
                    "sam2_box_point_anomaly_select_iou": anomaly_bp_iou,
                    "sam2_box_point_anomaly_select_f1": anomaly_bp_f1,
                    "sam2_box_sam_score": sam_score_value,
                    "sam2_box_anomaly_sam_score": anomaly_box_sam_score,
                    "sam2_box_point_anomaly_sam_score": anomaly_bp_sam_score,
                    "sam2_box_anomaly_consistency": anomaly_box_consistency,
                    "sam2_box_point_anomaly_consistency": anomaly_bp_consistency,
                    "visual_path": str(visual_path),
                }
            )

            if args.max_images > 0 and sample_idx >= args.max_images:
                break

        if args.max_images > 0 and sample_idx >= args.max_images:
            break

    out_csv = out_dir / "anomaly_aware_sam2_results.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "category",
            "image_path",
            "threshold",
            "patchcore_component_iou",
            "patchcore_component_f1",
            "sam2_box_sam_score_iou",
            "sam2_box_sam_score_f1",
            "sam2_box_anomaly_select_iou",
            "sam2_box_anomaly_select_f1",
            "sam2_box_point_anomaly_select_iou",
            "sam2_box_point_anomaly_select_f1",
            "sam2_box_sam_score",
            "sam2_box_anomaly_sam_score",
            "sam2_box_point_anomaly_sam_score",
            "sam2_box_anomaly_consistency",
            "sam2_box_point_anomaly_consistency",
            "visual_path",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    if rows:
        df = pd.DataFrame(rows)
        summary = {
            "category": category,
            "num_images": len(df),
            "patchcore_component_f1_mean": df["patchcore_component_f1"].mean(),
            "sam2_box_sam_score_f1_mean": df["sam2_box_sam_score_f1"].mean(),
            "sam2_box_anomaly_select_f1_mean": df["sam2_box_anomaly_select_f1"].mean(),
            "sam2_box_point_anomaly_select_f1_mean": df["sam2_box_point_anomaly_select_f1"].mean(),
            "patchcore_component_iou_mean": df["patchcore_component_iou"].mean(),
            "sam2_box_sam_score_iou_mean": df["sam2_box_sam_score_iou"].mean(),
            "sam2_box_anomaly_select_iou_mean": df["sam2_box_anomaly_select_iou"].mean(),
            "sam2_box_point_anomaly_select_iou_mean": df["sam2_box_point_anomaly_select_iou"].mean(),
            "result_csv": str(out_csv),
        }
    else:
        summary = {
            "category": category,
            "num_images": 0,
            "patchcore_component_f1_mean": 0.0,
            "sam2_box_sam_score_f1_mean": 0.0,
            "sam2_box_anomaly_select_f1_mean": 0.0,
            "sam2_box_point_anomaly_select_f1_mean": 0.0,
            "patchcore_component_iou_mean": 0.0,
            "sam2_box_sam_score_iou_mean": 0.0,
            "sam2_box_anomaly_select_iou_mean": 0.0,
            "sam2_box_point_anomaly_select_iou_mean": 0.0,
            "result_csv": str(out_csv),
        }

    print(
        f"[DONE] {category}: "
        f"PatchCore={summary['patchcore_component_f1_mean']:.4f}, "
        f"SAM2-score={summary['sam2_box_sam_score_f1_mean']:.4f}, "
        f"SAM2-AA-box={summary['sam2_box_anomaly_select_f1_mean']:.4f}, "
        f"SAM2-AA-box-point={summary['sam2_box_point_anomaly_select_f1_mean']:.4f}"
    )

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="datasets/MVTecAD")
    parser.add_argument("--sam2_root", type=str, default="third_party/sam2")
    parser.add_argument("--checkpoint", type=str, default="third_party/sam2/checkpoints/sam2.1_hiera_tiny.pt")
    parser.add_argument("--model_cfg", type=str, default="configs/sam2.1/sam2.1_hiera_t.yaml")
    parser.add_argument("--threshold_csv", type=str, default="results/analysis/patchcore_threshold_diagnosis/threshold_diagnosis_summary.csv")
    parser.add_argument("--categories", nargs="+", default=["grid"])
    parser.add_argument("--output_root", type=str, default="results/analysis/sam2_anomaly_aware_selection")
    parser.add_argument("--work_root", type=str, default="runs/analysis/sam2_anomaly_aware_selection")
    parser.add_argument("--min_area", type=int, default=20)
    parser.add_argument("--num_negative_points", type=int, default=4)
    parser.add_argument("--max_images", type=int, default=8)
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.sam2_root = str((PROJECT_ROOT / args.sam2_root).resolve())
    args.checkpoint = str((PROJECT_ROOT / args.checkpoint).resolve())

    thresholds = load_thresholds(args.threshold_csv)
    predictor = load_sam2_predictor(args)

    summaries = []

    for category in args.categories:
        if category not in thresholds:
            raise KeyError(f"Category {category} not found in threshold CSV: {args.threshold_csv}")

        summaries.append(run_category(args, predictor, category, thresholds[category]))

    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    summary_csv = out_root / "sam2_anomaly_aware_selection_summary.csv"
    pd.DataFrame(summaries).to_csv(summary_csv, index=False)

    print(f"[DONE] Saved summary to: {summary_csv}")


if __name__ == "__main__":
    main()
