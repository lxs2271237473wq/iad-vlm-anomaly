import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def add_sam2_to_path(sam2_root):
    sam2_root = Path(sam2_root).resolve()
    if str(sam2_root) not in sys.path:
        sys.path.insert(0, str(sam2_root))


def load_sam2_predictor(args):
    add_sam2_to_path(args.sam2_root)

    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = build_sam2(
        args.model_cfg,
        args.checkpoint,
        device=device,
    )
    predictor = SAM2ImagePredictor(model)

    print(f"[INFO] Loaded SAM2 on device: {device}")
    print(f"[INFO] Checkpoint: {args.checkpoint}")

    return predictor


def read_image_rgb(path, image_size):
    image = Image.open(path).convert("RGB")
    image = image.resize((image_size, image_size), Image.BILINEAR)
    return np.array(image)


def read_gt_mask(mask_path, image_size):
    mask = Image.open(mask_path).convert("L")
    mask = mask.resize((image_size, image_size), Image.NEAREST)
    mask = np.array(mask)
    return mask > 0


def derive_gt_mask_path(image_path):
    image_path = Path(image_path)

    if not image_path.is_absolute():
        image_path = PROJECT_ROOT / image_path

    parts = list(image_path.parts)

    if "test" not in parts:
        return None

    test_idx = parts.index("test")
    defect_type = parts[test_idx + 1]

    if defect_type == "good":
        return None

    category_root = Path(*parts[:test_idx])
    mask_name = image_path.stem + "_mask" + image_path.suffix
    mask_path = category_root / "ground_truth" / defect_type / mask_name

    return mask_path


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


def clamp_box(box, image_size):
    x1, y1, x2, y2 = [float(v) for v in box]

    x1 = max(0, min(image_size - 1, x1))
    y1 = max(0, min(image_size - 1, y1))
    x2 = max(0, min(image_size - 1, x2))
    y2 = max(0, min(image_size - 1, y2))

    if x2 <= x1:
        x2 = min(image_size - 1, x1 + 1)
    if y2 <= y1:
        y2 = min(image_size - 1, y1 + 1)

    return np.array([x1, y1, x2, y2], dtype=np.float32)


def save_visual(image, gt_mask, sam_mask, box, out_path, title):
    fig = plt.figure(figsize=(16, 4))

    ax1 = fig.add_subplot(1, 4, 1)
    ax1.imshow(image)
    ax1.set_title("Image")
    ax1.axis("off")

    ax2 = fig.add_subplot(1, 4, 2)
    ax2.imshow(gt_mask, cmap="gray")
    ax2.set_title("GT Mask")
    ax2.axis("off")

    ax3 = fig.add_subplot(1, 4, 3)
    ax3.imshow(image)
    x1, y1, x2, y2 = box
    ax3.add_patch(
        plt.Rectangle(
            (x1, y1),
            x2 - x1 + 1,
            y2 - y1 + 1,
            fill=False,
            linewidth=2,
        )
    )
    ax3.set_title("PatchCore Box Prompt")
    ax3.axis("off")

    ax4 = fig.add_subplot(1, 4, 4)
    ax4.imshow(image)
    ax4.imshow(sam_mask, alpha=0.45)
    ax4.set_title("SAM2 Mask")
    ax4.axis("off")

    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run_category(args, predictor, category):
    candidate_csv = (
        Path(args.candidate_root)
        / "MVTecAD"
        / category
        / "candidate_regions"
        / "candidate_regions.csv"
    )

    if not candidate_csv.exists():
        raise FileNotFoundError(f"Candidate CSV not found: {candidate_csv}")

    df = pd.read_csv(candidate_csv)

    for col in ["component_rank", "x1", "y1", "x2", "y2", "gt_iou", "gt_f1"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[df["component_rank"] == 1].copy()
    df = df.dropna(subset=["x1", "y1", "x2", "y2"])

    if args.max_images > 0:
        df = df.head(args.max_images)

    out_dir = Path(args.output_root) / "MVTecAD" / category / "sam2_box_prompt"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    for idx, row in df.iterrows():
        image_path = Path(row["image_path"])
        if not image_path.is_absolute():
            image_path = PROJECT_ROOT / image_path

        gt_mask_path = derive_gt_mask_path(image_path)

        if gt_mask_path is None or not gt_mask_path.exists():
            print(f"[WARN] Missing GT mask for image: {image_path}")
            continue

        image = read_image_rgb(image_path, args.image_size)
        gt_mask = read_gt_mask(gt_mask_path, args.image_size)

        box = clamp_box(
            [row["x1"], row["y1"], row["x2"], row["y2"]],
            args.image_size,
        )

        predictor.set_image(image)

        masks, scores, logits = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=box,
            multimask_output=True,
        )

        best_idx = int(np.argmax(scores))
        sam_mask = masks[best_idx] > 0
        sam_score = float(scores[best_idx])

        sam_iou, sam_f1 = compute_iou_f1(sam_mask, gt_mask)

        visual_name = f"{category}_{len(rows)+1:04d}_sam2_box.png"
        visual_path = out_dir / visual_name

        title = (
            f"{category} | SAM2 box prompt | "
            f"sam_f1={sam_f1:.4f} | sam_iou={sam_iou:.4f}"
        )

        save_visual(image, gt_mask, sam_mask, box, visual_path, title)

        rows.append(
            {
                "category": category,
                "image_path": str(image_path),
                "gt_mask_path": str(gt_mask_path),
                "box_x1": float(box[0]),
                "box_y1": float(box[1]),
                "box_x2": float(box[2]),
                "box_y2": float(box[3]),
                "patchcore_top1_gt_iou": float(row["gt_iou"]) if not pd.isna(row["gt_iou"]) else "",
                "patchcore_top1_gt_f1": float(row["gt_f1"]) if not pd.isna(row["gt_f1"]) else "",
                "sam2_score": sam_score,
                "sam2_gt_iou": sam_iou,
                "sam2_gt_f1": sam_f1,
                "visual_path": str(visual_path),
            }
        )

    out_csv = out_dir / "sam2_box_prompt_results.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "category",
            "image_path",
            "gt_mask_path",
            "box_x1",
            "box_y1",
            "box_x2",
            "box_y2",
            "patchcore_top1_gt_iou",
            "patchcore_top1_gt_f1",
            "sam2_score",
            "sam2_gt_iou",
            "sam2_gt_f1",
            "visual_path",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    if len(rows) == 0:
        summary = {
            "category": category,
            "num_images": 0,
            "patchcore_top1_gt_iou_mean": 0.0,
            "patchcore_top1_gt_f1_mean": 0.0,
            "sam2_gt_iou_mean": 0.0,
            "sam2_gt_f1_mean": 0.0,
            "result_csv": str(out_csv),
        }
    else:
        out_df = pd.DataFrame(rows)
        summary = {
            "category": category,
            "num_images": len(rows),
            "patchcore_top1_gt_iou_mean": pd.to_numeric(out_df["patchcore_top1_gt_iou"], errors="coerce").mean(),
            "patchcore_top1_gt_f1_mean": pd.to_numeric(out_df["patchcore_top1_gt_f1"], errors="coerce").mean(),
            "sam2_gt_iou_mean": pd.to_numeric(out_df["sam2_gt_iou"], errors="coerce").mean(),
            "sam2_gt_f1_mean": pd.to_numeric(out_df["sam2_gt_f1"], errors="coerce").mean(),
            "result_csv": str(out_csv),
        }

    print(
        f"[DONE] {category}: "
        f"PatchCore top1 F1={summary['patchcore_top1_gt_f1_mean']:.4f}, "
        f"SAM2 F1={summary['sam2_gt_f1_mean']:.4f}"
    )

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sam2_root", type=str, default="third_party/sam2")
    parser.add_argument("--checkpoint", type=str, default="third_party/sam2/checkpoints/sam2.1_hiera_tiny.pt")
    parser.add_argument("--model_cfg", type=str, default="configs/sam2.1/sam2.1_hiera_t.yaml")
    parser.add_argument("--candidate_root", type=str, default="results/analysis/patchcore_candidate_regions")
    parser.add_argument("--categories", nargs="+", default=["grid"])
    parser.add_argument("--output_root", type=str, default="results/analysis/sam2_prompt_feasibility")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--max_images", type=int, default=8)
    args = parser.parse_args()

    args.sam2_root = str((PROJECT_ROOT / args.sam2_root).resolve())
    args.checkpoint = str((PROJECT_ROOT / args.checkpoint).resolve())

    predictor = load_sam2_predictor(args)

    summaries = []
    for category in args.categories:
        summaries.append(run_category(args, predictor, category))

    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    summary_csv = out_root / "sam2_prompt_feasibility_summary.csv"
    pd.DataFrame(summaries).to_csv(summary_csv, index=False)

    print(f"[DONE] Saved summary to: {summary_csv}")


if __name__ == "__main__":
    main()
