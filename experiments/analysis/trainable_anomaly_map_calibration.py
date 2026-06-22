import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

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


class TinyAnomalyCalibrator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, kernel_size=1),
        )

    def forward(self, x):
        return self.net(x)


def resize_bool_mask(mask, target_hw):
    h, w = target_hw
    img = Image.fromarray(mask.astype(np.uint8) * 255)
    img = img.resize((w, h), Image.NEAREST)
    return np.array(img) > 0


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


def dice_loss_with_logits(logits, targets, eps=1e-6):
    probs = torch.sigmoid(logits)
    dims = (1, 2, 3)

    inter = (probs * targets).sum(dim=dims)
    denom = probs.sum(dim=dims) + targets.sum(dim=dims)

    dice = (2.0 * inter + eps) / (denom + eps)
    return 1.0 - dice.mean()


def bce_dice_loss(logits, targets):
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    dice = dice_loss_with_logits(logits, targets)
    return bce + dice


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

            if gt_mask.shape != anomaly_map.shape:
                gt_mask = resize_bool_mask(gt_mask, anomaly_map.shape)

            if gt_mask.sum() == 0:
                continue

            samples.append(
                {
                    "image_path": image_path,
                    "anomaly_map": anomaly_map.astype(np.float32),
                    "gt_mask": gt_mask.astype(np.float32),
                }
            )

    return samples


def split_samples(samples, seed=42):
    rng = np.random.default_rng(seed)
    indices = np.arange(len(samples))
    rng.shuffle(indices)

    mid = max(1, len(indices) // 2)
    train_idx = indices[:mid]
    eval_idx = indices[mid:]

    if len(eval_idx) == 0:
        eval_idx = train_idx

    train_samples = [samples[i] for i in train_idx]
    eval_samples = [samples[i] for i in eval_idx]

    return train_samples, eval_samples


def samples_to_tensor(samples, device):
    maps = np.stack([s["anomaly_map"] for s in samples], axis=0)
    masks = np.stack([s["gt_mask"] for s in samples], axis=0)

    x = torch.from_numpy(maps[:, None, :, :]).float().to(device)
    y = torch.from_numpy(masks[:, None, :, :]).float().to(device)

    return x, y


def predict_model(model, samples, device):
    model.eval()
    x, _ = samples_to_tensor(samples, device)

    outputs = []

    with torch.no_grad():
        logits = model(x)
        probs = torch.sigmoid(logits).detach().cpu().numpy()[:, 0]

    for i, sample in enumerate(samples):
        outputs.append(
            {
                "image_path": sample["image_path"],
                "score_map": probs[i],
                "gt_mask": sample["gt_mask"] > 0,
            }
        )

    return outputs


def raw_outputs(samples):
    return [
        {
            "image_path": s["image_path"],
            "score_map": s["anomaly_map"],
            "gt_mask": s["gt_mask"] > 0,
        }
        for s in samples
    ]


def evaluate_outputs(outputs, threshold):
    rows = []
    ious = []
    f1s = []

    for item in outputs:
        pred = item["score_map"] >= threshold
        gt = item["gt_mask"]

        iou, f1 = compute_iou_f1(pred, gt)

        rows.append(
            {
                "image_path": item["image_path"],
                "threshold": threshold,
                "iou": iou,
                "f1": f1,
                "pred_area": int(pred.sum()),
                "gt_area": int(gt.sum()),
            }
        )

        ious.append(iou)
        f1s.append(f1)

    return {
        "mean_iou": float(np.mean(ious)) if ious else 0.0,
        "mean_f1": float(np.mean(f1s)) if f1s else 0.0,
        "rows": rows,
    }


def tune_threshold(outputs, thresholds):
    best = None

    for threshold in thresholds:
        result = evaluate_outputs(outputs, threshold)

        if best is None or result["mean_f1"] > best["mean_f1"]:
            best = {
                "threshold": float(threshold),
                "mean_iou": result["mean_iou"],
                "mean_f1": result["mean_f1"],
            }

    return best


def train_calibrator(args, category):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    samples = collect_maps(args, category)
    train_samples, eval_samples = split_samples(samples, seed=args.seed)

    train_x, train_y = samples_to_tensor(train_samples, device)

    model = TinyAnomalyCalibrator().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    log_rows = []

    for epoch in range(1, args.epochs + 1):
        model.train()

        optimizer.zero_grad(set_to_none=True)
        logits = model(train_x)
        loss = bce_dice_loss(logits, train_y)
        loss.backward()
        optimizer.step()

        if epoch == 1 or epoch % args.log_interval == 0 or epoch == args.epochs:
            log_rows.append(
                {
                    "category": category,
                    "epoch": epoch,
                    "loss": float(loss.detach().cpu().item()),
                }
            )
            print(f"[TRAIN] {category} epoch={epoch}/{args.epochs} loss={float(loss.detach().cpu().item()):.6f}")

    thresholds = np.linspace(0.05, 0.95, 19)

    raw_train_outputs = raw_outputs(train_samples)
    raw_eval_outputs = raw_outputs(eval_samples)

    cal_train_outputs = predict_model(model, train_samples, device)
    cal_eval_outputs = predict_model(model, eval_samples, device)

    raw_best = tune_threshold(raw_train_outputs, thresholds)
    cal_best = tune_threshold(cal_train_outputs, thresholds)

    raw_eval = evaluate_outputs(raw_eval_outputs, raw_best["threshold"])
    cal_eval = evaluate_outputs(cal_eval_outputs, cal_best["threshold"])

    out_dir = Path(args.output_root) / "MVTecAD" / category
    out_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(log_rows).to_csv(out_dir / "training_log.csv", index=False)
    pd.DataFrame(raw_eval["rows"]).to_csv(out_dir / "raw_eval_metrics.csv", index=False)
    pd.DataFrame(cal_eval["rows"]).to_csv(out_dir / "calibrated_eval_metrics.csv", index=False)

    # Save model weights under runs/, not results/, to avoid uploading weights.
    weight_dir = Path(args.work_root) / "MVTecAD" / category
    weight_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), weight_dir / "tiny_calibrator.pth")

    summary = {
        "category": category,
        "num_samples": len(samples),
        "num_train_samples": len(train_samples),
        "num_eval_samples": len(eval_samples),
        "raw_tuned_threshold": raw_best["threshold"],
        "calibrated_tuned_threshold": cal_best["threshold"],
        "eval_raw_iou": raw_eval["mean_iou"],
        "eval_raw_f1": raw_eval["mean_f1"],
        "eval_calibrated_iou": cal_eval["mean_iou"],
        "eval_calibrated_f1": cal_eval["mean_f1"],
        "eval_delta_f1": cal_eval["mean_f1"] - raw_eval["mean_f1"],
        "training_log": str(out_dir / "training_log.csv"),
        "raw_eval_metrics": str(out_dir / "raw_eval_metrics.csv"),
        "calibrated_eval_metrics": str(out_dir / "calibrated_eval_metrics.csv"),
        "weight_path": str(weight_dir / "tiny_calibrator.pth"),
    }

    print(
        f"[DONE] {category}: "
        f"raw F1={summary['eval_raw_f1']:.4f}, "
        f"calibrated F1={summary['eval_calibrated_f1']:.4f}, "
        f"delta={summary['eval_delta_f1']:.4f}"
    )

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="datasets/MVTecAD")
    parser.add_argument("--categories", nargs="+", default=["grid", "screw", "leather", "wood"])
    parser.add_argument("--output_root", type=str, default="results/analysis/trainable_anomaly_map_calibration")
    parser.add_argument("--work_root", type=str, default="runs/analysis/trainable_anomaly_map_calibration")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--log_interval", type=int, default=20)
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = []

    for category in args.categories:
        rows.append(train_calibrator(args, category))

    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    summary_df = pd.DataFrame(rows)

    mean_row = {
        "category": "MEAN",
        "num_samples": summary_df["num_samples"].sum(),
        "num_train_samples": summary_df["num_train_samples"].sum(),
        "num_eval_samples": summary_df["num_eval_samples"].sum(),
        "raw_tuned_threshold": "",
        "calibrated_tuned_threshold": "",
        "eval_raw_iou": summary_df["eval_raw_iou"].mean(),
        "eval_raw_f1": summary_df["eval_raw_f1"].mean(),
        "eval_calibrated_iou": summary_df["eval_calibrated_iou"].mean(),
        "eval_calibrated_f1": summary_df["eval_calibrated_f1"].mean(),
        "eval_delta_f1": summary_df["eval_delta_f1"].mean(),
        "training_log": "",
        "raw_eval_metrics": "",
        "calibrated_eval_metrics": "",
        "weight_path": "",
    }

    summary_df = pd.concat([summary_df, pd.DataFrame([mean_row])], ignore_index=True)

    summary_csv = out_root / "trainable_anomaly_map_calibration_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    print(summary_df.to_string(index=False))
    print(f"[DONE] Saved summary to: {summary_csv}")


if __name__ == "__main__":
    main()
