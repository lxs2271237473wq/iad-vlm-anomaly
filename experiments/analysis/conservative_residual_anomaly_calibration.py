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


class ConservativeResidualCalibrator(nn.Module):
    def __init__(self, max_delta=0.15):
        super().__init__()
        self.max_delta = float(max_delta)

        self.net = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, kernel_size=1),
        )

        # Start from identity mapping: calibrated_map ≈ raw_map.
        final_conv = self.net[-1]
        nn.init.zeros_(final_conv.weight)
        nn.init.zeros_(final_conv.bias)

    def forward(self, raw_map):
        residual = self.max_delta * torch.tanh(self.net(raw_map))
        calibrated = torch.clamp(raw_map + residual, 0.0, 1.0)
        return calibrated, residual


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


def dice_loss_probs(probs, targets, eps=1e-6):
    dims = (1, 2, 3)
    inter = (probs * targets).sum(dim=dims)
    denom = probs.sum(dim=dims) + targets.sum(dim=dims)
    dice = (2.0 * inter + eps) / (denom + eps)
    return 1.0 - dice.mean()


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

            is_anomaly = bool(gt_mask.sum() > 0)

            samples.append(
                {
                    "image_path": image_path,
                    "anomaly_map": anomaly_map.astype(np.float32),
                    "gt_mask": gt_mask.astype(np.float32),
                    "is_anomaly": is_anomaly,
                }
            )

    return samples


def split_samples_stratified(samples, seed=42):
    abnormal = [s for s in samples if s["is_anomaly"]]
    normal = [s for s in samples if not s["is_anomaly"]]

    rng = np.random.default_rng(seed)

    def split_group(group):
        if len(group) == 0:
            return [], []

        idx = np.arange(len(group))
        rng.shuffle(idx)

        mid = max(1, len(idx) // 2)
        train_idx = idx[:mid]
        eval_idx = idx[mid:]

        if len(eval_idx) == 0:
            eval_idx = train_idx

        return [group[i] for i in train_idx], [group[i] for i in eval_idx]

    abnormal_train, abnormal_eval = split_group(abnormal)
    normal_train, normal_eval = split_group(normal)

    train_samples = abnormal_train + normal_train
    eval_samples = abnormal_eval + normal_eval

    rng.shuffle(train_samples)
    rng.shuffle(eval_samples)

    return train_samples, eval_samples


def samples_to_tensor(samples, device):
    maps = np.stack([s["anomaly_map"] for s in samples], axis=0)
    masks = np.stack([s["gt_mask"] for s in samples], axis=0)
    flags = np.array([s["is_anomaly"] for s in samples], dtype=np.float32)

    x = torch.from_numpy(maps[:, None, :, :]).float().to(device)
    y = torch.from_numpy(masks[:, None, :, :]).float().to(device)
    is_anomaly = torch.from_numpy(flags).float().to(device)

    return x, y, is_anomaly


def raw_outputs(samples):
    return [
        {
            "image_path": s["image_path"],
            "score_map": s["anomaly_map"],
            "gt_mask": s["gt_mask"] > 0,
            "is_anomaly": s["is_anomaly"],
        }
        for s in samples
    ]


def predict_model(model, samples, device):
    model.eval()
    x, _, _ = samples_to_tensor(samples, device)

    outputs = []

    with torch.no_grad():
        calibrated, residual = model(x)
        probs = calibrated.detach().cpu().numpy()[:, 0]
        residual_np = residual.detach().cpu().numpy()[:, 0]

    for i, sample in enumerate(samples):
        outputs.append(
            {
                "image_path": sample["image_path"],
                "score_map": probs[i],
                "residual_map": residual_np[i],
                "gt_mask": sample["gt_mask"] > 0,
                "is_anomaly": sample["is_anomaly"],
            }
        )

    return outputs


def evaluate_outputs(outputs, threshold):
    abnormal_rows = []
    normal_rows = []

    abnormal_ious = []
    abnormal_f1s = []
    normal_fp_ratios = []
    normal_pred_areas = []

    for item in outputs:
        score_map = item["score_map"]
        pred = score_map >= threshold
        gt = item["gt_mask"]

        pred_area = int(pred.sum())
        image_area = int(pred.size)
        fp_ratio = float(pred_area / image_area)

        if item["is_anomaly"]:
            iou, f1 = compute_iou_f1(pred, gt)
            abnormal_ious.append(iou)
            abnormal_f1s.append(f1)

            abnormal_rows.append(
                {
                    "image_path": item["image_path"],
                    "threshold": threshold,
                    "iou": iou,
                    "f1": f1,
                    "pred_area": pred_area,
                    "gt_area": int(gt.sum()),
                }
            )
        else:
            normal_fp_ratios.append(fp_ratio)
            normal_pred_areas.append(pred_area)

            normal_rows.append(
                {
                    "image_path": item["image_path"],
                    "threshold": threshold,
                    "normal_pred_area": pred_area,
                    "normal_fp_ratio": fp_ratio,
                }
            )

    return {
        "abnormal_mean_iou": float(np.mean(abnormal_ious)) if abnormal_ious else 0.0,
        "abnormal_mean_f1": float(np.mean(abnormal_f1s)) if abnormal_f1s else 0.0,
        "normal_mean_fp_ratio": float(np.mean(normal_fp_ratios)) if normal_fp_ratios else 0.0,
        "normal_mean_pred_area": float(np.mean(normal_pred_areas)) if normal_pred_areas else 0.0,
        "abnormal_rows": abnormal_rows,
        "normal_rows": normal_rows,
    }


def tune_threshold(outputs, thresholds, normal_penalty=0.25):
    best = None

    for threshold in thresholds:
        result = evaluate_outputs(outputs, threshold)
        objective = result["abnormal_mean_f1"] - normal_penalty * result["normal_mean_fp_ratio"]

        current = {
            "threshold": float(threshold),
            "objective": float(objective),
            "abnormal_mean_iou": result["abnormal_mean_iou"],
            "abnormal_mean_f1": result["abnormal_mean_f1"],
            "normal_mean_fp_ratio": result["normal_mean_fp_ratio"],
        }

        if best is None or current["objective"] > best["objective"]:
            best = current

    return best


def conservative_loss(calibrated, residual, raw, target, is_anomaly, args):
    bce = F.binary_cross_entropy(calibrated, target)

    abnormal_mask = is_anomaly > 0.5
    if abnormal_mask.sum() > 0:
        dice = dice_loss_probs(calibrated[abnormal_mask], target[abnormal_mask])
    else:
        dice = torch.tensor(0.0, device=calibrated.device)

    identity = F.mse_loss(calibrated, raw)
    residual_l1 = residual.abs().mean()

    normal_mask = is_anomaly < 0.5
    if normal_mask.sum() > 0:
        normal_fp = calibrated[normal_mask].mean()
    else:
        normal_fp = torch.tensor(0.0, device=calibrated.device)

    loss = (
        args.bce_weight * bce
        + args.dice_weight * dice
        + args.identity_weight * identity
        + args.residual_weight * residual_l1
        + args.normal_fp_weight * normal_fp
    )

    logs = {
        "loss": float(loss.detach().cpu().item()),
        "bce": float(bce.detach().cpu().item()),
        "dice": float(dice.detach().cpu().item()),
        "identity": float(identity.detach().cpu().item()),
        "residual_l1": float(residual_l1.detach().cpu().item()),
        "normal_fp": float(normal_fp.detach().cpu().item()),
    }

    return loss, logs


def train_category(args, category):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    samples = collect_maps(args, category)
    train_samples, eval_samples = split_samples_stratified(samples, seed=args.seed)

    train_x, train_y, train_is_anomaly = samples_to_tensor(train_samples, device)

    model = ConservativeResidualCalibrator(max_delta=args.max_delta).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    log_rows = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        calibrated, residual = model(train_x)
        loss, logs = conservative_loss(
            calibrated=calibrated,
            residual=residual,
            raw=train_x,
            target=train_y,
            is_anomaly=train_is_anomaly,
            args=args,
        )

        loss.backward()
        optimizer.step()

        if epoch == 1 or epoch % args.log_interval == 0 or epoch == args.epochs:
            row = {
                "category": category,
                "epoch": epoch,
            }
            row.update(logs)
            log_rows.append(row)

            print(
                f"[TRAIN] {category} epoch={epoch}/{args.epochs} "
                f"loss={logs['loss']:.6f} "
                f"identity={logs['identity']:.6f} "
                f"normal_fp={logs['normal_fp']:.6f}"
            )

    thresholds = np.linspace(0.05, 0.95, 19)

    raw_train_outputs = raw_outputs(train_samples)
    raw_eval_outputs = raw_outputs(eval_samples)

    cal_train_outputs = predict_model(model, train_samples, device)
    cal_eval_outputs = predict_model(model, eval_samples, device)

    raw_best = tune_threshold(raw_train_outputs, thresholds, normal_penalty=args.threshold_normal_penalty)
    cal_best = tune_threshold(cal_train_outputs, thresholds, normal_penalty=args.threshold_normal_penalty)

    raw_eval = evaluate_outputs(raw_eval_outputs, raw_best["threshold"])
    cal_eval = evaluate_outputs(cal_eval_outputs, cal_best["threshold"])

    out_dir = Path(args.output_root) / "MVTecAD" / category
    out_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(log_rows).to_csv(out_dir / "training_log.csv", index=False)
    pd.DataFrame(raw_eval["abnormal_rows"]).to_csv(out_dir / "raw_abnormal_eval_metrics.csv", index=False)
    pd.DataFrame(cal_eval["abnormal_rows"]).to_csv(out_dir / "calibrated_abnormal_eval_metrics.csv", index=False)
    pd.DataFrame(raw_eval["normal_rows"]).to_csv(out_dir / "raw_normal_eval_metrics.csv", index=False)
    pd.DataFrame(cal_eval["normal_rows"]).to_csv(out_dir / "calibrated_normal_eval_metrics.csv", index=False)

    weight_dir = Path(args.work_root) / "MVTecAD" / category
    weight_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), weight_dir / "conservative_residual_calibrator.pth")

    summary = {
        "category": category,
        "num_samples": len(samples),
        "num_train_samples": len(train_samples),
        "num_eval_samples": len(eval_samples),
        "num_train_abnormal": sum(s["is_anomaly"] for s in train_samples),
        "num_train_normal": sum(not s["is_anomaly"] for s in train_samples),
        "num_eval_abnormal": sum(s["is_anomaly"] for s in eval_samples),
        "num_eval_normal": sum(not s["is_anomaly"] for s in eval_samples),
        "raw_tuned_threshold": raw_best["threshold"],
        "calibrated_tuned_threshold": cal_best["threshold"],
        "eval_raw_abnormal_iou": raw_eval["abnormal_mean_iou"],
        "eval_raw_abnormal_f1": raw_eval["abnormal_mean_f1"],
        "eval_calibrated_abnormal_iou": cal_eval["abnormal_mean_iou"],
        "eval_calibrated_abnormal_f1": cal_eval["abnormal_mean_f1"],
        "eval_delta_abnormal_f1": cal_eval["abnormal_mean_f1"] - raw_eval["abnormal_mean_f1"],
        "eval_raw_normal_fp_ratio": raw_eval["normal_mean_fp_ratio"],
        "eval_calibrated_normal_fp_ratio": cal_eval["normal_mean_fp_ratio"],
        "eval_delta_normal_fp_ratio": cal_eval["normal_mean_fp_ratio"] - raw_eval["normal_mean_fp_ratio"],
        "training_log": str(out_dir / "training_log.csv"),
        "weight_path": str(weight_dir / "conservative_residual_calibrator.pth"),
    }

    print(
        f"[DONE] {category}: "
        f"raw abnormal F1={summary['eval_raw_abnormal_f1']:.4f}, "
        f"calibrated abnormal F1={summary['eval_calibrated_abnormal_f1']:.4f}, "
        f"delta={summary['eval_delta_abnormal_f1']:.4f}, "
        f"normal FP delta={summary['eval_delta_normal_fp_ratio']:.6f}"
    )

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="datasets/MVTecAD")
    parser.add_argument("--categories", nargs="+", default=["grid", "screw", "leather", "wood"])
    parser.add_argument("--output_root", type=str, default="results/analysis/conservative_residual_anomaly_calibration")
    parser.add_argument("--work_root", type=str, default="runs/analysis/conservative_residual_anomaly_calibration")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--max_delta", type=float, default=0.10)
    parser.add_argument("--bce_weight", type=float, default=1.0)
    parser.add_argument("--dice_weight", type=float, default=1.0)
    parser.add_argument("--identity_weight", type=float, default=2.0)
    parser.add_argument("--residual_weight", type=float, default=1.0)
    parser.add_argument("--normal_fp_weight", type=float, default=1.0)
    parser.add_argument("--threshold_normal_penalty", type=float, default=0.25)
    parser.add_argument("--log_interval", type=int, default=20)
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = []

    for category in args.categories:
        rows.append(train_category(args, category))

    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    summary_df = pd.DataFrame(rows)

    mean_row = {
        "category": "MEAN",
        "num_samples": summary_df["num_samples"].sum(),
        "num_train_samples": summary_df["num_train_samples"].sum(),
        "num_eval_samples": summary_df["num_eval_samples"].sum(),
        "num_train_abnormal": summary_df["num_train_abnormal"].sum(),
        "num_train_normal": summary_df["num_train_normal"].sum(),
        "num_eval_abnormal": summary_df["num_eval_abnormal"].sum(),
        "num_eval_normal": summary_df["num_eval_normal"].sum(),
        "raw_tuned_threshold": "",
        "calibrated_tuned_threshold": "",
        "eval_raw_abnormal_iou": summary_df["eval_raw_abnormal_iou"].mean(),
        "eval_raw_abnormal_f1": summary_df["eval_raw_abnormal_f1"].mean(),
        "eval_calibrated_abnormal_iou": summary_df["eval_calibrated_abnormal_iou"].mean(),
        "eval_calibrated_abnormal_f1": summary_df["eval_calibrated_abnormal_f1"].mean(),
        "eval_delta_abnormal_f1": summary_df["eval_delta_abnormal_f1"].mean(),
        "eval_raw_normal_fp_ratio": summary_df["eval_raw_normal_fp_ratio"].mean(),
        "eval_calibrated_normal_fp_ratio": summary_df["eval_calibrated_normal_fp_ratio"].mean(),
        "eval_delta_normal_fp_ratio": summary_df["eval_delta_normal_fp_ratio"].mean(),
        "training_log": "",
        "weight_path": "",
    }

    summary_df = pd.concat([summary_df, pd.DataFrame([mean_row])], ignore_index=True)

    summary_csv = out_root / "conservative_residual_anomaly_calibration_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    print(summary_df.to_string(index=False))
    print(f"[DONE] Saved summary to: {summary_csv}")


if __name__ == "__main__":
    main()
