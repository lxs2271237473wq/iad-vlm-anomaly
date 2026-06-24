import argparse
import time
import inspect
import csv
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
import pandas as pd
import torch
from anomalib.models import Patchcore, Fastflow, ReverseDistillation, Stfpm, Padim

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from anomalib.data import Folder
from anomalib.engine import Engine
from anomalib.models import Patchcore

from experiments.stage7_generalization.progress_utils import OneLineProgressCallback

from experiments.analysis.patchcore_candidate_regions import (
    get_field,
    take_item,
    mask_to_2d,
    normalize_map,
    connected_components,
    component_to_record,
)


VISA_CATEGORIES = [
    "candle",
    "capsules",
    "cashew",
    "chewinggum",
    "fryum",
    "macaroni1",
    "macaroni2",
    "pcb1",
    "pcb2",
    "pcb3",
    "pcb4",
    "pipe_fryum",
]

def build_pre_processor_for_model(model_cls, args):
    kwargs = {
        "image_size": (args.image_size, args.image_size),
        "center_crop_size": (args.center_crop_size, args.center_crop_size),
    }

    signature = inspect.signature(model_cls.configure_pre_processor)
    accepted = set(signature.parameters.keys())

    filtered_kwargs = {
        key: value for key, value in kwargs.items()
        if key in accepted
    }

    return model_cls.configure_pre_processor(**filtered_kwargs)

def canonical_path(path):
    s = str(path).replace("\\", "/")
    marker = "datasets/VisA_anomalib_1cls/"
    if marker in s:
        return s[s.index(marker):]
    marker = "datasets/VisA/"
    if marker in s:
        return s[s.index(marker):]
    return s


def infer_label_from_path(path):
    s = str(path).replace("\\", "/").lower()
    if "/test/anomaly/" in s:
        return 1
    if "/test/good/" in s:
        return 0
    if "/train/good/" in s:
        return 0
    return 0


def best_f1_threshold(y_true, y_score, max_thresholds=256):
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)

    if len(np.unique(y_true)) < 2:
        return 0.5, 0.0

    unique_scores = np.unique(y_score)
    if len(unique_scores) > max_thresholds:
        thresholds = np.quantile(y_score, np.linspace(0.0, 1.0, max_thresholds))
        thresholds = np.unique(thresholds)
    else:
        thresholds = unique_scores

    best_thr = float(thresholds[0])
    best_f1 = -1.0

    for thr in thresholds:
        pred = (y_score >= thr).astype(int)
        f1 = f1_score(y_true, pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = float(f1)
            best_thr = float(thr)

    return best_thr, best_f1


def safe_roc_auc(y_true, y_score):
    y_true = np.asarray(y_true).astype(int)
    if len(np.unique(y_true)) < 2:
        return np.nan
    return float(roc_auc_score(y_true, y_score))


def safe_ap(y_true, y_score):
    y_true = np.asarray(y_true).astype(int)
    if len(np.unique(y_true)) < 2:
        return np.nan
    return float(average_precision_score(y_true, y_score))


def extract_score(batch, index, anomaly_map):
    for field in ["pred_score", "pred_scores", "anomaly_score", "image_score", "score"]:
        try:
            values = get_field(batch, field)
            value = take_item(values, index)
            if hasattr(value, "detach"):
                value = value.detach().cpu().item()
            elif hasattr(value, "item"):
                value = value.item()
            return float(value)
        except Exception:
            pass

    if anomaly_map is None:
        return 0.0
    return float(np.max(anomaly_map))


def build_datamodule(args, category):
    category_root = Path(args.data_root) / category

    required = [
        category_root / "train" / "good",
        category_root / "test" / "good",
        category_root / "test" / "anomaly",
        category_root / "ground_truth" / "anomaly",
    ]

    for path in required:
        if not path.exists():
            raise FileNotFoundError(f"Required path missing: {path}")

    datamodule = Folder(
        name=f"VisA_{category}",
        root=str(category_root),
        normal_dir="train/good",
        abnormal_dir=["test/anomaly"],
        normal_test_dir="test/good",
        mask_dir=["ground_truth/anomaly"],
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        val_split_mode="same_as_test",
    )

    return datamodule


def collect_predictions(args, category, category_index=1, total_categories=1):
    datamodule = build_datamodule(args, category)

    if args.backbone_model == "patchcore":
        pre_processor = build_pre_processor_for_model(Patchcore, args)
        model = Patchcore(
            backbone=args.backbone,
            layers=["layer2", "layer3"],
            pre_trained=True,
            coreset_sampling_ratio=args.coreset_sampling_ratio,
            num_neighbors=args.num_neighbors,
            pre_processor=pre_processor,
        )

    elif args.backbone_model == "fastflow":
        pre_processor = build_pre_processor_for_model(Fastflow, args)
        model = Fastflow(
            backbone="resnet18",
            pre_trained=True,
            flow_steps=8,
            conv3x3_only=False,
            hidden_ratio=1.0,
            pre_processor=pre_processor,
        )

    elif args.backbone_model == "reverse_distillation":
        pre_processor = build_pre_processor_for_model(ReverseDistillation, args)
        model = ReverseDistillation(
            backbone="wide_resnet50_2",
            layers=["layer1", "layer2", "layer3"],
            pre_trained=True,
            pre_processor=pre_processor,
        )

    elif args.backbone_model == "stfpm":
        pre_processor = build_pre_processor_for_model(Stfpm, args)
        model = Stfpm(
            backbone="resnet18",
            layers=["layer1", "layer2", "layer3"],
            pre_processor=pre_processor,
        )

    elif args.backbone_model == "padim":
        pre_processor = build_pre_processor_for_model(Padim, args)
        model = Padim(
            backbone="resnet18",
            layers=["layer1", "layer2", "layer3"],
            pre_trained=True,
            pre_processor=pre_processor,
        )

    else:
        raise ValueError(f"Unknown backbone_model: {args.backbone_model}")

    work_dir = Path(args.work_root) / args.backbone_model / category
    work_dir.mkdir(parents=True, exist_ok=True)

    progress_callback = OneLineProgressCallback(category=f"{args.backbone_model}/{category}")

    engine = Engine(
        default_root_dir=str(work_dir),
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        callbacks=[progress_callback],
        max_epochs=args.max_epochs,
        limit_train_batches=args.limit_train_batches,
        limit_predict_batches=args.limit_predict_batches,
    )

    print(f"[INFO] Fitting {args.backbone_model} on VisA category: {category}")
    engine.fit(model=model, datamodule=datamodule)

    print(f"[INFO] Predicting {args.backbone_model} on VisA category: {category}")
    predictions = engine.predict(model=model, datamodule=datamodule)

    return predictions
    datamodule = build_datamodule(args, category)

    

    work_dir = Path(args.work_root) / category
    work_dir.mkdir(parents=True, exist_ok=True)

    progress_callback = OneLineProgressCallback(
        category=f"{args.backbone_model}/{category}",
        category_index=category_index,
        total_categories=total_categories,
        refresh_interval=args.progress_refresh_interval,
        run_start_time=args.run_start_time,
    )

    engine = Engine(
        default_root_dir=str(work_dir),
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        callbacks=[progress_callback],
    )

    print(f"[INFO] Fitting PatchCore on VisA category: {category}")
    engine.fit(model=model, datamodule=datamodule)

    print(f"[INFO] Predicting PatchCore on VisA category: {category}")
    predictions = engine.predict(model=model, datamodule=datamodule)

    return predictions


def evaluate_and_extract_candidates(args, category, predictions):
    image_records = []
    candidate_rows = []

    image_labels = []
    image_scores = []

    pixel_labels_all = []
    pixel_scores_all = []

    per_image_cache = []

    for batch in predictions:
        image_paths = get_field(batch, "image_path")
        gt_masks = get_field(batch, "gt_mask")
        anomaly_maps = get_field(batch, "anomaly_map")

        batch_size = len(image_paths) if isinstance(image_paths, (list, tuple)) else 1

        for i in range(batch_size):
            image_path = str(take_item(image_paths, i))
            y_image = infer_label_from_path(image_path)

            anomaly_map = normalize_map(take_item(anomaly_maps, i))
            gt_mask = mask_to_2d(take_item(gt_masks, i))

            if anomaly_map is None:
                continue

            if gt_mask is None:
                gt_mask_bin = np.zeros_like(anomaly_map, dtype=bool)
            else:
                gt_mask_bin = gt_mask > 0

            score = extract_score(batch, i, anomaly_map)

            image_labels.append(y_image)
            image_scores.append(score)

            pixel_labels_all.append(gt_mask_bin.astype(np.uint8).reshape(-1))
            pixel_scores_all.append(anomaly_map.astype(np.float32).reshape(-1))

            per_image_cache.append(
                {
                    "category": category,
                    "image_path": image_path,
                    "canonical_image_path": canonical_path(image_path),
                    "label": "anomaly" if y_image == 1 else "normal",
                    "is_anomaly": y_image,
                    "image_score": score,
                    "anomaly_map": anomaly_map,
                    "gt_mask": gt_mask_bin,
                }
            )

    image_labels_np = np.asarray(image_labels).astype(int)
    image_scores_np = np.asarray(image_scores).astype(float)

    pixel_labels_np = np.concatenate(pixel_labels_all).astype(int)
    pixel_scores_np = np.concatenate(pixel_scores_all).astype(float)

    image_thr, image_f1 = best_f1_threshold(image_labels_np, image_scores_np)
    pixel_thr, pixel_f1 = best_f1_threshold(pixel_labels_np, pixel_scores_np)

    image_auroc = safe_roc_auc(image_labels_np, image_scores_np)
    image_ap = safe_ap(image_labels_np, image_scores_np)

    pixel_auroc = safe_roc_auc(pixel_labels_np, pixel_scores_np)
    pixel_ap = safe_ap(pixel_labels_np, pixel_scores_np)

    for item in per_image_cache:
        pred_image_label = int(item["image_score"] >= image_thr)

        image_records.append(
            {
                "dataset": "VisA",
                "category": category,
                "image_path": item["image_path"],
                "canonical_image_path": item["canonical_image_path"],
                "label": item["label"],
                "is_anomaly": item["is_anomaly"],
                "image_score": item["image_score"],
                "image_threshold": image_thr,
                "pred_is_anomaly": pred_image_label,
                "image_correct": int(pred_image_label == item["is_anomaly"]),
            }
        )

        if item["is_anomaly"] != 1:
            continue

        binary_mask = item["anomaly_map"] >= pixel_thr
        components = connected_components(binary_mask)

        records = []
        for comp in components:
            if len(comp) < args.min_area:
                continue
            rec = component_to_record(comp, item["anomaly_map"], item["gt_mask"])
            records.append(rec)

        records = sorted(
            records,
            key=lambda r: (r["area"], r["mean_score"]),
            reverse=True,
        )[: args.top_components]

        if not records:
            candidate_rows.append(
                {
                    "dataset": "VisA",
                    "category": category,
                    "image_path": item["image_path"],
                    "canonical_image_path": item["canonical_image_path"],
                    "component_rank": 0,
                    "threshold": pixel_thr,
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
                }
            )
            continue

        for rank, rec in enumerate(records, start=1):
            candidate_rows.append(
                {
                    "dataset": "VisA",
                    "category": category,
                    "image_path": item["image_path"],
                    "canonical_image_path": item["canonical_image_path"],
                    "component_rank": rank,
                    "threshold": pixel_thr,
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
                }
            )

    metric_row = {
        "dataset": "VisA",
        "category": category,
        "num_test_images": len(image_labels_np),
        "num_test_normal": int((image_labels_np == 0).sum()),
        "num_test_anomaly": int((image_labels_np == 1).sum()),
        "image_auroc": image_auroc,
        "image_ap": image_ap,
        "image_best_f1": image_f1,
        "image_best_threshold": image_thr,
        "pixel_auroc": pixel_auroc,
        "pixel_ap": pixel_ap,
        "pixel_best_f1": pixel_f1,
        "pixel_best_threshold": pixel_thr,
    }

    return metric_row, image_records, candidate_rows


def save_category_outputs(args, category, metric_row, image_records, candidate_rows):
    out_root = Path(args.output_root)
    category_root = out_root / "VisA" / category
    category_root.mkdir(parents=True, exist_ok=True)

    image_csv = category_root / f"{args.backbone_model}_image_predictions.csv"
    candidates_csv = category_root / "candidate_regions.csv"

    pd.DataFrame(image_records).to_csv(image_csv, index=False)
    pd.DataFrame(candidate_rows).to_csv(candidates_csv, index=False)

    valid_candidates = pd.DataFrame(candidate_rows)
    if len(valid_candidates):
        valid = valid_candidates[pd.to_numeric(valid_candidates["component_rank"], errors="coerce") > 0]
        covered_images = valid["canonical_image_path"].nunique()
    else:
        covered_images = 0

    coverage_row = {
        "dataset": "VisA",
        "category": category,
        "num_anomaly_images": metric_row["num_test_anomaly"],
        "covered_anomaly_images": int(covered_images),
        "coverage_ratio": covered_images / metric_row["num_test_anomaly"] if metric_row["num_test_anomaly"] else 0.0,
        "num_candidate_rows": int(len(valid_candidates)),
        "candidate_csv": str(candidates_csv),
    }

    print(f"[DONE] Saved image predictions: {image_csv}")
    print(f"[DONE] Saved candidate regions: {candidates_csv}")

    return coverage_row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
    "--backbone_model",
    type=str,
    default="fastflow",
    choices=["patchcore", "fastflow", "reverse_distillation", "stfpm", "padim"],)
    parser.add_argument("--max_epochs", type=int, default=1)
    parser.add_argument("--limit_train_batches", type=float, default=1.0)
    parser.add_argument("--limit_predict_batches", type=float, default=1.0)
    parser.add_argument("--progress_refresh_interval", type=float, default=1.0)
    parser.add_argument("--data_root", type=str, default="datasets/VisA_anomalib_1cls")
    parser.add_argument("--categories", nargs="+", default=VISA_CATEGORIES)
    parser.add_argument("--output_root", type=str, default="results/stage7_generalization/visa_patchcore")
    parser.add_argument("--work_root", type=str, default="runs/stage7_generalization/visa_patchcore")
    parser.add_argument("--backbone", type=str, default="wide_resnet50_2")
    parser.add_argument("--coreset_sampling_ratio", type=float, default=0.1)
    parser.add_argument("--num_neighbors", type=int, default=9)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--center_crop_size", type=int, default=224)
    parser.add_argument("--top_components", type=int, default=3)
    parser.add_argument("--min_area", type=int, default=20)
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.run_start_time = time.time()

    torch.set_float32_matmul_precision("high")

    metric_rows = []
    coverage_rows = []

    for category_index, category in enumerate(args.categories, start=1):
        predictions = collect_predictions(args, category, category_index, len(args.categories))
        metric_row, image_records, candidate_rows = evaluate_and_extract_candidates(
            args=args,
            category=category,
            predictions=predictions,
        )
        coverage_row = save_category_outputs(
            args=args,
            category=category,
            metric_row=metric_row,
            image_records=image_records,
            candidate_rows=candidate_rows,
        )

        metric_rows.append(metric_row)
        coverage_rows.append(coverage_row)

        print("\n========== Category Result ==========")
        print(pd.DataFrame([metric_row]).to_string(index=False))
        print(pd.DataFrame([coverage_row]).to_string(index=False))

    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    metrics_df = pd.DataFrame(metric_rows)
    coverage_df = pd.DataFrame(coverage_rows)

    mean_metrics = {
        "dataset": "VisA",
        "category": "MEAN",
        "num_test_images": int(metrics_df["num_test_images"].sum()),
        "num_test_normal": int(metrics_df["num_test_normal"].sum()),
        "num_test_anomaly": int(metrics_df["num_test_anomaly"].sum()),
        "image_auroc": float(metrics_df["image_auroc"].mean()),
        "image_ap": float(metrics_df["image_ap"].mean()),
        "image_best_f1": float(metrics_df["image_best_f1"].mean()),
        "image_best_threshold": "",
        "pixel_auroc": float(metrics_df["pixel_auroc"].mean()),
        "pixel_ap": float(metrics_df["pixel_ap"].mean()),
        "pixel_best_f1": float(metrics_df["pixel_best_f1"].mean()),
        "pixel_best_threshold": "",
    }

    mean_coverage = {
        "dataset": "VisA",
        "category": "MEAN",
        "num_anomaly_images": int(coverage_df["num_anomaly_images"].sum()),
        "covered_anomaly_images": int(coverage_df["covered_anomaly_images"].sum()),
        "coverage_ratio": float(coverage_df["covered_anomaly_images"].sum() / coverage_df["num_anomaly_images"].sum()),
        "num_candidate_rows": int(coverage_df["num_candidate_rows"].sum()),
        "candidate_csv": "",
    }

    metrics_df = pd.concat([metrics_df, pd.DataFrame([mean_metrics])], ignore_index=True)
    coverage_df = pd.concat([coverage_df, pd.DataFrame([mean_coverage])], ignore_index=True)

    metrics_csv = out_root / f"visa_{args.backbone_model}_baseline_summary.csv"
    coverage_csv = out_root / f"visa_{args.backbone_model}_candidate_coverage_summary.csv"

    metrics_df.to_csv(metrics_csv, index=False)
    coverage_df.to_csv(coverage_csv, index=False)

    print("\n========== VisA Multi-backbone Baseline Summary ==========")
    print(metrics_df.to_string(index=False))

    print("\n========== VisA Multi-backbone Candidate Coverage ==========")
    print(coverage_df.to_string(index=False))

    print(f"\n[DONE] Saved baseline summary to: {metrics_csv}")
    print(f"[DONE] Saved coverage summary to: {coverage_csv}")


if __name__ == "__main__":
    main()
