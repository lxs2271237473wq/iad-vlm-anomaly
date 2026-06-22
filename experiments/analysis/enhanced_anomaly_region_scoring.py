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
    connected_components,
    component_to_record,
)


def minmax_norm(values):
    values = np.asarray(values, dtype=np.float32)
    if len(values) == 0:
        return values

    vmin = float(np.nanmin(values))
    vmax = float(np.nanmax(values))

    if vmax - vmin < 1e-8:
        return np.zeros_like(values, dtype=np.float32)

    return (values - vmin) / (vmax - vmin)


def load_thresholds(path):
    df = pd.read_csv(path)
    return {str(row["category"]): float(row["best_threshold"]) for _, row in df.iterrows()}


def threshold_list(base_threshold):
    values = [
        base_threshold - 0.15,
        base_threshold - 0.10,
        base_threshold - 0.05,
        base_threshold,
        base_threshold + 0.05,
        base_threshold + 0.10,
        base_threshold + 0.15,
    ]
    values = [min(0.95, max(0.05, float(v))) for v in values]
    return sorted(list(dict.fromkeys(values)))


def bbox_geometry(record, shape):
    h, w = shape

    x1, y1 = int(record["x1"]), int(record["y1"])
    x2, y2 = int(record["x2"]), int(record["y2"])

    width = max(1, x2 - x1 + 1)
    height = max(1, y2 - y1 + 1)
    bbox_area = width * height

    area = max(1, int(record["area"]))
    fill_ratio = float(area / bbox_area)

    aspect_ratio = float(width / height)
    aspect_penalty = abs(np.log(max(aspect_ratio, 1e-6)))
    aspect_score = 1.0 / (1.0 + aspect_penalty)

    relative_area = float(area / (h * w))

    return {
        "bbox_width": width,
        "bbox_height": height,
        "bbox_area": bbox_area,
        "fill_ratio": fill_ratio,
        "aspect_ratio": aspect_ratio,
        "aspect_score": aspect_score,
        "relative_area": relative_area,
    }


def local_contrast_features(anomaly_map, component_mask, record, pad=8):
    h, w = anomaly_map.shape

    x1, y1 = int(record["x1"]), int(record["y1"])
    x2, y2 = int(record["x2"]), int(record["y2"])

    ex1 = max(0, x1 - pad)
    ey1 = max(0, y1 - pad)
    ex2 = min(w - 1, x2 + pad)
    ey2 = min(h - 1, y2 + pad)

    local_box = np.zeros_like(component_mask, dtype=bool)
    local_box[ey1:ey2 + 1, ex1:ex2 + 1] = True

    ring_mask = local_box & (~component_mask)

    inside_mean = float(anomaly_map[component_mask].mean()) if component_mask.sum() > 0 else 0.0
    inside_max = float(anomaly_map[component_mask].max()) if component_mask.sum() > 0 else 0.0
    inside_std = float(anomaly_map[component_mask].std()) if component_mask.sum() > 0 else 0.0

    if ring_mask.sum() > 0:
        ring_mean = float(anomaly_map[ring_mask].mean())
        ring_max = float(anomaly_map[ring_mask].max())
    else:
        ring_mean = 0.0
        ring_max = 0.0

    local_contrast = inside_mean - ring_mean
    peak_contrast = inside_max - ring_mean
    peak_margin = inside_max - ring_max

    return {
        "inside_mean": inside_mean,
        "inside_max": inside_max,
        "inside_std": inside_std,
        "ring_mean": ring_mean,
        "ring_max": ring_max,
        "local_contrast": local_contrast,
        "peak_contrast": peak_contrast,
        "peak_margin": peak_margin,
    }


def multi_threshold_stability(anomaly_map, component_mask, thresholds):
    if component_mask.sum() == 0:
        return 0.0

    values = []
    comp_scores = anomaly_map[component_mask]

    for threshold in thresholds:
        values.append(float((comp_scores >= threshold).mean()))

    return float(np.mean(values))


def safe_float(value, default=0.0):
    try:
        if value is None:
            return default
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def extract_category_candidates(args, category, base_threshold):
    predictions = collect_predictions(args, category)
    thresholds = threshold_list(base_threshold)

    rows = []

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

            binary = anomaly_map >= base_threshold
            components = connected_components(binary)

            component_records = []

            for component in components:
                if len(component) < args.min_area:
                    continue

                record = component_to_record(component, anomaly_map, gt_mask)
                component_mask = record["mask"]

                geo = bbox_geometry(record, anomaly_map.shape)
                contrast = local_contrast_features(
                    anomaly_map=anomaly_map,
                    component_mask=component_mask,
                    record=record,
                    pad=args.local_pad,
                )
                stability = multi_threshold_stability(
                    anomaly_map=anomaly_map,
                    component_mask=component_mask,
                    thresholds=thresholds,
                )

                row = {
                    "category": category,
                    "image_path": image_path,
                    "threshold": base_threshold,
                    "x1": record["x1"],
                    "y1": record["y1"],
                    "x2": record["x2"],
                    "y2": record["y2"],
                    "area": record["area"],
                    "mean_score": record["mean_score"],
                    "max_score": record["max_score"],
                    "gt_iou": safe_float(record["gt_iou"]),
                    "gt_f1": safe_float(record["gt_f1"]),
                    "stability": stability,
                }

                row.update(geo)
                row.update(contrast)
                component_records.append(row)

            component_records = sorted(
                component_records,
                key=lambda r: (r["mean_score"], r["area"]),
                reverse=True,
            )

            for rank, row in enumerate(component_records, start=1):
                row["component_rank"] = rank
                rows.append(row)

    return pd.DataFrame(rows)


def split_images(df, seed=42):
    images = sorted(df["image_path"].unique().tolist())
    rng = np.random.default_rng(seed)
    rng.shuffle(images)

    mid = max(1, len(images) // 2)
    tune_images = set(images[:mid])
    eval_images = set(images[mid:])

    if len(eval_images) == 0:
        eval_images = tune_images

    tune_df = df[df["image_path"].isin(tune_images)].copy()
    eval_df = df[df["image_path"].isin(eval_images)].copy()

    return tune_df, eval_df


def add_normalized_features(df):
    df = df.copy()

    feature_cols = [
        "mean_score",
        "max_score",
        "inside_mean",
        "inside_max",
        "inside_std",
        "local_contrast",
        "peak_contrast",
        "peak_margin",
        "stability",
        "fill_ratio",
        "aspect_score",
        "relative_area",
    ]

    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        df[col + "_norm"] = minmax_norm(df[col].values)

    # Very large regions often become over-segmentation. Prefer moderate areas.
    df["area_penalty_norm"] = 1.0 - df["relative_area_norm"]

    return df


def generate_weight_grid():
    weights = []

    for mean_w in [0.2, 0.3, 0.4, 0.5]:
        for contrast_w in [0.0, 0.1, 0.2, 0.3]:
            for peak_w in [0.0, 0.1, 0.2]:
                for stability_w in [0.0, 0.1, 0.2, 0.3]:
                    for compact_w in [0.0, 0.1, 0.2]:
                        for aspect_w in [0.0, 0.05, 0.1]:
                            for area_penalty_w in [0.0, 0.05, 0.1]:
                                total = (
                                    mean_w
                                    + contrast_w
                                    + peak_w
                                    + stability_w
                                    + compact_w
                                    + aspect_w
                                    + area_penalty_w
                                )

                                if total <= 0:
                                    continue

                                weights.append(
                                    {
                                        "mean_w": mean_w / total,
                                        "contrast_w": contrast_w / total,
                                        "peak_w": peak_w / total,
                                        "stability_w": stability_w / total,
                                        "compact_w": compact_w / total,
                                        "aspect_w": aspect_w / total,
                                        "area_penalty_w": area_penalty_w / total,
                                    }
                                )

    return weights


def apply_score(df, weight):
    return (
        weight["mean_w"] * df["mean_score_norm"]
        + weight["contrast_w"] * df["local_contrast_norm"]
        + weight["peak_w"] * df["peak_contrast_norm"]
        + weight["stability_w"] * df["stability_norm"]
        + weight["compact_w"] * df["fill_ratio_norm"]
        + weight["aspect_w"] * df["aspect_score_norm"]
        + weight["area_penalty_w"] * df["area_penalty_norm"]
    )


def pick_patchcore_top1(df, metric_col):
    values = []

    for _, group in df.groupby("image_path"):
        row = group.sort_values("component_rank", ascending=True).iloc[0]
        values.append(row[metric_col])

    return float(np.nanmean(values)) if values else 0.0


def pick_score_top1(df, score_col, metric_col):
    values = []

    for _, group in df.groupby("image_path"):
        row = group.sort_values(score_col, ascending=False).iloc[0]
        values.append(row[metric_col])

    return float(np.nanmean(values)) if values else 0.0


def evaluate_category(args, category, base_threshold):
    df = extract_category_candidates(args, category, base_threshold)

    if len(df) == 0:
        raise RuntimeError(f"No candidates generated for category: {category}")

    df = add_normalized_features(df)

    out_dir = Path(args.output_root) / "MVTecAD" / category
    out_dir.mkdir(parents=True, exist_ok=True)

    candidate_csv = out_dir / "enhanced_region_candidates.csv"
    df.to_csv(candidate_csv, index=False)

    tune_df, eval_df = split_images(df, seed=args.seed)

    tune_patchcore_f1 = pick_patchcore_top1(tune_df, "gt_f1")
    eval_patchcore_f1 = pick_patchcore_top1(eval_df, "gt_f1")
    eval_patchcore_iou = pick_patchcore_top1(eval_df, "gt_iou")

    best_weight = None
    best_tune_f1 = -1.0
    detail_rows = []

    for weight_id, weight in enumerate(generate_weight_grid()):
        tune_df = tune_df.copy()
        tune_df["enhanced_region_score"] = apply_score(tune_df, weight)

        tune_f1 = pick_score_top1(tune_df, "enhanced_region_score", "gt_f1")

        row = {
            "category": category,
            "weight_id": weight_id,
            "tune_patchcore_f1": tune_patchcore_f1,
            "tune_enhanced_f1": tune_f1,
        }
        row.update(weight)
        detail_rows.append(row)

        if tune_f1 > best_tune_f1:
            best_tune_f1 = tune_f1
            best_weight = weight.copy()

    eval_df = eval_df.copy()
    eval_df["enhanced_region_score"] = apply_score(eval_df, best_weight)

    eval_enhanced_f1 = pick_score_top1(eval_df, "enhanced_region_score", "gt_f1")
    eval_enhanced_iou = pick_score_top1(eval_df, "enhanced_region_score", "gt_iou")

    detail_csv = out_dir / "enhanced_region_weight_detail.csv"
    pd.DataFrame(detail_rows).to_csv(detail_csv, index=False)

    eval_csv = out_dir / "enhanced_region_eval_candidates.csv"
    eval_df.to_csv(eval_csv, index=False)

    summary = {
        "category": category,
        "num_total_images": int(df["image_path"].nunique()),
        "num_tune_images": int(tune_df["image_path"].nunique()),
        "num_eval_images": int(eval_df["image_path"].nunique()),
        "eval_patchcore_top1_iou": eval_patchcore_iou,
        "eval_patchcore_top1_f1": eval_patchcore_f1,
        "eval_enhanced_region_iou": eval_enhanced_iou,
        "eval_enhanced_region_f1": eval_enhanced_f1,
        "eval_delta_f1": eval_enhanced_f1 - eval_patchcore_f1,
        "tune_patchcore_f1": tune_patchcore_f1,
        "tune_best_enhanced_f1": best_tune_f1,
        "best_mean_w": best_weight["mean_w"],
        "best_contrast_w": best_weight["contrast_w"],
        "best_peak_w": best_weight["peak_w"],
        "best_stability_w": best_weight["stability_w"],
        "best_compact_w": best_weight["compact_w"],
        "best_aspect_w": best_weight["aspect_w"],
        "best_area_penalty_w": best_weight["area_penalty_w"],
        "candidate_csv": str(candidate_csv),
        "detail_csv": str(detail_csv),
        "eval_csv": str(eval_csv),
    }

    print(
        f"[DONE] {category}: "
        f"PatchCore F1={eval_patchcore_f1:.4f}, "
        f"Enhanced F1={eval_enhanced_f1:.4f}, "
        f"delta={summary['eval_delta_f1']:.4f}"
    )

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="datasets/MVTecAD")
    parser.add_argument("--threshold_csv", type=str, default="results/analysis/patchcore_threshold_diagnosis/threshold_diagnosis_summary.csv")
    parser.add_argument("--categories", nargs="+", default=["grid", "screw", "leather", "wood"])
    parser.add_argument("--output_root", type=str, default="results/analysis/enhanced_anomaly_region_scoring")
    parser.add_argument("--work_root", type=str, default="runs/analysis/enhanced_anomaly_region_scoring")
    parser.add_argument("--min_area", type=int, default=20)
    parser.add_argument("--local_pad", type=int, default=8)
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    thresholds = load_thresholds(args.threshold_csv)

    rows = []
    for category in args.categories:
        if category not in thresholds:
            raise KeyError(f"Missing threshold for category: {category}")

        rows.append(evaluate_category(args, category, thresholds[category]))

    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    summary_df = pd.DataFrame(rows)

    mean_row = {
        "category": "MEAN",
        "num_total_images": summary_df["num_total_images"].sum(),
        "num_tune_images": summary_df["num_tune_images"].sum(),
        "num_eval_images": summary_df["num_eval_images"].sum(),
        "eval_patchcore_top1_iou": summary_df["eval_patchcore_top1_iou"].mean(),
        "eval_patchcore_top1_f1": summary_df["eval_patchcore_top1_f1"].mean(),
        "eval_enhanced_region_iou": summary_df["eval_enhanced_region_iou"].mean(),
        "eval_enhanced_region_f1": summary_df["eval_enhanced_region_f1"].mean(),
        "eval_delta_f1": summary_df["eval_delta_f1"].mean(),
        "tune_patchcore_f1": summary_df["tune_patchcore_f1"].mean(),
        "tune_best_enhanced_f1": summary_df["tune_best_enhanced_f1"].mean(),
        "best_mean_w": summary_df["best_mean_w"].mean(),
        "best_contrast_w": summary_df["best_contrast_w"].mean(),
        "best_peak_w": summary_df["best_peak_w"].mean(),
        "best_stability_w": summary_df["best_stability_w"].mean(),
        "best_compact_w": summary_df["best_compact_w"].mean(),
        "best_aspect_w": summary_df["best_aspect_w"].mean(),
        "best_area_penalty_w": summary_df["best_area_penalty_w"].mean(),
        "candidate_csv": "",
        "detail_csv": "",
        "eval_csv": "",
    }

    summary_df = pd.concat([summary_df, pd.DataFrame([mean_row])], ignore_index=True)

    summary_csv = out_root / "enhanced_anomaly_region_scoring_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    print(summary_df.to_string(index=False))
    print(f"[DONE] Saved summary to: {summary_csv}")


if __name__ == "__main__":
    main()
