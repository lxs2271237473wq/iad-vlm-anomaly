import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def minmax_norm(values):
    values = np.asarray(values, dtype=np.float32)
    if len(values) == 0:
        return values
    vmin = float(np.nanmin(values))
    vmax = float(np.nanmax(values))
    if vmax - vmin < 1e-8:
        return np.zeros_like(values, dtype=np.float32)
    return (values - vmin) / (vmax - vmin)


def safe_numeric(df, cols):
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def build_features(candidate_df, clip_df=None):
    df = candidate_df.copy()

    df = safe_numeric(
        df,
        [
            "component_rank",
            "x1",
            "y1",
            "x2",
            "y2",
            "area",
            "mean_score",
            "max_score",
            "gt_iou",
            "gt_f1",
        ],
    )

    df = df[df["component_rank"] > 0].copy()
    df = df.dropna(subset=["x1", "y1", "x2", "y2", "area", "mean_score", "max_score"])

    if clip_df is not None and len(clip_df) > 0:
        clip = clip_df.copy()
        clip = safe_numeric(
            clip,
            [
                "component_rank",
                "x1",
                "y1",
                "x2",
                "y2",
                "clip_semantic_score",
            ],
        )

        keep_cols = [
            "image_path",
            "component_rank",
            "x1",
            "y1",
            "x2",
            "y2",
            "clip_semantic_score",
        ]

        clip = clip[keep_cols].copy()

        df = df.merge(
            clip,
            on=["image_path", "component_rank", "x1", "y1", "x2", "y2"],
            how="left",
        )
    else:
        df["clip_semantic_score"] = 0.0

    df["clip_semantic_score"] = df["clip_semantic_score"].fillna(0.0)

    width = (df["x2"] - df["x1"] + 1).clip(lower=1)
    height = (df["y2"] - df["y1"] + 1).clip(lower=1)
    bbox_area = width * height

    df["bbox_area"] = bbox_area
    df["fill_ratio"] = (df["area"] / bbox_area).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    df["aspect_ratio"] = (width / height).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    df["aspect_penalty_raw"] = np.abs(np.log(df["aspect_ratio"].clip(lower=1e-6)))
    df["score_gap"] = df["max_score"] - df["mean_score"]

    # Per-category normalization.
    df["mean_score_norm"] = minmax_norm(df["mean_score"].values)
    df["max_score_norm"] = minmax_norm(df["max_score"].values)
    df["score_gap_norm"] = minmax_norm(df["score_gap"].values)
    df["area_norm"] = minmax_norm(np.log1p(df["area"].values))
    df["bbox_area_norm"] = minmax_norm(np.log1p(df["bbox_area"].values))
    df["fill_ratio_norm"] = minmax_norm(df["fill_ratio"].values)
    df["compactness_norm"] = df["fill_ratio_norm"]
    df["aspect_penalty_norm"] = minmax_norm(df["aspect_penalty_raw"].values)
    df["semantic_norm"] = minmax_norm(df["clip_semantic_score"].values)

    # Smaller elongated penalty is better, so reverse it.
    df["aspect_score_norm"] = 1.0 - df["aspect_penalty_norm"]

    return df


def pick_by_component_rank(df, metric_col):
    values = []
    for _, group in df.groupby("image_path"):
        row = group.sort_values("component_rank", ascending=True).iloc[0]
        values.append(row[metric_col])
    return float(np.nanmean(values)) if values else 0.0


def pick_by_score(df, score_col, metric_col):
    values = []
    for _, group in df.groupby("image_path"):
        row = group.sort_values(score_col, ascending=False).iloc[0]
        values.append(row[metric_col])
    return float(np.nanmean(values)) if values else 0.0


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


def generate_weight_grid():
    weights = []

    # Features:
    # mean, max, gap, area, compactness, aspect, semantic
    for mean_w in [0.2, 0.3, 0.4, 0.5, 0.6]:
        for max_w in [0.0, 0.1, 0.2]:
            for gap_w in [0.0, 0.1, 0.2]:
                for area_w in [0.0, 0.1, 0.2]:
                    for compact_w in [0.0, 0.1, 0.2]:
                        for aspect_w in [0.0, 0.05, 0.1]:
                            for semantic_w in [0.0, 0.05, 0.1, 0.2]:
                                total = mean_w + max_w + gap_w + area_w + compact_w + aspect_w + semantic_w
                                if total <= 0:
                                    continue

                                weights.append(
                                    {
                                        "mean_w": mean_w / total,
                                        "max_w": max_w / total,
                                        "gap_w": gap_w / total,
                                        "area_w": area_w / total,
                                        "compact_w": compact_w / total,
                                        "aspect_w": aspect_w / total,
                                        "semantic_w": semantic_w / total,
                                    }
                                )

    return weights


def apply_score(df, w):
    return (
        w["mean_w"] * df["mean_score_norm"]
        + w["max_w"] * df["max_score_norm"]
        + w["gap_w"] * df["score_gap_norm"]
        + w["area_w"] * df["area_norm"]
        + w["compact_w"] * df["compactness_norm"]
        + w["aspect_w"] * df["aspect_score_norm"]
        + w["semantic_w"] * df["semantic_norm"]
    )


def evaluate_category(args, category):
    candidate_csv = (
        Path(args.candidate_root)
        / "MVTecAD"
        / category
        / "candidate_regions"
        / "candidate_regions.csv"
    )

    if not candidate_csv.exists():
        raise FileNotFoundError(f"Missing candidate CSV: {candidate_csv}")

    clip_csv = (
        Path(args.clip_root)
        / "MVTecAD"
        / category
        / "clip_candidate_scores.csv"
    )

    candidate_df = pd.read_csv(candidate_csv)
    clip_df = pd.read_csv(clip_csv) if clip_csv.exists() else None

    df = build_features(candidate_df, clip_df)

    tune_df, eval_df = split_images(df, seed=args.seed)
    weight_grid = generate_weight_grid()

    detail_rows = []

    baseline_tune_f1 = pick_by_component_rank(tune_df, "gt_f1")
    baseline_eval_f1 = pick_by_component_rank(eval_df, "gt_f1")
    baseline_eval_iou = pick_by_component_rank(eval_df, "gt_iou")

    best_weight = None
    best_tune_f1 = -1.0

    for idx, w in enumerate(weight_grid):
        tune_df = tune_df.copy()
        tune_df["region_score"] = apply_score(tune_df, w)

        tune_f1 = pick_by_score(tune_df, "region_score", "gt_f1")

        if tune_f1 > best_tune_f1:
            best_tune_f1 = tune_f1
            best_weight = w.copy()

        detail_row = {
            "category": category,
            "weight_id": idx,
            "tune_patchcore_f1": baseline_tune_f1,
            "tune_rerank_f1": tune_f1,
        }
        detail_row.update(w)
        detail_rows.append(detail_row)

    eval_df = eval_df.copy()
    eval_df["region_score"] = apply_score(eval_df, best_weight)

    eval_rerank_f1 = pick_by_score(eval_df, "region_score", "gt_f1")
    eval_rerank_iou = pick_by_score(eval_df, "region_score", "gt_iou")

    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    category_dir = out_root / "MVTecAD" / category
    category_dir.mkdir(parents=True, exist_ok=True)

    detail_csv = category_dir / "region_scoring_weight_detail.csv"
    pd.DataFrame(detail_rows).to_csv(detail_csv, index=False)

    scored_csv = category_dir / "region_scoring_eval_candidates.csv"
    eval_df.to_csv(scored_csv, index=False)

    summary = {
        "category": category,
        "num_total_images": int(df["image_path"].nunique()),
        "num_tune_images": int(tune_df["image_path"].nunique()),
        "num_eval_images": int(eval_df["image_path"].nunique()),
        "eval_patchcore_top1_iou": baseline_eval_iou,
        "eval_patchcore_top1_f1": baseline_eval_f1,
        "eval_region_score_iou": eval_rerank_iou,
        "eval_region_score_f1": eval_rerank_f1,
        "eval_delta_f1": eval_rerank_f1 - baseline_eval_f1,
        "tune_patchcore_top1_f1": baseline_tune_f1,
        "tune_best_region_score_f1": best_tune_f1,
        "best_mean_w": best_weight["mean_w"],
        "best_max_w": best_weight["max_w"],
        "best_gap_w": best_weight["gap_w"],
        "best_area_w": best_weight["area_w"],
        "best_compact_w": best_weight["compact_w"],
        "best_aspect_w": best_weight["aspect_w"],
        "best_semantic_w": best_weight["semantic_w"],
        "detail_csv": str(detail_csv),
        "scored_eval_csv": str(scored_csv),
    }

    print(
        f"[DONE] {category}: "
        f"eval PatchCore F1={baseline_eval_f1:.4f}, "
        f"eval RegionScore F1={eval_rerank_f1:.4f}, "
        f"delta={summary['eval_delta_f1']:.4f}"
    )

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate_root", type=str, default="results/analysis/patchcore_candidate_regions")
    parser.add_argument("--clip_root", type=str, default="results/analysis/clip_semantic_candidate_scoring")
    parser.add_argument("--categories", nargs="+", default=["grid", "screw", "leather", "wood"])
    parser.add_argument("--output_root", type=str, default="results/analysis/anomaly_region_scoring_baseline")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = []

    for category in args.categories:
        rows.append(evaluate_category(args, category))

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
        "eval_region_score_iou": summary_df["eval_region_score_iou"].mean(),
        "eval_region_score_f1": summary_df["eval_region_score_f1"].mean(),
        "eval_delta_f1": summary_df["eval_delta_f1"].mean(),
        "tune_patchcore_top1_f1": summary_df["tune_patchcore_top1_f1"].mean(),
        "tune_best_region_score_f1": summary_df["tune_best_region_score_f1"].mean(),
        "best_mean_w": summary_df["best_mean_w"].mean(),
        "best_max_w": summary_df["best_max_w"].mean(),
        "best_gap_w": summary_df["best_gap_w"].mean(),
        "best_area_w": summary_df["best_area_w"].mean(),
        "best_compact_w": summary_df["best_compact_w"].mean(),
        "best_aspect_w": summary_df["best_aspect_w"].mean(),
        "best_semantic_w": summary_df["best_semantic_w"].mean(),
        "detail_csv": "",
        "scored_eval_csv": "",
    }

    summary_df = pd.concat([summary_df, pd.DataFrame([mean_row])], ignore_index=True)

    summary_csv = out_root / "anomaly_region_scoring_baseline_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    print(summary_df.to_string(index=False))
    print(f"[DONE] Saved summary to: {summary_csv}")


if __name__ == "__main__":
    main()
