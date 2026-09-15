import argparse
import csv
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


def pick_top1_mean(df, selector_col, metric_col):
    selected = []
    for _, group in df.groupby("image_path"):
        row = group.sort_values(selector_col, ascending=False).iloc[0]
        selected.append(row[metric_col])
    return float(np.nanmean(selected)) if selected else 0.0


def pick_patchcore_top1_mean(df, metric_col):
    selected = []
    for _, group in df.groupby("image_path"):
        row = group.sort_values("component_rank", ascending=True).iloc[0]
        selected.append(row[metric_col])
    return float(np.nanmean(selected)) if selected else 0.0


def evaluate_category(detail_csv, category, weights):
    df = pd.read_csv(detail_csv)

    for col in [
        "component_rank",
        "mean_score",
        "max_score",
        "area",
        "clip_semantic_score",
        "gt_iou",
        "gt_f1",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[df["component_rank"] > 0].copy()

    df["clip_semantic_score_norm"] = minmax_norm(df["clip_semantic_score"].values)
    df["mean_score_norm"] = minmax_norm(df["mean_score"].values)
    df["max_score_norm"] = minmax_norm(df["max_score"].values)
    df["area_norm"] = minmax_norm(np.log1p(df["area"].values))

    patchcore_f1 = pick_patchcore_top1_mean(df, "gt_f1")
    patchcore_iou = pick_patchcore_top1_mean(df, "gt_iou")

    rows = []

    for semantic_w, anomaly_w, max_w, area_w in weights:
        df["sweep_score"] = (
            semantic_w * df["clip_semantic_score_norm"]
            + anomaly_w * df["mean_score_norm"]
            + max_w * df["max_score_norm"]
            + area_w * df["area_norm"]
        )

        f1 = pick_top1_mean(df, "sweep_score", "gt_f1")
        iou = pick_top1_mean(df, "sweep_score", "gt_iou")

        rows.append(
            {
                "category": category,
                "semantic_weight": semantic_w,
                "anomaly_weight": anomaly_w,
                "max_score_weight": max_w,
                "area_weight": area_w,
                "patchcore_top1_gt_iou": patchcore_iou,
                "patchcore_top1_gt_f1": patchcore_f1,
                "rerank_gt_iou": iou,
                "rerank_gt_f1": f1,
                "delta_f1": f1 - patchcore_f1,
            }
        )

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_root", type=str, default="results/analysis/clip_semantic_candidate_scoring")
    parser.add_argument("--categories", nargs="+", default=["grid", "screw", "leather", "wood"])
    parser.add_argument("--output_root", type=str, default="results/analysis/clip_semantic_weight_sweep")
    args = parser.parse_args()

    weight_grid = []

    # semantic weight from 0 to 0.8; the rest is mostly anomaly-map based.
    for semantic_w in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        remain = 1.0 - semantic_w

        configs = [
            (semantic_w, remain, 0.0, 0.0),
            (semantic_w, remain * 0.8, remain * 0.1, remain * 0.1),
            (semantic_w, remain * 0.6, remain * 0.2, remain * 0.2),
        ]

        for cfg in configs:
            weight_grid.append(cfg)

    all_rows = []

    for category in args.categories:
        detail_csv = (
            Path(args.input_root)
            / "MVTecAD"
            / category
            / "clip_candidate_scores.csv"
        )

        if not detail_csv.exists():
            raise FileNotFoundError(f"Missing detail CSV: {detail_csv}")

        rows = evaluate_category(detail_csv, category, weight_grid)
        all_rows.extend(rows)

    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    detail_out = out_root / "clip_semantic_weight_sweep_detail.csv"
    pd.DataFrame(all_rows).to_csv(detail_out, index=False)

    detail_df = pd.DataFrame(all_rows)

    summary_rows = []

    for category, group in detail_df.groupby("category"):
        best = group.sort_values("rerank_gt_f1", ascending=False).iloc[0]
        baseline = group.iloc[0]["patchcore_top1_gt_f1"]

        summary_rows.append(
            {
                "category": category,
                "patchcore_top1_gt_f1": baseline,
                "best_rerank_gt_f1": best["rerank_gt_f1"],
                "best_delta_f1": best["delta_f1"],
                "best_semantic_weight": best["semantic_weight"],
                "best_anomaly_weight": best["anomaly_weight"],
                "best_max_score_weight": best["max_score_weight"],
                "best_area_weight": best["area_weight"],
            }
        )

    summary_df = pd.DataFrame(summary_rows)

    mean_row = {
        "category": "MEAN",
        "patchcore_top1_gt_f1": summary_df["patchcore_top1_gt_f1"].mean(),
        "best_rerank_gt_f1": summary_df["best_rerank_gt_f1"].mean(),
        "best_delta_f1": summary_df["best_delta_f1"].mean(),
        "best_semantic_weight": summary_df["best_semantic_weight"].mean(),
        "best_anomaly_weight": summary_df["best_anomaly_weight"].mean(),
        "best_max_score_weight": summary_df["best_max_score_weight"].mean(),
        "best_area_weight": summary_df["best_area_weight"].mean(),
    }

    summary_df = pd.concat([summary_df, pd.DataFrame([mean_row])], ignore_index=True)

    summary_out = out_root / "clip_semantic_weight_sweep_summary.csv"
    summary_df.to_csv(summary_out, index=False)

    print(summary_df.to_string(index=False))
    print(f"[DONE] Detail saved to: {detail_out}")
    print(f"[DONE] Summary saved to: {summary_out}")


if __name__ == "__main__":
    main()
