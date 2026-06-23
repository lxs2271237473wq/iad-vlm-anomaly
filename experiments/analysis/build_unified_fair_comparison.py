from pathlib import Path
import pandas as pd
import numpy as np


OUT_ROOT = Path("results/analysis/final_comparison")
OUT_ROOT.mkdir(parents=True, exist_ok=True)


def read_csv_safe(path):
    path = Path(path)
    if not path.exists():
        print(f"[WARN] missing: {path}")
        return None
    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"[WARN] failed to read {path}: {e}")
        return None


def to_float(value):
    try:
        if pd.isna(value):
            return np.nan
        return float(value)
    except Exception:
        return np.nan


def to_int(value):
    try:
        if pd.isna(value):
            return 0
        return int(float(value))
    except Exception:
        return 0


def add_row(rows, **kwargs):
    base = {
        "stage": "",
        "branch": "",
        "method": "",
        "dataset": "MVTecAD",
        "categories": "grid,screw,leather,wood",
        "setting_type": "",
        "is_upper_bound": False,
        "is_realistic_full_test": False,
        "input_mode": "",
        "candidate_source": "",
        "prompt_strategy": "",
        "baseline_reference": "",
        "num_images_total": "",
        "num_images_used": "",
        "coverage_ratio": "",
        "fallback_count": "",
        "skipped_count": "",
        "top1_accuracy": "",
        "top2_accuracy": "",
        "macro_f1": "",
        "pixel_iou": "",
        "pixel_f1": "",
        "normal_fp_ratio": "",
        "delta_top1_vs_fair_baseline": "",
        "delta_macro_f1_vs_fair_baseline": "",
        "main_gain": "",
        "main_limitation": "",
        "source_file": "",
    }
    base.update(kwargs)
    rows.append(base)


def add_reasoning_rows(rows):
    # Stage 6.1-B: full image prompt reasoning
    path = "results/analysis/defect_type_prompt_reasoning/defect_type_prompt_reasoning_mean_summary.csv"
    df = read_csv_safe(path)
    if df is not None:
        for _, r in df.iterrows():
            add_row(
                rows,
                stage="Stage 6.1-B",
                branch="defect_type_reasoning",
                method=f"{r['strategy']} + full image",
                setting_type="realistic_full_image",
                is_realistic_full_test=True,
                input_mode="full",
                candidate_source="none",
                prompt_strategy=r["strategy"],
                num_images_total=328,
                num_images_used=328,
                coverage_ratio=1.0,
                top1_accuracy=to_float(r["top1_accuracy"]),
                top2_accuracy=to_float(r["top2_accuracy"]),
                macro_f1=to_float(r["macro_f1"]),
                main_gain="establishes full-image CLIP prompt baseline",
                main_limitation="full image is affected by normal background and object context",
                source_file=path,
            )

    # Stage 6.1-C: old partial PatchCore crop
    path = "results/analysis/defect_type_prompt_reasoning_crop/defect_type_prompt_reasoning_crop_mean_summary.csv"
    df = read_csv_safe(path)
    if df is not None:
        for _, r in df.iterrows():
            total = to_int(r.get("num_images", 328))
            fallback = to_int(r.get("fallback_count", 0))
            add_row(
                rows,
                stage="Stage 6.1-C",
                branch="defect_type_reasoning",
                method=f"{r['strategy']} + {r['image_mode']}",
                setting_type="partial_candidate_or_fallback",
                is_realistic_full_test=False,
                input_mode=r["image_mode"],
                candidate_source="old PatchCore candidates",
                prompt_strategy=r["strategy"],
                num_images_total=total,
                num_images_used=total,
                coverage_ratio=(total - fallback) / total if total else "",
                fallback_count=fallback,
                top1_accuracy=to_float(r["top1_accuracy"]),
                top2_accuracy=to_float(r["top2_accuracy"]),
                macro_f1=to_float(r["macro_f1"]),
                main_gain="shows anomaly crop can help but coverage was incomplete",
                main_limitation="old candidate coverage caused many fallback full-image samples",
                source_file=path,
            )

    # Stage 6.1-D: GT crop upper bound
    path = "results/analysis/defect_type_prompt_reasoning_gt_crop/defect_type_prompt_reasoning_gt_crop_mean_summary.csv"
    df = read_csv_safe(path)
    if df is not None:
        for _, r in df.iterrows():
            add_row(
                rows,
                stage="Stage 6.1-D",
                branch="defect_type_reasoning",
                method=f"{r['strategy']} + {r['image_mode']}",
                setting_type="GT_crop_upper_bound" if r["image_mode"] == "gt_crop" else "realistic_full_image",
                is_upper_bound=(r["image_mode"] == "gt_crop"),
                is_realistic_full_test=(r["image_mode"] == "full"),
                input_mode=r["image_mode"],
                candidate_source="GT mask" if r["image_mode"] == "gt_crop" else "none",
                prompt_strategy=r["strategy"],
                num_images_total=328,
                num_images_used=328,
                coverage_ratio=1.0,
                fallback_count=to_int(r.get("fallback_count", 0)),
                top1_accuracy=to_float(r["top1_accuracy"]),
                top2_accuracy=to_float(r["top2_accuracy"]),
                macro_f1=to_float(r["macro_f1"]),
                main_gain="diagnoses the upper bound of accurate defect-region focus",
                main_limitation="uses ground-truth masks and is not a deployable setting",
                source_file=path,
            )

    # Stage 6.2: visual prompt refinement
    path = "results/analysis/defect_type_visual_prompt_refinement/defect_type_visual_prompt_refinement_mean_summary.csv"
    df = read_csv_safe(path)
    if df is not None:
        for _, r in df.iterrows():
            add_row(
                rows,
                stage="Stage 6.2",
                branch="defect_type_reasoning",
                method=f"{r['strategy']} + {r['image_mode']}",
                setting_type="GT_crop_upper_bound" if r["image_mode"] == "gt_crop" else "realistic_full_image",
                is_upper_bound=(r["image_mode"] == "gt_crop"),
                is_realistic_full_test=(r["image_mode"] == "full"),
                input_mode=r["image_mode"],
                candidate_source="GT mask" if r["image_mode"] == "gt_crop" else "none",
                prompt_strategy=r["strategy"],
                num_images_total=328,
                num_images_used=328,
                coverage_ratio=1.0,
                fallback_count=to_int(r.get("fallback_count", 0)),
                top1_accuracy=to_float(r["top1_accuracy"]),
                top2_accuracy=to_float(r["top2_accuracy"]),
                macro_f1=to_float(r["macro_f1"]),
                main_gain="short visual prompts improve defect type recognition",
                main_limitation="GT-crop rows are upper-bound; full-image rows still include background noise",
                source_file=path,
            )

    # Stage 6.3: old partial real anomaly crop
    path = "results/analysis/real_anomaly_crop_visual_prompt_reasoning/real_anomaly_crop_visual_prompt_mean_summary.csv"
    df = read_csv_safe(path)
    if df is not None:
        for _, r in df.iterrows():
            add_row(
                rows,
                stage="Stage 6.3",
                branch="defect_type_reasoning",
                method=f"{r['strategy']} + {r['eval_mode']}",
                setting_type="partial_candidate_or_fallback",
                is_realistic_full_test=False,
                input_mode=r["eval_mode"],
                candidate_source="old PatchCore candidates",
                prompt_strategy=r["strategy"],
                num_images_total=to_int(r["num_images_total"]),
                num_images_used=to_int(r["num_images_used"]),
                coverage_ratio=to_float(r["coverage_ratio"]),
                fallback_count=to_int(r["fallback_count"]),
                skipped_count=to_int(r["skipped_count"]),
                top1_accuracy=to_float(r["top1_accuracy"]),
                top2_accuracy=to_float(r["top2_accuracy"]),
                macro_f1=to_float(r["macro_f1"]),
                main_gain="shows real anomaly crops help on covered subset",
                main_limitation="candidate coverage was only about half of the abnormal images",
                source_file=path,
            )

    # Stage 6.4: full-test real anomaly crop
    path = "results/analysis/real_anomaly_crop_visual_prompt_reasoning_full_test/real_anomaly_crop_visual_prompt_mean_summary.csv"
    df = read_csv_safe(path)
    if df is not None:
        for _, r in df.iterrows():
            add_row(
                rows,
                stage="Stage 6.4",
                branch="defect_type_reasoning",
                method=f"{r['strategy']} + {r['eval_mode']}",
                setting_type="realistic_full_test" if r["eval_mode"] in ["full_all", "crop_or_full", "crop_topk_ensemble"] else "near_full_candidate_subset",
                is_realistic_full_test=(r["eval_mode"] in ["full_all", "crop_or_full", "crop_topk_ensemble"]),
                input_mode=r["eval_mode"],
                candidate_source="full-test PatchCore candidates" if "crop" in r["eval_mode"] else "none",
                prompt_strategy=r["strategy"],
                num_images_total=to_int(r["num_images_total"]),
                num_images_used=to_int(r["num_images_used"]),
                coverage_ratio=to_float(r["coverage_ratio"]),
                fallback_count=to_int(r["fallback_count"]),
                skipped_count=to_int(r["skipped_count"]),
                top1_accuracy=to_float(r["top1_accuracy"]),
                top2_accuracy=to_float(r["top2_accuracy"]),
                macro_f1=to_float(r["macro_f1"]),
                main_gain="full-test PatchCore anomaly crops improve realistic defect type reasoning",
                main_limitation="still limited by noisy candidate crops and low CLIP confidence margin",
                source_file=path,
            )

    # Stage 6.5: explanation generation
    path = "results/analysis/manufacturing_defect_explanations/manufacturing_defect_explanation_summary.csv"
    df = read_csv_safe(path)
    if df is not None:
        mean = df[df["category"] == "MEAN"]
        if len(mean) > 0:
            r = mean.iloc[0]
            add_row(
                rows,
                stage="Stage 6.5",
                branch="manufacturing_explanation",
                method="manufacturing-aware explanation generation",
                setting_type="same_as_Stage_6.4_generic_label_crop_topk_only",
                is_realistic_full_test=False,
                input_mode="crop_topk_only",
                candidate_source="full-test PatchCore candidates",
                prompt_strategy="generic_label classification + manufacturing knowledge explanation",
                baseline_reference="Stage 6.4 generic_label + crop_topk_only",
                num_images_total=328,
                num_images_used=to_int(r["num_reports"]),
                coverage_ratio=to_int(r["num_reports"]) / 328,
                top1_accuracy=to_float(r["top1_accuracy"]),
                top2_accuracy=to_float(r["top2_accuracy"]),
                macro_f1="",
                main_gain="adds structured manufacturing-aware explanation without changing classification output",
                main_limitation="explanations are candidate reasoning, not verified causal diagnosis",
                source_file=path,
            )


def add_localization_rows(rows):
    # PatchCore full 15-class baseline
    path = "results/baselines/patchcore_mvtec_summary.csv"
    df = read_csv_safe(path)
    if df is not None:
        # Column names may vary; preserve robust detection.
        cols = {c.lower(): c for c in df.columns}
        pixel_f1_col = next((c for c in df.columns if "pixel" in c.lower() and "f1" in c.lower()), None)
        pixel_auroc_col = next((c for c in df.columns if "pixel" in c.lower() and "auroc" in c.lower()), None)
        image_auroc_col = next((c for c in df.columns if "image" in c.lower() and "auroc" in c.lower()), None)

        weak = df[df.iloc[:, 0].astype(str).isin(["grid", "screw", "leather", "wood"])]
        if len(weak) > 0 and pixel_f1_col:
            add_row(
                rows,
                stage="Baseline",
                branch="localization",
                method="PatchCore baseline weak-category subset",
                setting_type="realistic_patchcore_baseline",
                is_realistic_full_test=True,
                candidate_source="PatchCore anomaly map",
                num_images_total="weak-category test set",
                pixel_f1=weak[pixel_f1_col].astype(float).mean(),
                main_gain="strong unsupervised anomaly localization baseline",
                main_limitation="pixel-level F1 remains weak on texture and small-structure defects",
                source_file=path,
            )

        if pixel_f1_col:
            add_row(
                rows,
                stage="Baseline",
                branch="localization",
                method="PatchCore baseline all MVTecAD classes",
                setting_type="realistic_patchcore_baseline",
                is_realistic_full_test=True,
                candidate_source="PatchCore anomaly map",
                num_images_total="15 categories",
                pixel_f1=df[pixel_f1_col].astype(float).mean(),
                main_gain="image-level anomaly detection is strong",
                main_limitation="pixel-level localization still leaves room for improvement",
                source_file=path,
            )

    # Stage 4/5 localization-calibration summaries
    summaries = [
        (
            "Stage 4.1",
            "anomaly region scoring baseline",
            "results/analysis/anomaly_region_scoring_baseline/anomaly_region_scoring_baseline_summary.csv",
            "eval_patchcore_top1_f1",
            "eval_region_score_f1",
            "weak positive candidate ranking",
            "uses candidate subset and hand-designed features",
        ),
        (
            "Stage 4.2",
            "enhanced anomaly region scoring",
            "results/analysis/enhanced_anomaly_region_scoring/enhanced_anomaly_region_scoring_summary.csv",
            "eval_patchcore_top1_f1",
            "eval_enhanced_region_f1",
            "adds contrast/stability/shape features",
            "does not outperform Stage 4.1",
        ),
        (
            "Stage 5.1",
            "simple anomaly-map calibration",
            "results/analysis/anomaly_map_calibration/anomaly_map_calibration_summary.csv",
            "",
            "best_mean_f1",
            "tests fixed/percentile/mean-std thresholding",
            "best method remains fixed threshold",
        ),
        (
            "Stage 5.2",
            "trainable Tiny CNN anomaly-map calibration",
            "results/analysis/trainable_anomaly_map_calibration/trainable_anomaly_map_calibration_summary.csv",
            "eval_raw_f1",
            "eval_calibrated_f1",
            "tests trainable calibration",
            "naive trainable calibration collapses on several categories",
        ),
        (
            "Stage 5.3",
            "conservative residual anomaly-map calibration",
            "results/analysis/conservative_residual_anomaly_calibration/conservative_residual_anomaly_calibration_summary.csv",
            "eval_raw_abnormal_f1",
            "eval_calibrated_abnormal_f1",
            "stable residual calibration",
            "gain is too small to use as main module",
        ),
    ]

    for stage, method, path, base_col, new_col, gain, limitation in summaries:
        df = read_csv_safe(path)
        if df is None:
            continue
        mean = df[df["category"] == "MEAN"] if "category" in df.columns else df.tail(1)
        if len(mean) == 0:
            continue
        r = mean.iloc[0]
        base_v = to_float(r[base_col]) if base_col and base_col in r else np.nan
        new_v = to_float(r[new_col]) if new_col in r else np.nan
        delta = new_v - base_v if not np.isnan(base_v) and not np.isnan(new_v) else ""
        add_row(
            rows,
            stage=stage,
            branch="localization",
            method=method,
            setting_type="diagnostic_localization_or_calibration",
            candidate_source="PatchCore anomaly map",
            baseline_reference=base_col,
            pixel_f1=new_v,
            delta_macro_f1_vs_fair_baseline=delta,
            main_gain=gain,
            main_limitation=limitation,
            source_file=path,
        )

    # Stage 5.4 sweep best
    path = "results/analysis/conservative_residual_calibration_sweep/conservative_residual_calibration_sweep_summary.csv"
    df = read_csv_safe(path)
    if df is not None and "mean_delta_abnormal_f1" in df.columns:
        best = df.sort_values("mean_delta_abnormal_f1", ascending=False).iloc[0]
        add_row(
            rows,
            stage="Stage 5.4",
            branch="localization",
            method=f"conservative residual calibration sweep best: {best['config_name']}",
            setting_type="diagnostic_hyperparameter_sweep",
            candidate_source="PatchCore anomaly map",
            pixel_f1=to_float(best["mean_calibrated_abnormal_f1"]),
            delta_macro_f1_vs_fair_baseline=to_float(best["mean_delta_abnormal_f1"]),
            normal_fp_ratio=to_float(best["mean_calibrated_normal_fp_ratio"]),
            main_gain="best residual calibration setting found by sweep",
            main_limitation="best gain remains too small; trainable calibration line stopped",
            source_file=path,
        )


def build_pairwise_deltas():
    path = "results/analysis/real_anomaly_crop_visual_prompt_reasoning_full_test/real_anomaly_crop_visual_prompt_mean_summary.csv"
    df = read_csv_safe(path)
    rows = []

    if df is None:
        return pd.DataFrame(rows)

    for strategy in sorted(df["strategy"].unique()):
        base = df[(df["strategy"] == strategy) & (df["eval_mode"] == "full_all")]
        if len(base) == 0:
            continue
        b = base.iloc[0]

        for mode in ["crop_or_full", "crop_topk_ensemble", "crop_only", "crop_topk_only"]:
            cur = df[(df["strategy"] == strategy) & (df["eval_mode"] == mode)]
            if len(cur) == 0:
                continue
            r = cur.iloc[0]

            same_used = int(r["num_images_used"]) == int(b["num_images_used"])
            rows.append(
                {
                    "comparison_scope": "Stage 6.4 full-test reasoning",
                    "strategy": strategy,
                    "baseline_mode": "full_all",
                    "method_mode": mode,
                    "baseline_num_used": int(b["num_images_used"]),
                    "method_num_used": int(r["num_images_used"]),
                    "same_image_count": same_used,
                    "method_coverage_ratio": float(r["coverage_ratio"]),
                    "baseline_top1": float(b["top1_accuracy"]),
                    "method_top1": float(r["top1_accuracy"]),
                    "delta_top1": float(r["top1_accuracy"]) - float(b["top1_accuracy"]),
                    "baseline_top2": float(b["top2_accuracy"]),
                    "method_top2": float(r["top2_accuracy"]),
                    "delta_top2": float(r["top2_accuracy"]) - float(b["top2_accuracy"]),
                    "baseline_macro_f1": float(b["macro_f1"]),
                    "method_macro_f1": float(r["macro_f1"]),
                    "delta_macro_f1": float(r["macro_f1"]) - float(b["macro_f1"]),
                    "fairness_note": (
                        "same 328-image setting"
                        if same_used
                        else "near-full candidate subset; not identical image count"
                    ),
                }
            )

    return pd.DataFrame(rows)


def write_markdown(unified_df, delta_df):
    out = OUT_ROOT / "unified_experiment_comparison.md"

    lines = [
        "# Unified Fair Experiment Comparison",
        "",
        "## 1. Purpose",
        "",
        "This document consolidates localization, defect type reasoning, and manufacturing-aware explanation results.",
        "",
        "The table explicitly separates realistic full-test settings, partial candidate subsets, and GT-crop upper-bound settings.",
        "",
        "## 2. Key Fair-comparison Rules",
        "",
        "- Compare methods only under the same dataset, category set, image set, and evaluation protocol.",
        "- Treat GT-crop results as upper-bound diagnostics, not fair deployable results.",
        "- Treat crop-only results as candidate-covered subset results unless image count matches the full test set.",
        "- Report coverage, fallback, and skipped counts for crop-based reasoning.",
        "- Stage 6.5 inherits classification metrics from Stage 6.4 and adds explanation capability.",
        "",
        "## 3. Unified Experiment Table",
        "",
    ]

    display_cols = [
        "stage",
        "branch",
        "method",
        "setting_type",
        "num_images_used",
        "coverage_ratio",
        "fallback_count",
        "skipped_count",
        "top1_accuracy",
        "top2_accuracy",
        "macro_f1",
        "pixel_f1",
        "normal_fp_ratio",
        "main_gain",
        "main_limitation",
    ]

    table_df = unified_df[display_cols].copy()
    lines.append(table_df.to_markdown(index=False))

    lines += [
        "",
        "## 4. Fair Pairwise Deltas for Stage 6.4",
        "",
        "These comparisons use full image reasoning as the baseline for each prompt strategy.",
        "",
    ]

    if len(delta_df) > 0:
        lines.append(delta_df.to_markdown(index=False))
    else:
        lines.append("No pairwise delta table was generated.")

    lines += [
        "",
        "## 5. Current Main Conclusion",
        "",
        "The strongest realistic positive branch is full-test PatchCore anomaly-crop defect reasoning with short visual prompts.",
        "",
        "Manufacturing process knowledge should be used as an explanation layer rather than directly as a long CLIP classification prompt.",
        "",
    ]

    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"[DONE] Wrote markdown: {out}")


def main():
    rows = []
    add_localization_rows(rows)
    add_reasoning_rows(rows)

    unified_df = pd.DataFrame(rows)

    # Stable column order.
    unified_csv = OUT_ROOT / "unified_experiment_comparison.csv"
    unified_df.to_csv(unified_csv, index=False)

    delta_df = build_pairwise_deltas()
    delta_csv = OUT_ROOT / "fair_pairwise_delta_comparison.csv"
    delta_df.to_csv(delta_csv, index=False)

    write_markdown(unified_df, delta_df)

    print(f"[DONE] Wrote unified CSV: {unified_csv}")
    print(f"[DONE] Wrote pairwise delta CSV: {delta_csv}")

    print("\n========== Main realistic Stage 6.4 deltas ==========")
    if len(delta_df) > 0:
        print(delta_df.sort_values("delta_top1", ascending=False).to_string(index=False))
    else:
        print("No deltas generated.")


if __name__ == "__main__":
    main()
