from pathlib import Path
import pandas as pd
import numpy as np


OUT_ROOT = Path("results/analysis/paper_ready_tables")
OUT_ROOT.mkdir(parents=True, exist_ok=True)

DOC_PATH = Path("docs/paper_ready_experiment_tables.md")


def read_csv(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path)


def num(x):
    try:
        if pd.isna(x) or x == "":
            return np.nan
        return float(x)
    except Exception:
        return np.nan


def fmt(x, digits=4):
    x = num(x)
    if np.isnan(x):
        return "-"
    return f"{x:.{digits}f}"


def delta_fmt(x, digits=4):
    x = num(x)
    if np.isnan(x):
        return "-"
    sign = "+" if x >= 0 else ""
    return f"{sign}{x:.{digits}f}"


def md_table(df):
    if len(df) == 0:
        return "No data.\n"

    cols = list(df.columns)
    lines = []
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")

    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")

    return "\n".join(lines)


def build_main_result_table(delta_df):
    # Fair full-test comparison only: same 328 images.
    fair = delta_df[
        (delta_df["same_image_count"] == True)
        & (delta_df["method_mode"] == "crop_topk_ensemble")
    ].copy()

    fair["delta_top1"] = fair["delta_top1"].astype(float)
    fair["delta_top2"] = fair["delta_top2"].astype(float)
    fair["delta_macro_f1"] = fair["delta_macro_f1"].astype(float)

    fair = fair.sort_values(["delta_top1", "delta_macro_f1"], ascending=[False, False])

    rows = []
    for _, r in fair.iterrows():
        rows.append({
            "Prompt Strategy": r["strategy"],
            "Baseline": "full_all",
            "Method": "crop_topk_ensemble",
            "Images": f"{int(r['method_num_used'])}",
            "Top-1": fmt(r["method_top1"]),
            "Top-1 Δ": delta_fmt(r["delta_top1"]),
            "Top-2": fmt(r["method_top2"]),
            "Top-2 Δ": delta_fmt(r["delta_top2"]),
            "Macro-F1": fmt(r["method_macro_f1"]),
            "Macro-F1 Δ": delta_fmt(r["delta_macro_f1"]),
            "Fair Setting": r["fairness_note"],
        })

    return pd.DataFrame(rows)


def build_ablation_table(unified_df):
    # Stage 6.4 realistic and near-full anomaly-crop reasoning rows.
    df = unified_df[
        (unified_df["stage"] == "Stage 6.4")
        & (unified_df["branch"] == "defect_type_reasoning")
    ].copy()

    rows = []
    for _, r in df.iterrows():
        rows.append({
            "Strategy": r["prompt_strategy"],
            "Input Mode": r["input_mode"],
            "Setting Type": r["setting_type"],
            "Used Images": r["num_images_used"],
            "Coverage": fmt(r["coverage_ratio"]),
            "Fallback": r["fallback_count"],
            "Skipped": r["skipped_count"],
            "Top-1": fmt(r["top1_accuracy"]),
            "Top-2": fmt(r["top2_accuracy"]),
            "Macro-F1": fmt(r["macro_f1"]),
        })

    out = pd.DataFrame(rows)

    order = {
        "full_all": 0,
        "crop_or_full": 1,
        "crop_topk_ensemble": 2,
        "crop_only": 3,
        "crop_topk_only": 4,
    }
    out["_order"] = out["Input Mode"].map(order).fillna(99)
    out = out.sort_values(["Strategy", "_order"]).drop(columns=["_order"])

    return out


def build_upper_bound_table(unified_df):
    df = unified_df[
        (unified_df["setting_type"] == "GT_crop_upper_bound")
        & (unified_df["branch"] == "defect_type_reasoning")
    ].copy()

    rows = []
    for _, r in df.iterrows():
        rows.append({
            "Stage": r["stage"],
            "Method": r["method"],
            "Prompt Strategy": r["prompt_strategy"],
            "Input": r["input_mode"],
            "Candidate Source": r["candidate_source"],
            "Images": r["num_images_used"],
            "Top-1": fmt(r["top1_accuracy"]),
            "Top-2": fmt(r["top2_accuracy"]),
            "Macro-F1": fmt(r["macro_f1"]),
            "Role": "upper-bound diagnostic, not fair deployable result",
        })

    out = pd.DataFrame(rows)
    return out.sort_values(["Top-1", "Macro-F1"], ascending=[False, False])


def build_negative_auxiliary_table(unified_df):
    keep_keywords = [
        "SAM2",
        "calibration",
        "region scoring",
        "manufacturing-aware explanation",
    ]

    rows = []

    # Pull from unified table where available.
    for _, r in unified_df.iterrows():
        text = f"{r.get('stage', '')} {r.get('method', '')} {r.get('main_gain', '')} {r.get('main_limitation', '')}"
        if any(k.lower() in text.lower() for k in keep_keywords):
            rows.append({
                "Stage": r["stage"],
                "Branch": r["branch"],
                "Method": r["method"],
                "Metric": r["pixel_f1"] if pd.notna(r.get("pixel_f1", np.nan)) and r.get("pixel_f1", "") != "" else r.get("top1_accuracy", ""),
                "Main Gain": r["main_gain"],
                "Main Limitation": r["main_limitation"],
                "Use in Paper": "auxiliary / negative / motivation result",
            })

    # Add SAM2 explicit negative summary, because it may not be in unified CSV.
    rows.append({
        "Stage": "SAM2 analysis",
        "Branch": "localization",
        "Method": "naive SAM2 box prompt and anomaly-aware SAM2 selection",
        "Metric": "below or similar to PatchCore component mask",
        "Main Gain": "shows that generic segmentation does not directly solve defect mask refinement",
        "Main Limitation": "SAM2 tends to segment object/texture regions rather than defect regions",
        "Use in Paper": "negative result / motivation",
    })

    return pd.DataFrame(rows)


def build_explanation_table():
    path = "results/analysis/manufacturing_defect_explanations/manufacturing_defect_explanation_summary.csv"
    df = read_csv(path)

    rows = []
    for _, r in df.iterrows():
        rows.append({
            "Category": r["category"],
            "Reports": int(float(r["num_reports"])),
            "Predicted Defect Types": int(float(r["num_pred_defect_types"])),
            "Defect Families": int(float(r["num_defect_families"])),
            "Top-1": fmt(r["top1_accuracy"]),
            "Top-2": fmt(r["top2_accuracy"]),
            "Confidence Margin": fmt(r["mean_confidence_margin"]),
            "Role": "explanation layer, not accuracy-improvement module",
        })

    return pd.DataFrame(rows)


def main():
    unified_df = read_csv("results/analysis/final_comparison/unified_experiment_comparison.csv")
    delta_df = read_csv("results/analysis/final_comparison/fair_pairwise_delta_comparison.csv")

    main_result = build_main_result_table(delta_df)
    ablation = build_ablation_table(unified_df)
    upper_bound = build_upper_bound_table(unified_df)
    negative_aux = build_negative_auxiliary_table(unified_df)
    explanation = build_explanation_table()

    main_result.to_csv(OUT_ROOT / "main_result_table.csv", index=False)
    ablation.to_csv(OUT_ROOT / "ablation_prompt_crop_table.csv", index=False)
    upper_bound.to_csv(OUT_ROOT / "gt_crop_upper_bound_table.csv", index=False)
    negative_aux.to_csv(OUT_ROOT / "negative_auxiliary_result_table.csv", index=False)
    explanation.to_csv(OUT_ROOT / "manufacturing_explanation_table.csv", index=False)

    lines = [
        "# Paper-ready Experiment Tables",
        "",
        "## 1. Main Fair Result Table",
        "",
        "This table uses the same 328-image full-test setting. The baseline is full-image reasoning under the same prompt strategy.",
        "",
        md_table(main_result),
        "",
        "## 2. Ablation: Prompt Strategy and Crop Mode",
        "",
        "This table shows full image, crop-or-full, top-k crop ensemble, crop-only, and crop-topk-only settings.",
        "",
        md_table(ablation),
        "",
        "## 3. GT-crop Upper-bound Diagnostic Table",
        "",
        "These results use ground-truth masks for cropping. They diagnose the potential of accurate localization but should not be reported as fair deployable results.",
        "",
        md_table(upper_bound),
        "",
        "## 4. Negative / Auxiliary Result Table",
        "",
        "These results explain why several branches are not used as the final main module.",
        "",
        md_table(negative_aux),
        "",
        "## 5. Manufacturing-aware Explanation Table",
        "",
        "Stage 6.5 adds structured manufacturing-aware explanations on top of the Stage 6.4 prediction results. It does not modify classification accuracy.",
        "",
        md_table(explanation),
        "",
        "## 6. Paper-level Conclusion",
        "",
        "The cleanest main result is the realistic full-test comparison:",
        "",
        "```text",
        "generic_label + full_all -> generic_label + crop_topk_ensemble",
        "same 328 images",
        "Top-1 improves from 0.2850 to 0.3388",
        "Macro-F1 improves from 0.1543 to 0.2206",
        "```",
        "",
        "This supports the final method direction:",
        "",
        "```text",
        "PatchCore anomaly crop -> short visual prompt defect reasoning -> manufacturing-aware explanation",
        "```",
        "",
    ]

    DOC_PATH.write_text("\n".join(lines), encoding="utf-8")

    print(f"[DONE] Wrote: {OUT_ROOT / 'main_result_table.csv'}")
    print(f"[DONE] Wrote: {OUT_ROOT / 'ablation_prompt_crop_table.csv'}")
    print(f"[DONE] Wrote: {OUT_ROOT / 'gt_crop_upper_bound_table.csv'}")
    print(f"[DONE] Wrote: {OUT_ROOT / 'negative_auxiliary_result_table.csv'}")
    print(f"[DONE] Wrote: {OUT_ROOT / 'manufacturing_explanation_table.csv'}")
    print(f"[DONE] Wrote: {DOC_PATH}")


if __name__ == "__main__":
    main()
