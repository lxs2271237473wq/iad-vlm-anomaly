from __future__ import annotations

from pathlib import Path
import pandas as pd


ROOT = Path(".").resolve()

STAGE11B_QUALITY = ROOT / "results/stage11_mvtecad2_multicategory/stage11_b1_patchcore_detector_quality_analysis.csv"
STAGE11C_CANDIDATE = ROOT / "results/stage11_mvtecad2_multicategory/stage11_c_candidate_summary.csv"
STAGE11D_SUMMARY = ROOT / "results/stage11_mvtecad2_multicategory/stage11_d_vlm_summary.csv"
STAGE10F_SUMMARY = ROOT / "results/stage10_dataset_expansion/stage10_f_multiscale_context_summary.csv"

OUT_DIR = ROOT / "results/stage11_mvtecad2_multicategory"
DOC_DIR = ROOT / "docs/stage11_mvtecad2_multicategory"

OUT_TABLE = OUT_DIR / "stage11_e_multicategory_evidence_table.csv"
OUT_REPORT = DOC_DIR / "stage11_e_multicategory_evidence_report.md"


def pick(df: pd.DataFrame, category: str, method: str) -> pd.Series | None:
    part = df[(df["category"] == category) & (df["method"] == method)]
    if part.empty:
        return None
    return part.iloc[0]


def best_method(df: pd.DataFrame, category: str, include_patchcore: bool = False) -> pd.Series:
    part = df[df["category"] == category].copy()
    if not include_patchcore:
        part = part[part["method"] != "patchcore_score"].copy()
    return part.sort_values("auroc", ascending=False).iloc[0]


def best_prefix(df: pd.DataFrame, category: str, prefix: str) -> pd.Series | None:
    part = df[(df["category"] == category) & (df["method"].str.startswith(prefix))].copy()
    if part.empty:
        return None
    return part.sort_values("auroc", ascending=False).iloc[0]


def f4(x) -> str:
    if x is None or pd.isna(x):
        return ""
    return f"{float(x):.4f}"


def decision_from_rows(full, best_vlm, best_context, best_tight) -> tuple[str, str]:
    full_auroc = float(full["auroc"])
    best_vlm_method = str(best_vlm["method"])
    best_vlm_delta = float(best_vlm["auroc"]) - full_auroc

    context_delta = None
    if best_context is not None:
        context_delta = float(best_context["auroc"]) - full_auroc

    tight_delta = None
    if best_tight is not None:
        tight_delta = float(best_tight["auroc"]) - full_auroc

    if best_vlm_method.startswith("context") and best_vlm_delta > 0:
        return (
            "main_positive_context",
            "Context-aware crop is the best VLM method and improves over full-image prompting.",
        )

    if context_delta is not None and context_delta > 0:
        return (
            "supportive_context",
            "Context-aware crop improves over full image but is not necessarily the best crop aggregation.",
        )

    if tight_delta is not None and tight_delta > 0:
        return (
            "supportive_tight_only",
            "A crop variant improves over full image, but context-aware crop is not the strongest evidence.",
        )

    return (
        "negative_or_full_stronger",
        "Full-image prompting is stronger or tied; this category should be used for limitation/failure analysis.",
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DOC_DIR.mkdir(parents=True, exist_ok=True)

    quality = pd.read_csv(STAGE11B_QUALITY)
    candidate = pd.read_csv(STAGE11C_CANDIDATE)
    summary = pd.read_csv(STAGE11D_SUMMARY)

    stage10f = pd.read_csv(STAGE10F_SUMMARY) if STAGE10F_SUMMARY.exists() else pd.DataFrame()

    categories = sorted([c for c in summary["category"].unique().tolist() if c != "ALL_PRIMARY"])
    rows = []

    for category in categories + ["ALL_PRIMARY"]:
        full = pick(summary, category, "full_image")
        patchcore = pick(summary, category, "patchcore_score")
        best_vlm = best_method(summary, category, include_patchcore=False)
        best_context = best_prefix(summary, category, "context")
        best_tight = best_prefix(summary, category, "tight")

        decision, interpretation = decision_from_rows(full, best_vlm, best_context, best_tight)

        qrow = quality[quality["category"] == category]
        crow = candidate[candidate["category"] == category]

        row = {
            "dataset": "MVTec AD 2",
            "category": category,
            "detector_priority_group": "" if qrow.empty else qrow.iloc[0].get("stage11_c_priority_group", ""),
            "image_AUROC_patchcore_stage11b": "" if qrow.empty else qrow.iloc[0].get("image_AUROC", ""),
            "pixel_AUROC_patchcore_stage11b": "" if qrow.empty else qrow.iloc[0].get("pixel_AUROC", ""),
            "pixel_F1_patchcore_stage11b": "" if qrow.empty else qrow.iloc[0].get("pixel_F1Score", ""),
            "candidate_coverage": "" if crow.empty else crow.iloc[0].get("candidate_coverage", ""),
            "top1_tight_gt_coverage": "" if crow.empty else crow.iloc[0].get("top1_tight_mean_gt_coverage_anomaly", ""),
            "top1_context_gt_coverage": "" if crow.empty else crow.iloc[0].get("top1_context_mean_gt_coverage_anomaly", ""),
            "full_image_auroc": float(full["auroc"]),
            "best_vlm_method": best_vlm["method"],
            "best_vlm_auroc": float(best_vlm["auroc"]),
            "best_vlm_delta_vs_full": float(best_vlm["auroc"]) - float(full["auroc"]),
            "best_context_method": "" if best_context is None else best_context["method"],
            "best_context_auroc": "" if best_context is None else float(best_context["auroc"]),
            "best_context_delta_vs_full": "" if best_context is None else float(best_context["auroc"]) - float(full["auroc"]),
            "best_tight_method": "" if best_tight is None else best_tight["method"],
            "best_tight_auroc": "" if best_tight is None else float(best_tight["auroc"]),
            "best_tight_delta_vs_full": "" if best_tight is None else float(best_tight["auroc"]) - float(full["auroc"]),
            "patchcore_score_auroc_stage11d": "" if patchcore is None else float(patchcore["auroc"]),
            "evidence_decision": decision,
            "interpretation": interpretation,
        }

        if category == "vial" and not stage10f.empty:
            s10_full = pick(stage10f, "vial", "full_image")
            s10_context = pick(stage10f, "vial", "context_1.50_top1")

            if s10_full is not None and s10_context is not None:
                row["stage10_vial_full_auroc"] = float(s10_full["auroc"])
                row["stage10_vial_context_1p50_top1_auroc"] = float(s10_context["auroc"])
                row["stage10_vial_context_delta"] = float(s10_context["auroc"]) - float(s10_full["auroc"])
                row["cross_stage_note"] = (
                    "Stage 10-G showed positive vial context evidence, but Stage 11-D batch candidate construction did not reproduce the same margin."
                )
            else:
                row["cross_stage_note"] = ""
        else:
            row["cross_stage_note"] = ""

        rows.append(row)

    table = pd.DataFrame(rows)
    table.to_csv(OUT_TABLE, index=False)

    lines = []
    lines.append("# Stage 11-E Multi-category Evidence Consolidation")
    lines.append("")
    lines.append("## 1. Purpose")
    lines.append("")
    lines.append("This report consolidates Stage 11-B detector quality, Stage 11-C candidate quality, and Stage 11-D VLM full-image versus crop reasoning results.")
    lines.append("It does not run any model, generate crops, or modify datasets.")
    lines.append("")
    lines.append("## 2. Main Result")
    lines.append("")
    all_row = table[table["category"] == "ALL_PRIMARY"].iloc[0]
    lines.append(
        f"Across all primary categories, the best VLM method is `{all_row['best_vlm_method']}` "
        f"with AUROC `{float(all_row['best_vlm_auroc']):.4f}`, compared with full-image AUROC "
        f"`{float(all_row['full_image_auroc']):.4f}`. "
        f"The AUROC gain is `{float(all_row['best_vlm_delta_vs_full']):.4f}`."
    )
    lines.append("")
    lines.append("This supports the claim that localization-guided context-aware crops can improve VLM anomaly reasoning, but the effect is category-dependent.")
    lines.append("")
    lines.append("## 3. Category-level Evidence")
    lines.append("")
    lines.append("| Category | Best VLM | Full AUROC | Best VLM AUROC | ΔAUROC | Best context | Context Δ | Decision |")
    lines.append("|---|---|---:|---:|---:|---|---:|---|")

    for _, r in table[table["category"] != "ALL_PRIMARY"].iterrows():
        lines.append(
            f"| {r['category']} | {r['best_vlm_method']} | {f4(r['full_image_auroc'])} | "
            f"{f4(r['best_vlm_auroc'])} | {f4(r['best_vlm_delta_vs_full'])} | "
            f"{r['best_context_method']} | {f4(r['best_context_delta_vs_full'])} | "
            f"{r['evidence_decision']} |"
        )

    lines.append("")
    lines.append("## 4. Candidate Quality Reference")
    lines.append("")
    lines.append("| Category | Candidate coverage | Tight GT coverage | Context GT coverage |")
    lines.append("|---|---:|---:|---:|")

    for _, r in table[table["category"] != "ALL_PRIMARY"].iterrows():
        lines.append(
            f"| {r['category']} | {f4(r['candidate_coverage'])} | "
            f"{f4(r['top1_tight_gt_coverage'])} | {f4(r['top1_context_gt_coverage'])} |"
        )

    lines.append("")
    lines.append("## 5. Vial Cross-stage Consistency Note")
    lines.append("")
    vial = table[table["category"] == "vial"].iloc[0]
    if "stage10_vial_context_delta" in table.columns and not pd.isna(vial.get("stage10_vial_context_delta", None)):
        lines.append(
            f"Stage 10-G vial: full-image AUROC `{float(vial['stage10_vial_full_auroc']):.4f}`, "
            f"context_1.50_top1 AUROC `{float(vial['stage10_vial_context_1p50_top1_auroc']):.4f}`, "
            f"delta `{float(vial['stage10_vial_context_delta']):.4f}`."
        )
        lines.append("")
        lines.append(
            f"Stage 11-D vial: full-image AUROC `{float(vial['full_image_auroc']):.4f}`, "
            f"best context AUROC `{float(vial['best_context_auroc']):.4f}`, "
            f"context delta `{float(vial['best_context_delta_vs_full']):.4f}`."
        )
        lines.append("")
        lines.append("This discrepancy should be treated as implementation/candidate-construction sensitivity, not as a direct contradiction of the method idea.")
    else:
        lines.append("Stage 10-F/G vial summary was unavailable for cross-stage comparison.")

    lines.append("")
    lines.append("## 6. Paper-level Interpretation")
    lines.append("")
    lines.append("The strongest claim is not that every crop improves every category.")
    lines.append("The defensible claim is:")
    lines.append("")
    lines.append("```text")
    lines.append("PatchCore localization can serve as a visual bridge for VLM anomaly reasoning when candidate regions preserve sufficient object context; the benefit is category- and candidate-quality-dependent.")
    lines.append("```")
    lines.append("")
    lines.append("The current evidence supports context-aware crop reasoning on the aggregate primary set, fruit_jelly, and walnuts.")
    lines.append("sheet_metal should be discussed as a failure/limitation case, and vial requires a consistency check between the Stage 10 single-category pipeline and Stage 11 batch pipeline.")
    lines.append("")
    lines.append("## 7. Next Step")
    lines.append("")
    lines.append("The next step should not be fabric expansion yet. First, inspect the Stage 10 vs Stage 11 vial candidate construction difference and decide whether Stage 11-C needs to reuse the Stage 10 candidate policy exactly.")
    lines.append("")
    lines.append("## 8. Output")
    lines.append("")
    lines.append(f"- Evidence table: `{OUT_TABLE.relative_to(ROOT)}`")
    lines.append(f"- Report: `{OUT_REPORT.relative_to(ROOT)}`")

    OUT_REPORT.write_text("\n".join(lines), encoding="utf-8")

    print("[DONE]", OUT_TABLE)
    print("[DONE]", OUT_REPORT)
    print("")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
