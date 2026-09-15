from __future__ import annotations

from pathlib import Path
import pandas as pd


ROOT = Path(".").resolve()

STAGE11E_TABLE = ROOT / "results/stage11_mvtecad2_multicategory/stage11_e_multicategory_evidence_table.csv"
STAGE11F3_SUMMARY = ROOT / "results/stage11_mvtecad2_multicategory/stage11_f3_vial_image_set_overlap_summary.csv"
STAGE11D_SUMMARY = ROOT / "results/stage11_mvtecad2_multicategory/stage11_d_vlm_summary.csv"
STAGE10F_SUMMARY = ROOT / "results/stage10_dataset_expansion/stage10_f_multiscale_context_summary.csv"

OUT_DIR = ROOT / "results/stage11_mvtecad2_multicategory"
DOC_DIR = ROOT / "docs/stage11_mvtecad2_multicategory"

OUT_TABLE = OUT_DIR / "stage11_g_unified_experiment_policy_table.csv"
OUT_REPORT = DOC_DIR / "stage11_g_unified_experiment_policy_report.md"


def f4(x) -> str:
    if x is None or pd.isna(x):
        return ""
    return f"{float(x):.4f}"


def get_row(df: pd.DataFrame, category: str, method: str):
    part = df[(df["category"] == category) & (df["method"] == method)]
    if part.empty:
        return None
    return part.iloc[0]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DOC_DIR.mkdir(parents=True, exist_ok=True)

    e = pd.read_csv(STAGE11E_TABLE)
    f3 = pd.read_csv(STAGE11F3_SUMMARY)
    d = pd.read_csv(STAGE11D_SUMMARY)
    s10 = pd.read_csv(STAGE10F_SUMMARY)

    f3_row = f3.iloc[0]

    all_primary = e[e["category"] == "ALL_PRIMARY"].iloc[0]

    stage10_vial_full = get_row(s10, "vial", "full_image")
    stage10_vial_ctx = get_row(s10, "vial", "context_1.50_top1")

    stage11_vial_full = get_row(d, "vial", "full_image")
    stage11_vial_ctx = get_row(d, "vial", "context_1.50_top1")

    policy_rows = [
        {
            "item": "main_experiment_source",
            "decision": "Use Stage 11-D/E unified multi-category pipeline as the paper-level main evidence.",
            "reason": "Stage 11 uses one unified data adapter, candidate construction script, and VLM scoring script across primary AD2 categories.",
        },
        {
            "item": "stage10_vial_status",
            "decision": "Keep Stage 10-G vial as historical single-category observation, not as a directly comparable main-table result.",
            "reason": "Stage 11-F3 shows Stage 10-G and Stage 11-D vial image sets are not sufficiently aligned.",
        },
        {
            "item": "vial_cross_stage_comparison",
            "decision": "Do not claim Stage 10-G and Stage 11-D vial numbers contradict or reproduce each other.",
            "reason": f"Best image-set overlap intersection is {int(f3_row['best_intersection'])}, directly_comparable={bool(f3_row['directly_comparable'])}.",
        },
        {
            "item": "main_claim",
            "decision": "Claim category-dependent benefit of localization-guided context-aware crops, strongest on aggregate primary set, fruit_jelly, and walnuts.",
            "reason": f"ALL_PRIMARY best VLM method is {all_primary['best_vlm_method']} with AUROC gain {float(all_primary['best_vlm_delta_vs_full']):.4f}.",
        },
        {
            "item": "limitation_cases",
            "decision": "Use sheet_metal and Stage 11 vial as limitation/failure-analysis cases, not as evidence against the whole method.",
            "reason": "Candidate quality and image-set/pipeline sensitivity affect crop-based VLM reasoning.",
        },
    ]

    table = pd.DataFrame(policy_rows)
    table.to_csv(OUT_TABLE, index=False)

    lines = []
    lines.append("# Stage 11-G Unified Experiment Policy")
    lines.append("")
    lines.append("## 1. Purpose")
    lines.append("")
    lines.append("This report fixes the experimental interpretation policy after the Stage 11-F3 vial image-set overlap audit.")
    lines.append("It does not run PatchCore, VLM inference, or crop generation.")
    lines.append("")
    lines.append("## 2. Key Decision")
    lines.append("")
    lines.append("The paper-level main evidence should use Stage 11-D/E, not a mixed comparison between Stage 10-G and Stage 11-D.")
    lines.append("")
    lines.append("Stage 10-G vial remains useful as a historical single-category observation, but it should not be placed in the same fair-comparison table as Stage 11-D vial.")
    lines.append("")
    lines.append("## 3. Why Stage 10-G and Stage 11-D vial are not directly comparable")
    lines.append("")
    lines.append("| Item | Value |")
    lines.append("|---|---:|")
    lines.append(f"| Best alignment | {f3_row['best_comparison']} |")
    lines.append(f"| Best intersection | {int(f3_row['best_intersection'])} |")
    lines.append(f"| Stage 10 unique keys | {int(f3_row['stage10_unique'])} |")
    lines.append(f"| Stage 11 unique keys | {int(f3_row['stage11_unique'])} |")
    lines.append(f"| Jaccard | {f4(f3_row['jaccard'])} |")
    lines.append(f"| Directly comparable | {bool(f3_row['directly_comparable'])} |")
    lines.append("")
    lines.append("## 4. Vial numbers should be interpreted separately")
    lines.append("")
    lines.append("| Source | Full-image AUROC | Context AUROC | Delta | Interpretation |")
    lines.append("|---|---:|---:|---:|---|")

    if stage10_vial_full is not None and stage10_vial_ctx is not None:
        delta10 = float(stage10_vial_ctx["auroc"]) - float(stage10_vial_full["auroc"])
        lines.append(
            f"| Stage 10-G vial | {f4(stage10_vial_full['auroc'])} | "
            f"{f4(stage10_vial_ctx['auroc'])} | {f4(delta10)} | Historical single-category observation |"
        )

    if stage11_vial_full is not None and stage11_vial_ctx is not None:
        delta11 = float(stage11_vial_ctx["auroc"]) - float(stage11_vial_full["auroc"])
        lines.append(
            f"| Stage 11-D vial | {f4(stage11_vial_full['auroc'])} | "
            f"{f4(stage11_vial_ctx['auroc'])} | {f4(delta11)} | Unified multi-category pipeline result |"
        )

    lines.append("")
    lines.append("## 5. Main Stage 11 result")
    lines.append("")
    lines.append("| Scope | Full AUROC | Best VLM Method | Best VLM AUROC | Delta |")
    lines.append("|---|---:|---|---:|---:|")
    lines.append(
        f"| ALL_PRIMARY | {f4(all_primary['full_image_auroc'])} | "
        f"{all_primary['best_vlm_method']} | {f4(all_primary['best_vlm_auroc'])} | "
        f"{f4(all_primary['best_vlm_delta_vs_full'])} |"
    )
    lines.append("")
    lines.append("## 6. Final paper-level wording")
    lines.append("")
    lines.append("Recommended claim:")
    lines.append("")
    lines.append("```text")
    lines.append("PatchCore localization can serve as a visual bridge for VLM anomaly reasoning when candidate regions preserve sufficient object context. On the unified MVTec AD 2 primary-category evaluation, context-aware crop aggregation improves over full-image VLM prompting, while category-level failures reveal sensitivity to candidate quality and object context.")
    lines.append("```")
    lines.append("")
    lines.append("Avoid this claim:")
    lines.append("")
    lines.append("```text")
    lines.append("Context-aware crops consistently improve every category.")
    lines.append("```")
    lines.append("")
    lines.append("## 7. Policy Table")
    lines.append("")
    lines.append("| Item | Decision | Reason |")
    lines.append("|---|---|---|")
    for _, r in table.iterrows():
        lines.append(f"| {r['item']} | {r['decision']} | {r['reason']} |")
    lines.append("")
    lines.append("## 8. Next Step")
    lines.append("")
    lines.append("After this policy is committed, the next practical step is to decide whether to include the secondary category `fabric` under the same Stage 11 pipeline, or to move directly to a paper-ready Stage 11 final table and method narrative.")
    lines.append("")
    lines.append("## 9. Output")
    lines.append("")
    lines.append(f"- Policy table: `{OUT_TABLE.relative_to(ROOT)}`")
    lines.append(f"- Report: `{OUT_REPORT.relative_to(ROOT)}`")

    OUT_REPORT.write_text("\n".join(lines), encoding="utf-8")

    print("[DONE]", OUT_TABLE)
    print("[DONE]", OUT_REPORT)
    print("")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
