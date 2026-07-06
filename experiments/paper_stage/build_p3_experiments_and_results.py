from __future__ import annotations

from pathlib import Path
import pandas as pd


ROOT = Path(".").resolve()

IN_SYSTEM = ROOT / "results/stage16_qcru_ablation/stage16_d_paper_facing_system_baseline_table.csv"
IN_QCR = ROOT / "results/stage16_qcru_ablation/stage16_d_paper_facing_qcr_ablation_table.csv"
IN_DELTAS = ROOT / "results/stage16_qcru_ablation/stage16_d_paper_facing_claim_ready_deltas.csv"

IN_BOUNDARY_DECISION = ROOT / "results/stage16_qcru_ablation/stage16_e_boundary_decision_summary.csv"
IN_BOUNDARY_CATEGORY = ROOT / "results/stage16_qcru_ablation/stage16_e_category_boundary_summary.csv"
IN_CASES = ROOT / "results/stage16_qcru_ablation/stage16_e_failure_boundary_case_inventory.csv"

IN_E17_DELTA = ROOT / "results/stage17_defensive_sensitivity/stage17_a_efficientad100_vs_30_delta.csv"

OUT_DIR = ROOT / "results/paper_p3"
DOC_DIR = ROOT / "docs/paper_p3"

OUT_PARAGRAPH_INVENTORY = OUT_DIR / "paper_p3_result_paragraph_inventory.csv"
OUT_TABLE_MAP = OUT_DIR / "paper_p3_table_to_text_map.csv"
OUT_DOC = DOC_DIR / "paper_p3_experiments_and_results_draft.md"


def read_csv_strict(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    if len(df.columns) <= 1:
        raise RuntimeError(f"{path} read as <=1 column. Repair CSV formatting before P3.")
    return df


def fmt(x) -> str:
    try:
        if pd.isna(x):
            return "NA"
        return f"{float(x):.4f}"
    except Exception:
        return str(x)


def signed(x) -> str:
    try:
        if pd.isna(x):
            return "NA"
        return f"{float(x):+.4f}"
    except Exception:
        return str(x)


def get_system_score(system: pd.DataFrame, method: str):
    rows = system[system["method"] == method]
    if rows.empty:
        return None
    return float(rows.iloc[0]["mean_image_auroc"])


def get_delta(deltas: pd.DataFrame, contains: str):
    rows = deltas[deltas["comparison"].astype(str).str.contains(contains, regex=False, na=False)]
    if rows.empty:
        return None
    return float(rows.iloc[0]["delta"])


def get_e17_metric(e17: pd.DataFrame, metric: str):
    rows = e17[e17["metric"] == metric]
    if rows.empty:
        return None
    return rows.iloc[0].to_dict()


def qcr_mean(qcr: pd.DataFrame, method_contains: str):
    rows = qcr[qcr["method"].astype(str).str.contains(method_contains, regex=False, na=False)]
    if rows.empty:
        return None
    return float(rows["image_auroc"].mean())


def boundary_metric(category: pd.DataFrame, col: str):
    if col not in category.columns:
        return {"mean": None, "wins": 0, "total": 0}
    s = pd.to_numeric(category[col], errors="coerce").dropna()
    return {
        "mean": float(s.mean()) if len(s) else None,
        "wins": int((s > 0).sum()) if len(s) else 0,
        "total": int(len(s)),
    }


def make_inventory() -> pd.DataFrame:
    rows = [
        {
            "paragraph_id": "P3-1",
            "section": "Experimental Setup",
            "purpose": "Define datasets, categories, metrics, and protocol split.",
            "source_files": "Stage 15/16 result tables",
            "status": "drafted",
        },
        {
            "paragraph_id": "P3-2",
            "section": "Baselines",
            "purpose": "Describe detector, VLM, and external VLM anomaly baselines.",
            "source_files": "stage16_d_paper_facing_system_baseline_table.csv",
            "status": "drafted",
        },
        {
            "paragraph_id": "P3-3",
            "section": "Main Results",
            "purpose": "Report system-level strong baseline comparison.",
            "source_files": "stage16_d_paper_facing_system_baseline_table.csv; stage16_d_paper_facing_claim_ready_deltas.csv",
            "status": "drafted",
        },
        {
            "paragraph_id": "P3-4",
            "section": "QCR Ablation",
            "purpose": "Report Quality-Calibrated QCR and adaptive refinement ablation.",
            "source_files": "stage16_d_paper_facing_qcr_ablation_table.csv; stage16_d_paper_facing_claim_ready_deltas.csv",
            "status": "drafted",
        },
        {
            "paragraph_id": "P3-5",
            "section": "Failure and Boundary Analysis",
            "purpose": "Explain quality calibration boundaries and fixed consistency risk.",
            "source_files": "stage16_e_category_boundary_summary.csv; stage16_e_boundary_decision_summary.csv",
            "status": "drafted",
        },
        {
            "paragraph_id": "P3-6",
            "section": "EfficientAD Sensitivity",
            "purpose": "Defend EfficientAD-30 fixed-budget with fruit_jelly 100-epoch sensitivity.",
            "source_files": "stage17_a_efficientad100_vs_30_delta.csv",
            "status": "drafted",
        },
        {
            "paragraph_id": "P3-7",
            "section": "Restrictions",
            "purpose": "State result-writing constraints to prevent overclaiming.",
            "source_files": "stage16_f_final_claim_evidence_map.csv",
            "status": "drafted",
        },
    ]
    return pd.DataFrame(rows)


def make_table_map() -> pd.DataFrame:
    rows = [
        {
            "paper_table_id": "Table 1",
            "paper_table_title": "System-level strong baseline comparison",
            "source_file": "results/stage16_qcru_ablation/stage16_d_paper_facing_system_baseline_table.csv",
            "recommended_location": "Main Results",
            "paper_text_use": "Main system-level evidence. Use LOCO as fair result; same-set as upper-bound diagnostic only.",
        },
        {
            "paper_table_id": "Table 2",
            "paper_table_title": "Quality-Calibrated QCR ablation",
            "source_file": "results/stage16_qcru_ablation/stage16_d_paper_facing_qcr_ablation_table.csv",
            "recommended_location": "Ablation Study",
            "paper_text_use": "Show detector-only, crop VLM, naive fusion, quality calibration, fixed Q+C diagnostic, and adaptive refinement.",
        },
        {
            "paper_table_id": "Table 3",
            "paper_table_title": "Claim-ready deltas",
            "source_file": "results/stage16_qcru_ablation/stage16_d_paper_facing_claim_ready_deltas.csv",
            "recommended_location": "Analysis or appendix",
            "paper_text_use": "Use exact deltas in prose; may not need full table in main paper.",
        },
        {
            "paper_table_id": "Table 4",
            "paper_table_title": "Boundary and failure summary",
            "source_file": "results/stage16_qcru_ablation/stage16_e_category_boundary_summary.csv",
            "recommended_location": "Failure Analysis",
            "paper_text_use": "Show quality calibration is useful but not universal; fixed consistency remains diagnostic.",
        },
        {
            "paper_table_id": "Appendix Table A1",
            "paper_table_title": "EfficientAD-100 sensitivity",
            "source_file": "results/stage17_defensive_sensitivity/stage17_a_efficientad100_vs_30_delta.csv",
            "recommended_location": "Appendix or baseline-budget note",
            "paper_text_use": "Defensive baseline-budget sensitivity. Do not claim full EfficientAD defeat.",
        },
        {
            "paper_table_id": "Figure 1",
            "paper_table_title": "Method overview",
            "source_file": "to_be_drawn",
            "recommended_location": "Method",
            "paper_text_use": "Detector localization -> candidate crop -> VLM evidence -> quality calibration -> adaptive refinement.",
        },
        {
            "paper_table_id": "Figure 2",
            "paper_table_title": "Boundary cases",
            "source_file": "results/stage16_qcru_ablation/stage16_e_failure_boundary_case_inventory.csv",
            "recommended_location": "Failure Analysis",
            "paper_text_use": "Manual visual inspection required before choosing paper examples.",
        },
    ]
    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DOC_DIR.mkdir(parents=True, exist_ok=True)

    system = read_csv_strict(IN_SYSTEM)
    qcr = read_csv_strict(IN_QCR)
    deltas = read_csv_strict(IN_DELTAS)
    boundary_decision = read_csv_strict(IN_BOUNDARY_DECISION)
    boundary_category = read_csv_strict(IN_BOUNDARY_CATEGORY)
    cases = read_csv_strict(IN_CASES)
    e17 = read_csv_strict(IN_E17_DELTA)

    full_vlm = get_system_score(system, "full-image VLM")
    context_vlm = get_system_score(system, "context-aware VLM")
    winclip = get_system_score(system, "WinCLIP fixed protocol")
    efficientad30 = get_system_score(system, "EfficientAD-30 fixed-budget")
    patchcore = get_system_score(system, "PatchCore")
    loco = get_system_score(system, "PatchCore + context VLM, LOCO")
    same_set = get_system_score(system, "PatchCore + context VLM, same-set")

    d_loco_patch = get_delta(deltas, "LOCO fusion vs PatchCore")
    d_loco_eff = get_delta(deltas, "LOCO fusion vs EfficientAD-30")
    d_loco_win = get_delta(deltas, "LOCO fusion vs WinCLIP")
    d_context_full = get_delta(deltas, "context-aware VLM vs full-image VLM")
    d_quality_naive = get_delta(deltas, "Quality-Calibrated QCR vs naive fusion")
    d_adaptive_quality = get_delta(deltas, "Adaptive refinement vs Quality-Calibrated QCR")
    d_adaptive_naive = get_delta(deltas, "Adaptive refinement vs naive fusion")
    d_fixed_quality = get_delta(deltas, "Fixed Q+C vs Quality-Calibrated QCR")

    qcr_detector = qcr_mean(qcr, "Detector only")
    qcr_crop = qcr_mean(qcr, "Crop VLM only")
    qcr_naive = qcr_mean(qcr, "Naive detector-crop fusion")
    qcr_quality = qcr_mean(qcr, "Quality-Calibrated QCR")
    qcr_adaptive = qcr_mean(qcr, "adaptive consistency refinement")
    qcr_fixed = qcr_mean(qcr, "Fixed Q+C fusion")

    q_boundary = boundary_metric(boundary_category, "delta_v4_quality_minus_v3_naive")
    a_boundary = boundary_metric(boundary_category, "delta_v6_adaptive_minus_v4_quality")
    f_boundary = boundary_metric(boundary_category, "delta_v5_fixed_minus_v4_quality")

    e17_image = get_e17_metric(e17, "image_AUROC")
    e17_pixel = get_e17_metric(e17, "pixel_AUROC")

    case_counts = cases["case_type"].value_counts().to_dict()

    inventory = make_inventory()
    table_map = make_table_map()

    inventory.to_csv(OUT_PARAGRAPH_INVENTORY, index=False, lineterminator="\n")
    table_map.to_csv(OUT_TABLE_MAP, index=False, lineterminator="\n")

    lines = []
    lines += [
        "# Paper Stage P3: Experiments and Results Draft",
        "",
        "## 1. Experimental Setup Draft",
        "",
        "We evaluate industrial anomaly recognition under two complementary experimental views. "
        "First, we report a system-level strong baseline comparison over the primary categories, including full-image VLM, context-aware VLM, WinCLIP, PatchCore, EfficientAD-30 fixed-budget, and PatchCore+context fusion. "
        "Second, we evaluate the proposed Quality-Calibrated QCR family under the QCR primary protocol to isolate the effect of candidate quality calibration and adaptive consistency refinement.",
        "",
        "Image-level AUROC is the primary metric because the paper targets image-level anomaly recognition and candidate-level reasoning. "
        "Pixel-level quantities are treated as auxiliary localization evidence and are not used to claim segmentation SOTA. "
        "For protocol fairness, LOCO fusion is used as the fair system-level result, while same-set fusion is reported only as an upper-bound diagnostic.",
        "",
        "## 2. Baselines Draft",
        "",
        "The system-level baselines include three groups. "
        "The first group consists of VLM-based baselines: full-image VLM, context-aware VLM, and WinCLIP under the fixed protocol used in this study. "
        "The second group consists of detector baselines: PatchCore and EfficientAD-30 fixed-budget. "
        "The third group consists of localization-guided fusion variants, including PatchCore+context VLM under LOCO and same-set settings.",
        "",
        "EfficientAD is reported as a fixed-budget detector baseline rather than a fully optimized EfficientAD result. "
        "This distinction is important: the paper should not claim full EfficientAD defeat. "
        "Instead, EfficientAD-30 is used to test whether the proposed localization-guided VLM route remains competitive against a modern non-VLM detector under a controlled fixed budget.",
        "",
        "## 3. Main System-level Results Draft",
        "",
        f"Table 1 reports the system-level comparison. Full-image VLM reaches mean image AUROC `{fmt(full_vlm)}`, while context-aware VLM reaches `{fmt(context_vlm)}`, giving a localization/context gain of `{signed(d_context_full)}`. "
        f"The external WinCLIP fixed protocol obtains `{fmt(winclip)}`. Among detector baselines, PatchCore obtains `{fmt(patchcore)}` and EfficientAD-30 fixed-budget obtains `{fmt(efficientad30)}`.",
        "",
        f"The fair PatchCore+context VLM LOCO fusion reaches `{fmt(loco)}`, improving over PatchCore by `{signed(d_loco_patch)}` and over EfficientAD-30 fixed-budget by `{signed(d_loco_eff)}`. "
        f"The same-set fusion reaches `{fmt(same_set)}`, but this result is an upper-bound diagnostic and should not be used as the fair deployment claim. "
        "These results support the central system-level conclusion: localization-guided VLM evidence is complementary to detector evidence, but fair evaluation must distinguish LOCO from same-set fusion.",
        "",
        "Recommended wording:",
        "",
        "```text",
        "Under the fair LOCO protocol, localization-guided VLM fusion improves over the detector-only PatchCore baseline and the fixed-budget EfficientAD baseline. The same-set fusion result is reported only as a diagnostic upper bound.",
        "```",
        "",
        "## 4. QCR Ablation Results Draft",
        "",
        f"Table 2 reports the QCR primary-protocol ablation. Detector-only scoring obtains mean AUROC `{fmt(qcr_detector)}`, crop VLM only obtains `{fmt(qcr_crop)}`, and naive detector-crop fusion obtains `{fmt(qcr_naive)}`. "
        f"Quality-Calibrated QCR improves to `{fmt(qcr_quality)}`, corresponding to a mean AUROC gain of `{signed(d_quality_naive)}` over naive fusion.",
        "",
        f"The adaptive consistency refinement obtains `{fmt(qcr_adaptive)}`, improving over naive fusion by `{signed(d_adaptive_naive)}` but only improving over the quality-calibrated core by `{signed(d_adaptive_quality)}`. "
        f"Fixed Q+C fusion obtains `{fmt(qcr_fixed)}` in the primary protocol and has a primary-protocol delta of `{signed(d_fixed_quality)}` over quality-only, but it is not used as the final method because the robustness analysis showed that fixed consistency is not stable across protocols.",
        "",
        "The correct interpretation is therefore not that consistency is the main source of improvement. "
        "The main effective component is candidate quality calibration. "
        "Adaptive consistency is retained only as a conservative refinement that avoids overcommitting to fixed consistency.",
        "",
        "Recommended wording:",
        "",
        "```text",
        "Quality calibration provides the main ablation gain over naive detector-crop fusion. Adaptive consistency yields only a small additional refinement and should not be interpreted as the main source of improvement.",
        "```",
        "",
        "## 5. Failure and Boundary Analysis Draft",
        "",
        f"Table 4 summarizes boundary behavior. Per-category, quality calibration has mean V4-V3 AUROC delta `{signed(q_boundary['mean'])}` and wins `{q_boundary['wins']}/{q_boundary['total']}` cases. "
        f"Adaptive consistency has mean V6-V4 delta `{signed(a_boundary['mean'])}` and wins `{a_boundary['wins']}/{a_boundary['total']}` cases. "
        f"Fixed Q+C has mean V5-V4 delta `{signed(f_boundary['mean'])}` and is positive in `{f_boundary['wins']}/{f_boundary['total']}` cases, but it remains diagnostic only.",
        "",
        "The case inventory includes the following extracted case types:",
        "",
    ]

    for k, v in sorted(case_counts.items()):
        lines.append(f"- `{k}`: `{v}` cases")

    lines += [
        "",
        "These cases should be manually inspected before selecting paper figures. "
        "The intended qualitative examples are: quality helping anomaly boost, quality suppressing normal false positives, quality suppressing true anomalies as a boundary case, fixed consistency causing risky score changes, and detector-VLM disagreement.",
        "",
        "Recommended wording:",
        "",
        "```text",
        "Failure analysis shows that quality calibration is useful but not universal. It can fail when candidate quality is misleading or when detector and VLM evidence disagree. This motivates conservative claim boundaries and prevents treating fixed consistency as the final method.",
        "```",
        "",
        "## 6. EfficientAD-100 Sensitivity Draft",
        "",
        f"To check whether EfficientAD-30 severely underestimates EfficientAD, we ran a 100-epoch sensitivity check on fruit_jelly. "
        f"The image-AUROC delta from EfficientAD-30 to EfficientAD-100 is `{signed(e17_image['delta_100_minus_30'])}`. "
        f"The pixel-AUROC delta is `{signed(e17_pixel['delta_100_minus_30'])}`.",
        "",
        "This result supports the use of EfficientAD-30 as a fixed-budget image-level baseline in the current paper. "
        "However, because the sensitivity check is only on fruit_jelly, the paper should still avoid claiming full EfficientAD defeat. "
        "The pixel-AUROC improvement should be mentioned only as auxiliary because the paper does not claim pixel-level segmentation SOTA.",
        "",
        "Recommended wording:",
        "",
        "```text",
        "We additionally run a 100-epoch EfficientAD sensitivity check on fruit_jelly. The image-level result does not indicate severe underestimation of EfficientAD-30, so we retain EfficientAD-30 as a fixed-budget baseline while avoiding claims of full EfficientAD superiority.",
        "```",
        "",
        "## 7. Result-writing Restrictions",
        "",
        "The Experiments section must follow these restrictions:",
        "",
        "- Do not merge Panel A and Panel B into one global ranking.",
        "- Do not use same-set fusion as the fair system-level claim.",
        "- Do not call EfficientAD-30 a full-budget EfficientAD result.",
        "- Do not claim consistency is universally beneficial.",
        "- Do not describe adaptive consistency as the main performance source.",
        "- Do not claim pixel-level segmentation SOTA.",
        "- Do not claim manufacturing-cause reasoning.",
        "",
        "## 8. Table-to-text Map",
        "",
        "| Paper ID | Source | Location | Text Use |",
        "|---|---|---|---|",
    ]

    for _, r in table_map.iterrows():
        lines.append(
            f"| {r['paper_table_id']} | `{r['source_file']}` | {r['recommended_location']} | {r['paper_text_use']} |"
        )

    lines += [
        "",
        "## 9. Paragraph Inventory",
        "",
        "| Paragraph ID | Section | Purpose | Status |",
        "|---|---|---|---|",
    ]

    for _, r in inventory.iterrows():
        lines.append(
            f"| {r['paragraph_id']} | {r['section']} | {r['purpose']} | {r['status']} |"
        )

    lines += [
        "",
        "## 10. Next Step",
        "",
        "Next stage:",
        "",
        "```text",
        "Paper Stage P4: Related Work and positioning",
        "```",
        "",
        "P4 should position the paper against PatchCore/EfficientAD-style detectors, WinCLIP/CLIP anomaly baselines, and VLM reasoning work, while avoiding broad SOTA claims.",
        "",
        "## 11. Outputs",
        "",
        f"- `{OUT_DOC.relative_to(ROOT)}`",
        f"- `{OUT_PARAGRAPH_INVENTORY.relative_to(ROOT)}`",
        f"- `{OUT_TABLE_MAP.relative_to(ROOT)}`",
        "",
    ]

    OUT_DOC.write_text("\n".join(lines), encoding="utf-8", newline="\n")

    print("[DONE]", OUT_DOC)
    print("[DONE]", OUT_PARAGRAPH_INVENTORY)
    print("[DONE]", OUT_TABLE_MAP)
    print()
    print("===== paragraph inventory =====")
    print(inventory.to_string(index=False))
    print()
    print("===== table-to-text map =====")
    print(table_map.to_string(index=False))


if __name__ == "__main__":
    main()
