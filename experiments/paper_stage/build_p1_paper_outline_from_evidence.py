from __future__ import annotations

from pathlib import Path
import pandas as pd


ROOT = Path(".").resolve()

IN_CLAIM_MAP = ROOT / "results/stage16_qcru_ablation/stage16_f_final_claim_evidence_map.csv"
IN_CLAIM_STATUS = ROOT / "results/stage16_qcru_ablation/stage16_f_paper_claim_status.csv"
IN_REJECTED = ROOT / "results/stage16_qcru_ablation/stage16_f_rejected_or_forbidden_claims.csv"

IN_SYSTEM = ROOT / "results/stage16_qcru_ablation/stage16_d_paper_facing_system_baseline_table.csv"
IN_QCR = ROOT / "results/stage16_qcru_ablation/stage16_d_paper_facing_qcr_ablation_table.csv"
IN_DELTAS = ROOT / "results/stage16_qcru_ablation/stage16_d_paper_facing_claim_ready_deltas.csv"

IN_E17_DELTA = ROOT / "results/stage17_defensive_sensitivity/stage17_a_efficientad100_vs_30_delta.csv"
IN_E17_REPORT = ROOT / "docs/stage17_defensive_sensitivity/stage17_a_efficientad100_fruit_jelly_sensitivity_report.md"

OUT_DIR = ROOT / "results/paper_p1"
DOC_DIR = ROOT / "docs/paper_p1"

OUT_TABLES = OUT_DIR / "paper_table_inventory.csv"
OUT_RISKS = OUT_DIR / "paper_remaining_risks.csv"
OUT_DOC = DOC_DIR / "paper_outline_quality_calibrated_qcr.md"


def read_csv_strict(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    if len(df.columns) <= 1:
        raise RuntimeError(f"{path} read as <=1 column. Repair local CSV formatting first.")
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
    r = system[system["method"] == method]
    if r.empty:
        return None
    return float(r.iloc[0]["mean_image_auroc"])


def get_delta(deltas: pd.DataFrame, text: str):
    r = deltas[deltas["comparison"].astype(str).str.contains(text, regex=False, na=False)]
    if r.empty:
        return None
    return r.iloc[0]


def get_stage17_metric(delta: pd.DataFrame, metric: str):
    r = delta[delta["metric"] == metric]
    if r.empty:
        return None
    return r.iloc[0]


def build_table_inventory() -> pd.DataFrame:
    rows = [
        {
            "table_id": "Table 1",
            "table_title": "System-level strong baseline comparison",
            "source_file": "results/stage16_qcru_ablation/stage16_d_paper_facing_system_baseline_table.csv",
            "paper_section": "Experiments / Main Results",
            "purpose": "Compare WinCLIP, full-image VLM, context-aware VLM, PatchCore, EfficientAD-30, and LOCO fusion.",
            "must_include": True,
            "notes": "Do not merge with QCR ablation table because protocols differ.",
        },
        {
            "table_id": "Table 2",
            "table_title": "QCR primary-protocol ablation",
            "source_file": "results/stage16_qcru_ablation/stage16_d_paper_facing_qcr_ablation_table.csv",
            "paper_section": "Experiments / Ablation",
            "purpose": "Show detector-only, crop VLM, naive fusion, Quality-Calibrated QCR, fixed Q+C diagnostic, and adaptive refinement.",
            "must_include": True,
            "notes": "Quality-Calibrated QCR is the main effective core; adaptive consistency is refinement.",
        },
        {
            "table_id": "Table 3",
            "table_title": "Claim-ready deltas",
            "source_file": "results/stage16_qcru_ablation/stage16_d_paper_facing_claim_ready_deltas.csv",
            "paper_section": "Experiments / Analysis",
            "purpose": "Report exact deltas supporting main claims.",
            "must_include": True,
            "notes": "Use for text evidence, not necessarily as a full paper table.",
        },
        {
            "table_id": "Table 4",
            "table_title": "Failure and boundary summary",
            "source_file": "results/stage16_qcru_ablation/stage16_e_category_boundary_summary.csv",
            "paper_section": "Failure Analysis / Limitations",
            "purpose": "Show quality calibration is useful but not universal, and fixed consistency is diagnostic only.",
            "must_include": True,
            "notes": "Important because method improvement is reliability-oriented, not huge SOTA margin.",
        },
        {
            "table_id": "Table 5",
            "table_title": "EfficientAD-100 fruit_jelly sensitivity",
            "source_file": "results/stage17_defensive_sensitivity/stage17_a_efficientad100_vs_30_delta.csv",
            "paper_section": "Appendix or Baseline Budget Sensitivity",
            "purpose": "Defend against the concern that EfficientAD-30 severely underestimates EfficientAD.",
            "must_include": False,
            "notes": "Use in appendix or footnote; do not claim full EfficientAD defeat.",
        },
        {
            "table_id": "Figure 1",
            "table_title": "Framework overview",
            "source_file": "to_be_drawn",
            "paper_section": "Method",
            "purpose": "Show detector localization -> candidate crop -> VLM evidence -> quality calibration -> adaptive refinement.",
            "must_include": True,
            "notes": "This should be a schematic, not a result table.",
        },
        {
            "table_id": "Figure 2",
            "table_title": "Representative boundary cases",
            "source_file": "results/stage16_qcru_ablation/stage16_e_failure_boundary_case_inventory.csv",
            "paper_section": "Failure Analysis",
            "purpose": "Visualize where quality helps, where quality misleads, and where detector-VLM disagreement occurs.",
            "must_include": True,
            "notes": "Images must be manually inspected before paper use.",
        },
    ]

    return pd.DataFrame(rows)


def build_remaining_risks() -> pd.DataFrame:
    rows = [
        {
            "risk_id": "R1",
            "risk": "EfficientAD is still not a full four-category 100-epoch baseline.",
            "severity": "medium",
            "mitigation": "Stage 17-A fruit_jelly sensitivity shows EfficientAD-100 does not improve image_AUROC over EfficientAD-30 on fruit_jelly. Keep EfficientAD as fixed-budget.",
            "paper_handling": "Label EfficientAD as EfficientAD-30 fixed-budget. Do not claim full EfficientAD defeat.",
        },
        {
            "risk_id": "R2",
            "risk": "AnomalyCLIP is not included.",
            "severity": "medium_high",
            "mitigation": "Avoid broad CLIP-family SOTA claims. Present WinCLIP as the fixed external VLM anomaly baseline used in this study.",
            "paper_handling": "Mention as future baseline extension or limitation.",
        },
        {
            "risk_id": "R3",
            "risk": "Adaptive consistency gain over quality-only is very small.",
            "severity": "medium",
            "mitigation": "Do not present adaptive consistency as the main innovation. The main method is quality calibration.",
            "paper_handling": "Write adaptive consistency as conservative refinement only.",
        },
        {
            "risk_id": "R4",
            "risk": "Quality calibration is not universally positive per category.",
            "severity": "medium",
            "mitigation": "Use Stage 16-E boundary analysis. Claim reliability calibration, not universal improvement.",
            "paper_handling": "Include failure/boundary section.",
        },
        {
            "risk_id": "R5",
            "risk": "Method may look heuristic because it uses score fusion.",
            "severity": "medium_high",
            "mitigation": "Emphasize fixed protocol, ablations, robustness checks, and boundary analysis.",
            "paper_handling": "Avoid overclaiming; frame as reliability calibration for localization-guided VLM evidence.",
        },
        {
            "risk_id": "R6",
            "risk": "Pixel-level claims are weak.",
            "severity": "high_if_overclaimed",
            "mitigation": "Do not claim segmentation SOTA.",
            "paper_handling": "Frame localization as candidate generation evidence, not final segmentation output.",
        },
    ]

    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DOC_DIR.mkdir(parents=True, exist_ok=True)

    claim_map = read_csv_strict(IN_CLAIM_MAP)
    claim_status = read_csv_strict(IN_CLAIM_STATUS)
    rejected = read_csv_strict(IN_REJECTED)
    system = read_csv_strict(IN_SYSTEM)
    qcr = read_csv_strict(IN_QCR)
    deltas = read_csv_strict(IN_DELTAS)
    e17 = read_csv_strict(IN_E17_DELTA)

    table_inventory = build_table_inventory()
    risks = build_remaining_risks()

    table_inventory.to_csv(OUT_TABLES, index=False, lineterminator="\n")
    risks.to_csv(OUT_RISKS, index=False, lineterminator="\n")

    loco = get_system_score(system, "PatchCore + context VLM, LOCO")
    same = get_system_score(system, "PatchCore + context VLM, same-set")
    patchcore = get_system_score(system, "PatchCore")
    efficientad30 = get_system_score(system, "EfficientAD-30 fixed-budget")
    winclip = get_system_score(system, "WinCLIP fixed protocol")
    full_vlm = get_system_score(system, "full-image VLM")
    context_vlm = get_system_score(system, "context-aware VLM")

    d_loco_patch = get_delta(deltas, "LOCO fusion vs PatchCore")
    d_loco_eff = get_delta(deltas, "LOCO fusion vs EfficientAD-30")
    d_quality_naive = get_delta(deltas, "Quality-Calibrated QCR vs naive fusion")
    d_adaptive_quality = get_delta(deltas, "Adaptive refinement vs Quality-Calibrated QCR")
    d_adaptive_naive = get_delta(deltas, "Adaptive refinement vs naive fusion")

    e17_image = get_stage17_metric(e17, "image_AUROC")
    e17_pixel = get_stage17_metric(e17, "pixel_AUROC")

    lines = []
    lines += [
        "# Paper Stage P1: Paper Outline from Evidence",
        "",
        "## 1. Working Title Options",
        "",
        "Preferred title:",
        "",
        "```text",
        "Quality-Calibrated Localization-Guided VLM Reasoning for Industrial Anomaly Recognition",
        "```",
        "",
        "Shorter method-oriented title:",
        "",
        "```text",
        "Quality-Calibrated QCR for Localization-Guided VLM Anomaly Recognition",
        "```",
        "",
        "Avoid titles centered on fixed QCR-U or universal consistency.",
        "",
        "## 2. Core Thesis",
        "",
        "The paper should argue:",
        "",
        "```text",
        "Industrial anomaly VLM reasoning is more reliable when it is guided by detector localization and calibrated by candidate quality. The main effective mechanism is quality calibration of crop-level VLM evidence. Detector-VLM consistency is useful only as a conservative adaptive refinement, not as the primary source of improvement.",
        "```",
        "",
        "## 3. Main Contributions",
        "",
        "### Contribution 1: Localization-guided VLM anomaly recognition",
        "",
        "The method converts anomaly localization evidence into candidate-level VLM evidence instead of relying on full-image VLM inference.",
        "",
        f"Evidence: context-aware VLM mean AUROC `{fmt(context_vlm)}` vs full-image VLM `{fmt(full_vlm)}`.",
        "",
        "### Contribution 2: Quality-calibrated candidate reasoning",
        "",
        "The main method core calibrates crop-level VLM evidence using candidate quality derived from localization evidence.",
        "",
        f"Evidence: Quality-Calibrated QCR vs naive fusion delta `{signed(d_quality_naive['delta']) if d_quality_naive is not None else 'NA'}` in the QCR primary protocol.",
        "",
        "### Contribution 3: Boundary-aware adaptive consistency",
        "",
        "The paper analyzes detector-VLM consistency and shows fixed consistency is not robust enough to be the final method. Adaptive consistency is retained only as a small conservative refinement.",
        "",
        f"Evidence: adaptive refinement vs quality core delta `{signed(d_adaptive_quality['delta']) if d_adaptive_quality is not None else 'NA'}`.",
        "",
        "### Contribution 4: Strong baseline and claim discipline",
        "",
        "The paper includes WinCLIP, EfficientAD-30 fixed-budget, PatchCore, VLM baselines, LOCO fusion, ablations, and failure/boundary analysis.",
        "",
        f"Evidence: LOCO fusion `{fmt(loco)}` vs PatchCore `{fmt(patchcore)}` and EfficientAD-30 `{fmt(efficientad30)}`.",
        "",
        "## 4. Abstract Skeleton",
        "",
        "```text",
        "Industrial anomaly recognition with general-purpose vision-language models remains unreliable when images are evaluated globally. We study a localization-guided formulation that converts detector localization evidence into candidate-level visual-language evidence. To make crop-level VLM scores reliable, we propose Quality-Calibrated QCR, which calibrates VLM anomaly evidence using candidate quality derived from anomaly localization. We further analyze detector-VLM consistency and find that fixed consistency is not robust; therefore, consistency is used only as a conservative adaptive refinement. Experiments with detector, VLM, and external anomaly baselines show that localization-guided VLM evidence complements detector scores, while ablations identify candidate quality as the main effective component. Failure analysis clarifies the method boundaries under misleading localization and detector-VLM disagreement.",
        "```",
        "",
        "## 5. Proposed Paper Structure",
        "",
        "### 1. Introduction",
        "",
        "- Motivate industrial anomaly recognition.",
        "- Explain why full-image VLM is weak for localized industrial defects.",
        "- Introduce localization-guided VLM evidence.",
        "- State that naive fusion is insufficient and requires reliability calibration.",
        "- Present Quality-Calibrated QCR as the method family.",
        "",
        "### 2. Related Work",
        "",
        "- Industrial anomaly detection: PatchCore, EfficientAD, FastFlow-type detectors.",
        "- Vision-language anomaly detection: WinCLIP and related CLIP/VLM anomaly baselines.",
        "- Localization-guided reasoning and candidate-based evidence.",
        "- Calibration/reliability in multimodal scoring.",
        "",
        "### 3. Method",
        "",
        "- Detector localization and candidate extraction.",
        "- Crop-level VLM anomaly evidence.",
        "- Candidate quality score.",
        "- Quality-Calibrated QCR formula.",
        "- Adaptive consistency refinement.",
        "- Explicitly state fixed Q+C is diagnostic, not final.",
        "",
        "### 4. Experimental Setup",
        "",
        "- Datasets and primary categories.",
        "- Baselines: full-image VLM, context-aware VLM, WinCLIP, PatchCore, EfficientAD-30 fixed-budget.",
        "- Protocols: LOCO vs same-set; QCR primary protocol.",
        "- Metrics: image AUROC as main metric; pixel metrics auxiliary only.",
        "",
        "### 5. Main Results",
        "",
        "- Panel A: system-level baseline comparison.",
        "- Use LOCO as fair system-level result.",
        "- Use same-set only as diagnostic upper bound.",
        "",
        "### 6. Ablation Study",
        "",
        "- Detector only.",
        "- Crop VLM only.",
        "- Naive fusion.",
        "- Quality-Calibrated QCR.",
        "- Fixed Q+C diagnostic.",
        "- Adaptive consistency refinement.",
        "",
        "### 7. Failure and Boundary Analysis",
        "",
        "- Quality helps anomaly boost.",
        "- Quality helps normal suppression.",
        "- Quality can suppress true anomalies when candidate quality is misleading.",
        "- Fixed consistency can mislead.",
        "- Detector-VLM disagreement cases.",
        "",
        "### 8. Limitations",
        "",
        "- No full-budget EfficientAD sweep.",
        "- No AnomalyCLIP yet.",
        "- No pixel-level SOTA claim.",
        "- No manufacturing-cause reasoning claim.",
        "- Adaptive consistency has small gain over quality-only.",
        "",
        "## 6. Paper Tables and Figures",
        "",
        "| ID | Title | Source | Must Include | Notes |",
        "|---|---|---|---:|---|",
    ]

    for _, r in table_inventory.iterrows():
        lines.append(
            f"| {r['table_id']} | {r['table_title']} | `{r['source_file']}` | "
            f"{int(bool(r['must_include']))} | {r['notes']} |"
        )

    lines += [
        "",
        "## 7. Key Numbers to Use",
        "",
        "### System-level results",
        "",
        f"- full-image VLM: `{fmt(full_vlm)}`",
        f"- context-aware VLM: `{fmt(context_vlm)}`",
        f"- WinCLIP fixed protocol: `{fmt(winclip)}`",
        f"- EfficientAD-30 fixed-budget: `{fmt(efficientad30)}`",
        f"- PatchCore: `{fmt(patchcore)}`",
        f"- PatchCore + context VLM, LOCO: `{fmt(loco)}`",
        f"- PatchCore + context VLM, same-set upper bound: `{fmt(same)}`",
        "",
        "### QCR method results",
        "",
        f"- Quality-Calibrated QCR vs naive fusion: `{signed(d_quality_naive['delta']) if d_quality_naive is not None else 'NA'}`",
        f"- adaptive refinement vs quality core: `{signed(d_adaptive_quality['delta']) if d_adaptive_quality is not None else 'NA'}`",
        f"- adaptive refinement vs naive fusion: `{signed(d_adaptive_naive['delta']) if d_adaptive_naive is not None else 'NA'}`",
        "",
        "### EfficientAD-100 sensitivity",
        "",
        f"- EfficientAD-100 minus EfficientAD-30 image_AUROC on fruit_jelly: `{signed(e17_image['delta_100_minus_30']) if e17_image is not None else 'NA'}`",
        f"- EfficientAD-100 minus EfficientAD-30 pixel_AUROC on fruit_jelly: `{signed(e17_pixel['delta_100_minus_30']) if e17_pixel is not None else 'NA'}`",
        "",
        "Interpretation: the image-level EfficientAD-100 check does not show severe underestimation of EfficientAD-30. Pixel improvement is auxiliary and should not become the main claim.",
        "",
        "## 8. Forbidden Claims",
        "",
        "| Claim ID | Forbidden | Allowed Replacement |",
        "|---|---|---|",
    ]

    for _, r in rejected.iterrows():
        lines.append(
            f"| {r['claim_id']} | {r['forbidden_wording']} | {r['allowed_wording']} |"
        )

    lines += [
        "",
        "## 9. Remaining Risks",
        "",
        "| Risk ID | Risk | Severity | Mitigation | Paper Handling |",
        "|---|---|---|---|---|",
    ]

    for _, r in risks.iterrows():
        lines.append(
            f"| {r['risk_id']} | {r['risk']} | {r['severity']} | {r['mitigation']} | {r['paper_handling']} |"
        )

    lines += [
        "",
        "## 10. Next Writing Step",
        "",
        "Next stage should be:",
        "",
        "```text",
        "Paper Stage P2: draft Introduction + Contributions + Method Overview",
        "```",
        "",
        "Do not start by writing the full paper. First draft the Introduction and Method Overview using the locked claims above.",
        "",
        "## 11. Outputs",
        "",
        f"- `{OUT_DOC.relative_to(ROOT)}`",
        f"- `{OUT_TABLES.relative_to(ROOT)}`",
        f"- `{OUT_RISKS.relative_to(ROOT)}`",
        "",
    ]

    OUT_DOC.write_text("\n".join(lines), encoding="utf-8", newline="\n")

    print("[DONE]", OUT_DOC)
    print("[DONE]", OUT_TABLES)
    print("[DONE]", OUT_RISKS)
    print()
    print("===== table inventory =====")
    print(table_inventory.to_string(index=False))
    print()
    print("===== remaining risks =====")
    print(risks.to_string(index=False))


if __name__ == "__main__":
    main()
