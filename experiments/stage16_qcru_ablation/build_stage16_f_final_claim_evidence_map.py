from __future__ import annotations

from pathlib import Path
import pandas as pd


ROOT = Path(".").resolve()

IN_SYSTEM = ROOT / "results/stage16_qcru_ablation/stage16_d_paper_facing_system_baseline_table.csv"
IN_QCR = ROOT / "results/stage16_qcru_ablation/stage16_d_paper_facing_qcr_ablation_table.csv"
IN_DELTAS = ROOT / "results/stage16_qcru_ablation/stage16_d_paper_facing_claim_ready_deltas.csv"
IN_CLAIMS = ROOT / "results/stage16_qcru_ablation/stage16_c_final_method_claims.csv"
IN_BOUNDARY = ROOT / "results/stage16_qcru_ablation/stage16_e_boundary_decision_summary.csv"
IN_CATEGORY = ROOT / "results/stage16_qcru_ablation/stage16_e_category_boundary_summary.csv"

OUT_DIR = ROOT / "results/stage16_qcru_ablation"
DOC_DIR = ROOT / "docs/stage16_qcru_ablation"

OUT_MAP = OUT_DIR / "stage16_f_final_claim_evidence_map.csv"
OUT_STATUS = OUT_DIR / "stage16_f_paper_claim_status.csv"
OUT_REJECTED = OUT_DIR / "stage16_f_rejected_or_forbidden_claims.csv"
OUT_DOC = DOC_DIR / "stage16_f_final_claim_evidence_map_report.md"


def read_csv_strict(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    if len(df.columns) <= 1:
        raise RuntimeError(f"{path} read as <=1 column. Repair local CSV formatting first.")
    return df


def safe_float(x, default=None):
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def fmt(x) -> str:
    x = safe_float(x, None)
    if x is None:
        return "NA"
    return f"{x:+.4f}"


def score_of(system: pd.DataFrame, method: str):
    r = system[system["method"] == method]
    if r.empty:
        return None
    return safe_float(r.iloc[0]["mean_image_auroc"])


def delta_row(deltas: pd.DataFrame, contains: str) -> dict:
    mask = deltas["comparison"].astype(str).str.contains(contains, regex=False, na=False)
    rows = deltas[mask]
    if rows.empty:
        return {}
    r = rows.iloc[0].to_dict()
    return r


def boundary_decision(boundary: pd.DataFrame, decision_id: str) -> dict:
    rows = boundary[boundary["decision_id"] == decision_id]
    if rows.empty:
        return {}
    return rows.iloc[0].to_dict()


def compute_category_stats(category: pd.DataFrame) -> dict:
    out = {}

    for col, name in [
        ("delta_v4_quality_minus_v3_naive", "quality_minus_naive"),
        ("delta_v6_adaptive_minus_v4_quality", "adaptive_minus_quality"),
        ("delta_v5_fixed_minus_v4_quality", "fixed_minus_quality"),
    ]:
        if col not in category.columns:
            continue
        s = pd.to_numeric(category[col], errors="coerce").dropna()
        out[name + "_mean"] = float(s.mean()) if len(s) else None
        out[name + "_wins"] = int((s > 0).sum()) if len(s) else 0
        out[name + "_total"] = int(len(s))

    return out


def make_claim_map(system: pd.DataFrame, deltas: pd.DataFrame, boundary: pd.DataFrame, category: pd.DataFrame) -> pd.DataFrame:
    loco = score_of(system, "PatchCore + context VLM, LOCO")
    same = score_of(system, "PatchCore + context VLM, same-set")
    patchcore = score_of(system, "PatchCore")
    ead30 = score_of(system, "EfficientAD-30 fixed-budget")
    winclip = score_of(system, "WinCLIP fixed protocol")
    full_vlm = score_of(system, "full-image VLM")
    context_vlm = score_of(system, "context-aware VLM")

    d_loco_patch = delta_row(deltas, "LOCO fusion vs PatchCore")
    d_loco_ead = delta_row(deltas, "LOCO fusion vs EfficientAD-30")
    d_loco_winclip = delta_row(deltas, "LOCO fusion vs WinCLIP")
    d_context_full = delta_row(deltas, "context-aware VLM vs full-image VLM")
    d_quality_naive = delta_row(deltas, "Quality-Calibrated QCR vs naive fusion")
    d_adaptive_quality = delta_row(deltas, "Adaptive refinement vs Quality-Calibrated QCR")
    d_adaptive_naive = delta_row(deltas, "Adaptive refinement vs naive fusion")
    d_fixed_quality = delta_row(deltas, "Fixed Q+C vs Quality-Calibrated QCR")

    e_quality = boundary_decision(boundary, "E1")
    e_adaptive = boundary_decision(boundary, "E2")
    e_fixed = boundary_decision(boundary, "E3")
    e_cases = boundary_decision(boundary, "E4")
    e_boundary = boundary_decision(boundary, "E5")

    cat_stats = compute_category_stats(category)

    rows = [
        {
            "claim_id": "P1",
            "claim_category": "problem_framing",
            "paper_claim": "Industrial anomaly VLM reasoning should be localization-guided rather than full-image only.",
            "allowed_wording": "We study localization-guided VLM anomaly recognition, where detector localization evidence is converted into candidate-level visual-language evidence.",
            "forbidden_wording": "We solve full industrial anomaly understanding with a general-purpose VLM.",
            "evidence_files": "stage16_d_paper_facing_system_baseline_table.csv; stage16_d_paper_facing_claim_ready_deltas.csv",
            "evidence_summary": (
                f"context-aware VLM AUROC={context_vlm}; full-image VLM AUROC={full_vlm}; "
                f"context minus full-image delta={fmt(d_context_full.get('delta', None))}."
            ),
            "support_level": "moderate",
            "paper_section": "Introduction; Method motivation; Experiments",
            "caveat": "Do not claim semantic understanding or manufacturing-cause reasoning.",
            "status": "use",
        },
        {
            "claim_id": "P2",
            "claim_category": "system_level_result",
            "paper_claim": "Localization-guided VLM evidence is complementary to detector baselines.",
            "allowed_wording": "The fair LOCO fusion improves over the detector-only PatchCore baseline and the fixed-budget EfficientAD baseline.",
            "forbidden_wording": "The method fully beats all detector baselines under all budgets.",
            "evidence_files": "stage16_d_paper_facing_system_baseline_table.csv; stage16_d_paper_facing_claim_ready_deltas.csv",
            "evidence_summary": (
                f"LOCO AUROC={loco}; PatchCore AUROC={patchcore}; EfficientAD-30 AUROC={ead30}; "
                f"LOCO-PatchCore={fmt(d_loco_patch.get('delta', None))}; "
                f"LOCO-EfficientAD30={fmt(d_loco_ead.get('delta', None))}."
            ),
            "support_level": "strong_but_protocol_limited",
            "paper_section": "Main Results",
            "caveat": "EfficientAD is fixed-budget; same-set fusion is upper-bound only.",
            "status": "use",
        },
        {
            "claim_id": "P3",
            "claim_category": "external_baseline",
            "paper_claim": "The proposed localization-guided route is stronger than the fixed WinCLIP protocol used in this study.",
            "allowed_wording": "Under our fixed protocol, LOCO fusion outperforms WinCLIP.",
            "forbidden_wording": "We comprehensively outperform all CLIP-based anomaly detection methods.",
            "evidence_files": "stage16_d_paper_facing_system_baseline_table.csv; stage16_d_paper_facing_claim_ready_deltas.csv",
            "evidence_summary": (
                f"LOCO AUROC={loco}; WinCLIP AUROC={winclip}; "
                f"delta={fmt(d_loco_winclip.get('delta', None))}."
            ),
            "support_level": "moderate",
            "paper_section": "Baselines",
            "caveat": "AnomalyCLIP is not yet included; avoid broad CLIP-family claims.",
            "status": "use_with_caution",
        },
        {
            "claim_id": "P4",
            "claim_category": "main_method_component",
            "paper_claim": "Candidate quality calibration is the main effective method component.",
            "allowed_wording": "Candidate quality calibration provides the main gain over naive detector-crop fusion.",
            "forbidden_wording": "Every category benefits from quality calibration.",
            "evidence_files": "stage16_d_paper_facing_claim_ready_deltas.csv; stage16_e_boundary_decision_summary.csv; stage16_e_category_boundary_summary.csv",
            "evidence_summary": (
                f"primary QCR quality-minus-naive delta={fmt(d_quality_naive.get('delta', None))}; "
                f"{e_quality.get('evidence', '')}; "
                f"per-category mean={fmt(cat_stats.get('quality_minus_naive_mean'))}, "
                f"wins={cat_stats.get('quality_minus_naive_wins')}/{cat_stats.get('quality_minus_naive_total')}."
            ),
            "support_level": "strong_as_core_but_not_universal",
            "paper_section": "Method; Ablation",
            "caveat": "Per-category wins are not universal; use boundary-aware wording.",
            "status": "use",
        },
        {
            "claim_id": "P5",
            "claim_category": "final_method_variant",
            "paper_claim": "Adaptive consistency is a conservative refinement, not the main source of improvement.",
            "allowed_wording": "Adaptive consistency slightly refines the quality-calibrated score while avoiding overcommitting to fixed consistency.",
            "forbidden_wording": "Adaptive consistency produces the main performance gain.",
            "evidence_files": "stage16_d_paper_facing_claim_ready_deltas.csv; stage16_e_boundary_decision_summary.csv; stage16_e_category_boundary_summary.csv",
            "evidence_summary": (
                f"primary adaptive-minus-quality delta={fmt(d_adaptive_quality.get('delta', None))}; "
                f"adaptive-minus-naive delta={fmt(d_adaptive_naive.get('delta', None))}; "
                f"{e_adaptive.get('evidence', '')}; "
                f"per-category mean={fmt(cat_stats.get('adaptive_minus_quality_mean'))}, "
                f"wins={cat_stats.get('adaptive_minus_quality_wins')}/{cat_stats.get('adaptive_minus_quality_total')}."
            ),
            "support_level": "weak_as_gain_strong_as_safety_caveat",
            "paper_section": "Ablation; Discussion",
            "caveat": "The gain over quality-only is very small.",
            "status": "use_with_caution",
        },
        {
            "claim_id": "P6",
            "claim_category": "diagnostic_component",
            "paper_claim": "Fixed Q+C fusion is diagnostic only and should not be the final method.",
            "allowed_wording": "Fixed consistency can peak in some settings but lacks robustness, so it is not used as the final method.",
            "forbidden_wording": "Fixed Q+C is the proposed final method.",
            "evidence_files": "stage16_d_paper_facing_claim_ready_deltas.csv; stage16_e_boundary_decision_summary.csv; stage16_e_category_boundary_summary.csv",
            "evidence_summary": (
                f"primary fixed-minus-quality delta={fmt(d_fixed_quality.get('delta', None))}; "
                f"{e_fixed.get('evidence', '')}; "
                f"per-category mean={fmt(cat_stats.get('fixed_minus_quality_mean'))}, "
                f"positive cases={cat_stats.get('fixed_minus_quality_wins')}/{cat_stats.get('fixed_minus_quality_total')}."
            ),
            "support_level": "strong_as_rejection",
            "paper_section": "Ablation; Discussion",
            "caveat": "Do not hide that fixed Q+C can be high in primary protocol; explain robustness tradeoff.",
            "status": "reject_as_final_method",
        },
        {
            "claim_id": "P7",
            "claim_category": "upper_bound",
            "paper_claim": "Same-set fusion is an upper-bound diagnostic, not a fair main result.",
            "allowed_wording": "Same-set fusion is reported only as a diagnostic upper bound.",
            "forbidden_wording": "Same-set fusion is the primary deployment result.",
            "evidence_files": "stage16_d_paper_facing_system_baseline_table.csv",
            "evidence_summary": f"same-set AUROC={same}; LOCO AUROC={loco}.",
            "support_level": "strong_as_protocol_boundary",
            "paper_section": "Main Results; Protocol",
            "caveat": "Use LOCO as the fair system-level claim.",
            "status": "use_as_diagnostic_only",
        },
        {
            "claim_id": "P8",
            "claim_category": "boundary_analysis",
            "paper_claim": "The method is a reliability-calibrated recognition framework, not a complete anomaly understanding system.",
            "allowed_wording": "Failure analysis shows boundaries from misleading localization, candidate quality errors, and detector-VLM disagreement.",
            "forbidden_wording": "The method explains manufacturing causes or solves all detector localization errors.",
            "evidence_files": "stage16_e_failure_boundary_case_inventory.csv; stage16_e_boundary_decision_summary.csv",
            "evidence_summary": (
                f"{e_cases.get('evidence', '')}; {e_boundary.get('evidence', '')}"
            ),
            "support_level": "strong_as_boundary_claim",
            "paper_section": "Failure Cases; Limitations",
            "caveat": "Representative images should be manually inspected before paper figures.",
            "status": "use",
        },
        {
            "claim_id": "P9",
            "claim_category": "segmentation_boundary",
            "paper_claim": "Do not claim pixel-level segmentation SOTA.",
            "allowed_wording": "Pixel-level/localization signals are used to generate candidate evidence for image-level anomaly recognition.",
            "forbidden_wording": "The method achieves pixel-level segmentation SOTA.",
            "evidence_files": "stage15_modern_detector_baselines; stage16_qcru_ablation",
            "evidence_summary": "The current method is evaluated and framed primarily for image-level anomaly recognition and candidate reasoning.",
            "support_level": "strong_as_restriction",
            "paper_section": "Limitations",
            "caveat": "Pixel metrics may be reported only as auxiliary detector evidence, not as the main claim.",
            "status": "reject",
        },
    ]

    return pd.DataFrame(rows)


def make_status_table(claim_map: pd.DataFrame) -> pd.DataFrame:
    rows = []

    groups = [
        ("main_claims_ready", claim_map[claim_map["status"].isin(["use", "use_with_caution"])]),
        ("claims_to_reject_or_downgrade", claim_map[claim_map["status"].isin(["reject", "reject_as_final_method", "use_as_diagnostic_only"])]),
    ]

    for group_name, g in groups:
        rows.append(
            {
                "status_group": group_name,
                "num_claims": len(g),
                "claim_ids": ";".join(g["claim_id"].astype(str).tolist()),
                "summary": "; ".join(g["paper_claim"].astype(str).tolist()),
            }
        )

    # Paper readiness flags.
    rows.extend(
        [
            {
                "status_group": "paper_ready_method_name",
                "num_claims": 1,
                "claim_ids": "P4;P5;P6",
                "summary": "Use Quality-Calibrated QCR as the method family; adaptive consistency is refinement; fixed Q+C is diagnostic only.",
            },
            {
                "status_group": "remaining_experiment_risks",
                "num_claims": 3,
                "claim_ids": "R1;R2;R3",
                "summary": "EfficientAD remains fixed-budget; AnomalyCLIP is absent; representative failure figures still need manual visual inspection.",
            },
            {
                "status_group": "next_actions",
                "num_claims": 2,
                "claim_ids": "N1;N2",
                "summary": "Run defensive EfficientAD-100 fruit_jelly sensitivity later; start paper outline/table-to-text drafting after claim map.",
            },
        ]
    )

    return pd.DataFrame(rows)


def make_rejected_claims(claim_map: pd.DataFrame) -> pd.DataFrame:
    rejected = claim_map[
        claim_map["status"].isin(["reject", "reject_as_final_method", "use_as_diagnostic_only"])
    ].copy()

    out = rejected[
        [
            "claim_id",
            "claim_category",
            "forbidden_wording",
            "allowed_wording",
            "evidence_summary",
            "caveat",
            "status",
        ]
    ].copy()

    return out


def write_report(claim_map: pd.DataFrame, status: pd.DataFrame, rejected: pd.DataFrame) -> None:
    lines = []
    lines += [
        "# Stage 16-F Final Claim-Evidence Map",
        "",
        "## 1. Purpose",
        "",
        "This stage maps every paper-facing claim to concrete experimental evidence and locks the forbidden claims.",
        "",
        "No new model is trained and no score is tuned in this stage.",
        "",
        "## 2. Final Method Naming",
        "",
        "Use this method family name:",
        "",
        "```text",
        "Quality-Calibrated QCR",
        "```",
        "",
        "Use this longer descriptive phrase when needed:",
        "",
        "```text",
        "Quality-Calibrated Localization-Guided VLM Reasoning",
        "```",
        "",
        "Use this only as the full variant name:",
        "",
        "```text",
        "Quality-Calibrated QCR with adaptive consistency refinement",
        "```",
        "",
        "Do not write the method as fixed Q+C QCR-U.",
        "",
        "## 3. Claim-Evidence Map",
        "",
        "| Claim ID | Category | Paper Claim | Support | Status | Section |",
        "|---|---|---|---|---|---|",
    ]

    for _, r in claim_map.iterrows():
        lines.append(
            f"| {r['claim_id']} | {r['claim_category']} | {r['paper_claim']} | "
            f"{r['support_level']} | {r['status']} | {r['paper_section']} |"
        )

    lines += [
        "",
        "## 4. Evidence Details",
        "",
        "| Claim ID | Evidence Summary | Caveat |",
        "|---|---|---|",
    ]

    for _, r in claim_map.iterrows():
        lines.append(
            f"| {r['claim_id']} | {r['evidence_summary']} | {r['caveat']} |"
        )

    lines += [
        "",
        "## 5. Rejected / Forbidden Claims",
        "",
        "| Claim ID | Forbidden Wording | Allowed Replacement |",
        "|---|---|---|",
    ]

    for _, r in rejected.iterrows():
        lines.append(
            f"| {r['claim_id']} | {r['forbidden_wording']} | {r['allowed_wording']} |"
        )

    lines += [
        "",
        "## 6. Paper Readiness Status",
        "",
        "| Status Group | Claim IDs | Summary |",
        "|---|---|---|",
    ]

    for _, r in status.iterrows():
        lines.append(
            f"| {r['status_group']} | {r['claim_ids']} | {r['summary']} |"
        )

    lines += [
        "",
        "## 7. Safe Abstract-level Wording",
        "",
        "A safe abstract-level claim is:",
        "",
        "```text",
        "We propose a quality-calibrated localization-guided VLM reasoning framework for industrial anomaly recognition. The framework converts detector localization evidence into candidate-level visual-language evidence and calibrates crop-level VLM scores using candidate quality. Experiments with strong detector and VLM baselines show that localization-guided VLM evidence complements detector scores, while ablations reveal that candidate quality is the main reliable component and consistency should be used only as a conservative adaptive refinement.",
        "```",
        "",
        "## 8. Remaining Risks Before Submission",
        "",
        "1. EfficientAD is still fixed-budget. Do not claim full EfficientAD defeat.",
        "2. AnomalyCLIP is not yet included. Avoid broad CLIP-family SOTA claims.",
        "3. Adaptive consistency gain is small. Do not present it as the main contribution.",
        "4. Failure-case examples should be visually inspected before choosing paper figures.",
        "5. The method is image-level anomaly recognition / candidate reasoning, not pixel-level segmentation SOTA.",
        "",
        "## 9. Next Step",
        "",
        "After this stage, the experimental evidence chain is mostly closed. The next practical step is either:",
        "",
        "```text",
        "Stage 17-A: EfficientAD-100 fruit_jelly sensitivity check",
        "```",
        "",
        "or:",
        "",
        "```text",
        "Paper Stage P1: draft paper outline from claim-evidence map",
        "```",
        "",
        "If the goal is submission defense, run EfficientAD-100 fruit_jelly first. If the goal is writing, start the paper outline.",
        "",
        "## 10. Outputs",
        "",
        f"- `{OUT_MAP.relative_to(ROOT)}`",
        f"- `{OUT_STATUS.relative_to(ROOT)}`",
        f"- `{OUT_REJECTED.relative_to(ROOT)}`",
        f"- `{OUT_DOC.relative_to(ROOT)}`",
        "",
    ]

    OUT_DOC.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DOC_DIR.mkdir(parents=True, exist_ok=True)

    system = read_csv_strict(IN_SYSTEM)
    _ = read_csv_strict(IN_QCR)
    deltas = read_csv_strict(IN_DELTAS)
    _ = read_csv_strict(IN_CLAIMS)
    boundary = read_csv_strict(IN_BOUNDARY)
    category = read_csv_strict(IN_CATEGORY)

    claim_map = make_claim_map(system, deltas, boundary, category)
    status = make_status_table(claim_map)
    rejected = make_rejected_claims(claim_map)

    claim_map.to_csv(OUT_MAP, index=False, lineterminator="\n")
    status.to_csv(OUT_STATUS, index=False, lineterminator="\n")
    rejected.to_csv(OUT_REJECTED, index=False, lineterminator="\n")

    write_report(claim_map, status, rejected)

    print("[DONE]", OUT_MAP)
    print("[DONE]", OUT_STATUS)
    print("[DONE]", OUT_REJECTED)
    print("[DONE]", OUT_DOC)
    print()
    print("===== claim map =====")
    print(claim_map[["claim_id", "claim_category", "support_level", "status", "paper_claim"]].to_string(index=False))
    print()
    print("===== rejected / forbidden =====")
    print(rejected[["claim_id", "forbidden_wording", "allowed_wording"]].to_string(index=False))
    print()
    print("===== status =====")
    print(status.to_string(index=False))


if __name__ == "__main__":
    main()
