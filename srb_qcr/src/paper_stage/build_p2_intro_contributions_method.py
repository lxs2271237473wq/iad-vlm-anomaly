from __future__ import annotations

from pathlib import Path
import pandas as pd


ROOT = Path(".").resolve()

IN_CLAIM_MAP = ROOT / "results/stage16_qcru_ablation/stage16_f_final_claim_evidence_map.csv"
IN_SYSTEM = ROOT / "results/stage16_qcru_ablation/stage16_d_paper_facing_system_baseline_table.csv"
IN_QCR = ROOT / "results/stage16_qcru_ablation/stage16_d_paper_facing_qcr_ablation_table.csv"
IN_DELTAS = ROOT / "results/stage16_qcru_ablation/stage16_d_paper_facing_claim_ready_deltas.csv"
IN_RISKS = ROOT / "results/paper_p1/paper_remaining_risks.csv"

OUT_DIR = ROOT / "results/paper_p2"
DOC_DIR = ROOT / "docs/paper_p2"

OUT_SECTION_INVENTORY = OUT_DIR / "paper_p2_section_inventory.csv"
OUT_CLAIM_USAGE = OUT_DIR / "paper_p2_claim_usage_map.csv"
OUT_DOC = DOC_DIR / "paper_p2_intro_contributions_method_overview.md"


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


def get_delta(deltas: pd.DataFrame, contains: str):
    r = deltas[deltas["comparison"].astype(str).str.contains(contains, regex=False, na=False)]
    if r.empty:
        return None
    return float(r.iloc[0]["delta"])


def get_qcr_score(qcr: pd.DataFrame, method_contains: str, backbone: str | None = None):
    mask = qcr["method"].astype(str).str.contains(method_contains, regex=False, na=False)
    if backbone is not None:
        mask &= qcr["backbone"].astype(str) == backbone
    rows = qcr[mask]
    if rows.empty:
        return None
    return float(rows["image_auroc"].mean())


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DOC_DIR.mkdir(parents=True, exist_ok=True)

    claim_map = read_csv_strict(IN_CLAIM_MAP)
    system = read_csv_strict(IN_SYSTEM)
    qcr = read_csv_strict(IN_QCR)
    deltas = read_csv_strict(IN_DELTAS)
    risks = read_csv_strict(IN_RISKS)

    full_vlm = get_system_score(system, "full-image VLM")
    context_vlm = get_system_score(system, "context-aware VLM")
    winclip = get_system_score(system, "WinCLIP fixed protocol")
    efficientad30 = get_system_score(system, "EfficientAD-30 fixed-budget")
    patchcore = get_system_score(system, "PatchCore")
    loco = get_system_score(system, "PatchCore + context VLM, LOCO")
    same_set = get_system_score(system, "PatchCore + context VLM, same-set")

    delta_loco_patch = get_delta(deltas, "LOCO fusion vs PatchCore")
    delta_loco_ead = get_delta(deltas, "LOCO fusion vs EfficientAD-30")
    delta_quality_naive = get_delta(deltas, "Quality-Calibrated QCR vs naive fusion")
    delta_adaptive_quality = get_delta(deltas, "Adaptive refinement vs Quality-Calibrated QCR")
    delta_adaptive_naive = get_delta(deltas, "Adaptive refinement vs naive fusion")

    qcr_quality = get_qcr_score(qcr, "Quality-Calibrated QCR")
    qcr_adaptive = get_qcr_score(qcr, "adaptive consistency refinement")
    qcr_naive = get_qcr_score(qcr, "Naive detector-crop fusion")

    section_rows = [
        {
            "section_id": "P2-S1",
            "section": "Introduction",
            "purpose": "Motivate localization-guided VLM anomaly recognition and reliability calibration.",
            "status": "drafted",
        },
        {
            "section_id": "P2-S2",
            "section": "Contributions",
            "purpose": "State conservative contribution bullets aligned with Stage 16-F claim-evidence map.",
            "status": "drafted",
        },
        {
            "section_id": "P2-S3",
            "section": "Method Overview",
            "purpose": "Describe detector localization, candidate VLM evidence, quality calibration, and adaptive refinement.",
            "status": "drafted",
        },
        {
            "section_id": "P2-S4",
            "section": "Notation and Scoring",
            "purpose": "Define D, M, Q, K, S_quality, and S_adaptive.",
            "status": "drafted",
        },
        {
            "section_id": "P2-S5",
            "section": "Claim Boundaries",
            "purpose": "Explicitly prevent overclaims around consistency, segmentation, and manufacturing-cause reasoning.",
            "status": "drafted",
        },
    ]

    pd.DataFrame(section_rows).to_csv(OUT_SECTION_INVENTORY, index=False, lineterminator="\n")

    usage_rows = []
    for _, r in claim_map.iterrows():
        cid = str(r["claim_id"])
        usage_rows.append(
            {
                "claim_id": cid,
                "paper_claim": r["paper_claim"],
                "used_in_p2_section": (
                    "Introduction" if cid in ["P1", "P2", "P3"]
                    else "Contributions" if cid in ["P4", "P5", "P6", "P7"]
                    else "Claim Boundaries"
                ),
                "status": r["status"],
                "support_level": r["support_level"],
            }
        )
    pd.DataFrame(usage_rows).to_csv(OUT_CLAIM_USAGE, index=False, lineterminator="\n")

    lines = []
    lines += [
        "# Paper Stage P2: Introduction, Contributions, and Method Overview Draft",
        "",
        "## 1. Working Title",
        "",
        "```text",
        "Quality-Calibrated Localization-Guided VLM Reasoning for Industrial Anomaly Recognition",
        "```",
        "",
        "Short method name:",
        "",
        "```text",
        "Quality-Calibrated QCR",
        "```",
        "",
        "Full variant name when adaptive consistency is mentioned:",
        "",
        "```text",
        "Quality-Calibrated QCR with Adaptive Consistency Refinement",
        "```",
        "",
        "---",
        "",
        "## 2. Introduction Draft",
        "",
        "Industrial anomaly recognition requires identifying subtle, localized deviations from normal product appearance. "
        "Recent vision-language models provide broad visual reasoning ability, but applying them directly to full industrial images is often unreliable because defects may occupy only a small region and may not dominate the global image semantics. "
        "In our experiments, full-image VLM inference remains weak compared with localization-guided variants, motivating a formulation that first converts anomaly localization evidence into candidate-level visual-language evidence.",
        "",
        "Classical anomaly detectors such as PatchCore, FastFlow-style backbones, and EfficientAD provide useful localization or anomaly evidence, but their scores are not designed to express visual-language-level abnormality. "
        "Conversely, a VLM can compare localized visual evidence against textual abnormality prompts, but it is easily misled when the candidate crop is poorly localized or visually ambiguous. "
        "This creates a practical gap: detector localization and VLM reasoning are complementary, but naive score fusion does not explicitly model whether the crop-level score fusion does not explicitly model whether the crop-level VLM evidence should be trusted.",
        "",
        "We address this gap with a quality-calibrated localization-guided VLM reasoning framework. "
        "The framework uses detector localization to generate candidate regions, obtains crop-level VLM anomaly evidence, and calibrates that evidence using candidate quality. "
        "The resulting method, Quality-Calibrated QCR, treats candidate quality as the main reliability mechanism. "
        "We further analyze detector-VLM consistency and find that fixed consistency is not robust enough to serve as the final method; therefore, consistency is retained only as a conservative adaptive refinement.",
        "",
        "Empirically, the system-level comparison shows that localization-guided VLM evidence complements detector baselines. "
        f"The fair LOCO fusion reaches mean image AUROC `{fmt(loco)}`, compared with PatchCore `{fmt(patchcore)}` and EfficientAD-30 fixed-budget `{fmt(efficientad30)}`. "
        f"In the QCR primary protocol, Quality-Calibrated QCR improves over naive detector-crop fusion by `{signed(delta_quality_naive)}` AUROC, while adaptive consistency adds only `{signed(delta_adaptive_quality)}` over the quality-calibrated core. "
        "These results support a conservative conclusion: the main effective component is candidate quality calibration, while adaptive consistency is a refinement rather than the main source of improvement.",
        "",
        "---",
        "",
        "## 3. Contribution Draft",
        "",
        "The paper should state the contributions as follows:",
        "",
        "1. **Localization-guided VLM anomaly recognition.** "
        "We formulate industrial anomaly recognition as a localization-guided visual-language reasoning problem, where detector localization evidence is converted into candidate-level VLM evidence rather than relying on full-image VLM inference.",
        "",
        "2. **Quality-calibrated candidate reasoning.** "
        "We propose Quality-Calibrated QCR, which calibrates crop-level VLM anomaly evidence using candidate quality derived from anomaly localization. "
        "This is the main effective method component and provides the primary gain over naive detector-crop fusion.",
        "",
        "3. **Boundary-aware consistency refinement.** "
        "We analyze detector-VLM consistency and show that fixed consistency can be unstable. "
        "Instead of using fixed Q+C fusion as the final method, we retain consistency only as a reliability-gated adaptive refinement.",
        "",
        "4. **Strong baseline and claim-disciplined evaluation.** "
        "We compare against full-image VLM, context-aware VLM, WinCLIP, PatchCore, EfficientAD-30 fixed-budget, LOCO fusion, and QCR ablations. "
        "We also include boundary analysis and explicitly reject unsupported claims such as pixel-level segmentation SOTA or manufacturing-cause reasoning.",
        "",
        "---",
        "",
        "## 4. Method Overview Draft",
        "",
        "### 4.1 Localization-guided candidate generation",
        "",
        "Given an input industrial image, an anomaly detector produces localization evidence and an image-level anomaly score. "
        "The localization evidence is used to generate candidate regions that are likely to contain abnormal visual patterns. "
        "This candidate-based design reduces the burden on the VLM: instead of interpreting the full image globally, the VLM evaluates localized evidence that is selected by the detector.",
        "",
        "### 4.2 Crop-level VLM anomaly evidence",
        "",
        "For each candidate crop, the VLM produces an abnormality score based on localized visual-language comparison. "
        "The crop-level score is denoted as `M`. "
        "This score is useful but not sufficient, because a crop may be poorly localized, too broad, too small, or visually misleading. "
        "Therefore, crop-level VLM evidence must be calibrated before it is fused with detector evidence.",
        "",
        "### 4.3 Candidate quality calibration",
        "",
        "We define candidate quality, denoted as `Q`, to measure whether the selected crop is a reliable carrier of anomaly evidence. "
        "The detector image-level anomaly score is denoted as `D`, and the crop-level VLM abnormality score is denoted as `M`. "
        "Naive fusion uses:",
        "",
        "```text",
        "S_naive = 0.5D + 0.5M",
        "```",
        "",
        "Quality-Calibrated QCR instead uses candidate quality to modulate the VLM term:",
        "",
        "```text",
        "S_quality = 0.5D + 0.5 * M * (0.5 + 0.5Q)",
        "```",
        "",
        "This design reflects the core assumption of the method: crop-level VLM evidence should contribute more when the candidate region is reliable, and less when the candidate region is uncertain.",
        "",
        "### 4.4 Adaptive consistency refinement",
        "",
        "We also examine detector-VLM consistency. "
        "Let `K` denote a high-high consistency signal between detector and VLM evidence. "
        "Fixed consistency fusion is diagnostic only because it can produce high scores in some settings but is not robust across protocols. "
        "Therefore, the final refinement uses a conservative adaptive gate:",
        "",
        "```text",
        "agreement = 1 - |D - M|",
        "mutual_anomaly_evidence = min(D, M)",
        "gate = Q * K * agreement * mutual_anomaly_evidence",
        "S_adaptive = S_quality + 0.05 * gate",
        "```",
        "",
        "The adaptive term is intentionally small. "
        "It is not presented as the main source of improvement; its role is to preserve the quality-calibrated core while adding consistency only when detector evidence, VLM evidence, candidate quality, and agreement are jointly reliable.",
        "",
        "### 4.5 Final scoring interpretation",
        "",
        "The final method family is Quality-Calibrated QCR. "
        "When the adaptive refinement is included, the full method can be described as Quality-Calibrated QCR with adaptive consistency refinement. "
        "However, the main paper claim should remain centered on candidate quality calibration, not on consistency.",
        "",
        "---",
        "",
        "## 5. Evidence Anchors for Writing",
        "",
        "Use these numbers in the paper text:",
        "",
        "### System-level evidence",
        "",
        f"- full-image VLM mean AUROC: `{fmt(full_vlm)}`",
        f"- context-aware VLM mean AUROC: `{fmt(context_vlm)}`",
        f"- WinCLIP fixed protocol mean AUROC: `{fmt(winclip)}`",
        f"- EfficientAD-30 fixed-budget mean AUROC: `{fmt(efficientad30)}`",
        f"- PatchCore mean AUROC: `{fmt(patchcore)}`",
        f"- PatchCore + context VLM, LOCO mean AUROC: `{fmt(loco)}`",
        f"- PatchCore + context VLM, same-set upper bound: `{fmt(same_set)}`",
        f"- LOCO minus PatchCore: `{signed(delta_loco_patch)}`",
        f"- LOCO minus EfficientAD-30 fixed-budget: `{signed(delta_loco_ead)}`",
        "",
        "### QCR evidence",
        "",
        f"- naive detector-crop fusion mean primary AUROC: `{fmt(qcr_naive)}`",
        f"- Quality-Calibrated QCR mean primary AUROC: `{fmt(qcr_quality)}`",
        f"- adaptive refinement mean primary AUROC: `{fmt(qcr_adaptive)}`",
        f"- Quality-Calibrated QCR minus naive fusion: `{signed(delta_quality_naive)}`",
        f"- adaptive refinement minus quality core: `{signed(delta_adaptive_quality)}`",
        f"- adaptive refinement minus naive fusion: `{signed(delta_adaptive_naive)}`",
        "",
        "---",
        "",
        "## 6. Required Claim Boundaries",
        "",
        "The following boundaries must be preserved throughout the paper:",
        "",
        "- Do not claim fixed Q+C fusion is the final method.",
        "- Do not claim consistency is universally beneficial.",
        "- Do not claim adaptive consistency is the main source of improvement.",
        "- Do not claim pixel-level segmentation SOTA.",
        "- Do not claim manufacturing-cause reasoning.",
        "- Do not claim full EfficientAD defeat; EfficientAD is reported as a 30-epoch fixed-budget baseline.",
        "- Do not merge same-set upper bound with LOCO fair deployment results.",
        "",
        "---",
        "",
        "## 7. Risk-aware Writing Notes",
        "",
    ]

    for _, r in risks.iterrows():
        lines.append(
            f"- `{r['risk_id']}`: {r['risk']} Mitigation: {r['mitigation']}"
        )

    lines += [
        "",
        "---",
        "",
        "## 8. Next Step",
        "",
        "Next stage should be:",
        "",
        "```text",
        "Paper Stage P3: draft Experiments section and paper-facing result text",
        "```",
        "",
        "P3 should turn Stage 16-D and Stage 16-E tables into paper-ready experimental paragraphs.",
        "",
        "## 9. Outputs",
        "",
        f"- `{OUT_DOC.relative_to(ROOT)}`",
        f"- `{OUT_SECTION_INVENTORY.relative_to(ROOT)}`",
        f"- `{OUT_CLAIM_USAGE.relative_to(ROOT)}`",
        "",
    ]

    OUT_DOC.write_text("\n".join(lines), encoding="utf-8", newline="\n")

    print("[DONE]", OUT_DOC)
    print("[DONE]", OUT_SECTION_INVENTORY)
    print("[DONE]", OUT_CLAIM_USAGE)


if __name__ == "__main__":
    main()
