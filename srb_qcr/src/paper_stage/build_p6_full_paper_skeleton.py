from __future__ import annotations

from pathlib import Path
import re
import pandas as pd


ROOT = Path(".").resolve()

IN_P1_OUTLINE = ROOT / "docs/paper_p1/paper_outline_quality_calibrated_qcr.md"
IN_P2 = ROOT / "docs/paper_p2/paper_p2_intro_contributions_method_overview.md"
IN_P3 = ROOT / "docs/paper_p3/paper_p3_experiments_and_results_draft.md"
IN_P4 = ROOT / "docs/paper_p4/paper_p4_related_work_and_positioning.md"
IN_P5 = ROOT / "docs/paper_p5/paper_p5_method_full_draft.md"

IN_TABLE_INV = ROOT / "results/paper_p1/paper_table_inventory.csv"
IN_RISKS = ROOT / "results/paper_p1/paper_remaining_risks.csv"
IN_CLAIM_MAP = ROOT / "results/stage16_qcru_ablation/stage16_f_final_claim_evidence_map.csv"
IN_REJECTED = ROOT / "results/stage16_qcru_ablation/stage16_f_rejected_or_forbidden_claims.csv"

OUT_DIR = ROOT / "results/paper_p6"
DOC_DIR = ROOT / "docs/paper_p6"

OUT_DOC = DOC_DIR / "paper_p6_full_paper_skeleton.md"
OUT_SECTION_INV = OUT_DIR / "paper_p6_section_assembly_inventory.csv"
OUT_CHECKLIST = OUT_DIR / "paper_p6_missing_items_checklist.csv"
OUT_READINESS = OUT_DIR / "paper_p6_submission_readiness.csv"


def read_text(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(path)
    return path.read_text(encoding="utf-8")


def read_csv_optional(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if len(df.columns) <= 1:
        raise RuntimeError(f"{path} read as <=1 column. Repair CSV formatting first.")
    return df


def extract_section(text: str, heading_contains: str) -> str:
    """
    Extract a markdown section beginning at a heading containing `heading_contains`.
    Stops at the next same-level or higher-level heading.
    """
    lines = text.splitlines()
    start = None
    start_level = None

    for i, line in enumerate(lines):
        m = re.match(r"^(#{1,6})\s+(.*)$", line.strip())
        if not m:
            continue
        title = m.group(2)
        if heading_contains.lower() in title.lower():
            start = i
            start_level = len(m.group(1))
            break

    if start is None:
        return f"[MISSING SECTION: {heading_contains}]"

    end = len(lines)
    for j in range(start + 1, len(lines)):
        m = re.match(r"^(#{1,6})\s+(.*)$", lines[j].strip())
        if not m:
            continue
        level = len(m.group(1))
        if level <= start_level:
            end = j
            break

    return "\n".join(lines[start:end]).strip()


def strip_first_heading(section: str) -> str:
    lines = section.splitlines()
    if lines and re.match(r"^#{1,6}\s+", lines[0].strip()):
        return "\n".join(lines[1:]).strip()
    return section.strip()


def make_section_inventory() -> pd.DataFrame:
    rows = [
        {
            "paper_section": "Title",
            "source": "P1/P2",
            "assembly_status": "drafted",
            "notes": "Working title is fixed but can be shortened later.",
        },
        {
            "paper_section": "Abstract",
            "source": "P1/P2/P3",
            "assembly_status": "drafted",
            "notes": "Needs final word-limit adjustment after venue selection.",
        },
        {
            "paper_section": "Introduction",
            "source": "P2",
            "assembly_status": "drafted",
            "notes": "Must keep claim boundaries around VLM reasoning.",
        },
        {
            "paper_section": "Related Work",
            "source": "P4",
            "assembly_status": "drafted",
            "notes": "Needs exact BibTeX entries before submission.",
        },
        {
            "paper_section": "Method",
            "source": "P5",
            "assembly_status": "drafted",
            "notes": "Core formulas and algorithm are included.",
        },
        {
            "paper_section": "Experiments",
            "source": "P3",
            "assembly_status": "drafted",
            "notes": "Tables are referenced by source CSV; formatting still needed.",
        },
        {
            "paper_section": "Failure and Boundary Analysis",
            "source": "P3/Stage16E",
            "assembly_status": "drafted_data_ready",
            "notes": "Representative images still need manual inspection.",
        },
        {
            "paper_section": "Limitations",
            "source": "P1/P3/P6 checklist",
            "assembly_status": "drafted",
            "notes": "Must mention EfficientAD fixed-budget and missing AnomalyCLIP.",
        },
        {
            "paper_section": "Conclusion",
            "source": "P6",
            "assembly_status": "drafted",
            "notes": "Keep conservative wording.",
        },
    ]
    return pd.DataFrame(rows)


def make_missing_checklist() -> pd.DataFrame:
    rows = [
        {
            "item_id": "M1",
            "item": "Draw Figure 1 framework overview",
            "priority": "high",
            "reason": "Method needs a visual pipeline: detector localization -> candidate crop -> VLM evidence -> quality calibration -> adaptive refinement.",
            "status": "missing",
        },
        {
            "item_id": "M2",
            "item": "Select Figure 2 representative boundary cases",
            "priority": "high",
            "reason": "Stage 16-E generated case inventory, but image examples must be visually inspected before paper use.",
            "status": "missing",
        },
        {
            "item_id": "M3",
            "item": "Convert CSV tables into paper-formatted LaTeX tables",
            "priority": "high",
            "reason": "Paper skeleton references CSV outputs; final paper needs compact formatted tables.",
            "status": "missing",
        },
        {
            "item_id": "M4",
            "item": "Prepare BibTeX references",
            "priority": "high",
            "reason": "P4 has reference inventory but not exact BibTeX entries.",
            "status": "missing",
        },
        {
            "item_id": "M5",
            "item": "Venue template and page budget",
            "priority": "medium",
            "reason": "Abstract length, table count, and appendix size depend on target venue.",
            "status": "missing",
        },
        {
            "item_id": "M6",
            "item": "Decide whether to add AnomalyCLIP or explicitly list it as limitation",
            "priority": "medium_high",
            "reason": "AnomalyCLIP is a likely reviewer question for VLM anomaly work.",
            "status": "open_decision",
        },
        {
            "item_id": "M7",
            "item": "Polish English academic writing",
            "priority": "medium",
            "reason": "Current text is claim-safe but still draft-like.",
            "status": "missing",
        },
        {
            "item_id": "M8",
            "item": "Write exact protocol details",
            "priority": "medium_high",
            "reason": "LOCO, same-set, QCR primary protocol, and EfficientAD fixed-budget must be unambiguous.",
            "status": "needs_detailing",
        },
    ]
    return pd.DataFrame(rows)


def make_readiness() -> pd.DataFrame:
    rows = [
        {
            "dimension": "experimental_evidence_chain",
            "status": "mostly_closed",
            "evidence": "Stage 15 strong baselines, Stage 16 method/claim map, Stage 17 EfficientAD sensitivity are complete.",
            "blocking_for_first_draft": False,
        },
        {
            "dimension": "method_name_and_claims",
            "status": "closed",
            "evidence": "Quality-Calibrated QCR is locked; fixed Q+C rejected as final method.",
            "blocking_for_first_draft": False,
        },
        {
            "dimension": "paper_text_skeleton",
            "status": "created_by_p6",
            "evidence": "P6 assembles P2/P3/P4/P5 into one skeleton.",
            "blocking_for_first_draft": False,
        },
        {
            "dimension": "figures",
            "status": "missing",
            "evidence": "Framework and boundary-case figures are not drawn/selected.",
            "blocking_for_first_draft": True,
        },
        {
            "dimension": "references",
            "status": "incomplete",
            "evidence": "Reference inventory exists, but BibTeX is not prepared.",
            "blocking_for_first_draft": True,
        },
        {
            "dimension": "external_vlm_baseline_risk",
            "status": "open",
            "evidence": "AnomalyCLIP is not included; currently handled as limitation.",
            "blocking_for_first_draft": False,
        },
        {
            "dimension": "submission_readiness",
            "status": "draftable_not_submission_ready",
            "evidence": "Evidence chain is coherent, but figures, BibTeX, table formatting, and final polishing remain.",
            "blocking_for_first_draft": False,
        },
    ]
    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DOC_DIR.mkdir(parents=True, exist_ok=True)

    p1 = read_text(IN_P1_OUTLINE)
    p2 = read_text(IN_P2)
    p3 = read_text(IN_P3)
    p4 = read_text(IN_P4)
    p5 = read_text(IN_P5)

    table_inv = read_csv_optional(IN_TABLE_INV)
    risks = read_csv_optional(IN_RISKS)
    claim_map = read_csv_optional(IN_CLAIM_MAP)
    rejected = read_csv_optional(IN_REJECTED)

    section_inv = make_section_inventory()
    checklist = make_missing_checklist()
    readiness = make_readiness()

    section_inv.to_csv(OUT_SECTION_INV, index=False, lineterminator="\n")
    checklist.to_csv(OUT_CHECKLIST, index=False, lineterminator="\n")
    readiness.to_csv(OUT_READINESS, index=False, lineterminator="\n")

    intro = strip_first_heading(extract_section(p2, "Introduction Draft"))
    contributions = strip_first_heading(extract_section(p2, "Contribution Draft"))
    related = strip_first_heading(extract_section(p4, "Related Work Draft"))
    method = strip_first_heading(extract_section(p5, "Method Overview"))
    notation = extract_section(p5, "Notation")
    candidate_generation = extract_section(p5, "Localization-guided Candidate Generation")
    crop_vlm = extract_section(p5, "Crop-level VLM Anomaly Evidence")
    quality = extract_section(p5, "Candidate Quality Calibration")
    fixed_qc = extract_section(p5, "Diagnostic Fixed Q+C Fusion")
    adaptive = extract_section(p5, "Adaptive Consistency Refinement")
    algorithm = extract_section(p5, "Algorithm")
    exp_setup = strip_first_heading(extract_section(p3, "Experimental Setup Draft"))
    baselines = strip_first_heading(extract_section(p3, "Baselines Draft"))
    main_results = strip_first_heading(extract_section(p3, "Main System-level Results Draft"))
    qcr_ablation = strip_first_heading(extract_section(p3, "QCR Ablation Results Draft"))
    failure = strip_first_heading(extract_section(p3, "Failure and Boundary Analysis Draft"))
    ead_sens = strip_first_heading(extract_section(p3, "EfficientAD-100 Sensitivity Draft"))
    restrictions = strip_first_heading(extract_section(p3, "Result-writing Restrictions"))

    lines = []
    lines += [
        "# Paper Stage P6: First Full Paper Draft Skeleton",
        "",
        "## Working Title",
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
        "---",
        "",
        "# Abstract",
        "",
        "Industrial anomaly recognition with general-purpose vision-language models remains unreliable when images are evaluated globally, because defects are often small, localized, and visually subtle. "
        "We propose a quality-calibrated localization-guided VLM reasoning framework that converts detector localization evidence into candidate-level visual-language evidence. "
        "The core method, Quality-Calibrated QCR, calibrates crop-level VLM anomaly scores using candidate quality derived from localization evidence. "
        "We further study detector-VLM consistency and find that fixed consistency is not robust enough to serve as the final method; therefore, consistency is used only as a conservative adaptive refinement. "
        "Experiments with detector, VLM, and external anomaly baselines show that localization-guided VLM evidence complements detector scores, while ablations identify candidate quality calibration as the main effective component. "
        "Failure and boundary analyses clarify when quality calibration helps, when it can fail, and why fixed consistency should remain diagnostic.",
        "",
        "---",
        "",
        "# 1. Introduction",
        "",
        intro,
        "",
        "## Contributions",
        "",
        contributions,
        "",
        "---",
        "",
        "# 2. Related Work",
        "",
        related,
        "",
        "---",
        "",
        "# 3. Method",
        "",
        method,
        "",
        notation,
        "",
        candidate_generation,
        "",
        crop_vlm,
        "",
        quality,
        "",
        fixed_qc,
        "",
        adaptive,
        "",
        algorithm,
        "",
        "---",
        "",
        "# 4. Experimental Setup",
        "",
        exp_setup,
        "",
        baselines,
        "",
        "---",
        "",
        "# 5. Main Results",
        "",
        main_results,
        "",
        "---",
        "",
        "# 6. Ablation Study",
        "",
        qcr_ablation,
        "",
        "---",
        "",
        "# 7. Failure and Boundary Analysis",
        "",
        failure,
        "",
        "---",
        "",
        "# 8. Baseline-budget Sensitivity",
        "",
        ead_sens,
        "",
        "---",
        "",
        "# 9. Limitations",
        "",
        "The method should be interpreted as image-level anomaly recognition and candidate-level VLM evidence calibration, not as pixel-level segmentation SOTA or manufacturing-cause reasoning. "
        "EfficientAD is reported as a fixed-budget baseline, with a fruit_jelly 100-epoch sensitivity check used only as defensive evidence. "
        "AnomalyCLIP is not included in the current experiments and should be listed as a remaining external VLM anomaly baseline risk. "
        "Adaptive consistency provides only a small refinement over the quality-calibrated core, so the main method claim must remain centered on candidate quality calibration.",
        "",
        "Risk inventory:",
        "",
    ]

    if risks.empty:
        lines.append("- [MISSING] P1 risk inventory not found.")
    else:
        for _, r in risks.iterrows():
            lines.append(
                f"- `{r['risk_id']}` **{r['risk']}** Mitigation: {r['mitigation']}"
            )

    lines += [
        "",
        "---",
        "",
        "# 10. Conclusion",
        "",
        "We presented Quality-Calibrated QCR, a localization-guided VLM reasoning framework for industrial anomaly recognition. "
        "The main finding is that detector localization can provide useful candidate-level evidence for VLM anomaly scoring, but crop-level VLM evidence must be calibrated by candidate quality. "
        "The resulting quality-calibrated score improves over naive detector-crop fusion and provides a more reliable method core. "
        "Consistency is useful only as a conservative adaptive refinement and should not be treated as a universally beneficial fixed fusion term. "
        "The evidence chain supports a conservative but coherent conclusion: quality-calibrated localization-guided VLM reasoning is a practical way to combine industrial anomaly localization and VLM-based anomaly evidence.",
        "",
        "---",
        "",
        "# Appendix / Paper Assembly Notes",
        "",
        "## A. Result-writing Restrictions",
        "",
        restrictions,
        "",
        "## B. Rejected or Forbidden Claims",
        "",
    ]

    if rejected.empty:
        lines.append("- [MISSING] rejected-claim file not found.")
    else:
        for _, r in rejected.iterrows():
            lines.append(
                f"- Forbidden: **{r['forbidden_wording']}**  \n"
                f"  Replacement: {r['allowed_wording']}"
            )

    lines += [
        "",
        "## C. Tables and Figures Still Needed",
        "",
    ]

    if table_inv.empty:
        lines.append("- [MISSING] table inventory not found.")
    else:
        for _, r in table_inv.iterrows():
            must = "required" if bool(r.get("must_include", False)) else "optional"
            lines.append(
                f"- `{r['table_id']}` **{r['table_title']}** ({must}). "
                f"Source: `{r['source_file']}`. Notes: {r['notes']}"
            )

    lines += [
        "",
        "## D. Missing Items Checklist",
        "",
        "| ID | Item | Priority | Reason | Status |",
        "|---|---|---|---|---|",
    ]

    for _, r in checklist.iterrows():
        lines.append(
            f"| {r['item_id']} | {r['item']} | {r['priority']} | {r['reason']} | {r['status']} |"
        )

    lines += [
        "",
        "## E. Submission Readiness",
        "",
        "| Dimension | Status | Evidence | Blocking for First Draft |",
        "|---|---|---|---:|",
    ]

    for _, r in readiness.iterrows():
        lines.append(
            f"| {r['dimension']} | {r['status']} | {r['evidence']} | {int(bool(r['blocking_for_first_draft']))} |"
        )

    lines += [
        "",
        "## F. Next Step",
        "",
        "Next stage:",
        "",
        "```text",
        "Paper Stage P7: convert skeleton into LaTeX-style paper draft and compact tables",
        "```",
        "",
        "Before P7, inspect this skeleton once and remove duplicated paragraphs caused by importing earlier stage drafts.",
        "",
        "## G. Outputs",
        "",
        f"- `{OUT_DOC.relative_to(ROOT)}`",
        f"- `{OUT_SECTION_INV.relative_to(ROOT)}`",
        f"- `{OUT_CHECKLIST.relative_to(ROOT)}`",
        f"- `{OUT_READINESS.relative_to(ROOT)}`",
        "",
    ]

    OUT_DOC.write_text("\n".join(lines), encoding="utf-8", newline="\n")

    print("[DONE]", OUT_DOC)
    print("[DONE]", OUT_SECTION_INV)
    print("[DONE]", OUT_CHECKLIST)
    print("[DONE]", OUT_READINESS)
    print()
    print("===== section inventory =====")
    print(section_inv.to_string(index=False))
    print()
    print("===== missing checklist =====")
    print(checklist.to_string(index=False))
    print()
    print("===== readiness =====")
    print(readiness.to_string(index=False))


if __name__ == "__main__":
    main()
