from __future__ import annotations

from pathlib import Path
import pandas as pd


ROOT = Path(".").resolve()

STAGE11_MAIN = ROOT / "results/stage11_mvtecad2_multicategory/stage11_i_paper_ready_main_table.csv"
STAGE11_METHOD = ROOT / "results/stage11_mvtecad2_multicategory/stage11_i_paper_ready_method_table.csv"
STAGE11_USAGE = ROOT / "results/stage11_mvtecad2_multicategory/stage11_i_category_usage_decision_table.csv"
STAGE12_CLAIMS = ROOT / "results/stage12_paper_writing/stage12_a_claim_evidence_table.csv"

OUT_DIR = ROOT / "results/stage12_paper_writing"
DOC_DIR = ROOT / "docs/stage12_paper_writing"

OUT_SECTION_PLAN = OUT_DIR / "stage12_b_section_plan.csv"
OUT_TITLE_ABSTRACT = OUT_DIR / "stage12_b_title_abstract_candidates.csv"
OUT_REPORT = DOC_DIR / "stage12_b_paper_outline.md"


def read(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def f4(x) -> str:
    if x is None or pd.isna(x) or x == "":
        return ""
    return f"{float(x):.4f}"


def build_section_plan() -> pd.DataFrame:
    rows = [
        {
            "section": "Title",
            "purpose": "Emphasize localization-guided, context-aware VLM reasoning for industrial anomaly detection.",
            "key_content": "Avoid claiming a new detector SOTA. Highlight visual bridge and context-aware crop reasoning.",
            "evidence_or_input": "Stage 11-I main table and Stage 12-A claim table.",
        },
        {
            "section": "Abstract",
            "purpose": "Summarize problem, method, main result, and limitation in one compact paragraph.",
            "key_content": "Industrial anomaly reasoning with VLMs; detector localization as visual bridge; context-aware crop aggregation; +0.0835 AUROC on ALL_PRIMARY.",
            "evidence_or_input": "ALL_PRIMARY full=0.5201, context=0.6036, delta=+0.0835.",
        },
        {
            "section": "Introduction",
            "purpose": "Motivate why full-image VLM prompting and naive tight crop prompting are both insufficient.",
            "key_content": "Full image dilutes small defects; tight crop loses object context; context-aware crop keeps both anomaly evidence and product semantics.",
            "evidence_or_input": "Stage 10-G and Stage 11-D/E observations.",
        },
        {
            "section": "Related Work",
            "purpose": "Position against anomaly detection, VLM industrial inspection, and crop/region-guided reasoning.",
            "key_content": "PatchCore-style localization is not the contribution; contribution is using localization as a VLM visual bridge.",
            "evidence_or_input": "Method framing from Stage 12-A.",
        },
        {
            "section": "Method",
            "purpose": "Describe the full pipeline.",
            "key_content": "Anomaly map generation, candidate region proposal, context-aware crop construction, VLM prompt scoring, top-k aggregation, detector-quality-aware evaluation.",
            "evidence_or_input": "Stage 12-A method narrative.",
        },
        {
            "section": "Experiments",
            "purpose": "Report detector quality, candidate quality, VLM comparison, secondary and failure cases.",
            "key_content": "MVTec AD 2; primary/secondary/detector-risk split; full/tight/context comparison; PatchCore score as detector reference.",
            "evidence_or_input": "Stage 11-B/C/D/H/I.",
        },
        {
            "section": "Discussion",
            "purpose": "Explain category-dependent behavior.",
            "key_content": "Positive on ALL_PRIMARY aggregate, fruit_jelly, walnuts; limitation on sheet_metal, vial, fabric; detector-risk categories excluded.",
            "evidence_or_input": "Stage 11-I usage decision table.",
        },
        {
            "section": "Conclusion",
            "purpose": "State the conditional contribution without overclaiming.",
            "key_content": "Context-aware localization-guided VLM reasoning is useful when candidate regions preserve sufficient object context.",
            "evidence_or_input": "Claim C1-C4.",
        },
    ]

    return pd.DataFrame(rows)


def build_title_abstract(main: pd.DataFrame) -> pd.DataFrame:
    all_primary = main[main["category_or_scope"] == "ALL_PRIMARY"].iloc[0]

    main_result = (
        f"On the unified MVTec AD 2 primary-category evaluation, "
        f"full-image VLM prompting obtains AUROC {f4(all_primary['full_image_auroc'])}, "
        f"whereas {all_primary['reported_method']} obtains AUROC "
        f"{f4(all_primary['reported_method_auroc'])}, improving by "
        f"{f4(all_primary['delta_auroc_vs_full'])}."
    )

    rows = [
        {
            "type": "title",
            "candidate_id": "T1",
            "text": "Context-Aware Localization-Guided Visual-Language Reasoning for Industrial Anomaly Detection",
            "comment": "Most balanced title. Emphasizes method and task.",
        },
        {
            "type": "title",
            "candidate_id": "T2",
            "text": "Using Anomaly Localization as a Visual Bridge for VLM-based Industrial Defect Reasoning",
            "comment": "Emphasizes the core insight: localization as bridge.",
        },
        {
            "type": "title",
            "candidate_id": "T3",
            "text": "Beyond Full-image Prompting: Context-Aware Region Guidance for VLM Industrial Anomaly Reasoning",
            "comment": "Good if the paper focuses on full-image vs crop comparison.",
        },
        {
            "type": "title",
            "candidate_id": "T4",
            "text": "Detector-Guided Context Crops for Visual-Language Industrial Anomaly Understanding",
            "comment": "Shorter title, but less explicit about localization bridge.",
        },
        {
            "type": "abstract",
            "candidate_id": "A1",
            "text": (
                "Visual-language models provide a promising interface for industrial anomaly understanding, "
                "but direct full-image prompting often dilutes small defect evidence, while naive tight cropping "
                "can remove necessary object-level context. This work studies a localization-guided VLM reasoning "
                "pipeline that uses a classical anomaly detector as a visual bridge: anomaly maps produce candidate "
                "regions, candidate boxes are expanded into context-aware crops, and a VLM scores normal-versus-abnormal "
                "prompt margins over full images and localized crops. Experiments on MVTec AD 2 show that the benefit "
                "is conditional on detector and candidate quality. "
                + main_result
                + " Category-level analysis further shows positive evidence on fruit_jelly and walnuts, while "
                  "sheet_metal, vial, and fabric reveal limitations caused by candidate quality and object-context sensitivity."
            ),
            "comment": "Recommended abstract draft. Honest and not overclaimed.",
        },
    ]

    return pd.DataFrame(rows)


def write_report(
    main: pd.DataFrame,
    method: pd.DataFrame,
    usage: pd.DataFrame,
    claims: pd.DataFrame,
    section_plan: pd.DataFrame,
    title_abstract: pd.DataFrame,
) -> None:
    DOC_DIR.mkdir(parents=True, exist_ok=True)

    all_primary = main[main["category_or_scope"] == "ALL_PRIMARY"].iloc[0]

    title_rows = title_abstract[title_abstract["type"] == "title"]
    abstract_rows = title_abstract[title_abstract["type"] == "abstract"]

    lines = []

    lines.append("# Stage 12-B Paper Outline, Title, Abstract, and Contributions")
    lines.append("")
    lines.append("## 1. Purpose")
    lines.append("")
    lines.append("This stage converts the current experimental evidence into a paper-level structure.")
    lines.append("It does not run models, regenerate crops, or modify experimental results.")
    lines.append("")
    lines.append("## 2. Recommended Paper Positioning")
    lines.append("")
    lines.append("The paper should not be positioned as a new anomaly detector or a detector SOTA paper.")
    lines.append("")
    lines.append("Recommended positioning:")
    lines.append("")
    lines.append("```text")
    lines.append("A localization-guided VLM reasoning framework for industrial anomaly understanding, where classical anomaly localization serves as a visual bridge and context-aware crop construction preserves object semantics for VLM reasoning.")
    lines.append("```")
    lines.append("")
    lines.append("Avoid positioning:")
    lines.append("")
    lines.append("```text")
    lines.append("A new state-of-the-art industrial anomaly detector.")
    lines.append("```")
    lines.append("")
    lines.append("## 3. Title Candidates")
    lines.append("")
    lines.append("| ID | Title | Comment |")
    lines.append("|---|---|---|")
    for _, r in title_rows.iterrows():
        lines.append(f"| {r['candidate_id']} | {r['text']} | {r['comment']} |")

    lines.append("")
    lines.append("Recommended title:")
    lines.append("")
    lines.append("```text")
    lines.append("Context-Aware Localization-Guided Visual-Language Reasoning for Industrial Anomaly Detection")
    lines.append("```")
    lines.append("")
    lines.append("## 4. Abstract Draft")
    lines.append("")
    for _, r in abstract_rows.iterrows():
        lines.append(f"### {r['candidate_id']}")
        lines.append("")
        lines.append(r["text"])
        lines.append("")

    lines.append("## 5. Core Contributions")
    lines.append("")
    lines.append("1. **Localization-as-bridge formulation.** The paper formulates classical anomaly localization as a visual bridge between industrial anomaly detectors and VLM reasoning, rather than treating PatchCore as the final contribution.")
    lines.append("2. **Context-aware crop construction.** The method explicitly distinguishes tight anomaly crops from context-aware crops, showing that preserving object context is important for VLM anomaly reasoning.")
    lines.append("3. **Detector-quality-aware evaluation.** The evaluation reports detector quality, candidate coverage, VLM reasoning metrics, and limitation categories together, avoiding misleading crop-based conclusions when localization is weak.")
    lines.append("4. **Unified MVTec AD 2 multi-category evidence.** The final Stage 11 pipeline provides a unified multi-category evaluation and shows aggregate improvement while documenting category-dependent failure cases.")
    lines.append("")
    lines.append("## 6. Main Numeric Evidence")
    lines.append("")
    lines.append("| Scope | Full-image AUROC | Reported Method | Method AUROC | ΔAUROC | PatchCore Reference |")
    lines.append("|---|---:|---|---:|---:|---:|")
    lines.append(
        f"| ALL_PRIMARY | {f4(all_primary['full_image_auroc'])} | "
        f"{all_primary['reported_method']} | {f4(all_primary['reported_method_auroc'])} | "
        f"{f4(all_primary['delta_auroc_vs_full'])} | {f4(all_primary['patchcore_reference_auroc'])} |"
    )

    lines.append("")
    lines.append("## 7. Paper Section Plan")
    lines.append("")
    lines.append("| Section | Purpose | Key Content | Evidence/Input |")
    lines.append("|---|---|---|---|")
    for _, r in section_plan.iterrows():
        lines.append(f"| {r['section']} | {r['purpose']} | {r['key_content']} | {r['evidence_or_input']} |")

    lines.append("")
    lines.append("## 8. Experiment Table Placement")
    lines.append("")
    lines.append("| Table | Content | Source | Paper Role |")
    lines.append("|---|---|---|---|")
    lines.append("| Table 1 | Detector quality over all AD2 categories | Stage 11-B1 | Justify primary/secondary/risk split |")
    lines.append("| Table 2 | Main VLM comparison on ALL_PRIMARY and primary categories | Stage 11-I method table | Main result |")
    lines.append("| Table 3 | Secondary and risk category usage decision | Stage 11-I usage table | Boundary and fairness analysis |")
    lines.append("| Table 4 | Candidate quality statistics | Stage 11-C/H | Explain failure and sensitivity cases |")
    lines.append("")
    lines.append("## 9. Claim–Evidence Mapping")
    lines.append("")
    lines.append("| Claim ID | Claim | Evidence | Status | Paper Usage |")
    lines.append("|---|---|---|---|---|")
    for _, r in claims.iterrows():
        lines.append(f"| {r['claim_id']} | {r['claim']} | {r['evidence']} | {r['status']} | {r['paper_usage']} |")

    lines.append("")
    lines.append("## 10. Risk Control for Review")
    lines.append("")
    lines.append("| Potential reviewer concern | Response strategy |")
    lines.append("|---|---|")
    lines.append("| This is only cropping anomalous regions for a VLM. | Emphasize context-aware crop construction, aggregation, and detector-quality-aware evaluation. |")
    lines.append("| PatchCore is not novel. | State that PatchCore is a localization bridge and detector reference, not the claimed contribution. |")
    lines.append("| Results are not positive on every category. | Present category-dependent behavior as an honest limitation and show detector/candidate quality explains boundary cases. |")
    lines.append("| VLM score is weaker than PatchCore detector score. | Clarify that PatchCore is the detector reference; the VLM branch targets anomaly reasoning/semantic interpretation, not detector replacement. |")
    lines.append("")
    lines.append("## 11. Recommended Final Claim")
    lines.append("")
    lines.append("```text")
    lines.append("Localization-guided context-aware crops improve VLM anomaly reasoning on the unified MVTec AD 2 primary-category evaluation, but the benefit is category-dependent and sensitive to detector/candidate quality.")
    lines.append("```")
    lines.append("")
    lines.append("## 12. Next Step")
    lines.append("")
    lines.append("Stage 12-C should draft the full Introduction and Method sections in paper-style language, using this outline as the source of truth.")
    lines.append("")
    lines.append("## 13. Output")
    lines.append("")
    lines.append(f"- Section plan: `{OUT_SECTION_PLAN.relative_to(ROOT)}`")
    lines.append(f"- Title/abstract candidates: `{OUT_TITLE_ABSTRACT.relative_to(ROOT)}`")
    lines.append(f"- Paper outline: `{OUT_REPORT.relative_to(ROOT)}`")

    OUT_REPORT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DOC_DIR.mkdir(parents=True, exist_ok=True)

    main_table = read(STAGE11_MAIN)
    method_table = read(STAGE11_METHOD)
    usage_table = read(STAGE11_USAGE)
    claim_table = read(STAGE12_CLAIMS)

    section_plan = build_section_plan()
    title_abstract = build_title_abstract(main_table)

    section_plan.to_csv(OUT_SECTION_PLAN, index=False)
    title_abstract.to_csv(OUT_TITLE_ABSTRACT, index=False)

    write_report(
        main=main_table,
        method=method_table,
        usage=usage_table,
        claims=claim_table,
        section_plan=section_plan,
        title_abstract=title_abstract,
    )

    print("[DONE]", OUT_SECTION_PLAN)
    print("[DONE]", OUT_TITLE_ABSTRACT)
    print("[DONE]", OUT_REPORT)

    print("\n===== title / abstract =====")
    print(title_abstract.to_string(index=False))

    print("\n===== section plan =====")
    print(section_plan.to_string(index=False))


if __name__ == "__main__":
    main()
