from __future__ import annotations

from pathlib import Path
import pandas as pd


ROOT = Path(".").resolve()

STAGE12_B_OUTLINE = ROOT / "docs/stage12_paper_writing/stage12_b_paper_outline.md"
STAGE12_CLAIMS = ROOT / "results/stage12_paper_writing/stage12_a_claim_evidence_table.csv"
STAGE11_MAIN = ROOT / "results/stage11_mvtecad2_multicategory/stage11_i_paper_ready_main_table.csv"
STAGE11_METHOD = ROOT / "results/stage11_mvtecad2_multicategory/stage11_i_paper_ready_method_table.csv"
STAGE11_USAGE = ROOT / "results/stage11_mvtecad2_multicategory/stage11_i_category_usage_decision_table.csv"

OUT_DIR = ROOT / "results/stage12_paper_writing"
DOC_DIR = ROOT / "docs/stage12_paper_writing"

OUT_STATUS = OUT_DIR / "stage12_c_intro_method_draft_status.csv"
OUT_REPORT = DOC_DIR / "stage12_c_intro_method_draft.md"


def read(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def f4(x) -> str:
    if x is None or pd.isna(x) or x == "":
        return ""
    return f"{float(x):.4f}"


def write_report(main: pd.DataFrame, method: pd.DataFrame, usage: pd.DataFrame, claims: pd.DataFrame) -> None:
    all_primary = main[main["category_or_scope"] == "ALL_PRIMARY"].iloc[0]

    positive = usage[usage["paper_usage"].astype(str).str.contains("positive", case=False, na=False)]
    limitation = usage[usage["paper_usage"].astype(str).str.contains("limitation|boundary", case=False, na=False)]
    excluded = usage[usage["paper_usage"].astype(str).str.contains("excluded", case=False, na=False)]

    lines = []

    lines.append("# Stage 12-C Introduction and Method Draft")
    lines.append("")
    lines.append("## 1. Purpose")
    lines.append("")
    lines.append("This document drafts the Introduction and Method sections using the Stage 11 evidence and Stage 12-B paper outline.")
    lines.append("It is a paper-writing artifact only. It does not run models, regenerate crops, or modify experimental results.")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("# Introduction Draft")
    lines.append("")
    lines.append("Industrial anomaly detection aims to identify defective or abnormal products under limited supervision, often using only normal training samples. Classical anomaly detection methods have achieved strong localization and detection performance by modeling normal visual patterns and highlighting deviations at test time. However, these methods usually produce numerical anomaly scores and pixel-level heatmaps, while downstream industrial inspection often requires a more interpretable reasoning interface: users need to understand what visual evidence indicates an abnormality and how the suspected region relates to the product structure.")
    lines.append("")
    lines.append("Recent visual-language models provide a potential interface for such anomaly understanding because they can compare visual content with textual descriptions of normal and abnormal states. A straightforward strategy is to apply full-image VLM prompting to industrial inspection images. This strategy is simple, but it is often suboptimal for industrial defects. Many defects are small, local, and visually weak compared with the complete product appearance. When the whole image is passed to the VLM, the abnormal evidence can be diluted by large normal regions and background context.")
    lines.append("")
    lines.append("An opposite strategy is to crop the most anomalous region and feed only that local patch to the VLM. This also has an important limitation. Industrial anomalies are not purely local texture patterns; whether a visual cue is abnormal often depends on object-level context. A small crop may remove the surrounding product structure, making it difficult for the VLM to determine whether the local appearance is a defect or a normal part of the object.")
    lines.append("")
    lines.append("This motivates a localization-guided and context-aware VLM reasoning framework. Instead of treating classical anomaly localization as the final output, we use it as a visual bridge between anomaly detectors and VLM reasoning. The anomaly detector first proposes suspicious regions from anomaly maps. These regions are then converted into context-aware crops that preserve both local abnormal evidence and surrounding object semantics. Finally, a VLM scores normal-versus-abnormal prompt margins over full images, tight crops, and context-aware crops.")
    lines.append("")
    lines.append("The key design principle is that the VLM branch should not simply receive the smallest anomalous patch. It should receive a region that is sufficiently localized to focus on defect evidence, but sufficiently contextualized to maintain product semantics. This distinction is central to our method and avoids reducing the work to a trivial crop-and-prompt pipeline.")
    lines.append("")
    lines.append("We evaluate this idea on MVTec AD 2 using a unified multi-category pipeline. Our results show a conditional but meaningful benefit: on the primary-category aggregate, full-image VLM prompting obtains AUROC "
                 f"{f4(all_primary['full_image_auroc'])}, while the reported context-aware crop aggregation method `{all_primary['reported_method']}` obtains AUROC {f4(all_primary['reported_method_auroc'])}, improving by {f4(all_primary['delta_auroc_vs_full'])}. "
                 "At the same time, category-level analysis shows that this benefit is not universal. Positive evidence appears on some categories, while other categories reveal sensitivity to detector localization quality and candidate construction.")
    lines.append("")
    lines.append("The main contributions are summarized as follows:")
    lines.append("")
    lines.append("1. We formulate anomaly localization as a visual bridge for VLM-based industrial anomaly reasoning, rather than presenting the detector itself as the contribution.")
    lines.append("2. We introduce a context-aware crop construction strategy that preserves object-level semantics around detector-localized suspicious regions.")
    lines.append("3. We provide a detector-quality-aware evaluation protocol that jointly reports detector quality, candidate quality, VLM reasoning performance, and limitation cases.")
    lines.append("4. We conduct a unified MVTec AD 2 multi-category study showing aggregate improvement of context-aware crop reasoning over full-image VLM prompting, while explicitly documenting category-dependent failures.")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("# Method Draft")
    lines.append("")
    lines.append("## 1. Overview")
    lines.append("")
    lines.append("Given an industrial inspection image, the proposed framework produces VLM-based anomaly reasoning scores through four stages: anomaly localization, candidate region proposal, context-aware crop construction, and VLM prompt scoring. The detector and the VLM play different roles. The detector provides spatial priors through anomaly maps, while the VLM evaluates visual-language normality and abnormality cues over image regions.")
    lines.append("")
    lines.append("The complete pipeline is:")
    lines.append("")
    lines.append("```text")
    lines.append("input image -> anomaly map -> candidate region proposal -> context-aware crop construction -> VLM anomaly reasoning")
    lines.append("```")
    lines.append("")
    lines.append("## 2. Anomaly Localization as a Visual Bridge")
    lines.append("")
    lines.append("We first apply a classical anomaly localization model to the test image. In our implementation, PatchCore is used as the localization backbone. The role of PatchCore is not to serve as a new detector contribution, but to provide anomaly maps and image-level detector scores. The anomaly map highlights spatial regions that deviate from normal training samples.")
    lines.append("")
    lines.append("Let an inspection image be denoted as `I`. The anomaly detector produces an anomaly map `A`, where higher values indicate stronger local abnormality. This map is used to construct candidate regions for downstream VLM reasoning.")
    lines.append("")
    lines.append("## 3. Candidate Region Proposal")
    lines.append("")
    lines.append("Candidate regions are extracted from high-response areas in the anomaly map. Connected components or top-response regions are converted into bounding boxes. Each candidate has a detector confidence derived from the anomaly map response. Multiple candidates can be retained for one image, allowing the VLM stage to aggregate evidence across suspicious regions.")
    lines.append("")
    lines.append("This stage produces tight candidate boxes. These tight boxes focus on the most suspicious visual areas but may not contain enough object context for VLM reasoning.")
    lines.append("")
    lines.append("## 4. Context-aware Crop Construction")
    lines.append("")
    lines.append("To address the context loss of tight crops, each candidate box is expanded into a context-aware crop. The expansion preserves surrounding product structure while still focusing on the detector-localized suspicious region. This is important because industrial defects are often defined relative to the product geometry, material, or normal manufacturing pattern.")
    lines.append("")
    lines.append("We therefore evaluate both tight crops and context-aware crops. The tight crop tests whether local abnormal evidence alone is sufficient. The context-aware crop tests whether adding object context improves VLM reasoning. The final method emphasizes the latter because Stage 11 evidence shows that context-aware aggregation improves the primary-category aggregate result.")
    lines.append("")
    lines.append("## 5. VLM Prompt Scoring")
    lines.append("")
    lines.append("For each full image or crop, the VLM computes similarity scores against normal and abnormal textual prompts. The anomaly score is defined as the margin between abnormal-prompt similarity and normal-prompt similarity. Category-aware prompts are used to describe normal and defective industrial products.")
    lines.append("")
    lines.append("For each image, we evaluate:")
    lines.append("")
    lines.append("- full-image VLM score;")
    lines.append("- tight crop top-1 score;")
    lines.append("- tight crop top-k maximum and mean scores;")
    lines.append("- context-aware crop top-1 score;")
    lines.append("- context-aware crop top-k maximum and mean scores;")
    lines.append("- PatchCore image score as a detector reference.")
    lines.append("")
    lines.append("PatchCore score is reported only as a detector reference. It should not be interpreted as a VLM reasoning method.")
    lines.append("")
    lines.append("## 6. Detector-quality-aware Evaluation")
    lines.append("")
    lines.append("Crop-based VLM reasoning depends strongly on localization quality. If the detector proposes poor candidate regions, VLM crop scores may fail for reasons unrelated to the VLM itself. Therefore, categories are separated into primary, secondary, and detector-risk groups according to detector quality and localization reliability.")
    lines.append("")
    lines.append("Primary categories are used for the main VLM crop comparison. Secondary categories are used as boundary cases. Detector-risk categories are excluded from the main VLM evidence because weak localization would make crop-based reasoning unfair or difficult to interpret.")
    lines.append("")
    lines.append("## 7. Current Evidence Used by the Method Section")
    lines.append("")
    lines.append("The method should be described with the following evidence constraints:")
    lines.append("")
    lines.append("| Evidence Type | Current Result | Usage |")
    lines.append("|---|---|---|")
    lines.append(f"| Main aggregate | `{all_primary['reported_method']}` improves AUROC from {f4(all_primary['full_image_auroc'])} to {f4(all_primary['reported_method_auroc'])} | Main positive evidence |")
    lines.append("| Positive categories | fruit_jelly and walnuts | Category-level support |")
    lines.append("| Limitation categories | sheet_metal, vial, and fabric | Boundary/failure analysis |")
    lines.append("| Detector-risk categories | can, rice, wallplugs | Excluded from main VLM evidence |")
    lines.append("")
    lines.append("## 8. Recommended Method Claim")
    lines.append("")
    lines.append("The method section should use this claim:")
    lines.append("")
    lines.append("```text")
    lines.append("Localization-guided context-aware crops improve VLM anomaly reasoning when candidate regions preserve sufficient object context and detector localization is reliable.")
    lines.append("```")
    lines.append("")
    lines.append("It should avoid this claim:")
    lines.append("")
    lines.append("```text")
    lines.append("Context-aware crops consistently improve all industrial anomaly categories.")
    lines.append("```")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("# Writing Notes")
    lines.append("")
    lines.append("- Do not overstate novelty of PatchCore.")
    lines.append("- Do not claim detector SOTA.")
    lines.append("- Do not mix Stage 10-G vial with the Stage 11 unified main table.")
    lines.append("- Present category-dependent behavior as part of the analysis, not as a hidden weakness.")
    lines.append("- Use `context-aware crop construction` and `localization as a visual bridge` as the two central phrases.")
    lines.append("")
    lines.append("# Output Dependency")
    lines.append("")
    lines.append(f"- Main evidence table: `results/stage11_mvtecad2_multicategory/stage11_i_paper_ready_main_table.csv`")
    lines.append(f"- Claim evidence table: `results/stage12_paper_writing/stage12_a_claim_evidence_table.csv`")
    lines.append(f"- Paper outline: `docs/stage12_paper_writing/stage12_b_paper_outline.md`")
    lines.append(f"- This draft: `{OUT_REPORT.relative_to(ROOT)}`")

    OUT_REPORT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DOC_DIR.mkdir(parents=True, exist_ok=True)

    main = read(STAGE11_MAIN)
    method = read(STAGE11_METHOD)
    usage = read(STAGE11_USAGE)
    claims = read(STAGE12_CLAIMS)

    write_report(main, method, usage, claims)

    status = pd.DataFrame([
        {
            "artifact": str(OUT_REPORT.relative_to(ROOT)),
            "status": "created",
            "purpose": "Introduction and Method paper draft based on Stage 11 and Stage 12-B evidence.",
        }
    ])
    status.to_csv(OUT_STATUS, index=False)

    print("[DONE]", OUT_REPORT)
    print("[DONE]", OUT_STATUS)


if __name__ == "__main__":
    main()
