from __future__ import annotations

from pathlib import Path
import pandas as pd


ROOT = Path(".").resolve()

STAGE11_MAIN = ROOT / "results/stage11_mvtecad2_multicategory/stage11_i_paper_ready_main_table.csv"
STAGE11_METHOD = ROOT / "results/stage11_mvtecad2_multicategory/stage11_i_paper_ready_method_table.csv"
STAGE11_USAGE = ROOT / "results/stage11_mvtecad2_multicategory/stage11_i_category_usage_decision_table.csv"
STAGE11_REPORT = ROOT / "docs/stage11_mvtecad2_multicategory/stage11_i_paper_ready_evidence_report.md"

OUT_DIR = ROOT / "results/stage12_paper_writing"
DOC_DIR = ROOT / "docs/stage12_paper_writing"

OUT_CLAIM_TABLE = OUT_DIR / "stage12_a_claim_evidence_table.csv"
OUT_REPORT = DOC_DIR / "stage12_a_method_experiment_narrative.md"


def f4(x) -> str:
    if x is None or pd.isna(x) or x == "":
        return ""
    return f"{float(x):.4f}"


def read(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def build_claim_table(main: pd.DataFrame, method: pd.DataFrame, usage: pd.DataFrame) -> pd.DataFrame:
    all_primary = main[main["category_or_scope"] == "ALL_PRIMARY"].iloc[0]

    rows = [
        {
            "claim_id": "C1",
            "claim": "Localization-guided context-aware crops improve VLM anomaly reasoning on the unified MVTec AD 2 primary-category evaluation.",
            "evidence": (
                f"ALL_PRIMARY full-image AUROC={f4(all_primary['full_image_auroc'])}; "
                f"{all_primary['reported_method']} AUROC={f4(all_primary['reported_method_auroc'])}; "
                f"delta={f4(all_primary['delta_auroc_vs_full'])}."
            ),
            "status": "main_claim_supported",
            "paper_usage": "Main result table and abstract-level claim.",
        },
        {
            "claim_id": "C2",
            "claim": "The benefit is category-dependent rather than universal.",
            "evidence": "fruit_jelly and walnuts are positive; sheet_metal and vial are limitation cases under the unified pipeline.",
            "status": "supported_with_limitations",
            "paper_usage": "Discussion and limitations.",
        },
        {
            "claim_id": "C3",
            "claim": "Detector/candidate quality must be reported alongside VLM crop reasoning.",
            "evidence": "fabric is a secondary boundary case; can, rice, and wallplugs are excluded from main VLM evidence due to weak detector quality.",
            "status": "supported",
            "paper_usage": "Experiment protocol and fairness discussion.",
        },
        {
            "claim_id": "C4",
            "claim": "The method should not be described as merely cropping anomaly regions for a VLM.",
            "evidence": "The final pipeline requires detector localization, candidate construction, context-aware crop generation, and VLM scoring/aggregation.",
            "status": "method_framing",
            "paper_usage": "Method section.",
        },
    ]

    return pd.DataFrame(rows)


def write_report(main: pd.DataFrame, method: pd.DataFrame, usage: pd.DataFrame, claim_table: pd.DataFrame) -> None:
    DOC_DIR.mkdir(parents=True, exist_ok=True)

    all_primary = main[main["category_or_scope"] == "ALL_PRIMARY"].iloc[0]

    positive = usage[usage["paper_usage"].astype(str).str.contains("positive", case=False, na=False)]
    limitations = usage[usage["paper_usage"].astype(str).str.contains("limitation|boundary", case=False, na=False)]
    excluded = usage[usage["paper_usage"].astype(str).str.contains("excluded", case=False, na=False)]

    lines = []

    lines.append("# Stage 12-A Method and Experiment Narrative")
    lines.append("")
    lines.append("## 1. Purpose")
    lines.append("")
    lines.append("This file converts the Stage 11 evidence tables into a paper-oriented method and experiment narrative.")
    lines.append("It does not run models, regenerate crops, or modify experimental results.")
    lines.append("")
    lines.append("## 2. Final Method Framing")
    lines.append("")
    lines.append("The method should be framed as:")
    lines.append("")
    lines.append("```text")
    lines.append("classical anomaly localization -> candidate region proposal -> context-aware crop construction -> VLM anomaly reasoning")
    lines.append("```")
    lines.append("")
    lines.append("The core idea is not simply to crop the most anomalous patch and send it to a VLM. The important design point is to preserve enough object-level context around detector-localized suspicious regions so that the VLM can jointly observe local abnormal evidence and global product semantics.")
    lines.append("")
    lines.append("## 3. Proposed Method Section Outline")
    lines.append("")
    lines.append("### 3.1 Problem Definition")
    lines.append("")
    lines.append("Given normal training images and test inspection images from industrial categories, the goal is to produce anomaly reasoning scores using a visual-language model without training a new task-specific VLM.")
    lines.append("")
    lines.append("### 3.2 Detector-side Localization Bridge")
    lines.append("")
    lines.append("A PatchCore-style anomaly detector is used as a localization bridge. It is not presented as the final contribution itself; rather, it provides anomaly maps and candidate regions that guide VLM attention toward suspicious visual areas.")
    lines.append("")
    lines.append("### 3.3 Context-aware Candidate Crop Construction")
    lines.append("")
    lines.append("Instead of using only tight anomaly crops, each candidate is expanded into a context-aware crop. This prevents the VLM from losing product-level semantics, which is critical in industrial inspection where the same local texture can be normal or abnormal depending on object context.")
    lines.append("")
    lines.append("### 3.4 VLM Reasoning and Aggregation")
    lines.append("")
    lines.append("The VLM branch compares normal and abnormal textual prompts and computes anomaly margins for full images, tight crops, and context-aware crops. Multi-candidate aggregation is evaluated through top-1, top-k maximum, and top-k mean strategies.")
    lines.append("")
    lines.append("### 3.5 Detector-quality-aware Evaluation")
    lines.append("")
    lines.append("Categories with weak detector quality are not used as main VLM crop evidence. This avoids attributing detector localization failures to the VLM reasoning branch.")
    lines.append("")
    lines.append("## 4. Experiment Section Outline")
    lines.append("")
    lines.append("### 4.1 Dataset and Splits")
    lines.append("")
    lines.append("The main multi-category evaluation is conducted on MVTec AD 2. Public test splits are used for measurable evaluation; private/unlabeled splits are excluded from metric computation.")
    lines.append("")
    lines.append("### 4.2 Detector Baseline")
    lines.append("")
    lines.append("PatchCore is first evaluated on all eight AD2 categories. The detector-quality analysis separates categories into primary, secondary, and detector-risk groups.")
    lines.append("")
    lines.append("### 4.3 Candidate Region Generation")
    lines.append("")
    lines.append("For primary categories, anomaly maps are converted into candidate regions and expanded into context-aware crops. Candidate coverage and GT mask coverage are reported to diagnose crop reliability.")
    lines.append("")
    lines.append("### 4.4 VLM Reasoning Comparison")
    lines.append("")
    lines.append("The main VLM comparison includes full-image prompting, tight crop prompting, context-aware crop prompting, and PatchCore score as a detector reference.")
    lines.append("")
    lines.append("### 4.5 Secondary and Failure Analysis")
    lines.append("")
    lines.append("The secondary category fabric and detector-risk categories are used to show boundary conditions, not to inflate the main claim.")
    lines.append("")
    lines.append("## 5. Main Numeric Result")
    lines.append("")
    lines.append("| Scope | Full-image AUROC | Reported Method | Method AUROC | ΔAUROC | PatchCore Reference |")
    lines.append("|---|---:|---|---:|---:|---:|")
    lines.append(
        f"| ALL_PRIMARY | {f4(all_primary['full_image_auroc'])} | "
        f"{all_primary['reported_method']} | {f4(all_primary['reported_method_auroc'])} | "
        f"{f4(all_primary['delta_auroc_vs_full'])} | {f4(all_primary['patchcore_reference_auroc'])} |"
    )
    lines.append("")
    lines.append("## 6. Method Comparison Table")
    lines.append("")
    lines.append("| Category | Role | Full | Tight top1 | Tight mean | Context top1 | Context mean | Best context Δ | Best VLM | Best VLM Δ | PatchCore |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|")

    for _, r in method.iterrows():
        lines.append(
            f"| {r['category']} | {r['role']} | {f4(r['full_image_auroc'])} | "
            f"{f4(r['tight_crop_top1_auroc'])} | {f4(r['tight_crop_topk_mean_auroc'])} | "
            f"{f4(r['context_1p50_top1_auroc'])} | {f4(r['context_1p50_topk_mean_auroc'])} | "
            f"{f4(r['best_context_delta_vs_full'])} | {r['best_vlm_method']} | "
            f"{f4(r['best_vlm_delta_vs_full'])} | {f4(r['patchcore_score_auroc'])} |"
        )

    lines.append("")
    lines.append("## 7. Category Usage Decision")
    lines.append("")
    lines.append("### Positive / supportive categories")
    lines.append("")
    if positive.empty:
        lines.append("- None")
    else:
        for _, r in positive.iterrows():
            lines.append(f"- `{r['category']}`: context Δ={f4(r['best_context_delta_vs_full'])}; {r['reason']}")

    lines.append("")
    lines.append("### Limitation / boundary categories")
    lines.append("")
    if limitations.empty:
        lines.append("- None")
    else:
        for _, r in limitations.iterrows():
            lines.append(f"- `{r['category']}`: context Δ={f4(r['best_context_delta_vs_full'])}; {r['reason']}")

    lines.append("")
    lines.append("### Excluded detector-risk categories")
    lines.append("")
    if excluded.empty:
        lines.append("- None")
    else:
        for _, r in excluded.iterrows():
            lines.append(f"- `{r['category']}`: image AUROC={f4(r['image_AUROC_patchcore'])}, pixel F1={f4(r['pixel_F1_patchcore'])}; {r['reason']}")

    lines.append("")
    lines.append("## 8. Claim–Evidence Mapping")
    lines.append("")
    lines.append("| Claim ID | Claim | Evidence | Status | Paper Usage |")
    lines.append("|---|---|---|---|---|")
    for _, r in claim_table.iterrows():
        lines.append(f"| {r['claim_id']} | {r['claim']} | {r['evidence']} | {r['status']} | {r['paper_usage']} |")

    lines.append("")
    lines.append("## 9. Recommended Wording")
    lines.append("")
    lines.append("Use:")
    lines.append("")
    lines.append("```text")
    lines.append("Localization-guided context-aware crops improve VLM anomaly reasoning on the unified MVTec AD 2 primary-category evaluation, but the benefit is category-dependent and sensitive to detector/candidate quality.")
    lines.append("```")
    lines.append("")
    lines.append("Avoid:")
    lines.append("")
    lines.append("```text")
    lines.append("Context-aware crops consistently improve all industrial anomaly categories.")
    lines.append("```")
    lines.append("")
    lines.append("## 10. Next Step")
    lines.append("")
    lines.append("Stage 12-B should convert this narrative into a structured paper outline: title, abstract, introduction contributions, method section, experiment section, and limitations.")
    lines.append("")
    lines.append("## 11. Output")
    lines.append("")
    lines.append(f"- Claim-evidence table: `{OUT_CLAIM_TABLE.relative_to(ROOT)}`")
    lines.append(f"- Narrative report: `{OUT_REPORT.relative_to(ROOT)}`")

    OUT_REPORT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DOC_DIR.mkdir(parents=True, exist_ok=True)

    main = read(STAGE11_MAIN)
    method = read(STAGE11_METHOD)
    usage = read(STAGE11_USAGE)

    claim_table = build_claim_table(main, method, usage)
    claim_table.to_csv(OUT_CLAIM_TABLE, index=False)

    write_report(main, method, usage, claim_table)

    print("[DONE]", OUT_CLAIM_TABLE)
    print("[DONE]", OUT_REPORT)
    print("")
    print(claim_table.to_string(index=False))


if __name__ == "__main__":
    main()
