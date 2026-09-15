from __future__ import annotations

from pathlib import Path
import pandas as pd


ROOT = Path(".").resolve()

IN_P1_RISKS = ROOT / "results/paper_p1/paper_remaining_risks.csv"
IN_P3_TABLE_MAP = ROOT / "results/paper_p3/paper_p3_table_to_text_map.csv"
IN_CLAIM_MAP = ROOT / "results/stage16_qcru_ablation/stage16_f_final_claim_evidence_map.csv"

OUT_DIR = ROOT / "results/paper_p4"
DOC_DIR = ROOT / "docs/paper_p4"

OUT_REF = OUT_DIR / "paper_p4_reference_inventory.csv"
OUT_POSITIONING = OUT_DIR / "paper_p4_positioning_map.csv"
OUT_SECTION_INV = OUT_DIR / "paper_p4_related_work_section_inventory.csv"
OUT_DOC = DOC_DIR / "paper_p4_related_work_and_positioning.md"


def read_csv_optional(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if len(df.columns) <= 1:
        raise RuntimeError(f"{path} read as <=1 column. Repair CSV formatting first.")
    return df


def build_reference_inventory() -> pd.DataFrame:
    rows = [
        {
            "cite_key": "Bergmann2019MVTecAD",
            "work": "MVTec AD",
            "year": 2019,
            "venue_or_type": "CVPR dataset paper",
            "category": "industrial_anomaly_dataset",
            "why_relevant": "Standard industrial anomaly detection and localization benchmark.",
            "how_to_position": "Use as background benchmark context, especially for industrial inspection framing.",
            "avoid_claim": "Do not claim our method solves all MVTec-style segmentation/localization problems.",
        },
        {
            "cite_key": "Zou2022VisA",
            "work": "VisA / SPot-the-Difference",
            "year": 2022,
            "venue_or_type": "arXiv / dataset",
            "category": "industrial_anomaly_dataset",
            "why_relevant": "Visual anomaly dataset with multiple industrial object categories and image/pixel annotations.",
            "how_to_position": "Use as dataset context for VisA-based experiments and candidate reasoning.",
            "avoid_claim": "Do not overgeneralize VisA results to all industrial domains.",
        },
        {
            "cite_key": "Roth2022PatchCore",
            "work": "PatchCore",
            "year": 2022,
            "venue_or_type": "CVPR",
            "category": "industrial_anomaly_detector",
            "why_relevant": "Patch-level memory-bank detector using nominal patch features; strong detector/localization baseline.",
            "how_to_position": "Our method uses detector localization evidence and shows VLM evidence can complement detector scores.",
            "avoid_claim": "Do not claim PatchCore is weak; it is a strong baseline.",
        },
        {
            "cite_key": "Yu2021FastFlow",
            "work": "FastFlow",
            "year": 2021,
            "venue_or_type": "arXiv",
            "category": "industrial_anomaly_detector",
            "why_relevant": "Normalizing-flow anomaly detection/localization method used as a detector-style backbone in our QCR analysis.",
            "how_to_position": "Use as detector/localization backbone context, not as the main novelty target.",
            "avoid_claim": "Do not claim flow-based detectors are replaced by VLM reasoning.",
        },
        {
            "cite_key": "Batzner2024EfficientAD",
            "work": "EfficientAD",
            "year": 2024,
            "venue_or_type": "WACV",
            "category": "industrial_anomaly_detector",
            "why_relevant": "Modern efficient student-teacher anomaly detection method; important strong detector baseline.",
            "how_to_position": "Report as EfficientAD-30 fixed-budget baseline and add fruit_jelly 100-epoch sensitivity.",
            "avoid_claim": "Do not claim full EfficientAD defeat.",
        },
        {
            "cite_key": "Radford2021CLIP",
            "work": "CLIP",
            "year": 2021,
            "venue_or_type": "ICML",
            "category": "vision_language_model",
            "why_relevant": "Foundation vision-language model enabling zero-shot text-image scoring.",
            "how_to_position": "Use as background for VLM anomaly reasoning and prompt-based visual evidence.",
            "avoid_claim": "Do not claim CLIP-style global semantics alone solves industrial anomaly recognition.",
        },
        {
            "cite_key": "Jeong2023WinCLIP",
            "work": "WinCLIP",
            "year": 2023,
            "venue_or_type": "CVPR",
            "category": "clip_anomaly_detection",
            "why_relevant": "Window-based CLIP method for zero-/few-shot anomaly classification and segmentation.",
            "how_to_position": "Use as external CLIP/VLM anomaly baseline under our fixed protocol.",
            "avoid_claim": "Do not claim we comprehensively outperform all WinCLIP settings.",
        },
        {
            "cite_key": "Zhou2024AnomalyCLIP",
            "work": "AnomalyCLIP",
            "year": 2024,
            "venue_or_type": "ICLR",
            "category": "clip_anomaly_detection",
            "why_relevant": "Object-agnostic prompt learning for zero-shot anomaly detection.",
            "how_to_position": "Mention as important related CLIP anomaly work and as a limitation if not included experimentally.",
            "avoid_claim": "Do not claim superiority over AnomalyCLIP unless it is run under a matched protocol.",
        },
    ]
    return pd.DataFrame(rows)


def build_positioning_map() -> pd.DataFrame:
    rows = [
        {
            "positioning_id": "RW-P1",
            "related_area": "Industrial anomaly detection and localization",
            "representative_work": "MVTec AD; VisA; PatchCore; FastFlow; EfficientAD",
            "what_prior_work_does": "Detects and localizes anomalies using normal-only training, patch features, flows, or student-teacher signals.",
            "our_gap": "Detector scores and localization maps do not directly provide reliable visual-language evidence.",
            "our_position": "We use detector localization as candidate evidence and calibrate crop-level VLM scores with candidate quality.",
            "safe_wording": "Our method complements detector evidence rather than replacing strong detectors.",
        },
        {
            "positioning_id": "RW-P2",
            "related_area": "CLIP / VLM anomaly detection",
            "representative_work": "CLIP; WinCLIP; AnomalyCLIP",
            "what_prior_work_does": "Uses image-text alignment, window features, prompts, or object-agnostic prompt learning for anomaly scoring.",
            "our_gap": "Global or window-level VLM scores may still be unreliable for tiny or poorly localized industrial defects.",
            "our_position": "We study localization-guided VLM evidence and reliability calibration of candidate crops.",
            "safe_wording": "We do not claim broad CLIP-family SOTA; we focus on candidate reliability calibration.",
        },
        {
            "positioning_id": "RW-P3",
            "related_area": "Localization-guided reasoning",
            "representative_work": "Detector-to-candidate pipelines; crop-based VLM scoring",
            "what_prior_work_does": "Uses localized regions to reduce irrelevant visual context.",
            "our_gap": "Candidate crops are not equally reliable; naive fusion ignores whether crop evidence should be trusted.",
            "our_position": "Candidate quality explicitly modulates crop-level VLM evidence.",
            "safe_wording": "The novelty is not merely cropping, but quality-calibrated crop evidence.",
        },
        {
            "positioning_id": "RW-P4",
            "related_area": "Reliability calibration and consistency",
            "representative_work": "Score fusion and agreement-based heuristics",
            "what_prior_work_does": "Combines model scores or agreement signals.",
            "our_gap": "Fixed consistency can be unstable and can hurt when detector/VLM evidence is unreliable.",
            "our_position": "Quality calibration is the stable core; adaptive consistency is a conservative refinement only.",
            "safe_wording": "Consistency is not claimed as universally beneficial.",
        },
    ]
    return pd.DataFrame(rows)


def build_section_inventory() -> pd.DataFrame:
    rows = [
        {
            "section_id": "RW-1",
            "section_title": "Industrial anomaly detection and localization",
            "main_refs": "Bergmann2019MVTecAD; Zou2022VisA; Roth2022PatchCore; Yu2021FastFlow; Batzner2024EfficientAD",
            "purpose": "Establish detector/localization context and explain why detector evidence is useful but not sufficient for VLM reasoning.",
            "status": "drafted",
        },
        {
            "section_id": "RW-2",
            "section_title": "Vision-language anomaly detection",
            "main_refs": "Radford2021CLIP; Jeong2023WinCLIP; Zhou2024AnomalyCLIP",
            "purpose": "Position VLM/CLIP anomaly methods and avoid broad CLIP-family superiority claims.",
            "status": "drafted",
        },
        {
            "section_id": "RW-3",
            "section_title": "Localization-guided VLM evidence",
            "main_refs": "PatchCore; FastFlow; VLM crop reasoning",
            "purpose": "Explain the gap between detector localization and trustworthy VLM evidence.",
            "status": "drafted",
        },
        {
            "section_id": "RW-4",
            "section_title": "Reliability calibration of candidate evidence",
            "main_refs": "Quality-Calibrated QCR evidence from our ablations",
            "purpose": "Position the main method as reliability calibration, not raw score fusion.",
            "status": "drafted",
        },
        {
            "section_id": "RW-5",
            "section_title": "Positioning summary",
            "main_refs": "All above",
            "purpose": "State exactly how this paper differs from detector-only and CLIP-only anomaly methods.",
            "status": "drafted",
        },
    ]
    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DOC_DIR.mkdir(parents=True, exist_ok=True)

    _ = read_csv_optional(IN_P1_RISKS)
    _ = read_csv_optional(IN_P3_TABLE_MAP)
    _ = read_csv_optional(IN_CLAIM_MAP)

    ref = build_reference_inventory()
    pos = build_positioning_map()
    sections = build_section_inventory()

    ref.to_csv(OUT_REF, index=False, lineterminator="\n")
    pos.to_csv(OUT_POSITIONING, index=False, lineterminator="\n")
    sections.to_csv(OUT_SECTION_INV, index=False, lineterminator="\n")

    lines = []
    lines += [
        "# Paper Stage P4: Related Work and Positioning",
        "",
        "## 1. Purpose",
        "",
        "This stage drafts the Related Work section and locks the paper's positioning against industrial anomaly detectors, CLIP/VLM anomaly methods, localization-guided reasoning, and reliability calibration.",
        "",
        "The goal is not to claim broad SOTA. The goal is to state precisely where Quality-Calibrated QCR fits.",
        "",
        "## 2. Related Work Draft",
        "",
        "### 2.1 Industrial anomaly detection and localization",
        "",
        "Industrial anomaly detection is commonly studied under settings where only normal training examples are available. Benchmarks such as MVTec AD and VisA provide industrial inspection images with image-level and localization-oriented annotations, making them central testbeds for anomaly recognition and localization. Detector-oriented methods typically learn normal appearance and identify deviations at test time.",
        "",
        "PatchCore represents a strong patch-feature memory-bank line of work. It stores representative nominal patch features and detects anomalies through deviations from normal patch-level statistics. Flow-based detectors such as FastFlow model feature distributions with normalizing flows and provide anomaly localization evidence. EfficientAD follows a different efficient student-teacher direction and is designed for accurate anomaly detection with low latency.",
        "",
        "These detector methods are strong baselines and provide useful localization signals. However, their anomaly scores are not visual-language explanations, and their localization maps do not directly determine whether a candidate crop is reliable VLM evidence. Our work is therefore complementary: we use detector localization to generate candidate regions and then calibrate crop-level VLM evidence using candidate quality.",
        "",
        "### 2.2 Vision-language anomaly detection",
        "",
        "Vision-language models such as CLIP enable zero-shot image-text matching and have motivated prompt-based anomaly detection. WinCLIP adapts CLIP to anomaly classification and segmentation by aggregating window-level visual features and text prompts. AnomalyCLIP further studies object-agnostic prompt learning for zero-shot anomaly detection.",
        "",
        "These methods show that language-supervised representations can support anomaly recognition. However, industrial anomalies are often small, localized, and visually subtle. Global image-text matching can be unreliable when the abnormal evidence occupies only a small region. Window-based or prompt-learning approaches address part of this problem, but they do not directly study how detector localization quality should modulate crop-level VLM evidence.",
        "",
        "Our work should therefore not be positioned as a broad replacement for CLIP anomaly methods. Instead, it studies a narrower but practical question: given localization evidence from an anomaly detector, how can crop-level VLM evidence be made more reliable?",
        "",
        "### 2.3 Localization-guided VLM evidence",
        "",
        "A natural way to improve VLM-based anomaly recognition is to reduce irrelevant context by presenting localized candidate regions to the VLM. This converts detector localization evidence into localized visual-language evidence. However, localization-guided VLM reasoning is not solved by cropping alone. Candidate crops can be too small, too broad, poorly centered, or produced by a misleading detector response.",
        "",
        "This motivates the core design of Quality-Calibrated QCR. The method does not treat every crop-level VLM score as equally trustworthy. Instead, it uses candidate quality to calibrate how strongly crop-level VLM evidence should contribute to the final anomaly score. The key contribution is therefore quality-calibrated candidate reasoning rather than simple crop extraction.",
        "",
        "### 2.4 Reliability calibration and consistency",
        "",
        "Naive detector-VLM fusion assumes that detector scores and VLM scores are directly comparable and equally reliable. Our ablation results show that this assumption is too weak. Candidate quality calibration provides the main improvement over naive fusion by modulating crop-level VLM evidence according to localization-derived reliability.",
        "",
        "We also study detector-VLM consistency. Fixed consistency can produce high scores in some primary settings, but robustness analysis shows that it is not stable enough to be the final method. Consequently, the final method treats consistency only as an adaptive, conservative refinement. This distinction is important: the paper should not claim that consistency is universally beneficial or that fixed Q+C fusion is the proposed method.",
        "",
        "### 2.5 Positioning summary",
        "",
        "The paper is positioned between detector-only industrial anomaly detection and VLM-only anomaly reasoning. Detector-only methods provide strong anomaly localization but do not provide calibrated visual-language evidence. VLM-only methods can use textual abnormality concepts but are unreliable when abnormal regions are small or poorly localized. Quality-Calibrated QCR connects these two directions by converting detector localization into candidate-level VLM evidence and calibrating that evidence using candidate quality.",
        "",
        "The safe positioning statement is:",
        "",
        "```text",
        "We propose a quality-calibrated localization-guided VLM reasoning framework for industrial anomaly recognition. Unlike detector-only methods, the framework converts localization evidence into visual-language anomaly evidence. Unlike VLM-only anomaly methods, it does not trust crop-level VLM scores blindly; instead, it calibrates them with candidate quality and uses consistency only as a conservative adaptive refinement.",
        "```",
        "",
        "## 3. Reference Inventory",
        "",
        "| Cite Key | Work | Year | Category | Positioning Use |",
        "|---|---|---:|---|---|",
    ]

    for _, r in ref.iterrows():
        lines.append(
            f"| {r['cite_key']} | {r['work']} | {int(r['year'])} | {r['category']} | {r['how_to_position']} |"
        )

    lines += [
        "",
        "## 4. Positioning Map",
        "",
        "| ID | Related Area | Prior Work Does | Our Gap | Our Position |",
        "|---|---|---|---|---|",
    ]

    for _, r in pos.iterrows():
        lines.append(
            f"| {r['positioning_id']} | {r['related_area']} | {r['what_prior_work_does']} | {r['our_gap']} | {r['our_position']} |"
        )

    lines += [
        "",
        "## 5. Forbidden Related-work Positioning",
        "",
        "- Do not claim to be the first VLM anomaly detection method.",
        "- Do not claim broad CLIP-family SOTA.",
        "- Do not claim superiority over AnomalyCLIP unless AnomalyCLIP is run under a matched protocol.",
        "- Do not claim to replace PatchCore, FastFlow, or EfficientAD.",
        "- Do not claim pixel-level segmentation SOTA.",
        "- Do not claim manufacturing-cause reasoning.",
        "",
        "## 6. Related Work Section Inventory",
        "",
        "| Section ID | Title | Main Refs | Purpose | Status |",
        "|---|---|---|---|---|",
    ]

    for _, r in sections.iterrows():
        lines.append(
            f"| {r['section_id']} | {r['section_title']} | {r['main_refs']} | {r['purpose']} | {r['status']} |"
        )

    lines += [
        "",
        "## 7. Next Step",
        "",
        "Next stage:",
        "",
        "```text",
        "Paper Stage P5: Method section full draft",
        "```",
        "",
        "P5 should turn the P2 method overview into a full method section with notation, algorithm steps, and scoring formulas.",
        "",
        "## 8. Outputs",
        "",
        f"- `{OUT_DOC.relative_to(ROOT)}`",
        f"- `{OUT_REF.relative_to(ROOT)}`",
        f"- `{OUT_POSITIONING.relative_to(ROOT)}`",
        f"- `{OUT_SECTION_INV.relative_to(ROOT)}`",
        "",
    ]

    OUT_DOC.write_text("\n".join(lines), encoding="utf-8", newline="\n")

    print("[DONE]", OUT_DOC)
    print("[DONE]", OUT_REF)
    print("[DONE]", OUT_POSITIONING)
    print("[DONE]", OUT_SECTION_INV)
    print()
    print("===== reference inventory =====")
    print(ref.to_string(index=False))
    print()
    print("===== positioning map =====")
    print(pos.to_string(index=False))


if __name__ == "__main__":
    main()
