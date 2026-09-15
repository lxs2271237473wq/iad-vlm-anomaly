# Stage 12-B Paper Outline, Title, Abstract, and Contributions

## 1. Purpose

This stage converts the current experimental evidence into a paper-level structure.
It does not run models, regenerate crops, or modify experimental results.

## 2. Recommended Paper Positioning

The paper should not be positioned as a new anomaly detector or a detector SOTA paper.

Recommended positioning:

```text
A localization-guided VLM reasoning framework for industrial anomaly understanding, where classical anomaly localization serves as a visual bridge and context-aware crop construction preserves object semantics for VLM reasoning.
```

Avoid positioning:

```text
A new state-of-the-art industrial anomaly detector.
```

## 3. Title Candidates

| ID | Title | Comment |
|---|---|---|
| T1 | Context-Aware Localization-Guided Visual-Language Reasoning for Industrial Anomaly Detection | Most balanced title. Emphasizes method and task. |
| T2 | Using Anomaly Localization as a Visual Bridge for VLM-based Industrial Defect Reasoning | Emphasizes the core insight: localization as bridge. |
| T3 | Beyond Full-image Prompting: Context-Aware Region Guidance for VLM Industrial Anomaly Reasoning | Good if the paper focuses on full-image vs crop comparison. |
| T4 | Detector-Guided Context Crops for Visual-Language Industrial Anomaly Understanding | Shorter title, but less explicit about localization bridge. |

Recommended title:

```text
Context-Aware Localization-Guided Visual-Language Reasoning for Industrial Anomaly Detection
```

## 4. Abstract Draft

### A1

Visual-language models provide a promising interface for industrial anomaly understanding, but direct full-image prompting often dilutes small defect evidence, while naive tight cropping can remove necessary object-level context. This work studies a localization-guided VLM reasoning pipeline that uses a classical anomaly detector as a visual bridge: anomaly maps produce candidate regions, candidate boxes are expanded into context-aware crops, and a VLM scores normal-versus-abnormal prompt margins over full images and localized crops. Experiments on MVTec AD 2 show that the benefit is conditional on detector and candidate quality. On the unified MVTec AD 2 primary-category evaluation, full-image VLM prompting obtains AUROC 0.5201, whereas context_1.50_topk_mean obtains AUROC 0.6036, improving by 0.0835. Category-level analysis further shows positive evidence on fruit_jelly and walnuts, while sheet_metal, vial, and fabric reveal limitations caused by candidate quality and object-context sensitivity.

## 5. Core Contributions

1. **Localization-as-bridge formulation.** The paper formulates classical anomaly localization as a visual bridge between industrial anomaly detectors and VLM reasoning, rather than treating PatchCore as the final contribution.
2. **Context-aware crop construction.** The method explicitly distinguishes tight anomaly crops from context-aware crops, showing that preserving object context is important for VLM anomaly reasoning.
3. **Detector-quality-aware evaluation.** The evaluation reports detector quality, candidate coverage, VLM reasoning metrics, and limitation categories together, avoiding misleading crop-based conclusions when localization is weak.
4. **Unified MVTec AD 2 multi-category evidence.** The final Stage 11 pipeline provides a unified multi-category evaluation and shows aggregate improvement while documenting category-dependent failure cases.

## 6. Main Numeric Evidence

| Scope | Full-image AUROC | Reported Method | Method AUROC | ΔAUROC | PatchCore Reference |
|---|---:|---|---:|---:|---:|
| ALL_PRIMARY | 0.5201 | context_1.50_topk_mean | 0.6036 | 0.0835 | 0.8087 |

## 7. Paper Section Plan

| Section | Purpose | Key Content | Evidence/Input |
|---|---|---|---|
| Title | Emphasize localization-guided, context-aware VLM reasoning for industrial anomaly detection. | Avoid claiming a new detector SOTA. Highlight visual bridge and context-aware crop reasoning. | Stage 11-I main table and Stage 12-A claim table. |
| Abstract | Summarize problem, method, main result, and limitation in one compact paragraph. | Industrial anomaly reasoning with VLMs; detector localization as visual bridge; context-aware crop aggregation; +0.0835 AUROC on ALL_PRIMARY. | ALL_PRIMARY full=0.5201, context=0.6036, delta=+0.0835. |
| Introduction | Motivate why full-image VLM prompting and naive tight crop prompting are both insufficient. | Full image dilutes small defects; tight crop loses object context; context-aware crop keeps both anomaly evidence and product semantics. | Stage 10-G and Stage 11-D/E observations. |
| Related Work | Position against anomaly detection, VLM industrial inspection, and crop/region-guided reasoning. | PatchCore-style localization is not the contribution; contribution is using localization as a VLM visual bridge. | Method framing from Stage 12-A. |
| Method | Describe the full pipeline. | Anomaly map generation, candidate region proposal, context-aware crop construction, VLM prompt scoring, top-k aggregation, detector-quality-aware evaluation. | Stage 12-A method narrative. |
| Experiments | Report detector quality, candidate quality, VLM comparison, secondary and failure cases. | MVTec AD 2; primary/secondary/detector-risk split; full/tight/context comparison; PatchCore score as detector reference. | Stage 11-B/C/D/H/I. |
| Discussion | Explain category-dependent behavior. | Positive on ALL_PRIMARY aggregate, fruit_jelly, walnuts; limitation on sheet_metal, vial, fabric; detector-risk categories excluded. | Stage 11-I usage decision table. |
| Conclusion | State the conditional contribution without overclaiming. | Context-aware localization-guided VLM reasoning is useful when candidate regions preserve sufficient object context. | Claim C1-C4. |

## 8. Experiment Table Placement

| Table | Content | Source | Paper Role |
|---|---|---|---|
| Table 1 | Detector quality over all AD2 categories | Stage 11-B1 | Justify primary/secondary/risk split |
| Table 2 | Main VLM comparison on ALL_PRIMARY and primary categories | Stage 11-I method table | Main result |
| Table 3 | Secondary and risk category usage decision | Stage 11-I usage table | Boundary and fairness analysis |
| Table 4 | Candidate quality statistics | Stage 11-C/H | Explain failure and sensitivity cases |

## 9. Claim–Evidence Mapping

| Claim ID | Claim | Evidence | Status | Paper Usage |
|---|---|---|---|---|
| C1 | Localization-guided context-aware crops improve VLM anomaly reasoning on the unified MVTec AD 2 primary-category evaluation. | ALL_PRIMARY full-image AUROC=0.5201; context_1.50_topk_mean AUROC=0.6036; delta=0.0835. | main_claim_supported | Main result table and abstract-level claim. |
| C2 | The benefit is category-dependent rather than universal. | fruit_jelly and walnuts are positive; sheet_metal and vial are limitation cases under the unified pipeline. | supported_with_limitations | Discussion and limitations. |
| C3 | Detector/candidate quality must be reported alongside VLM crop reasoning. | fabric is a secondary boundary case; can, rice, and wallplugs are excluded from main VLM evidence due to weak detector quality. | supported | Experiment protocol and fairness discussion. |
| C4 | The method should not be described as merely cropping anomaly regions for a VLM. | The final pipeline requires detector localization, candidate construction, context-aware crop generation, and VLM scoring/aggregation. | method_framing | Method section. |

## 10. Risk Control for Review

| Potential reviewer concern | Response strategy |
|---|---|
| This is only cropping anomalous regions for a VLM. | Emphasize context-aware crop construction, aggregation, and detector-quality-aware evaluation. |
| PatchCore is not novel. | State that PatchCore is a localization bridge and detector reference, not the claimed contribution. |
| Results are not positive on every category. | Present category-dependent behavior as an honest limitation and show detector/candidate quality explains boundary cases. |
| VLM score is weaker than PatchCore detector score. | Clarify that PatchCore is the detector reference; the VLM branch targets anomaly reasoning/semantic interpretation, not detector replacement. |

## 11. Recommended Final Claim

```text
Localization-guided context-aware crops improve VLM anomaly reasoning on the unified MVTec AD 2 primary-category evaluation, but the benefit is category-dependent and sensitive to detector/candidate quality.
```

## 12. Next Step

Stage 12-C should draft the full Introduction and Method sections in paper-style language, using this outline as the source of truth.

## 13. Output

- Section plan: `results/stage12_paper_writing/stage12_b_section_plan.csv`
- Title/abstract candidates: `results/stage12_paper_writing/stage12_b_title_abstract_candidates.csv`
- Paper outline: `docs/stage12_paper_writing/stage12_b_paper_outline.md`