# Stage 12-C Introduction and Method Draft

## 1. Purpose

This document drafts the Introduction and Method sections using the Stage 11 evidence and Stage 12-B paper outline.
It is a paper-writing artifact only. It does not run models, regenerate crops, or modify experimental results.

---

# Introduction Draft

Industrial anomaly detection aims to identify defective or abnormal products under limited supervision, often using only normal training samples. Classical anomaly detection methods have achieved strong localization and detection performance by modeling normal visual patterns and highlighting deviations at test time. However, these methods usually produce numerical anomaly scores and pixel-level heatmaps, while downstream industrial inspection often requires a more interpretable reasoning interface: users need to understand what visual evidence indicates an abnormality and how the suspected region relates to the product structure.

Recent visual-language models provide a potential interface for such anomaly understanding because they can compare visual content with textual descriptions of normal and abnormal states. A straightforward strategy is to apply full-image VLM prompting to industrial inspection images. This strategy is simple, but it is often suboptimal for industrial defects. Many defects are small, local, and visually weak compared with the complete product appearance. When the whole image is passed to the VLM, the abnormal evidence can be diluted by large normal regions and background context.

An opposite strategy is to crop the most anomalous region and feed only that local patch to the VLM. This also has an important limitation. Industrial anomalies are not purely local texture patterns; whether a visual cue is abnormal often depends on object-level context. A small crop may remove the surrounding product structure, making it difficult for the VLM to determine whether the local appearance is a defect or a normal part of the object.

This motivates a localization-guided and context-aware VLM reasoning framework. Instead of treating classical anomaly localization as the final output, we use it as a visual bridge between anomaly detectors and VLM reasoning. The anomaly detector first proposes suspicious regions from anomaly maps. These regions are then converted into context-aware crops that preserve both local abnormal evidence and surrounding object semantics. Finally, a VLM scores normal-versus-abnormal prompt margins over full images, tight crops, and context-aware crops.

The key design principle is that the VLM branch should not simply receive the smallest anomalous patch. It should receive a region that is sufficiently localized to focus on defect evidence, but sufficiently contextualized to maintain product semantics. This distinction is central to our method and avoids reducing the work to a trivial crop-and-prompt pipeline.

We evaluate this idea on MVTec AD 2 using a unified multi-category pipeline. Our results show a conditional but meaningful benefit: on the primary-category aggregate, full-image VLM prompting obtains AUROC 0.5201, while the reported context-aware crop aggregation method `context_1.50_topk_mean` obtains AUROC 0.6036, improving by 0.0835. At the same time, category-level analysis shows that this benefit is not universal. Positive evidence appears on some categories, while other categories reveal sensitivity to detector localization quality and candidate construction.

The main contributions are summarized as follows:

1. We formulate anomaly localization as a visual bridge for VLM-based industrial anomaly reasoning, rather than presenting the detector itself as the contribution.
2. We introduce a context-aware crop construction strategy that preserves object-level semantics around detector-localized suspicious regions.
3. We provide a detector-quality-aware evaluation protocol that jointly reports detector quality, candidate quality, VLM reasoning performance, and limitation cases.
4. We conduct a unified MVTec AD 2 multi-category study showing aggregate improvement of context-aware crop reasoning over full-image VLM prompting, while explicitly documenting category-dependent failures.

---

# Method Draft

## 1. Overview

Given an industrial inspection image, the proposed framework produces VLM-based anomaly reasoning scores through four stages: anomaly localization, candidate region proposal, context-aware crop construction, and VLM prompt scoring. The detector and the VLM play different roles. The detector provides spatial priors through anomaly maps, while the VLM evaluates visual-language normality and abnormality cues over image regions.

The complete pipeline is:

```text
input image -> anomaly map -> candidate region proposal -> context-aware crop construction -> VLM anomaly reasoning
```

## 2. Anomaly Localization as a Visual Bridge

We first apply a classical anomaly localization model to the test image. In our implementation, PatchCore is used as the localization backbone. The role of PatchCore is not to serve as a new detector contribution, but to provide anomaly maps and image-level detector scores. The anomaly map highlights spatial regions that deviate from normal training samples.

Let an inspection image be denoted as `I`. The anomaly detector produces an anomaly map `A`, where higher values indicate stronger local abnormality. This map is used to construct candidate regions for downstream VLM reasoning.

## 3. Candidate Region Proposal

Candidate regions are extracted from high-response areas in the anomaly map. Connected components or top-response regions are converted into bounding boxes. Each candidate has a detector confidence derived from the anomaly map response. Multiple candidates can be retained for one image, allowing the VLM stage to aggregate evidence across suspicious regions.

This stage produces tight candidate boxes. These tight boxes focus on the most suspicious visual areas but may not contain enough object context for VLM reasoning.

## 4. Context-aware Crop Construction

To address the context loss of tight crops, each candidate box is expanded into a context-aware crop. The expansion preserves surrounding product structure while still focusing on the detector-localized suspicious region. This is important because industrial defects are often defined relative to the product geometry, material, or normal manufacturing pattern.

We therefore evaluate both tight crops and context-aware crops. The tight crop tests whether local abnormal evidence alone is sufficient. The context-aware crop tests whether adding object context improves VLM reasoning. The final method emphasizes the latter because Stage 11 evidence shows that context-aware aggregation improves the primary-category aggregate result.

## 5. VLM Prompt Scoring

For each full image or crop, the VLM computes similarity scores against normal and abnormal textual prompts. The anomaly score is defined as the margin between abnormal-prompt similarity and normal-prompt similarity. Category-aware prompts are used to describe normal and defective industrial products.

For each image, we evaluate:

- full-image VLM score;
- tight crop top-1 score;
- tight crop top-k maximum and mean scores;
- context-aware crop top-1 score;
- context-aware crop top-k maximum and mean scores;
- PatchCore image score as a detector reference.

PatchCore score is reported only as a detector reference. It should not be interpreted as a VLM reasoning method.

## 6. Detector-quality-aware Evaluation

Crop-based VLM reasoning depends strongly on localization quality. If the detector proposes poor candidate regions, VLM crop scores may fail for reasons unrelated to the VLM itself. Therefore, categories are separated into primary, secondary, and detector-risk groups according to detector quality and localization reliability.

Primary categories are used for the main VLM crop comparison. Secondary categories are used as boundary cases. Detector-risk categories are excluded from the main VLM evidence because weak localization would make crop-based reasoning unfair or difficult to interpret.

## 7. Current Evidence Used by the Method Section

The method should be described with the following evidence constraints:

| Evidence Type | Current Result | Usage |
|---|---|---|
| Main aggregate | `context_1.50_topk_mean` improves AUROC from 0.5201 to 0.6036 | Main positive evidence |
| Positive categories | fruit_jelly and walnuts | Category-level support |
| Limitation categories | sheet_metal, vial, and fabric | Boundary/failure analysis |
| Detector-risk categories | can, rice, wallplugs | Excluded from main VLM evidence |

## 8. Recommended Method Claim

The method section should use this claim:

```text
Localization-guided context-aware crops improve VLM anomaly reasoning when candidate regions preserve sufficient object context and detector localization is reliable.
```

It should avoid this claim:

```text
Context-aware crops consistently improve all industrial anomaly categories.
```

---

# Writing Notes

- Do not overstate novelty of PatchCore.
- Do not claim detector SOTA.
- Do not mix Stage 10-G vial with the Stage 11 unified main table.
- Present category-dependent behavior as part of the analysis, not as a hidden weakness.
- Use `context-aware crop construction` and `localization as a visual bridge` as the two central phrases.

# Output Dependency

- Main evidence table: `results/stage11_mvtecad2_multicategory/stage11_i_paper_ready_main_table.csv`
- Claim evidence table: `results/stage12_paper_writing/stage12_a_claim_evidence_table.csv`
- Paper outline: `docs/stage12_paper_writing/stage12_b_paper_outline.md`
- This draft: `docs/stage12_paper_writing/stage12_c_intro_method_draft.md`