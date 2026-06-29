# Stage 12-A Method and Experiment Narrative

## 1. Purpose

This file converts the Stage 11 evidence tables into a paper-oriented method and experiment narrative.
It does not run models, regenerate crops, or modify experimental results.

## 2. Final Method Framing

The method should be framed as:

```text
classical anomaly localization -> candidate region proposal -> context-aware crop construction -> VLM anomaly reasoning
```

The core idea is not simply to crop the most anomalous patch and send it to a VLM. The important design point is to preserve enough object-level context around detector-localized suspicious regions so that the VLM can jointly observe local abnormal evidence and global product semantics.

## 3. Proposed Method Section Outline

### 3.1 Problem Definition

Given normal training images and test inspection images from industrial categories, the goal is to produce anomaly reasoning scores using a visual-language model without training a new task-specific VLM.

### 3.2 Detector-side Localization Bridge

A PatchCore-style anomaly detector is used as a localization bridge. It is not presented as the final contribution itself; rather, it provides anomaly maps and candidate regions that guide VLM attention toward suspicious visual areas.

### 3.3 Context-aware Candidate Crop Construction

Instead of using only tight anomaly crops, each candidate is expanded into a context-aware crop. This prevents the VLM from losing product-level semantics, which is critical in industrial inspection where the same local texture can be normal or abnormal depending on object context.

### 3.4 VLM Reasoning and Aggregation

The VLM branch compares normal and abnormal textual prompts and computes anomaly margins for full images, tight crops, and context-aware crops. Multi-candidate aggregation is evaluated through top-1, top-k maximum, and top-k mean strategies.

### 3.5 Detector-quality-aware Evaluation

Categories with weak detector quality are not used as main VLM crop evidence. This avoids attributing detector localization failures to the VLM reasoning branch.

## 4. Experiment Section Outline

### 4.1 Dataset and Splits

The main multi-category evaluation is conducted on MVTec AD 2. Public test splits are used for measurable evaluation; private/unlabeled splits are excluded from metric computation.

### 4.2 Detector Baseline

PatchCore is first evaluated on all eight AD2 categories. The detector-quality analysis separates categories into primary, secondary, and detector-risk groups.

### 4.3 Candidate Region Generation

For primary categories, anomaly maps are converted into candidate regions and expanded into context-aware crops. Candidate coverage and GT mask coverage are reported to diagnose crop reliability.

### 4.4 VLM Reasoning Comparison

The main VLM comparison includes full-image prompting, tight crop prompting, context-aware crop prompting, and PatchCore score as a detector reference.

### 4.5 Secondary and Failure Analysis

The secondary category fabric and detector-risk categories are used to show boundary conditions, not to inflate the main claim.

## 5. Main Numeric Result

| Scope | Full-image AUROC | Reported Method | Method AUROC | ΔAUROC | PatchCore Reference |
|---|---:|---|---:|---:|---:|
| ALL_PRIMARY | 0.5201 | context_1.50_topk_mean | 0.6036 | 0.0835 | 0.8087 |

## 6. Method Comparison Table

| Category | Role | Full | Tight top1 | Tight mean | Context top1 | Context mean | Best context Δ | Best VLM | Best VLM Δ | PatchCore |
|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|
| ALL_PRIMARY | primary_aggregate | 0.5201 | 0.5231 | 0.5036 | 0.5773 | 0.6036 | 0.0835 | context_1.50_topk_mean | 0.0835 | 0.8087 |
| fruit_jelly | primary_category | 0.7533 | 0.7367 | 0.6767 | 0.8367 | 0.8567 | 0.1033 | context_1.50_topk_mean | 0.1033 | 0.7167 |
| sheet_metal | primary_category | 0.7130 | 0.4926 | 0.3519 | 0.2444 | 0.5870 | -0.0556 | full_image | 0.0000 | 0.7463 |
| vial | primary_category | 0.6876 | 0.3753 | 0.3973 | 0.6834 | 0.5231 | -0.0042 | full_image | 0.0000 | 0.8732 |
| walnuts | primary_category | 0.4296 | 0.5630 | 0.6067 | 0.5193 | 0.6430 | 0.2133 | context_1.50_topk_mean | 0.2133 | 0.8052 |
| fabric | secondary_category | 0.4168 | 0.4949 | 0.4936 | 0.3980 | 0.3468 | -0.0189 | tight_crop_top1 | 0.0781 | 0.7232 |

## 7. Category Usage Decision

### Positive / supportive categories

- `fruit_jelly`: context Δ=0.1033; Primary detector quality is acceptable and best context crop improves over full-image prompting.
- `walnuts`: context Δ=0.2133; Primary detector quality is acceptable and best context crop improves over full-image prompting.

### Limitation / boundary categories

- `sheet_metal`: context Δ=-0.0556; Primary detector quality is acceptable, but context crop does not improve under the unified Stage 11 pipeline.
- `vial`: context Δ=-0.0042; Primary detector quality is acceptable, but context crop does not improve under the unified Stage 11 pipeline.
- `fabric`: context Δ=-0.0189; Image-level detector is acceptable but pixel localization is weak; context crop is not positive.

### Excluded detector-risk categories

- `can`: image AUROC=0.3901, pixel F1=0.0002; Detector quality is too weak for fair crop-based VLM reasoning.
- `rice`: image AUROC=0.5630, pixel F1=0.0552; Detector quality is too weak for fair crop-based VLM reasoning.
- `wallplugs`: image AUROC=0.4626, pixel F1=0.0391; Detector quality is too weak for fair crop-based VLM reasoning.

## 8. Claim–Evidence Mapping

| Claim ID | Claim | Evidence | Status | Paper Usage |
|---|---|---|---|---|
| C1 | Localization-guided context-aware crops improve VLM anomaly reasoning on the unified MVTec AD 2 primary-category evaluation. | ALL_PRIMARY full-image AUROC=0.5201; context_1.50_topk_mean AUROC=0.6036; delta=0.0835. | main_claim_supported | Main result table and abstract-level claim. |
| C2 | The benefit is category-dependent rather than universal. | fruit_jelly and walnuts are positive; sheet_metal and vial are limitation cases under the unified pipeline. | supported_with_limitations | Discussion and limitations. |
| C3 | Detector/candidate quality must be reported alongside VLM crop reasoning. | fabric is a secondary boundary case; can, rice, and wallplugs are excluded from main VLM evidence due to weak detector quality. | supported | Experiment protocol and fairness discussion. |
| C4 | The method should not be described as merely cropping anomaly regions for a VLM. | The final pipeline requires detector localization, candidate construction, context-aware crop generation, and VLM scoring/aggregation. | method_framing | Method section. |

## 9. Recommended Wording

Use:

```text
Localization-guided context-aware crops improve VLM anomaly reasoning on the unified MVTec AD 2 primary-category evaluation, but the benefit is category-dependent and sensitive to detector/candidate quality.
```

Avoid:

```text
Context-aware crops consistently improve all industrial anomaly categories.
```

## 10. Next Step

Stage 12-B should convert this narrative into a structured paper outline: title, abstract, introduction contributions, method section, experiment section, and limitations.

## 11. Output

- Claim-evidence table: `results/stage12_paper_writing/stage12_a_claim_evidence_table.csv`
- Narrative report: `docs/stage12_paper_writing/stage12_a_method_experiment_narrative.md`