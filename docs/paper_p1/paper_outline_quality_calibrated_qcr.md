# Paper Stage P1: Paper Outline from Evidence

## 1. Working Title Options

Preferred title:

```text
Quality-Calibrated Localization-Guided VLM Reasoning for Industrial Anomaly Recognition
```

Shorter method-oriented title:

```text
Quality-Calibrated QCR for Localization-Guided VLM Anomaly Recognition
```

Avoid titles centered on fixed QCR-U or universal consistency.

## 2. Core Thesis

The paper should argue:

```text
Industrial anomaly VLM reasoning is more reliable when it is guided by detector localization and calibrated by candidate quality. The main effective mechanism is quality calibration of crop-level VLM evidence. Detector-VLM consistency is useful only as a conservative adaptive refinement, not as the primary source of improvement.
```

## 3. Main Contributions

### Contribution 1: Localization-guided VLM anomaly recognition

The method converts anomaly localization evidence into candidate-level VLM evidence instead of relying on full-image VLM inference.

Evidence: context-aware VLM mean AUROC `0.7101` vs full-image VLM `0.6459`.

### Contribution 2: Quality-calibrated candidate reasoning

The main method core calibrates crop-level VLM evidence using candidate quality derived from localization evidence.

Evidence: Quality-Calibrated QCR vs naive fusion delta `+0.0096` in the QCR primary protocol.

### Contribution 3: Boundary-aware adaptive consistency

The paper analyzes detector-VLM consistency and shows fixed consistency is not robust enough to be the final method. Adaptive consistency is retained only as a small conservative refinement.

Evidence: adaptive refinement vs quality core delta `+0.0004`.

### Contribution 4: Strong baseline and claim discipline

The paper includes WinCLIP, EfficientAD-30 fixed-budget, PatchCore, VLM baselines, LOCO fusion, ablations, and failure/boundary analysis.

Evidence: LOCO fusion `0.8210` vs PatchCore `0.7853` and EfficientAD-30 `0.7604`.

## 4. Abstract Skeleton

```text
Industrial anomaly recognition with general-purpose vision-language models remains unreliable when images are evaluated globally. We study a localization-guided formulation that converts detector localization evidence into candidate-level visual-language evidence. To make crop-level VLM scores reliable, we propose Quality-Calibrated QCR, which calibrates VLM anomaly evidence using candidate quality derived from anomaly localization. We further analyze detector-VLM consistency and find that fixed consistency is not robust; therefore, consistency is used only as a conservative adaptive refinement. Experiments with detector, VLM, and external anomaly baselines show that localization-guided VLM evidence complements detector scores, while ablations identify candidate quality as the main effective component. Failure analysis clarifies the method boundaries under misleading localization and detector-VLM disagreement.
```

## 5. Proposed Paper Structure

### 1. Introduction

- Motivate industrial anomaly recognition.
- Explain why full-image VLM is weak for localized industrial defects.
- Introduce localization-guided VLM evidence.
- State that naive fusion is insufficient and requires reliability calibration.
- Present Quality-Calibrated QCR as the method family.

### 2. Related Work

- Industrial anomaly detection: PatchCore, EfficientAD, FastFlow-type detectors.
- Vision-language anomaly detection: WinCLIP and related CLIP/VLM anomaly baselines.
- Localization-guided reasoning and candidate-based evidence.
- Calibration/reliability in multimodal scoring.

### 3. Method

- Detector localization and candidate extraction.
- Crop-level VLM anomaly evidence.
- Candidate quality score.
- Quality-Calibrated QCR formula.
- Adaptive consistency refinement.
- Explicitly state fixed Q+C is diagnostic, not final.

### 4. Experimental Setup

- Datasets and primary categories.
- Baselines: full-image VLM, context-aware VLM, WinCLIP, PatchCore, EfficientAD-30 fixed-budget.
- Protocols: LOCO vs same-set; QCR primary protocol.
- Metrics: image AUROC as main metric; pixel metrics auxiliary only.

### 5. Main Results

- Panel A: system-level baseline comparison.
- Use LOCO as fair system-level result.
- Use same-set only as diagnostic upper bound.

### 6. Ablation Study

- Detector only.
- Crop VLM only.
- Naive fusion.
- Quality-Calibrated QCR.
- Fixed Q+C diagnostic.
- Adaptive consistency refinement.

### 7. Failure and Boundary Analysis

- Quality helps anomaly boost.
- Quality helps normal suppression.
- Quality can suppress true anomalies when candidate quality is misleading.
- Fixed consistency can mislead.
- Detector-VLM disagreement cases.

### 8. Limitations

- No full-budget EfficientAD sweep.
- No AnomalyCLIP yet.
- No pixel-level SOTA claim.
- No manufacturing-cause reasoning claim.
- Adaptive consistency has small gain over quality-only.

## 6. Paper Tables and Figures

| ID | Title | Source | Must Include | Notes |
|---|---|---|---:|---|
| Table 1 | System-level strong baseline comparison | `results/stage16_qcru_ablation/stage16_d_paper_facing_system_baseline_table.csv` | 1 | Do not merge with QCR ablation table because protocols differ. |
| Table 2 | QCR primary-protocol ablation | `results/stage16_qcru_ablation/stage16_d_paper_facing_qcr_ablation_table.csv` | 1 | Quality-Calibrated QCR is the main effective core; adaptive consistency is refinement. |
| Table 3 | Claim-ready deltas | `results/stage16_qcru_ablation/stage16_d_paper_facing_claim_ready_deltas.csv` | 1 | Use for text evidence, not necessarily as a full paper table. |
| Table 4 | Failure and boundary summary | `results/stage16_qcru_ablation/stage16_e_category_boundary_summary.csv` | 1 | Important because method improvement is reliability-oriented, not huge SOTA margin. |
| Table 5 | EfficientAD-100 fruit_jelly sensitivity | `results/stage17_defensive_sensitivity/stage17_a_efficientad100_vs_30_delta.csv` | 0 | Use in appendix or footnote; do not claim full EfficientAD defeat. |
| Figure 1 | Framework overview | `to_be_drawn` | 1 | This should be a schematic, not a result table. |
| Figure 2 | Representative boundary cases | `results/stage16_qcru_ablation/stage16_e_failure_boundary_case_inventory.csv` | 1 | Images must be manually inspected before paper use. |

## 7. Key Numbers to Use

### System-level results

- full-image VLM: `0.6459`
- context-aware VLM: `0.7101`
- WinCLIP fixed protocol: `0.6138`
- EfficientAD-30 fixed-budget: `0.7604`
- PatchCore: `0.7853`
- PatchCore + context VLM, LOCO: `0.8210`
- PatchCore + context VLM, same-set upper bound: `0.8453`

### QCR method results

- Quality-Calibrated QCR vs naive fusion: `+0.0096`
- adaptive refinement vs quality core: `+0.0004`
- adaptive refinement vs naive fusion: `+0.0100`

### EfficientAD-100 sensitivity

- EfficientAD-100 minus EfficientAD-30 image_AUROC on fruit_jelly: `-0.0167`
- EfficientAD-100 minus EfficientAD-30 pixel_AUROC on fruit_jelly: `+0.0531`

Interpretation: the image-level EfficientAD-100 check does not show severe underestimation of EfficientAD-30. Pixel improvement is auxiliary and should not become the main claim.

## 8. Forbidden Claims

| Claim ID | Forbidden | Allowed Replacement |
|---|---|---|
| P6 | Fixed Q+C is the proposed final method. | Fixed consistency can peak in some settings but lacks robustness, so it is not used as the final method. |
| P7 | Same-set fusion is the primary deployment result. | Same-set fusion is reported only as a diagnostic upper bound. |
| P9 | The method achieves pixel-level segmentation SOTA. | Pixel-level/localization signals are used to generate candidate evidence for image-level anomaly recognition. |

## 9. Remaining Risks

| Risk ID | Risk | Severity | Mitigation | Paper Handling |
|---|---|---|---|---|
| R1 | EfficientAD is still not a full four-category 100-epoch baseline. | medium | Stage 17-A fruit_jelly sensitivity shows EfficientAD-100 does not improve image_AUROC over EfficientAD-30 on fruit_jelly. Keep EfficientAD as fixed-budget. | Label EfficientAD as EfficientAD-30 fixed-budget. Do not claim full EfficientAD defeat. |
| R2 | AnomalyCLIP is not included. | medium_high | Avoid broad CLIP-family SOTA claims. Present WinCLIP as the fixed external VLM anomaly baseline used in this study. | Mention as future baseline extension or limitation. |
| R3 | Adaptive consistency gain over quality-only is very small. | medium | Do not present adaptive consistency as the main innovation. The main method is quality calibration. | Write adaptive consistency as conservative refinement only. |
| R4 | Quality calibration is not universally positive per category. | medium | Use Stage 16-E boundary analysis. Claim reliability calibration, not universal improvement. | Include failure/boundary section. |
| R5 | Method may look heuristic because it uses score fusion. | medium_high | Emphasize fixed protocol, ablations, robustness checks, and boundary analysis. | Avoid overclaiming; frame as reliability calibration for localization-guided VLM evidence. |
| R6 | Pixel-level claims are weak. | high_if_overclaimed | Do not claim segmentation SOTA. | Frame localization as candidate generation evidence, not final segmentation output. |

## 10. Next Writing Step

Next stage should be:

```text
Paper Stage P2: draft Introduction + Contributions + Method Overview
```

Do not start by writing the full paper. First draft the Introduction and Method Overview using the locked claims above.

## 11. Outputs

- `docs/paper_p1/paper_outline_quality_calibrated_qcr.md`
- `results/paper_p1/paper_table_inventory.csv`
- `results/paper_p1/paper_remaining_risks.csv`
