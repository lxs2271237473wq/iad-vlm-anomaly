# Paper Stage P9: Citation-marked Compact Draft

This file is generated from the P7 compact draft with first-pass citation commands inserted.
It is still a Markdown/LaTeX hybrid draft; final venue formatting is not done.

---
# Paper Stage P7: LaTeX-style Compact Draft

## Title

**Quality-Calibrated Localization-Guided VLM Reasoning for Industrial Anomaly Recognition**

## Abstract

Industrial anomaly recognition with general-purpose vision-language models remains unreliable when defects are small, localized, and visually subtle. We propose Quality-Calibrated QCR, a localization-guided VLM reasoning framework that converts detector localization evidence into candidate-level visual-language evidence and calibrates crop-level VLM anomaly scores using candidate quality. The method treats candidate quality as the main reliability mechanism and uses detector-VLM consistency only as a conservative adaptive refinement. Experiments with detector, VLM, and external anomaly baselines show that localization-guided VLM evidence complements detector scores. Ablations further show that quality calibration provides the main gain over naive detector-crop fusion, while adaptive consistency yields only a small refinement and fixed consistency should remain diagnostic. Failure and boundary analyses clarify when quality calibration helps, when it fails, and why broad claims about segmentation or manufacturing-cause reasoning are unsupported.

## 1. Introduction

Industrial anomalies often occupy small localized regions and may not dominate global image semantics~\cite{Bergmann2019MVTecAD,Zou2022VisA}. This makes direct full-image VLM reasoning unreliable for industrial inspection. A detector can provide localization evidence, while a VLM can provide visual-language abnormality evidence; however, naive fusion does not model whether the localized crop is reliable. We address this gap by calibrating crop-level VLM scores with candidate quality derived from localization evidence.

Our method, **Quality-Calibrated QCR**, uses detector localization to generate candidate crops, obtains crop-level VLM anomaly evidence, and modulates that evidence using candidate quality. We also study detector-VLM consistency, but robustness analysis shows that fixed consistency is not reliable enough to be the final method. Thus, consistency is retained only as a small adaptive refinement.

Our contributions are:

1. We formulate industrial anomaly recognition as localization-guided VLM evidence reasoning.
2. We propose candidate-quality calibration as the main reliability mechanism for crop-level VLM evidence.
3. We analyze fixed and adaptive detector-VLM consistency and show why consistency must be treated conservatively.
4. We provide strong baseline comparisons, ablations, boundary analysis, and fixed-budget sensitivity checks with explicit claim restrictions.

## 2. Related Work

**Industrial anomaly detection.** PatchCore~\cite{Roth2022PatchCore}, FastFlow-style detectors~\cite{Yu2021FastFlow}, and EfficientAD~\cite{Batzner2024EfficientAD} provide strong anomaly detection and localization evidence. Our method does not replace them; it uses localization evidence to produce candidate-level VLM evidence.

**Vision-language anomaly detection.** CLIP~\cite{Radford2021CLIP}-based anomaly methods such as WinCLIP~\cite{Jeong2023WinCLIP} and AnomalyCLIP~\cite{Zhou2024AnomalyCLIP} show the value of language-supervised representations. Our work is narrower: it studies how detector-guided crop evidence should be calibrated before being trusted by a VLM-based anomaly scorer.

**Reliability calibration.** Naive score fusion assumes detector and VLM scores are equally reliable. We show that candidate quality is the stable calibration signal, whereas fixed consistency is not robust enough to be the final method.

## 3. Method

Given image $x$, a detector produces localization evidence $A$ and normalized anomaly score $D$. Candidate crops $C=\{c_i\}$ are generated from $A$. The VLM scores each crop and the crop-level scores are aggregated into $M$. Candidate quality $Q$ measures the reliability of the localized crop evidence.

The naive detector-crop fusion baseline is:

$$S_{\mathrm{naive}} = 0.5D + 0.5M.$$

Quality-Calibrated QCR modulates the VLM contribution by candidate quality:

$$S_{\mathrm{quality}} = 0.5D + 0.5M(0.5 + 0.5Q).$$

A diagnostic fixed Q+C variant is:

$$S_{\mathrm{fixed}} = 0.4D + 0.4M + 0.1Q + 0.1K,$$

where $K$ is detector-VLM high-high consistency. This variant is diagnostic only because fixed consistency is not robust across protocols.

The adaptive refinement uses a conservative gate:

$$g = QK(1-|D-M|)\min(D,M),$$

$$S_{\mathrm{adaptive}} = S_{\mathrm{quality}} + 0.05g.$$

The main method is $S_{\mathrm{quality}}$. The adaptive variant is reported as a refinement, not as the main performance source.

## 4. Experiments

We evaluate two complementary views. The system-level view compares VLM baselines, detector baselines, and localization-guided fusion, using VisA-style industrial anomaly evaluation context~\cite{Zou2022VisA}. The QCR view isolates detector-only scoring, crop VLM scoring, naive fusion, quality calibration, fixed Q+C, and adaptive refinement under the QCR primary protocol. Image AUROC is the main metric; pixel metrics are auxiliary only.

## 5. Main Results

Full-image VLM obtains mean AUROC `0.6459`, while context-aware VLM obtains `0.7101`. PatchCore~\cite{Roth2022PatchCore} obtains `0.7853` and EfficientAD-30 fixed-budget~\cite{Batzner2024EfficientAD} obtains `0.7604`. The fair LOCO fusion reaches `0.8210`, improving over PatchCore by `+0.0356` and over EfficientAD-30 by `+0.0606`. The same-set result `0.8453` is reported only as an upper-bound diagnostic.

# Table 1. System-level strong baseline comparison

| Rank | Method | Mean AUROC | Paper role | Protocol tag |
| --- | --- | --- | --- | --- |
| 1 | PatchCore + context VLM, same-set | 0.8453 | upper_bound_diagnostic_only | mean_summary |
| 2 | PatchCore + context VLM, LOCO | 0.8210 | primary_fair_system_result | mean_summary |
| 3 | PatchCore | 0.7853 | classic_detector_baseline | mean_summary |
| 4 | EfficientAD-30 fixed-budget | 0.7604 | modern_detector_fixed_budget_baseline | mean_summary |
| 5 | context-aware VLM | 0.7101 | vlm_baseline | mean_summary |
| 6 | full-image VLM | 0.6459 | vlm_baseline | mean_summary |
| 7 | WinCLIP fixed protocol | 0.6138 | external_vlm_anomaly_baseline | mean_summary |

**Note.** LOCO is the fair system-level result. Same-set is an upper-bound diagnostic only.


## 6. Ablation Study

Under the QCR primary protocol, naive detector-crop fusion obtains mean AUROC `0.9652`. Quality-Calibrated QCR obtains `0.9748`, a gain of `+0.0096`. The adaptive refinement obtains `0.9752`, improving over naive fusion by `+0.0100` but over the quality core by only `+0.0004`. Therefore, the main effective component is candidate quality calibration, not adaptive consistency.

# Table 2. QCR primary-protocol ablation

| ID | Method | Role | FastFlow | PatchCore | Mean AUROC |
| --- | --- | --- | --- | --- | --- |
| V0 | Detector only | anchor_baseline | 0.8955 | 0.9131 | 0.9043 |
| V2 | Crop VLM only | vlm_crop_baseline | 0.9269 | 0.8846 | 0.9057 |
| V3 | Naive detector-crop fusion | naive_fusion_baseline | 0.9688 | 0.9616 | 0.9652 |
| V4 | Quality-Calibrated QCR | main_effective_method_core | 0.9778 | 0.9718 | 0.9748 |
| V5 | Fixed Q+C fusion | diagnostic_not_final | 0.9842 | 0.9740 | 0.9791 |
| V6 | Quality-Calibrated QCR + adaptive consistency refinement | final_refinement_variant | 0.9783 | 0.9722 | 0.9752 |

**Note.** Quality-Calibrated QCR is the main method core. Fixed Q+C is diagnostic only.


## 7. Failure and Boundary Analysis

Boundary analysis shows that quality calibration is useful but not universal. It can help by boosting true localized anomalies or suppressing normal false positives, but it can also fail when candidate quality is misleading or detector-VLM evidence disagrees. Fixed Q+C can peak in some cases but is not robust enough to be final.

# Table 3. Boundary and robustness summary

| Comparison | Mean delta | Wins | Min | Max | Interpretation |
| --- | --- | --- | --- | --- | --- |
| Quality-Calibrated QCR - Naive fusion | +0.0119 | 13/24 | -0.0132 | +0.0908 | Quality calibration effect |
| Adaptive refinement - Quality-Calibrated QCR | +0.0003 | 14/24 | +0.0000 | +0.0014 | Adaptive consistency refinement effect |
| Fixed Q+C - Quality-Calibrated QCR | +0.0027 | 18/24 | -0.0506 | +0.0202 | Diagnostic fixed consistency effect |
| Adaptive refinement - Fixed Q+C | -0.0024 | 3/24 | -0.0189 | +0.0508 | Robustness tradeoff |

**Note.** Per-category deltas show quality calibration is useful but not universal; adaptive consistency is only a refinement.


## 8. EfficientAD Budget Sensitivity

A 100-epoch EfficientAD sensitivity check on fruit_jelly gives image-AUROC delta `-0.0167` relative to EfficientAD-30. This does not indicate severe image-level underestimation of EfficientAD-30, but it remains a fixed-budget baseline and should not be described as full EfficientAD defeat.

# Appendix Table A1. EfficientAD-100 fruit_jelly sensitivity

| Metric | EfficientAD-30 | EfficientAD-100 | Delta 100-30 |
| --- | --- | --- | --- |
| image_AUROC | 0.8433 | 0.8267 | -0.0167 |
| image_F1Score | 0.8571 | 0.8438 | -0.0134 |
| pixel_AUROC | 0.7894 | 0.8424 | +0.0531 |
| pixel_F1Score | 0.5395 | 0.5561 | +0.0166 |

**Note.** This is a defensive fixed-budget sensitivity check, not a full EfficientAD sweep.


## 9. Limitations

The method is image-level anomaly recognition and candidate-level VLM evidence calibration. It does not claim pixel-level segmentation SOTA, manufacturing-cause reasoning, or full anomaly understanding. EfficientAD is fixed-budget, AnomalyCLIP~\cite{Zhou2024AnomalyCLIP} is not included, and adaptive consistency has only a small gain over quality-only.

## 10. Conclusion

Quality-Calibrated QCR provides a reliability-calibrated bridge between industrial anomaly localization and VLM anomaly evidence. Candidate quality is the main effective component, while adaptive consistency is a conservative refinement. The evidence supports a cautious but coherent claim: localization-guided VLM anomaly recognition becomes more reliable when crop-level evidence is calibrated by candidate quality.

## Appendix A. Compact Table Inventory

| table_id | title | path | paper_location | status |
| --- | --- | --- | --- | --- |
| Table 1 | System-level strong baseline comparison | docs/paper_p7/tables/table1_system_baselines.md | Main Results | generated |
| Table 2 | QCR primary-protocol ablation | docs/paper_p7/tables/table2_qcr_ablation.md | Ablation Study | generated |
| Table 3 | Boundary and robustness summary | docs/paper_p7/tables/table3_boundary_summary.md | Failure / Boundary Analysis | generated |
| Appendix Table A1 | EfficientAD-100 fruit_jelly sensitivity | docs/paper_p7/tables/table4_efficientad_sensitivity.md | Appendix / Baseline Budget Sensitivity | generated |

## Appendix B. Draft Checklist

| item_id | item | status | next_action |
| --- | --- | --- | --- |
| P7-C1 | Compact draft generated | done | Manual polish after figures and references are added. |
| P7-C2 | Paper-facing compact tables generated | done | Convert Markdown tables to LaTeX tabular/booktabs. |
| P7-C3 | Framework figure missing | missing | Create Figure 1 pipeline schematic. |
| P7-C4 | Boundary case figure missing | missing | Inspect Stage 16-E case inventory and select representative images. |
| P7-C5 | BibTeX missing | missing | Prepare references for MVTec AD, VisA, PatchCore, FastFlow, EfficientAD, CLIP, WinCLIP, AnomalyCLIP. |
| P7-C6 | AnomalyCLIP risk unresolved | open | Either run it later or keep it explicitly as limitation. |
