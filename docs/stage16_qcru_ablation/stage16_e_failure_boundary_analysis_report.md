# Stage 16-E Failure Cases and Boundary Analysis

## 1. Purpose

Stage 16-D created the paper-facing main comparison. Stage 16-E explains method boundaries.

This stage does not train models or rerun VLM inference. It mines the existing Stage 9 prediction table for representative boundary cases.

## 2. Primary Scope

The case inventory uses the QCR primary protocol:

```text
dataset = VisA
strategy = inspection_binary
eval_mode = crop_topk_ensemble
```

## 3. Category-level Boundary Summary

| Backbone | Category | V3 Naive | V4 Quality | V5 Fixed Q+C | V6 Adaptive | V4-V3 | V6-V4 | V5-V4 | Boundary Label |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| FastFlow | capsules | 0.9858 | 0.9767 | 0.9950 | 0.9768 | -0.0092 | +0.0002 | +0.0183 | quality_not_helpful;adaptive_gain_negligible;fixed_consistency_can_peak_but_diagnostic |
| FastFlow | cashew | 0.9976 | 0.9946 | 0.9994 | 0.9946 | -0.0030 | +0.0000 | +0.0048 | quality_not_helpful;adaptive_gain_negligible;fixed_consistency_can_peak_but_diagnostic |
| FastFlow | pcb2 | 0.9920 | 0.9893 | 0.9969 | 0.9894 | -0.0027 | +0.0001 | +0.0076 | quality_not_helpful;adaptive_gain_negligible;fixed_consistency_can_peak_but_diagnostic |
| FastFlow | fryum | 1.0000 | 0.9992 | 1.0000 | 0.9992 | -0.0008 | +0.0000 | +0.0008 | quality_not_helpful;adaptive_gain_negligible;fixed_consistency_can_peak_but_diagnostic |
| FastFlow | candle | 0.9937 | 0.9936 | 0.9973 | 0.9937 | -0.0001 | +0.0001 | +0.0037 | quality_not_helpful;adaptive_gain_negligible;fixed_consistency_can_peak_but_diagnostic |
| FastFlow | chewinggum | 1.0000 | 1.0000 | 1.0000 | 1.0000 | +0.0000 | +0.0000 | +0.0000 | quality_not_helpful;adaptive_gain_negligible |
| FastFlow | macaroni1 | 0.9767 | 0.9779 | 0.9891 | 0.9786 | +0.0012 | +0.0007 | +0.0112 | adaptive_gain_negligible;fixed_consistency_can_peak_but_diagnostic |
| FastFlow | pipe_fryum | 0.9862 | 0.9930 | 0.9950 | 0.9930 | +0.0068 | +0.0000 | +0.0020 | adaptive_gain_negligible;fixed_consistency_can_peak_but_diagnostic |
| FastFlow | pcb4 | 0.9638 | 0.9797 | 0.9812 | 0.9808 | +0.0159 | +0.0011 | +0.0015 | fixed_consistency_can_peak_but_diagnostic |
| FastFlow | macaroni2 | 0.8998 | 0.9191 | 0.9393 | 0.9204 | +0.0193 | +0.0013 | +0.0202 | fixed_consistency_can_peak_but_diagnostic |
| FastFlow | pcb1 | 0.8995 | 0.9487 | 0.9464 | 0.9495 | +0.0492 | +0.0008 | -0.0023 | adaptive_gain_negligible |
| FastFlow | pcb3 | 0.9167 | 0.9659 | 0.9703 | 0.9668 | +0.0492 | +0.0009 | +0.0044 | adaptive_gain_negligible;fixed_consistency_can_peak_but_diagnostic |
| PatchCore | capsules | 0.9920 | 0.9788 | 0.9968 | 0.9788 | -0.0132 | +0.0000 | +0.0180 | quality_not_helpful;adaptive_gain_negligible;fixed_consistency_can_peak_but_diagnostic |
| PatchCore | pcb2 | 0.9847 | 0.9803 | 0.9890 | 0.9805 | -0.0044 | +0.0002 | +0.0087 | quality_not_helpful;adaptive_gain_negligible;fixed_consistency_can_peak_but_diagnostic |
| PatchCore | cashew | 1.0000 | 0.9998 | 1.0000 | 0.9998 | -0.0002 | +0.0000 | +0.0002 | quality_not_helpful;adaptive_gain_negligible;fixed_consistency_can_peak_but_diagnostic |
| PatchCore | chewinggum | 1.0000 | 1.0000 | 1.0000 | 1.0000 | +0.0000 | +0.0000 | +0.0000 | quality_not_helpful;adaptive_gain_negligible |
| PatchCore | pipe_fryum | 1.0000 | 1.0000 | 1.0000 | 1.0000 | +0.0000 | +0.0000 | +0.0000 | quality_not_helpful;adaptive_gain_negligible |
| PatchCore | fryum | 0.9978 | 0.9984 | 0.9994 | 0.9984 | +0.0006 | +0.0000 | +0.0010 | adaptive_gain_negligible;fixed_consistency_can_peak_but_diagnostic |
| PatchCore | candle | 0.9952 | 0.9979 | 0.9982 | 0.9979 | +0.0027 | +0.0000 | +0.0003 | adaptive_gain_negligible;fixed_consistency_can_peak_but_diagnostic |
| PatchCore | macaroni1 | 0.9755 | 0.9793 | 0.9882 | 0.9795 | +0.0038 | +0.0002 | +0.0089 | adaptive_gain_negligible;fixed_consistency_can_peak_but_diagnostic |
| PatchCore | pcb4 | 0.9854 | 0.9952 | 0.9896 | 0.9953 | +0.0098 | +0.0001 | -0.0056 | adaptive_gain_negligible |
| PatchCore | pcb3 | 0.9676 | 0.9821 | 0.9849 | 0.9824 | +0.0145 | +0.0003 | +0.0028 | adaptive_gain_negligible;fixed_consistency_can_peak_but_diagnostic |
| PatchCore | macaroni2 | 0.8076 | 0.8622 | 0.8722 | 0.8636 | +0.0546 | +0.0014 | +0.0100 | fixed_consistency_can_peak_but_diagnostic;low_absolute_qcr_auc |
| PatchCore | pcb1 | 0.8508 | 0.9416 | 0.8910 | 0.9418 | +0.0908 | +0.0002 | -0.0506 | adaptive_gain_negligible |

## 4. Case Types Extracted

| Case Type | Meaning | Paper Use |
|---|---|---|
| quality_helps_anomaly_boost | anomaly images whose score is boosted by quality calibration | positive qualitative example |
| quality_helps_normal_suppression | normal images suppressed by quality calibration | false-positive reduction example |
| quality_boundary_anomaly_suppression | anomaly images suppressed by quality calibration | boundary / failure case |
| quality_boundary_normal_boost | normal images boosted by quality calibration | boundary / failure case |
| fixed_consistency_boundary_anomaly_suppression | anomaly images where fixed consistency hurts | explains why fixed Q+C is not final |
| fixed_consistency_boundary_normal_boost | normal images where fixed consistency increases risk | explains false-positive boundary |
| adaptive_refinement_high_gate | images with strongest adaptive gate | explains refinement behavior |
| detector_vlm_disagreement_boundary | images with high detector/VLM disagreement | explains detector-VLM conflict |

Case counts:

| Case Type | Count |
|---|---:|
| quality_helps_anomaly_boost | 10 |
| quality_boundary_anomaly_suppression | 10 |
| fixed_consistency_boundary_anomaly_suppression | 10 |
| quality_helps_normal_suppression | 10 |
| quality_boundary_normal_boost | 10 |
| fixed_consistency_boundary_normal_boost | 10 |
| adaptive_refinement_high_gate | 10 |
| detector_vlm_disagreement_boundary | 10 |

## 5. Boundary Decisions

| Decision ID | Topic | Decision | Paper Action |
|---|---|---|---|
| E1 | quality_calibration | Keep candidate quality calibration as the main method core. | Use as main contribution. |
| E2 | adaptive_consistency | Keep adaptive consistency only as a refinement. | Use with caution; do not call it the main source of improvement. |
| E3 | fixed_consistency | Do not use fixed Q+C as the final method even if it peaks on some categories. | Mention as diagnostic only. |
| E4 | case_inventory | Use selected cases for qualitative boundary analysis. | Inspect representative cases manually before paper figures. |
| E5 | paper_boundary | The method should be claimed as reliability calibration, not full anomaly understanding. | Use boundary-aware wording in paper. |

## 6. Paper Interpretation

The correct interpretation is:

```text
Quality calibration is the main reliability mechanism. It helps when candidate quality aligns with true localized anomaly evidence, but it can still fail when localization quality is misleading or when the VLM and detector disagree. Fixed consistency can produce high peak AUROC in the primary protocol, but it is not robust enough to be the final method. Adaptive consistency is retained only as a conservative refinement.
```

## 7. Claims to Avoid

- Do not claim the method solves all detector localization errors.
- Do not claim consistency is universally beneficial.
- Do not claim adaptive consistency is the main source of improvement.
- Do not claim pixel-level segmentation SOTA.
- Do not claim manufacturing-cause understanding.

## 8. Next Step

Next stage:

```text
Stage 16-F: final claim-evidence map
```

Stage 16-F should map every paper claim to the exact table/result that supports it.

## 9. Outputs

- `results/stage16_qcru_ablation/stage16_e_failure_boundary_case_inventory.csv`
- `results/stage16_qcru_ablation/stage16_e_category_boundary_summary.csv`
- `results/stage16_qcru_ablation/stage16_e_boundary_decision_summary.csv`
- `docs/stage16_qcru_ablation/stage16_e_failure_boundary_analysis_report.md`
