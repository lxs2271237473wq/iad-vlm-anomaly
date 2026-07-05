# Stage 16-D Paper-facing Final Comparison

## 1. Purpose

This stage creates the final paper-facing comparison tables after the method claim was locked in Stage 16-C.

The final method family is:

```text
Quality-Calibrated QCR
```

The adaptive consistency term is treated only as a conservative refinement, not as the main performance source.

## 2. Important Comparison Rule

This report uses two panels because Stage 15 system baselines and Stage 16 QCR ablations are not the same protocol.

- Panel A compares system-level baselines from Stage 15.
- Panel B compares QCR variants under the Stage 16-B QCR primary protocol.

Do not merge the two panels into a single global ranking.

## 3. Panel A: System-level Strong Baseline Comparison

| Rank | Method | Mean Image AUROC | Role | Fairness Tag |
|---:|---|---:|---|---|
| 1 | PatchCore + context VLM, same-set | 0.8453 | upper_bound_diagnostic_only | mean_summary |
| 2 | PatchCore + context VLM, LOCO | 0.8210 | primary_fair_system_result | mean_summary |
| 3 | PatchCore | 0.7853 | classic_detector_baseline | mean_summary |
| 4 | EfficientAD-30 fixed-budget | 0.7604 | modern_detector_fixed_budget_baseline | mean_summary |
| 5 | context-aware VLM | 0.7101 | vlm_baseline | mean_summary |
| 6 | full-image VLM | 0.6459 | vlm_baseline | mean_summary |
| 7 | WinCLIP fixed protocol | 0.6138 | external_vlm_anomaly_baseline | mean_summary |

Paper use:

- Use `PatchCore + context VLM, LOCO` as the fair system-level result.
- Use `same-set` only as an upper-bound diagnostic.
- Keep `EfficientAD-30` explicitly labeled as fixed-budget.

## 4. Panel B: QCR Primary-protocol Ablation

| Backbone | Method | Variant | Image AUROC | AP | Best F1 | Role |
|---|---|---|---:|---:|---:|---|
| FastFlow | Detector only | V0 | 0.8955 | 0.9205 | 0.8445 | anchor_baseline |
| FastFlow | Crop VLM only | V2 | 0.9269 | 0.9485 | 0.8765 | vlm_crop_baseline |
| FastFlow | Naive detector-crop fusion | V3 | 0.9688 | 0.9750 | 0.9167 | naive_fusion_baseline |
| FastFlow | Quality-Calibrated QCR | V4 | 0.9778 | 0.9822 | 0.9304 | main_effective_method_core |
| FastFlow | Fixed Q+C fusion | V5 | 0.9842 | 0.9873 | 0.9403 | diagnostic_not_final |
| FastFlow | Quality-Calibrated QCR + adaptive consistency refinement | V6 | 0.9783 | 0.9827 | 0.9312 | final_refinement_variant |
| PatchCore | Detector only | V0 | 0.9131 | 0.9242 | 0.8606 | anchor_baseline |
| PatchCore | Crop VLM only | V2 | 0.8846 | 0.9096 | 0.8340 | vlm_crop_baseline |
| PatchCore | Naive detector-crop fusion | V3 | 0.9616 | 0.9681 | 0.9127 | naive_fusion_baseline |
| PatchCore | Quality-Calibrated QCR | V4 | 0.9718 | 0.9752 | 0.9319 | main_effective_method_core |
| PatchCore | Fixed Q+C fusion | V5 | 0.9740 | 0.9784 | 0.9312 | diagnostic_not_final |
| PatchCore | Quality-Calibrated QCR + adaptive consistency refinement | V6 | 0.9722 | 0.9756 | 0.9320 | final_refinement_variant |

Paper use:

- Treat `Quality-Calibrated QCR` as the main effective method core.
- Treat `Quality-Calibrated QCR + adaptive consistency refinement` as the final conservative refinement.
- Treat `Fixed Q+C fusion` as diagnostic only, because it is not robust across protocols.

## 5. Claim-ready Deltas

| Scope | Comparison | Left Score | Right Score | Delta | Interpretation |
|---|---|---:|---:|---:|---|
| system_panel | LOCO fusion vs PatchCore | 0.8210 | 0.7853 | +0.0356 | Localization-guided VLM evidence complements the detector baseline. |
| system_panel | LOCO fusion vs EfficientAD-30 fixed-budget | 0.8210 | 0.7604 | +0.0606 | LOCO fusion remains above the fixed-budget modern detector baseline; do not claim full EfficientAD defeat. |
| system_panel | LOCO fusion vs WinCLIP fixed protocol | 0.8210 | 0.6138 | +0.2072 | The proposed localization-guided route is stronger than this fixed WinCLIP protocol. |
| system_panel | LOCO fusion vs context-aware VLM | 0.8210 | 0.7101 | +0.1109 | Claim-supporting delta. |
| system_panel | context-aware VLM vs full-image VLM | 0.7101 | 0.6459 | +0.0642 | Localization/context improves over full-image VLM. |
| system_panel | same-set upper bound vs LOCO fair result | 0.8453 | 0.8210 | +0.0243 | Same-set is diagnostic upper bound only; LOCO is the fair result. |
| qcr_primary_protocol | Quality-Calibrated QCR vs naive fusion | 0.9748 | 0.9652 | +0.0096 | Candidate quality calibration is the main method gain. |
| qcr_primary_protocol | Adaptive refinement vs Quality-Calibrated QCR | 0.9752 | 0.9748 | +0.0004 | Adaptive consistency is only a small refinement, not a main contribution. |
| qcr_primary_protocol | Adaptive refinement vs naive fusion | 0.9752 | 0.9652 | +0.0100 | Final refinement variant improves over naive fusion. |
| qcr_primary_protocol | Fixed Q+C vs Quality-Calibrated QCR | 0.9791 | 0.9748 | +0.0043 | Fixed consistency is diagnostic only because robustness is not stable across protocols. |
| qcr_primary_protocol | Adaptive refinement vs fixed Q+C | 0.9752 | 0.9791 | -0.0039 | Adaptive refinement trades peak primary-protocol AUROC for robustness. |
| primary_protocol | adaptive_qcru_minus_naive |  |  | +0.0100 | wins=2/2, win_rate=1.0 |
| primary_protocol | adaptive_qcru_minus_quality |  |  | +0.0004 | wins=2/2, win_rate=1.0 |
| primary_protocol | adaptive_qcru_minus_fixed_qc |  |  | -0.0039 | wins=0/2, win_rate=0.0 |
| primary_protocol | quality_minus_naive |  |  | +0.0096 | wins=2/2, win_rate=1.0 |
| all_protocols | adaptive_qcru_minus_naive |  |  | +0.0371 | wins=12/12, win_rate=1.0 |
| all_protocols | adaptive_qcru_minus_quality |  |  | +0.0006 | wins=12/12, win_rate=1.0 |
| all_protocols | adaptive_qcru_minus_fixed_qc |  |  | +0.0099 | wins=6/12, win_rate=0.5 |
| all_protocols | quality_minus_naive |  |  | +0.0365 | wins=12/12, win_rate=1.0 |

## 6. Final Paper Claims

| Claim ID | Type | Claim | Status |
|---|---|---|---|
| C1 | final_method_name | Use Quality-Calibrated QCR as the main paper-facing method family. | use |
| C2 | main_effective_component | Candidate quality calibration is the main effective component. | use |
| C3 | auxiliary_component | Adaptive consistency is a conservative refinement, not the main source of improvement. | use_with_caution |
| C4 | rejected_claim | Do not claim fixed quality-consistency fusion as the final method. | reject |
| C5 | rejected_claim | Do not claim consistency is universally beneficial. | reject |
| C6 | safe_paper_claim | Localization-guided VLM evidence becomes more reliable when crop evidence is calibrated by candidate quality. | use |
| C7 | safe_paper_claim | Adaptive consistency can be retained as a reliability-gated refinement that avoids overcommitting to unstable fixed consistency. | use_with_caution |
| C8 | final_recommendation | Use Quality-Calibrated QCR as the main method; describe adaptive consistency as a small gated refinement. | use |

## 7. Safe Main Claim

Use this as the central claim:

```text
Localization-guided VLM anomaly recognition becomes more reliable when crop-level VLM evidence is calibrated by candidate quality. Adaptive consistency is retained as a conservative refinement, but the main effective component is candidate quality calibration.
```

## 8. Claims to Avoid

- Do not claim fixed Q+C fusion as the final method.
- Do not claim consistency is universally beneficial.
- Do not claim adaptive consistency is the main source of improvement.
- Do not claim full industrial anomaly understanding.
- Do not claim pixel-level segmentation SOTA.

## 9. Next Step

Next stage:

```text
Stage 16-E: failure cases and boundary analysis
```

Stage 16-E should explain where quality calibration helps, where fixed consistency fails, and where detector localization errors mislead VLM reasoning.

## 10. Outputs

- `results/stage16_qcru_ablation/stage16_d_paper_facing_system_baseline_table.csv`
- `results/stage16_qcru_ablation/stage16_d_paper_facing_qcr_ablation_table.csv`
- `results/stage16_qcru_ablation/stage16_d_paper_facing_claim_ready_deltas.csv`
- `docs/stage16_qcru_ablation/stage16_d_paper_facing_final_comparison_report.md`
