# Paper Stage P12: Final Figure 2 Montage

## 1. Summary

- final panels: `5`
- panel order: `A, B, C, E, F`
- dropped panel: `D`
- montage: `docs/paper_p12/figures/figure2_boundary_cases_montage.png`
- paper copy: `paper/quality_calibrated_qcr/figures/figure2_boundary_cases_montage.png`
- LaTeX snippet: `docs/paper_p12/figure2_latex_snippet.tex`

## 2. Final panel manifest

| Panel | Source | Case type | Category | GT | Purpose | Asset |
|---|---|---|---|---:|---|---|
| A | P11 | quality_helps_anomaly_boost | chewinggum | 1 | True anomaly with reliable candidate quality. | `docs/paper_p12/figure2_final_assets/panel_A_A_chewinggum_test_anomaly_015.JPG` |
| B | P11 | quality_helps_normal_suppression | pcb1 | 0 | Normal image with high VLM evidence suppressed by low quality. | `docs/paper_p12/figure2_final_assets/panel_B_B_pcb1_test_normal_0490.JPG` |
| C | P11 | quality_boundary_anomaly_suppression | pipe_fryum | 1 | True anomaly suppressed when quality is misleading. | `docs/paper_p12/figure2_final_assets/panel_C_C_pipe_fryum_test_anomaly_032.JPG` |
| E | P11-B | fixed_consistency_boundary_normal_boost | pcb3 | 0 | Fixed consistency boosts a normal case. | `docs/paper_p12/figure2_final_assets/panel_E_E_replacement_5_pcb3_test_normal_0402.JPG` |
| F | P11 | detector_vlm_disagreement_boundary | pipe_fryum | 1 | Detector high, VLM low: evidence conflict. | `docs/paper_p12/figure2_final_assets/panel_F_F_pipe_fryum_test_anomaly_096.JPG` |

## 3. Caption

```text
Representative boundary cases for quality-calibrated localization-guided VLM reasoning. Quality calibration can boost true anomaly evidence and suppress normal false positives, but it can also fail when candidate quality is misleading. Fixed Q+C can over-boost normal evidence, and detector-VLM disagreement remains a boundary case. These examples illustrate method boundaries rather than universal behavior.
```

## 4. Notes

- D is intentionally dropped because replacement candidates were weak.
- E uses the P11-B selected replacement: fixed-consistency normal boost, pcb3, FastFlow.
- The figure is boundary analysis, not proof of universal behavior.
- The figure must not be described as segmentation output or manufacturing-cause explanation.

## 5. Next step

```text
Paper Stage P13: compile-check LaTeX manuscript and patch figure/table issues
```
