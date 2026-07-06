# Figure 2 Boundary Case Selection Plan

## Purpose

Figure 2 should visually support the paper's boundary-aware claim:

```text
Quality calibration is the main reliability mechanism, but it is not universally correct. Fixed consistency can be risky, and detector-VLM disagreement remains a boundary case.
```

This file is a selection plan only. Do not use these panels in the paper until the original images/crops are manually inspected.

## Selected Panels

| Panel | Case Type | Category | Image Key | GT | Purpose | Manual Status |
|---|---|---|---|---:|---|---|
| A | quality_helps_anomaly_boost | chewinggum | datasets/VisA_anomalib_1cls/chewinggum/test/anomaly/chewinggum_test_anomaly_015.JPG | 1 | Positive example: quality calibration boosts true anomaly evidence. | selected_for_manual_inspection |
| B | quality_helps_normal_suppression | pcb1 | datasets/VisA_anomalib_1cls/pcb1/test/good/pcb1_test_normal_0490.JPG | 0 | Positive example: quality calibration suppresses normal false-positive evidence. | selected_for_manual_inspection |
| C | quality_boundary_anomaly_suppression | pipe_fryum | datasets/VisA_anomalib_1cls/pipe_fryum/test/anomaly/pipe_fryum_test_anomaly_032.JPG | 1 | Boundary case: quality calibration can suppress true anomaly evidence. | selected_for_manual_inspection |
| D | quality_boundary_normal_boost | macaroni2 | datasets/VisA_anomalib_1cls/macaroni2/test/good/macaroni2_test_normal_0159.JPG | 0 | Boundary case: quality calibration can boost normal evidence. | selected_for_manual_inspection |
| E | fixed_consistency_boundary_normal_boost | pcb1 | datasets/VisA_anomalib_1cls/pcb1/test/good/pcb1_test_normal_0490.JPG | 0 | Explains why fixed Q+C can be risky. | selected_for_manual_inspection |
| F | detector_vlm_disagreement_boundary | pipe_fryum | datasets/VisA_anomalib_1cls/pipe_fryum/test/anomaly/pipe_fryum_test_anomaly_096.JPG | 1 | Shows detector-VLM conflict as a method boundary. | selected_for_manual_inspection |

## Required Manual Inspection

For each selected panel, inspect:

1. Original image.
2. Detector heatmap or localization evidence, if available.
3. Candidate crop used for VLM scoring.
4. Whether the case visually matches the intended paper purpose.
5. Whether the case could accidentally overclaim segmentation or cause reasoning.

## Panel Interpretation

- Panel A/B should show positive behavior of quality calibration.
- Panel C/D should show quality calibration boundaries.
- Panel E should explain why fixed Q+C is diagnostic only.
- Panel F should explain detector-VLM disagreement.
