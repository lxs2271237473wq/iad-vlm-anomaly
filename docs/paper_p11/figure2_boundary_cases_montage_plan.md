# Figure 2 Boundary Case Montage Plan

This is not the final figure. It is the manual inspection plan for selecting valid boundary cases.

## Required panels

| Panel | Case type | Category | Asset status | Paper purpose | Manual decision |
|---|---|---|---|---|---|
| A | quality_helps_anomaly_boost | chewinggum | resolved | Positive example: quality calibration boosts true anomaly evidence. | pending |
| B | quality_helps_normal_suppression | pcb1 | resolved | Positive example: quality calibration suppresses normal false-positive evidence. | pending |
| C | quality_boundary_anomaly_suppression | pipe_fryum | resolved | Boundary case: quality calibration can suppress true anomaly evidence. | pending |
| D | quality_boundary_normal_boost | macaroni2 | resolved | Boundary case: quality calibration can boost normal evidence. | pending |
| E | fixed_consistency_boundary_normal_boost | pcb1 | resolved | Explains why fixed Q+C can be risky. | pending |
| F | detector_vlm_disagreement_boundary | pipe_fryum | resolved | Shows detector-VLM conflict as a method boundary. | pending |

## Manual acceptance criteria

Keep a panel only if:

1. the original image is visually interpretable;
2. the visible defect or normal region matches the intended case type;
3. the example illustrates a method boundary without implying universal behavior;
4. the caption can describe it as a representative case, not proof of a general rule;
5. the image does not require unsupported manufacturing-cause explanation.

Reject a panel if:

- the defect is not visible;
- the image/crop does not match the selected case type;
- the example would force an unsupported segmentation or causal claim;
- the asset path cannot be resolved.

## Safe caption wording

```text
Representative boundary cases for quality-calibrated localization-guided VLM reasoning. Quality calibration can boost true anomaly evidence and suppress normal false positives, but it can also fail when candidate quality is misleading or detector and VLM evidence disagree. These examples illustrate method boundaries rather than universal behavior.
```
