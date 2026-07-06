# Paper Stage P11: Figure 2 Manual Image Montage Preparation

## 1. Updated Manual Review Summary

- selected panels: `6`
- keep: `4`
- replace: `2`
- reject: `0`
- pending: `0`
- unresolved image assets: `0`
- contact sheet: `docs/paper_p11/figures/figure2_boundary_cases_contact_sheet.png`

This stage prepares manual inspection assets only. It does not finalize Figure 2.

## 2. Panel Decisions

| Panel | Case type | Category | Asset status | Manual decision | Paper use allowed | Notes |
|---|---|---|---|---|---|---|
| A | quality_helps_anomaly_boost | chewinggum | resolved | keep | yes | Positive anomaly example: high detector score, high candidate quality, and quality calibration boosts anomaly evidence. |
| B | quality_helps_normal_suppression | pcb1 | resolved | keep | yes | Positive normal-suppression example: high VLM score but low quality, so quality calibration suppresses likely false-positive evidence. |
| C | quality_boundary_anomaly_suppression | pipe_fryum | resolved | keep | yes | Boundary anomaly-suppression example: VLM score is high but candidate quality is low, showing quality calibration can suppress true anomaly evidence. |
| D | quality_boundary_normal_boost | macaroni2 | resolved | replace | no | Weak boundary normal-boost example; quality delta is too small to be visually compelling. |
| E | fixed_consistency_boundary_normal_boost | pcb1 | resolved | replace | no | Likely duplicates panel B numerically; replace with a distinct fixed-consistency boundary case. |
| F | detector_vlm_disagreement_boundary | pipe_fryum | resolved | keep | yes | Detector-VLM disagreement example: detector evidence is high while VLM evidence is low. |

## 3. Current Decision

Use panels A/B/C/F as provisional keep cases. Panels D/E require replacement before final Figure 2 montage.

## 4. Next stage

```text
Paper Stage P11-B: select replacement candidates for D/E
```
