# Stage 9-A2 QCR-U Sanity Check Report

## 1. Purpose

This stage checks whether Stage 9-A1 QCR-U fusion can be used as a paper-level module.
It reads existing Stage 9-A1 predictions only. It does not train models, rerun CLIP, or regenerate anomaly maps.

## 2. Why This Check Is Necessary

Stage 9-A1 shows strong QCR-U performance, but candidate_quality_only is also extremely strong.
Therefore, the current result must be treated carefully: QCR-U may be a useful calibration module, but candidate quality may already encode most anomaly evidence.

## 3. Output Files

- `results/stage9_qcr_u/stage9_a2_qcr_u_per_category.csv`
- `results/stage9_qcr_u/stage9_a2_qcr_u_macro_summary.csv`
- `results/stage9_qcr_u/stage9_a2_qcr_u_signal_diagnostics.csv`
- `results/stage9_qcr_u/stage9_a2_qcr_u_sanity_report.md`

## 4. Best QCR-U Macro Results

| Backbone | Strategy | Eval mode | Fusion | Macro AUROC | Macro AP | Macro F1 | Positive categories | Negative categories | Mean ΔAUROC vs VLM |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| FastFlow | inspection_binary | crop_topk_ensemble | qcr_u_detector_aware | 0.9871 | 0.9899 | 0.9698 | 9 | 0 | 0.0650 |
| FastFlow | inspection_binary | crop_topk_ensemble | qcr_u_fixed | 0.9758 | 0.9828 | 0.9574 | 9 | 0 | 0.0536 |
| FastFlow | inspection_binary | crop_or_full | qcr_u_detector_aware | 0.9747 | 0.9813 | 0.9507 | 11 | 0 | 0.1043 |
| PatchCore | inspection_binary | crop_topk_ensemble | qcr_u_detector_aware | 0.9667 | 0.9771 | 0.9516 | 11 | 0 | 0.0823 |
| PatchCore | generic_binary | crop_topk_ensemble | qcr_u_detector_aware | 0.9656 | 0.9773 | 0.9498 | 9 | 0 | 0.0864 |
| PatchCore | generic_binary | crop_or_full | qcr_u_detector_aware | 0.9592 | 0.9728 | 0.9411 | 9 | 0 | 0.0926 |
| PatchCore | inspection_binary | crop_or_full | qcr_u_detector_aware | 0.9572 | 0.9705 | 0.9418 | 10 | 1 | 0.0900 |
| FastFlow | inspection_binary | crop_or_full | qcr_u_fixed | 0.9567 | 0.9701 | 0.9334 | 11 | 0 | 0.0863 |
| PatchCore | inspection_binary | crop_topk_ensemble | qcr_u_fixed | 0.9448 | 0.9619 | 0.9247 | 11 | 0 | 0.0603 |
| PatchCore | generic_binary | crop_topk_ensemble | qcr_u_fixed | 0.9375 | 0.9594 | 0.9247 | 9 | 0 | 0.0583 |
| PatchCore | inspection_binary | crop_or_full | qcr_u_fixed | 0.9335 | 0.9537 | 0.9142 | 11 | 0 | 0.0663 |
| PatchCore | category_binary | crop_topk_ensemble | qcr_u_detector_aware | 0.9300 | 0.9470 | 0.9175 | 11 | 0 | 0.1412 |

## 5. Candidate-quality-only Macro Results

| Backbone | Strategy | Eval mode | Macro AUROC | Mean ΔAUROC vs VLM |
|---|---|---|---:|---:|
| FastFlow | inspection_binary | crop_or_full | 0.9950 | 0.1246 |
| FastFlow | inspection_binary | crop_topk_ensemble | 0.9950 | 0.0728 |
| FastFlow | inspection_binary | full_all | 0.9950 | 0.4000 |
| PatchCore | category_binary | crop_or_full | 0.9950 | 0.2215 |
| PatchCore | category_binary | crop_topk_ensemble | 0.9950 | 0.2061 |
| PatchCore | category_binary | full_all | 0.9950 | 0.4276 |
| PatchCore | generic_binary | crop_or_full | 0.9950 | 0.1285 |
| PatchCore | generic_binary | crop_topk_ensemble | 0.9950 | 0.1158 |
| PatchCore | generic_binary | full_all | 0.9950 | 0.4076 |
| PatchCore | inspection_binary | crop_or_full | 0.9950 | 0.1278 |
| PatchCore | inspection_binary | crop_topk_ensemble | 0.9950 | 0.1106 |
| PatchCore | inspection_binary | full_all | 0.9950 | 0.4000 |

## 6. QCR-U Negative-delta Categories

| Backbone | Strategy | Eval mode | Category | Fusion | AUROC | VLM AUROC | ΔAUROC |
|---|---|---|---|---|---:|---:|---:|
| PatchCore | inspection_binary | crop_or_full | capsules | qcr_u_detector_aware | 0.9987 | 0.9988 | -0.0002 |

## 7. Strongest Candidate-quality Separation

| Backbone | Strategy | Eval mode | Category | Candidate rate normal | Candidate rate anomaly | Q normal | Q anomaly | corr(Q,y) |
|---|---|---|---|---:|---:|---:|---:|---:|
| FastFlow | inspection_binary | crop_or_full | pcb3 | 0.0000 | 1.0000 | 0.0000 | 0.9215 | 0.9195 |
| FastFlow | inspection_binary | crop_topk_ensemble | pcb3 | 0.0000 | 1.0000 | 0.0000 | 0.9215 | 0.9195 |
| FastFlow | inspection_binary | full_all | pcb3 | 0.0000 | 1.0000 | 0.0000 | 0.9215 | 0.9195 |
| FastFlow | inspection_binary | crop_or_full | pcb4 | 0.0000 | 1.0000 | 0.0000 | 0.6504 | 0.9195 |
| FastFlow | inspection_binary | crop_topk_ensemble | pcb4 | 0.0000 | 1.0000 | 0.0000 | 0.6504 | 0.9195 |
| FastFlow | inspection_binary | full_all | pcb4 | 0.0000 | 1.0000 | 0.0000 | 0.6504 | 0.9195 |
| PatchCore | category_binary | crop_or_full | pcb3 | 0.0000 | 1.0000 | 0.0000 | 0.3900 | 0.9195 |
| PatchCore | category_binary | crop_topk_ensemble | pcb3 | 0.0000 | 1.0000 | 0.0000 | 0.3900 | 0.9195 |
| PatchCore | category_binary | full_all | pcb3 | 0.0000 | 1.0000 | 0.0000 | 0.3900 | 0.9195 |
| PatchCore | generic_binary | crop_or_full | pcb3 | 0.0000 | 1.0000 | 0.0000 | 0.3900 | 0.9195 |
| PatchCore | generic_binary | crop_topk_ensemble | pcb3 | 0.0000 | 1.0000 | 0.0000 | 0.3900 | 0.9195 |
| PatchCore | generic_binary | full_all | pcb3 | 0.0000 | 1.0000 | 0.0000 | 0.3900 | 0.9195 |
| PatchCore | inspection_binary | crop_or_full | pcb3 | 0.0000 | 1.0000 | 0.0000 | 0.3900 | 0.9195 |
| PatchCore | inspection_binary | crop_topk_ensemble | pcb3 | 0.0000 | 1.0000 | 0.0000 | 0.3900 | 0.9195 |
| PatchCore | inspection_binary | full_all | pcb3 | 0.0000 | 1.0000 | 0.0000 | 0.3900 | 0.9195 |
| PatchCore | category_binary | crop_or_full | pcb4 | 0.0000 | 1.0000 | 0.0000 | 0.2946 | 0.9195 |
| PatchCore | category_binary | crop_topk_ensemble | pcb4 | 0.0000 | 1.0000 | 0.0000 | 0.2946 | 0.9195 |
| PatchCore | category_binary | full_all | pcb4 | 0.0000 | 1.0000 | 0.0000 | 0.2946 | 0.9195 |
| PatchCore | generic_binary | crop_or_full | pcb4 | 0.0000 | 1.0000 | 0.0000 | 0.2946 | 0.9195 |
| PatchCore | generic_binary | crop_topk_ensemble | pcb4 | 0.0000 | 1.0000 | 0.0000 | 0.2946 | 0.9195 |

## 8. Decision Guidance

Use the following rule for the next paper decision:

| Condition | Decision |
|---|---|
| QCR-U improves macro AUROC over vlm_only across most categories and candidate_quality_only is not overwhelmingly dominant | Keep QCR-U as a main module |
| QCR-U improves some settings, but candidate_quality_only is much stronger | Keep QCR-U as an ablation/calibration module, not the core contribution |
| QCR-U often hurts per-category AUROC | Keep QCR-U only as diagnostic analysis |

## 9. Current Conservative Interpretation

Before Stage 9-A2 results are inspected, the safest interpretation is:

```text
QCR-U is a candidate calibration module that combines VLM abnormality, detector-region quality, and detector/VLM consistency.
However, because candidate_quality_only is very strong in Stage 9-A1, the module should not yet be claimed as the main source of anomaly reasoning improvement.
```

## 10. Next Step

After this report is committed, inspect whether QCR-U has stable macro-category gains.
Then decide whether Stage 9-B should write QCR-U into the method section or keep it as an ablation/diagnostic module.