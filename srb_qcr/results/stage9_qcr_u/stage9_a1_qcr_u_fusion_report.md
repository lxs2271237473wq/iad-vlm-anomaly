# Stage 9-A1 QCR-U Fusion Report

## 1. Purpose

This stage implements fixed-weight QCR-U fusion on existing VisA predictions.
It reads existing detector predictions, candidate regions, and VLM binary prompt scores.
It does not train models, rerun CLIP, or regenerate anomaly maps.

## 2. Score Definition

```text
M = normalized VLM anomaly score
Q = normalized candidate-region quality
D = normalized detector image score
K = high-high detector/VLM consistency
F_qcr = alpha * M + beta * Q + gamma * K
F_qcr_detector_aware = alpha * M + beta * Q + gamma * K + delta * D
```

## 3. Output Files

- `results/stage9_qcr_u/stage9_a1_qcr_u_fusion_predictions.csv`
- `results/stage9_qcr_u/stage9_a1_qcr_u_fusion_summary.csv`
- `results/stage9_qcr_u/stage9_a1_qcr_u_fusion_report.md`

## 4. Best Overall Rows

| Backbone | Strategy | Eval mode | Fusion | AUROC | AP | Best F1 | Delta AUROC vs VLM |
|---|---|---|---|---:|---:|---:|---:|
| FastFlow | inspection_binary | crop_or_full | candidate_quality_only | 0.9950 | 0.9976 | 0.9950 | 0.1240 |
| FastFlow | inspection_binary | crop_topk_ensemble | candidate_quality_only | 0.9950 | 0.9976 | 0.9950 | 0.0681 |
| FastFlow | inspection_binary | full_all | candidate_quality_only | 0.9950 | 0.9976 | 0.9950 | 0.4059 |
| FastFlow | inspection_binary | crop_topk_ensemble | qcr_u_detector_aware | 0.9898 | 0.9925 | 0.9581 | 0.0629 |
| FastFlow | inspection_binary | crop_topk_ensemble | qcr_u_fixed | 0.9806 | 0.9865 | 0.9428 | 0.0537 |
| PatchCore | category_binary | crop_or_full | candidate_quality_only | 0.9950 | 0.9976 | 0.9950 | 0.2130 |
| PatchCore | category_binary | crop_topk_ensemble | candidate_quality_only | 0.9950 | 0.9976 | 0.9950 | 0.1959 |
| PatchCore | category_binary | full_all | candidate_quality_only | 0.9950 | 0.9976 | 0.9950 | 0.4689 |
| PatchCore | generic_binary | crop_or_full | candidate_quality_only | 0.9950 | 0.9976 | 0.9950 | 0.1150 |
| PatchCore | generic_binary | crop_topk_ensemble | candidate_quality_only | 0.9950 | 0.9976 | 0.9950 | 0.0991 |

## 5. Best QCR-U Rows

| Backbone | Strategy | Eval mode | Fusion | AUROC | AP | Best F1 | Delta AUROC vs VLM |
|---|---|---|---|---:|---:|---:|---:|
| FastFlow | inspection_binary | crop_topk_ensemble | qcr_u_detector_aware | 0.9898 | 0.9925 | 0.9581 | 0.0629 |
| FastFlow | inspection_binary | crop_topk_ensemble | qcr_u_fixed | 0.9806 | 0.9865 | 0.9428 | 0.0537 |
| FastFlow | inspection_binary | crop_or_full | qcr_u_detector_aware | 0.9781 | 0.9840 | 0.9346 | 0.1071 |
| PatchCore | generic_binary | crop_topk_ensemble | qcr_u_detector_aware | 0.9708 | 0.9809 | 0.9301 | 0.0748 |
| PatchCore | inspection_binary | crop_topk_ensemble | qcr_u_detector_aware | 0.9675 | 0.9769 | 0.9154 | 0.0829 |
| PatchCore | generic_binary | crop_or_full | qcr_u_detector_aware | 0.9633 | 0.9761 | 0.9172 | 0.0833 |

## 6. Interpretation Boundary

- These are fixed-weight fusion results, not supervised grid-search results.
- Positive delta means Q/K/D adds useful signal beyond VLM score alone.
- Negative delta means the current QCR-U construction should remain diagnostic evidence rather than a main performance claim.
- Since labels are not used to tune weights, this is safer than an oracle weight search.

## 7. Next Step

Stage 9-A2 should decide whether QCR-U is kept as a main module, an ablation module, or a diagnostic analysis.