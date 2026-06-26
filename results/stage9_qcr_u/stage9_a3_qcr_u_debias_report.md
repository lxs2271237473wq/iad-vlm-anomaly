# Stage 9-A3 QCR-U Debias Check Report

## 1. Purpose

Stage 9-A2 showed that candidate_quality_only is extremely strong.
This stage removes candidate-existence bias by assigning a neutral Q value to images without candidates.
It reads Stage 9-A1 predictions only and does not train models or regenerate anomaly maps.

## 2. Debias Setting

```text
q_original = candidate_quality_norm
q_neutral = candidate_quality_norm if candidate exists else 0.5
```

The neutral value prevents no-candidate images from being automatically treated as normal through Q=0.

## 3. Output Files

- `results/stage9_qcr_u/stage9_a3_qcr_u_debiased_predictions.csv`
- `results/stage9_qcr_u/stage9_a3_qcr_u_debiased_summary.csv`
- `results/stage9_qcr_u/stage9_a3_qcr_u_debiased_per_category.csv`
- `results/stage9_qcr_u/stage9_a3_qcr_u_debias_report.md`

## 4. Best QCR-U / Debiased Rows

| Backbone | Strategy | Eval mode | Method | AUROC | AP | Best F1 | ΔAUROC vs VLM |
|---|---|---|---|---:|---:|---:|---:|
| FastFlow | inspection_binary | crop_topk_ensemble | qcr_u_original_detector_aware | 0.9898 | 0.9925 | 0.9581 | 0.0629 |
| FastFlow | inspection_binary | crop_topk_ensemble | qcr_u_original_fixed | 0.9806 | 0.9865 | 0.9428 | 0.0537 |
| FastFlow | inspection_binary | crop_or_full | qcr_u_original_detector_aware | 0.9781 | 0.9840 | 0.9346 | 0.1071 |
| PatchCore | generic_binary | crop_topk_ensemble | qcr_u_original_detector_aware | 0.9708 | 0.9809 | 0.9301 | 0.0748 |
| PatchCore | inspection_binary | crop_topk_ensemble | qcr_u_original_detector_aware | 0.9675 | 0.9769 | 0.9154 | 0.0829 |
| PatchCore | generic_binary | crop_or_full | qcr_u_original_detector_aware | 0.9633 | 0.9761 | 0.9172 | 0.0833 |
| FastFlow | inspection_binary | crop_or_full | qcr_u_original_fixed | 0.9614 | 0.9724 | 0.9060 | 0.0904 |
| PatchCore | inspection_binary | crop_or_full | qcr_u_original_detector_aware | 0.9600 | 0.9715 | 0.9022 | 0.0918 |
| FastFlow | inspection_binary | crop_topk_ensemble | qcr_u_neutral_q_detector_aware | 0.9596 | 0.9716 | 0.9107 | 0.0326 |
| FastFlow | inspection_binary | crop_topk_ensemble | qcr_u_no_q_detector_aware | 0.9560 | 0.9687 | 0.9089 | 0.0290 |
| PatchCore | generic_binary | crop_topk_ensemble | qcr_u_original_fixed | 0.9493 | 0.9684 | 0.9024 | 0.0533 |
| PatchCore | inspection_binary | crop_topk_ensemble | qcr_u_original_fixed | 0.9456 | 0.9619 | 0.8910 | 0.0610 |
| PatchCore | generic_binary | crop_topk_ensemble | qcr_u_no_q_detector_aware | 0.9455 | 0.9638 | 0.8971 | 0.0495 |
| FastFlow | inspection_binary | crop_topk_ensemble | qcr_u_neutral_q_fixed | 0.9412 | 0.9594 | 0.8883 | 0.0142 |
| PatchCore | generic_binary | crop_or_full | qcr_u_original_fixed | 0.9406 | 0.9627 | 0.8886 | 0.0606 |
| PatchCore | inspection_binary | crop_or_full | qcr_u_original_fixed | 0.9366 | 0.9548 | 0.8803 | 0.0684 |
| PatchCore | inspection_binary | crop_topk_ensemble | qcr_u_no_q_detector_aware | 0.9366 | 0.9541 | 0.8794 | 0.0520 |
| PatchCore | generic_binary | crop_or_full | qcr_u_no_q_detector_aware | 0.9323 | 0.9552 | 0.8787 | 0.0523 |
| FastFlow | inspection_binary | crop_topk_ensemble | qcr_u_no_q_fixed | 0.9315 | 0.9525 | 0.8812 | 0.0046 |
| PatchCore | category_binary | crop_topk_ensemble | qcr_u_original_detector_aware | 0.9300 | 0.9419 | 0.8892 | 0.1308 |

## 5. Candidate Quality Original vs Neutral

| Backbone | Strategy | Eval mode | Method | AUROC | AP | Best F1 | ΔAUROC vs VLM |
|---|---|---|---|---:|---:|---:|---:|
| FastFlow | inspection_binary | crop_or_full | candidate_quality_original | 0.9950 | 0.9976 | 0.9950 | 0.1240 |
| FastFlow | inspection_binary | crop_topk_ensemble | candidate_quality_original | 0.9950 | 0.9976 | 0.9950 | 0.0681 |
| PatchCore | category_binary | crop_or_full | candidate_quality_original | 0.9950 | 0.9976 | 0.9950 | 0.2130 |
| FastFlow | inspection_binary | full_all | candidate_quality_original | 0.9950 | 0.9976 | 0.9950 | 0.4059 |
| PatchCore | generic_binary | crop_or_full | candidate_quality_original | 0.9950 | 0.9976 | 0.9950 | 0.1150 |
| PatchCore | generic_binary | crop_topk_ensemble | candidate_quality_original | 0.9950 | 0.9976 | 0.9950 | 0.0991 |
| PatchCore | category_binary | full_all | candidate_quality_original | 0.9950 | 0.9976 | 0.9950 | 0.4689 |
| PatchCore | category_binary | crop_topk_ensemble | candidate_quality_original | 0.9950 | 0.9976 | 0.9950 | 0.1959 |
| PatchCore | inspection_binary | crop_topk_ensemble | candidate_quality_original | 0.9950 | 0.9976 | 0.9950 | 0.1104 |
| PatchCore | inspection_binary | full_all | candidate_quality_original | 0.9950 | 0.9976 | 0.9950 | 0.4059 |
| PatchCore | inspection_binary | crop_or_full | candidate_quality_original | 0.9950 | 0.9976 | 0.9950 | 0.1268 |
| PatchCore | generic_binary | full_all | candidate_quality_original | 0.9950 | 0.9976 | 0.9950 | 0.4513 |
| FastFlow | inspection_binary | crop_or_full | candidate_quality_neutral | 0.7250 | 0.8673 | 0.8406 | -0.1460 |
| FastFlow | inspection_binary | full_all | candidate_quality_neutral | 0.7250 | 0.8673 | 0.8406 | 0.1359 |
| FastFlow | inspection_binary | crop_topk_ensemble | candidate_quality_neutral | 0.7250 | 0.8673 | 0.8406 | -0.2019 |
| PatchCore | category_binary | crop_or_full | candidate_quality_neutral | 0.2683 | 0.5824 | 0.7139 | -0.5137 |
| PatchCore | generic_binary | crop_topk_ensemble | candidate_quality_neutral | 0.2683 | 0.5824 | 0.7139 | -0.6276 |
| PatchCore | generic_binary | crop_or_full | candidate_quality_neutral | 0.2683 | 0.5824 | 0.7139 | -0.6116 |
| PatchCore | category_binary | crop_topk_ensemble | candidate_quality_neutral | 0.2683 | 0.5824 | 0.7139 | -0.5308 |
| PatchCore | category_binary | full_all | candidate_quality_neutral | 0.2683 | 0.5824 | 0.7139 | -0.2578 |

## 6. Stability Counts

- Neutral-Q QCR-U positive settings: 16/24
- No-Q QCR-U positive settings: 24/24

## 7. Decision Rule

| Observation | Paper Decision |
|---|---|
| Neutral-Q QCR-U still improves most settings | Keep QCR-U as a calibration/ablation module |
| No-Q QCR-U still improves most settings | Emphasize detector-VLM consistency K rather than candidate quality Q |
| Only original-Q works | Do not use QCR-U as a method claim; keep as leakage/diagnostic analysis |

## 8. Conservative Claim

Until this check is inspected, the safest claim remains:

```text
QCR-U is a diagnostic calibration mechanism. The paper's core contribution should remain localization-guided VLM reasoning, not candidate-quality fusion.
```