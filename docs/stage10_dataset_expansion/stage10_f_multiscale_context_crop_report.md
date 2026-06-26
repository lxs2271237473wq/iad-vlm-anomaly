# Stage 10-F Multiscale Context Crop Diagnostic

## 1. Purpose

Stage 10-E showed that direct candidate crops underperform full-image VLM prompting.
This stage tests whether adding spatial context around PatchCore candidate boxes improves VLM reasoning.

## 2. Inputs

- Candidate regions: `results/stage10_dataset_expansion/stage10_d_patchcore_candidate_regions.csv`
- Stage 10-E image predictions: `results/stage10_dataset_expansion/stage10_e_vlm_image_predictions.csv`

## 3. Context Configurations

| Context | Scale | Square |
|---|---:|---:|
| context_0.20 | 0.2 | False |
| context_0.50 | 0.5 | False |
| context_1.00 | 1.0 | False |
| context_1.50 | 1.5 | False |
| square_context_1.00 | 1.0 | True |

## 4. VLM Backend

- Backend: `open_clip:ViT-B-32/openai`

## 5. Output Files

- Crop scores: `results/stage10_dataset_expansion/stage10_f_multiscale_context_crop_scores.csv`
- Image predictions: `results/stage10_dataset_expansion/stage10_f_multiscale_context_image_predictions.csv`
- Summary: `results/stage10_dataset_expansion/stage10_f_multiscale_context_summary.csv`
- Report: `docs/stage10_dataset_expansion/stage10_f_multiscale_context_crop_report.md`
- Generated crops: `results/stage10_dataset_expansion/stage10_f_multiscale_context_crops`

## 6. Summary

| Method | Images | AUROC | AP | Best F1 | Best Acc | ΔAUROC vs full |
|---|---:|---:|---:|---:|---:|---:|
| patchcore_score | 71 | 0.8899 | 0.9605 | 0.9204 | 0.8732 | 0.2411 |
| context_1.50_top1 | 71 | 0.7746 | 0.9081 | 0.8667 | 0.7746 | 0.1258 |
| context_1.50_topk_mean | 71 | 0.6981 | 0.8345 | 0.8667 | 0.7746 | 0.0493 |
| square_context_1.00_top1 | 71 | 0.6520 | 0.8324 | 0.8618 | 0.7606 | 0.0031 |
| full_image | 71 | 0.6488 | 0.8722 | 0.8548 | 0.7465 | 0.0000 |
| context_1.50_topk_max | 71 | 0.6447 | 0.7718 | 0.8679 | 0.8028 | -0.0042 |
| square_context_1.00_topk_mean | 71 | 0.5493 | 0.7906 | 0.8595 | 0.7606 | -0.0996 |
| context_0.50_topk_max | 71 | 0.5010 | 0.8149 | 0.8548 | 0.7465 | -0.1478 |
| context_1.00_top1 | 71 | 0.4990 | 0.7736 | 0.8548 | 0.7465 | -0.1499 |
| square_context_1.00_topk_max | 71 | 0.4927 | 0.7044 | 0.8595 | 0.7606 | -0.1562 |
| context_0.50_topk_mean | 71 | 0.4916 | 0.7929 | 0.8548 | 0.7465 | -0.1572 |
| context_0.20_topk_mean | 71 | 0.4906 | 0.7436 | 0.8667 | 0.7746 | -0.1583 |
| context_0.20_topk_max | 71 | 0.4853 | 0.7802 | 0.8618 | 0.7606 | -0.1635 |
| stage10e_crop_topk_mean | 71 | 0.4811 | 0.7514 | 0.8689 | 0.7746 | -0.1677 |
| stage10e_crop_topk_max | 71 | 0.4675 | 0.7777 | 0.8618 | 0.7606 | -0.1813 |
| context_1.00_topk_mean | 71 | 0.4602 | 0.7844 | 0.8548 | 0.7465 | -0.1887 |
| context_1.00_topk_max | 71 | 0.4214 | 0.7333 | 0.8548 | 0.7465 | -0.2275 |
| context_0.50_top1 | 71 | 0.4140 | 0.7649 | 0.8548 | 0.7465 | -0.2348 |
| context_0.20_top1 | 71 | 0.3899 | 0.7195 | 0.8618 | 0.7606 | -0.2589 |
| stage10e_crop_top1 | 71 | 0.3753 | 0.7080 | 0.8689 | 0.7746 | -0.2736 |

## 7. Decision Rule

- If any context-crop method exceeds full_image, keep MVTec AD 2 as positive evidence for context-aware localization-guided VLM reasoning.
- If context crops remain below full_image, treat vial as a negative case and move to either another AD2 category or logical-anomaly data such as MVTec LOCO AD.
- PatchCore score is a detector reference, not VLM reasoning evidence.

<!-- stage10_f1_summary_refreshed_20260626_181042_202497 -->
