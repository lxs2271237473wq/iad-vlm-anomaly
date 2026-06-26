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
| full_image | 71 | 0.6488 | 0.8722 | 0.8548 | 0.7465 | 0.0000 |
| stage10e_crop_topk_mean | 71 | 0.4811 | 0.7514 | 0.8689 | 0.7746 | -0.1677 |
| stage10e_crop_topk_max | 71 | 0.4675 | 0.7777 | 0.8618 | 0.7606 | -0.1813 |
| stage10e_crop_top1 | 71 | 0.3753 | 0.7080 | 0.8689 | 0.7746 | -0.2736 |

## 7. Decision Rule

- If any context-crop method exceeds full_image, keep MVTec AD 2 as positive evidence for context-aware localization-guided VLM reasoning.
- If context crops remain below full_image, treat vial as a negative case and move to either another AD2 category or logical-anomaly data such as MVTec LOCO AD.
- PatchCore score is a detector reference, not VLM reasoning evidence.