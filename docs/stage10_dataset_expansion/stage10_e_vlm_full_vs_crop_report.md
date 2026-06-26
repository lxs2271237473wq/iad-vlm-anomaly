# Stage 10-E MVTec AD 2 VLM Full-image vs Candidate-crop Reasoning

## 1. Purpose

This stage evaluates whether PatchCore candidate crops improve CLIP/VLM binary anomaly reasoning on MVTec AD 2 vial.
It compares full-image prompting with crop-based prompting using the candidate crops generated in Stage 10-D.

## 2. Input

- Candidate CSV: `results/stage10_dataset_expansion/stage10_d_patchcore_candidate_regions.csv`
- Candidate crop directory: `results/stage10_dataset_expansion/stage10_d_patchcore_candidate_crops/`

## 3. VLM Backend

- Backend: `open_clip:ViT-B-32/openai`

## 4. Prompts

Normal prompts:

- a quality inspection image of a normal vial
- an industrial inspection photo of an intact vial
- a clean and defect-free vial surface
- a normal product image of a vial with no defect

Anomaly prompts:

- a quality inspection image of a defective vial
- an industrial inspection photo of a damaged vial
- a vial with visible defects or anomalies
- an abnormal product image of a vial with defect

## 5. Output Files

- Candidate scores: `results/stage10_dataset_expansion/stage10_e_vlm_candidate_scores.csv`
- Image predictions: `results/stage10_dataset_expansion/stage10_e_vlm_image_predictions.csv`
- Summary: `results/stage10_dataset_expansion/stage10_e_vlm_summary.csv`
- Report: `docs/stage10_dataset_expansion/stage10_e_vlm_full_vs_crop_report.md`

## 6. Summary

| Method | Images | AUROC | AP | Best F1 | Best Acc | ΔAUROC vs full |
|---|---:|---:|---:|---:|---:|---:|
| patchcore_score | 71 | 0.8899 | 0.9605 | 0.9204 | 0.8732 | 0.2411 |
| full_image | 71 | 0.6488 | 0.8722 | 0.8548 | 0.7465 | 0.0000 |
| crop_topk_mean | 71 | 0.4811 | 0.7514 | 0.8689 | 0.7746 | -0.1677 |
| crop_topk_max | 71 | 0.4675 | 0.7777 | 0.8618 | 0.7606 | -0.1813 |
| crop_top1 | 71 | 0.3753 | 0.7080 | 0.8689 | 0.7746 | -0.2736 |

## 7. Interpretation Rule

- If crop_topk_max > full_image, MVTec AD 2 supports the localization-guided VLM reasoning claim.
- If crop_topk_max is close to or worse than full_image, inspect whether candidate crops are too small, visually ambiguous, or dominated by background.
- PatchCore_score is included only as a detector reference, not as a VLM reasoning result.

## 8. Next Step

Stage 10-F should generate a comparison table that integrates Stage 10-C detector metrics, Stage 10-D candidate coverage, and Stage 10-E VLM reasoning.