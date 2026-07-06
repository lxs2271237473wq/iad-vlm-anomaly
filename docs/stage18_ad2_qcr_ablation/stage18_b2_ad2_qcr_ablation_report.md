# Stage 18-B2 AD2 Four-category QCR Ablation

## Purpose

Assemble AD2 four-category QCR ablation from existing Stage11 image-level VLM predictions and candidate-level quality evidence.

This aligns the QCR ablation with the AD2 four-category system-level baseline setting.

## Data

- input image-level predictions: `results/stage11_mvtecad2_multicategory/stage11_d_vlm_image_predictions.csv`
- input candidate scores: `results/stage11_mvtecad2_multicategory/stage11_d_vlm_candidate_scores.csv`
- assembled images: `243`
- categories: `fruit_jelly; sheet_metal; vial; walnuts`
- detector evidence `D`: normalized `patchcore_score`
- crop VLM evidence `M`: normalized context top-k VLM score
- candidate quality `Q`: normalized non-GT candidate score evidence
- consistency `K`: soft high-high consistency `D*M`

## Summary table

| Variant | Method | Role | Mean AUROC | Mean F1 |
|---|---|---|---:|---:|
| V0 | Detector only | baseline_detector | 0.7853 | 0.8717 |
| V1 | Full-image VLM | baseline_vlm | 0.6459 | 0.8535 |
| V2 | Crop top-k VLM | baseline_crop_vlm | 0.6524 | 0.8447 |
| V3 | Naive detector-crop fusion | fusion_baseline | 0.8286 | 0.8741 |
| V4 | Quality-Calibrated QCR | main_method_core | 0.8191 | 0.8811 |
| V5 | Fixed Q+C fusion | diagnostic_not_final | 0.8226 | 0.8753 |
| V6 | Quality-Calibrated QCR + adaptive refinement | final_refinement | 0.8194 | 0.8811 |

## Claim-ready deltas

| Comparison | Delta AUROC | A | B |
|---|---:|---:|---:|
| Quality-Calibrated QCR vs naive fusion | -0.0095 | 0.8191 | 0.8286 |
| Adaptive refinement vs Quality-Calibrated QCR | +0.0003 | 0.8194 | 0.8191 |
| Adaptive refinement vs naive fusion | -0.0093 | 0.8194 | 0.8286 |
| Fixed Q+C diagnostic vs Quality-Calibrated QCR | +0.0035 | 0.8226 | 0.8191 |
| Quality-Calibrated QCR vs detector only | +0.0338 | 0.8191 | 0.7853 |
| Quality-Calibrated QCR vs crop top-k VLM | +0.1667 | 0.8191 | 0.6524 |

## Per-category AUROC

| Category | Variant | Method | AUROC | F1 |
|---|---|---|---:|---:|
| fruit_jelly | V0 | Detector only | 0.7167 | 0.8696 |
| fruit_jelly | V1 | Full-image VLM | 0.7533 | 0.9091 |
| fruit_jelly | V2 | Crop top-k VLM | 0.8567 | 0.8852 |
| fruit_jelly | V3 | Naive detector-crop fusion | 0.8767 | 0.9000 |
| fruit_jelly | V4 | Quality-Calibrated QCR | 0.8833 | 0.9000 |
| fruit_jelly | V5 | Fixed Q+C fusion | 0.8800 | 0.9000 |
| fruit_jelly | V6 | Quality-Calibrated QCR + adaptive refinement | 0.8833 | 0.9000 |
| sheet_metal | V0 | Detector only | 0.7463 | 0.8911 |
| sheet_metal | V1 | Full-image VLM | 0.7130 | 0.9000 |
| sheet_metal | V2 | Crop top-k VLM | 0.5870 | 0.8824 |
| sheet_metal | V3 | Naive detector-crop fusion | 0.7426 | 0.9000 |
| sheet_metal | V4 | Quality-Calibrated QCR | 0.6537 | 0.9000 |
| sheet_metal | V5 | Fixed Q+C fusion | 0.6889 | 0.8889 |
| sheet_metal | V6 | Quality-Calibrated QCR + adaptive refinement | 0.6537 | 0.9000 |
| vial | V0 | Detector only | 0.8732 | 0.9298 |
| vial | V1 | Full-image VLM | 0.6876 | 0.8548 |
| vial | V2 | Crop top-k VLM | 0.5231 | 0.8548 |
| vial | V3 | Naive detector-crop fusion | 0.9182 | 0.9204 |
| vial | V4 | Quality-Calibrated QCR | 0.9476 | 0.9455 |
| vial | V5 | Fixed Q+C fusion | 0.9319 | 0.9298 |
| vial | V6 | Quality-Calibrated QCR + adaptive refinement | 0.9486 | 0.9455 |
| walnuts | V0 | Detector only | 0.8052 | 0.7963 |
| walnuts | V1 | Full-image VLM | 0.4296 | 0.7500 |
| walnuts | V2 | Crop top-k VLM | 0.6430 | 0.7563 |
| walnuts | V3 | Naive detector-crop fusion | 0.7770 | 0.7759 |
| walnuts | V4 | Quality-Calibrated QCR | 0.7919 | 0.7788 |
| walnuts | V5 | Fixed Q+C fusion | 0.7896 | 0.7826 |
| walnuts | V6 | Quality-Calibrated QCR + adaptive refinement | 0.7919 | 0.7788 |

## Interpretation rules

- If V4 improves over V3, AD2 supports candidate quality calibration.
- If V6 only slightly improves over V4, keep adaptive consistency as refinement.
- If V5 is strong but unstable or not selected, keep fixed Q+C as diagnostic.
- Do not use this table to claim pixel-level segmentation SOTA.

## Outputs

- `results/stage18_ad2_qcr_ablation/stage18_b2_ad2_qcr_assembled_predictions.csv`
- `results/stage18_ad2_qcr_ablation/stage18_b2_ad2_qcr_ablation_per_category.csv`
- `results/stage18_ad2_qcr_ablation/stage18_b2_ad2_qcr_ablation_summary.csv`
- `results/stage18_ad2_qcr_ablation/stage18_b2_ad2_qcr_claim_ready_deltas.csv`
