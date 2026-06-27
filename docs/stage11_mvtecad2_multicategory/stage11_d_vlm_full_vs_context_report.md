# Stage 11-D MVTec AD 2 Full-image vs Context-aware Crop VLM Reasoning

## 1. Purpose

This stage evaluates whether localization-guided context-aware crops improve VLM anomaly reasoning over full-image prompting on the selected MVTec AD 2 primary categories.

This step does not rerun PatchCore and does not regenerate candidate crops. It reads Stage 11-C candidate regions and scores full images, tight crops, and context-aware crops using a CLIP-style VLM backend.

## 2. Selected Categories

- fruit_jelly
- sheet_metal
- vial
- walnuts

## 3. Inputs

- Candidate regions: `results/stage11_mvtecad2_multicategory/stage11_c_candidate_regions.csv`
- Candidate summary: `results/stage11_mvtecad2_multicategory/stage11_c_candidate_summary.csv`
- Detector quality analysis: `results/stage11_mvtecad2_multicategory/stage11_b1_patchcore_detector_quality_analysis.csv`

## 4. VLM Backend

- Backend: `open_clip:ViT-B-32/openai`
- Scoring: anomaly prompt similarity minus normal prompt similarity
- Prompt type: category-aware normal/anomaly prompts

## 5. Evaluated Methods

| Method | Meaning |
|---|---|
| full_image | VLM score on the original full inspection image |
| tight_crop_top1 | VLM score on rank-1 tight PatchCore candidate crop |
| tight_crop_topk_max | maximum VLM score over tight candidate crops |
| tight_crop_topk_mean | mean VLM score over tight candidate crops |
| context_1.50_top1 | VLM score on rank-1 context crop expanded from the candidate box |
| context_1.50_topk_max | maximum VLM score over context crops |
| context_1.50_topk_mean | mean VLM score over context crops |
| patchcore_score | detector score reference, not a VLM reasoning method |

## 6. Output Files

- Candidate scores: `results/stage11_mvtecad2_multicategory/stage11_d_vlm_candidate_scores.csv`
- Image predictions: `results/stage11_mvtecad2_multicategory/stage11_d_vlm_image_predictions.csv`
- Summary: `results/stage11_mvtecad2_multicategory/stage11_d_vlm_summary.csv`
- Report: `docs/stage11_mvtecad2_multicategory/stage11_d_vlm_full_vs_context_report.md`

## 7. Stage 11-C Candidate Quality Reference

| Category | Images | Candidate rows | Coverage | Top1 tight GT coverage | Top1 context GT coverage |
|---|---:|---:|---:|---:|---:|
| fruit_jelly | 40 | 61 | 1.0000 | 0.4302 | 0.8661 |
| sheet_metal | 57 | 153 | 1.0000 | 0.1132 | 0.2509 |
| vial | 71 | 117 | 1.0000 | 0.4275 | 0.8016 |
| walnuts | 75 | 171 | 1.0000 | 0.3606 | 0.4851 |

## 8. VLM Reasoning Summary

| Category | Method | Images | AUROC | AP | Best F1 | Best Acc | ΔAUROC vs full |
|---|---|---:|---:|---:|---:|---:|---:|
| ALL_PRIMARY | patchcore_score | 243 | 0.8087 | 0.9170 | 0.8641 | 0.7942 | 0.2885 |
| ALL_PRIMARY | context_1.50_topk_mean | 243 | 0.6036 | 0.7983 | 0.8337 | 0.7160 | 0.0835 |
| ALL_PRIMARY | context_1.50_top1 | 243 | 0.5773 | 0.7469 | 0.8341 | 0.7202 | 0.0571 |
| ALL_PRIMARY | context_1.50_topk_max | 243 | 0.5652 | 0.7876 | 0.8337 | 0.7160 | 0.0450 |
| ALL_PRIMARY | tight_crop_top1 | 243 | 0.5231 | 0.7224 | 0.8321 | 0.7160 | 0.0030 |
| ALL_PRIMARY | full_image | 243 | 0.5201 | 0.7260 | 0.8398 | 0.7284 | 0.0000 |
| ALL_PRIMARY | tight_crop_topk_mean | 243 | 0.5036 | 0.7149 | 0.8317 | 0.7119 | -0.0166 |
| ALL_PRIMARY | tight_crop_topk_max | 243 | 0.4648 | 0.6883 | 0.8321 | 0.7160 | -0.0553 |
| fruit_jelly | context_1.50_topk_mean | 40 | 0.8567 | 0.9556 | 0.8852 | 0.8250 | 0.1033 |
| fruit_jelly | context_1.50_top1 | 40 | 0.8367 | 0.9412 | 0.9153 | 0.8750 | 0.0833 |
| fruit_jelly | context_1.50_topk_max | 40 | 0.7667 | 0.9171 | 0.8852 | 0.8250 | 0.0133 |
| fruit_jelly | full_image | 40 | 0.7533 | 0.8997 | 0.9091 | 0.8500 | 0.0000 |
| fruit_jelly | tight_crop_top1 | 40 | 0.7367 | 0.9072 | 0.8696 | 0.7750 | -0.0167 |
| fruit_jelly | tight_crop_topk_max | 40 | 0.7333 | 0.9102 | 0.8696 | 0.7750 | -0.0200 |
| fruit_jelly | patchcore_score | 40 | 0.7167 | 0.9105 | 0.8696 | 0.7750 | -0.0367 |
| fruit_jelly | tight_crop_topk_mean | 40 | 0.6767 | 0.8849 | 0.8571 | 0.7500 | -0.0767 |
| sheet_metal | patchcore_score | 57 | 0.7463 | 0.9260 | 0.8911 | 0.8070 | 0.0333 |
| sheet_metal | full_image | 57 | 0.7130 | 0.9026 | 0.9000 | 0.8246 | 0.0000 |
| sheet_metal | context_1.50_topk_max | 57 | 0.6574 | 0.8859 | 0.8824 | 0.7895 | -0.0556 |
| sheet_metal | context_1.50_topk_mean | 57 | 0.5870 | 0.8096 | 0.8824 | 0.7895 | -0.1259 |
| sheet_metal | tight_crop_top1 | 57 | 0.4926 | 0.7578 | 0.8824 | 0.7895 | -0.2204 |
| sheet_metal | tight_crop_topk_mean | 57 | 0.3519 | 0.7006 | 0.8824 | 0.7895 | -0.3611 |
| sheet_metal | tight_crop_topk_max | 57 | 0.3481 | 0.7113 | 0.8824 | 0.7895 | -0.3648 |
| sheet_metal | context_1.50_top1 | 57 | 0.2444 | 0.6679 | 0.8824 | 0.7895 | -0.4685 |
| vial | patchcore_score | 71 | 0.8732 | 0.9525 | 0.9298 | 0.8873 | 0.1855 |
| vial | full_image | 71 | 0.6876 | 0.8733 | 0.8548 | 0.7465 | 0.0000 |
| vial | context_1.50_top1 | 71 | 0.6834 | 0.8882 | 0.8548 | 0.7465 | -0.0042 |
| vial | context_1.50_topk_mean | 71 | 0.5231 | 0.8041 | 0.8548 | 0.7465 | -0.1646 |
| vial | context_1.50_topk_max | 71 | 0.4560 | 0.7186 | 0.8548 | 0.7465 | -0.2317 |
| vial | tight_crop_topk_mean | 71 | 0.3973 | 0.7396 | 0.8548 | 0.7465 | -0.2904 |
| vial | tight_crop_top1 | 71 | 0.3753 | 0.7207 | 0.8571 | 0.7606 | -0.3124 |
| vial | tight_crop_topk_max | 71 | 0.3543 | 0.6683 | 0.8571 | 0.7606 | -0.3333 |
| walnuts | patchcore_score | 75 | 0.8052 | 0.8853 | 0.7963 | 0.7067 | 0.3756 |
| walnuts | context_1.50_topk_mean | 75 | 0.6430 | 0.7580 | 0.7563 | 0.6133 | 0.2133 |
| walnuts | context_1.50_topk_max | 75 | 0.6319 | 0.7442 | 0.7563 | 0.6133 | 0.2022 |
| walnuts | tight_crop_topk_mean | 75 | 0.6067 | 0.7406 | 0.7563 | 0.6133 | 0.1770 |
| walnuts | tight_crop_top1 | 75 | 0.5630 | 0.6908 | 0.7500 | 0.6000 | 0.1333 |
| walnuts | context_1.50_top1 | 75 | 0.5193 | 0.6871 | 0.7500 | 0.6000 | 0.0896 |
| walnuts | tight_crop_topk_max | 75 | 0.5074 | 0.6775 | 0.7586 | 0.6267 | 0.0778 |
| walnuts | full_image | 75 | 0.4296 | 0.6286 | 0.7500 | 0.6000 | 0.0000 |

## 9. Category-level Decision

| Category | Best VLM method | Best VLM AUROC | Full-image AUROC | ΔAUROC | Decision |
|---|---|---:|---:|---:|---|
| ALL_PRIMARY | context_1.50_topk_mean | 0.6036 | 0.5201 | 0.0835 | positive_context_evidence |
| fruit_jelly | context_1.50_topk_mean | 0.8567 | 0.7533 | 0.1033 | positive_context_evidence |
| sheet_metal | full_image | 0.7130 | 0.7130 | 0.0000 | full_image_stronger_or_tie |
| vial | full_image | 0.6876 | 0.6876 | 0.0000 | full_image_stronger_or_tie |
| walnuts | context_1.50_topk_mean | 0.6430 | 0.4296 | 0.2133 | positive_context_evidence |

## 10. Interpretation

The main question is not whether PatchCore alone is strong, but whether PatchCore localization can serve as a useful visual bridge for VLM reasoning.
A positive result is strongest when context-aware crops outperform full-image prompting, because this supports the Stage 10-G conclusion that object context is necessary for crop-based VLM anomaly reasoning.

Detector-risk categories are intentionally excluded from this stage because weak localization would make crop-based VLM results difficult to interpret.

## 11. Next Step

Stage 11-E should consolidate Stage 11-B, 11-C, and 11-D into a paper-ready multi-category evidence table and decide whether to include secondary category fabric.