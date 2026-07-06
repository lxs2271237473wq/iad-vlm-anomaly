# Stage 18-B1 AD2 QCR Source Schema Profile

## Purpose

Inspect Stage11/Stage13 AD2 source files to decide whether AD2 four-category QCR predictions can be assembled from existing files.

## Summary

- qcr_ready files: `3`
- partial_join_source files: `2`

## File profile

| File | Rows | AD2 coverage | Readiness | Key cols | Label cols | Detector-like | VLM-like | Quality-like |
|---|---:|---:|---|---|---|---|---|---|
| `results/stage11_mvtecad2_multicategory/stage11_b_patchcore_multicategory_predictions.csv` | 543 | 4/4 | ad2_summary_or_auxiliary | category;image_path;mask_path | gt_label;pred_label;anomaly_map_min;anomaly_map_max;anomaly_map_mean;anomaly_map_shape |  |  |  |
| `results/stage11_mvtecad2_multicategory/stage11_b1_patchcore_detector_quality_analysis.csv` | 8 | 4/4 | ad2_summary_or_auxiliary | category |  |  |  |  |
| `results/stage11_mvtecad2_multicategory/stage11_c_candidate_regions.csv` | 502 | 4/4 | partial_join_source | category;candidate_rank;image_path;mask_path;map_x1;map_y1;map_x2;map_y2;bbox_x1;bbox_y1;bbox_x2;bbox_y2;context_1p50_x1;context_1p50_y1;context_1p50_x2;context_1p50_y2;tight_crop_path;context_1p50_crop_path | gt_label;pred_label;anomaly_map_height;anomaly_map_width;tight_gt_mask_pixels;tight_candidate_covers_gt_ratio;context_gt_mask_pixels;context_candidate_covers_gt_ratio |  | context_candidate_mask_pixels;context_candidate_mask_density | map_area;candidate_score_max;candidate_score_mean;tight_candidate_mask_pixels;tight_candidate_mask_density |
| `results/stage11_mvtecad2_multicategory/stage11_d_vlm_candidate_scores.csv` | 502 | 4/4 | qcr_ready | category;candidate_rank;image_path;mask_path;map_x1;map_y1;map_x2;map_y2;bbox_x1;bbox_y1;bbox_x2;bbox_y2;context_1p50_x1;context_1p50_y1;context_1p50_x2;context_1p50_y2;tight_crop_path;context_1p50_crop_path | gt_label;pred_label;anomaly_map_height;anomaly_map_width;tight_gt_mask_pixels;tight_candidate_covers_gt_ratio;context_gt_mask_pixels;context_candidate_covers_gt_ratio;gt_binary;tight_anomaly_score;context_anomaly_score | patchcore_score | context_candidate_mask_pixels;context_candidate_mask_density;clip_backend;tight_vlm_margin;context_normal_score;context_vlm_margin | map_area;candidate_score_max;candidate_score_mean;tight_candidate_mask_pixels;tight_candidate_mask_density |
| `results/stage11_mvtecad2_multicategory/stage11_d_vlm_image_predictions.csv` | 243 | 4/4 | qcr_ready | category;image_path;tight_best_crop_path;context_best_crop_path | gt_binary;full_image_anomaly_score | full_image_score;patchcore_score | clip_backend;context_top1_score;context_topk_max_score;context_topk_mean_score | tight_topk_max_score;tight_topk_mean_score |
| `results/stage11_mvtecad2_multicategory/stage11_e_multicategory_evidence_table.csv` | 5 | 4/4 | qcr_ready | category | top1_tight_gt_coverage;top1_context_gt_coverage | detector_priority_group;image_AUROC_patchcore_stage11b;pixel_AUROC_patchcore_stage11b;pixel_F1_patchcore_stage11b;patchcore_score_auroc_stage11d | best_vlm_method;best_vlm_auroc;best_vlm_delta_vs_full;best_context_method;best_context_auroc;best_context_delta_vs_full;stage10_vial_context_1p50_top1_auroc;stage10_vial_context_delta | candidate_coverage |
| `results/stage13_strong_baseline/stage13_a_patchcore_vlm_fusion_per_category.csv` | 8 | 4/4 | partial_join_source | category | num_anomaly | alpha_patchcore |  |  |
| `results/stage13_strong_baseline/stage13_a_patchcore_vlm_score_complementarity.csv` | 5 | 4/4 | ad2_summary_or_auxiliary | category |  | patchcore_full_spearman | patchcore_context_spearman |  |

## Strong join candidates

| File A | File B | Key A | Key B | Overlap | Ratio A | Ratio B | Notes |
|---|---|---|---|---:|---:|---:|---|
| `results/stage11_mvtecad2_multicategory/stage11_c_candidate_regions.csv` | `results/stage11_mvtecad2_multicategory/stage11_d_vlm_candidate_scores.csv` | image_path | image_path | 243 | 1.000 | 1.000 | strong possible join |
| `results/stage11_mvtecad2_multicategory/stage11_c_candidate_regions.csv` | `results/stage11_mvtecad2_multicategory/stage11_d_vlm_image_predictions.csv` | image_path | image_path | 243 | 1.000 | 1.000 | strong possible join |
| `results/stage11_mvtecad2_multicategory/stage11_d_vlm_candidate_scores.csv` | `results/stage11_mvtecad2_multicategory/stage11_d_vlm_image_predictions.csv` | image_path | image_path | 243 | 1.000 | 1.000 | strong possible join |
| `results/stage11_mvtecad2_multicategory/stage11_b_patchcore_multicategory_predictions.csv` | `results/stage11_mvtecad2_multicategory/stage11_c_candidate_regions.csv` | image_path | image_path | 118 | 0.217 | 0.486 | partial possible join |
| `results/stage11_mvtecad2_multicategory/stage11_b_patchcore_multicategory_predictions.csv` | `results/stage11_mvtecad2_multicategory/stage11_d_vlm_candidate_scores.csv` | image_path | image_path | 118 | 0.217 | 0.486 | partial possible join |
| `results/stage11_mvtecad2_multicategory/stage11_b_patchcore_multicategory_predictions.csv` | `results/stage11_mvtecad2_multicategory/stage11_d_vlm_image_predictions.csv` | image_path | image_path | 118 | 0.217 | 0.486 | partial possible join |
| `results/stage11_mvtecad2_multicategory/stage11_b_patchcore_multicategory_predictions.csv` | `results/stage11_mvtecad2_multicategory/stage11_b1_patchcore_detector_quality_analysis.csv` | category | category | 8 | 1.000 | 1.000 | strong possible join |
| `results/stage11_mvtecad2_multicategory/stage11_e_multicategory_evidence_table.csv` | `results/stage13_strong_baseline/stage13_a_patchcore_vlm_score_complementarity.csv` | category | category | 5 | 1.000 | 1.000 | strong possible join |
| `results/stage11_mvtecad2_multicategory/stage11_c_candidate_regions.csv` | `results/stage11_mvtecad2_multicategory/stage11_e_multicategory_evidence_table.csv` | category | category | 4 | 1.000 | 0.800 | strong possible join |
| `results/stage11_mvtecad2_multicategory/stage11_c_candidate_regions.csv` | `results/stage13_strong_baseline/stage13_a_patchcore_vlm_fusion_per_category.csv` | category | category | 4 | 1.000 | 1.000 | strong possible join |
| `results/stage11_mvtecad2_multicategory/stage11_c_candidate_regions.csv` | `results/stage13_strong_baseline/stage13_a_patchcore_vlm_score_complementarity.csv` | category | category | 4 | 1.000 | 0.800 | strong possible join |
| `results/stage11_mvtecad2_multicategory/stage11_d_vlm_candidate_scores.csv` | `results/stage11_mvtecad2_multicategory/stage11_e_multicategory_evidence_table.csv` | category | category | 4 | 1.000 | 0.800 | strong possible join |
| `results/stage11_mvtecad2_multicategory/stage11_d_vlm_candidate_scores.csv` | `results/stage13_strong_baseline/stage13_a_patchcore_vlm_fusion_per_category.csv` | category | category | 4 | 1.000 | 1.000 | strong possible join |
| `results/stage11_mvtecad2_multicategory/stage11_d_vlm_candidate_scores.csv` | `results/stage13_strong_baseline/stage13_a_patchcore_vlm_score_complementarity.csv` | category | category | 4 | 1.000 | 0.800 | strong possible join |
| `results/stage11_mvtecad2_multicategory/stage11_d_vlm_image_predictions.csv` | `results/stage11_mvtecad2_multicategory/stage11_e_multicategory_evidence_table.csv` | category | category | 4 | 1.000 | 0.800 | strong possible join |
| `results/stage11_mvtecad2_multicategory/stage11_d_vlm_image_predictions.csv` | `results/stage13_strong_baseline/stage13_a_patchcore_vlm_fusion_per_category.csv` | category | category | 4 | 1.000 | 1.000 | strong possible join |
| `results/stage11_mvtecad2_multicategory/stage11_d_vlm_image_predictions.csv` | `results/stage13_strong_baseline/stage13_a_patchcore_vlm_score_complementarity.csv` | category | category | 4 | 1.000 | 0.800 | strong possible join |
| `results/stage13_strong_baseline/stage13_a_patchcore_vlm_fusion_per_category.csv` | `results/stage13_strong_baseline/stage13_a_patchcore_vlm_score_complementarity.csv` | category | category | 4 | 1.000 | 0.800 | strong possible join |
| `results/stage11_mvtecad2_multicategory/stage11_e_multicategory_evidence_table.csv` | `results/stage13_strong_baseline/stage13_a_patchcore_vlm_fusion_per_category.csv` | category | category | 4 | 0.800 | 1.000 | strong possible join |
| `results/stage11_mvtecad2_multicategory/stage11_b_patchcore_multicategory_predictions.csv` | `results/stage11_mvtecad2_multicategory/stage11_e_multicategory_evidence_table.csv` | category | category | 4 | 0.500 | 0.800 | partial possible join |

## Decision rule

- If a qcr_ready file exists, proceed to Stage 18-B2 directly.
- If partial files have strong joins and contain D/M/Q/label across files, assemble in Stage 18-B2.
- If VLM or quality is missing, proceed to Stage 18-C to generate missing AD2 QCR predictions.

## Recommended next action

`proceed_to_stage18_b2_direct_qcr_assembly`

## Outputs

- `results/stage18_ad2_qcr_ablation/stage18_b1_ad2_qcr_source_schema_profile.csv`
- `results/stage18_ad2_qcr_ablation/stage18_b1_ad2_qcr_source_columns_long.csv`
- `results/stage18_ad2_qcr_ablation/stage18_b1_ad2_qcr_join_feasibility.csv`
