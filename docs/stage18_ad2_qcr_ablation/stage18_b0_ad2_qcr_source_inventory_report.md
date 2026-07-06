# Stage 18-B0 AD2 QCR Source Inventory

## Purpose

Scan existing result/run files to determine whether AD2 four-category QCR predictions can be assembled from existing per-image sources.

## Summary

- scanned files: `551`
- AD2 high-value QCR-ready/near-ready files: `0`
- AD2 medium-value partial per-image files: `1`
- AD2 low-value summary/category-level files: `58`

## Candidate files

| File | Coverage | Role | Value | Image ID | Label | Detector | VLM | Quality | Notes |
|---|---:|---|---|---|---|---|---|---|---|
| `results/stage11_mvtecad2_multicategory/stage11_d_vlm_candidate_scores.csv` | 4/4 | ad2_partial_per_image_source | medium | image_path | gt_label | patchcore_score |  |  | detector;label |
| `results/stage10_dataset_expansion/stage10_b0_mvtecad2_layout_validation.csv` | 4/4 | ad2_summary_or_category_level_source | low | path |  |  |  |  |  |
| `results/stage10_dataset_expansion/stage10_b1_mvtecad2_manifest_summary.csv` | 4/4 | ad2_summary_or_category_level_source | low |  | is_anomaly |  |  |  | label |
| `results/stage11_mvtecad2_multicategory/stage11_a_folder_adapter_mapping.csv` | 4/4 | ad2_summary_or_category_level_source | low |  | is_anomaly |  |  |  | label |
| `results/stage11_mvtecad2_multicategory/stage11_a_folder_adapter_summary.csv` | 4/4 | ad2_summary_or_category_level_source | low |  | is_anomaly |  |  |  | label |
| `results/stage11_mvtecad2_multicategory/stage11_a_folder_adapter_validation.csv` | 4/4 | ad2_summary_or_category_level_source | low |  |  |  |  |  |  |
| `results/stage11_mvtecad2_multicategory/stage11_b1_patchcore_detector_quality_analysis.csv` | 4/4 | ad2_summary_or_category_level_source | low |  |  |  |  |  |  |
| `results/stage11_mvtecad2_multicategory/stage11_b_patchcore_multicategory_metrics.csv` | 4/4 | ad2_summary_or_category_level_source | low |  |  |  |  |  |  |
| `results/stage11_mvtecad2_multicategory/stage11_b_patchcore_multicategory_predictions.csv` | 4/4 | ad2_summary_or_category_level_source | low | image_path | gt_label |  |  |  | label |
| `results/stage11_mvtecad2_multicategory/stage11_b_patchcore_multicategory_status.csv` | 4/4 | ad2_summary_or_category_level_source | low |  |  |  |  |  |  |
| `results/stage11_mvtecad2_multicategory/stage11_c_candidate_regions.csv` | 4/4 | ad2_summary_or_category_level_source | low | image_path | gt_label |  |  |  | label |
| `results/stage11_mvtecad2_multicategory/stage11_c_candidate_status.csv` | 4/4 | ad2_summary_or_category_level_source | low |  |  |  |  |  |  |
| `results/stage11_mvtecad2_multicategory/stage11_c_candidate_summary.csv` | 4/4 | ad2_summary_or_category_level_source | low |  |  |  |  |  |  |
| `results/stage11_mvtecad2_multicategory/stage11_d_vlm_image_predictions.csv` | 4/4 | ad2_summary_or_category_level_source | low | image_path |  | patchcore_score |  |  | detector |
| `results/stage11_mvtecad2_multicategory/stage11_d_vlm_summary.csv` | 4/4 | ad2_summary_or_category_level_source | low |  |  |  |  |  |  |
| `results/stage11_mvtecad2_multicategory/stage11_e_multicategory_evidence_table.csv` | 4/4 | ad2_summary_or_category_level_source | low |  |  |  |  |  |  |
| `results/stage11_mvtecad2_multicategory/stage11_i_category_usage_decision_table.csv` | 4/4 | ad2_summary_or_category_level_source | low |  |  |  |  |  |  |
| `results/stage11_mvtecad2_multicategory/stage11_i_paper_ready_method_table.csv` | 4/4 | ad2_summary_or_category_level_source | low |  |  |  |  |  |  |
| `results/stage13_strong_baseline/stage13_a_patchcore_vlm_fusion_grid.csv` | 4/4 | ad2_summary_or_category_level_source | low |  |  |  |  |  |  |
| `results/stage13_strong_baseline/stage13_a_patchcore_vlm_fusion_loco_category.csv` | 4/4 | ad2_summary_or_category_level_source | low |  |  |  |  |  |  |
| `results/stage13_strong_baseline/stage13_a_patchcore_vlm_fusion_per_category.csv` | 4/4 | ad2_summary_or_category_level_source | low |  |  |  |  |  |  |
| `results/stage13_strong_baseline/stage13_a_patchcore_vlm_fusion_summary.csv` | 4/4 | ad2_summary_or_category_level_source | low |  |  |  |  |  |  |
| `results/stage13_strong_baseline/stage13_a_patchcore_vlm_score_complementarity.csv` | 4/4 | ad2_summary_or_category_level_source | low |  |  |  |  |  |  |
| `results/stage14_strong_vlm_baselines/stage14_e_primary_external_baseline_comparison.csv` | 4/4 | ad2_summary_or_category_level_source | low |  |  |  |  |  |  |
| `results/stage14_strong_vlm_baselines/stage14_e_winclip_primary_fixed_protocol.csv` | 4/4 | ad2_summary_or_category_level_source | low |  |  |  |  |  |  |
| `results/stage15_modern_detector_baselines/stage15_d_efficientad_primary_fixed_budget.csv` | 4/4 | ad2_summary_or_category_level_source | low |  |  |  |  |  |  |
| `results/stage15_modern_detector_baselines/stage15_d_efficientad_primary_fixed_budget_epoch30_primary4_backup.csv` | 4/4 | ad2_summary_or_category_level_source | low |  |  |  |  |  |  |
| `results/stage15_modern_detector_baselines/stage15_e_primary_unified_baseline_comparison.csv` | 4/4 | ad2_summary_or_category_level_source | low |  |  |  |  |  |  |
| `results/stage10_dataset_expansion/stage10_b1_mvtecad2_manifest.csv` | 3/4 | ad2_summary_or_category_level_source | low | image_path | is_anomaly |  |  |  | label |
| `results/stage10_dataset_expansion/stage10_b2_mvtecad2_folder_mapping.csv` | 1/4 | ad2_summary_or_category_level_source | low |  | is_anomaly |  |  |  | label |
| `results/stage10_dataset_expansion/stage10_b2_mvtecad2_folder_summary.csv` | 1/4 | ad2_summary_or_category_level_source | low |  | is_anomaly |  |  |  | label |
| `results/stage10_dataset_expansion/stage10_c_mvtecad2_vial_patchcore_metrics.csv` | 1/4 | ad2_summary_or_category_level_source | low |  |  |  |  |  |  |
| `results/stage10_dataset_expansion/stage10_c_mvtecad2_vial_patchcore_predictions.csv` | 1/4 | ad2_summary_or_category_level_source | low | image_path | gt_label |  |  |  | label |
| `results/stage10_dataset_expansion/stage10_d_patchcore_candidate_regions.csv` | 1/4 | ad2_summary_or_category_level_source | low | image_path | gt_label |  |  |  | label |
| `results/stage10_dataset_expansion/stage10_d_patchcore_candidate_summary.csv` | 1/4 | ad2_summary_or_category_level_source | low |  |  |  |  |  |  |
| `results/stage10_dataset_expansion/stage10_e_vlm_candidate_scores.csv` | 1/4 | ad2_summary_or_category_level_source | low | image_path | gt_label |  |  |  | label |
| `results/stage10_dataset_expansion/stage10_e_vlm_image_predictions.csv` | 1/4 | ad2_summary_or_category_level_source | low | image_path |  |  |  |  |  |
| `results/stage10_dataset_expansion/stage10_e_vlm_summary.csv` | 1/4 | ad2_summary_or_category_level_source | low |  |  |  |  |  |  |
| `results/stage10_dataset_expansion/stage10_f_multiscale_context_crop_scores.csv` | 1/4 | ad2_summary_or_category_level_source | low | image_path | gt_label |  |  |  | label |
| `results/stage10_dataset_expansion/stage10_f_multiscale_context_image_predictions.csv` | 1/4 | ad2_summary_or_category_level_source | low | image_path |  |  |  |  |  |
| `results/stage10_dataset_expansion/stage10_f_multiscale_context_summary.csv` | 1/4 | ad2_summary_or_category_level_source | low |  |  |  |  |  |  |
| `results/stage10_dataset_expansion/stage10_g_mvtecad2_vial_final_table.csv` | 1/4 | ad2_summary_or_category_level_source | low |  |  |  |  |  |  |
| `results/stage11_mvtecad2_multicategory/stage11_f1_vial_path_aligned_drift_matched.csv` | 1/4 | ad2_summary_or_category_level_source | low |  |  |  |  |  |  |
| `results/stage11_mvtecad2_multicategory/stage11_f1_vial_path_aligned_drift_summary.csv` | 1/4 | ad2_summary_or_category_level_source | low |  |  |  |  |  |  |
| `results/stage11_mvtecad2_multicategory/stage11_f2_vial_source_aligned_drift_matched.csv` | 1/4 | ad2_summary_or_category_level_source | low |  |  |  |  |  |  |
| `results/stage11_mvtecad2_multicategory/stage11_f2_vial_source_aligned_drift_summary.csv` | 1/4 | ad2_summary_or_category_level_source | low |  |  |  |  |  |  |
| `results/stage11_mvtecad2_multicategory/stage11_f2_vial_source_aligned_drift_unmatched.csv` | 1/4 | ad2_summary_or_category_level_source | low |  |  |  |  |  |  |
| `results/stage11_mvtecad2_multicategory/stage11_f3_vial_image_set_keys.csv` | 1/4 | ad2_summary_or_category_level_source | low | image_path | label |  |  |  | label |
| `results/stage11_mvtecad2_multicategory/stage11_f3_vial_image_set_overlap.csv` | 1/4 | ad2_summary_or_category_level_source | low |  |  |  |  |  |  |
| `results/stage11_mvtecad2_multicategory/stage11_f3_vial_image_set_overlap_summary.csv` | 1/4 | ad2_summary_or_category_level_source | low |  |  |  |  |  |  |
| `results/stage11_mvtecad2_multicategory/stage11_f_vial_candidate_policy_drift_matched.csv` | 1/4 | ad2_summary_or_category_level_source | low | image_key |  |  |  |  |  |
| `results/stage11_mvtecad2_multicategory/stage11_f_vial_candidate_policy_drift_summary.csv` | 1/4 | ad2_summary_or_category_level_source | low |  |  |  |  |  |  |
| `results/stage14_strong_vlm_baselines/stage14_c2_winclip_fruit_jelly_metrics.csv` | 1/4 | ad2_summary_or_category_level_source | low |  |  |  |  |  |  |
| `results/stage14_strong_vlm_baselines/stage14_c3_fruit_jelly_external_baseline_comparison.csv` | 1/4 | ad2_summary_or_category_level_source | low |  |  |  |  |  |  |
| `results/stage14_strong_vlm_baselines/stage14_d_winclip_fruit_jelly_sensitivity.csv` | 1/4 | ad2_summary_or_category_level_source | low |  |  |  |  |  |  |
| `results/stage15_modern_detector_baselines/stage15_b_efficientad_fruit_jelly_metrics.csv` | 1/4 | ad2_summary_or_category_level_source | low |  |  |  |  |  |  |
| `results/stage15_modern_detector_baselines/stage15_c_fruit_jelly_modern_baseline_comparison.csv` | 1/4 | ad2_summary_or_category_level_source | low |  |  |  |  |  |  |
| `results/stage17_defensive_sensitivity/stage17_a_efficientad100_fruit_jelly.csv` | 1/4 | ad2_summary_or_category_level_source | low |  |  |  |  |  |  |
| `results/stage17_defensive_sensitivity/stage17_a_efficientad100_vs_30_delta.csv` | 1/4 | ad2_summary_or_category_level_source | low |  |  |  |  |  |  |

## Decision rule

- If high-value files exist, proceed to Stage 18-B1: assemble AD2 QCR predictions from existing sources.
- If only medium-value files exist, inspect whether detector/VLM/quality sources can be joined by image key.
- If no high/medium-value files exist, proceed to Stage 18-C: generate AD2 QCR predictions from scratch.

## Recommended next action

`inspect_medium_sources_then_assemble_or_generate_missing_parts`
