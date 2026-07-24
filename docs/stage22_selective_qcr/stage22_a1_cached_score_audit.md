# Stage 22-A1: Cached Score Artifact Audit

## Purpose

Locate existing per-image cached scores required for
offline Selective QCR experiments without repeating
detector or VLM inference.

## Required information

- detector evidence `D`
- crop/VLM evidence `M`
- candidate quality `Q`
- optional consistency `C`
- binary anomaly label
- category and image identifier

## Candidate files (190)

| Rank | File | Size MiB | Rows | Detected groups |
|---:|---|---:|---:|---|
| 1 | `results/stage18_ad2_qcr_ablation/stage18_b2_ad2_qcr_assembled_predictions.csv` | 0.153 | 243 | detector=D; vlm=M; quality=Q; consistency=agreement; category=category; path=image_path |
| 2 | `results/stage16_qcru_ablation/stage16_e_failure_boundary_case_inventory.csv` | 0.038 | 80 | detector=D; vlm=M; quality=Q; consistency=agreement; category=category |
| 3 | `results/stage18_ad2_qcr_ablation/stage18_b2_ad2_qcr_ablation_per_category.csv` | 0.005 | 28 | category=category |
| 4 | `results/stage18_ad2_qcr_ablation/stage18_b3_ad2_q_source_sweep_per_category.csv` | 0.044 | 200 | category=category |
| 5 | `results/stage18_ad2_qcr_ablation/stage18_b5_ad2_loco_qcr_all_configs_per_category.csv` | 0.581 | 4256 | category=category |
| 6 | `results/stage10_dataset_expansion/stage10_b1_mvtecad2_manifest.csv` | 1.311 | 8004 | label=is_anomaly; category=category; path=filename,image_path; split=split |
| 7 | `results/stage7_generalization/visa_manifest/visa_image_manifest.csv` | 2.428 | 10821 | label=is_anomaly,label; category=category; path=image_path; split=split |
| 8 | `results/stage11_mvtecad2_multicategory/stage11_d_vlm_candidate_scores.csv` | 0.408 | 502 | detector=patchcore_score; category=category; path=image_path |
| 9 | `results/stage11_mvtecad2_multicategory/stage11_d_vlm_image_predictions.csv` | 0.144 | 243 | detector=patchcore_score; category=category; path=image_path |
| 10 | `results/stage11_mvtecad2_multicategory/stage11_h_fabric_vlm_candidate_scores.csv` | 0.155 | 200 | detector=patchcore_score; category=category; path=image_path |
| 11 | `results/stage11_mvtecad2_multicategory/stage11_h_fabric_vlm_image_predictions.csv` | 0.047 | 78 | detector=patchcore_score; category=category; path=image_path |
| 12 | `results/stage7_generalization/visa_binary_prompt_reasoning/visa_binary_prompt_predictions.csv` | 5.107 | 19458 | label=is_anomaly,label; category=category; path=image_path |
| 13 | `results/stage7_generalization/visa_multibackbone/fastflow_12cls/VisA/candle/fastflow_image_predictions.csv` | 0.044 | 200 | label=is_anomaly,label; category=category; path=image_path |
| 14 | `results/stage7_generalization/visa_multibackbone/fastflow_12cls/VisA/capsules/fastflow_image_predictions.csv` | 0.037 | 160 | label=is_anomaly,label; category=category; path=image_path |
| 15 | `results/stage7_generalization/visa_multibackbone/fastflow_12cls/VisA/cashew/fastflow_image_predictions.csv` | 0.033 | 150 | label=is_anomaly,label; category=category; path=image_path |
| 16 | `results/stage7_generalization/visa_multibackbone/fastflow_12cls/VisA/chewinggum/fastflow_image_predictions.csv` | 0.036 | 150 | label=is_anomaly,label; category=category; path=image_path |
| 17 | `results/stage7_generalization/visa_multibackbone/fastflow_12cls/VisA/fryum/fastflow_image_predictions.csv` | 0.032 | 150 | label=is_anomaly,label; category=category; path=image_path |
| 18 | `results/stage7_generalization/visa_multibackbone/fastflow_12cls/VisA/macaroni1/fastflow_image_predictions.csv` | 0.047 | 200 | label=is_anomaly,label; category=category; path=image_path |
| 19 | `results/stage7_generalization/visa_multibackbone/fastflow_12cls/VisA/macaroni2/fastflow_image_predictions.csv` | 0.047 | 200 | label=is_anomaly,label; category=category; path=image_path |
| 20 | `results/stage7_generalization/visa_multibackbone/fastflow_12cls/VisA/pcb1/fastflow_image_predictions.csv` | 0.042 | 200 | label=is_anomaly,label; category=category; path=image_path |
| 21 | `results/stage7_generalization/visa_multibackbone/fastflow_12cls/VisA/pcb2/fastflow_image_predictions.csv` | 0.042 | 200 | label=is_anomaly,label; category=category; path=image_path |
| 22 | `results/stage7_generalization/visa_multibackbone/fastflow_12cls/VisA/pcb3/fastflow_image_predictions.csv` | 0.042 | 201 | label=is_anomaly,label; category=category; path=image_path |
| 23 | `results/stage7_generalization/visa_multibackbone/fastflow_12cls/VisA/pcb4/fastflow_image_predictions.csv` | 0.043 | 201 | label=is_anomaly,label; category=category; path=image_path |
| 24 | `results/stage7_generalization/visa_multibackbone/fastflow_12cls/VisA/pipe_fryum/fastflow_image_predictions.csv` | 0.036 | 150 | label=is_anomaly,label; category=category; path=image_path |
| 25 | `results/stage7_generalization/visa_multibackbone/fastflow_binary_prompt_reasoning/visa_binary_prompt_predictions.csv` | 1.712 | 6486 | label=is_anomaly,label; category=category; path=image_path |
| 26 | `results/stage7_generalization/visa_multibackbone/fastflow_candle_full/VisA/candle/fastflow_image_predictions.csv` | 0.044 | 200 | label=is_anomaly,label; category=category; path=image_path |
| 27 | `results/stage7_generalization/visa_multibackbone/fastflow_smoke/VisA/candle/fastflow_image_predictions.csv` | 0.044 | 200 | label=is_anomaly,label; category=category; path=image_path |
| 28 | `results/stage7_generalization/visa_patchcore/VisA/candle/patchcore_image_predictions.csv` | 0.044 | 200 | label=is_anomaly,label; category=category; path=image_path |
| 29 | `results/stage7_generalization/visa_patchcore/VisA/capsules/patchcore_image_predictions.csv` | 0.037 | 160 | label=is_anomaly,label; category=category; path=image_path |
| 30 | `results/stage7_generalization/visa_patchcore/VisA/cashew/patchcore_image_predictions.csv` | 0.033 | 150 | label=is_anomaly,label; category=category; path=image_path |
| 31 | `results/stage7_generalization/visa_patchcore/VisA/chewinggum/patchcore_image_predictions.csv` | 0.036 | 150 | label=is_anomaly,label; category=category; path=image_path |
| 32 | `results/stage7_generalization/visa_patchcore/VisA/fryum/patchcore_image_predictions.csv` | 0.032 | 150 | label=is_anomaly,label; category=category; path=image_path |
| 33 | `results/stage7_generalization/visa_patchcore/VisA/macaroni1/patchcore_image_predictions.csv` | 0.047 | 200 | label=is_anomaly,label; category=category; path=image_path |
| 34 | `results/stage7_generalization/visa_patchcore/VisA/macaroni2/patchcore_image_predictions.csv` | 0.047 | 200 | label=is_anomaly,label; category=category; path=image_path |
| 35 | `results/stage7_generalization/visa_patchcore/VisA/pcb1/patchcore_image_predictions.csv` | 0.042 | 200 | label=is_anomaly,label; category=category; path=image_path |
| 36 | `results/stage7_generalization/visa_patchcore/VisA/pcb2/patchcore_image_predictions.csv` | 0.042 | 200 | label=is_anomaly,label; category=category; path=image_path |
| 37 | `results/stage7_generalization/visa_patchcore/VisA/pcb3/patchcore_image_predictions.csv` | 0.043 | 201 | label=is_anomaly,label; category=category; path=image_path |
| 38 | `results/stage7_generalization/visa_patchcore/VisA/pcb4/patchcore_image_predictions.csv` | 0.042 | 201 | label=is_anomaly,label; category=category; path=image_path |
| 39 | `results/stage7_generalization/visa_patchcore/VisA/pipe_fryum/patchcore_image_predictions.csv` | 0.036 | 150 | label=is_anomaly,label; category=category; path=image_path |
| 40 | `results/stage18_ad2_qcr_ablation/stage18_a0_ad2_qcr_inventory.csv` | 0.001 | 4 | none |
| 41 | `results/stage18_ad2_qcr_ablation/stage18_b0_ad2_qcr_source_candidates.csv` | 0.013 | 59 | none |
| 42 | `results/stage18_ad2_qcr_ablation/stage18_b0_ad2_qcr_source_inventory.csv` | 0.106 | 551 | none |
| 43 | `results/stage18_ad2_qcr_ablation/stage18_b1_ad2_qcr_join_feasibility.csv` | 0.006 | 28 | none |
| 44 | `results/stage18_ad2_qcr_ablation/stage18_b1_ad2_qcr_source_columns_long.csv` | 0.039 | 177 | none |
| 45 | `results/stage18_ad2_qcr_ablation/stage18_b1_ad2_qcr_source_schema_profile.csv` | 0.006 | 8 | none |
| 46 | `results/stage18_ad2_qcr_ablation/stage18_b2_ad2_qcr_ablation_summary.csv` | 0.001 | 7 | none |
| 47 | `results/stage18_ad2_qcr_ablation/stage18_b2_ad2_qcr_claim_ready_deltas.csv` | 0.001 | 6 | none |
| 48 | `results/stage18_ad2_qcr_ablation/stage18_b3_ad2_q_source_sweep_ranked.csv` | 0.013 | 50 | none |
| 49 | `results/stage18_ad2_qcr_ablation/stage18_b3_ad2_q_source_sweep_summary.csv` | 0.013 | 50 | none |
| 50 | `results/stage18_ad2_qcr_ablation/stage18_b4_ad2_qcr_claim_safe_decision.csv` | 0.001 | 4 | none |

## Interpretation rule

A file is immediately usable only if sample-level detector
evidence, VLM evidence, quality, labels, and sample/category
identifiers can be aligned without reconstructing values
from aggregate metrics.
