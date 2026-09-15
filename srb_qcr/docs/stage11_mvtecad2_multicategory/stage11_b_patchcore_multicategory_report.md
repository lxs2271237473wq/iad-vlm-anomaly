# Stage 11-B MVTec AD 2 Multi-category PatchCore Baseline

## 1. Purpose

This stage runs PatchCore baseline experiments over all validated MVTec AD 2 categories using the Stage 11-A Anomalib Folder adapters.

This step trains and evaluates PatchCore. It does not run VLM reasoning, generate candidate crops, or modify the original dataset.

## 2. Input

- Folder adapter root: `datasets/MVTec_AD_2_anomalib_all`
- Stage 11-A summary: `results/stage11_mvtecad2_multicategory/stage11_a_folder_adapter_summary.csv`
- Normal training split: `train/good`
- Public test normal split: `test/good`
- Public test anomaly split: `test/bad`
- Mask split: `ground_truth/bad`

## 3. PatchCore Configuration

| Item | Value |
|---|---|
| Backbone | wide_resnet50_2 |
| Layers | layer2, layer3 |
| Pretrained | True |
| Neighbors | 9 |
| Train batch size | 8 |
| Eval batch size | 8 |
| Workers | 0 |

## 4. Output Files

- Metrics: `results/stage11_mvtecad2_multicategory/stage11_b_patchcore_multicategory_metrics.csv`
- Predictions: `results/stage11_mvtecad2_multicategory/stage11_b_patchcore_multicategory_predictions.csv`
- Status: `results/stage11_mvtecad2_multicategory/stage11_b_patchcore_multicategory_status.csv`
- Report: `docs/stage11_mvtecad2_multicategory/stage11_b_patchcore_multicategory_report.md`
- Run root: `runs/stage11_mvtecad2_multicategory/patchcore_baseline`

## 5. Dataset Size by Category

| Category | Train good | Test good | Test bad | Test total |
|---|---:|---:|---:|---:|
| can | 458 | 72 | 90 | 162 |
| fabric | 430 | 66 | 90 | 156 |
| fruit_jelly | 300 | 20 | 60 | 80 |
| rice | 348 | 42 | 90 | 132 |
| sheet_metal | 156 | 24 | 90 | 114 |
| vial | 332 | 35 | 105 | 140 |
| wallplugs | 326 | 60 | 90 | 150 |
| walnuts | 480 | 60 | 90 | 150 |

## 6. Execution Status

| Category | Success | Fit | Test | Predict | Prediction rows | Time sec | Error |
|---|---:|---:|---:|---:|---:|---:|---|
| can | True | True | True | True | 81 | 922.1 | `` |
| fabric | True | True | True | True | 78 | 944.3 | `` |
| fruit_jelly | True | True | True | True | 40 | 501.0 | `` |
| rice | True | True | True | True | 66 | 730.2 | `` |
| sheet_metal | True | True | True | True | 57 | 382.2 | `` |
| vial | True | True | True | True | 71 | 618.4 | `` |
| wallplugs | True | True | True | True | 75 | 668.6 | `` |
| walnuts | True | True | True | True | 75 | 963.3 | `` |

## 7. Metrics

| category | image_AUROC | image_F1Score | pixel_AUROC | pixel_F1Score |
|---|---|---|---|---|
| can | 0.3901 | 0.6441 | 0.7119 | 0.0002 |
| fabric | 0.7582 | 0.7789 | 0.7871 | 0.0765 |
| fruit_jelly | 0.7900 | 0.8788 | 0.9476 | 0.4963 |
| rice | 0.5630 | 0.8108 | 0.7637 | 0.0552 |
| sheet_metal | 0.8315 | 0.9000 | 0.8595 | 0.3765 |
| vial | 0.7987 | 0.8598 | 0.9484 | 0.3366 |
| wallplugs | 0.4626 | 0.7458 | 0.8675 | 0.0391 |
| walnuts | 0.7822 | 0.6471 | 0.9193 | 0.3918 |

## 8. Interpretation

This table establishes the detector-side baseline across all MVTec AD 2 categories.
The next step should not compare VLM methods before confirming detector quality per category, because poor localization can make crop-based VLM reasoning look artificially weak.

## 9. Next Step

Stage 11-C should generate PatchCore candidate regions for the successfully validated categories, then Stage 11-D should evaluate full-image versus context-aware crop VLM reasoning.