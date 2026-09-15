# Stage 10-C MVTec AD 2 Vial PatchCore Pilot

## 1. Purpose

This stage runs the first PatchCore pilot on MVTec AD 2 vial using the Anomalib Folder adapter.
It trains PatchCore on normal train/validation images and evaluates on public test normal/anomaly images.

## 2. Input Dataset

- Folder root: `datasets/MVTec_AD_2_anomalib/vial_folder`
- Train normal: `train/good`
- Test normal: `test/good`
- Test anomaly: `test/bad`
- Masks: `ground_truth/bad`

## 3. Output Files

- Metrics: `results/stage10_dataset_expansion/stage10_c_mvtecad2_vial_patchcore_metrics.csv`
- Predictions: `results/stage10_dataset_expansion/stage10_c_mvtecad2_vial_patchcore_predictions.csv`
- Report: `docs/stage10_dataset_expansion/stage10_c_mvtecad2_vial_patchcore_report.md`
- Run dir: `runs/stage10_mvtecad2_patchcore/vial_patchcore_pilot`

## 4. Test Metrics

| Metric | Value |
|---|---:|
| image_AUROC | 0.8375262022018433 |
| image_F1Score | 0.8807339668273926 |
| pixel_AUROC | 0.9298676252365112 |
| pixel_F1Score | 0.22310766577720642 |

## 5. Prediction Extraction

- Prediction rows: `71`
- Prediction extraction error: ``

## 6. Interpretation

This is the first detector baseline on MVTec AD 2 vial.
If metrics are valid and prediction rows contain anomaly maps or scores, Stage 10-D should generate candidate crops from these PatchCore outputs.
If prediction rows are empty, Stage 10-D should first patch prediction extraction before VLM reasoning.