# Stage 10-D PatchCore Candidate Crop Extraction

## 1. Purpose

This stage extracts candidate anomaly regions from PatchCore anomaly maps on MVTec AD 2 vial.
It reruns PatchCore fit/predict because Stage 10-C stored only metrics and summarized predictions.

## 2. Input

- Folder dataset: `datasets/MVTec_AD_2_anomalib/vial_folder`
- Model: PatchCore / wide_resnet50_2 / layer2+layer3

## 3. Candidate Extraction Rule

- Threshold quantile: `0.97`
- Top-k candidates per image: `3`
- Minimum area ratio: `0.0005`
- Padding ratio: `0.12`

## 4. Output Files

- Candidate CSV: `results/stage10_dataset_expansion/stage10_d_patchcore_candidate_regions.csv`
- Summary CSV: `results/stage10_dataset_expansion/stage10_d_patchcore_candidate_summary.csv`
- Crop directory: `results/stage10_dataset_expansion/stage10_d_patchcore_candidate_crops`
- Report: `docs/stage10_dataset_expansion/stage10_d_patchcore_candidate_report.md`

## 5. Summary

| Dataset | Category | Images | Candidate rows | Images with candidates | Coverage | Mean candidates/image | Mean pred score |
|---|---|---:|---:|---:|---:|---:|---:|
| MVTec AD 2 | vial | 71 | 123 | 71 | 1.0000 | 1.7324 | 0.7242 |

## 6. Error Status

- Predict error: ``

Error counts:

| Error | Count |
|---|---:|
| none | 123 |

## 7. Next Step

Stage 10-E should run VLM binary prompt reasoning on full images versus PatchCore candidate crops.
The crop directory is generated locally and should not be committed to GitHub unless a small qualitative subset is selected.