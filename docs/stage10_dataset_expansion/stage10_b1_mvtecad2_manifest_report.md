# Stage 10-B1 MVTec AD 2 Manifest Report

## 1. Purpose

This stage builds a unified manifest for MVTec AD 2.
It does not train models, run anomaly detectors, run VLM reasoning, or modify previous results.

## 2. Dataset Root

`datasets/MVTec_AD_2`

## 3. Output Files

- `results/stage10_dataset_expansion/stage10_b1_mvtecad2_manifest.csv`
- `results/stage10_dataset_expansion/stage10_b1_mvtecad2_manifest_summary.csv`
- `docs/stage10_dataset_expansion/stage10_b1_mvtecad2_manifest_report.md`

## 4. Overall Statistics

- Total manifest images: `1024`
- Labeled images: `472`
- Unlabeled private images: `552`
- Public anomalous images with masks detected: `105` / `105`

## 5. Split Summary

| Category | Split | Label available | is_anomaly | Anomaly type | Images | Masks | Evaluation scope |
|---|---|---:|---:|---|---:|---:|---|
| vial | test_private | False | -1 | unknown_private | 276 | 0 | official_server_only |
| vial | test_private_mixed | False | -1 | unknown_private | 276 | 0 | official_server_only |
| vial | test_public | True | 0 | good | 35 | 0 | local_labeled_test |
| vial | test_public | True | 1 | bad | 105 | 105 | local_labeled_test |
| vial | train | True | 0 | good | 291 | 0 | normal_only |
| vial | validation | True | 0 | good | 41 | 0 | normal_only |

## 6. Important Notes

- `train` and `validation` are treated as normal-only splits.
- `test_public` is treated as the local labeled test split.
- `test_private` and `test_private_mixed` are marked as label-unavailable and should not be used for local AUROC/AP/F1.
- If mask coverage is zero, Stage 10 should first run image-level detector/crop/VLM reasoning; pixel-level evaluation should wait until mask pairing is confirmed.

## 7. Next Step

Stage 10-B2 should adapt this manifest into the existing PatchCore/FastFlow and crop-reasoning pipeline.
For the first pilot, use only:

```text
category = vial
train + validation for normal reference
test_public for local evaluation
test_private/test_private_mixed excluded from local metric computation
```