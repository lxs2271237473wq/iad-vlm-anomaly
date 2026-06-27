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

- Total manifest images: `8004`
- Labeled images: `3914`
- Unlabeled private images: `4090`
- Public anomalous images with masks detected: `705` / `705`

## 5. Split Summary

| Category | Split | Label available | is_anomaly | Anomaly type | Images | Masks | Evaluation scope |
|---|---|---:|---:|---|---:|---:|---|
| can | test_private | False | -1 | unknown_private | 321 | 0 | official_server_only |
| can | test_private_mixed | False | -1 | unknown_private | 321 | 0 | official_server_only |
| can | test_public | True | 0 | good | 72 | 0 | local_labeled_test |
| can | test_public | True | 1 | bad | 90 | 90 | local_labeled_test |
| can | train | True | 0 | good | 412 | 0 | normal_only |
| can | validation | True | 0 | good | 46 | 0 | normal_only |
| fabric | test_private | False | -1 | unknown_private | 314 | 0 | official_server_only |
| fabric | test_private_mixed | False | -1 | unknown_private | 314 | 0 | official_server_only |
| fabric | test_public | True | 0 | good | 66 | 0 | local_labeled_test |
| fabric | test_public | True | 1 | bad | 90 | 90 | local_labeled_test |
| fabric | train | True | 0 | good | 387 | 0 | normal_only |
| fabric | validation | True | 0 | good | 43 | 0 | normal_only |
| fruit_jelly | test_private | False | -1 | unknown_private | 255 | 0 | official_server_only |
| fruit_jelly | test_private_mixed | False | -1 | unknown_private | 255 | 0 | official_server_only |
| fruit_jelly | test_public | True | 0 | good | 20 | 0 | local_labeled_test |
| fruit_jelly | test_public | True | 1 | bad | 60 | 60 | local_labeled_test |
| fruit_jelly | train | True | 0 | good | 263 | 0 | normal_only |
| fruit_jelly | validation | True | 0 | good | 37 | 0 | normal_only |
| rice | test_private | False | -1 | unknown_private | 277 | 0 | official_server_only |
| rice | test_private_mixed | False | -1 | unknown_private | 277 | 0 | official_server_only |
| rice | test_public | True | 0 | good | 42 | 0 | local_labeled_test |
| rice | test_public | True | 1 | bad | 90 | 90 | local_labeled_test |
| rice | train | True | 0 | good | 313 | 0 | normal_only |
| rice | validation | True | 0 | good | 35 | 0 | normal_only |
| sheet_metal | test_private | False | -1 | unknown_private | 142 | 0 | official_server_only |
| sheet_metal | test_private_mixed | False | -1 | unknown_private | 142 | 0 | official_server_only |
| sheet_metal | test_public | True | 0 | good | 24 | 0 | local_labeled_test |
| sheet_metal | test_public | True | 1 | bad | 90 | 90 | local_labeled_test |
| sheet_metal | train | True | 0 | good | 137 | 0 | normal_only |
| sheet_metal | validation | True | 0 | good | 19 | 0 | normal_only |
| vial | test_private | False | -1 | unknown_private | 276 | 0 | official_server_only |
| vial | test_private_mixed | False | -1 | unknown_private | 276 | 0 | official_server_only |
| vial | test_public | True | 0 | good | 35 | 0 | local_labeled_test |
| vial | test_public | True | 1 | bad | 105 | 105 | local_labeled_test |
| vial | train | True | 0 | good | 291 | 0 | normal_only |
| vial | validation | True | 0 | good | 41 | 0 | normal_only |
| wallplugs | test_private | False | -1 | unknown_private | 232 | 0 | official_server_only |
| wallplugs | test_private_mixed | False | -1 | unknown_private | 232 | 0 | official_server_only |
| wallplugs | test_public | True | 0 | good | 60 | 0 | local_labeled_test |
| wallplugs | test_public | True | 1 | bad | 90 | 90 | local_labeled_test |
| wallplugs | train | True | 0 | good | 293 | 0 | normal_only |
| wallplugs | validation | True | 0 | good | 33 | 0 | normal_only |
| walnuts | test_private | False | -1 | unknown_private | 228 | 0 | official_server_only |
| walnuts | test_private_mixed | False | -1 | unknown_private | 228 | 0 | official_server_only |
| walnuts | test_public | True | 0 | good | 60 | 0 | local_labeled_test |
| walnuts | test_public | True | 1 | bad | 90 | 90 | local_labeled_test |
| walnuts | train | True | 0 | good | 432 | 0 | normal_only |
| walnuts | validation | True | 0 | good | 48 | 0 | normal_only |

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

<!-- stage10_full_ad2_manifest_sync_20260627_140302_089820 categories=can,fabric,fruit_jelly,rice,sheet_metal,vial,wallplugs,walnuts -->
