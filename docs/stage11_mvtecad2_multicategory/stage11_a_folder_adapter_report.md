# Stage 11-A MVTec AD 2 Multi-category Folder Adapter Report

## 1. Purpose

This stage builds Anomalib Folder-style datasets for all MVTec AD 2 categories.
It does not train PatchCore, run VLM reasoning, generate crops, or modify original datasets.

## 2. Categories

- can
- fabric
- fruit_jelly
- rice
- sheet_metal
- vial
- wallplugs
- walnuts

## 3. Output Files

- Mapping: `results/stage11_mvtecad2_multicategory/stage11_a_folder_adapter_mapping.csv`
- Summary: `results/stage11_mvtecad2_multicategory/stage11_a_folder_adapter_summary.csv`
- Validation: `results/stage11_mvtecad2_multicategory/stage11_a_folder_adapter_validation.csv`
- Report: `docs/stage11_mvtecad2_multicategory/stage11_a_folder_adapter_report.md`
- Generated folder root: `datasets/MVTec_AD_2_anomalib_all`

## 4. Summary

| Category | Subset | Label | is_anomaly | Images | Masks | Folder root |
|---|---|---|---:|---:|---:|---|
| can | test | bad | 1 | 90 | 90 | `datasets/MVTec_AD_2_anomalib_all/can_folder` |
| can | test | good | 0 | 72 | 0 | `datasets/MVTec_AD_2_anomalib_all/can_folder` |
| can | train | good | 0 | 458 | 0 | `datasets/MVTec_AD_2_anomalib_all/can_folder` |
| fabric | test | bad | 1 | 90 | 90 | `datasets/MVTec_AD_2_anomalib_all/fabric_folder` |
| fabric | test | good | 0 | 66 | 0 | `datasets/MVTec_AD_2_anomalib_all/fabric_folder` |
| fabric | train | good | 0 | 430 | 0 | `datasets/MVTec_AD_2_anomalib_all/fabric_folder` |
| fruit_jelly | test | bad | 1 | 60 | 60 | `datasets/MVTec_AD_2_anomalib_all/fruit_jelly_folder` |
| fruit_jelly | test | good | 0 | 20 | 0 | `datasets/MVTec_AD_2_anomalib_all/fruit_jelly_folder` |
| fruit_jelly | train | good | 0 | 300 | 0 | `datasets/MVTec_AD_2_anomalib_all/fruit_jelly_folder` |
| rice | test | bad | 1 | 90 | 90 | `datasets/MVTec_AD_2_anomalib_all/rice_folder` |
| rice | test | good | 0 | 42 | 0 | `datasets/MVTec_AD_2_anomalib_all/rice_folder` |
| rice | train | good | 0 | 348 | 0 | `datasets/MVTec_AD_2_anomalib_all/rice_folder` |
| sheet_metal | test | bad | 1 | 90 | 90 | `datasets/MVTec_AD_2_anomalib_all/sheet_metal_folder` |
| sheet_metal | test | good | 0 | 24 | 0 | `datasets/MVTec_AD_2_anomalib_all/sheet_metal_folder` |
| sheet_metal | train | good | 0 | 156 | 0 | `datasets/MVTec_AD_2_anomalib_all/sheet_metal_folder` |
| vial | test | bad | 1 | 105 | 105 | `datasets/MVTec_AD_2_anomalib_all/vial_folder` |
| vial | test | good | 0 | 35 | 0 | `datasets/MVTec_AD_2_anomalib_all/vial_folder` |
| vial | train | good | 0 | 332 | 0 | `datasets/MVTec_AD_2_anomalib_all/vial_folder` |
| wallplugs | test | bad | 1 | 90 | 90 | `datasets/MVTec_AD_2_anomalib_all/wallplugs_folder` |
| wallplugs | test | good | 0 | 60 | 0 | `datasets/MVTec_AD_2_anomalib_all/wallplugs_folder` |
| wallplugs | train | good | 0 | 326 | 0 | `datasets/MVTec_AD_2_anomalib_all/wallplugs_folder` |
| walnuts | test | bad | 1 | 90 | 90 | `datasets/MVTec_AD_2_anomalib_all/walnuts_folder` |
| walnuts | test | good | 0 | 60 | 0 | `datasets/MVTec_AD_2_anomalib_all/walnuts_folder` |
| walnuts | train | good | 0 | 480 | 0 | `datasets/MVTec_AD_2_anomalib_all/walnuts_folder` |

## 5. Anomalib Validation

- Successful categories: `8` / `8`

| Category | Success | Train batch | Test batch | Error |
|---|---:|---|---|---|
| can | True | ImageBatch | ImageBatch | `` |
| fabric | True | ImageBatch | ImageBatch | `` |
| fruit_jelly | True | ImageBatch | ImageBatch | `` |
| rice | True | ImageBatch | ImageBatch | `` |
| sheet_metal | True | ImageBatch | ImageBatch | `` |
| vial | True | ImageBatch | ImageBatch | `` |
| wallplugs | True | ImageBatch | ImageBatch | `` |
| walnuts | True | ImageBatch | ImageBatch | `` |

## 6. Next Step

Stage 11-B should run PatchCore pilot in batch mode over all validated categories.
The generated folder datasets are local artifacts and should not be committed to GitHub.