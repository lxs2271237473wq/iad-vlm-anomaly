# Stage 10-B0 MVTec AD 2 Layout Validation

## 1. Purpose

This stage validates whether MVTec AD 2 has been placed under the expected project path.
It does not train models, run anomaly detectors, run VLM reasoning, or modify old results.

## 2. Expected Dataset Root

`datasets/MVTec_AD_2`

## 3. Validation Status

- Status: `images_found`
- Total files: `1131`
- Total images: `1129`

## 4. Directory Summary

| Level | Path | Exists | Dirs | Files | Images |
|---|---|---:|---:|---:|---:|
| dataset_root | `datasets/MVTec_AD_2` | True | 12 | 1131 | 1129 |
| top_level_dir | `datasets/MVTec_AD_2/vial` | True | 11 | 1129 | 1129 |
| second_level_dir | `datasets/MVTec_AD_2/vial/test_private` | True | 0 | 276 | 276 |
| second_level_dir | `datasets/MVTec_AD_2/vial/test_private_mixed` | True | 0 | 276 | 276 |
| second_level_dir | `datasets/MVTec_AD_2/vial/test_public` | True | 4 | 245 | 245 |
| second_level_dir | `datasets/MVTec_AD_2/vial/train` | True | 1 | 291 | 291 |
| second_level_dir | `datasets/MVTec_AD_2/vial/validation` | True | 1 | 41 | 41 |

## 5. Next Step

MVTec AD 2 images are available. Next step: implement Stage 10-B1 manifest builder.