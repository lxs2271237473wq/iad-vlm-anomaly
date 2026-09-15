# Stage 10-B0 MVTec AD 2 Layout Validation

## 1. Purpose

This stage validates whether MVTec AD 2 has been placed under the expected project path.
It does not train models, run anomaly detectors, run VLM reasoning, or modify old results.

## 2. Expected Dataset Root

`datasets/MVTec_AD_2`

## 3. Validation Status

- Status: `images_found`
- Total files: `8711`
- Total images: `8709`

## 4. Directory Summary

| Level | Path | Exists | Dirs | Files | Images |
|---|---|---:|---:|---:|---:|
| dataset_root | `datasets/MVTec_AD_2` | True | 96 | 8711 | 8709 |
| top_level_dir | `datasets/MVTec_AD_2/can` | True | 11 | 1352 | 1352 |
| second_level_dir | `datasets/MVTec_AD_2/can/test_private` | True | 0 | 321 | 321 |
| second_level_dir | `datasets/MVTec_AD_2/can/test_private_mixed` | True | 0 | 321 | 321 |
| second_level_dir | `datasets/MVTec_AD_2/can/test_public` | True | 4 | 252 | 252 |
| second_level_dir | `datasets/MVTec_AD_2/can/train` | True | 1 | 412 | 412 |
| second_level_dir | `datasets/MVTec_AD_2/can/validation` | True | 1 | 46 | 46 |
| top_level_dir | `datasets/MVTec_AD_2/fabric` | True | 11 | 1304 | 1304 |
| second_level_dir | `datasets/MVTec_AD_2/fabric/test_private` | True | 0 | 314 | 314 |
| second_level_dir | `datasets/MVTec_AD_2/fabric/test_private_mixed` | True | 0 | 314 | 314 |
| second_level_dir | `datasets/MVTec_AD_2/fabric/test_public` | True | 4 | 246 | 246 |
| second_level_dir | `datasets/MVTec_AD_2/fabric/train` | True | 1 | 387 | 387 |
| second_level_dir | `datasets/MVTec_AD_2/fabric/validation` | True | 1 | 43 | 43 |
| top_level_dir | `datasets/MVTec_AD_2/fruit_jelly` | True | 11 | 950 | 950 |
| second_level_dir | `datasets/MVTec_AD_2/fruit_jelly/test_private` | True | 0 | 255 | 255 |
| second_level_dir | `datasets/MVTec_AD_2/fruit_jelly/test_private_mixed` | True | 0 | 255 | 255 |
| second_level_dir | `datasets/MVTec_AD_2/fruit_jelly/test_public` | True | 4 | 140 | 140 |
| second_level_dir | `datasets/MVTec_AD_2/fruit_jelly/train` | True | 1 | 263 | 263 |
| second_level_dir | `datasets/MVTec_AD_2/fruit_jelly/validation` | True | 1 | 37 | 37 |
| top_level_dir | `datasets/MVTec_AD_2/rice` | True | 11 | 1124 | 1124 |
| second_level_dir | `datasets/MVTec_AD_2/rice/test_private` | True | 0 | 277 | 277 |
| second_level_dir | `datasets/MVTec_AD_2/rice/test_private_mixed` | True | 0 | 277 | 277 |
| second_level_dir | `datasets/MVTec_AD_2/rice/test_public` | True | 4 | 222 | 222 |
| second_level_dir | `datasets/MVTec_AD_2/rice/train` | True | 1 | 313 | 313 |
| second_level_dir | `datasets/MVTec_AD_2/rice/validation` | True | 1 | 35 | 35 |
| top_level_dir | `datasets/MVTec_AD_2/sheet_metal` | True | 11 | 644 | 644 |
| second_level_dir | `datasets/MVTec_AD_2/sheet_metal/test_private` | True | 0 | 142 | 142 |
| second_level_dir | `datasets/MVTec_AD_2/sheet_metal/test_private_mixed` | True | 0 | 142 | 142 |
| second_level_dir | `datasets/MVTec_AD_2/sheet_metal/test_public` | True | 4 | 204 | 204 |
| second_level_dir | `datasets/MVTec_AD_2/sheet_metal/train` | True | 1 | 137 | 137 |
| second_level_dir | `datasets/MVTec_AD_2/sheet_metal/validation` | True | 1 | 19 | 19 |
| top_level_dir | `datasets/MVTec_AD_2/vial` | True | 11 | 1129 | 1129 |
| second_level_dir | `datasets/MVTec_AD_2/vial/test_private` | True | 0 | 276 | 276 |
| second_level_dir | `datasets/MVTec_AD_2/vial/test_private_mixed` | True | 0 | 276 | 276 |
| second_level_dir | `datasets/MVTec_AD_2/vial/test_public` | True | 4 | 245 | 245 |
| second_level_dir | `datasets/MVTec_AD_2/vial/train` | True | 1 | 291 | 291 |
| second_level_dir | `datasets/MVTec_AD_2/vial/validation` | True | 1 | 41 | 41 |
| top_level_dir | `datasets/MVTec_AD_2/wallplugs` | True | 11 | 1030 | 1030 |
| second_level_dir | `datasets/MVTec_AD_2/wallplugs/test_private` | True | 0 | 232 | 232 |
| second_level_dir | `datasets/MVTec_AD_2/wallplugs/test_private_mixed` | True | 0 | 232 | 232 |
| second_level_dir | `datasets/MVTec_AD_2/wallplugs/test_public` | True | 4 | 240 | 240 |
| second_level_dir | `datasets/MVTec_AD_2/wallplugs/train` | True | 1 | 293 | 293 |
| second_level_dir | `datasets/MVTec_AD_2/wallplugs/validation` | True | 1 | 33 | 33 |
| top_level_dir | `datasets/MVTec_AD_2/walnuts` | True | 11 | 1176 | 1176 |
| second_level_dir | `datasets/MVTec_AD_2/walnuts/test_private` | True | 0 | 228 | 228 |
| second_level_dir | `datasets/MVTec_AD_2/walnuts/test_private_mixed` | True | 0 | 228 | 228 |
| second_level_dir | `datasets/MVTec_AD_2/walnuts/test_public` | True | 4 | 240 | 240 |
| second_level_dir | `datasets/MVTec_AD_2/walnuts/train` | True | 1 | 432 | 432 |
| second_level_dir | `datasets/MVTec_AD_2/walnuts/validation` | True | 1 | 48 | 48 |

## 5. Next Step

MVTec AD 2 images are available. Next step: implement Stage 10-B1 manifest builder.