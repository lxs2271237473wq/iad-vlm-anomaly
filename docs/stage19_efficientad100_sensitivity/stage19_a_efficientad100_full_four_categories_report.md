# Stage 19-A EfficientAD-100 Full Four-category Sensitivity

## Purpose

Run EfficientAD with a 100-epoch budget on the AD2 four-category setting and compare against the existing EfficientAD-30 fixed-budget baseline.

## Config

- categories requested: `fruit_jelly; sheet_metal; vial; walnuts`
- max_epochs: `100`
- eval_batch_size: `64`
- num_workers: `16`
- precision: `16-mixed`
- seeded fruit_jelly from Stage17: `True`

## Completion

- successful categories: `4/4`
- success list: `fruit_jelly; sheet_metal; vial; walnuts`

## Decision

`EfficientAD-100 improves image AUROC over EfficientAD-30 on average.`

## EfficientAD-100 vs EfficientAD-30 summary

| Metric | Categories | EfficientAD-30 mean | EfficientAD-100 mean | Delta | 100 better | 100 worse |
|---|---:|---:|---:|---:|---:|---:|
| image_AUROC | 4 | 0.7604 | 0.7657 | +0.0052 | 1 | 3 |
| image_F1Score | 4 | 0.8423 | 0.8585 | +0.0162 | 2 | 2 |
| pixel_AUROC | 4 | 0.8093 | 0.8346 | +0.0253 | 4 | 0 |
| pixel_F1Score | 4 | 0.3104 | 0.3298 | +0.0194 | 3 | 1 |

## Per-category delta

| Category | Metric | EAD-30 | EAD-100 | Delta |
|---|---|---:|---:|---:|
| fruit_jelly | image_AUROC | 0.8433 | 0.8267 | -0.0167 |
| fruit_jelly | image_F1Score | 0.8571 | 0.8438 | -0.0134 |
| fruit_jelly | pixel_AUROC | 0.7894 | 0.8424 | +0.0531 |
| fruit_jelly | pixel_F1Score | 0.5395 | 0.5561 | +0.0166 |
| sheet_metal | image_AUROC | 0.7333 | 0.7083 | -0.0250 |
| sheet_metal | image_F1Score | 0.8842 | 0.8824 | -0.0019 |
| sheet_metal | pixel_AUROC | 0.7510 | 0.7777 | +0.0267 |
| sheet_metal | pixel_F1Score | 0.2995 | 0.2956 | -0.0039 |
| vial | image_AUROC | 0.9099 | 0.9036 | -0.0063 |
| vial | image_F1Score | 0.8991 | 0.9444 | +0.0454 |
| vial | pixel_AUROC | 0.9281 | 0.9424 | +0.0143 |
| vial | pixel_F1Score | 0.3605 | 0.4055 | +0.0450 |
| walnuts | image_AUROC | 0.5552 | 0.6241 | +0.0689 |
| walnuts | image_F1Score | 0.7288 | 0.7636 | +0.0348 |
| walnuts | pixel_AUROC | 0.7687 | 0.7760 | +0.0073 |
| walnuts | pixel_F1Score | 0.0422 | 0.0620 | +0.0197 |

## Paper interpretation rule

- If EfficientAD-100 does not improve image AUROC on average, keep EfficientAD-30 as a fixed-budget baseline and cite Stage19 as sensitivity analysis.
- If EfficientAD-100 improves strongly, update the system baseline table and weaken any comparison against EfficientAD.
- Pixel AUROC improvements are auxiliary unless the paper makes pixel-level localization claims.

## Outputs

- `results/stage19_efficientad100_sensitivity/stage19_a_efficientad100_full_four_categories.csv`
- `results/stage19_efficientad100_sensitivity/stage19_a_efficientad100_vs_30_delta.csv`
- `results/stage19_efficientad100_sensitivity/stage19_a_efficientad100_sensitivity_summary.csv`
- `results/stage19_efficientad100_sensitivity/stage19_a_efficientad100_full_four_categories_raw.json`
- `results/stage19_efficientad100_sensitivity/stage19_a_efficientad100_full_four_categories_errors.txt`
