# Stage 15-D EfficientAD Fixed-Budget Baseline

## 1. Purpose

This stage runs EfficientAD as a modern non-VLM detector baseline under a fixed 30-epoch budget on the four primary MVTec AD 2 categories.

This is not the proposed method. It is used only as a detector baseline for later strong-baseline comparison.

## 2. Fixed Protocol

- method: `EfficientAD-30 fixed-budget`
- categories: `fruit_jelly`, `sheet_metal`, `vial`, `walnuts`
- max_epochs: `30`
- train_batch_size: `1`
- eval_batch_size: `64`
- num_workers: `16`
- precision: `16-mixed`
- check_val_every_n_epoch: `10`
- model_size: `small`

## 3. Results

| Category | Status | Fit sec | Test sec | Image AUROC | Image F1 | Pixel AUROC | Pixel F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| fruit_jelly | success | 4870.582 | 17.338 | 0.8433 | 0.8571 | 0.7894 | 0.5395 |
| sheet_metal | success | 2557.149 | 29.355 | 0.7333 | 0.8842 | 0.7510 | 0.2995 |
| vial | success | 5116.335 | 23.059 | 0.9099 | 0.8991 | 0.9281 | 0.3605 |
| walnuts | success | 8219.740 | 57.985 | 0.5552 | 0.7288 | 0.7687 | 0.0422 |

## 4. Aggregate

- successful_categories: `4`
- mean_image_AUROC: `0.7604`
- mean_image_F1Score: `0.8423`
- mean_pixel_AUROC: `0.8093`
- mean_pixel_F1Score: `0.3104`

## 5. Interpretation

EfficientAD-30 provides a stronger modern non-VLM detector baseline than the earlier fruit_jelly-only 20-epoch pilot.

However, this result should not be described as an official or full-budget EfficientAD baseline. It should be reported as `EfficientAD-30 fixed-budget`.

A 100-epoch sensitivity check on `fruit_jelly` is still needed to determine whether the 30-epoch budget substantially underestimates EfficientAD.

## 6. Runtime Note

Anomalib EfficientAD forces `train_batch_size=1` and performs validation quantile/metric computation. RTX 4090 utilization can therefore remain low for this baseline.
