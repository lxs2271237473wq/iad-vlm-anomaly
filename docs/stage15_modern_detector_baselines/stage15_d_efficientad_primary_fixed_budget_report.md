# Stage 15-D EfficientAD Fixed-Budget Baseline

## Purpose

This script runs a robust fixed-budget EfficientAD baseline and writes status to CSV before and after each major phase.

EfficientAD is only a modern non-VLM detector baseline. It is not the proposed method.

## Config

- categories: `['fruit_jelly', 'sheet_metal', 'vial', 'walnuts']`
- max_epochs: `30`
- eval_batch_size: `64`
- num_workers: `16`
- check_val_every_n_epoch: `10`
- precision: `16-mixed`
- model_size: `small`
- lr: `0.0001`
- weight_decay: `1e-05`
- enable_progress_bar: `False`
- reset_outputs: `True`
- train_batch_size: `1`

## Results

| Category | Status | Fit sec | Test sec | Image AUROC | Image F1 | Pixel AUROC | Pixel F1 | Error |
|---|---|---:|---:|---:|---:|---:|---:|---|
| fruit_jelly | success | 4870.582 | 17.338 | 0.8433333039283752 | 0.8571428656578064 | 0.7893707156181335 | 0.5394811034202576 | `` |
| sheet_metal | success | 2557.149 | 29.355 | 0.7333333492279053 | 0.8842105269432068 | 0.7509881854057312 | 0.29947036504745483 | `` |
| vial | success | 5116.335 | 23.059 | 0.9098532199859619 | 0.8990825414657593 | 0.9281267523765564 | 0.3604857921600342 | `` |
| walnuts | success | 8219.74 | 57.985 | 0.5551851987838745 | 0.7288135886192322 | 0.7686794996261597 | 0.04223956540226936 | `` |

## Aggregate

- successful_categories: `4`
- mean_image_AUROC: `0.7604`
- mean_image_F1Score: `0.8423`
- mean_pixel_AUROC: `0.8093`
- mean_pixel_F1Score: `0.3104`

## Note

Anomalib EfficientAD forces train_batch_size=1 and performs validation quantile/metric computation. RTX 4090 utilization can therefore remain low for this baseline.