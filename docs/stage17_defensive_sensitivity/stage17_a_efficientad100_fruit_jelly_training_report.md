# Stage 15-D EfficientAD Fixed-Budget Baseline

## Purpose

This script runs a robust fixed-budget EfficientAD baseline and writes status to CSV before and after each major phase.

EfficientAD is only a modern non-VLM detector baseline. It is not the proposed method.

## Config

- categories: `['fruit_jelly']`
- max_epochs: `100`
- eval_batch_size: `64`
- num_workers: `16`
- check_val_every_n_epoch: `20`
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
| fruit_jelly | success | 15268.667 | 16.217 | 0.8266666531562805 | 0.84375 | 0.8424254655838013 | 0.5561066269874573 | `` |

## Aggregate

- successful_categories: `1`
- mean_image_AUROC: `0.8267`
- mean_image_F1Score: `0.8438`
- mean_pixel_AUROC: `0.8424`
- mean_pixel_F1Score: `0.5561`

## Note

Anomalib EfficientAD forces train_batch_size=1 and performs validation quantile/metric computation. RTX 4090 utilization can therefore remain low for this baseline.