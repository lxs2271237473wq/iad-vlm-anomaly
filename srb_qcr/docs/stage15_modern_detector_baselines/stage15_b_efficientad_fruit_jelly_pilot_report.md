# Stage 15-B EfficientAD fruit_jelly Pilot

## 1. Purpose

This stage runs a one-category EfficientAD pilot on MVTec AD 2 fruit_jelly.

The purpose is to introduce a modern non-VLM anomaly detector baseline, instead of relying only on PatchCore.

## 2. Dataset

- Category: `fruit_jelly`
- Data root: `datasets/MVTec_AD_2_anomalib_all/fruit_jelly_folder`
- Train normal dir: `train/good`
- Test normal dir: `test/good`
- Test abnormal dir: `test/bad`
- Mask dir: `ground_truth/bad`

## 3. Model and Training Config

- model: `EfficientAd`
- category: `fruit_jelly`
- imagenet_dir: `/root/private_data/iad-vlm-anomaly/datasets/imagenette`
- model_size: `small`
- max_epochs: `20`
- train_batch_size: `1`
- eval_batch_size: `8`
- num_workers: `0`
- lr: `0.0001`
- weight_decay: `1e-05`

## 4. Status

- Status: `success`
- Timestamp: `2026-06-30T23:38:55`

## 5. Metrics

```json
[
  {
    "image_AUROC": 0.6566667556762695,
    "image_F1Score": 0.774193525314331,
    "pixel_AUROC": 0.7702546119689941,
    "pixel_F1Score": 0.46490851044654846
  }
]
```

## 6. Error

No error.

## 7. Interpretation

EfficientAD successfully completed fit and test on fruit_jelly. The next step is to compare it against PatchCore, WinCLIP, context-aware VLM, and PatchCore+context fusion on the same category.

## 8. Output Files

- Metrics JSON: `results/stage15_modern_detector_baselines/stage15_b_efficientad_fruit_jelly_metrics.json`
- Metrics CSV: `results/stage15_modern_detector_baselines/stage15_b_efficientad_fruit_jelly_metrics.csv`
- Error log: `results/stage15_modern_detector_baselines/stage15_b_efficientad_fruit_jelly_error.txt`
- Report: `docs/stage15_modern_detector_baselines/stage15_b_efficientad_fruit_jelly_pilot_report.md`
- Engine output root: `runs/stage15_modern_detector_baselines/efficientad_fruit_jelly_pilot`