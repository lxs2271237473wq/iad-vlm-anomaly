# Stage 14-C2 WinCLIP fruit_jelly Pilot

## 1. Purpose

This stage runs a one-category WinCLIP pilot on MVTec AD 2 fruit_jelly.

The purpose is to introduce an external vision-language anomaly detection baseline, instead of comparing only with full-image VLM or PatchCore.

## 2. Dataset

- Category: `fruit_jelly`
- Class name for WinCLIP: `fruit jelly`
- Data root: `datasets/MVTec_AD_2_anomalib_all/fruit_jelly_folder`
- Train normal dir: `train/good`
- Test normal dir: `test/good`
- Test abnormal dir: `test/bad`
- Mask dir: `ground_truth/bad`

## 3. Model

- Model: `WinClip`
- k-shot: `0`
- scales: `(2, 3)`

## 4. Status

- Status: `success`
- Timestamp: `2026-06-30T16:49:58`

## 5. Metrics

```json
[
  {
    "image_AUROC": 0.4266666769981384,
    "image_F1Score": 0.8235294222831726,
    "pixel_AUROC": 0.530823826789856,
    "pixel_F1Score": 0.006259575951844454
  }
]
```

## 6. Error

No error.

## 7. Interpretation

The WinCLIP pilot ran successfully on fruit_jelly. Next step is to parse the reported metrics and compare them with PatchCore, our context VLM score, and PatchCore+context fusion on the same category.

## 8. Output Files

- Metrics JSON: `results/stage14_strong_vlm_baselines/stage14_c2_winclip_fruit_jelly_metrics.json`
- Metrics CSV: `results/stage14_strong_vlm_baselines/stage14_c2_winclip_fruit_jelly_metrics.csv`
- Error log: `results/stage14_strong_vlm_baselines/stage14_c2_winclip_fruit_jelly_error.txt`
- Report: `docs/stage14_strong_vlm_baselines/stage14_c2_winclip_fruit_jelly_pilot_report.md`
- Engine output root: `runs/stage14_strong_vlm_baselines/winclip_fruit_jelly_pilot`