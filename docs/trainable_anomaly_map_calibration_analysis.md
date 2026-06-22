# Trainable Anomaly-map Calibration Analysis

## 1. Purpose

This document analyzes Stage 5.2: trainable lightweight anomaly-map calibration.

The tested pipeline is:

```text
PatchCore anomaly map -> Tiny CNN calibration head -> calibrated anomaly probability map -> thresholded mask
```

The tested categories are:

- grid
- screw
- leather
- wood

This is a supervised diagnostic experiment using MVTec AD ground-truth masks. It should not be directly compared as a fair unsupervised method against PatchCore.

## 2. Summary Results

| Category | Samples | Train | Eval | Raw Threshold | Calibrated Threshold | Raw IoU | Raw F1 | Calibrated IoU | Calibrated F1 | Delta F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| grid | 28 | 14 | 14 | 0.8000 | 0.5000 | 0.3686 | 0.5182 | 0.3897 | 0.5383 | 0.0202 |
| screw | 60 | 30 | 30 | 0.8500 | 0.4500 | 0.2899 | 0.4146 | 0.0492 | 0.0902 | -0.3244 |
| leather | 46 | 23 | 23 | 0.9000 | 0.8000 | 0.3571 | 0.5156 | 0.2617 | 0.4077 | -0.1079 |
| wood | 30 | 15 | 15 | 0.7000 | 0.5500 | 0.4558 | 0.6073 | 0.3926 | 0.5461 | -0.0612 |

## 3. Mean Result

| Metric | Value |
|---|---:|
| Mean Raw IoU | 0.3679 |
| Mean Raw F1 | 0.5139 |
| Mean Calibrated IoU | 0.2733 |
| Mean Calibrated F1 | 0.3956 |
| Mean Delta F1 | -0.1183 |

## 4. Observations

1. The trainable calibration head improves `grid`, but it significantly degrades `screw`, `leather`, and `wood`.

2. The mean F1 decreases strongly after calibration. This shows that the current Tiny CNN calibration head is not robust.

3. The result is especially poor on `screw`, where the calibrated F1 is much lower than the raw anomaly-map F1. This indicates that the model destroys useful fine-grained anomaly responses.

4. The current training protocol uses a small number of abnormal samples per category. This makes the calibration head prone to overfitting.

5. The current version does not explicitly include normal images with all-zero masks. This weakens the model's ability to distinguish real defect regions from normal background.

## 5. Conclusion

Stage 5.2 should be treated as a negative diagnostic result.

```text
A naive Tiny CNN trained directly on PatchCore anomaly maps is not sufficient.
The trainable calibration direction remains possible, but the current design is unstable.
```

## 6. Next Direction

The next version should not simply increase model size. It should fix the training protocol and make the calibration conservative.

Recommended next stage:

```text
Stage 5.3: Conservative residual anomaly-map calibration
```

Key changes:

1. Include normal images with all-zero masks.
2. Use a residual calibration form instead of replacing the original anomaly map.
3. Add a constraint that discourages large changes from the raw PatchCore anomaly map.
4. Use a validation split for early stopping.
5. Report both raw map F1 and calibrated map F1 on the same held-out split.
