# Anomaly-map Calibration Analysis

## 1. Purpose

This document analyzes Stage 5.1: lightweight anomaly-map calibration.

The tested strategies are:

1. fixed threshold
2. percentile threshold
3. mean-std adaptive threshold
4. optional largest-component filtering

The goal is to check whether simple calibration of PatchCore anomaly maps can improve pixel-level localization.

## 2. Summary Results

| Category | Best Method | Best Value | Keep Largest | Best IoU | Best F1 | Mean Pred Area | Mean Threshold |
|---|---|---:|---|---:|---:|---:|---:|
| grid | fixed | 0.8000 | False | 0.3298 | 0.4758 | 764.5 | 0.8000 |
| screw | fixed | 0.8500 | False | 0.2806 | 0.4024 | 491.4 | 0.8500 |
| leather | fixed | 0.9000 | False | 0.3456 | 0.5036 | 348.2 | 0.9000 |
| wood | fixed | 0.7000 | False | 0.3607 | 0.5004 | 3549.4 | 0.7000 |

## 3. Mean Result

| Metric | Value |
|---|---:|
| Mean Best IoU | 0.3292 |
| Mean Best F1 | 0.4706 |
| Mean Predicted Area | 1288.4 |
| Mean Threshold | 0.8125 |

## 4. Observations

1. The best strategy for all four categories is still fixed thresholding.

2. Percentile thresholding and mean-std adaptive thresholding do not outperform the fixed thresholds found in the earlier threshold diagnosis.

3. This means that simple per-image threshold calibration is not enough to solve the weak pixel-level localization problem.

4. Compared with previous stages, Stage 5.1 does not produce a stronger result:

| Stage | Mean F1 |
|---|---:|
| Stage 4.1 Region Score | 0.4712 |
| Stage 4.2 Enhanced Region Score | 0.4746 |
| Stage 5.1 Anomaly-map Calibration | 0.4706 |

## 5. Conclusion

Stage 5.1 should be treated as a negative or weak diagnostic result.

```text
Simple anomaly-map calibration by thresholding is not sufficient.
The project should move from rule-based calibration to a trainable calibration module.
```

## 6. Next Direction

Recommended next stage:

```text
Stage 5.2: Trainable lightweight anomaly-map calibration module
```

The next method should learn how to refine anomaly maps instead of only choosing thresholds.

Candidate design:

- input: PatchCore anomaly map
- optional input: normalized image intensity or edge map
- output: calibrated anomaly probability map
- supervision: MVTec ground-truth masks
- model: very small CNN / UNet-like calibration head
- evaluation: pixel AUROC, pixel F1, mask quality on weak categories
