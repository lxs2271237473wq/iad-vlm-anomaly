# Conservative Residual Anomaly-map Calibration Analysis

## 1. Purpose

This document analyzes Stage 5.3: conservative residual anomaly-map calibration.

The tested pipeline is:

```text
PatchCore anomaly map -> conservative residual calibration head -> calibrated anomaly map -> thresholded mask
```

Unlike the naive Tiny CNN calibration in Stage 5.2, this version does not replace the original anomaly map. It only learns a bounded residual:

```text
calibrated_map = raw_anomaly_map + small_residual
```

The design includes:

- normal images with all-zero masks
- residual constraint
- identity regularization
- normal false-positive penalty
- abnormal-mask F1 evaluation
- normal false-positive evaluation

## 2. Summary Results

| Category | Samples | Train Abnormal | Train Normal | Eval Abnormal | Eval Normal | Raw F1 | Calibrated F1 | Delta F1 | Raw Normal FP | Calibrated Normal FP | Normal FP Delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| grid | 40 | 14 | 6 | 14 | 6 | 0.5182 | 0.5105 | -0.0076 | 0.0282 | 0.0226 | -0.0056 |
| screw | 81 | 30 | 10 | 30 | 11 | 0.4146 | 0.4206 | 0.0060 | 0.0177 | 0.0142 | -0.0035 |
| leather | 62 | 23 | 8 | 23 | 8 | 0.5156 | 0.5194 | 0.0038 | 0.0058 | 0.0062 | 0.0004 |
| wood | 40 | 15 | 5 | 15 | 5 | 0.6019 | 0.6095 | 0.0076 | 0.0486 | 0.0489 | 0.0003 |

## 3. Mean Result

| Metric | Value |
|---|---:|
| Mean Raw Abnormal IoU | 0.3655 |
| Mean Raw Abnormal F1 | 0.5126 |
| Mean Calibrated Abnormal IoU | 0.3682 |
| Mean Calibrated Abnormal F1 | 0.5150 |
| Mean Delta Abnormal F1 | 0.0025 |
| Mean Raw Normal FP Ratio | 0.0251 |
| Mean Calibrated Normal FP Ratio | 0.0230 |
| Mean Delta Normal FP Ratio | -0.0021 |

## 4. Observations

1. Conservative residual calibration avoids the severe collapse observed in Stage 5.2.

2. The average abnormal F1 improves slightly, from raw PatchCore maps to calibrated maps.

3. The mean normal false-positive ratio decreases slightly, which means the conservative loss helps suppress normal-background activation.

4. The gain is still very small. The module is stable but not strong enough to serve as the main method.

5. The strongest positive categories are `screw` and `wood`; `grid` slightly degrades.

## 5. Comparison with Stage 5.2

| Stage | Mean Raw F1 | Mean Calibrated F1 | Mean Delta F1 | Judgment |
|---|---:|---:|---:|---|
| Stage 5.2 Naive Tiny CNN Calibration | 0.5139 | 0.3956 | -0.1183 | failed |
| Stage 5.3 Conservative Residual Calibration | 0.5126 | 0.5150 | 0.0025 | stable but weak |

## 6. Conclusion

Stage 5.3 should be treated as a stable but weak positive result.

```text
Conservative residual calibration is safer than naive trainable calibration,
but the current improvement is too small to become the main module.
```

## 7. Next Direction

The next step should run a small hyperparameter sweep to check whether the conservative residual module has unused potential.

Recommended next stage:

```text
Stage 5.4: Conservative residual calibration hyperparameter sweep
```

The sweep should vary:

- max_delta
- identity_weight
- residual_weight
- normal_fp_weight
- learning rate

If Stage 5.4 still gives only near-zero gains, the project should stop the trainable calibration line and move to method consolidation.
