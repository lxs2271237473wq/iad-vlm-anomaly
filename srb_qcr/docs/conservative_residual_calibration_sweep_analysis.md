# Conservative Residual Calibration Hyperparameter Sweep Analysis

## 1. Purpose

This document analyzes Stage 5.4: conservative residual anomaly-map calibration hyperparameter sweep.

The purpose is to determine whether the conservative residual calibration module still has meaningful improvement potential after Stage 5.3.

The tested module follows:

```text
calibrated_map = raw_anomaly_map + bounded_residual
```

The sweep varies:

- max_delta
- learning rate
- identity regularization weight
- residual regularization weight
- normal false-positive penalty
- dice loss weight

The tested categories are:

- grid
- screw
- leather
- wood

## 2. Sweep Summary

| Config | Raw F1 | Calibrated F1 | Delta F1 | Raw Normal FP | Calibrated Normal FP | Normal FP Delta | max_delta | lr | identity_w | residual_w | normal_fp_w | dice_w |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cfg01_baseline | 0.5126 | 0.5146 | 0.0020 | 0.0251 | 0.0269 | 0.0018 | 0.10 | 0.00050 | 2.0 | 1.0 | 1.0 | 1.0 |
| cfg02_smaller_delta | 0.5126 | 0.5052 | -0.0074 | 0.0251 | 0.0311 | 0.0060 | 0.05 | 0.00050 | 2.0 | 1.0 | 1.0 | 1.0 |
| cfg03_larger_delta | 0.5126 | 0.5184 | 0.0058 | 0.0251 | 0.0256 | 0.0005 | 0.15 | 0.00050 | 2.0 | 1.0 | 1.0 | 1.0 |
| cfg04_stronger_identity | 0.5126 | 0.5140 | 0.0014 | 0.0251 | 0.0256 | 0.0005 | 0.10 | 0.00050 | 5.0 | 2.0 | 1.0 | 1.0 |
| cfg05_weaker_identity | 0.5126 | 0.5136 | 0.0011 | 0.0251 | 0.0278 | 0.0028 | 0.10 | 0.00050 | 1.0 | 0.5 | 1.0 | 1.0 |
| cfg06_stronger_normal_fp | 0.5126 | 0.5157 | 0.0031 | 0.0251 | 0.0250 | -0.0001 | 0.10 | 0.00050 | 2.0 | 1.0 | 3.0 | 1.0 |
| cfg07_lower_lr | 0.5126 | 0.5146 | 0.0020 | 0.0251 | 0.0327 | 0.0076 | 0.10 | 0.00020 | 2.0 | 1.0 | 1.0 | 1.0 |
| cfg08_weaker_dice | 0.5126 | 0.5127 | 0.0002 | 0.0251 | 0.0273 | 0.0022 | 0.10 | 0.00050 | 2.0 | 1.0 | 1.0 | 0.5 |

## 3. Best Configurations

### 3.1 Best by abnormal F1 gain

| Metric | Value |
|---|---:|
| Config | cfg03_larger_delta |
| Mean Raw Abnormal F1 | 0.5126 |
| Mean Calibrated Abnormal F1 | 0.5184 |
| Mean Delta Abnormal F1 | 0.0058 |
| Mean Delta Normal FP Ratio | 0.0005 |

### 3.2 Safest config by normal false-positive control

| Metric | Value |
|---|---:|
| Config | cfg06_stronger_normal_fp |
| Mean Raw Abnormal F1 | 0.5126 |
| Mean Calibrated Abnormal F1 | 0.5157 |
| Mean Delta Abnormal F1 | 0.0031 |
| Mean Delta Normal FP Ratio | -0.0001 |

## 4. Observations

1. The best F1 configuration is `cfg03_larger_delta`, but the average F1 gain is still very small.

2. The safest configuration is `cfg06_stronger_normal_fp`, which slightly improves abnormal F1 while slightly reducing normal false positives.

3. Smaller residual range does not help. `cfg02_smaller_delta` reduces F1 and increases normal false positives.

4. Stronger identity regularization makes the module safer but does not produce meaningful F1 gain.

5. Lower learning rate and weaker dice loss do not provide useful improvement.

## 5. Conclusion

Stage 5.4 shows that conservative residual anomaly-map calibration has limited remaining potential.

```text
The residual calibration module is stable, but the best gain remains below a useful threshold.
It should not be used as the main method.
```

Final judgment:

```text
Stop the trainable anomaly-map calibration line.
Use it only as an auxiliary diagnostic / negative-result branch.
```

## 6. Next Direction

The project should now move from exploratory branches to method consolidation.

Recommended next stage:

```text
Stage 6.1: Method consolidation and paper-oriented framework design
```

The consolidated method should keep only the parts with defensible value:

- PatchCore as the strong anomaly-map baseline.
- Manufacturing-aware prompt bank as semantic prior / explanation component.
- CLIP semantic scoring as weak auxiliary evidence, not the main driver.
- Region scoring as diagnostic evidence, not the final core module.
- SAM2 and trainable calibration as negative or motivation experiments.
