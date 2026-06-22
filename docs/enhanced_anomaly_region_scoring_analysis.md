# Enhanced Anomaly-region Scoring Analysis

## 1. Purpose

This document analyzes Stage 4.2: enhanced anomaly-region scoring.

The tested pipeline is:

```text
PatchCore anomaly map -> connected components -> enhanced region features -> weighted region score -> candidate re-ranking
```

Compared with Stage 4.1, this stage adds:

- local anomaly contrast
- peak contrast
- multi-threshold stability
- compactness and aspect-ratio constraints
- area penalty

The tested categories are:

- grid
- screw
- leather
- wood

## 2. Summary Results

| Category | Eval Images | PatchCore Top-1 IoU | PatchCore Top-1 F1 | Enhanced IoU | Enhanced F1 | Delta F1 |
|---|---:|---:|---:|---:|---:|---:|
| grid | 14 | 0.3378 | 0.4819 | 0.3378 | 0.4819 | 0.0000 |
| screw | 30 | 0.2829 | 0.3987 | 0.2905 | 0.4111 | 0.0123 |
| leather | 23 | 0.3609 | 0.5191 | 0.3609 | 0.5191 | 0.0000 |
| wood | 15 | 0.3453 | 0.4862 | 0.3453 | 0.4862 | 0.0000 |
| MEAN | 82 | 0.3318 | 0.4715 | 0.3336 | 0.4746 | 0.0031 |

## 3. Mean Result

| Metric | Value |
|---|---:|
| Mean PatchCore Top-1 IoU | 0.3318 |
| Mean PatchCore Top-1 F1 | 0.4715 |
| Mean Enhanced Region IoU | 0.3336 |
| Mean Enhanced Region F1 | 0.4746 |
| Mean Delta F1 | 0.0031 |

## 4. Selected Weight Pattern

| Category | Mean W | Contrast W | Peak W | Stability W | Compact W | Aspect W | Area Penalty W |
|---|---:|---:|---:|---:|---:|---:|---:|
| grid | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| screw | 0.6667 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.3333 |
| leather | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| wood | 0.6667 | 0.0000 | 0.0000 | 0.0000 | 0.3333 | 0.0000 | 0.0000 |
| MEAN | 0.8333 | 0.0000 | 0.0000 | 0.0000 | 0.0833 | 0.0000 | 0.0833 |

## 5. Observations

1. Enhanced anomaly-region scoring gives only a weak positive mean improvement.

2. The gain mainly comes from `screw`; `grid`, `leather`, and `wood` remain unchanged.

3. Compared with Stage 4.1, the improvement is smaller. Stage 4.1 achieved a stronger mean gain than this enhanced scoring version.

4. The selected weights are dominated by mean anomaly score. Contrast, stability, and shape features are not consistently selected as useful factors.

5. This suggests that simply adding more hand-designed region features does not reliably improve candidate ranking.

## 6. Conclusion

Stage 4.2 should be treated as a weak positive but non-promising result.

```text
Enhanced hand-crafted anomaly-region scoring does not provide enough improvement to become the main module.
```

The project should stop expanding this rule-based scoring line and move toward a more principled method.

## 7. Next Direction

The next stage should move from hand-designed region scoring to a trainable or calibration-based module.

Recommended next stage:

```text
Stage 5.1: Lightweight anomaly-map calibration module
```

Instead of only re-ranking connected components, the next method should learn or calibrate the anomaly map itself.

Possible directions:

- learnable pixel-level calibration using normal validation statistics
- category-adaptive anomaly score normalization
- multi-scale anomaly-map fusion
- small trainable refinement head over PatchCore anomaly maps
- semantic prior as weak auxiliary input rather than the main scoring signal
