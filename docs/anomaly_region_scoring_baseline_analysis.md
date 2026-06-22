# Anomaly-map Guided Region Scoring Baseline Analysis

## 1. Purpose

This document analyzes whether anomaly-map-derived region features can improve candidate-region ranking over the default PatchCore top-1 connected component.

The tested pipeline is:

```text
PatchCore anomaly map -> connected components -> region features -> weighted region scoring -> candidate re-ranking
```

The tested categories are:

- grid
- screw
- leather
- wood

## 2. Evaluation Protocol

For each category, abnormal images are split into two subsets:

- tuning subset: used to select the best region-scoring weights
- evaluation subset: used to compare PatchCore top-1 and the learned region score

This avoids directly reporting the best score on the same images used for selecting weights.

## 3. Summary Results

| Category | Eval Images | PatchCore Top-1 IoU | PatchCore Top-1 F1 | Region Score IoU | Region Score F1 | Delta F1 |
|---|---:|---:|---:|---:|---:|---:|
| grid | 14 | 0.3344 | 0.4779 | 0.3298 | 0.4720 | -0.0059 |
| screw | 30 | 0.2741 | 0.3828 | 0.2916 | 0.4105 | 0.0278 |
| leather | 23 | 0.3583 | 0.5170 | 0.3583 | 0.5170 | 0.0000 |
| wood | 15 | 0.3379 | 0.4731 | 0.3455 | 0.4855 | 0.0123 |
| MEAN | 82 | 0.3262 | 0.4627 | 0.3313 | 0.4712 | 0.0085 |

## 4. Mean Result

| Metric | Value |
|---|---:|
| Mean PatchCore Top-1 IoU | 0.3262 |
| Mean PatchCore Top-1 F1 | 0.4627 |
| Mean Region Score IoU | 0.3313 |
| Mean Region Score F1 | 0.4712 |
| Mean Delta F1 | 0.0085 |

## 5. Selected Weight Pattern

| Category | Mean-score W | Max-score W | Gap W | Area W | Compact W | Aspect W | Semantic W |
|---|---:|---:|---:|---:|---:|---:|---:|
| grid | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| screw | 0.3077 | 0.0000 | 0.0000 | 0.0000 | 0.3077 | 0.0769 | 0.3077 |
| leather | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| wood | 0.5714 | 0.0000 | 0.0000 | 0.2857 | 0.0000 | 0.0000 | 0.1429 |
| MEAN | 0.7198 | 0.0000 | 0.0000 | 0.0714 | 0.0769 | 0.0192 | 0.1126 |

## 6. Observations

1. Region scoring provides a weak but positive average improvement over PatchCore top-1 candidate selection.

2. The strongest gain appears on `screw`, which suggests that region-level scoring is useful for small structured object defects.

3. `wood` also benefits moderately, while `grid` slightly degrades and `leather` remains unchanged.

4. The selected weights show that anomaly-map features are more reliable than the semantic feature. The average semantic weight is small, so CLIP-based semantic scoring should remain an auxiliary term.

5. The improvement is still modest. This means the current scoring rule is useful as a diagnostic baseline, but it needs stronger region features or a better scoring formulation before becoming the main method.

## 7. Conclusion

The Stage 4.1 result is a weak positive result:

```text
Anomaly-map guided region scoring is more promising than direct SAM2 refinement or pure CLIP semantic re-ranking,
but the current hand-designed score still provides only limited improvement.
```

Therefore, the next stage should improve the region scoring formulation itself.

## 8. Next Stage

Recommended next stage:

```text
Stage 4.2: Enhanced anomaly-region scoring with local contrast and multi-threshold stability
```

The next module should add stronger region features:

- local anomaly contrast
- inside/outside score difference
- multi-threshold region stability
- region compactness and aspect constraints
- optional weak semantic prior
