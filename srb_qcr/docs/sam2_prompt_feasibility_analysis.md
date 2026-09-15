# SAM2 Prompt Feasibility Analysis

## 1. Purpose

This document analyzes whether SAM2 can directly refine PatchCore candidate boxes into better pixel-level defect masks.

The tested pipeline is:

```text
PatchCore anomaly map -> candidate box -> SAM2 box prompt -> SAM2 mask
```

The tested categories are weak PatchCore localization categories:

- grid
- screw
- leather
- wood

## 2. Summary

| Category | Images | PatchCore Top-1 IoU | PatchCore Top-1 F1 | SAM2 IoU | SAM2 F1 | Delta F1 |
|---|---:|---:|---:|---:|---:|---:|
| grid | 28 | 0.3149 | 0.4559 | 0.0756 | 0.1280 | -0.3279 |
| screw | 60 | 0.2736 | 0.3841 | 0.0357 | 0.0611 | -0.3229 |
| leather | 46 | 0.3444 | 0.5024 | 0.0278 | 0.0469 | -0.4555 |
| wood | 30 | 0.2679 | 0.3925 | 0.2504 | 0.3614 | -0.0311 |

## 3. Mean Result

| Metric | Value |
|---|---:|
| Mean PatchCore Top-1 F1 | 0.4337 |
| Mean SAM2 Box Prompt F1 | 0.1494 |
| Mean Delta F1 | -0.2844 |

## 4. Observations

1. Naive SAM2 box prompting does not improve PatchCore localization on these weak categories.

2. In most categories, SAM2 box-prompt masks are worse than PatchCore candidate regions. This suggests that SAM2 tends to segment visually coherent object or texture regions, not necessarily the abnormal defect region.

3. The result is especially poor on `leather` and `screw`, where the defect regions are either fine-grained, thin, local, or texture-like. A box prompt alone does not provide enough defect-specific guidance.

4. `wood` is the only category where SAM2 is relatively close to PatchCore, but it still does not clearly improve the baseline.

## 5. Conclusion

The current SAM2 test should be treated as a negative feasibility result:

```text
PatchCore candidate box + naive SAM2 box prompt is not sufficient for industrial defect mask refinement.
```

Therefore, SAM2 should not be used as a simple plug-in refinement module at this stage.

## 6. Next Direction

The next stage should make SAM2 anomaly-aware rather than using the raw box prompt directly.

Recommended next experiments:

1. SAM2 multimask selection using PatchCore anomaly-map consistency.
2. Box + positive point prompt from the anomaly peak.
3. Negative point prompt from low-anomaly background regions.
4. Candidate ranking based on anomaly score, component size, and semantic defect prior.
5. If SAM2 still fails, shift the main method toward anomaly-map enhancement instead of mask refinement.

Recommended next stage:

```text
Stage 2.4: Anomaly-aware SAM2 prompt and mask selection
```
