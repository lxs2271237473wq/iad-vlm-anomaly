# Anomaly-aware SAM2 Selection Analysis

## 1. Purpose

This document analyzes whether anomaly-aware SAM2 mask selection can improve PatchCore candidate-region localization.

The tested variants are:

1. PatchCore connected component mask.
2. SAM2 box prompt with SAM2's own score selection.
3. SAM2 box prompt with anomaly-aware mask selection.
4. SAM2 box + positive/negative point prompts with anomaly-aware mask selection.

The tested categories are:

- grid
- screw
- leather
- wood

## 2. Summary Results

| Category | Images | PatchCore F1 | SAM2 Score F1 | SAM2 AA-box F1 | SAM2 AA-box-point F1 | Delta AA-box F1 | Delta AA-box-point F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| grid | 28 | 0.4588 | 0.4340 | 0.4353 | 0.4382 | -0.0235 | -0.0206 |
| screw | 60 | 0.3963 | 0.3774 | 0.3780 | 0.3803 | -0.0183 | -0.0160 |
| leather | 46 | 0.5047 | 0.5076 | 0.5088 | 0.5021 | 0.0041 | -0.0027 |
| wood | 30 | 0.4030 | 0.3574 | 0.3608 | 0.3586 | -0.0422 | -0.0444 |

## 3. Mean Results

| Method | Mean Pixel F1 |
|---|---:|
| PatchCore component mask | 0.4407 |
| SAM2 box prompt + SAM score selection | 0.4191 |
| SAM2 box prompt + anomaly-aware selection | 0.4207 |
| SAM2 box + point prompt + anomaly-aware selection | 0.4198 |

## 4. Observations

1. Anomaly-aware selection improves over the earlier naive SAM2 box prompt, but it still does not consistently outperform the PatchCore connected component mask.

2. `leather` is the only category where SAM2 box prompt with anomaly-aware selection slightly improves F1 over PatchCore.

3. `grid`, `screw`, and `wood` still degrade after SAM2 refinement, which indicates that SAM2 tends to segment visually coherent regions rather than defect-specific regions.

4. Positive and negative point prompts do not provide a stable gain in the current setting. In some cases, they even reduce performance compared with box-only anomaly-aware selection.

## 5. Conclusion

The current result should be treated as a weak/negative feasibility result for SAM2-based refinement.

```text
SAM2 can be used for visualization and exploratory refinement,
but the current SAM2 prompt strategy is not strong enough to become the main improvement module.
```

Therefore, the next main direction should shift from pure SAM2 mask refinement to anomaly-map enhancement and semantic-guided candidate ranking.

## 6. Next Stage

Recommended next stage:

```text
Stage 3.1: Semantic-guided anomaly candidate ranking
```

The next module should use defect semantics and manufacturing-aware knowledge to rank or reweight candidate regions, rather than relying only on SAM2 segmentation.

Possible inputs:

- PatchCore anomaly map.
- Candidate box, center point, and area.
- Defect type semantics such as scratch, crack, cut, color, contamination, deformation.
- Manufacturing-aware defect knowledge from `knowledge/defect_knowledge_generic.json`.
