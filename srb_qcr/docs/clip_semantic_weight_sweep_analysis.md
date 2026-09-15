# CLIP Semantic Candidate Re-ranking Weight Sweep Analysis

## 1. Purpose

This document analyzes whether manufacturing-aware semantic prompts and CLIP-based crop-text similarity can improve PatchCore candidate-region ranking.

The tested pipeline is:

```text
PatchCore candidate regions -> crop CLIP scoring -> semantic/anomaly weighted re-ranking
```

The tested categories are:

- grid
- screw
- leather
- wood

## 2. Summary Results

| Category | PatchCore Top-1 F1 | Best Rerank F1 | Delta F1 | Best Semantic Weight | Best Anomaly Weight | Best Max-score Weight | Best Area Weight |
|---|---:|---:|---:|---:|---:|---:|---:|
| grid | 0.4559 | 0.4530 | -0.0029 | 0.0000 | 1.0000 | 0.0000 | 0.0000 |
| leather | 0.5024 | 0.5024 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 |
| screw | 0.3841 | 0.4095 | 0.0254 | 0.8000 | 0.1600 | 0.0200 | 0.0200 |
| wood | 0.3925 | 0.3930 | 0.0006 | 0.3000 | 0.7000 | 0.0000 | 0.0000 |
| MEAN | 0.4337 | 0.4395 | 0.0058 | 0.2750 | 0.7150 | 0.0050 | 0.0050 |

## 3. Mean Result

| Metric | Value |
|---|---:|
| Mean PatchCore Top-1 F1 | 0.4337 |
| Mean Best Rerank F1 | 0.4395 |
| Mean Delta F1 | 0.0058 |
| Mean Best Semantic Weight | 0.2750 |

## 4. Observations

1. CLIP semantic re-ranking provides only a weak average improvement over PatchCore candidate top-1 ranking.

2. The main positive gain comes from `screw`, where the best semantic weight is high. This suggests that semantic prompts are more useful for structured object defects than for texture-like defects.

3. For `grid` and `leather`, the best semantic weight is zero, which means the CLIP semantic term is not helpful for these categories under the current prompt bank.

4. For `wood`, the improvement is very small. Natural texture variation likely makes defect semantics difficult for generic CLIP prompts.

## 5. Conclusion

The current CLIP semantic candidate re-ranking result should be treated as a weak positive result, not a strong main-module result.

```text
Manufacturing-aware prompts are useful as auxiliary evidence,
but the current CLIP-based semantic score is not strong enough to drive candidate ranking by itself.
```

Therefore, the next main direction should focus on improving the anomaly map or the region scoring mechanism itself, while keeping semantic prompts as an auxiliary signal.

## 6. Next Stage

Recommended next stage:

```text
Stage 4.1: Anomaly-map enhancement and region scoring baseline
```

The next module should combine:

- PatchCore anomaly score.
- Candidate area and compactness.
- Local anomaly contrast.
- Optional weak semantic score.
- Defect-type-aware priors from the manufacturing knowledge base.
