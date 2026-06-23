# Manufacturing-aware Defect Explanation Analysis

## 1. Purpose

This document analyzes Stage 6.5: manufacturing-aware defect explanation generation.

Stage 6.5 does not modify the detection or classification model. Instead, it uses the Stage 6.4 prediction results and combines them with the manufacturing knowledge base to produce structured explanations.

The explanation pipeline is:

```text
PatchCore candidate crop -> refined visual prompt defect prediction -> manufacturing knowledge base -> structured explanation
```

The generated explanation contains:

- predicted defect type
- top-2 candidate defect types
- PatchCore candidate region
- anomaly score of the candidate region
- defect family
- visual evidence
- related manufacturing processes
- possible manufacturing causes
- inspection focus

## 2. Fair-comparison Note

Stage 6.5 should be compared under the same data and prediction setting as Stage 6.4:

```text
strategy = generic_label
eval_mode = crop_topk_only
candidate source = full-test PatchCore candidate regions
evaluation set = 327 abnormal images with candidate crops
```

Therefore, the classification metrics are inherited from the same Stage 6.4 prediction setting. Stage 6.5 adds explanation capability, not classification accuracy improvement.

## 3. Stage 6.4 Prediction Setting Used by Stage 6.5

| Metric | Value |
|---|---:|
| Images Used | 327 / 328 |
| Coverage Ratio | 0.9970 |
| Top-1 Accuracy | 0.3398 |
| Top-2 Accuracy | 0.5045 |
| Macro-F1 | 0.2209 |

## 4. Explanation Summary

| Category | Reports | Predicted Defect Types | Defect Families | Mean Confidence Margin | Top-1 Accuracy | Top-2 Accuracy |
|---|---:|---:|---:|---:|---:|---:|
| grid | 56 | 4 | 3 | 0.0043 | 0.2321 | 0.3750 |
| leather | 92 | 5 | 4 | 0.0026 | 0.2826 | 0.4130 |
| screw | 119 | 5 | 2 | 0.0058 | 0.3277 | 0.5798 |
| wood | 60 | 3 | 3 | 0.0138 | 0.5167 | 0.6500 |
| MEAN | 327 | 16 | 6 | 0.0061 | 0.3398 | 0.5045 |

## 5. Mean Result

| Metric | Value |
|---|---:|
| Total Reports | 327 |
| Unique Predicted Defect Types | 16 |
| Unique Defect Families | 6 |
| Mean Confidence Margin | 0.0061 |
| Top-1 Accuracy | 0.3398 |
| Top-2 Accuracy | 0.5045 |

## 6. Observations

1. Stage 6.5 generates structured explanations for 327 abnormal images, matching the Stage 6.4 `generic_label + crop_topk_only` prediction setting.

2. The explanation layer successfully attaches candidate region, anomaly score, defect family, visual evidence, manufacturing processes, possible causes, and inspection focus to each prediction.

3. The confidence margins are small, which means CLIP-based defect type predictions remain uncertain. Therefore, explanations should be treated as candidate reasoning rather than ground-truth causal diagnosis.

4. Wood has the strongest defect type prediction accuracy, while grid and leather remain weaker.

5. Manufacturing knowledge is more suitable for explanation generation than for direct CLIP classification prompts.

## 7. Conclusion

Stage 6.5 completes the current method pipeline:

```text
anomaly localization -> defect type reasoning -> manufacturing-aware explanation
```

The main contribution of this stage is not a higher classification score, but the addition of interpretable manufacturing-aware output on top of the strongest realistic Stage 6.4 setting.

## 8. Next Direction

Recommended next stage:

```text
Stage 6.6: Unified fair comparison table
```

The next stage should consolidate all positive, weak-positive, negative, upper-bound, and realistic-setting results into a single comparison table.

The unified table should explicitly record:

- dataset and category set
- number of images used
- candidate coverage
- fallback and skipped counts
- whether the setting is full-test, partial subset, or GT upper-bound
- prompt strategy
- candidate source
- Top-1 / Top-2 / Macro-F1
- Pixel F1 / IoU when applicable
- main gain and limitation
