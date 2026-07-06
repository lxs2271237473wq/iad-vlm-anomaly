# Stage 18-B6 AD2 LOCO Robust QCR Selector Sweep

## Purpose

Test whether AD2 QCR can be rescued by a more robust train-category selector rather than the B5 selector that maximizes train adaptive AUROC.

No held-out category labels are used for selecting Q source, direction, eta, or gamma.

## Valid candidate-quality sources

```text
candidate_score_max_mean
candidate_score_max_min
candidate_score_mean_max
candidate_score_mean_mean
candidate_score_mean_min
context_candidate_mask_density_max
context_candidate_mask_density_mean
context_candidate_mask_density_min
map_area_max
map_area_mean
map_area_min
num_candidates_x
num_candidates_y
tight_candidate_mask_density_max
tight_candidate_mask_density_mean
tight_candidate_mask_density_min
```

## Best selector

- selector: `semantic_candidate_score_max_min_inverted`
- claim_status: `weak_positive_boundary_support`
- mean test V3: `0.8286`
- mean test adaptive QCR: `0.8468`
- adaptive QCR minus V3: `+0.0182`
- adaptive wins over V3: `2/4`
- worst adaptive category: `fruit_jelly`
- worst adaptive delta: `-0.0133`

## Selector summary

| Selector | Status | V3 | Adaptive | Delta | Wins | Worst category | Worst delta |
|---|---|---:|---:|---:|---:|---|---:|
| B5_baseline_max_train_adaptive_auroc | do_not_promote | 0.8286 | 0.8028 | -0.0258 | 1/4 | sheet_metal | -0.1148 |
| max_train_delta_adaptive | do_not_promote | 0.8286 | 0.8028 | -0.0258 | 1/4 | sheet_metal | -0.1148 |
| wins_then_delta_adaptive | do_not_promote | 0.8286 | 0.8010 | -0.0276 | 1/4 | sheet_metal | -0.1148 |
| robust_delta_score | do_not_promote | 0.8286 | 0.8004 | -0.0282 | 1/4 | sheet_metal | -0.1148 |
| robust_quality_score | do_not_promote | 0.8286 | 0.8004 | -0.0282 | 1/4 | sheet_metal | -0.1148 |
| worst_delta_then_mean_delta_adaptive | do_not_promote | 0.8286 | 0.7957 | -0.0329 | 0/4 | sheet_metal | -0.1148 |
| semantic_candidate_score_max_mean_inverted | promote_ad2_qcr_support | 0.8286 | 0.8465 | +0.0178 | 3/4 | fruit_jelly | -0.0067 |
| semantic_candidate_score_max_min_inverted | weak_positive_boundary_support | 0.8286 | 0.8468 | +0.0182 | 2/4 | fruit_jelly | -0.0133 |

## Decision rule

- If at least one selector has positive mean adaptive delta and wins at least 3/4 held-out categories, AD2 QCR can be used as supporting cross-category evidence.
- If all selectors have negative mean delta, stop optimizing AD2 QCR and report AD2 as boundary/sensitivity evidence.

## Outputs

- `results/stage18_ad2_qcr_ablation/stage18_b6_ad2_loco_robust_selector_folds.csv`
- `results/stage18_ad2_qcr_ablation/stage18_b6_ad2_loco_robust_selector_summary.csv`
