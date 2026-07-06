# Stage 18-B5 AD2 LOCO QCR Policy Optimization

## Purpose

Optimize QCR policy without using the held-out AD2 category labels for selection.

Each fold selects Q source, Q direction, eta, and gamma on three AD2 categories, then evaluates on the held-out category.

## Safe Q source candidates

```text
num_candidates_x
tight_top1_score
tight_topk_max_score
tight_topk_mean_score
candidate_score_mean_max
candidate_score_mean_mean
candidate_score_mean_min
candidate_score_max_mean
candidate_score_max_min
tight_candidate_mask_density_max
tight_candidate_mask_density_mean
tight_candidate_mask_density_min
context_candidate_mask_density_max
context_candidate_mask_density_mean
context_candidate_mask_density_min
map_area_max
map_area_mean
map_area_min
num_candidates_y
```

## Summary

- final_status: `do_not_promote_ad2_qcr_main_claim`
- mean test V3 naive: `0.8286`
- mean test quality QCR: `0.8028`
- mean test adaptive QCR: `0.8028`
- quality QCR minus V3: `-0.0258`
- adaptive QCR minus V3: `-0.0258`
- quality QCR wins over V3: `1/4`
- adaptive QCR wins over V3: `1/4`
- worst adaptive category: `sheet_metal`
- worst adaptive delta: `-0.1148`

## Selected folds

| Held-out | Selected Q | Direction | eta | gamma | Test V3 | Test Quality | Test Adaptive | Adaptive-V3 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| fruit_jelly | candidate_score_max_min | inverted | 0.30 | 0.00 | 0.8767 | 0.8700 | 0.8700 | -0.0067 |
| sheet_metal | candidate_score_mean_max | direct | 0.35 | 0.03 | 0.7426 | 0.6278 | 0.6278 | -0.1148 |
| vial | candidate_score_max_min | inverted | 0.30 | 0.00 | 0.9182 | 0.9151 | 0.9151 | -0.0031 |
| walnuts | candidate_score_max_min | inverted | 0.30 | 0.00 | 0.7770 | 0.7985 | 0.7985 | +0.0215 |

## Decision rule

- If adaptive QCR has positive mean delta and wins at least 3/4 held-out categories, QCR can be promoted as cross-category calibrated AD2 support.
- If only mean delta is positive but wins fewer than 3/4, report AD2 as weak/boundary support.
- If mean delta is negative, keep AD2 QCR as source-sensitivity diagnostic and retain VisA as the main ablation.

## Outputs

- `results/stage18_ad2_qcr_ablation/stage18_b5_ad2_loco_qcr_all_configs_per_category.csv`
- `results/stage18_ad2_qcr_ablation/stage18_b5_ad2_loco_qcr_selected_folds.csv`
- `results/stage18_ad2_qcr_ablation/stage18_b5_ad2_loco_qcr_summary.csv`
