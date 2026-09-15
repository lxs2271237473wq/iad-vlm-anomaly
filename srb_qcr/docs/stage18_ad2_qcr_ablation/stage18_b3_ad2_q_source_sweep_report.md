# Stage 18-B3 AD2 Q Source Sweep

## Purpose

Diagnose whether the Stage 18-B2 AD2 QCR drop against naive fusion is caused by the selected candidate quality source.

The sweep keeps the same QCR formulas as the current paper method and only changes the non-GT candidate quality source.

## Best ranked source

- q_source: `full_image_score`
- q_direction: `direct`
- mean V3 AUROC: `0.8286`
- mean V4 AUROC: `0.8505`
- mean V6 AUROC: `0.8500`
- V4 minus V3: `+0.0219`
- V6 minus V3: `+0.0214`
- V4 wins over V3: `1/4`
- worst category: `fruit_jelly`
- worst category delta V4-V3: `-0.0200`

## Top 10 sources

| Rank | Q source | Direction | V3 | V4 | V6 | V4-V3 | V6-V3 | Wins V4/V3 | Worst category |
|---:|---|---|---:|---:|---:|---:|---:|---:|---|
| 1 | full_image_score | direct | 0.8286 | 0.8505 | 0.8500 | +0.0219 | +0.0214 | 1/4 | fruit_jelly |
| 2 | candidate_score_max_min | inverted | 0.8286 | 0.8491 | 0.8484 | +0.0204 | +0.0198 | 3/4 | fruit_jelly |
| 3 | context_top1_score | inverted | 0.8286 | 0.8471 | 0.8467 | +0.0185 | +0.0180 | 2/4 | fruit_jelly |
| 4 | candidate_score_max_mean | inverted | 0.8286 | 0.8469 | 0.8469 | +0.0183 | +0.0183 | 3/4 | fruit_jelly |
| 5 | candidate_score_mean_min | inverted | 0.8286 | 0.8432 | 0.8420 | +0.0145 | +0.0134 | 2/4 | fruit_jelly |
| 6 | tight_topk_max_score | inverted | 0.8286 | 0.8414 | 0.8407 | +0.0127 | +0.0121 | 2/4 | fruit_jelly |
| 7 | tight_topk_mean_score | inverted | 0.8286 | 0.8408 | 0.8408 | +0.0121 | +0.0121 | 2/4 | fruit_jelly |
| 8 | candidate_score_mean_mean | inverted | 0.8286 | 0.8391 | 0.8383 | +0.0105 | +0.0097 | 2/4 | fruit_jelly |
| 9 | context_topk_max_score | direct | 0.8286 | 0.8387 | 0.8385 | +0.0101 | +0.0098 | 3/4 | fruit_jelly |
| 10 | map_area_max | direct | 0.8286 | 0.8356 | 0.8342 | +0.0070 | +0.0055 | 2/4 | fruit_jelly |

## Decision rule

- If a non-GT Q source gives V4 > V3 on mean AUROC and wins at least 3/4 categories, AD2 QCR can be promoted to a stronger supporting ablation.
- If no Q source passes that threshold, AD2 QCR should be reported as a boundary/diagnostic result rather than a main claim.
- Do not select a Q source using ground-truth overlap, ground-truth mask quality, or label-derived information.

## Outputs

- `results/stage18_ad2_qcr_ablation/stage18_b3_ad2_q_source_sweep_per_category.csv`
- `results/stage18_ad2_qcr_ablation/stage18_b3_ad2_q_source_sweep_summary.csv`
- `results/stage18_ad2_qcr_ablation/stage18_b3_ad2_q_source_sweep_ranked.csv`
