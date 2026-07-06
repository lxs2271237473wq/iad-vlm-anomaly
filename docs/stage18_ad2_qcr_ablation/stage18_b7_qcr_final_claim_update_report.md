# Stage 18-B7 QCR Final Claim Update

## Decision

QCR remains viable as the main innovation, but the claim must focus on candidate-quality calibration rather than adaptive refinement.

## Locked AD2 support setting

- locked selector: `semantic_candidate_score_max_mean_inverted`
- Q source: `candidate_score_max_mean`
- Q direction: `inverted`
- protocol: AD2 four-category leave-one-category-out policy selection

## Final AD2 paper-facing result

| Method | Mean AUROC | Delta vs naive | Wins | Paper role |
|---|---:|---:|---:|---|
| Naive detector-crop fusion | 0.8286 | +0.0000 |  | baseline |
| Quality-Calibrated QCR | 0.8469 | +0.0183 | 3/4 | main_qcr_support |
| Quality-Calibrated QCR + adaptive refinement | 0.8465 | +0.0178 | 3/4 | auxiliary_refinement |

## Final method choice

- selected AD2-facing QCR variant: `Quality-Calibrated QCR`
- selected AUROC: `0.8469`
- selected delta vs naive: `+0.0183`
- selected wins vs naive: `3/4`
- worst category delta: `-0.0067`
- note: Quality-only calibration is selected because it is slightly stronger than adaptive refinement on AD2.

## Paper wording

Use this wording:

```text
The main contribution is Quality-Calibrated QCR, which calibrates localization-guided crop-level VLM evidence using candidate-region quality. On VisA, the controlled ablation shows consistent gains over naive detector-crop fusion. On the AD2 four-category setting, a fixed semantic candidate-quality source with leave-one-category-out policy selection improves over naive fusion, providing supporting cross-category evidence. Adaptive consistency refinement is retained only as an auxiliary analysis rather than the primary source of improvement.
```

Avoid this wording:

```text
Adaptive QCR is universally beneficial across all datasets and all candidate-quality sources.
```

## Fold details

| Held-out | Q source | Direction | eta | gamma | V3 | Quality QCR | Adaptive QCR | Quality-V3 | Adaptive-V3 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| fruit_jelly | candidate_score_max_mean | inverted | 0.25 | 0.03 | 0.8767 | 0.8700 | 0.8700 | -0.0067 | -0.0067 |
| sheet_metal | candidate_score_max_mean | inverted | 0.25 | 0.03 | 0.7426 | 0.8019 | 0.8000 | +0.0593 | +0.0574 |
| vial | candidate_score_max_mean | inverted | 0.25 | 0.01 | 0.9182 | 0.9203 | 0.9203 | +0.0021 | +0.0021 |
| walnuts | candidate_score_max_mean | inverted | 0.25 | 0.00 | 0.7770 | 0.7956 | 0.7956 | +0.0185 | +0.0185 |

## Outputs

- `results/stage18_ad2_qcr_ablation/stage18_b7_ad2_qcr_final_paper_facing_table.csv`
- `results/stage18_ad2_qcr_ablation/stage18_b7_ad2_qcr_final_loco_folds.csv`
- `results/stage18_ad2_qcr_ablation/stage18_b7_qcr_final_claim_update.csv`
