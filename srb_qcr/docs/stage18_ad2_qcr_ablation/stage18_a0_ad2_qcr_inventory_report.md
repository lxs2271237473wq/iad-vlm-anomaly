# Stage 18-A0 AD2 QCR Inventory

## Purpose

Check whether existing QCR prediction/result files already contain AD2 four-category data for:

```text
fruit_jelly
sheet_metal
vial
walnuts
```

## Decision

- can_directly_run_ad2_qcr_ablation: `False`

## Inventory

| File | Readable | Rows | AD2 coverage | Directly usable | Notes |
|---|---:|---:|---:|---:|---|
| `results/stage9_qcr_u/stage9_a1_qcr_u_fusion_predictions.csv` | 1 | 155664 | 0/4 | 0 | does not contain full AD2 QCR-ready prediction columns/categories |
| `results/stage16_qcru_ablation/stage16_b_adaptive_qcru_all_variants_per_config.csv` | 1 | 72 | 0/4 | 0 | does not contain full AD2 QCR-ready prediction columns/categories |
| `results/stage16_qcru_ablation/stage16_b_adaptive_qcru_all_variants_per_category.csv` | 1 | 864 | 0/4 | 0 | does not contain full AD2 QCR-ready prediction columns/categories |
| `results/stage16_qcru_ablation/stage16_d_paper_facing_qcr_ablation_table.csv` | 1 | 12 | 0/4 | 0 | does not contain full AD2 QCR-ready prediction columns/categories |

## Next Action

Proceed to Stage 18-B: generate missing AD2 QCR prediction file first.

This means the current QCR ablation evidence is not yet aligned with the AD2 four-category system-level baseline.