# Stage 18-B4 AD2 QCR Claim-safe Decision

## Purpose

Convert the AD2 Q-source sweep into a claim-safe decision table.

## Key decision

- final_status: `ad2_qcr_supporting_sensitivity_not_main_claim_yet`
- recommended_q_if_formalizing_new_definition: `candidate_score_max_mean / inverted`

## Why the best overall source is excluded

The top overall source in B3 is `full_image_score/direct`, but this is full-image VLM evidence, not candidate quality. It must not be used as Q in a candidate-quality calibration claim.

## Claim-safe cases

| Case | Status | Q source | Direction | V3 | V4 | V6 | V4-V3 | Wins | Worst category |
|---|---|---|---|---:|---:|---:|---:|---:|---|
| B2_default_q_source | boundary_result_not_main_claim | candidate_score_mean_max | direct | 0.8286 | 0.8191 | 0.8194 | -0.0095 | NA |  |
| B3_best_overall_invalid_as_Q | exclude_from_qcr_claim | full_image_score | direct | 0.8286 | 0.8505 | 0.8500 | +0.0219 | 1 | fruit_jelly |
| B3_performance_best_valid_candidate_Q | supporting_source_sensitivity | candidate_score_max_min | inverted | 0.8286 | 0.8491 | 0.8484 | +0.0204 | 3 | fruit_jelly |
| B3_stability_preferred_valid_candidate_Q | recommended_if_formalizing_new_Q_definition | candidate_score_max_mean | inverted | 0.8286 | 0.8469 | 0.8469 | +0.0183 | 3 | fruit_jelly |

## Paper recommendation

Use AD2 four-category QCR as a source-sensitivity/boundary-supporting result unless the new Q definition is formally locked and rerun consistently across the main VisA ablation.

Recommended wording:

```text
On the AD2 four-category setting, the default transferred Q source is not uniformly beneficial. A non-GT candidate-region score source recovers a positive mean gain over naive detector-crop fusion, but we report this as a candidate-quality source sensitivity rather than as the primary QCR claim.
```

## Outputs

- `results/stage18_ad2_qcr_ablation/stage18_b4_ad2_qcr_valid_q_sources_ranked.csv`
- `results/stage18_ad2_qcr_ablation/stage18_b4_ad2_qcr_claim_safe_decision.csv`
- `results/stage18_ad2_qcr_ablation/stage18_b4_ad2_qcr_paper_facing_table.csv`
