# Stage 11-F Vial Candidate Policy Drift Diagnostic

## 1. Purpose

This diagnostic compares the single-category Stage 10 vial candidate construction with the multi-category Stage 11 vial candidate construction.
It does not rerun PatchCore, VLM inference, or crop generation.

## 2. Why this is needed

Stage 10-G reported positive vial context evidence, while Stage 11-D did not reproduce the same margin.
Therefore, this diagnostic checks whether the mismatch is caused by candidate policy drift rather than by the idea of context-aware crop reasoning itself.

## 3. Vial VLM Result Difference

| Stage | full_image AUROC | context_1.50_top1 AUROC | ΔAUROC |
|---|---:|---:|---:|
| Stage 10-F/G | 0.6488 | 0.7746 | 0.1258 |
| Stage 11-D | 0.6876 | 0.6834 | -0.0042 |

## 4. Candidate Construction Summary

| Scope | Candidate rows | Images | Mean cand/img | Top1 bbox area | Top1 context area | Top1 tight GT coverage | Top1 context GT coverage |
|---|---:|---:|---:|---:|---:|---:|---:|
| stage10 | 123 | 60 | 2.0500 | 0.0381 |  |  |  |
| stage11 | 117 | 66 | 1.7727 | 0.0273 | 0.3940 | 0.4275 | 0.8016 |

## 5. Matched-image Top1 BBox Difference

- Matched images: `39`
- Mean top1 bbox IoU: `0.6003`
- Median top1 bbox IoU: `0.6802`
- Mean absolute bbox area-ratio difference: `0.0133`
- Mean absolute tight GT coverage difference: ``

## 6. Interpretation Rule

- If bbox IoU is low, Stage 10 and Stage 11 are selecting different visual regions.
- If GT coverage differs strongly, the VLM discrepancy is likely caused by candidate construction.
- If candidate count or bbox area ratio differs strongly, Stage 11-C should be patched to reuse the Stage 10 candidate policy before extending to fabric.

## 7. Output

- Matched CSV: `results/stage11_mvtecad2_multicategory/stage11_f_vial_candidate_policy_drift_matched.csv`
- Summary CSV: `results/stage11_mvtecad2_multicategory/stage11_f_vial_candidate_policy_drift_summary.csv`
- Report: `docs/stage11_mvtecad2_multicategory/stage11_f_vial_candidate_policy_drift_report.md`