# Stage 11-F2 Vial Source-aligned Candidate Drift Diagnostic

## 1. Purpose

This report compares Stage 10 and Stage 11 vial candidate construction using source-path alignment.
Stage 11 adapter paths are mapped back to original AD2 source paths through the Stage 11-A folder adapter mapping.

This step does not rerun PatchCore, VLM inference, or crop generation.

## 2. Alignment Result

| Item | Value |
|---|---:|
| Stage 10-D summary images | 71 |
| Stage 11-C summary images | 71 |
| Stage 10 unique source keys | 71 |
| Stage 11 unique source keys | 71 |
| Matched top1 images | 38 |
| Unmatched keys | 66 |

## 3. Candidate Policy Difference

| Metric | Value |
|---|---:|
| Stage 10 candidate rows | 123 |
| Stage 11 candidate rows | 117 |
| Stage 10 mean candidates/image | 1.7324 |
| Stage 11 mean candidates/image | 1.6479 |
| Mean tight bbox IoU | 0.7129 |
| Median tight bbox IoU | 0.7036 |
| Mean context bbox IoU | 0.1864 |
| Median context bbox IoU | 0.1750 |
| Mean abs tight area-ratio diff | 0.0124 |
| Mean abs context area-ratio diff | 0.3427 |

## 4. GT Coverage Difference

| Metric | Value |
|---|---:|
| Stage 10 mean tight GT coverage | 0.5576 |
| Stage 11 mean tight GT coverage | 0.4916 |
| Stage 10 mean context GT coverage | 0.6647 |
| Stage 11 mean context GT coverage | 0.8759 |
| Mean abs tight GT coverage diff | 0.0660 |
| Mean abs context GT coverage diff | 0.2112 |

## 5. Interpretation Rule

- If matched top1 images is close to 71 and bbox IoU is high, the Stage 10/Stage 11 vial discrepancy is unlikely to be candidate selection.
- If matched top1 images is close to 71 but bbox IoU or GT coverage differs strongly, Stage 11-C should be patched to reuse the Stage 10 candidate/crop policy.
- If matched top1 images is still far below 71, the alignment key remains wrong and the unmatched CSV should be inspected.

## 6. Current Decision

Alignment is still insufficient. Inspect unmatched source keys before drawing a policy conclusion.

## 7. Output

- Matched CSV: `results/stage11_mvtecad2_multicategory/stage11_f2_vial_source_aligned_drift_matched.csv`
- Unmatched CSV: `results/stage11_mvtecad2_multicategory/stage11_f2_vial_source_aligned_drift_unmatched.csv`
- Summary CSV: `results/stage11_mvtecad2_multicategory/stage11_f2_vial_source_aligned_drift_summary.csv`
- Report: `docs/stage11_mvtecad2_multicategory/stage11_f2_vial_source_aligned_drift_report.md`