# Stage 11-F1 Vial Path-aligned Candidate Drift Diagnostic

## 1. Purpose

This report fixes the Stage 11-F image matching issue by comparing Stage 10 and Stage 11 vial candidates using path-aware keys.
It does not rerun PatchCore, VLM inference, or crop generation.

## 2. Original Candidate Summary

| Source | Images | Candidate rows |
|---|---:|---:|
| Stage 10-D summary | 71 | 123 |
| Stage 11-C summary | 71 | 117 |

## 3. Matching Diagnostics

| Match key | Stage10 images | Stage11 images | Matched images | Stage10 cand/img | Stage11 cand/img | Mean top1 bbox IoU | Median IoU | Mean abs bbox-area diff |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| key_label_stem | 71 | 71 | 38 | 1.7324 | 1.6479 | nan | nan | nan |
| key_stem | 60 | 66 | 39 | 2.0500 | 1.7727 | nan | nan | nan |

## 4. Selected Matching Key

The selected matching key is `key_stem`, because it gives the largest matched image count while preserving label information when possible.

## 5. Interpretation

If the selected key still matches far fewer than 71 images, the previous Stage 11-F conclusion is not reliable and the next step should inspect the exact unmatched image lists.
If the selected key matches close to 71 images and bbox IoU is high, the vial inconsistency is less likely to be from candidate region selection and more likely to come from crop image construction, prompt/backend differences, or aggregation.
If the selected key matches close to 71 images but bbox IoU is low, Stage 11-C should be patched to reuse the Stage 10 candidate policy.

## 6. Output

- Matched CSV: `results/stage11_mvtecad2_multicategory/stage11_f1_vial_path_aligned_drift_matched.csv`
- Summary CSV: `results/stage11_mvtecad2_multicategory/stage11_f1_vial_path_aligned_drift_summary.csv`
- Report: `docs/stage11_mvtecad2_multicategory/stage11_f1_vial_path_aligned_drift_report.md`