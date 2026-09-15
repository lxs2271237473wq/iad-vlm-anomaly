# Stage 11-F3 Vial Image-set Overlap Audit

## 1. Purpose

This report audits whether Stage 10-D/G and Stage 11-C/D use the same vial image set.
It does not run PatchCore, VLM inference, or crop generation.

## 2. Why this matters

Stage 10-G and Stage 11-D reported different vial context-crop effects. Before attributing this to candidate policy drift, the exact image sets must be aligned.

## 3. Key-overlap diagnostics

| Comparison | Stage10 unique | Stage11 unique | Intersection | Stage10 only | Stage11 only | Jaccard |
|---|---:|---:|---:|---:|---:|---:|
| stage10_labeled_key_vs_stage11_source_labeled_key | 71 | 71 | 38 | 33 | 33 | 0.3654 |
| stage10_stem_vs_stage11_source_stem | 60 | 66 | 39 | 21 | 27 | 0.4483 |
| stage10_labeled_key_vs_stage11_labeled_key | 71 | 71 | 38 | 33 | 33 | 0.3654 |
| stage10_stem_vs_stage11_stem | 60 | 66 | 39 | 21 | 27 | 0.4483 |

## 4. Best available alignment

The best available alignment is `stage10_stem_vs_stage11_source_stem`, with intersection `39` out of Stage 10 `60` and Stage 11 `66` unique keys.

## 5. Decision

The image sets are not sufficiently aligned. Stage 10-G and Stage 11-D vial AUROC should not be treated as directly comparable until the source of the image-set mismatch is resolved.

The next step should inspect unmatched examples and decide whether to rerun vial under a single unified Stage 11 pipeline or retire Stage 10-G vial as a historical single-category result.

## 6. Output

- Keys CSV: `results/stage11_mvtecad2_multicategory/stage11_f3_vial_image_set_keys.csv`
- Overlap CSV: `results/stage11_mvtecad2_multicategory/stage11_f3_vial_image_set_overlap.csv`
- Summary CSV: `results/stage11_mvtecad2_multicategory/stage11_f3_vial_image_set_overlap_summary.csv`
- Report: `docs/stage11_mvtecad2_multicategory/stage11_f3_vial_image_set_overlap_report.md`