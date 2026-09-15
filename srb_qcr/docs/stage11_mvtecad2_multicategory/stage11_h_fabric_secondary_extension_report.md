# Stage 11-H Fabric Secondary Category Extension

## 1. Purpose

This stage evaluates `fabric` as a secondary MVTec AD 2 category under the same Stage 11 candidate-region and VLM reasoning pipeline.

It does not modify Stage 11-C/D/E primary results. All outputs are stored as `stage11_h_*` files.

## 2. Why fabric is secondary

Stage 11-B1 categorized `fabric` as secondary because image-level PatchCore detection is acceptable, but pixel-level localization quality is weak.

| Metric | Value |
|---|---:|
| image AUROC | 0.7582 |
| image F1 | 0.7789 |
| pixel AUROC | 0.7871 |
| pixel F1 | 0.0765 |
| priority group | secondary |

## 3. Candidate Generation Status

| Category | Success | Images | Candidate rows | Time sec | Error |
|---|---:|---:|---:|---:|---|
| fabric | True | 78 | 200 | 992.8 | `` |

## 4. Candidate Quality

| Images | Candidate rows | Coverage | Mean cand/img | Top1 tight GT coverage | Top1 context GT coverage |
|---:|---:|---:|---:|---:|---:|
| 78 | 200 | 1.0000 | 2.5641 | 0.0684 | 0.1739 |

## 5. VLM Reasoning Results

| Method | AUROC | AP | Best F1 | Best Acc | ΔAUROC vs full |
|---|---:|---:|---:|---:|---:|
| patchcore_score | 0.7232 | 0.7736 | 0.8108 | 0.7308 | 0.3064 |
| tight_crop_top1 | 0.4949 | 0.5832 | 0.7563 | 0.6282 | 0.0781 |
| tight_crop_topk_mean | 0.4936 | 0.6057 | 0.7317 | 0.5769 | 0.0768 |
| full_image | 0.4168 | 0.5192 | 0.7317 | 0.5769 | 0.0000 |
| context_1.50_top1 | 0.3980 | 0.5477 | 0.7317 | 0.5769 | -0.0189 |
| context_1.50_topk_max | 0.3670 | 0.5374 | 0.7317 | 0.5769 | -0.0498 |
| context_1.50_topk_mean | 0.3468 | 0.5275 | 0.7317 | 0.5769 | -0.0700 |
| tight_crop_topk_max | 0.3320 | 0.5021 | 0.7317 | 0.5769 | -0.0848 |

## 6. Secondary Evidence Decision

| Item | Value |
|---|---:|
| full-image AUROC | 0.4168 |
| best VLM method | tight_crop_top1 |
| best VLM AUROC | 0.4949 |
| best VLM ΔAUROC vs full | 0.0781 |
| best context method | context_1.50_top1 |
| best context ΔAUROC vs full | -0.0189 |

Interpretation:

`fabric` should be used as a boundary/limitation case rather than main evidence.

## 7. Output Files

- Candidate regions: `results/stage11_mvtecad2_multicategory/stage11_h_fabric_candidate_regions.csv`
- Candidate summary: `results/stage11_mvtecad2_multicategory/stage11_h_fabric_candidate_summary.csv`
- Candidate status: `results/stage11_mvtecad2_multicategory/stage11_h_fabric_candidate_status.csv`
- VLM candidate scores: `results/stage11_mvtecad2_multicategory/stage11_h_fabric_vlm_candidate_scores.csv`
- VLM image predictions: `results/stage11_mvtecad2_multicategory/stage11_h_fabric_vlm_image_predictions.csv`
- VLM summary: `results/stage11_mvtecad2_multicategory/stage11_h_fabric_vlm_summary.csv`
- Evidence table: `results/stage11_mvtecad2_multicategory/stage11_h_fabric_secondary_evidence_table.csv`
- Report: `docs/stage11_mvtecad2_multicategory/stage11_h_fabric_secondary_extension_report.md`

## 8. Next Step

After this secondary extension is committed, Stage 11-I should build the final paper-ready Stage 11 evidence table.