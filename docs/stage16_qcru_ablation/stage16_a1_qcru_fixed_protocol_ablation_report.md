# Stage 16-A1 QCR-U Fixed-Protocol Ablation

## 1. Purpose

This stage evaluates fixed, non-tuned QCR-U ablation variants using the existing Stage 9 prediction table.

It does not train models, rerun VLM inference, or tune weights on the test set.

## 2. Input

- source: `results/stage9_qcr_u/stage9_a1_qcr_u_fusion_predictions.csv`
- deduplicated base rows: `25944`

The base table contains detector score, crop VLM score, candidate quality, and detector-VLM consistency.

## 3. Fixed Ablation Variants

| Variant | Formula | Meaning |
|---|---|---|
| detector_only | `D` | detector score only |
| crop_topk_vlm | `M` | crop VLM score only |
| naive_detector_crop_fusion | `0.5D + 0.5M` | naive fusion baseline |
| quality_weighted_crop | `0.5D + 0.5(M * (0.5 + 0.5Q))` | candidate quality modulates VLM evidence |
| quality_consistency_fusion | `0.4D + 0.4M + 0.1Q + 0.1K` | fixed Q+C fusion variant |

Where `D` is detector score, `M` is crop VLM abnormal score, `Q` is candidate quality, and `K` is detector-VLM high-high consistency.

## 4. Best Protocols by Q+C Fusion AUROC

| Rank | Backbone | Dataset | Strategy | Eval Mode | V5 AUROC | V5 AP | V5 Best F1 |
|---:|---|---|---|---|---:|---:|---:|
| 1 | FastFlow | VisA | inspection_binary | crop_topk_ensemble | 0.9842 | 0.9873 | 0.9403 |
| 2 | PatchCore | VisA | generic_binary | crop_topk_ensemble | 0.9775 | 0.9805 | 0.9409 |
| 3 | PatchCore | VisA | inspection_binary | crop_topk_ensemble | 0.9740 | 0.9784 | 0.9312 |
| 4 | FastFlow | VisA | inspection_binary | crop_or_full | 0.9724 | 0.9785 | 0.9206 |
| 5 | PatchCore | VisA | generic_binary | crop_or_full | 0.9702 | 0.9758 | 0.9302 |
| 6 | PatchCore | VisA | inspection_binary | crop_or_full | 0.9665 | 0.9731 | 0.9164 |
| 7 | PatchCore | VisA | category_binary | crop_topk_ensemble | 0.9525 | 0.9573 | 0.9007 |
| 8 | PatchCore | VisA | category_binary | crop_or_full | 0.9440 | 0.9518 | 0.8910 |
| 9 | FastFlow | VisA | inspection_binary | full_all | 0.9087 | 0.9261 | 0.8506 |
| 10 | PatchCore | VisA | inspection_binary | full_all | 0.8927 | 0.9090 | 0.8405 |
| 11 | PatchCore | VisA | generic_binary | full_all | 0.8822 | 0.9105 | 0.8263 |
| 12 | PatchCore | VisA | category_binary | full_all | 0.8802 | 0.9034 | 0.8277 |

## 5. Variant Comparison Within the Best Protocol

Best protocol by V5 AUROC: `FastFlow / VisA / inspection_binary / crop_topk_ensemble`.

| Variant | AUROC | AP | Best F1 | Best Accuracy |
|---|---:|---:|---:|---:|
| detector_only | 0.8955 | 0.9205 | 0.8445 | 0.8224 |
| crop_topk_vlm | 0.9269 | 0.9485 | 0.8765 | 0.8603 |
| naive_detector_crop_fusion | 0.9688 | 0.9750 | 0.9167 | 0.9047 |
| quality_weighted_crop | 0.9778 | 0.9822 | 0.9304 | 0.9214 |
| quality_consistency_fusion | 0.9842 | 0.9873 | 0.9403 | 0.9334 |

Key AUROC deltas in the best protocol:

- Q+C fusion minus naive fusion: `+0.0153`.
- Quality-weighted crop minus naive fusion: `+0.0090`.
- Q+C fusion minus detector-only: `+0.0887`.
- Q+C fusion minus crop VLM only: `+0.0572`.

## 6. Interpretation Rules

This stage is diagnostic. A positive result only means fixed Q+C evidence is useful under the existing Stage 9 signals.

It is not yet the final QCR-U method unless:

1. Q+C improves over naive fusion consistently, not only in one protocol.
2. The selected protocol is justified without test-set tuning.
3. Per-category results do not collapse on one or more primary categories.

## 7. Outputs

- `results/stage16_qcru_ablation/stage16_a1_qcru_fixed_ablation_per_config.csv`
- `results/stage16_qcru_ablation/stage16_a1_qcru_fixed_ablation_per_category.csv`
- `results/stage16_qcru_ablation/stage16_a1_qcru_fixed_ablation_best_by_protocol.csv`
- `docs/stage16_qcru_ablation/stage16_a1_qcru_fixed_protocol_ablation_report.md`
