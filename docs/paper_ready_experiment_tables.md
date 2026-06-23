# Paper-ready Experiment Tables

## 1. Main Fair Result Table

This table uses the same 328-image full-test setting. The baseline is full-image reasoning under the same prompt strategy.

| Prompt Strategy | Baseline | Method | Images | Top-1 | Top-1 Δ | Top-2 | Top-2 Δ | Macro-F1 | Macro-F1 Δ | Fair Setting |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| generic_label | full_all | crop_topk_ensemble | 328 | 0.3388 | +0.0537 | 0.5072 | +0.0082 | 0.2206 | +0.0663 | same 328-image setting |
| short_visual | full_all | crop_topk_ensemble | 328 | 0.3187 | +0.0390 | 0.5131 | +0.0426 | 0.2022 | +0.0422 | same 328-image setting |
| category_visual | full_all | crop_topk_ensemble | 328 | 0.2872 | +0.0281 | 0.5124 | +0.0492 | 0.2069 | +0.0213 | same 328-image setting |
| visual_ensemble | full_all | crop_topk_ensemble | 328 | 0.2801 | -0.0368 | 0.5722 | +0.0811 | 0.1681 | -0.0232 | same 328-image setting |

## 2. Ablation: Prompt Strategy and Crop Mode

This table shows full image, crop-or-full, top-k crop ensemble, crop-only, and crop-topk-only settings.

| Strategy | Input Mode | Setting Type | Used Images | Coverage | Fallback | Skipped | Top-1 | Top-2 | Macro-F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| category_visual | full_all | realistic_full_test | 328.0 | 1.0000 | 0.0 | 0.0 | 0.2592 | 0.4633 | 0.1857 |
| category_visual | crop_or_full | realistic_full_test | 328.0 | 0.9970 | 1.0 | 0.0 | 0.2787 | 0.4934 | 0.1952 |
| category_visual | crop_topk_ensemble | realistic_full_test | 328.0 | 0.9970 | 1.0 | 0.0 | 0.2872 | 0.5124 | 0.2069 |
| category_visual | crop_only | near_full_candidate_subset | 327.0 | 0.9970 | 0.0 | 1.0 | 0.2798 | 0.4955 | 0.1961 |
| category_visual | crop_topk_only | near_full_candidate_subset | 327.0 | 0.9970 | 0.0 | 1.0 | 0.2884 | 0.5146 | 0.2083 |
| generic_label | full_all | realistic_full_test | 328.0 | 1.0000 | 0.0 | 0.0 | 0.2850 | 0.4990 | 0.1543 |
| generic_label | crop_or_full | realistic_full_test | 328.0 | 0.9970 | 1.0 | 0.0 | 0.3178 | 0.5030 | 0.1950 |
| generic_label | crop_topk_ensemble | realistic_full_test | 328.0 | 0.9970 | 1.0 | 0.0 | 0.3388 | 0.5072 | 0.2206 |
| generic_label | crop_only | near_full_candidate_subset | 327.0 | 0.9970 | 0.0 | 1.0 | 0.3188 | 0.5002 | 0.1954 |
| generic_label | crop_topk_only | near_full_candidate_subset | 327.0 | 0.9970 | 0.0 | 1.0 | 0.3398 | 0.5045 | 0.2209 |
| short_visual | full_all | realistic_full_test | 328.0 | 1.0000 | 0.0 | 0.0 | 0.2798 | 0.4705 | 0.1600 |
| short_visual | crop_or_full | realistic_full_test | 328.0 | 0.9970 | 1.0 | 0.0 | 0.3104 | 0.5131 | 0.1972 |
| short_visual | crop_topk_ensemble | realistic_full_test | 328.0 | 0.9970 | 1.0 | 0.0 | 0.3187 | 0.5131 | 0.2022 |
| short_visual | crop_only | near_full_candidate_subset | 327.0 | 0.9970 | 0.0 | 1.0 | 0.3112 | 0.5151 | 0.1975 |
| short_visual | crop_topk_only | near_full_candidate_subset | 327.0 | 0.9970 | 0.0 | 1.0 | 0.3196 | 0.5151 | 0.2025 |
| visual_ensemble | full_all | realistic_full_test | 328.0 | 1.0000 | 0.0 | 0.0 | 0.3170 | 0.4911 | 0.1912 |
| visual_ensemble | crop_or_full | realistic_full_test | 328.0 | 0.9970 | 1.0 | 0.0 | 0.2822 | 0.5641 | 0.1743 |
| visual_ensemble | crop_topk_ensemble | realistic_full_test | 328.0 | 0.9970 | 1.0 | 0.0 | 0.2801 | 0.5722 | 0.1681 |
| visual_ensemble | crop_only | near_full_candidate_subset | 327.0 | 0.9970 | 0.0 | 1.0 | 0.2830 | 0.5663 | 0.1745 |
| visual_ensemble | crop_topk_only | near_full_candidate_subset | 327.0 | 0.9970 | 0.0 | 1.0 | 0.2809 | 0.5743 | 0.1683 |

## 3. GT-crop Upper-bound Diagnostic Table

These results use ground-truth masks for cropping. They diagnose the potential of accurate localization but should not be reported as fair deployable results.

| Stage | Method | Prompt Strategy | Input | Candidate Source | Images | Top-1 | Top-2 | Macro-F1 | Role |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Stage 6.2 | category_visual + gt_crop | category_visual | gt_crop | GT mask | 328.0 | 0.3102 | 0.4655 | 0.2460 | upper-bound diagnostic, not fair deployable result |
| Stage 6.2 | short_visual + gt_crop | short_visual | gt_crop | GT mask | 328.0 | 0.3045 | 0.4669 | 0.2100 | upper-bound diagnostic, not fair deployable result |
| Stage 6.1-D | category_aware + gt_crop | category_aware | gt_crop | GT mask | 328.0 | 0.2973 | 0.5022 | 0.2147 | upper-bound diagnostic, not fair deployable result |
| Stage 6.2 | generic_label + gt_crop | generic_label | gt_crop | GT mask | 328.0 | 0.2925 | 0.4721 | 0.1973 | upper-bound diagnostic, not fair deployable result |
| Stage 6.1-D | generic + gt_crop | generic | gt_crop | GT mask | 328.0 | 0.2904 | 0.4623 | 0.1865 | upper-bound diagnostic, not fair deployable result |
| Stage 6.1-D | manufacturing_aware + gt_crop | manufacturing_aware | gt_crop | GT mask | 328.0 | 0.2788 | 0.4539 | 0.1969 | upper-bound diagnostic, not fair deployable result |
| Stage 6.2 | visual_ensemble + gt_crop | visual_ensemble | gt_crop | GT mask | 328.0 | 0.2744 | 0.4974 | 0.1837 | upper-bound diagnostic, not fair deployable result |

## 4. Negative / Auxiliary Result Table

These results explain why several branches are not used as the final main module.

| Stage | Branch | Method | Metric | Main Gain | Main Limitation | Use in Paper |
| --- | --- | --- | --- | --- | --- | --- |
| Stage 4.1 | localization | anomaly region scoring baseline | 0.471249343426501 | weak positive candidate ranking | uses candidate subset and hand-designed features | auxiliary / negative / motivation result |
| Stage 4.2 | localization | enhanced anomaly region scoring | 0.4745791025522283 | adds contrast/stability/shape features | does not outperform Stage 4.1 | auxiliary / negative / motivation result |
| Stage 5.1 | localization | simple anomaly-map calibration | 0.4705774664749127 | tests fixed/percentile/mean-std thresholding | best method remains fixed threshold | auxiliary / negative / motivation result |
| Stage 5.2 | localization | trainable Tiny CNN anomaly-map calibration | 0.3955828949574317 | tests trainable calibration | naive trainable calibration collapses on several categories | auxiliary / negative / motivation result |
| Stage 5.3 | localization | conservative residual anomaly-map calibration | 0.5150067848961855 | stable residual calibration | gain is too small to use as main module | auxiliary / negative / motivation result |
| Stage 5.4 | localization | conservative residual calibration sweep best: cfg03_larger_delta | 0.5183686629377395 | best residual calibration setting found by sweep | best gain remains too small; trainable calibration line stopped | auxiliary / negative / motivation result |
| Stage 6.5 | manufacturing_explanation | manufacturing-aware explanation generation | 0.3397873279746681 | adds structured manufacturing-aware explanation without changing classification output | explanations are candidate reasoning, not verified causal diagnosis | auxiliary / negative / motivation result |
| SAM2 analysis | localization | naive SAM2 box prompt and anomaly-aware SAM2 selection | below or similar to PatchCore component mask | shows that generic segmentation does not directly solve defect mask refinement | SAM2 tends to segment object/texture regions rather than defect regions | negative result / motivation |

## 5. Manufacturing-aware Explanation Table

Stage 6.5 adds structured manufacturing-aware explanations on top of the Stage 6.4 prediction results. It does not modify classification accuracy.

| Category | Reports | Predicted Defect Types | Defect Families | Top-1 | Top-2 | Confidence Margin | Role |
| --- | --- | --- | --- | --- | --- | --- | --- |
| grid | 56 | 4 | 3 | 0.2321 | 0.3750 | 0.0043 | explanation layer, not accuracy-improvement module |
| leather | 92 | 5 | 4 | 0.2826 | 0.4130 | 0.0026 | explanation layer, not accuracy-improvement module |
| screw | 119 | 5 | 2 | 0.3277 | 0.5798 | 0.0058 | explanation layer, not accuracy-improvement module |
| wood | 60 | 3 | 3 | 0.5167 | 0.6500 | 0.0138 | explanation layer, not accuracy-improvement module |
| MEAN | 327 | 16 | 6 | 0.3398 | 0.5045 | 0.0061 | explanation layer, not accuracy-improvement module |

## 6. Paper-level Conclusion

The cleanest main result is the realistic full-test comparison:

```text
generic_label + full_all -> generic_label + crop_topk_ensemble
same 328 images
Top-1 improves from 0.2850 to 0.3388
Macro-F1 improves from 0.1543 to 0.2206
```

This supports the final method direction:

```text
PatchCore anomaly crop -> short visual prompt defect reasoning -> manufacturing-aware explanation
```
