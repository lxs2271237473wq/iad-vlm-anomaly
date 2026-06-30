# Stage 13-A PatchCore and VLM Score Fusion

## 1. Purpose

This stage tests whether the context-aware VLM branch provides complementary information to the strong PatchCore detector.

It does not train models, regenerate crops, or rerun VLM inference. It only fuses existing image-level scores from Stage 11-D.

## 2. Direct Baselines on ALL_PRIMARY

| Method | AUROC | AP | Best F1 | Best Acc |
|---|---:|---:|---:|---:|
| patchcore_score | 0.8087 | 0.9170 | 0.8641 | 0.7984 |
| full_image_score | 0.5201 | 0.7260 | 0.8398 | 0.7284 |
| context_topk_mean_score | 0.6036 | 0.7983 | 0.8337 | 0.7160 |
| patchcore_score | 0.7167 | 0.9105 | 0.8696 | 0.7750 |
| full_image_score | 0.7533 | 0.8997 | 0.9091 | 0.8500 |
| context_topk_mean_score | 0.8567 | 0.9556 | 0.8852 | 0.8250 |
| patchcore_score | 0.7463 | 0.9260 | 0.8911 | 0.8070 |
| full_image_score | 0.7130 | 0.9026 | 0.9000 | 0.8246 |
| context_topk_mean_score | 0.5870 | 0.8096 | 0.8824 | 0.7895 |
| patchcore_score | 0.8732 | 0.9525 | 0.9298 | 0.8873 |
| full_image_score | 0.6876 | 0.8733 | 0.8548 | 0.7606 |
| context_topk_mean_score | 0.5231 | 0.8041 | 0.8548 | 0.7465 |
| patchcore_score | 0.8052 | 0.8853 | 0.7963 | 0.7467 |
| full_image_score | 0.4296 | 0.6286 | 0.7500 | 0.6000 |
| context_topk_mean_score | 0.6430 | 0.7580 | 0.7563 | 0.6667 |

## 3. Same-set Fusion Result

Same-set fusion searches the fusion weight on the same evaluation set. It is an upper-bound diagnostic and should not be overclaimed as fair final evidence.

| Fusion | Best alpha for PatchCore | AUROC | Delta vs PatchCore | AP |
|---|---:|---:|---:|---:|
| patchcore_plus_full_image | 0.7000 | 0.8083 | -0.0003 | 0.9230 |
| patchcore_plus_context | 0.6800 | 0.8384 | 0.0297 | 0.9361 |

## 4. Leave-one-category-out Fusion

This protocol selects the fusion weight on three categories and evaluates on the held-out category.

| Category / Aggregate | Fusion | Alpha for PatchCore | AUROC | AP | Train AUROC |
|---|---|---:|---:|---:|---:|
| fruit_jelly | patchcore_plus_full_image | 0.8000 | 0.7333 | 0.9178 | 0.8179 |
| fruit_jelly | patchcore_plus_context | 0.7000 | 0.8333 | 0.9476 | 0.8408 |
| sheet_metal | patchcore_plus_full_image | 1.0000 | 0.7463 | 0.9260 | 0.8101 |
| sheet_metal | patchcore_plus_context | 0.6800 | 0.7481 | 0.9261 | 0.8537 |
| vial | patchcore_plus_full_image | 0.7000 | 0.8700 | 0.9529 | 0.7752 |
| vial | patchcore_plus_context | 0.6300 | 0.9224 | 0.9725 | 0.7994 |
| walnuts | patchcore_plus_full_image | 0.5500 | 0.6659 | 0.8009 | 0.8613 |
| walnuts | patchcore_plus_context | 0.6300 | 0.7800 | 0.8751 | 0.8609 |
| ALL_PRIMARY | patchcore_plus_full_image |  | 0.7496 | 0.8981 |  |
| ALL_PRIMARY | patchcore_plus_context |  | 0.8301 | 0.9333 |  |

## 5. Per-category Same-set Best Fusion

| Category | Fusion | Alpha for PatchCore | AUROC | AP |
|---|---|---:|---:|---:|
| fruit_jelly | patchcore_plus_full_image | 0.2700 | 0.8000 | 0.9345 |
| fruit_jelly | patchcore_plus_context | 0.3400 | 0.8933 | 0.9711 |
| sheet_metal | patchcore_plus_full_image | 0.5500 | 0.9185 | 0.9791 |
| sheet_metal | patchcore_plus_context | 0.6400 | 0.7556 | 0.9265 |
| vial | patchcore_plus_full_image | 0.8900 | 0.8816 | 0.9565 |
| vial | patchcore_plus_context | 0.6600 | 0.9256 | 0.9734 |
| walnuts | patchcore_plus_full_image | 1.0000 | 0.8052 | 0.8853 |
| walnuts | patchcore_plus_context | 0.9100 | 0.8067 | 0.8883 |

## 6. Score Complementarity

| Category | PatchCore vs full-image correlation | PatchCore vs context correlation |
|---|---:|---:|
| ALL_PRIMARY | -0.1516 | -0.1077 |
| fruit_jelly | 0.5649 | 0.1877 |
| sheet_metal | -0.3357 | -0.3083 |
| vial | 0.2400 | -0.2551 |
| walnuts | 0.2195 | 0.3115 |

## 7. Decision

Same-set PatchCore+context fusion improves over PatchCore by 0.0297 AUROC.
Leave-one-category-out PatchCore+context fusion improves over PatchCore by 0.0214 AUROC.

If fusion does not improve over PatchCore, the VLM branch should not be claimed as a detector-strength improvement. The next analysis should move to PatchCore error cases and uncertain samples.

## 8. Output Files

- Fusion grid: `results/stage13_strong_baseline/stage13_a_patchcore_vlm_fusion_grid.csv`
- Fusion summary: `results/stage13_strong_baseline/stage13_a_patchcore_vlm_fusion_summary.csv`
- Per-category fusion: `results/stage13_strong_baseline/stage13_a_patchcore_vlm_fusion_per_category.csv`
- Leave-one-category-out fusion: `results/stage13_strong_baseline/stage13_a_patchcore_vlm_fusion_loco_category.csv`
- Complementarity table: `results/stage13_strong_baseline/stage13_a_patchcore_vlm_score_complementarity.csv`
- Report: `docs/stage13_strong_baseline/stage13_a_patchcore_vlm_fusion_report.md`