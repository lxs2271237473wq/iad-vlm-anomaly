# Stage 11-B1 PatchCore Detector Quality Analysis

## 1. Purpose

This analysis summarizes the Stage 11-B multi-category PatchCore baseline and decides which MVTec AD 2 categories are suitable for Stage 11-C candidate-region generation and Stage 11-D VLM reasoning.

This step does not train PatchCore, run VLM inference, or generate candidate crops.

## 2. Decision Rule

| Group | Rule | Usage |
|---|---|---|
| primary | image AUROC >= 0.75, pixel AUROC >= 0.85, pixel F1 >= 0.30 | Main evidence for context-aware crop reasoning |
| secondary | image AUROC >= 0.70 and pixel AUROC >= 0.75 | Usable, but localization needs caution |
| detector_risk | image AUROC < 0.60 | Do not use as main VLM claim; detector may dominate failure |
| diagnostic | mixed metrics | Use only for qualitative or failure analysis |

## 3. Category Ranking

| Category | image AUROC | image F1 | pixel AUROC | pixel F1 | Group | Interpretation |
|---|---:|---:|---:|---:|---|---|
| sheet_metal | 0.8315 | 0.9000 | 0.8595 | 0.3765 | primary | Detector quality is sufficient for localization-guided crop/VLM reasoning. |
| vial | 0.7987 | 0.8598 | 0.9484 | 0.3366 | primary | Detector quality is sufficient for localization-guided crop/VLM reasoning. |
| fruit_jelly | 0.7900 | 0.8788 | 0.9476 | 0.4963 | primary | Detector quality is sufficient for localization-guided crop/VLM reasoning. |
| walnuts | 0.7822 | 0.6471 | 0.9193 | 0.3918 | primary | Detector quality is sufficient for localization-guided crop/VLM reasoning. |
| fabric | 0.7582 | 0.7789 | 0.7871 | 0.0765 | secondary | Image-level detection is acceptable, but localization quality should be treated cautiously. |
| rice | 0.5630 | 0.8108 | 0.7637 | 0.0552 | detector_risk | PatchCore image-level detection is weak; VLM crop results may mainly reflect detector failure. |
| wallplugs | 0.4626 | 0.7458 | 0.8675 | 0.0391 | detector_risk | PatchCore image-level detection is weak; VLM crop results may mainly reflect detector failure. |
| can | 0.3901 | 0.6441 | 0.7119 | 0.0002 | detector_risk | PatchCore image-level detection is weak; VLM crop results may mainly reflect detector failure. |

## 4. Stage 11-C Category Plan

- Primary categories: `sheet_metal, vial, fruit_jelly, walnuts`
- Secondary categories: `fabric`
- Diagnostic categories: `none`
- Detector-risk categories: `rice, wallplugs, can`

Recommended Stage 11-C execution plan:

1. First run candidate-region generation on the primary categories.
2. Then include secondary categories if runtime is acceptable.
3. Keep detector-risk categories for failure analysis, not for the main paper claim.

## 5. Paper-level Interpretation

The multi-category results show that the proposed VLM reasoning branch should be evaluated conditionally on detector quality.
If PatchCore localization is weak, crop-based VLM reasoning may fail because the crop is not a reliable visual bridge.
Therefore, the next module should report both detector quality and VLM reasoning quality instead of treating all categories as equally valid evidence.

## 6. Output

- Analysis CSV: `results/stage11_mvtecad2_multicategory/stage11_b1_patchcore_detector_quality_analysis.csv`
- Report: `docs/stage11_mvtecad2_multicategory/stage11_b1_patchcore_detector_quality_analysis.md`