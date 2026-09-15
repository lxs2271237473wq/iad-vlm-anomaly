# Stage 15-C fruit_jelly Modern Baseline Comparison

## 1. Purpose

This report compares EfficientAD, WinCLIP, PatchCore, context-aware VLM, and PatchCore+context fusion on AD2 fruit_jelly.

EfficientAD is included as a modern non-VLM detector baseline, while WinCLIP is included as an external VLM anomaly detection baseline.

## 2. Result Table

| Rank | Method | Group | Image AUROC | Image F1 | Pixel AUROC | Pixel F1 | Protocol |
|---:|---|---|---:|---:|---:|---:|---|
| 1 | PatchCore + context VLM, same-set | fusion_same_set | 0.8933 | 0.9286 |  |  | Stage 13 same-set fusion upper-bound |
| 2 | context-aware VLM | vlm_branch | 0.8567 |  |  |  | Stage 11 context-aware VLM |
| 3 | PatchCore + context VLM, LOCO | fusion_loco | 0.8333 | 0.8696 |  |  | Stage 13 leave-one-category-out fusion |
| 4 | full-image VLM | vlm_branch | 0.7533 |  |  |  | Stage 11 full-image baseline |
| 5 | PatchCore | classical_detector | 0.7167 |  |  |  | Stage 11 reference |
| 6 | EfficientAD pilot | modern_detector_baseline | 0.6567 | 0.7742 | 0.7703 | 0.4649 | model_size=small, max_epochs=20, train_batch_size=1 |
| 7 | WinCLIP sensitivity best | external_vlm_baseline | 0.6300 | 0.8571 | 0.8511 | 0.2399 | class_name=jelly, k_shot=1, scales=(1, 2, 3) |
| 8 | WinCLIP zero-shot | external_vlm_baseline | 0.4267 |  | 0.5308 | 0.0063 | Stage 14-C2 zero-shot pilot |

## 3. Main Observations

1. EfficientAD pilot reaches image AUROC `0.6567` on fruit_jelly.
2. EfficientAD is higher than the best tested WinCLIP fruit_jelly setting `0.6300`.
3. EfficientAD is lower than PatchCore `0.7167`.
4. EfficientAD is lower than context-aware VLM `0.8567` and PatchCore+context LOCO fusion `0.8333`.
5. EfficientAD has pixel F1 `0.4649`, which is useful for localization discussion but not directly comparable to image-only VLM scores.

## 4. Safe Interpretation

The EfficientAD fruit_jelly pilot verifies that the modern detector baseline can run in the current environment.

However, because this is a 20-epoch one-category pilot, it should not yet be used as the final EfficientAD baseline.

The next stage should run a fixed EfficientAD protocol on the four AD2 primary categories, or first increase the fruit_jelly training budget if the goal is to obtain a stronger formal detector baseline.

## 5. Output

- Comparison CSV: `results/stage15_modern_detector_baselines/stage15_c_fruit_jelly_modern_baseline_comparison.csv`
- Report: `docs/stage15_modern_detector_baselines/stage15_c_fruit_jelly_modern_baseline_comparison.md`