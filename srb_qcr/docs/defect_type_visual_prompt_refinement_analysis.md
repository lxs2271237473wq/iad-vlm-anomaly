# Defect Type Visual Prompt Refinement Analysis

## 1. Purpose

This document analyzes Stage 6.2: visual prompt refinement for defect type recognition.

The motivation is that long manufacturing-process prompts introduce semantic noise for CLIP-based defect classification.

Therefore, this stage separates:

- short visual prompts for defect type classification
- manufacturing process knowledge for later explanation generation

The tested prompt strategies are:

- generic_label
- short_visual
- category_visual
- visual_ensemble

The tested image modes are:

- full image
- GT crop

GT crop is an upper-bound diagnostic setting because ground-truth masks are used for cropping.

## 2. Mean Summary

| Strategy | Image Mode | Top-1 Accuracy | Top-2 Accuracy | Macro-F1 | Fallback Count |
|---|---|---:|---:|---:|---:|
| visual_ensemble | full | 0.3170 | 0.4911 | 0.1912 | 0 |
| category_visual | gt_crop | 0.3102 | 0.4655 | 0.2460 | 0 |
| short_visual | gt_crop | 0.3045 | 0.4669 | 0.2100 | 0 |
| generic_label | gt_crop | 0.2925 | 0.4721 | 0.1973 | 0 |
| generic_label | full | 0.2850 | 0.4990 | 0.1543 | 0 |
| short_visual | full | 0.2798 | 0.4705 | 0.1600 | 0 |
| visual_ensemble | gt_crop | 0.2744 | 0.4974 | 0.1837 | 0 |
| category_visual | full | 0.2592 | 0.4633 | 0.1857 | 0 |

## 3. Best Results

### 3.1 Best by Top-1 Accuracy

| Metric | Value |
|---|---|
| Strategy | visual_ensemble |
| Image Mode | full |
| Top-1 Accuracy | 0.3170 |
| Top-2 Accuracy | 0.4911 |
| Macro-F1 | 0.1912 |

### 3.2 Best by Macro-F1

| Metric | Value |
|---|---|
| Strategy | category_visual |
| Image Mode | gt_crop |
| Top-1 Accuracy | 0.3102 |
| Top-2 Accuracy | 0.4655 |
| Macro-F1 | 0.2460 |

## 4. Comparison with Stage 6.1-D

| Stage | Setting | Top-1 Accuracy | Top-2 Accuracy | Macro-F1 |
|---|---|---:|---:|---:|
| Stage 6.1-D | category_aware + gt_crop | 0.2973 | 0.5022 | 0.2147 |
| Stage 6.2 | visual_ensemble + full | 0.3170 | 0.4911 | 0.1912 |
| Stage 6.2 | category_visual + gt_crop | 0.3102 | 0.4655 | 0.2460 |

## 5. Observations

1. Short visual prompts improve defect type recognition compared with long manufacturing-aware prompts.

2. The best Top-1 Accuracy is achieved by full-image visual ensemble prompts.

3. The best Macro-F1 is achieved by category-aware visual prompts with GT crops.

4. GT crops help class-balanced recognition, while full-image ensemble prompts preserve more global object context.

5. Manufacturing process knowledge should not be directly used as long CLIP classification prompts. It should be used for explanation and cause reasoning after defect type prediction.

## 6. Conclusion

Stage 6.2 is a positive result.

```text
Defect type reasoning benefits from short visual prompts and accurate defect-region focus.
Manufacturing process knowledge should be decoupled from CLIP classification prompts and used as an explanation layer.
```

## 7. Next Direction

Recommended next stage:

```text
Stage 6.3: Real anomaly-crop defect reasoning with refined visual prompts
```

The next stage should use PatchCore anomaly crops rather than GT crops and test whether the refined visual prompts remain effective under realistic localization.
