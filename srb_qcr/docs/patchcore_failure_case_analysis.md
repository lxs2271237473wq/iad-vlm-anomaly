# PatchCore Failure Case Analysis on MVTec AD

## 1. Purpose

This document analyzes the worst pixel-level localization cases of PatchCore on selected weak MVTec AD categories.

The selected categories are:

- grid
- screw
- leather
- wood

These categories were selected because they show relatively low Pixel F1Score in the full PatchCore baseline.

## 2. Failure Case Summary

### grid

| Item | Value |
|---|---:|
| Number of exported cases | 8 |
| Cases with Pixel F1 = 0 | 3 |
| Cases with Predicted Area = 0 | 3 |
| Mean Pixel F1 | 0.0997 |
| Mean GT Area | 322.6 |
| Mean Predicted Area | 432.1 |

| Rank | Image Path | GT Area | Pred Area | Pixel F1 | Visual Path |
|---:|---|---:|---:|---:|---|
| 1 | `/root/private_data/iad-vlm-anomaly/datasets/MVTecAD/grid/test/glue/001.png` | 316 | 0 | 0.0000 | `results/visualizations/patchcore/MVTecAD/grid/failures/rank_01_f1_0.0000.png` |
| 2 | `/root/private_data/iad-vlm-anomaly/datasets/MVTecAD/grid/test/glue/007.png` | 323 | 0 | 0.0000 | `results/visualizations/patchcore/MVTecAD/grid/failures/rank_02_f1_0.0000.png` |
| 3 | `/root/private_data/iad-vlm-anomaly/datasets/MVTecAD/grid/test/metal_contamination/009.png` | 263 | 0 | 0.0000 | `results/visualizations/patchcore/MVTecAD/grid/failures/rank_03_f1_0.0000.png` |
| 4 | `/root/private_data/iad-vlm-anomaly/datasets/MVTecAD/grid/test/metal_contamination/007.png` | 124 | 22 | 0.0411 | `results/visualizations/patchcore/MVTecAD/grid/failures/rank_04_f1_0.0411.png` |
| 5 | `/root/private_data/iad-vlm-anomaly/datasets/MVTecAD/grid/test/bent/000.png` | 344 | 64 | 0.0588 | `results/visualizations/patchcore/MVTecAD/grid/failures/rank_05_f1_0.0588.png` |
| 6 | `/root/private_data/iad-vlm-anomaly/datasets/MVTecAD/grid/test/broken/004.png` | 288 | 1975 | 0.1838 | `results/visualizations/patchcore/MVTecAD/grid/failures/rank_06_f1_0.1838.png` |
| 7 | `/root/private_data/iad-vlm-anomaly/datasets/MVTecAD/grid/test/thread/006.png` | 834 | 814 | 0.2488 | `results/visualizations/patchcore/MVTecAD/grid/failures/rank_07_f1_0.2488.png` |
| 8 | `/root/private_data/iad-vlm-anomaly/datasets/MVTecAD/grid/test/broken/002.png` | 89 | 582 | 0.2653 | `results/visualizations/patchcore/MVTecAD/grid/failures/rank_08_f1_0.2653.png` |

### screw

| Item | Value |
|---|---:|
| Number of exported cases | 8 |
| Cases with Pixel F1 = 0 | 8 |
| Cases with Predicted Area = 0 | 7 |
| Mean Pixel F1 | 0.0000 |
| Mean GT Area | 178.6 |
| Mean Predicted Area | 16.8 |

| Rank | Image Path | GT Area | Pred Area | Pixel F1 | Visual Path |
|---:|---|---:|---:|---:|---|
| 1 | `/root/private_data/iad-vlm-anomaly/datasets/MVTecAD/screw/test/manipulated_front/009.png` | 421 | 0 | 0.0000 | `results/visualizations/patchcore/MVTecAD/screw/failures/rank_01_f1_0.0000.png` |
| 2 | `/root/private_data/iad-vlm-anomaly/datasets/MVTecAD/screw/test/manipulated_front/010.png` | 212 | 0 | 0.0000 | `results/visualizations/patchcore/MVTecAD/screw/failures/rank_02_f1_0.0000.png` |
| 3 | `/root/private_data/iad-vlm-anomaly/datasets/MVTecAD/screw/test/manipulated_front/011.png` | 191 | 134 | 0.0000 | `results/visualizations/patchcore/MVTecAD/screw/failures/rank_03_f1_0.0000.png` |
| 4 | `/root/private_data/iad-vlm-anomaly/datasets/MVTecAD/screw/test/manipulated_front/012.png` | 63 | 0 | 0.0000 | `results/visualizations/patchcore/MVTecAD/screw/failures/rank_04_f1_0.0000.png` |
| 5 | `/root/private_data/iad-vlm-anomaly/datasets/MVTecAD/screw/test/manipulated_front/013.png` | 194 | 0 | 0.0000 | `results/visualizations/patchcore/MVTecAD/screw/failures/rank_05_f1_0.0000.png` |
| 6 | `/root/private_data/iad-vlm-anomaly/datasets/MVTecAD/screw/test/manipulated_front/020.png` | 85 | 0 | 0.0000 | `results/visualizations/patchcore/MVTecAD/screw/failures/rank_06_f1_0.0000.png` |
| 7 | `/root/private_data/iad-vlm-anomaly/datasets/MVTecAD/screw/test/manipulated_front/022.png` | 94 | 0 | 0.0000 | `results/visualizations/patchcore/MVTecAD/screw/failures/rank_07_f1_0.0000.png` |
| 8 | `/root/private_data/iad-vlm-anomaly/datasets/MVTecAD/screw/test/scratch_head/007.png` | 169 | 0 | 0.0000 | `results/visualizations/patchcore/MVTecAD/screw/failures/rank_08_f1_0.0000.png` |

### leather

| Item | Value |
|---|---:|
| Number of exported cases | 8 |
| Cases with Pixel F1 = 0 | 8 |
| Cases with Predicted Area = 0 | 8 |
| Mean Pixel F1 | 0.0000 |
| Mean GT Area | 178.4 |
| Mean Predicted Area | 0.0 |

| Rank | Image Path | GT Area | Pred Area | Pixel F1 | Visual Path |
|---:|---|---:|---:|---:|---|
| 1 | `/root/private_data/iad-vlm-anomaly/datasets/MVTecAD/leather/test/color/002.png` | 363 | 0 | 0.0000 | `results/visualizations/patchcore/MVTecAD/leather/failures/rank_01_f1_0.0000.png` |
| 2 | `/root/private_data/iad-vlm-anomaly/datasets/MVTecAD/leather/test/color/004.png` | 446 | 0 | 0.0000 | `results/visualizations/patchcore/MVTecAD/leather/failures/rank_02_f1_0.0000.png` |
| 3 | `/root/private_data/iad-vlm-anomaly/datasets/MVTecAD/leather/test/color/005.png` | 125 | 0 | 0.0000 | `results/visualizations/patchcore/MVTecAD/leather/failures/rank_03_f1_0.0000.png` |
| 4 | `/root/private_data/iad-vlm-anomaly/datasets/MVTecAD/leather/test/color/006.png` | 59 | 0 | 0.0000 | `results/visualizations/patchcore/MVTecAD/leather/failures/rank_04_f1_0.0000.png` |
| 5 | `/root/private_data/iad-vlm-anomaly/datasets/MVTecAD/leather/test/color/014.png` | 130 | 0 | 0.0000 | `results/visualizations/patchcore/MVTecAD/leather/failures/rank_05_f1_0.0000.png` |
| 6 | `/root/private_data/iad-vlm-anomaly/datasets/MVTecAD/leather/test/color/018.png` | 85 | 0 | 0.0000 | `results/visualizations/patchcore/MVTecAD/leather/failures/rank_06_f1_0.0000.png` |
| 7 | `/root/private_data/iad-vlm-anomaly/datasets/MVTecAD/leather/test/cut/009.png` | 105 | 0 | 0.0000 | `results/visualizations/patchcore/MVTecAD/leather/failures/rank_07_f1_0.0000.png` |
| 8 | `/root/private_data/iad-vlm-anomaly/datasets/MVTecAD/leather/test/cut/010.png` | 114 | 0 | 0.0000 | `results/visualizations/patchcore/MVTecAD/leather/failures/rank_08_f1_0.0000.png` |

### wood

| Item | Value |
|---|---:|
| Number of exported cases | 8 |
| Cases with Pixel F1 = 0 | 2 |
| Cases with Predicted Area = 0 | 2 |
| Mean Pixel F1 | 0.1740 |
| Mean GT Area | 1076.6 |
| Mean Predicted Area | 3194.2 |

| Rank | Image Path | GT Area | Pred Area | Pixel F1 | Visual Path |
|---:|---|---:|---:|---:|---|
| 1 | `/root/private_data/iad-vlm-anomaly/datasets/MVTecAD/wood/test/hole/005.png` | 792 | 0 | 0.0000 | `results/visualizations/patchcore/MVTecAD/wood/failures/rank_01_f1_0.0000.png` |
| 2 | `/root/private_data/iad-vlm-anomaly/datasets/MVTecAD/wood/test/scratch/001.png` | 502 | 0 | 0.0000 | `results/visualizations/patchcore/MVTecAD/wood/failures/rank_02_f1_0.0000.png` |
| 3 | `/root/private_data/iad-vlm-anomaly/datasets/MVTecAD/wood/test/combined/002.png` | 233 | 2320 | 0.1825 | `results/visualizations/patchcore/MVTecAD/wood/failures/rank_03_f1_0.1825.png` |
| 4 | `/root/private_data/iad-vlm-anomaly/datasets/MVTecAD/wood/test/scratch/010.png` | 1782 | 4524 | 0.1916 | `results/visualizations/patchcore/MVTecAD/wood/failures/rank_04_f1_0.1916.png` |
| 5 | `/root/private_data/iad-vlm-anomaly/datasets/MVTecAD/wood/test/scratch/020.png` | 1528 | 1739 | 0.2143 | `results/visualizations/patchcore/MVTecAD/wood/failures/rank_05_f1_0.2143.png` |
| 6 | `/root/private_data/iad-vlm-anomaly/datasets/MVTecAD/wood/test/scratch/019.png` | 881 | 4667 | 0.2570 | `results/visualizations/patchcore/MVTecAD/wood/failures/rank_06_f1_0.2570.png` |
| 7 | `/root/private_data/iad-vlm-anomaly/datasets/MVTecAD/wood/test/scratch/008.png` | 1491 | 8869 | 0.2712 | `results/visualizations/patchcore/MVTecAD/wood/failures/rank_07_f1_0.2712.png` |
| 8 | `/root/private_data/iad-vlm-anomaly/datasets/MVTecAD/wood/test/scratch/009.png` | 1404 | 3435 | 0.2757 | `results/visualizations/patchcore/MVTecAD/wood/failures/rank_08_f1_0.2757.png` |

## 3. Main Observations

1. PatchCore image-level anomaly detection is strong, but pixel-level localization still fails on fine-grained and texture-heavy defects.

2. A common failure mode is under-segmentation: the image can receive a high anomaly score, but the final predicted binary mask has zero or very small area.

3. For texture categories such as `grid`, `leather`, and `wood`, the model is sensitive to local normal texture variation. This can cause either missed masks or over-expanded anomaly regions.

4. For object-structure categories such as `screw`, the anomaly is often small and localized. PatchCore can detect that the image is abnormal, but the final mask thresholding does not reliably isolate the abnormal region.

## 4. Implication for the Next Method

The next method should focus on pixel-level localization rather than only image-level classification.

Potential improvement directions:

1. Region proposal refinement based on anomaly maps.
2. Adaptive mask thresholding for different defect types.
3. SAM/SAM2-based mask refinement using PatchCore anomaly regions as prompts.
4. VLM/CLIP-based defect semantic guidance, especially for defect types such as scratch, cut, color, contamination, and deformation.
5. Manufacturing-aware defect knowledge to distinguish real defects from normal texture variations.
