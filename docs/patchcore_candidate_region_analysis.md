# PatchCore Candidate Region Analysis on MVTec AD

## 1. Purpose

This document analyzes whether PatchCore anomaly maps can provide useful candidate regions for later SAM/SAM2 prompt-based mask refinement.

Pipeline:

```text
PatchCore anomaly map -> thresholding -> connected components -> candidate boxes / points / coarse masks
```

Selected weak categories:

- grid
- screw
- leather
- wood

## 2. Summary Table

| Category | Images | Candidate Rows | No-candidate Images | Avg Candidates/Image | Top-1 GT IoU | Top-1 GT F1 | Best GT IoU | Best GT F1 | Hit@IoU0.1 | Hit@IoU0.3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| grid | 28 | 36 | 0 | 1.29 | 0.3149 | 0.4559 | 0.3179 | 0.4603 | 1.0000 | 0.3929 |
| screw | 60 | 83 | 0 | 1.38 | 0.2736 | 0.3841 | 0.3158 | 0.4479 | 0.8667 | 0.5333 |
| leather | 46 | 46 | 0 | 1.00 | 0.3444 | 0.5024 | 0.3444 | 0.5024 | 1.0000 | 0.6304 |
| wood | 30 | 57 | 0 | 1.90 | 0.2679 | 0.3925 | 0.3066 | 0.4455 | 0.9333 | 0.4000 |

## 3. Per-category Notes

### grid

| Item | Value |
|---|---:|
| Number of abnormal images | 28 |
| Candidate rows | 36 |
| No-candidate images | 0 |
| Average candidates per image | 1.29 |
| Top-1 candidate mean GT IoU | 0.3149 |
| Top-1 candidate mean GT F1 | 0.4559 |
| Best candidate mean GT IoU | 0.3179 |
| Best candidate mean GT F1 | 0.4603 |
| Hit rate @ IoU >= 0.1 | 1.0000 |
| Hit rate @ IoU >= 0.3 | 0.3929 |

### screw

| Item | Value |
|---|---:|
| Number of abnormal images | 60 |
| Candidate rows | 83 |
| No-candidate images | 0 |
| Average candidates per image | 1.38 |
| Top-1 candidate mean GT IoU | 0.2736 |
| Top-1 candidate mean GT F1 | 0.3841 |
| Best candidate mean GT IoU | 0.3158 |
| Best candidate mean GT F1 | 0.4479 |
| Hit rate @ IoU >= 0.1 | 0.8667 |
| Hit rate @ IoU >= 0.3 | 0.5333 |

### leather

| Item | Value |
|---|---:|
| Number of abnormal images | 46 |
| Candidate rows | 46 |
| No-candidate images | 0 |
| Average candidates per image | 1.00 |
| Top-1 candidate mean GT IoU | 0.3444 |
| Top-1 candidate mean GT F1 | 0.5024 |
| Best candidate mean GT IoU | 0.3444 |
| Best candidate mean GT F1 | 0.5024 |
| Hit rate @ IoU >= 0.1 | 1.0000 |
| Hit rate @ IoU >= 0.3 | 0.6304 |

### wood

| Item | Value |
|---|---:|
| Number of abnormal images | 30 |
| Candidate rows | 57 |
| No-candidate images | 0 |
| Average candidates per image | 1.90 |
| Top-1 candidate mean GT IoU | 0.2679 |
| Top-1 candidate mean GT F1 | 0.3925 |
| Best candidate mean GT IoU | 0.3066 |
| Best candidate mean GT F1 | 0.4455 |
| Hit rate @ IoU >= 0.1 | 0.9333 |
| Hit rate @ IoU >= 0.3 | 0.4000 |

## 4. Interpretation

1. If `no_candidate_images` is close to zero, PatchCore anomaly maps can at least provide candidate prompts for most abnormal images.

2. If `Top-1 GT IoU` and `Top-1 GT F1` are low, the largest/highest-score candidate is not always well aligned with the ground-truth defect region.

3. If `Best GT IoU` is much higher than `Top-1 GT IoU`, region ranking should be improved before sending prompts to SAM/SAM2.

4. If both `Top-1` and `Best` values are low, the problem is not only segmentation refinement. The anomaly map itself needs enhancement.

## 5. Implication for SAM/SAM2 Refinement

The next stage should use these candidate regions as prompts:

- box prompt: `[x1, y1, x2, y2]`
- point prompt: candidate center `(cx, cy)`
- coarse mask prompt: thresholded connected component

However, SAM/SAM2 should not be used blindly. Candidate quality must be checked first because SAM can refine object boundaries, but it cannot recover a defect region if the prompt is far from the true defect.

## 6. Next Stage

Recommended next stage:

```text
Stage 2.3: SAM/SAM2 prompt feasibility test
```

Start with the categories:

- grid
- screw
- leather
- wood
