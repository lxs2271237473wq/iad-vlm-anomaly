# VisA PatchCore Baseline and Candidate Generation Analysis

## 1. Purpose

This document summarizes Stage 7.2-B: VisA + PatchCore baseline and anomaly candidate generation.

The goal is to test whether the current anomaly-crop pipeline can generalize beyond MVTec AD.

VisA is used as the first cross-dataset validation target.

## 2. Overall Result

| Metric | Value |
|---|---:|
| Test Images | 2162 |
| Test Normal Images | 962 |
| Test Anomaly Images | 1200 |
| Image AUROC | 0.9138 |
| Image AP | 0.9258 |
| Image Best F1 | 0.8843 |
| Pixel AUROC | 0.8971 |
| Pixel AP | 0.1059 |
| Pixel Best F1 | 0.1814 |

## 3. Candidate Coverage

| Metric | Value |
|---|---:|
| Anomaly Images | 1200 |
| Covered Anomaly Images | 1200 |
| Coverage Ratio | 1.0000 |
| Candidate Rows | 1544 |

## 4. Category-level Baseline Summary

| Category | Test Images | Normal | Anomaly | Image AUROC | Image F1 | Pixel AUROC | Pixel F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| candle | 200 | 100 | 100 | 0.9840 | 0.9453 | 0.9687 | 0.1756 |
| capsules | 160 | 60 | 100 | 0.6995 | 0.7778 | 0.8909 | 0.1370 |
| cashew | 150 | 50 | 100 | 0.9660 | 0.9314 | 0.7596 | 0.1444 |
| chewinggum | 150 | 50 | 100 | 0.9918 | 0.9746 | 0.9664 | 0.3928 |
| fryum | 150 | 50 | 100 | 0.9544 | 0.9149 | 0.7141 | 0.1353 |
| macaroni1 | 200 | 100 | 100 | 0.8661 | 0.8131 | 0.9486 | 0.0844 |
| macaroni2 | 200 | 100 | 100 | 0.7119 | 0.7200 | 0.9641 | 0.0570 |
| pcb1 | 200 | 100 | 100 | 0.9436 | 0.8772 | 0.9274 | 0.2204 |
| pcb2 | 200 | 100 | 100 | 0.9304 | 0.8638 | 0.9149 | 0.1466 |
| pcb3 | 201 | 101 | 100 | 0.9331 | 0.8571 | 0.9641 | 0.2221 |
| pcb4 | 201 | 101 | 100 | 0.9906 | 0.9561 | 0.8710 | 0.2457 |
| pipe_fryum | 150 | 50 | 100 | 0.9936 | 0.9802 | 0.8757 | 0.2156 |
| MEAN | 2162 | 962 | 1200 | 0.9138 | 0.8843 | 0.8971 | 0.1814 |

## 5. Weak Image-level Categories

| Category | Image AUROC | Image F1 |
|---|---:|---:|
| capsules | 0.6995 | 0.7778 |
| macaroni2 | 0.7119 | 0.7200 |
| macaroni1 | 0.8661 | 0.8131 |

## 6. Weak Pixel-level Categories

| Category | Pixel AUROC | Pixel F1 |
|---|---:|---:|
| macaroni2 | 0.9641 | 0.0570 |
| macaroni1 | 0.9486 | 0.0844 |
| fryum | 0.7141 | 0.1353 |

## 7. Observations

1. PatchCore achieves strong image-level performance on VisA, with mean Image AUROC above 0.91.

2. Pixel-level F1 remains low, which is consistent with the earlier MVTec AD observation that mask refinement alone is difficult.

3. Candidate region coverage reaches 1200 / 1200 anomaly images, so VisA is suitable for full-test anomaly crop reasoning.

4. The weakest image-level categories are capsules and macaroni2.

5. The weakest pixel-level categories include macaroni2, macaroni1, and capsules.

## 8. Conclusion

Stage 7.2-B supports cross-dataset generalization at the anomaly localization and candidate generation level.

```text
PatchCore on VisA provides full anomaly candidate coverage, while pixel-level mask quality remains limited.
This makes VisA suitable for testing whether anomaly crops can improve downstream reasoning beyond MVTec AD.
```

## 9. Next Direction

Recommended next stage:

```text
Stage 7.3: VisA full-image vs anomaly-crop reasoning
```

Because VisA only provides normal/anomaly labels rather than fine-grained defect type labels, the next reasoning task should be adapted carefully.

The first VisA reasoning experiment should compare full image versus anomaly crop under a binary normal/anomaly visual-prompt setting, not MVTec-style defect-type classification.
