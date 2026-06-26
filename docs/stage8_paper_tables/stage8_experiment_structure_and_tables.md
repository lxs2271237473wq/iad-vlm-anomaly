# Stage 8 Experiment Structure and Paper Tables

## 1. Purpose

This document reorganizes the completed experiments into a paper-ready structure.

The goal is to make the experimental evidence clear for CCF-B style review: dataset generalization, backbone generalization, fair comparison, and ablation logic.

## 2. Main Experimental Claims

| Claim | Evidence |
|---|---|
| Localization helps reasoning | MVTec AD and VisA both show crop-based reasoning improves over full-image prompting. |
| Dataset-level generalization | The effect holds on MVTec AD and VisA. |
| Backbone-level generalization | The effect holds with PatchCore and FastFlow candidates. |
| Full-image prompting is insufficient | On VisA, full-image CLIP reasoning is much weaker than crop-guided reasoning. |

## 3. Paper-ready Main Result Table

| Stage | Dataset | Backbone | Task | Setting | Full-image Score | Crop Score | Improvement | Main Claim |
|---|---|---|---|---|---:|---:|---:|---|
| Stage 6 | MVTec AD | PatchCore | Defect-type reasoning | weak categories, fair full-test | 0.2850 Top-1 | 0.3388 Top-1 | +0.0538 | PatchCore-guided crops improve defect-type reasoning. |
| Stage 7.3 | VisA | PatchCore | Binary normal/anomaly reasoning | 12 categories, 2162 test images | 0.5950 AUROC | 0.8844 AUROC | +0.2894 | PatchCore candidates generalize crop reasoning to VisA. |
| Stage 7.4 | VisA | FastFlow | Binary normal/anomaly reasoning | 12 categories, 2162 test images | 0.5950 AUROC | 0.9222 AUROC | +0.3272 | Crop reasoning also works with a non-PatchCore backbone. |

## 4. VisA Backbone-level Detector and Reasoning Table

| Backbone | Image AUROC | Image F1 | Pixel AUROC | Pixel F1 | Candidate Coverage | Full AUROC | Crop AUROC | Delta AUROC | Crop F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| PatchCore | 0.9138 | 0.8843 | 0.8971 | 0.1814 | 1.0000 | 0.5950 | 0.8844 | +0.2894 | 0.8783 |
| FastFlow | 0.8934 | 0.8630 | 0.9511 | 0.2573 | 0.9992 | 0.5950 | 0.9222 | +0.3272 | 0.9042 |

## 5. Recommended Paper Experiment Layout

### 5.1 Baseline anomaly localization

Report PatchCore and FastFlow detection results on VisA, including image-level metrics, pixel-level metrics, and candidate coverage.

### 5.2 Full-image versus anomaly-crop reasoning

Compare full_all, crop_or_full, and crop_topk_ensemble under the same prompt strategy.

### 5.3 Dataset generalization

Use MVTec AD and VisA to show that localization-guided reasoning is not dataset-specific.

### 5.4 Backbone generalization

Use PatchCore and FastFlow on VisA to show that the proposed reasoning pipeline is not tied to one anomaly detector.

### 5.5 Limitation analysis

Discuss that the method depends on the quality of candidate localization. Very small, diffuse, or low-contrast anomalies may still reduce crop quality.

## 6. What Not To Claim

| Avoided Claim | Reason |
|---|---|
| The method solves pixel-perfect segmentation | Pixel F1 is still limited, especially on difficult categories. |
| The method discovers all defect causes | Current experiments validate reasoning from visual anomaly crops, not full causal discovery. |
| Full-image prompting is always useless | Full-image prompting is a baseline; the claim is that crop guidance is substantially stronger in these settings. |
| GT crop results are directly comparable | GT crops are upper-bound diagnostics and must not be mixed with realistic candidate crops. |

## 7. Current Paper-level Conclusion

Classical anomaly localization provides a practical bridge between industrial anomaly detectors and visual-language reasoning models. Across MVTec AD and VisA, and across PatchCore and FastFlow candidate generators, anomaly-crop prompting consistently improves reasoning over full-image prompting.

## 8. Next Stage

Stage 8-B should generate the final paper table files and decide which results belong in the main paper and which belong in appendix.