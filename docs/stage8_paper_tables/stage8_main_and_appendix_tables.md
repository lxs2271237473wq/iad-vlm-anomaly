# Stage 8-B Main and Appendix Tables

## 1. Purpose

This document splits the completed experimental evidence into main-paper tables and appendix tables.

## 2. Recommended Main-paper Tables

### Table 1. Generalization of localization-guided reasoning

| Dataset | Backbone | Task | Setting | Full image | Crop | Improvement | Claim |
|---|---|---|---|---:|---:|---:|---|
| MVTec AD | PatchCore | Defect-type reasoning | weak categories, fair full-test | 0.2850 Top-1 | 0.3388 Top-1 | +0.0538 | PatchCore-guided crops improve defect-type reasoning. |
| VisA | PatchCore | Binary normal/anomaly reasoning | 12 categories, 2162 test images | 0.5950 AUROC | 0.8844 AUROC | +0.2894 | PatchCore candidates generalize crop reasoning to VisA. |
| VisA | FastFlow | Binary normal/anomaly reasoning | 12 categories, 2162 test images | 0.5950 AUROC | 0.9222 AUROC | +0.3272 | Crop reasoning also works with a non-PatchCore backbone. |

### Table 2. Backbone-level reasoning comparison on VisA

| Backbone | Candidate Coverage | Full AUROC | Crop AUROC | ΔAUROC | Full F1 | Crop F1 | ΔF1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| PatchCore | 1.0000 | 0.5950 | 0.8844 | +0.2894 | 0.7395 | 0.8783 | +0.1388 |
| FastFlow | 0.9992 | 0.5950 | 0.9222 | +0.3272 | 0.7395 | 0.9042 | +0.1648 |

## 3. Recommended Appendix Table

### Table A1. VisA detector and candidate generation details

| Dataset | Backbone | Test Images | Anomaly Images | Image AUROC | Image F1 | Pixel AUROC | Pixel F1 | Candidate Coverage | Candidate Rows |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| VisA | PatchCore | 2162 | 1200 | 0.9138 | 0.8843 | 0.8971 | 0.1814 | 1.0000 | 1544 |
| VisA | FastFlow | 2162 | 1200 | 0.8934 | 0.8630 | 0.9511 | 0.2573 | 0.9992 | 2354 |

## 4. Main-paper Placement Recommendation

| Section | Table | Reason |
|---|---|---|
| Experiments / Main Results | Table 1 | Shows dataset and backbone generalization in one compact table. |
| Ablation / Generalization | Table 2 | Shows that the crop-reasoning gain is not tied to PatchCore. |
| Appendix | Table A1 | Gives detector and candidate-generation details without overloading the main paper. |

## 5. Reviewer-facing Interpretation

The main paper should emphasize that localization-guided crops consistently improve VLM reasoning over full-image prompting.

The appendix should contain detector quality and candidate coverage details, because these explain why crop reasoning succeeds or fails without distracting from the main contribution.