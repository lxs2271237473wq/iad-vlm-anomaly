# Stage 7 Generalization Final Summary

## 1. Purpose

This document summarizes the current Stage 7 generalization evidence.

The goal is to show that anomaly-crop reasoning is not limited to a single dataset or a single anomaly localization backbone.

## 2. Paper-ready Generalization Table

| Dataset | Backbone | Task | Full-image Metric | Crop Metric | Improvement | Note |
|---|---|---|---:|---:|---:|---|
| MVTec AD | PatchCore | Defect-type reasoning | 0.2850 Top-1 | 0.3388 Top-1 | +0.0538 | Weak-category full-test setting; crop_topk_ensemble improves defect-type reasoning. |
| VisA | PatchCore | Binary normal/anomaly reasoning | 0.5950 AUROC | 0.8844 AUROC | +0.2894 | Cross-dataset validation using PatchCore candidate crops. |
| VisA | FastFlow | Binary normal/anomaly reasoning | 0.5950 AUROC | 0.9222 AUROC | +0.3272 | Backbone-level validation using FastFlow candidate crops. |

## 3. Backbone-level Comparison on VisA

| Backbone | Image AUROC | Pixel AUROC | Candidate Coverage | Full AUROC | Crop AUROC | Delta AUROC | Crop F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| PatchCore | 0.9138 | 0.8971 | 1.0000 | 0.5950 | 0.8844 | +0.2894 | 0.8783 |
| FastFlow | 0.8934 | 0.9511 | 0.9992 | 0.5950 | 0.9222 | +0.3272 | 0.9042 |

## 4. Main Conclusion

```text
The localization-to-reasoning pipeline generalizes across both datasets and anomaly backbones.
On VisA, both PatchCore-guided and FastFlow-guided candidate crops substantially improve CLIP-based anomaly reasoning over full-image prompting.
```

## 5. Current Paper Claim

The current evidence supports the following claim:

```text
Classical anomaly localization can serve as an effective bridge between industrial anomaly detectors and visual-language reasoning models.
```

## 6. Next Experimental Option

The next optional experiment is to add one more lightweight backbone, such as STFPM or PaDiM, only if additional backbone diversity is needed.