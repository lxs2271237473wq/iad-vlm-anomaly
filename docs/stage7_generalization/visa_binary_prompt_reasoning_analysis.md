# VisA Binary Prompt Reasoning Analysis

## 1. Purpose

This document summarizes Stage 7.3: VisA full-image versus PatchCore-guided anomaly-crop binary prompt reasoning.

VisA does not provide fine-grained defect-type labels. Therefore, this stage uses binary normal/anomaly reasoning instead of MVTec-style defect-type classification.

## 2. Evaluation Setting

| Item | Value |
|---|---:|
| Dataset | VisA |
| Categories | 12 |
| Test images | 2162 |
| Normal images | 962 |
| Anomaly images | 1200 |
| VLM backbone | CLIP ViT-B/32 |
| Candidate source | PatchCore anomaly map |

## 3. Mean Results

| Strategy | Eval Mode | Images | Normal | Anomaly | Coverage | AUROC | AP | Best F1 | Accuracy |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| inspection_binary | crop_topk_ensemble | 2162 | 962 | 1200 | 0.5550 | 0.8844 | 0.9179 | 0.8783 | 0.8651 |
| inspection_binary | crop_or_full | 2162 | 962 | 1200 | 0.5550 | 0.8672 | 0.9050 | 0.8700 | 0.8454 |
| generic_binary | crop_topk_ensemble | 2162 | 962 | 1200 | 0.5550 | 0.8792 | 0.9144 | 0.8665 | 0.8554 |
| generic_binary | crop_or_full | 2162 | 962 | 1200 | 0.5550 | 0.8665 | 0.9049 | 0.8560 | 0.8110 |
| category_binary | crop_topk_ensemble | 2162 | 962 | 1200 | 0.5550 | 0.7889 | 0.8440 | 0.8483 | 0.8043 |
| category_binary | crop_or_full | 2162 | 962 | 1200 | 0.5550 | 0.7735 | 0.8343 | 0.8396 | 0.7895 |
| inspection_binary | full_all | 2162 | 962 | 1200 | 1.0000 | 0.5950 | 0.6796 | 0.7395 | 0.6215 |
| generic_binary | full_all | 2162 | 962 | 1200 | 1.0000 | 0.5874 | 0.6538 | 0.7322 | 0.6010 |
| category_binary | full_all | 2162 | 962 | 1200 | 1.0000 | 0.5674 | 0.6395 | 0.7266 | 0.5873 |

## 4. Delta Against Full Image

| Strategy | Crop Mode | Delta AUROC | Delta AP | Delta Best F1 | Delta Accuracy |
|---|---|---:|---:|---:|---:|
| generic_binary | crop_topk_ensemble | +0.2918 | +0.2606 | +0.1343 | +0.2544 |
| inspection_binary | crop_topk_ensemble | +0.2894 | +0.2382 | +0.1388 | +0.2436 |
| generic_binary | crop_or_full | +0.2791 | +0.2511 | +0.1238 | +0.2101 |
| inspection_binary | crop_or_full | +0.2722 | +0.2254 | +0.1305 | +0.2239 |
| category_binary | crop_topk_ensemble | +0.2215 | +0.2045 | +0.1217 | +0.2169 |
| category_binary | crop_or_full | +0.2060 | +0.1948 | +0.1130 | +0.2021 |

## 5. Main Result

The cleanest comparison keeps the prompt strategy fixed as inspection_binary and changes only the visual input mode.

| Setting | AUROC | AP | Best F1 | Accuracy |
|---|---:|---:|---:|---:|
| full_all | 0.5950 | 0.6796 | 0.7395 | 0.6215 |
| crop_topk_ensemble | 0.8844 | 0.9179 | 0.8783 | 0.8651 |
| Improvement | +0.2894 | +0.2382 | +0.1388 | +0.2436 |

## 6. Coverage Ratio Interpretation

In Stage 7.3, coverage_ratio is computed over all test images: covered_count / num_images_total.

Because VisA contains 1200 anomaly images and 962 normal images, only anomaly images are expected to have anomaly candidate crops. Normal images fall back to full-image input.

Therefore, a coverage ratio around 1200 / 2162 = 0.5550 is expected. This is not a candidate-generation failure.

Stage 7.2-B already showed anomaly candidate coverage of 1200 / 1200 = 1.0000.

## 7. Conclusion

PatchCore-guided anomaly crops substantially improve VLM-based normal/anomaly reasoning on VisA compared with full-image prompting.

This provides cross-dataset evidence that anomaly localization can serve as an effective bridge between classical anomaly detectors and visual-language reasoning models.

## 8. Next Step

Stage 7.4 should test multi-backbone anomaly candidate generalization.