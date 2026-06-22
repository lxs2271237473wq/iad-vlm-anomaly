# PatchCore Baseline Analysis on MVTec AD

## 1. Experiment Setting

This document records the first complete PatchCore baseline on MVTec AD for the `iad-vlm-anomaly` project.

| Item | Setting |
|---|---|
| Dataset | MVTec AD |
| Categories | 15 |
| Datamodule | Anomalib `Folder` |
| Model | PatchCore |
| Backbone | wide_resnet50_2 |
| Feature Layers | layer2, layer3 |
| Coreset Sampling Ratio | 0.1 |
| Number of Neighbors | 9 |
| Image Size | 256 |
| Center Crop Size | 224 |
| Result CSV | `results/baselines/patchcore_mvtec_summary.csv` |
| Per-category Metrics | `runs/baselines/patchcore/MVTecAD/<category>/metrics.json` |

## 2. Overall Results

| Metric | Mean Value |
|---|---:|
| Image AUROC | 0.9853 |
| Image F1Score | 0.9706 |
| Pixel AUROC | 0.9795 |
| Pixel F1Score | 0.5888 |

## 3. Per-category Results

| category | image_AUROC | image_F1Score | pixel_AUROC | pixel_F1Score |
| --- | --- | --- | --- | --- |
| bottle | 1.0000 | 1.0000 | 0.9797 | 0.7250 |
| cable | 0.9858 | 0.9787 | 0.9819 | 0.6494 |
| capsule | 0.9621 | 0.9541 | 0.9888 | 0.5435 |
| carpet | 0.9968 | 0.9888 | 0.9893 | 0.5597 |
| grid | 0.9718 | 0.9825 | 0.9840 | 0.3692 |
| hazelnut | 1.0000 | 1.0000 | 0.9842 | 0.6142 |
| leather | 1.0000 | 1.0000 | 0.9892 | 0.4031 |
| metal_nut | 0.9961 | 0.9783 | 0.9789 | 0.8320 |
| pill | 0.9415 | 0.9489 | 0.9734 | 0.7433 |
| screw | 0.9944 | 0.9483 | 0.9892 | 0.3933 |
| tile | 0.9818 | 0.9630 | 0.9541 | 0.6579 |
| toothbrush | 0.9889 | 0.9333 | 0.9875 | 0.5831 |
| transistor | 1.0000 | 0.9744 | 0.9788 | 0.7050 |
| wood | 0.9800 | 0.9333 | 0.9542 | 0.5049 |
| zipper | 0.9802 | 0.9748 | 0.9793 | 0.5483 |

## 4. Weak Categories at Image Level

The following categories have relatively lower image-level anomaly classification performance.

| category | image_AUROC | image_F1Score |
| --- | --- | --- |
| pill | 0.9415 | 0.9489 |
| capsule | 0.9621 | 0.9541 |
| grid | 0.9718 | 0.9825 |
| wood | 0.9800 | 0.9333 |
| zipper | 0.9802 | 0.9748 |

## 5. Weak Categories at Pixel Level

The following categories have relatively lower pixel-level anomaly localization performance.

| category | pixel_AUROC | pixel_F1Score |
| --- | --- | --- |
| grid | 0.9840 | 0.3692 |
| screw | 0.9892 | 0.3933 |
| leather | 0.9892 | 0.4031 |
| wood | 0.9542 | 0.5049 |
| capsule | 0.9888 | 0.5435 |

## 6. Preliminary Observations

1. PatchCore performs strongly on image-level anomaly detection. The mean Image AUROC is high, showing that normal-memory-based nearest-neighbor matching is already effective for judging whether an image is anomalous.

2. Pixel-level localization is clearly weaker than image-level classification. The mean Pixel F1Score is much lower than the mean Pixel AUROC, which indicates that PatchCore can often rank abnormal pixels correctly but does not always produce an accurate binary defect mask.

3. Categories such as `grid`, `screw`, `leather`, and `wood` are important failure-analysis targets because their Pixel F1Score is relatively low. These categories are suitable for later mask refinement, region reasoning, SAM-based segmentation, and manufacturing-aware semantic constraints.

4. The current baseline proves that the project pipeline is working:
   - MVTec AD directory structure is readable.
   - Anomalib `Folder` datamodule works without soft links.
   - PatchCore can run on all 15 categories.
   - Per-category metrics and summary CSV can be saved and uploaded to GitHub.

## 7. Follow-up Direction

The next stage should not only chase higher Image AUROC. The main improvement target should be pixel-level localization and interpretable defect reasoning.

Recommended next steps:

1. Add formal evaluation protocol fields:
   - train sample count
   - test sample count
   - validation split setting
   - full-test or split-test evaluation flag

2. Generate visual failure cases for weak categories:
   - `grid`
   - `screw`
   - `leather`
   - `wood`

3. Introduce the next baseline:
   - PaDiM
   - FastFlow
   - WinCLIP or CLIP-based zero-shot anomaly baseline

4. Prepare the later VLM/SAM/knowledge-enhanced module around pixel-level localization and defect explanation.
