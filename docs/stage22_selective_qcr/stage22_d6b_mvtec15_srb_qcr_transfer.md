# Stage 22-D6b: MVTec AD 15-Class Frozen SRB-QCR Transfer

## Protocol

- parameters selected on: `VisA PatchCore category-LOCO`
- target: `MVTec AD 15 categories`
- target labels used for parameters: `none`
- rows: `1725`
- `w_max = 0.35`
- `q_quantile = 0.25`
- target unlabeled Q threshold: `0.440713`
- `tau_delta = 0.75`
- resolved Q adapter: `fallback max(candidate_score_mean) per image`

The Q adapter is frozen in the script before metric computation. A direct candidate-quality column is preferred; otherwise the maximum candidate mean score per image is used and normalized within category. Old Adaptive QCR is diagnostic because MVTec has no cached K; K is reconstructed by the fixed high-high rule D>=0.5 and M>=0.5.

## Summary

| ID | Method | Macro AUROC | Macro AP | Macro F1 | Pooled AUROC | Potential call rate |
|---|---|---:|---:|---:|---:|---:|
| D0 | Detector only | 0.9882 | 0.9967 | 0.9844 | 0.9881 | - |
| M0 | Crop CLIP only | 0.5586 | 0.7901 | 0.8470 | 0.5303 | - |
| V3 | Naive detector-crop fusion | 0.9436 | 0.9802 | 0.9421 | 0.9385 | - |
| V4 | Old Quality-Calibrated QCR | 0.9509 | 0.9830 | 0.9457 | 0.9437 | - |
| V6 | Old Adaptive QCR diagnostic reconstruction | 0.9513 | 0.9831 | 0.9460 | 0.9440 | - |
| S1 | SRB-QCR frozen transfer | 0.9885 | 0.9968 | 0.9855 | 0.9881 | 0.7501 |

## Main deltas

- SRB minus detector: `+0.0003`
- SRB minus naive: `+0.0449`
- SRB minus old Quality QCR: `+0.0376`
- categories SRB > detector: `6/15`
- worst category delta vs detector: `-0.0012`
- potential VLM call rate: `0.7501`
- potential calls saved: `0.2499`

## Bootstrap

| Comparison | Delta | CI low | CI high | P(delta>0) | Two-sided p |
|---|---:|---:|---:|---:|---:|
| SRB-QCR vs detector | +0.0003 | -0.0009 | +0.0014 | 0.6860 | 0.628000 |
| SRB-QCR vs naive fusion | +0.0449 | +0.0341 | +0.0561 | 1.0000 | 0.000000 |
| Quality QCR vs naive fusion | +0.0072 | +0.0035 | +0.0113 | 1.0000 | 0.000000 |
| SRB-QCR vs old Quality QCR | +0.0376 | +0.0279 | +0.0482 | 1.0000 | 0.000000 |

## Source inventory

### bottle
- PatchCore: `results/stage22_selective_qcr/mvtec15_rerun_patchcore/MVTecAD/bottle/patchcore_image_predictions.csv`
- candidates: `results/stage22_selective_qcr/mvtec15_rerun_patchcore/MVTecAD/bottle/candidate_regions.csv`
- CLIP: `results/stage22_selective_qcr/mvtec15_clip_crop_reasoning/MVTecAD/bottle/clip_crop_predictions.csv`
- CLIP score column: `vlm_anomaly_score`
- Q rule: `fallback max(candidate_score_mean) per image`

### cable
- PatchCore: `results/stage22_selective_qcr/mvtec15_rerun_patchcore/MVTecAD/cable/patchcore_image_predictions.csv`
- candidates: `results/stage22_selective_qcr/mvtec15_rerun_patchcore/MVTecAD/cable/candidate_regions.csv`
- CLIP: `results/stage22_selective_qcr/mvtec15_clip_crop_reasoning/MVTecAD/cable/clip_crop_predictions.csv`
- CLIP score column: `vlm_anomaly_score`
- Q rule: `fallback max(candidate_score_mean) per image`

### capsule
- PatchCore: `results/stage22_selective_qcr/mvtec15_rerun_patchcore/MVTecAD/capsule/patchcore_image_predictions.csv`
- candidates: `results/stage22_selective_qcr/mvtec15_rerun_patchcore/MVTecAD/capsule/candidate_regions.csv`
- CLIP: `results/stage22_selective_qcr/mvtec15_clip_crop_reasoning/MVTecAD/capsule/clip_crop_predictions.csv`
- CLIP score column: `vlm_anomaly_score`
- Q rule: `fallback max(candidate_score_mean) per image`

### carpet
- PatchCore: `results/stage22_selective_qcr/mvtec15_rerun_patchcore/MVTecAD/carpet/patchcore_image_predictions.csv`
- candidates: `results/stage22_selective_qcr/mvtec15_rerun_patchcore/MVTecAD/carpet/candidate_regions.csv`
- CLIP: `results/stage22_selective_qcr/mvtec15_clip_crop_reasoning/MVTecAD/carpet/clip_crop_predictions.csv`
- CLIP score column: `vlm_anomaly_score`
- Q rule: `fallback max(candidate_score_mean) per image`

### grid
- PatchCore: `results/stage22_selective_qcr/mvtec15_rerun_patchcore/MVTecAD/grid/patchcore_image_predictions.csv`
- candidates: `results/stage22_selective_qcr/mvtec15_rerun_patchcore/MVTecAD/grid/candidate_regions.csv`
- CLIP: `results/stage22_selective_qcr/mvtec15_clip_crop_reasoning/MVTecAD/grid/clip_crop_predictions.csv`
- CLIP score column: `vlm_anomaly_score`
- Q rule: `fallback max(candidate_score_mean) per image`

### hazelnut
- PatchCore: `results/stage22_selective_qcr/mvtec15_rerun_patchcore/MVTecAD/hazelnut/patchcore_image_predictions.csv`
- candidates: `results/stage22_selective_qcr/mvtec15_rerun_patchcore/MVTecAD/hazelnut/candidate_regions.csv`
- CLIP: `results/stage22_selective_qcr/mvtec15_clip_crop_reasoning/MVTecAD/hazelnut/clip_crop_predictions.csv`
- CLIP score column: `vlm_anomaly_score`
- Q rule: `fallback max(candidate_score_mean) per image`

### leather
- PatchCore: `results/stage22_selective_qcr/mvtec15_rerun_patchcore/MVTecAD/leather/patchcore_image_predictions.csv`
- candidates: `results/stage22_selective_qcr/mvtec15_rerun_patchcore/MVTecAD/leather/candidate_regions.csv`
- CLIP: `results/stage22_selective_qcr/mvtec15_clip_crop_reasoning/MVTecAD/leather/clip_crop_predictions.csv`
- CLIP score column: `vlm_anomaly_score`
- Q rule: `fallback max(candidate_score_mean) per image`

### metal_nut
- PatchCore: `results/stage22_selective_qcr/mvtec15_rerun_patchcore/MVTecAD/metal_nut/patchcore_image_predictions.csv`
- candidates: `results/stage22_selective_qcr/mvtec15_rerun_patchcore/MVTecAD/metal_nut/candidate_regions.csv`
- CLIP: `results/stage22_selective_qcr/mvtec15_clip_crop_reasoning/MVTecAD/metal_nut/clip_crop_predictions.csv`
- CLIP score column: `vlm_anomaly_score`
- Q rule: `fallback max(candidate_score_mean) per image`

### pill
- PatchCore: `results/stage22_selective_qcr/mvtec15_rerun_patchcore/MVTecAD/pill/patchcore_image_predictions.csv`
- candidates: `results/stage22_selective_qcr/mvtec15_rerun_patchcore/MVTecAD/pill/candidate_regions.csv`
- CLIP: `results/stage22_selective_qcr/mvtec15_clip_crop_reasoning/MVTecAD/pill/clip_crop_predictions.csv`
- CLIP score column: `vlm_anomaly_score`
- Q rule: `fallback max(candidate_score_mean) per image`

### screw
- PatchCore: `results/stage22_selective_qcr/mvtec15_rerun_patchcore/MVTecAD/screw/patchcore_image_predictions.csv`
- candidates: `results/stage22_selective_qcr/mvtec15_rerun_patchcore/MVTecAD/screw/candidate_regions.csv`
- CLIP: `results/stage22_selective_qcr/mvtec15_clip_crop_reasoning/MVTecAD/screw/clip_crop_predictions.csv`
- CLIP score column: `vlm_anomaly_score`
- Q rule: `fallback max(candidate_score_mean) per image`

### tile
- PatchCore: `results/stage22_selective_qcr/mvtec15_rerun_patchcore/MVTecAD/tile/patchcore_image_predictions.csv`
- candidates: `results/stage22_selective_qcr/mvtec15_rerun_patchcore/MVTecAD/tile/candidate_regions.csv`
- CLIP: `results/stage22_selective_qcr/mvtec15_clip_crop_reasoning/MVTecAD/tile/clip_crop_predictions.csv`
- CLIP score column: `vlm_anomaly_score`
- Q rule: `fallback max(candidate_score_mean) per image`

### toothbrush
- PatchCore: `results/stage22_selective_qcr/mvtec15_rerun_patchcore/MVTecAD/toothbrush/patchcore_image_predictions.csv`
- candidates: `results/stage22_selective_qcr/mvtec15_rerun_patchcore/MVTecAD/toothbrush/candidate_regions.csv`
- CLIP: `results/stage22_selective_qcr/mvtec15_clip_crop_reasoning/MVTecAD/toothbrush/clip_crop_predictions.csv`
- CLIP score column: `vlm_anomaly_score`
- Q rule: `fallback max(candidate_score_mean) per image`

### transistor
- PatchCore: `results/stage22_selective_qcr/mvtec15_rerun_patchcore/MVTecAD/transistor/patchcore_image_predictions.csv`
- candidates: `results/stage22_selective_qcr/mvtec15_rerun_patchcore/MVTecAD/transistor/candidate_regions.csv`
- CLIP: `results/stage22_selective_qcr/mvtec15_clip_crop_reasoning/MVTecAD/transistor/clip_crop_predictions.csv`
- CLIP score column: `vlm_anomaly_score`
- Q rule: `fallback max(candidate_score_mean) per image`

### wood
- PatchCore: `results/stage22_selective_qcr/mvtec15_rerun_patchcore/MVTecAD/wood/patchcore_image_predictions.csv`
- candidates: `results/stage22_selective_qcr/mvtec15_rerun_patchcore/MVTecAD/wood/candidate_regions.csv`
- CLIP: `results/stage22_selective_qcr/mvtec15_clip_crop_reasoning/MVTecAD/wood/clip_crop_predictions.csv`
- CLIP score column: `vlm_anomaly_score`
- Q rule: `fallback max(candidate_score_mean) per image`

### zipper
- PatchCore: `results/stage22_selective_qcr/mvtec15_rerun_patchcore/MVTecAD/zipper/patchcore_image_predictions.csv`
- candidates: `results/stage22_selective_qcr/mvtec15_rerun_patchcore/MVTecAD/zipper/candidate_regions.csv`
- CLIP: `results/stage22_selective_qcr/mvtec15_clip_crop_reasoning/MVTecAD/zipper/clip_crop_predictions.csv`
- CLIP score column: `vlm_anomaly_score`
- Q rule: `fallback max(candidate_score_mean) per image`

Potential call saving is an offline pre-gate estimate on MVTec, not a measured wall-clock speedup.