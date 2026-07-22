# Stage 20-D: AnomalyCLIP AD2-four Full Evaluation

## Protocol

- implementation: official `zqhang/AnomalyCLIP`
- repository commit: `3911738c0867544f545a076ad78f3f11d9ecbfdf`
- dataset: `ad2four`
- categories: `fruit_jelly`, `sheet_metal`, `vial`, `walnuts`
- number of test images: `484`
- checkpoint: fixed checkpoint verified in Stage 20-C3
- checkpoint path: `/root/private_data/third_party/AnomalyCLIP/checkpoints/9_12_4_multiscale/epoch_10.pth`
- checkpoint SHA-256: `7205c05df3319984b349686cbfd8cc01d3ac241a82f33943e9217cbb85604b0b`
- model: `ViT-L/14@336px`
- image size: `518`
- feature layers: `6, 12, 18, 24`
- DPAM layer: `20`
- metrics: image AUROC, image AP, pixel AUROC, pixel AUPRO
- additional training on AD2: `none`

## Per-category results

| Category | Image AUROC | Image AP | Pixel AUROC | Pixel AUPRO |
|---|---:|---:|---:|---:|
| fruit_jelly | 0.6680 | 0.8800 | 0.6240 | 0.5010 |
| sheet_metal | 0.5330 | 0.8420 | 0.6520 | 0.1230 |
| vial | 0.5400 | 0.8190 | 0.6540 | 0.5420 |
| walnuts | 0.4130 | 0.5450 | 0.8750 | 0.4420 |
| mean | 0.5390 | 0.7710 | 0.7010 | 0.4020 |

## System-level image-AUROC comparison

| Method | Image AUROC | Difference relative to AnomalyCLIP | Role |
|---|---:|---:|---|
| PatchCore + context VLM, same-set | 0.8453 | +0.3063 | upper-bound diagnostic only |
| PatchCore + context VLM, LOCO | 0.8210 | +0.2820 | primary fair system result |
| PatchCore | 0.7853 | +0.2463 | classic detector baseline |
| EfficientAD-30 fixed-budget | 0.7604 | +0.2214 | modern detector fixed-budget baseline |
| context-aware VLM | 0.7101 | +0.1711 | context-aware VLM baseline |
| full-image VLM | 0.6459 | +0.1069 | full-image VLM baseline |
| WinCLIP fixed protocol | 0.6138 | +0.0748 | existing external VLM anomaly baseline |
| AnomalyCLIP fixed checkpoint | 0.5390 | +0.0000 | new external VLM anomaly baseline |

## Runtime

- complete evaluation time: `1131.48` seconds

## Claim restrictions

- This is a fixed-checkpoint evaluation with no AD2-specific tuning.
- The comparison should be described as an external baseline under the
  implemented fixed protocol.
- Do not claim universal superiority over AnomalyCLIP based on one
  adapted AD2 protocol.
- The same-set fusion remains an upper-bound diagnostic only.

## Main result

- AnomalyCLIP mean image AUROC: `0.5390`
- AnomalyCLIP mean image AP: `0.7710`
- AnomalyCLIP mean pixel AUROC: `0.7010`
- AnomalyCLIP mean pixel AUPRO: `0.4020`
