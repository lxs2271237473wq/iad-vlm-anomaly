# Stage 20-C3: AnomalyCLIP Single-image Smoke Test

## Status

- status: `success`
- category: `fruit_jelly`
- ground-truth anomaly: `1`
- image: `/root/private_data/anomalyclip_data/ad2four/fruit_jelly/test/bad/000320_000_overexposed.png`
- checkpoint: `/root/private_data/third_party/AnomalyCLIP/checkpoints/9_12_4_multiscale/epoch_10.pth`

## Outputs

- image anomaly score: `0.88681185`
- image tensor shape: `[1, 3, 518, 518]`
- number of patch feature maps: `4`
- anomaly-map shape: `[1, 518, 518]`
- anomaly-map minimum: `0.00401884`
- anomaly-map maximum: `0.24169862`
- anomaly-map mean: `0.04333393`

## Runtime

- elapsed seconds: `52.291`
- peak allocated VRAM: `2073.54 MiB`
- GPU: `NVIDIA GeForce RTX 4090`

## Interpretation

This smoke test confirms that the official model, prompt checkpoint,
AD2-four Dataset adapter, image-level score, and pixel-level anomaly
map can run together. It is not a final AnomalyCLIP benchmark result.

## Next step

Fix the official cross-dataset checkpoint explicitly and run the
complete 484-image AD2-four evaluation.
