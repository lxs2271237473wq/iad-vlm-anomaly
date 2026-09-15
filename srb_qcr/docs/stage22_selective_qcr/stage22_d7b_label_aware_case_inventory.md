# Stage 22-D7b: Label-Aware Case Inventory

The earlier D7a inventory ranked raw anomaly-score changes without accounting for the true image label. A score increase is beneficial for anomaly images but harmful for normal images. This corrected inventory uses:

```text
orientation = +1 for anomaly images
orientation = -1 for normal images
label-aware benefit(A vs B) = orientation * (score_A - score_B)
```

- source rows: `1725`
- categories: `15`
- selected rows: `45`

## Case counts

| Case type | Count |
|---|---:|
| gate_off_missed_vlm_opportunity | 5 |
| gate_off_protected_from_harmful_vlm | 5 |
| gate_on_harmful | 5 |
| gate_on_helpful | 5 |
| largest_naive_harm | 5 |
| largest_srb_repair_over_naive | 5 |
| largest_srb_repair_over_old_quality | 5 |
| srb_failure_vs_detector | 5 |
| srb_success_vs_detector | 5 |

## Top representative cases

| Case type | Rank | Category | Y | D | M | Q | SRB | Benefit vs detector | Image |
|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| srb_success_vs_detector | 1 | cable | 1 | 0.2644 | 0.5276 | 0.9835 | 0.3232 | +0.0588 | `/root/private_data/iad-vlm-anomaly/datasets/MVTecAD/cable/test/cable_swap/010.png` |
| srb_success_vs_detector | 2 | tile | 1 | 0.5037 | 0.7952 | 0.9279 | 0.5616 | +0.0579 | `/root/private_data/iad-vlm-anomaly/datasets/MVTecAD/tile/test/oil/014.png` |
| srb_success_vs_detector | 3 | cable | 1 | 0.2873 | 0.7095 | 0.8780 | 0.3440 | +0.0567 | `/root/private_data/iad-vlm-anomaly/datasets/MVTecAD/cable/test/cable_swap/004.png` |
| srb_failure_vs_detector | 1 | capsule | 1 | 1.0000 | 0.5843 | 1.0000 | 0.9351 | -0.0649 | `/root/private_data/iad-vlm-anomaly/datasets/MVTecAD/capsule/test/crack/001.png` |
| srb_failure_vs_detector | 2 | toothbrush | 1 | 1.0000 | 0.6682 | 1.0000 | 0.9352 | -0.0648 | `/root/private_data/iad-vlm-anomaly/datasets/MVTecAD/toothbrush/test/defective/022.png` |
| srb_failure_vs_detector | 3 | grid | 1 | 0.7381 | 0.3628 | 0.9715 | 0.6744 | -0.0638 | `/root/private_data/iad-vlm-anomaly/datasets/MVTecAD/grid/test/thread/010.png` |
| largest_naive_harm | 1 | bottle | 1 | 1.0000 | 0.0000 | 0.9762 | 1.0000 | +0.0000 | `/root/private_data/iad-vlm-anomaly/datasets/MVTecAD/bottle/test/contamination/007.png` |
| largest_naive_harm | 2 | metal_nut | 1 | 0.9847 | 0.0000 | 0.1158 | 0.9847 | +0.0000 | `/root/private_data/iad-vlm-anomaly/datasets/MVTecAD/metal_nut/test/color/006.png` |
| largest_naive_harm | 3 | bottle | 1 | 1.0000 | 0.0171 | 0.8920 | 1.0000 | +0.0000 | `/root/private_data/iad-vlm-anomaly/datasets/MVTecAD/bottle/test/contamination/017.png` |
| largest_srb_repair_over_naive | 1 | bottle | 1 | 1.0000 | 0.0000 | 0.9762 | 1.0000 | +0.0000 | `/root/private_data/iad-vlm-anomaly/datasets/MVTecAD/bottle/test/contamination/007.png` |
| largest_srb_repair_over_naive | 2 | metal_nut | 1 | 0.9847 | 0.0000 | 0.1158 | 0.9847 | +0.0000 | `/root/private_data/iad-vlm-anomaly/datasets/MVTecAD/metal_nut/test/color/006.png` |
| largest_srb_repair_over_naive | 3 | bottle | 1 | 1.0000 | 0.0171 | 0.8920 | 1.0000 | +0.0000 | `/root/private_data/iad-vlm-anomaly/datasets/MVTecAD/bottle/test/contamination/017.png` |
| largest_srb_repair_over_old_quality | 1 | bottle | 1 | 1.0000 | 0.0000 | 0.9762 | 1.0000 | +0.0000 | `/root/private_data/iad-vlm-anomaly/datasets/MVTecAD/bottle/test/contamination/007.png` |
| largest_srb_repair_over_old_quality | 2 | metal_nut | 1 | 0.9847 | 0.0000 | 0.1158 | 0.9847 | +0.0000 | `/root/private_data/iad-vlm-anomaly/datasets/MVTecAD/metal_nut/test/color/006.png` |
| largest_srb_repair_over_old_quality | 3 | bottle | 1 | 1.0000 | 0.0171 | 0.8920 | 1.0000 | +0.0000 | `/root/private_data/iad-vlm-anomaly/datasets/MVTecAD/bottle/test/contamination/017.png` |
| gate_off_protected_from_harmful_vlm | 1 | metal_nut | 1 | 0.9847 | 0.0000 | 0.1158 | 0.9847 | +0.0000 | `/root/private_data/iad-vlm-anomaly/datasets/MVTecAD/metal_nut/test/color/006.png` |
| gate_off_protected_from_harmful_vlm | 2 | leather | 1 | 1.0000 | 0.1049 | 0.4011 | 1.0000 | +0.0000 | `/root/private_data/iad-vlm-anomaly/datasets/MVTecAD/leather/test/glue/015.png` |
| gate_off_protected_from_harmful_vlm | 3 | grid | 1 | 0.8800 | 0.0000 | 0.1394 | 0.8800 | +0.0000 | `/root/private_data/iad-vlm-anomaly/datasets/MVTecAD/grid/test/metal_contamination/004.png` |
| gate_off_missed_vlm_opportunity | 1 | tile | 1 | 0.3147 | 0.9807 | 0.2054 | 0.3147 | +0.0000 | `/root/private_data/iad-vlm-anomaly/datasets/MVTecAD/tile/test/gray_stroke/013.png` |
| gate_off_missed_vlm_opportunity | 2 | cable | 1 | 0.3466 | 0.9264 | 0.0633 | 0.3466 | +0.0000 | `/root/private_data/iad-vlm-anomaly/datasets/MVTecAD/cable/test/poke_insulation/000.png` |
| gate_off_missed_vlm_opportunity | 3 | tile | 1 | 0.3522 | 0.9273 | 0.3687 | 0.3522 | +0.0000 | `/root/private_data/iad-vlm-anomaly/datasets/MVTecAD/tile/test/rough/000.png` |
| gate_on_helpful | 1 | cable | 1 | 0.2644 | 0.5276 | 0.9835 | 0.3232 | +0.0588 | `/root/private_data/iad-vlm-anomaly/datasets/MVTecAD/cable/test/cable_swap/010.png` |
| gate_on_helpful | 2 | tile | 1 | 0.5037 | 0.7952 | 0.9279 | 0.5616 | +0.0579 | `/root/private_data/iad-vlm-anomaly/datasets/MVTecAD/tile/test/oil/014.png` |
| gate_on_helpful | 3 | cable | 1 | 0.2873 | 0.7095 | 0.8780 | 0.3440 | +0.0567 | `/root/private_data/iad-vlm-anomaly/datasets/MVTecAD/cable/test/cable_swap/004.png` |
| gate_on_harmful | 1 | capsule | 1 | 1.0000 | 0.5843 | 1.0000 | 0.9351 | -0.0649 | `/root/private_data/iad-vlm-anomaly/datasets/MVTecAD/capsule/test/crack/001.png` |
| gate_on_harmful | 2 | toothbrush | 1 | 1.0000 | 0.6682 | 1.0000 | 0.9352 | -0.0648 | `/root/private_data/iad-vlm-anomaly/datasets/MVTecAD/toothbrush/test/defective/022.png` |
| gate_on_harmful | 3 | grid | 1 | 0.7381 | 0.3628 | 0.9715 | 0.6744 | -0.0638 | `/root/private_data/iad-vlm-anomaly/datasets/MVTecAD/grid/test/thread/010.png` |