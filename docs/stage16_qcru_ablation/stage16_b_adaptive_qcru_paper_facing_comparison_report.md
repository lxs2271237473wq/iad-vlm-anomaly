# Stage 16-B Adaptive QCR-U Paper-facing Comparison

## 1. Purpose

This stage connects the Adaptive QCR-U candidate back to a paper-facing comparison table.

It tests whether Adaptive QCR-U should be the final method name, or whether the method should be downgraded to quality-calibrated localization-guided fusion.

## 2. Primary Protocol

The primary protocol is:

```text
dataset = VisA
strategy = inspection_binary
eval_mode = crop_topk_ensemble
```

Reason: QCR-U is a candidate/crop reliability method. `full_all` is useful for diagnostics but is not the correct primary protocol for a crop-based reliability module.

## 3. Primary Protocol Table

| Backbone | Variant | AUROC | AP | Best F1 | Best Acc |
|---|---|---:|---:|---:|---:|
| FastFlow | detector_only | 0.8955 | 0.9205 | 0.8445 | 0.8224 |
| FastFlow | crop_topk_vlm | 0.9269 | 0.9485 | 0.8765 | 0.8603 |
| FastFlow | naive_detector_crop_fusion | 0.9688 | 0.9750 | 0.9167 | 0.9047 |
| FastFlow | quality_weighted_crop | 0.9778 | 0.9822 | 0.9304 | 0.9214 |
| FastFlow | fixed_quality_consistency | 0.9842 | 0.9873 | 0.9403 | 0.9334 |
| FastFlow | adaptive_qcru | 0.9783 | 0.9827 | 0.9312 | 0.9223 |
| PatchCore | detector_only | 0.9131 | 0.9242 | 0.8606 | 0.8390 |
| PatchCore | crop_topk_vlm | 0.8846 | 0.9096 | 0.8340 | 0.8016 |
| PatchCore | naive_detector_crop_fusion | 0.9616 | 0.9681 | 0.9127 | 0.8996 |
| PatchCore | quality_weighted_crop | 0.9718 | 0.9752 | 0.9319 | 0.9237 |
| PatchCore | fixed_quality_consistency | 0.9740 | 0.9784 | 0.9312 | 0.9232 |
| PatchCore | adaptive_qcru | 0.9722 | 0.9756 | 0.9320 | 0.9237 |

## 4. Decision Summary

| Scope | Comparison | Wins | Total | Win Rate | Mean Delta | Median Delta | Min Delta | Max Delta |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| primary_protocol | adaptive_qcru_minus_naive | 2 | 2 | 1.0000 | +0.0100 | +0.0100 | +0.0094 | +0.0105 |
| primary_protocol | adaptive_qcru_minus_quality | 2 | 2 | 1.0000 | +0.0004 | +0.0004 | +0.0003 | +0.0005 |
| primary_protocol | adaptive_qcru_minus_fixed_qc | 0 | 2 | 0.0000 | -0.0039 | -0.0039 | -0.0059 | -0.0018 |
| primary_protocol | quality_minus_naive | 2 | 2 | 1.0000 | +0.0096 | +0.0096 | +0.0090 | +0.0102 |
| all_protocols | adaptive_qcru_minus_naive | 12 | 12 | 1.0000 | +0.0371 | +0.0253 | +0.0082 | +0.0816 |
| all_protocols | adaptive_qcru_minus_quality | 12 | 12 | 1.0000 | +0.0006 | +0.0005 | +0.0003 | +0.0012 |
| all_protocols | adaptive_qcru_minus_fixed_qc | 6 | 12 | 0.5000 | +0.0099 | +0.0042 | -0.0059 | +0.0342 |
| all_protocols | quality_minus_naive | 12 | 12 | 1.0000 | +0.0365 | +0.0248 | +0.0079 | +0.0808 |

## 5. Final Trial Recommendation

- recommended method name: `Quality-Calibrated QCR with adaptive consistency refinement`
- recommendation: Use Quality-Calibrated QCR as the main method; describe adaptive consistency as a small gated refinement.

## 6. Interpretation Rule

If Adaptive QCR-U only improves over quality-only by a negligible margin, the paper should not overclaim adaptive consistency.

In that case, the correct claim is:

```text
Candidate quality provides the main reliability calibration gain, while adaptive consistency is a conservative refinement that avoids fixed-consistency degradation.
```

## 7. Outputs

- `results/stage16_qcru_ablation/stage16_b_adaptive_qcru_all_variants_per_config.csv`
- `results/stage16_qcru_ablation/stage16_b_adaptive_qcru_all_variants_per_category.csv`
- `results/stage16_qcru_ablation/stage16_b_adaptive_qcru_primary_protocol_table.csv`
- `results/stage16_qcru_ablation/stage16_b_adaptive_qcru_final_method_decision.csv`
- `docs/stage16_qcru_ablation/stage16_b_adaptive_qcru_paper_facing_comparison_report.md`
