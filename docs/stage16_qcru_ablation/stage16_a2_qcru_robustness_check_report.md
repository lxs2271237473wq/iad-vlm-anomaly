# Stage 16-A2 QCR-U Robustness Check

## 1. Purpose

Stage 16-A1 showed that fixed quality-consistency fusion can improve the best protocol.

Stage 16-A2 checks whether that gain is robust across all protocols, instead of only appearing in the best protocol.

## 2. Overall Robustness Summary

| Check | Wins | Total | Win Rate | Mean Delta | Median Delta | Min Delta | Max Delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| V5 > V3 naive fusion | 12 | 12 | 1.0000 | 0.0271 | 0.0224 | 0.0098 | 0.0536 |
| V5 > V4 quality-only | 6 | 12 | 0.5000 | -0.0094 | -0.0038 | -0.0334 | 0.0064 |
| V5 > V0 detector-only | 9 | 12 | 0.7500 | 0.0334 | 0.0464 | -0.0329 | 0.0887 |
| V5 > V2 crop-VLM-only | 12 | 12 | 1.0000 | 0.1791 | 0.1274 | 0.0572 | 0.3541 |
| V4 > V3 naive fusion | 12 | 12 | 1.0000 | 0.0365 | 0.0248 | 0.0079 | 0.0808 |
| V5 > V3 naive fusion by eval_mode=crop_or_full | 4 | 4 | 1.0000 | 0.0189 | 0.0190 | 0.0128 | 0.0248 |
| V5 > V4 quality-only by eval_mode=crop_or_full | 3 | 4 | 0.7500 | -0.0009 | 0.0012 | -0.0097 | 0.0038 |
| V5 > V3 naive fusion by eval_mode=crop_topk_ensemble | 4 | 4 | 1.0000 | 0.0149 | 0.0138 | 0.0098 | 0.0222 |
| V5 > V4 quality-only by eval_mode=crop_topk_ensemble | 3 | 4 | 0.7500 | 0.0005 | 0.0020 | -0.0086 | 0.0064 |
| V5 > V3 naive fusion by eval_mode=full_all | 4 | 4 | 1.0000 | 0.0475 | 0.0480 | 0.0405 | 0.0536 |
| V5 > V4 quality-only by eval_mode=full_all | 0 | 4 | 0.0000 | -0.0277 | -0.0288 | -0.0334 | -0.0197 |

## 3. Protocol-level Deltas

| Backbone | Strategy | Eval Mode | V5 AUROC | V3 AUROC | V4 AUROC | V5-V3 | V5-V4 |
|---|---|---|---:|---:|---:|---:|---:|
| FastFlow | inspection_binary | full_all | 0.9087 | 0.8550 | 0.9283 | +0.0536 | -0.0197 |
| PatchCore | generic_binary | full_all | 0.8822 | 0.8335 | 0.9136 | +0.0486 | -0.0314 |
| PatchCore | category_binary | full_all | 0.8802 | 0.8328 | 0.9136 | +0.0474 | -0.0334 |
| PatchCore | inspection_binary | full_all | 0.8927 | 0.8522 | 0.9189 | +0.0405 | -0.0262 |
| PatchCore | category_binary | crop_or_full | 0.9440 | 0.9192 | 0.9537 | +0.0248 | -0.0097 |
| FastFlow | inspection_binary | crop_or_full | 0.9724 | 0.9499 | 0.9686 | +0.0225 | +0.0038 |
| PatchCore | category_binary | crop_topk_ensemble | 0.9525 | 0.9303 | 0.9611 | +0.0222 | -0.0086 |
| PatchCore | inspection_binary | crop_or_full | 0.9665 | 0.9510 | 0.9650 | +0.0155 | +0.0015 |
| FastFlow | inspection_binary | crop_topk_ensemble | 0.9842 | 0.9688 | 0.9778 | +0.0153 | +0.0064 |
| PatchCore | generic_binary | crop_or_full | 0.9702 | 0.9574 | 0.9693 | +0.0128 | +0.0009 |
| PatchCore | inspection_binary | crop_topk_ensemble | 0.9740 | 0.9616 | 0.9718 | +0.0124 | +0.0022 |
| PatchCore | generic_binary | crop_topk_ensemble | 0.9775 | 0.9677 | 0.9756 | +0.0098 | +0.0019 |

## 4. Failure / Weakness Cases

| Backbone | Strategy | Eval Mode | V5-V3 | V5-V4 | V5-V0 | V5-V2 | Reason |
|---|---|---|---:|---:|---:|---:|---|
| PatchCore | category_binary | crop_or_full | +0.0248 | -0.0097 | +0.0309 | +0.1620 | V5_not_better_than_quality_only |
| PatchCore | category_binary | crop_topk_ensemble | +0.0222 | -0.0086 | +0.0394 | +0.1534 | V5_not_better_than_quality_only |
| PatchCore | inspection_binary | full_all | +0.0405 | -0.0262 | -0.0204 | +0.3036 | V5_not_better_than_quality_only;V5_not_better_than_detector |
| PatchCore | category_binary | full_all | +0.0474 | -0.0334 | -0.0329 | +0.3541 | V5_not_better_than_quality_only;V5_not_better_than_detector |
| PatchCore | generic_binary | full_all | +0.0486 | -0.0314 | -0.0309 | +0.3384 | V5_not_better_than_quality_only;V5_not_better_than_detector |
| FastFlow | inspection_binary | full_all | +0.0536 | -0.0197 | +0.0132 | +0.3196 | V5_not_better_than_quality_only |

## 5. Decision Rule

If V5 is consistently better than V3 naive fusion but often worse than V4 quality-only, the consistency term should not be claimed as universally beneficial.

In that case, the next method should be revised from fixed Q+C fusion to adaptive QCR-U:

```text
use quality-weighted crop as the stable core;
apply consistency only when detector and VLM evidence are both reliable;
avoid adding consistency under weak/full-image protocols where it hurts.
```

## 6. Outputs

- `results/stage16_qcru_ablation/stage16_a2_qcru_variant_delta_by_protocol.csv`
- `results/stage16_qcru_ablation/stage16_a2_qcru_robustness_summary.csv`
- `results/stage16_qcru_ablation/stage16_a2_qcru_failure_cases.csv`
- `docs/stage16_qcru_ablation/stage16_a2_qcru_robustness_check_report.md`
