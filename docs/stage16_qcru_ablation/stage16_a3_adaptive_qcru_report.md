# Stage 16-A3 Adaptive QCR-U

## 1. Purpose

Stage 16-A2 showed that candidate quality is stable, while fixed consistency is not universally beneficial.

This stage tests an adaptive QCR-U score that uses quality-weighted crop scoring as the stable core and applies consistency only as a conservative reliability-gated bonus.

## 2. Formula

```text
D = detector anomaly score
M = crop VLM anomaly score
Q = candidate quality
K = high-high detector/VLM consistency

S_quality = 0.5D + 0.5 * M * (0.5 + 0.5Q)
agreement = 1 - |D - M|
mutual_anomaly_evidence = min(D, M)
gate = Q * K * agreement * mutual_anomaly_evidence
S_adaptive = S_quality + 0.05 * gate
```

The coefficient `0.05` is fixed and intentionally conservative. It is not selected by test-set tuning.

## 3. Robustness Summary

| Check | Wins | Total | Win Rate | Mean Delta | Median Delta | Min Delta | Max Delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| V6 > V3 naive | 12 | 12 | 1.0000 | 0.0371 | 0.0253 | 0.0082 | 0.0816 |
| V6 > V4 quality | 12 | 12 | 1.0000 | 0.0006 | 0.0005 | 0.0003 | 0.0012 |
| V6 > V5 fixed Q+C | 6 | 12 | 0.5000 | 0.0099 | 0.0042 | -0.0059 | 0.0342 |
| V5 > V4 quality | 6 | 12 | 0.5000 | -0.0094 | -0.0038 | -0.0334 | 0.0064 |
| V6 > V4 quality by eval_mode=crop_or_full | 4 | 4 | 1.0000 | 0.0005 | 0.0004 | 0.0003 | 0.0006 |
| V6 > V5 fixed Q+C by eval_mode=crop_or_full | 1 | 4 | 0.2500 | 0.0013 | -0.0009 | -0.0032 | 0.0102 |
| V6 > V4 quality by eval_mode=crop_topk_ensemble | 4 | 4 | 1.0000 | 0.0004 | 0.0004 | 0.0003 | 0.0005 |
| V6 > V5 fixed Q+C by eval_mode=crop_topk_ensemble | 1 | 4 | 0.2500 | -0.0001 | -0.0017 | -0.0059 | 0.0090 |
| V6 > V4 quality by eval_mode=full_all | 4 | 4 | 1.0000 | 0.0009 | 0.0008 | 0.0007 | 0.0012 |
| V6 > V5 fixed Q+C by eval_mode=full_all | 4 | 4 | 1.0000 | 0.0286 | 0.0296 | 0.0209 | 0.0342 |

## 4. Adaptive QCR-U Protocol Ranking

| Rank | Backbone | Strategy | Eval Mode | V6 AUROC | AP | Best F1 |
|---:|---|---|---|---:|---:|---:|
| 1 | FastFlow | inspection_binary | crop_topk_ensemble | 0.9783 | 0.9827 | 0.9312 |
| 2 | PatchCore | generic_binary | crop_topk_ensemble | 0.9759 | 0.9780 | 0.9402 |
| 3 | PatchCore | inspection_binary | crop_topk_ensemble | 0.9722 | 0.9756 | 0.9320 |
| 4 | PatchCore | generic_binary | crop_or_full | 0.9696 | 0.9741 | 0.9305 |
| 5 | FastFlow | inspection_binary | crop_or_full | 0.9693 | 0.9763 | 0.9156 |
| 6 | PatchCore | inspection_binary | crop_or_full | 0.9653 | 0.9709 | 0.9212 |
| 7 | PatchCore | category_binary | crop_topk_ensemble | 0.9615 | 0.9654 | 0.9131 |
| 8 | PatchCore | category_binary | crop_or_full | 0.9542 | 0.9609 | 0.9043 |
| 9 | FastFlow | inspection_binary | full_all | 0.9296 | 0.9467 | 0.8704 |
| 10 | PatchCore | inspection_binary | full_all | 0.9196 | 0.9341 | 0.8665 |
| 11 | PatchCore | category_binary | full_all | 0.9145 | 0.9323 | 0.8616 |
| 12 | PatchCore | generic_binary | full_all | 0.9144 | 0.9352 | 0.8622 |

## 5. Protocol-level Delta Table

| Backbone | Strategy | Eval Mode | V3 | V4 | V5 | V6 | V6-V3 | V6-V4 | V6-V5 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| PatchCore | category_binary | full_all | 0.8328 | 0.9136 | 0.8802 | 0.9145 | +0.0816 | +0.0008 | +0.0342 |
| PatchCore | generic_binary | full_all | 0.8335 | 0.9136 | 0.8822 | 0.9144 | +0.0809 | +0.0009 | +0.0322 |
| FastFlow | inspection_binary | full_all | 0.8550 | 0.9283 | 0.9087 | 0.9296 | +0.0745 | +0.0012 | +0.0209 |
| PatchCore | inspection_binary | full_all | 0.8522 | 0.9189 | 0.8927 | 0.9196 | +0.0675 | +0.0007 | +0.0270 |
| PatchCore | category_binary | crop_or_full | 0.9192 | 0.9537 | 0.9440 | 0.9542 | +0.0350 | +0.0005 | +0.0102 |
| PatchCore | category_binary | crop_topk_ensemble | 0.9303 | 0.9611 | 0.9525 | 0.9615 | +0.0313 | +0.0004 | +0.0090 |
| FastFlow | inspection_binary | crop_or_full | 0.9499 | 0.9686 | 0.9724 | 0.9693 | +0.0193 | +0.0006 | -0.0032 |
| PatchCore | inspection_binary | crop_or_full | 0.9510 | 0.9650 | 0.9665 | 0.9653 | +0.0143 | +0.0004 | -0.0011 |
| PatchCore | generic_binary | crop_or_full | 0.9574 | 0.9693 | 0.9702 | 0.9696 | +0.0122 | +0.0003 | -0.0006 |
| PatchCore | inspection_binary | crop_topk_ensemble | 0.9616 | 0.9718 | 0.9740 | 0.9722 | +0.0105 | +0.0003 | -0.0018 |
| FastFlow | inspection_binary | crop_topk_ensemble | 0.9688 | 0.9778 | 0.9842 | 0.9783 | +0.0094 | +0.0005 | -0.0059 |
| PatchCore | generic_binary | crop_topk_ensemble | 0.9677 | 0.9756 | 0.9775 | 0.9759 | +0.0082 | +0.0003 | -0.0016 |

## 6. Decision Rule

If adaptive QCR-U beats naive fusion consistently and avoids the full_all degradation of fixed Q+C, it can replace fixed Q+C as the next method candidate.

If adaptive QCR-U still fails to beat quality-only, the method should be simplified to quality-weighted crop fusion and consistency should be moved to analysis rather than method.

## 7. Outputs

- `results/stage16_qcru_ablation/stage16_a3_adaptive_qcru_per_config.csv`
- `results/stage16_qcru_ablation/stage16_a3_adaptive_qcru_delta_by_protocol.csv`
- `results/stage16_qcru_ablation/stage16_a3_adaptive_qcru_failure_cases.csv`
- `docs/stage16_qcru_ablation/stage16_a3_adaptive_qcru_report.md`
