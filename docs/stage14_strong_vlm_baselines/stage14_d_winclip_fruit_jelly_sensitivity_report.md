# Stage 14-D WinCLIP fruit_jelly Sensitivity

## 1. Purpose

This stage tests whether the weak WinCLIP zero-shot result on AD2 fruit_jelly is caused by default configuration choices.

It varies class name, k-shot value, and WinCLIP scales.

## 2. Configuration Space

- Class names: `['fruit jelly', 'fruit_jelly', 'jelly']`
- k-shot values: `[0, 1, 2, 4]`
- scales: `[(2, 3), (1, 2, 3)]`

## 3. Results

| Rank | Status | Class name | k-shot | Scales | Image AUROC | Image F1 | Pixel AUROC | Pixel F1 | Error |
|---:|---|---|---:|---|---:|---:|---:|---:|---|
| 1 | success | `jelly` | 1 | `(1, 2, 3)` | 0.6300 | 0.8571 | 0.8511 | 0.2399 | `` |
| 2 | success | `fruit_jelly` | 4 | `(2, 3)` | 0.5667 | 0.8235 | 0.8051 | 0.1898 | `` |
| 3 | success | `jelly` | 1 | `(2, 3)` | 0.5567 | 0.8571 | 0.7800 | 0.1526 | `` |
| 4 | success | `fruit jelly` | 1 | `(2, 3)` | 0.5467 | 0.7119 | 0.7338 | 0.1992 | `` |
| 5 | success | `jelly` | 2 | `(2, 3)` | 0.5383 | 0.8571 | 0.8333 | 0.2117 | `` |
| 6 | success | `fruit jelly` | 4 | `(1, 2, 3)` | 0.5333 | 0.8571 | 0.8954 | 0.3111 | `` |
| 7 | success | `jelly` | 4 | `(2, 3)` | 0.5300 | 0.8235 | 0.7932 | 0.2382 | `` |
| 8 | success | `fruit_jelly` | 0 | `(1, 2, 3)` | 0.4950 | 0.8571 | 0.5757 | 0.0455 | `` |
| 9 | success | `fruit jelly` | 2 | `(1, 2, 3)` | 0.4917 | 0.8182 | 0.7473 | 0.1447 | `` |
| 10 | success | `jelly` | 0 | `(2, 3)` | 0.4783 | 0.8571 | 0.6535 | 0.1005 | `` |
| 11 | success | `fruit jelly` | 4 | `(2, 3)` | 0.4633 | 0.8060 | 0.7520 | 0.1765 | `` |
| 12 | success | `fruit_jelly` | 1 | `(1, 2, 3)` | 0.4500 | 0.8571 | 0.8452 | 0.2666 | `` |
| 13 | success | `fruit_jelly` | 1 | `(2, 3)` | 0.4500 | 0.7879 | 0.8227 | 0.2146 | `` |
| 14 | success | `fruit_jelly` | 2 | `(1, 2, 3)` | 0.4467 | 0.8571 | 0.7788 | 0.1507 | `` |
| 15 | success | `jelly` | 4 | `(1, 2, 3)` | 0.4433 | 0.8571 | 0.8582 | 0.2593 | `` |
| 16 | success | `fruit jelly` | 2 | `(2, 3)` | 0.4367 | 0.8235 | 0.7984 | 0.3136 | `` |
| 17 | success | `fruit jelly` | 1 | `(1, 2, 3)` | 0.4233 | 0.7879 | 0.8418 | 0.2709 | `` |
| 18 | success | `fruit_jelly` | 0 | `(2, 3)` | 0.4167 | 0.8571 | 0.5802 | 0.0055 | `` |
| 19 | success | `jelly` | 2 | `(1, 2, 3)` | 0.4117 | 0.8571 | 0.8301 | 0.2391 | `` |
| 20 | success | `fruit_jelly` | 4 | `(1, 2, 3)` | 0.4117 | 0.8235 | 0.7813 | 0.1640 | `` |
| 21 | success | `fruit jelly` | 0 | `(2, 3)` | 0.3917 | 0.8571 | 0.5302 | 0.0057 | `` |
| 22 | success | `fruit jelly` | 0 | `(1, 2, 3)` | 0.3717 | 0.8406 | 0.5840 | 0.0000 | `` |
| 23 | success | `fruit_jelly` | 2 | `(2, 3)` | 0.2833 | 0.8406 | 0.7576 | 0.1774 | `` |
| 24 | success | `jelly` | 0 | `(1, 2, 3)` | 0.2700 | 0.8571 | 0.5150 | 0.0796 | `` |

## 4. Best Successful Configuration

| Item | Value |
|---|---:|
| class_name | `jelly` |
| k_shot | 1 |
| scales | `(1, 2, 3)` |
| image AUROC | 0.6300 |
| image F1 | 0.8571 |
| pixel AUROC | 0.8511 |
| pixel F1 | 0.2399 |

Interpretation:

WinCLIP improves over the default zero-shot pilot, but still does not exceed the PatchCore fruit_jelly reference.

## 5. Errors

No configuration failed.

## 6. Output

- Sensitivity CSV: `results/stage14_strong_vlm_baselines/stage14_d_winclip_fruit_jelly_sensitivity.csv`
- Raw JSON: `results/stage14_strong_vlm_baselines/stage14_d_winclip_fruit_jelly_sensitivity_raw.json`
- Error log: `results/stage14_strong_vlm_baselines/stage14_d_winclip_fruit_jelly_sensitivity_errors.txt`
- Report: `docs/stage14_strong_vlm_baselines/stage14_d_winclip_fruit_jelly_sensitivity_report.md`