# Stage 14-E WinCLIP Primary Fixed Protocol

## 1. Purpose

This stage evaluates WinCLIP on all AD2 primary categories using one fixed protocol.

This avoids claiming a per-category over-tuned WinCLIP result.

## 2. Fixed Protocol

- k-shot: `1`
- scales: `(1, 2, 3)`

| Category | class_name |
|---|---|
| fruit_jelly | `jelly` |
| sheet_metal | `sheet metal` |
| vial | `vial` |
| walnuts | `walnut` |

## 3. WinCLIP Results

| Category | Status | Image AUROC | Image F1 | Pixel AUROC | Pixel F1 | Error |
|---|---|---:|---:|---:|---:|---|
| fruit_jelly | success | 0.4667 | 0.8571 | 0.8327 | 0.2824 | `` |
| sheet_metal | success | 0.7074 | 0.8824 | 0.8140 | 0.2274 | `` |
| vial | success | 0.8795 | 0.8621 | 0.8188 | 0.2076 | `` |
| walnuts | success | 0.4015 | 0.6726 | 0.8155 | 0.3743 | `` |

## 4. Unified Category-level Comparison

| Category | Method | Group | AUROC | AP | Pixel AUROC | Pixel F1 |
|---|---|---|---:|---:|---:|---:|
| fruit_jelly | WinCLIP fixed protocol | external_vlm_baseline | 0.4667 |  | 0.8327 | 0.2824 |
| fruit_jelly | PatchCore | classical_detector | 0.7167 |  |  |  |
| fruit_jelly | full-image VLM | vlm_branch | 0.7533 |  |  |  |
| fruit_jelly | context-aware VLM | vlm_branch | 0.8567 |  |  |  |
| fruit_jelly | PatchCore + context VLM, LOCO | fusion_loco | 0.8333 | 0.9476 |  |  |
| fruit_jelly | PatchCore + context VLM, same-set | fusion_same_set | 0.8933 | 0.9711 |  |  |
| sheet_metal | WinCLIP fixed protocol | external_vlm_baseline | 0.7074 |  | 0.8140 | 0.2274 |
| sheet_metal | PatchCore | classical_detector | 0.7463 |  |  |  |
| sheet_metal | full-image VLM | vlm_branch | 0.7130 |  |  |  |
| sheet_metal | context-aware VLM | vlm_branch | 0.6574 |  |  |  |
| sheet_metal | PatchCore + context VLM, LOCO | fusion_loco | 0.7481 | 0.9261 |  |  |
| sheet_metal | PatchCore + context VLM, same-set | fusion_same_set | 0.7556 | 0.9265 |  |  |
| vial | WinCLIP fixed protocol | external_vlm_baseline | 0.8795 |  | 0.8188 | 0.2076 |
| vial | PatchCore | classical_detector | 0.8732 |  |  |  |
| vial | full-image VLM | vlm_branch | 0.6876 |  |  |  |
| vial | context-aware VLM | vlm_branch | 0.6834 |  |  |  |
| vial | PatchCore + context VLM, LOCO | fusion_loco | 0.9224 | 0.9725 |  |  |
| vial | PatchCore + context VLM, same-set | fusion_same_set | 0.9256 | 0.9734 |  |  |
| walnuts | WinCLIP fixed protocol | external_vlm_baseline | 0.4015 |  | 0.8155 | 0.3743 |
| walnuts | PatchCore | classical_detector | 0.8052 |  |  |  |
| walnuts | full-image VLM | vlm_branch | 0.4296 |  |  |  |
| walnuts | context-aware VLM | vlm_branch | 0.6430 |  |  |  |
| walnuts | PatchCore + context VLM, LOCO | fusion_loco | 0.7800 | 0.8751 |  |  |
| walnuts | PatchCore + context VLM, same-set | fusion_same_set | 0.8067 | 0.8883 |  |  |

## 5. Aggregate Observation

- Mean WinCLIP image AUROC over successful categories: `0.6138`
- Best WinCLIP category AUROC: `0.8795`
- Worst WinCLIP category AUROC: `0.4015`

## 6. Decision

If WinCLIP remains below PatchCore or PatchCore+context fusion on most categories, it should be reported as an external VLM anomaly detection baseline that is not directly robust under this AD2 fixed protocol.

If WinCLIP outperforms our method on some categories, those categories should become failure-analysis cases rather than being ignored.

## 7. Errors

No category failed.

## 8. Output

- WinCLIP CSV: `results/stage14_strong_vlm_baselines/stage14_e_winclip_primary_fixed_protocol.csv`
- Raw JSON: `results/stage14_strong_vlm_baselines/stage14_e_winclip_primary_fixed_protocol_raw.json`
- Error log: `results/stage14_strong_vlm_baselines/stage14_e_winclip_primary_fixed_protocol_errors.txt`
- Report: `docs/stage14_strong_vlm_baselines/stage14_e_winclip_primary_fixed_protocol_report.md`