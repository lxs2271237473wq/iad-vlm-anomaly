# Stage 15-E Primary Unified Baseline Comparison

## 1. Purpose

This stage merges the four-category EfficientAD-30 fixed-budget baseline with the existing Stage 14-E primary external baseline comparison.

The goal is to check whether the newly added modern detector baseline changes the current research conclusion before running any 100-epoch sensitivity experiment.

## 2. Included Methods

- WinCLIP fixed protocol
- full-image VLM
- context-aware VLM
- PatchCore
- EfficientAD-30 fixed-budget
- PatchCore + context VLM, LOCO
- PatchCore + context VLM, same-set

Important: `same-set` fusion is an upper-bound diagnostic and must not be overclaimed as the final fair protocol.

## 3. Mean Image AUROC Ranking

| Rank | Method | Mean Image AUROC | Mean Pixel AUROC | Fairness tag |
|---:|---|---:|---:|---|
| 1 | PatchCore + context VLM, same-set | 0.8453 |  | mean_summary |
| 2 | PatchCore + context VLM, LOCO | 0.8210 |  | mean_summary |
| 3 | PatchCore | 0.7853 |  | mean_summary |
| 4 | EfficientAD-30 fixed-budget | 0.7604 | 0.8093 | mean_summary |
| 5 | context-aware VLM | 0.7101 |  | mean_summary |
| 6 | full-image VLM | 0.6459 |  | mean_summary |
| 7 | WinCLIP fixed protocol | 0.6138 | 0.8203 | mean_summary |

## 4. Main Deltas

- LOCO fusion minus EfficientAD-30: `+0.0606` mean image AUROC.
- EfficientAD-30 minus PatchCore: `-0.0249` mean image AUROC.
- EfficientAD-30 minus context-aware VLM: `+0.0503` mean image AUROC.
- EfficientAD-30 minus WinCLIP fixed protocol: `+0.1467` mean image AUROC.

## 5. Per-category Result Table

| Category | Method | Image AUROC | Pixel AUROC | Fairness tag |
|---|---|---:|---:|---|
| fruit_jelly | WinCLIP fixed protocol | 0.4667 | 0.8327 | primary_or_reference |
| fruit_jelly | full-image VLM | 0.7533 |  | primary_or_reference |
| fruit_jelly | context-aware VLM | 0.8567 |  | primary_or_reference |
| fruit_jelly | PatchCore | 0.7167 |  | primary_or_reference |
| fruit_jelly | EfficientAD-30 fixed-budget | 0.8433 | 0.7894 | fixed_budget_detector_baseline |
| fruit_jelly | PatchCore + context VLM, LOCO | 0.8333 |  | primary_or_reference |
| fruit_jelly | PatchCore + context VLM, same-set | 0.8933 |  | upper_bound_diagnostic |
| sheet_metal | WinCLIP fixed protocol | 0.7074 | 0.8140 | primary_or_reference |
| sheet_metal | full-image VLM | 0.7130 |  | primary_or_reference |
| sheet_metal | context-aware VLM | 0.6574 |  | primary_or_reference |
| sheet_metal | PatchCore | 0.7463 |  | primary_or_reference |
| sheet_metal | EfficientAD-30 fixed-budget | 0.7333 | 0.7510 | fixed_budget_detector_baseline |
| sheet_metal | PatchCore + context VLM, LOCO | 0.7481 |  | primary_or_reference |
| sheet_metal | PatchCore + context VLM, same-set | 0.7556 |  | upper_bound_diagnostic |
| vial | WinCLIP fixed protocol | 0.8795 | 0.8188 | primary_or_reference |
| vial | full-image VLM | 0.6876 |  | primary_or_reference |
| vial | context-aware VLM | 0.6834 |  | primary_or_reference |
| vial | PatchCore | 0.8732 |  | primary_or_reference |
| vial | EfficientAD-30 fixed-budget | 0.9099 | 0.9281 | fixed_budget_detector_baseline |
| vial | PatchCore + context VLM, LOCO | 0.9224 |  | primary_or_reference |
| vial | PatchCore + context VLM, same-set | 0.9256 |  | upper_bound_diagnostic |
| walnuts | WinCLIP fixed protocol | 0.4015 | 0.8155 | primary_or_reference |
| walnuts | full-image VLM | 0.4296 |  | primary_or_reference |
| walnuts | context-aware VLM | 0.6430 |  | primary_or_reference |
| walnuts | PatchCore | 0.8052 |  | primary_or_reference |
| walnuts | EfficientAD-30 fixed-budget | 0.5552 | 0.7687 | fixed_budget_detector_baseline |
| walnuts | PatchCore + context VLM, LOCO | 0.7800 |  | primary_or_reference |
| walnuts | PatchCore + context VLM, same-set | 0.8067 |  | upper_bound_diagnostic |

## 6. Interpretation

EfficientAD-30 is a useful modern non-VLM detector baseline, but it does not invalidate the current localization-guided VLM fusion direction.

The fairer fusion result, `PatchCore + context VLM, LOCO`, should be compared against EfficientAD-30. The same-set fusion result should remain an upper-bound diagnostic.

If EfficientAD-30 is close to or below the LOCO fusion mean, the next priority is not immediately a full four-category EfficientAD-100 run. A single-category 100-epoch sensitivity check is enough to test whether the 30-epoch budget severely underestimates EfficientAD.

## 7. Outputs

- CSV: `results/stage15_modern_detector_baselines/stage15_e_primary_unified_baseline_comparison.csv`
- Report: `docs/stage15_modern_detector_baselines/stage15_e_primary_unified_baseline_comparison_report.md`
