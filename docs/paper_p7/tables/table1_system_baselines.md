# Table 1. System-level strong baseline comparison

| Rank | Method | Mean AUROC | Paper role | Protocol tag |
| --- | --- | --- | --- | --- |
| 1 | PatchCore + context VLM, same-set | 0.8453 | upper_bound_diagnostic_only | mean_summary |
| 2 | PatchCore + context VLM, LOCO | 0.8210 | primary_fair_system_result | mean_summary |
| 3 | PatchCore | 0.7853 | classic_detector_baseline | mean_summary |
| 4 | EfficientAD-30 fixed-budget | 0.7604 | modern_detector_fixed_budget_baseline | mean_summary |
| 5 | context-aware VLM | 0.7101 | vlm_baseline | mean_summary |
| 6 | full-image VLM | 0.6459 | vlm_baseline | mean_summary |
| 7 | WinCLIP fixed protocol | 0.6138 | external_vlm_anomaly_baseline | mean_summary |

**Note.** LOCO is the fair system-level result. Same-set is an upper-bound diagnostic only.
