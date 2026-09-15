# Stage 20-E: Final AnomalyCLIP Integration

## 1. Locked protocol

- Stage 20-D status: `success`
- return code: `0`
- dataset: `AD2-four`
- categories: `fruit_jelly`, `sheet_metal`, `vial`, `walnuts`
- AD2-specific AnomalyCLIP training: `none`
- checkpoint: fixed and recorded in Stage 20-D

## 2. Locked AnomalyCLIP result

- mean image AUROC: `0.5390`
- mean image AP: `0.7709999999999999`
- mean pixel AUROC: `0.701`
- mean pixel AUPRO: `0.402`

## 3. Main comparison

- LOCO fusion image AUROC: `0.8210`
- AnomalyCLIP image AUROC: `0.5390`
- LOCO minus AnomalyCLIP: `+0.2820`
- decision: `loco_above_anomalyclip`

The LOCO fusion is stronger than the fixed AnomalyCLIP baseline on this adapted AD2-four protocol. The claim must remain protocol-specific. AnomalyCLIP=0.5390, LOCO=0.8210, delta_LOCO_minus_AnomalyCLIP=+0.2820. delta_AnomalyCLIP_minus_PatchCore=-0.2463. delta_AnomalyCLIP_minus_EfficientAD30=-0.2214. delta_AnomalyCLIP_minus_WinCLIP=-0.0748.

## 4. Final system table

| Rank | Method | Image AUROC | Role |
|---:|---|---:|---|
| 1 | PatchCore + context VLM, same-set | 0.8453 | upper_bound_diagnostic_only |
| 2 | PatchCore + context VLM, LOCO | 0.8210 | primary_fair_system_result |
| 3 | PatchCore | 0.7853 | classic_detector_baseline |
| 4 | EfficientAD-30 fixed-budget | 0.7604 | modern_detector_fixed_budget_baseline |
| 5 | context-aware VLM | 0.7101 | vlm_baseline |
| 6 | full-image VLM | 0.6459 | vlm_baseline |
| 7 | WinCLIP fixed protocol | 0.6138 | external_vlm_anomaly_baseline |
| 8 | AnomalyCLIP fixed checkpoint | 0.5390 | external VLM anomaly baseline |

## 5. Claim decision

The paper should report the numerical comparison directly,
but all superiority wording must remain limited to the
implemented AD2-four fixed-checkpoint protocol.

## 6. Next step

Update the paper-facing baseline table, Results section,
abstract, and limitations using this locked result.
