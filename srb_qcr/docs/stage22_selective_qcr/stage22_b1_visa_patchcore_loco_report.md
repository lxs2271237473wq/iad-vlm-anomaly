# Stage 22-B1: VisA PatchCore Category-LOCO SRB-QCR

## Protocol

- method: `Selective Reliability-Bounded QCR`
- development source: `VisA PatchCore`
- selection: leave one category out
- configurations per fold: `27`
- target-label use during selection: `none`
- detector/VLM inference rerun: `none`

## LOCO comparison

| Variant | Mean image AUROC | AP | Best F1 | Potential VLM call rate |
|---|---:|---:|---:|---:|
| Detector only | 0.9131 | 0.9242 | 0.8606 | - |
| Naive detector-crop fusion | 0.9616 | 0.9681 | 0.9130 | - |
| Old Quality-Calibrated QCR | 0.9718 | 0.9752 | 0.9319 | - |
| Old Adaptive QCR | 0.9722 | 0.9756 | 0.9320 | - |
| SRB-QCR | 0.9189 | 0.9227 | 0.8681 | 0.4117 |

## Main deltas

- SRB-QCR minus detector: `+0.0058`
- SRB-QCR minus naive fusion: `-0.0427`
- SRB-QCR minus old Quality QCR: `-0.0529`
- SRB-QCR minus old Adaptive QCR: `-0.0532`
- potential VLM call rate: `0.4117`
- potential calls saved: `0.5883`

## Fold selections

| Held-out category | Fallback | w_max | q quantile | q threshold | tau_delta | Dev AUROC | Dev delta vs detector |
|---|---|---:|---:|---:|---:|---:|---:|
| candle | False | 0.35 | 0.25 | 0.24434642685153096 | 0.75 | 0.9143 | +0.0069 |
| capsules | False | 0.35 | 0.25 | 0.24140247926151412 | 0.75 | 0.9386 | +0.0054 |
| cashew | False | 0.35 | 0.25 | 0.26254248173960815 | 0.75 | 0.9156 | +0.0066 |
| chewinggum | False | 0.35 | 0.25 | 0.2361108740578542 | 0.75 | 0.9137 | +0.0071 |
| fryum | False | 0.35 | 0.25 | 0.2408458987406098 | 0.75 | 0.9164 | +0.0064 |
| macaroni1 | False | 0.35 | 0.25 | 0.23828217850079597 | 0.75 | 0.9235 | +0.0054 |
| macaroni2 | False | 0.35 | 0.25 | 0.2376558375792334 | 0.75 | 0.9389 | +0.0068 |
| pcb1 | False | 0.35 | 0.25 | 0.24329970252391564 | 0.75 | 0.9179 | +0.0069 |
| pcb2 | False | 0.35 | 0.25 | 0.2677066139550875 | 0.75 | 0.9189 | +0.0067 |
| pcb3 | False | 0.35 | 0.25 | 0.24208995326546062 | 0.75 | 0.9186 | +0.0066 |
| pcb4 | False | 0.35 | 0.25 | 0.2533052107035941 | 0.75 | 0.9136 | +0.0069 |
| pipe_fryum | False | 0.35 | 0.25 | 0.23864054504341625 | 0.75 | 0.9138 | +0.0073 |

## Frozen transfer configuration

```json
{
  "method": "SRB-QCR",
  "selection_rule": "frequency, mean development rank, call rate, conservative tie order",
  "selected": {
    "w_max": 0.35,
    "q_quantile": 0.25,
    "tau_delta": 0.75,
    "selection_count": 12,
    "mean_dev_rank": 1.0,
    "mean_dev_call_rate": 0.41634174051458467
  },
  "all_candidates": [
    {
      "w_max": 0.35,
      "q_quantile": 0.25,
      "tau_delta": 0.75,
      "selection_count": 12,
      "mean_dev_rank": 1.0,
      "mean_dev_call_rate": 0.41634174051458467
    }
  ]
}
```

## Interpretation restriction

Potential call rate is an offline estimate based on
the pre-VLM quality gate. It is not yet a measured
wall-clock speedup.
