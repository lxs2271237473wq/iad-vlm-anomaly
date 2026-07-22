# Stage 22-B2b: Frozen SRB-QCR Transfer to AD2

## Protocol

- parameter source: `VisA PatchCore category-LOCO`
- transfer target: `AD2 four categories`
- AD2 labels used for parameter selection: `none`
- `w_max = 0.35`
- `q_quantile = 0.25`
- target unlabeled Q threshold: `0.466024`
- `tau_delta = 0.75`

## Summary

| Variant | Mean category AUROC | Pooled AUROC | Mean AP | Mean Best F1 | Potential call rate |
|---|---:|---:|---:|---:|---:|
| Detector only | 0.7853 | 0.8060 | 0.9186 | 0.8717 | - |
| Naive detector-crop fusion | 0.8286 | 0.8473 | 0.9327 | 0.8741 | - |
| Old Quality-Calibrated QCR | 0.8191 | 0.8434 | 0.9293 | 0.8811 | - |
| Old Adaptive QCR | 0.8194 | 0.8437 | 0.9294 | 0.8811 | - |
| SRB-QCR frozen transfer | 0.7939 | 0.8144 | 0.9232 | 0.8683 | 0.7490 |

## Frozen-transfer deltas

- SRB minus detector: `+0.0086`
- SRB minus naive: `-0.0347`
- SRB minus old Quality QCR: `-0.0252`
- SRB minus old Adaptive QCR: `-0.0255`
- categories SRB > detector: `3/4`
- worst category delta vs detector: `-0.0037`
- potential calls saved: `0.2510`
- frozen decision: `retain_as_cross_dataset_deployment_mode`

Potential call saving remains an offline estimate.
It is not a measured runtime speedup.
