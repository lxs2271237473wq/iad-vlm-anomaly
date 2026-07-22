# Stage 22-B2a: Frozen SRB-QCR Transfer to VisA FastFlow

## Protocol

- parameters selected on: `VisA PatchCore category-LOCO`
- target: `VisA FastFlow`
- target labels used for parameter selection: `none`
- `w_max = 0.35`
- `q_quantile = 0.25`
- target unlabeled Q threshold: `0.488337`
- `tau_delta = 0.75`

## Summary

| Variant | Image AUROC | AP | Best F1 | Potential call rate |
|---|---:|---:|---:|---:|
| Detector only | 0.8955 | 0.9198 | 0.8448 | - |
| Naive detector-crop fusion | 0.9688 | 0.9750 | 0.9169 | - |
| Old Quality-Calibrated QCR | 0.9778 | 0.9822 | 0.9307 | - |
| Old Adaptive QCR | 0.9783 | 0.9827 | 0.9312 | - |
| SRB-QCR frozen transfer | 0.9003 | 0.9167 | 0.8484 | 0.4158 |

## Frozen-transfer deltas

- SRB minus detector: `+0.0048`
- SRB minus naive: `-0.0685`
- SRB minus old Quality QCR: `-0.0775`
- SRB minus old Adaptive QCR: `-0.0779`
- categories SRB > detector: `8/12`
- worst category delta vs detector: `-0.0029`
- potential calls saved: `0.5842`

Potential call saving remains an offline estimate,
not a measured runtime speedup.
