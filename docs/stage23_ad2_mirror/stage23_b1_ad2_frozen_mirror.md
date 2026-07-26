# Stage 23-B1: AD2 Four-Category Frozen Mirror Evaluation

## Protocol

- parameters selected on: `VisA PatchCore category-LOCO`
- target labels used for parameter selection: `none`
- rows: `243`
- frozen parameters: `{'w_max': 0.35, 'q_quantile': 0.25, 'tau_delta': 0.75}`
- non-inferiority margin: `-0.002`

## Summary

| ID | Method | Macro AUROC | Macro AP | Macro F1 | Pooled AUROC | Call rate |
|---|---|---:|---:|---:|---:|---:|
| D0 | Detector only | 0.7853 | 0.9186 | 0.8717 | 0.8060 | - |
| M0 | Crop VLM only | 0.6524 | 0.8318 | 0.8447 | 0.6608 | - |
| V3 | Naive detector-crop fusion | 0.8286 | 0.9327 | 0.8741 | 0.8473 | - |
| V4 | Old Quality-Calibrated QCR | 0.8191 | 0.9293 | 0.8811 | 0.8434 | - |
| V6 | Old Adaptive QCR | 0.8194 | 0.9294 | 0.8811 | 0.8437 | - |
| S1 | SRB-QCR frozen transfer | 0.7939 | 0.9232 | 0.8683 | 0.8144 | 0.7490 |

## Main deltas

- SRB minus detector: `+0.0086`
- SRB minus crop VLM: `+0.1415`
- SRB minus naive: `-0.0347`
- SRB minus old Quality QCR: `-0.0252`
- SRB minus old Adaptive QCR: `-0.0255`
- categories SRB > detector: `3/4`
- worst category delta vs detector: `-0.0037`
- potential calls saved: `0.2510`

## Bootstrap

| Comparison | Delta | CI low | CI high | P(delta>0) | p | P(non-inferior) | CI non-inferior |
|---|---:|---:|---:|---:|---:|---:|---|
| SRB-QCR vs detector | +0.0086 | -0.0066 | +0.0241 | 0.8712 | 0.257600 | 0.9160 | False |
| SRB-QCR vs crop VLM | +0.1415 | +0.0350 | +0.2466 | 0.9966 | 0.006800 | 0.9970 | True |
| SRB-QCR vs naive fusion | -0.0347 | -0.0958 | +0.0252 | 0.1280 | 0.256000 | 0.1478 | False |
| Quality QCR vs naive fusion | -0.0095 | -0.0308 | +0.0102 | 0.1808 | 0.361600 | 0.2402 | False |
| SRB-QCR vs old Quality QCR | -0.0252 | -0.0845 | +0.0362 | 0.2104 | 0.420800 | 0.2342 | False |
| SRB-QCR vs old Adaptive QCR | -0.0255 | -0.0848 | +0.0362 | 0.2076 | 0.415200 | 0.2300 | False |

## Restrictions

- AD2 has only four categories; treat statistical conclusions as supplementary.
- Potential call saving is offline until Stage 23-C measured runtime.
- source: `results/stage22_selective_qcr/stage22_b2b_ad2_frozen_predictions.csv`