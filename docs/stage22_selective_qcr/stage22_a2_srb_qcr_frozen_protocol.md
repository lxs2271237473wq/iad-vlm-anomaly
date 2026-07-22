# Stage 22-A2: Frozen SRB-QCR Protocol

## Status

- status: `frozen_before_stage22_b_results`
- method: `Selective Reliability-Bounded QCR`
- short name: `SRB-QCR`
- source: `results/stage9_qcr_u/stage9_a1_qcr_u_fusion_predictions.csv`
- source SHA-256: `eccf302b453c544f6a9a73846a62eb8d3a7c547f3631d3d9a52500b75a588d82`

## Signal mapping

- `D = detector_score_norm`
- `M = vlm_score_norm`
- `Q = candidate_quality_norm`
- `Y = is_anomaly_final`
- sample ID: `image_key`
- group: `category`

## Frozen method

```text
G_pre = I(has_candidate) · I(not fallback) · I(Q >= tau_q)
A     = clip(1 - |D-M| / tau_delta, 0, 1)
w     = w_max · G_pre · Q · A
S_SRB = D + w(M-D)
```

with:

```text
0 <= w <= w_max < 0.5
|S_SRB-D| <= w_max |M-D| <= w_max
```

A missing or invalid candidate forces `w=0` and
`S_SRB=D`.

## Frozen grid

- `w_max`: `[0.15, 0.25, 0.35]`
- `tau_q`: `[0.25, 0.5, 0.75]`
- `tau_delta`: `[0.25, 0.5, 0.75]`
- total configurations: `27`

## Selection

- development source: VisA PatchCore
- selection: leave one category out
- primary criterion: development macro image AUROC
- detector non-inferiority tolerance: `0.002`
- AUROC tie tolerance: `0.001`
- tie priority: lower call rate, lower `w_max`,
  higher `tau_q`, lower `tau_delta`
- no eligible configuration: detector-only fallback

## Frozen transfer tests

- VisA FastFlow
- AD2 four categories
- complete MVTec AD when cached signals are available

## Data audit

- original rows: `155664`
- deduplicated rows: `25944`
- primary rows: `4324`
- primary categories: `12`
- PatchCore development backbone(s): `PatchCore`

## Claim restriction

Offline evaluation may report potential VLM call rate.
It may not claim measured runtime acceleration until
an execution-level timing experiment is completed.
