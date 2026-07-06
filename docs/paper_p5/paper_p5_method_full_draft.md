# Paper Stage P5: Method Section Full Draft

## 1. Method Overview

We propose **Quality-Calibrated QCR**, a localization-guided VLM reasoning framework for image-level industrial anomaly recognition. The method starts from detector localization evidence, converts this evidence into candidate image crops, obtains crop-level VLM abnormality scores, and calibrates those scores using candidate quality. The key design principle is that crop-level VLM evidence should not be trusted uniformly: it should contribute strongly only when the candidate region is reliable.

The method family contains two paper-facing variants. The main method core is **Quality-Calibrated QCR**, which uses candidate quality to calibrate crop-level VLM evidence. An optional full variant, **Quality-Calibrated QCR with adaptive consistency refinement**, adds a small reliability-gated consistency bonus. Fixed Q+C fusion is retained only as a diagnostic ablation and is not the final method.

## 2. Notation

| Symbol | Name | Definition | Range / Type |
|---|---|---|---|
| `x` | input image | industrial test image | image |
| `A` | localization evidence | detector-produced anomaly localization map or candidate evidence | map / spatial evidence |
| `D` | detector anomaly score | normalized image-level anomaly score from detector evidence | [0, 1] |
| `C = {c_i}` | candidate crop set | candidate regions generated from localization evidence | set of image crops |
| `m_i` | candidate VLM abnormality score | VLM abnormality score for candidate crop c_i | [0, 1] |
| `M` | aggregated crop VLM score | fixed aggregation of crop-level VLM abnormality evidence under the selected protocol | [0, 1] |
| `Q` | candidate quality | localization-derived reliability of the candidate crop evidence | [0, 1] |
| `K` | detector-VLM high-high consistency | consistency signal indicating jointly high detector and VLM abnormal evidence | [0, 1] |
| `S_naive` | naive detector-crop fusion score | unreliability-aware baseline score | [0, 1] |
| `S_quality` | Quality-Calibrated QCR score | main quality-calibrated anomaly score | [0, 1] after normalization |
| `S_adaptive` | adaptive-refinement score | quality-calibrated score plus conservative gated consistency bonus | [0, 1] after normalization |

## 3. Localization-guided Candidate Generation

Given an input image `x`, an anomaly detector produces localization evidence `A` and a normalized detector anomaly score `D`. The localization evidence is used to generate a set of candidate crops:

```text
Detector(x) -> A, D
A -> C = {c_i}
```

The candidate set focuses the VLM on spatial regions where abnormal evidence is likely to appear. This step is not claimed as the main novelty by itself. Its role is to convert detector localization into candidate-level visual evidence.

## 4. Crop-level VLM Anomaly Evidence

Each candidate crop `c_i` is evaluated by the VLM using abnormality-oriented prompts, producing a crop-level score `m_i`. The crop scores are aggregated under a fixed protocol to produce `M`, the aggregated crop-level VLM abnormality score:

```text
VLM(c_i, prompts) -> m_i
Aggregate({m_i}) -> M
```

The aggregated score `M` is useful but not sufficient. A high VLM score can be unreliable if the crop is poorly localized, too broad, too small, or visually ambiguous. Therefore, the method calibrates VLM evidence using candidate quality.

## 5. Candidate Quality Calibration

Candidate quality `Q` measures the reliability of the selected candidate evidence. The naive detector-crop fusion baseline is:

```text
S_naive = 0.5D + 0.5M
```

This baseline treats detector and VLM evidence as equally reliable. Quality-Calibrated QCR instead modulates the VLM contribution by candidate quality:

```text
S_quality = 0.5D + 0.5 * M * (0.5 + 0.5Q)
```

When `Q` is high, the crop-level VLM evidence contributes more strongly. When `Q` is low, the VLM contribution is reduced, preventing unreliable crops from dominating the image-level anomaly score. This quality-calibrated score is the main method core.

## 6. Diagnostic Fixed Q+C Fusion

We also evaluate a fixed quality-consistency fusion variant:

```text
S_fixed = 0.4D + 0.4M + 0.1Q + 0.1K
```

where `K` is a detector-VLM high-high consistency signal. This variant is useful diagnostically because it tests whether adding consistency can improve peak performance. However, robustness analysis shows that fixed consistency is not stable across protocols. Therefore, `S_fixed` is not used as the final method.

## 7. Adaptive Consistency Refinement

To avoid the instability of fixed consistency, the final refinement applies consistency only through a conservative reliability gate. We define:

```text
agreement = 1 - |D - M|
mutual_anomaly_evidence = min(D, M)
gate = Q * K * agreement * mutual_anomaly_evidence
S_adaptive = S_quality + 0.05 * gate
```

The coefficient `0.05` is intentionally small. The adaptive term is not intended to be the main performance source. Its role is to add a consistency bonus only when candidate quality, detector evidence, VLM evidence, and detector-VLM agreement are jointly reliable.

The full variant can be written as:

```text
Quality-Calibrated QCR with adaptive consistency refinement
```

but the core contribution remains candidate quality calibration.

## 8. Algorithm

```text
Algorithm 1: Quality-Calibrated QCR

Input:
    image x
    anomaly detector
    VLM scoring function

Output:
    image-level anomaly score S

1. Run detector on x to obtain localization evidence A and detector score D.
2. Generate candidate crop set C = {c_i} from A using the fixed candidate protocol.
3. Score each crop c_i with the VLM and aggregate crop scores into M.
4. Estimate candidate quality Q from localization-derived candidate evidence.
5. Compute the quality-calibrated score:
       S_quality = 0.5D + 0.5 * M * (0.5 + 0.5Q)
6. Optionally compute adaptive consistency refinement:
       agreement = 1 - |D - M|
       mutual_anomaly_evidence = min(D, M)
       gate = Q * K * agreement * mutual_anomaly_evidence
       S_adaptive = S_quality + 0.05 * gate
7. Use S_quality as the main method score, or S_adaptive when reporting the adaptive-refinement variant.
```

## 9. Method Components

| ID | Component | Role | Formula / Operation | Paper Status |
|---|---|---|---|---|
| M1 | Detector localization | Generate localization evidence and detector anomaly score. | `Detector(x) -> A, D` | use |
| M2 | Candidate crop generation | Convert localization evidence into candidate regions for VLM scoring. | `A -> C = {c_i}` | use |
| M3 | Crop-level VLM scoring | Obtain localized visual-language abnormality evidence. | `VLM(c_i, prompts) -> m_i; aggregate {m_i} -> M` | use |
| M4 | Candidate quality calibration | Main effective method component; calibrates crop-level VLM evidence. | `S_quality = 0.5D + 0.5 * M * (0.5 + 0.5Q)` | main_method_core |
| M5 | Fixed Q+C fusion | Diagnostic comparison only. | `S_fixed = 0.4D + 0.4M + 0.1Q + 0.1K` | diagnostic_only |
| M6 | Adaptive consistency refinement | Conservative optional refinement on top of quality-calibrated core. | `S_adaptive = S_quality + 0.05 * Q*K*(1-|D-M|)*min(D,M)` | use_with_caution |

## 10. Claim Boundaries

| ID | Topic | Safe Statement | Forbidden Statement |
|---|---|---|---|
| B1 | method name | Use Quality-Calibrated QCR as the main method family. | Use fixed Q+C QCR-U as the final method. |
| B2 | quality calibration | Candidate quality calibration is the main effective component. | Candidate quality improves every category and every case. |
| B3 | adaptive consistency | Adaptive consistency is a conservative refinement. | Adaptive consistency is the main source of improvement. |
| B4 | fixed consistency | Fixed Q+C is diagnostic because robustness is insufficient. | Fixed consistency is universally beneficial. |
| B5 | localization | Localization evidence is used for candidate generation and reliability calibration. | The method achieves pixel-level segmentation SOTA. |
| B6 | VLM reasoning | The method performs localization-guided VLM anomaly recognition. | The method explains manufacturing causes or full anomaly understanding. |

## 11. Method Section Writing Notes

The method section should emphasize the following:

- The paper is not a detector replacement paper.
- The paper is not a generic VLM anomaly understanding paper.
- The main contribution is reliability calibration of localization-guided VLM evidence.
- Candidate quality is the main effective component.
- Adaptive consistency is a conservative refinement.
- Fixed Q+C is a diagnostic ablation, not the final method.

## 12. Next Step

Next stage:

```text
Paper Stage P6: assemble first full paper draft skeleton
```

P6 should combine P2 Introduction/Contributions, P4 Related Work, P5 Method, and P3 Experiments into a single paper skeleton.

## 13. Outputs

- `docs/paper_p5/paper_p5_method_full_draft.md`
- `results/paper_p5/paper_p5_notation_table.csv`
- `results/paper_p5/paper_p5_method_components.csv`
- `results/paper_p5/paper_p5_algorithm_steps.csv`
- `results/paper_p5/paper_p5_method_claim_boundaries.csv`
