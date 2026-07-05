# Stage 16-C Final Method Claims

## 1. Decision

The final method should not be written as fixed QCR-U or as a consistency-driven method.

The paper-facing method family is:

```text
Quality-Calibrated QCR
```

A more descriptive paper title/method phrase is:

```text
Quality-Calibrated Localization-Guided VLM Reasoning
```

The safest extended method name is:

```text
Quality-Calibrated QCR with adaptive consistency refinement
```

## 2. Why this decision is necessary

Stage 16-B shows that the adaptive variant is consistently better than naive fusion, but almost all useful gain comes from quality calibration.

Adaptive consistency is retained only as a conservative gated refinement. It should not be described as the main performance source.

## 3. Primary Protocol Evidence

| Comparison | Wins | Total | Win Rate | Mean Delta | Median Delta | Min Delta | Max Delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| quality minus naive | 2 | 2 | 1.0000 | +0.0096 | +0.0096 | +0.0090 | +0.0102 |
| adaptive minus naive | 2 | 2 | 1.0000 | +0.0100 | +0.0100 | +0.0094 | +0.0105 |
| adaptive minus quality | 2 | 2 | 1.0000 | +0.0004 | +0.0004 | +0.0003 | +0.0005 |
| fixed Q+C minus quality | 2 | 2 | 1.0000 | +0.0043 | +0.0043 | +0.0022 | +0.0064 |
| adaptive minus fixed Q+C | 0 | 2 | 0.0000 | -0.0039 | -0.0039 | -0.0059 | -0.0018 |

## 4. All-Protocol Evidence From Stage 16-B

| Comparison | Wins | Total | Win Rate | Mean Delta | Median Delta | Min Delta | Max Delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| quality minus naive | 12 | 12 | 1.0000 | +0.0365 | +0.0248 | +0.0079 | +0.0808 |
| adaptive minus naive | 12 | 12 | 1.0000 | +0.0371 | +0.0253 | +0.0082 | +0.0816 |
| adaptive minus quality | 12 | 12 | 1.0000 | +0.0006 | +0.0005 | +0.0003 | +0.0012 |
| adaptive minus fixed Q+C | 6 | 12 | 0.5000 | +0.0099 | +0.0042 | -0.0059 | +0.0342 |

## 5. Final Claims Table

| Claim ID | Type | Claim | Paper Status |
|---|---|---|---|
| C1 | final_method_name | Use Quality-Calibrated QCR as the main paper-facing method family. | use |
| C2 | main_effective_component | Candidate quality calibration is the main effective component. | use |
| C3 | auxiliary_component | Adaptive consistency is a conservative refinement, not the main source of improvement. | use_with_caution |
| C4 | rejected_claim | Do not claim fixed quality-consistency fusion as the final method. | reject |
| C5 | rejected_claim | Do not claim consistency is universally beneficial. | reject |
| C6 | safe_paper_claim | Localization-guided VLM evidence becomes more reliable when crop evidence is calibrated by candidate quality. | use |
| C7 | safe_paper_claim | Adaptive consistency can be retained as a reliability-gated refinement that avoids overcommitting to unstable fixed consistency. | use_with_caution |
| C8 | final_recommendation | Use Quality-Calibrated QCR as the main method; describe adaptive consistency as a small gated refinement. | use |

## 6. Safe Contribution Wording

Use this wording in the paper:

```text
We propose a quality-calibrated localization-guided VLM reasoning framework for industrial anomaly recognition. Instead of directly fusing detector and VLM scores, the method calibrates crop-level VLM evidence using candidate quality derived from anomaly localization. We further analyze detector-VLM consistency and find that fixed consistency is not robust; therefore, consistency is used only as a conservative adaptive refinement.
```

## 7. Claims to Avoid

- Do not claim fixed Q+C fusion is the final method.
- Do not claim consistency is universally beneficial.
- Do not claim adaptive consistency is the main source of improvement.
- Do not claim full industrial anomaly understanding.
- Do not claim manufacturing-cause reasoning.
- Do not claim pixel-level segmentation SOTA.

## 8. Next Step

Next stage:

```text
Stage 16-D: Paper-facing final comparison table
```

Stage 16-D should compare the final method family against strong baselines and earlier fusion variants in one table.

## 9. Outputs

- `results/stage16_qcru_ablation/stage16_c_final_method_claims.csv`
- `docs/stage16_qcru_ablation/stage16_c_final_method_claims_report.md`
