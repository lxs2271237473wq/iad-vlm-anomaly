# Stage 16-C Final Method Claims

## 1. Decision

The final method should not be written as fixed QCR-U or as a consistency-driven method.

The paper-facing method name should be:

```text
Quality-Calibrated QCR
```

A longer descriptive name can be:

```text
Quality-Calibrated Localization-Guided VLM Reasoning
```

Adaptive consistency can be retained only as a small refinement:

```text
Quality-Calibrated QCR with Adaptive Consistency Refinement
```

## 2. Why this downgrade is necessary

Stage 16-B shows that adaptive QCR-U is consistently better than naive fusion, but almost all useful gain over naive fusion comes from quality calibration.

The adaptive consistency term improves over quality-only by only a very small margin. Therefore, consistency cannot be claimed as the main innovation.

## 3. Final Claims Table

| Claim ID | Type | Claim | Paper Status |
|---|---|---|---|
| C1 | final_method_name | Use Quality-Calibrated QCR as the main method name. | use |
| C2 | main_effective_component | Candidate quality calibration is the main effective component. | use |
| C3 | auxiliary_component | Adaptive consistency is a conservative refinement, not the main source of improvement. | use_with_caution |
| C4 | rejected_claim | Do not claim fixed quality-consistency fusion as the final method. | reject |
| C5 | rejected_claim | Do not claim consistency is universally beneficial. | reject |
| C6 | safe_paper_claim | Localization-guided VLM reasoning becomes substantially stronger when crop evidence is calibrated by candidate quality. | use |
| C7 | safe_paper_claim | The proposed method should be positioned as reliability calibration for localization-guided VLM anomaly recognition. | use |

## 4. Safe contribution wording

The safest contribution wording is:

```text
We propose a quality-calibrated localization-guided VLM reasoning framework for industrial anomaly recognition. Instead of directly fusing detector and VLM scores, the method calibrates crop-level VLM evidence using candidate quality derived from anomaly localization. We further study detector-VLM consistency and find that fixed consistency is not robust; therefore, consistency is used only as a conservative adaptive refinement.
```

## 5. Claims to avoid

- Do not claim consistency is universally beneficial.
- Do not claim fixed Q+C fusion is the final method.
- Do not claim adaptive consistency is the main source of improvement.
- Do not claim this solves full industrial anomaly understanding.
- Do not claim manufacturing-cause reasoning.

## 6. Next step

Next stage should generate the final paper-facing main table using this method naming:

```text
Stage 16-D: paper-facing final comparison table
```

That table should compare:

- WinCLIP fixed protocol
- EfficientAD-30 fixed-budget
- PatchCore
- detector-only
- crop VLM
- naive fusion
- quality-calibrated QCR
- quality-calibrated QCR + adaptive consistency refinement
