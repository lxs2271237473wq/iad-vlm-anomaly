# Stage 11-E Multi-category Evidence Consolidation

## 1. Purpose

This report consolidates Stage 11-B detector quality, Stage 11-C candidate quality, and Stage 11-D VLM full-image versus crop reasoning results.
It does not run any model, generate crops, or modify datasets.

## 2. Main Result

Across all primary categories, the best VLM method is `context_1.50_topk_mean` with AUROC `0.6036`, compared with full-image AUROC `0.5201`. The AUROC gain is `0.0835`.

This supports the claim that localization-guided context-aware crops can improve VLM anomaly reasoning, but the effect is category-dependent.

## 3. Category-level Evidence

| Category | Best VLM | Full AUROC | Best VLM AUROC | ΔAUROC | Best context | Context Δ | Decision |
|---|---|---:|---:|---:|---|---:|---|
| fruit_jelly | context_1.50_topk_mean | 0.7533 | 0.8567 | 0.1033 | context_1.50_topk_mean | 0.1033 | main_positive_context |
| sheet_metal | full_image | 0.7130 | 0.7130 | 0.0000 | context_1.50_topk_max | -0.0556 | negative_or_full_stronger |
| vial | full_image | 0.6876 | 0.6876 | 0.0000 | context_1.50_top1 | -0.0042 | negative_or_full_stronger |
| walnuts | context_1.50_topk_mean | 0.4296 | 0.6430 | 0.2133 | context_1.50_topk_mean | 0.2133 | main_positive_context |

## 4. Candidate Quality Reference

| Category | Candidate coverage | Tight GT coverage | Context GT coverage |
|---|---:|---:|---:|
| fruit_jelly | 1.0000 | 0.4302 | 0.8661 |
| sheet_metal | 1.0000 | 0.1132 | 0.2509 |
| vial | 1.0000 | 0.4275 | 0.8016 |
| walnuts | 1.0000 | 0.3606 | 0.4851 |

## 5. Vial Cross-stage Consistency Note

Stage 10-G vial: full-image AUROC `0.6488`, context_1.50_top1 AUROC `0.7746`, delta `0.1258`.

Stage 11-D vial: full-image AUROC `0.6876`, best context AUROC `0.6834`, context delta `-0.0042`.

This discrepancy should be treated as implementation/candidate-construction sensitivity, not as a direct contradiction of the method idea.

## 6. Paper-level Interpretation

The strongest claim is not that every crop improves every category.
The defensible claim is:

```text
PatchCore localization can serve as a visual bridge for VLM anomaly reasoning when candidate regions preserve sufficient object context; the benefit is category- and candidate-quality-dependent.
```

The current evidence supports context-aware crop reasoning on the aggregate primary set, fruit_jelly, and walnuts.
sheet_metal should be discussed as a failure/limitation case, and vial requires a consistency check between the Stage 10 single-category pipeline and Stage 11 batch pipeline.

## 7. Next Step

The next step should not be fabric expansion yet. First, inspect the Stage 10 vs Stage 11 vial candidate construction difference and decide whether Stage 11-C needs to reuse the Stage 10 candidate policy exactly.

## 8. Output

- Evidence table: `results/stage11_mvtecad2_multicategory/stage11_e_multicategory_evidence_table.csv`
- Report: `docs/stage11_mvtecad2_multicategory/stage11_e_multicategory_evidence_report.md`