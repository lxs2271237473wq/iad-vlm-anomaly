# Stage 11-G Unified Experiment Policy

## 1. Purpose

This report fixes the experimental interpretation policy after the Stage 11-F3 vial image-set overlap audit.
It does not run PatchCore, VLM inference, or crop generation.

## 2. Key Decision

The paper-level main evidence should use Stage 11-D/E, not a mixed comparison between Stage 10-G and Stage 11-D.

Stage 10-G vial remains useful as a historical single-category observation, but it should not be placed in the same fair-comparison table as Stage 11-D vial.

## 3. Why Stage 10-G and Stage 11-D vial are not directly comparable

| Item | Value |
|---|---:|
| Best alignment | stage10_stem_vs_stage11_source_stem |
| Best intersection | 39 |
| Stage 10 unique keys | 60 |
| Stage 11 unique keys | 66 |
| Jaccard | 0.4483 |
| Directly comparable | False |

## 4. Vial numbers should be interpreted separately

| Source | Full-image AUROC | Context AUROC | Delta | Interpretation |
|---|---:|---:|---:|---|
| Stage 10-G vial | 0.6488 | 0.7746 | 0.1258 | Historical single-category observation |
| Stage 11-D vial | 0.6876 | 0.6834 | -0.0042 | Unified multi-category pipeline result |

## 5. Main Stage 11 result

| Scope | Full AUROC | Best VLM Method | Best VLM AUROC | Delta |
|---|---:|---|---:|---:|
| ALL_PRIMARY | 0.5201 | context_1.50_topk_mean | 0.6036 | 0.0835 |

## 6. Final paper-level wording

Recommended claim:

```text
PatchCore localization can serve as a visual bridge for VLM anomaly reasoning when candidate regions preserve sufficient object context. On the unified MVTec AD 2 primary-category evaluation, context-aware crop aggregation improves over full-image VLM prompting, while category-level failures reveal sensitivity to candidate quality and object context.
```

Avoid this claim:

```text
Context-aware crops consistently improve every category.
```

## 7. Policy Table

| Item | Decision | Reason |
|---|---|---|
| main_experiment_source | Use Stage 11-D/E unified multi-category pipeline as the paper-level main evidence. | Stage 11 uses one unified data adapter, candidate construction script, and VLM scoring script across primary AD2 categories. |
| stage10_vial_status | Keep Stage 10-G vial as historical single-category observation, not as a directly comparable main-table result. | Stage 11-F3 shows Stage 10-G and Stage 11-D vial image sets are not sufficiently aligned. |
| vial_cross_stage_comparison | Do not claim Stage 10-G and Stage 11-D vial numbers contradict or reproduce each other. | Best image-set overlap intersection is 39, directly_comparable=False. |
| main_claim | Claim category-dependent benefit of localization-guided context-aware crops, strongest on aggregate primary set, fruit_jelly, and walnuts. | ALL_PRIMARY best VLM method is context_1.50_topk_mean with AUROC gain 0.0835. |
| limitation_cases | Use sheet_metal and Stage 11 vial as limitation/failure-analysis cases, not as evidence against the whole method. | Candidate quality and image-set/pipeline sensitivity affect crop-based VLM reasoning. |

## 8. Next Step

After this policy is committed, the next practical step is to decide whether to include the secondary category `fabric` under the same Stage 11 pipeline, or to move directly to a paper-ready Stage 11 final table and method narrative.

## 9. Output

- Policy table: `results/stage11_mvtecad2_multicategory/stage11_g_unified_experiment_policy_table.csv`
- Report: `docs/stage11_mvtecad2_multicategory/stage11_g_unified_experiment_policy_report.md`