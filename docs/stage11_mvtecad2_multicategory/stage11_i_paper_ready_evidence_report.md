# Stage 11-I Paper-ready Evidence Tables

## 1. Purpose

This stage converts the Stage 11 multi-category experiments into paper-ready evidence tables.
It does not train models, run VLM inference, generate crops, or modify datasets.

## 2. Main Claim Table

| Scope / Category | Role | Full-image AUROC | Reported Method | Method AUROC | ΔAUROC vs full | PatchCore reference | Paper usage |
|---|---|---:|---|---:|---:|---:|---|
| ALL_PRIMARY | main_aggregate_evidence | 0.5201 | context_1.50_topk_mean | 0.6036 | 0.0835 | 0.8087 | Main positive evidence: context-aware crop aggregation improves over full-image VLM prompting on the unified primary set. |
| fruit_jelly | primary_category | 0.7533 | context_1.50_topk_mean | 0.8567 | 0.1033 | 0.7167 | Positive category-level context evidence. |
| sheet_metal | primary_category | 0.7130 | context_1.50_topk_max | 0.6574 | -0.0556 | 0.7463 | Limitation/failure-analysis category under the unified pipeline. |
| vial | primary_category | 0.6876 | context_1.50_top1 | 0.6834 | -0.0042 | 0.8732 | Limitation/failure-analysis category under the unified pipeline. |
| walnuts | primary_category | 0.4296 | context_1.50_topk_mean | 0.6430 | 0.2133 | 0.8052 | Positive category-level context evidence. |
| fabric | secondary_boundary_case | 0.4168 | context_1.50_top1 | 0.3980 | -0.0189 | 0.7232 | Secondary boundary case: localization is weak and context crop does not improve. |

## 3. Method Comparison Table

| Scope / Category | Role | Full | Tight top1 | Tight mean | Context top1 | Context mean | Best context Δ | Best VLM | Best VLM Δ | PatchCore |
|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|
| ALL_PRIMARY | primary_aggregate | 0.5201 | 0.5231 | 0.5036 | 0.5773 | 0.6036 | 0.0835 | context_1.50_topk_mean | 0.0835 | 0.8087 |
| fruit_jelly | primary_category | 0.7533 | 0.7367 | 0.6767 | 0.8367 | 0.8567 | 0.1033 | context_1.50_topk_mean | 0.1033 | 0.7167 |
| sheet_metal | primary_category | 0.7130 | 0.4926 | 0.3519 | 0.2444 | 0.5870 | -0.0556 | full_image | 0.0000 | 0.7463 |
| vial | primary_category | 0.6876 | 0.3753 | 0.3973 | 0.6834 | 0.5231 | -0.0042 | full_image | 0.0000 | 0.8732 |
| walnuts | primary_category | 0.4296 | 0.5630 | 0.6067 | 0.5193 | 0.6430 | 0.2133 | context_1.50_topk_mean | 0.2133 | 0.8052 |
| fabric | secondary_category | 0.4168 | 0.4949 | 0.4936 | 0.3980 | 0.3468 | -0.0189 | tight_crop_top1 | 0.0781 | 0.7232 |

## 4. Category Usage Decision

| Category | Detector group | Image AUROC | Pixel AUROC | Pixel F1 | Context Δ | Paper usage | Reason |
|---|---|---:|---:|---:|---:|---|---|
| fruit_jelly | primary | 0.7900 | 0.9476 | 0.4963 | 0.1033 | main_positive_or_supportive | Primary detector quality is acceptable and best context crop improves over full-image prompting. |
| sheet_metal | primary | 0.8315 | 0.8595 | 0.3765 | -0.0556 | main_limitation | Primary detector quality is acceptable, but context crop does not improve under the unified Stage 11 pipeline. |
| vial | primary | 0.7987 | 0.9484 | 0.3366 | -0.0042 | main_limitation | Primary detector quality is acceptable, but context crop does not improve under the unified Stage 11 pipeline. |
| walnuts | primary | 0.7822 | 0.9193 | 0.3918 | 0.2133 | main_positive_or_supportive | Primary detector quality is acceptable and best context crop improves over full-image prompting. |
| fabric | secondary | 0.7582 | 0.7871 | 0.0765 | -0.0189 | secondary_boundary_case | Image-level detector is acceptable but pixel localization is weak; context crop is not positive. |
| can | detector_risk | 0.3901 | 0.7119 | 0.0002 |  | excluded_from_main_vlm_evidence | Detector quality is too weak for fair crop-based VLM reasoning. |
| rice | detector_risk | 0.5630 | 0.7637 | 0.0552 |  | excluded_from_main_vlm_evidence | Detector quality is too weak for fair crop-based VLM reasoning. |
| wallplugs | detector_risk | 0.4626 | 0.8675 | 0.0391 |  | excluded_from_main_vlm_evidence | Detector quality is too weak for fair crop-based VLM reasoning. |

## 5. Final Interpretation

The Stage 11 results support a conditional version of the method claim.

Recommended paper wording:

```text
Localization-guided context-aware crops improve VLM anomaly reasoning on the unified MVTec AD 2 primary-category evaluation, but the benefit is category-dependent and sensitive to detector/candidate quality.
```

Avoid overclaiming:

```text
Context-aware crops consistently improve all industrial anomaly categories.
```

## 6. Main Numeric Takeaway

On ALL_PRIMARY, the reported context-aware method `context_1.50_topk_mean` reaches AUROC `0.6036`, compared with full-image AUROC `0.5201`, giving ΔAUROC `0.0835`.

## 7. Output

- Main table: `results/stage11_mvtecad2_multicategory/stage11_i_paper_ready_main_table.csv`
- Method table: `results/stage11_mvtecad2_multicategory/stage11_i_paper_ready_method_table.csv`
- Category usage table: `results/stage11_mvtecad2_multicategory/stage11_i_category_usage_decision_table.csv`
- Report: `docs/stage11_mvtecad2_multicategory/stage11_i_paper_ready_evidence_report.md`

## 8. Next Step

The next step should be Stage 12: convert these evidence tables into a final method narrative and paper experiment section.