# Paper Stage P3: Experiments and Results Draft

## 1. Experimental Setup Draft

We evaluate industrial anomaly recognition under two complementary experimental views. First, we report a system-level strong baseline comparison over the primary categories, including full-image VLM, context-aware VLM, WinCLIP, PatchCore, EfficientAD-30 fixed-budget, and PatchCore+context fusion. Second, we evaluate the proposed Quality-Calibrated QCR family under the QCR primary protocol to isolate the effect of candidate quality calibration and adaptive consistency refinement.

Image-level AUROC is the primary metric because the paper targets image-level anomaly recognition and candidate-level reasoning. Pixel-level quantities are treated as auxiliary localization evidence and are not used to claim segmentation SOTA. For protocol fairness, LOCO fusion is used as the fair system-level result, while same-set fusion is reported only as an upper-bound diagnostic.

## 2. Baselines Draft

The system-level baselines include three groups. The first group consists of VLM-based baselines: full-image VLM, context-aware VLM, and WinCLIP under the fixed protocol used in this study. The second group consists of detector baselines: PatchCore and EfficientAD-30 fixed-budget. The third group consists of localization-guided fusion variants, including PatchCore+context VLM under LOCO and same-set settings.

EfficientAD is reported as a fixed-budget detector baseline rather than a fully optimized EfficientAD result. This distinction is important: the paper should not claim full EfficientAD defeat. Instead, EfficientAD-30 is used to test whether the proposed localization-guided VLM route remains competitive against a modern non-VLM detector under a controlled fixed budget.

## 3. Main System-level Results Draft

Table 1 reports the system-level comparison. Full-image VLM reaches mean image AUROC `0.6459`, while context-aware VLM reaches `0.7101`, giving a localization/context gain of `+0.0642`. The external WinCLIP fixed protocol obtains `0.6138`. Among detector baselines, PatchCore obtains `0.7853` and EfficientAD-30 fixed-budget obtains `0.7604`.

The fair PatchCore+context VLM LOCO fusion reaches `0.8210`, improving over PatchCore by `+0.0356` and over EfficientAD-30 fixed-budget by `+0.0606`. The same-set fusion reaches `0.8453`, but this result is an upper-bound diagnostic and should not be used as the fair deployment claim. These results support the central system-level conclusion: localization-guided VLM evidence is complementary to detector evidence, but fair evaluation must distinguish LOCO from same-set fusion.

Recommended wording:

```text
Under the fair LOCO protocol, localization-guided VLM fusion improves over the detector-only PatchCore baseline and the fixed-budget EfficientAD baseline. The same-set fusion result is reported only as a diagnostic upper bound.
```

## 4. QCR Ablation Results Draft

Table 2 reports the QCR primary-protocol ablation. Detector-only scoring obtains mean AUROC `0.9043`, crop VLM only obtains `0.9057`, and naive detector-crop fusion obtains `0.9652`. Quality-Calibrated QCR improves to `0.9750`, corresponding to a mean AUROC gain of `+0.0096` over naive fusion.

The adaptive consistency refinement obtains `0.9752`, improving over naive fusion by `+0.0100` but only improving over the quality-calibrated core by `+0.0004`. Fixed Q+C fusion obtains `0.9791` in the primary protocol and has a primary-protocol delta of `+0.0043` over quality-only, but it is not used as the final method because the robustness analysis showed that fixed consistency is not stable across protocols.

The correct interpretation is therefore not that consistency is the main source of improvement. The main effective component is candidate quality calibration. Adaptive consistency is retained only as a conservative refinement that avoids overcommitting to fixed consistency.

Recommended wording:

```text
Quality calibration provides the main ablation gain over naive detector-crop fusion. Adaptive consistency yields only a small additional refinement and should not be interpreted as the main source of improvement.
```

## 5. Failure and Boundary Analysis Draft

Table 4 summarizes boundary behavior. Per-category, quality calibration has mean V4-V3 AUROC delta `+0.0119` and wins `13/24` cases. Adaptive consistency has mean V6-V4 delta `+0.0003` and wins `14/24` cases. Fixed Q+C has mean V5-V4 delta `+0.0027` and is positive in `18/24` cases, but it remains diagnostic only.

The case inventory includes the following extracted case types:

- `adaptive_refinement_high_gate`: `10` cases
- `detector_vlm_disagreement_boundary`: `10` cases
- `fixed_consistency_boundary_anomaly_suppression`: `10` cases
- `fixed_consistency_boundary_normal_boost`: `10` cases
- `quality_boundary_anomaly_suppression`: `10` cases
- `quality_boundary_normal_boost`: `10` cases
- `quality_helps_anomaly_boost`: `10` cases
- `quality_helps_normal_suppression`: `10` cases

These cases should be manually inspected before selecting paper figures. The intended qualitative examples are: quality helping anomaly boost, quality suppressing normal false positives, quality suppressing true anomalies as a boundary case, fixed consistency causing risky score changes, and detector-VLM disagreement.

Recommended wording:

```text
Failure analysis shows that quality calibration is useful but not universal. It can fail when candidate quality is misleading or when detector and VLM evidence disagree. This motivates conservative claim boundaries and prevents treating fixed consistency as the final method.
```

## 6. EfficientAD-100 Sensitivity Draft

To check whether EfficientAD-30 severely underestimates EfficientAD, we ran a 100-epoch sensitivity check on fruit_jelly. The image-AUROC delta from EfficientAD-30 to EfficientAD-100 is `-0.0167`. The pixel-AUROC delta is `+0.0531`.

This result supports the use of EfficientAD-30 as a fixed-budget image-level baseline in the current paper. However, because the sensitivity check is only on fruit_jelly, the paper should still avoid claiming full EfficientAD defeat. The pixel-AUROC improvement should be mentioned only as auxiliary because the paper does not claim pixel-level segmentation SOTA.

Recommended wording:

```text
We additionally run a 100-epoch EfficientAD sensitivity check on fruit_jelly. The image-level result does not indicate severe underestimation of EfficientAD-30, so we retain EfficientAD-30 as a fixed-budget baseline while avoiding claims of full EfficientAD superiority.
```

## 7. Result-writing Restrictions

The Experiments section must follow these restrictions:

- Do not merge Panel A and Panel B into one global ranking.
- Do not use same-set fusion as the fair system-level claim.
- Do not call EfficientAD-30 a full-budget EfficientAD result.
- Do not claim consistency is universally beneficial.
- Do not describe adaptive consistency as the main performance source.
- Do not claim pixel-level segmentation SOTA.
- Do not claim manufacturing-cause reasoning.

## 8. Table-to-text Map

| Paper ID | Source | Location | Text Use |
|---|---|---|---|
| Table 1 | `results/stage16_qcru_ablation/stage16_d_paper_facing_system_baseline_table.csv` | Main Results | Main system-level evidence. Use LOCO as fair result; same-set as upper-bound diagnostic only. |
| Table 2 | `results/stage16_qcru_ablation/stage16_d_paper_facing_qcr_ablation_table.csv` | Ablation Study | Show detector-only, crop VLM, naive fusion, quality calibration, fixed Q+C diagnostic, and adaptive refinement. |
| Table 3 | `results/stage16_qcru_ablation/stage16_d_paper_facing_claim_ready_deltas.csv` | Analysis or appendix | Use exact deltas in prose; may not need full table in main paper. |
| Table 4 | `results/stage16_qcru_ablation/stage16_e_category_boundary_summary.csv` | Failure Analysis | Show quality calibration is useful but not universal; fixed consistency remains diagnostic. |
| Appendix Table A1 | `results/stage17_defensive_sensitivity/stage17_a_efficientad100_vs_30_delta.csv` | Appendix or baseline-budget note | Defensive baseline-budget sensitivity. Do not claim full EfficientAD defeat. |
| Figure 1 | `to_be_drawn` | Method | Detector localization -> candidate crop -> VLM evidence -> quality calibration -> adaptive refinement. |
| Figure 2 | `results/stage16_qcru_ablation/stage16_e_failure_boundary_case_inventory.csv` | Failure Analysis | Manual visual inspection required before choosing paper examples. |

## 9. Paragraph Inventory

| Paragraph ID | Section | Purpose | Status |
|---|---|---|---|
| P3-1 | Experimental Setup | Define datasets, categories, metrics, and protocol split. | drafted |
| P3-2 | Baselines | Describe detector, VLM, and external VLM anomaly baselines. | drafted |
| P3-3 | Main Results | Report system-level strong baseline comparison. | drafted |
| P3-4 | QCR Ablation | Report Quality-Calibrated QCR and adaptive refinement ablation. | drafted |
| P3-5 | Failure and Boundary Analysis | Explain quality calibration boundaries and fixed consistency risk. | drafted |
| P3-6 | EfficientAD Sensitivity | Defend EfficientAD-30 fixed-budget with fruit_jelly 100-epoch sensitivity. | drafted |
| P3-7 | Restrictions | State result-writing constraints to prevent overclaiming. | drafted |

## 10. Next Step

Next stage:

```text
Paper Stage P4: Related Work and positioning
```

P4 should position the paper against PatchCore/EfficientAD-style detectors, WinCLIP/CLIP anomaly baselines, and VLM reasoning work, while avoiding broad SOTA claims.

## 11. Outputs

- `docs/paper_p3/paper_p3_experiments_and_results_draft.md`
- `results/paper_p3/paper_p3_result_paragraph_inventory.csv`
- `results/paper_p3/paper_p3_table_to_text_map.csv`
