# Stage 16-F Final Claim-Evidence Map

## 1. Purpose

This stage maps every paper-facing claim to concrete experimental evidence and locks the forbidden claims.

No new model is trained and no score is tuned in this stage.

## 2. Final Method Naming

Use this method family name:

```text
Quality-Calibrated QCR
```

Use this longer descriptive phrase when needed:

```text
Quality-Calibrated Localization-Guided VLM Reasoning
```

Use this only as the full variant name:

```text
Quality-Calibrated QCR with adaptive consistency refinement
```

Do not write the method as fixed Q+C QCR-U.

## 3. Claim-Evidence Map

| Claim ID | Category | Paper Claim | Support | Status | Section |
|---|---|---|---|---|---|
| P1 | problem_framing | Industrial anomaly VLM reasoning should be localization-guided rather than full-image only. | moderate | use | Introduction; Method motivation; Experiments |
| P2 | system_level_result | Localization-guided VLM evidence is complementary to detector baselines. | strong_but_protocol_limited | use | Main Results |
| P3 | external_baseline | The proposed localization-guided route is stronger than the fixed WinCLIP protocol used in this study. | moderate | use_with_caution | Baselines |
| P4 | main_method_component | Candidate quality calibration is the main effective method component. | strong_as_core_but_not_universal | use | Method; Ablation |
| P5 | final_method_variant | Adaptive consistency is a conservative refinement, not the main source of improvement. | weak_as_gain_strong_as_safety_caveat | use_with_caution | Ablation; Discussion |
| P6 | diagnostic_component | Fixed Q+C fusion is diagnostic only and should not be the final method. | strong_as_rejection | reject_as_final_method | Ablation; Discussion |
| P7 | upper_bound | Same-set fusion is an upper-bound diagnostic, not a fair main result. | strong_as_protocol_boundary | use_as_diagnostic_only | Main Results; Protocol |
| P8 | boundary_analysis | The method is a reliability-calibrated recognition framework, not a complete anomaly understanding system. | strong_as_boundary_claim | use | Failure Cases; Limitations |
| P9 | segmentation_boundary | Do not claim pixel-level segmentation SOTA. | strong_as_restriction | reject | Limitations |

## 4. Evidence Details

| Claim ID | Evidence Summary | Caveat |
|---|---|---|
| P1 | context-aware VLM AUROC=0.7101187980433263; full-image VLM AUROC=0.6458892382948986; context minus full-image delta=+0.0642. | Do not claim semantic understanding or manufacturing-cause reasoning. |
| P2 | LOCO AUROC=0.8209783368273935; PatchCore AUROC=0.7853284416491964; EfficientAD-30 AUROC=0.7604262679815292; LOCO-PatchCore=+0.0356; LOCO-EfficientAD30=+0.0606. | EfficientAD is fixed-budget; same-set fusion is upper-bound only. |
| P3 | LOCO AUROC=0.8209783368273935; WinCLIP AUROC=0.6137526258826256; delta=+0.2072. | AnomalyCLIP is not yet included; avoid broad CLIP-family claims. |
| P4 | primary QCR quality-minus-naive delta=+0.0096; Per-category mean V4-V3 AUROC delta=+0.0119; wins=13/24.; per-category mean=+0.0119, wins=13/24. | Per-category wins are not universal; use boundary-aware wording. |
| P5 | primary adaptive-minus-quality delta=+0.0004; adaptive-minus-naive delta=+0.0100; Per-category mean V6-V4 AUROC delta=+0.0003; wins=14/24.; per-category mean=+0.0003, wins=14/24. | The gain over quality-only is very small. |
| P6 | primary fixed-minus-quality delta=+0.0043; Per-category mean V5-V4 AUROC delta=+0.0027; positive cases=18/24.; per-category mean=+0.0027, positive cases=18/24. | Do not hide that fixed Q+C can be high in primary protocol; explain robustness tradeoff. |
| P7 | same-set AUROC=0.8452830188679246; LOCO AUROC=0.8209783368273935. | Use LOCO as the fair system-level claim. |
| P8 | quality_helps_anomaly_boost=10; quality_boundary_anomaly_suppression=10; fixed_consistency_boundary_anomaly_suppression=10; quality_helps_normal_suppression=10; quality_boundary_normal_boost=10; fixed_consistency_boundary_normal_boost=10; adaptive_refinement_high_gate=10; detector_vlm_disagreement_boundary=10; The case taxonomy explicitly includes detector-VLM disagreement and candidate-quality boundary cases. | Representative images should be manually inspected before paper figures. |
| P9 | The current method is evaluated and framed primarily for image-level anomaly recognition and candidate reasoning. | Pixel metrics may be reported only as auxiliary detector evidence, not as the main claim. |

## 5. Rejected / Forbidden Claims

| Claim ID | Forbidden Wording | Allowed Replacement |
|---|---|---|
| P6 | Fixed Q+C is the proposed final method. | Fixed consistency can peak in some settings but lacks robustness, so it is not used as the final method. |
| P7 | Same-set fusion is the primary deployment result. | Same-set fusion is reported only as a diagnostic upper bound. |
| P9 | The method achieves pixel-level segmentation SOTA. | Pixel-level/localization signals are used to generate candidate evidence for image-level anomaly recognition. |

## 6. Paper Readiness Status

| Status Group | Claim IDs | Summary |
|---|---|---|
| main_claims_ready | P1;P2;P3;P4;P5;P8 | Industrial anomaly VLM reasoning should be localization-guided rather than full-image only.; Localization-guided VLM evidence is complementary to detector baselines.; The proposed localization-guided route is stronger than the fixed WinCLIP protocol used in this study.; Candidate quality calibration is the main effective method component.; Adaptive consistency is a conservative refinement, not the main source of improvement.; The method is a reliability-calibrated recognition framework, not a complete anomaly understanding system. |
| claims_to_reject_or_downgrade | P6;P7;P9 | Fixed Q+C fusion is diagnostic only and should not be the final method.; Same-set fusion is an upper-bound diagnostic, not a fair main result.; Do not claim pixel-level segmentation SOTA. |
| paper_ready_method_name | P4;P5;P6 | Use Quality-Calibrated QCR as the method family; adaptive consistency is refinement; fixed Q+C is diagnostic only. |
| remaining_experiment_risks | R1;R2;R3 | EfficientAD remains fixed-budget; AnomalyCLIP is absent; representative failure figures still need manual visual inspection. |
| next_actions | N1;N2 | Run defensive EfficientAD-100 fruit_jelly sensitivity later; start paper outline/table-to-text drafting after claim map. |

## 7. Safe Abstract-level Wording

A safe abstract-level claim is:

```text
We propose a quality-calibrated localization-guided VLM reasoning framework for industrial anomaly recognition. The framework converts detector localization evidence into candidate-level visual-language evidence and calibrates crop-level VLM scores using candidate quality. Experiments with strong detector and VLM baselines show that localization-guided VLM evidence complements detector scores, while ablations reveal that candidate quality is the main reliable component and consistency should be used only as a conservative adaptive refinement.
```

## 8. Remaining Risks Before Submission

1. EfficientAD is still fixed-budget. Do not claim full EfficientAD defeat.
2. AnomalyCLIP is not yet included. Avoid broad CLIP-family SOTA claims.
3. Adaptive consistency gain is small. Do not present it as the main contribution.
4. Failure-case examples should be visually inspected before choosing paper figures.
5. The method is image-level anomaly recognition / candidate reasoning, not pixel-level segmentation SOTA.

## 9. Next Step

After this stage, the experimental evidence chain is mostly closed. The next practical step is either:

```text
Stage 17-A: EfficientAD-100 fruit_jelly sensitivity check
```

or:

```text
Paper Stage P1: draft paper outline from claim-evidence map
```

If the goal is submission defense, run EfficientAD-100 fruit_jelly first. If the goal is writing, start the paper outline.

## 10. Outputs

- `results/stage16_qcru_ablation/stage16_f_final_claim_evidence_map.csv`
- `results/stage16_qcru_ablation/stage16_f_paper_claim_status.csv`
- `results/stage16_qcru_ablation/stage16_f_rejected_or_forbidden_claims.csv`
- `docs/stage16_qcru_ablation/stage16_f_final_claim_evidence_map_report.md`
