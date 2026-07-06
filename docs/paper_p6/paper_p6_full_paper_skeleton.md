# Paper Stage P6: First Full Paper Draft Skeleton

## Working Title

```text
Quality-Calibrated Localization-Guided VLM Reasoning for Industrial Anomaly Recognition
```

Short method name:

```text
Quality-Calibrated QCR
```

---

# Abstract

Industrial anomaly recognition with general-purpose vision-language models remains unreliable when images are evaluated globally, because defects are often small, localized, and visually subtle. We propose a quality-calibrated localization-guided VLM reasoning framework that converts detector localization evidence into candidate-level visual-language evidence. The core method, Quality-Calibrated QCR, calibrates crop-level VLM anomaly scores using candidate quality derived from localization evidence. We further study detector-VLM consistency and find that fixed consistency is not robust enough to serve as the final method; therefore, consistency is used only as a conservative adaptive refinement. Experiments with detector, VLM, and external anomaly baselines show that localization-guided VLM evidence complements detector scores, while ablations identify candidate quality calibration as the main effective component. Failure and boundary analyses clarify when quality calibration helps, when it can fail, and why fixed consistency should remain diagnostic.

---

# 1. Introduction

Industrial anomaly recognition requires identifying subtle, localized deviations from normal product appearance. Recent vision-language models provide broad visual reasoning ability, but applying them directly to full industrial images is often unreliable because defects may occupy only a small region and may not dominate the global image semantics. In our experiments, full-image VLM inference remains weak compared with localization-guided variants, motivating a formulation that first converts anomaly localization evidence into candidate-level visual-language evidence.

Classical anomaly detectors such as PatchCore, FastFlow-style backbones, and EfficientAD provide useful localization or anomaly evidence, but their scores are not designed to express visual-language-level abnormality. Conversely, a VLM can compare localized visual evidence against textual abnormality prompts, but it is easily misled when the candidate crop is poorly localized or visually ambiguous. This creates a practical gap: detector localization and VLM reasoning are complementary, but naive score fusion does not explicitly model whether the crop-level VLM evidence should be trusted.

We address this gap with a quality-calibrated localization-guided VLM reasoning framework. The framework uses detector localization to generate candidate regions, obtains crop-level VLM anomaly evidence, and calibrates that evidence using candidate quality. The resulting method, Quality-Calibrated QCR, treats candidate quality as the main reliability mechanism. We further analyze detector-VLM consistency and find that fixed consistency is not robust enough to serve as the final method; therefore, consistency is retained only as a conservative adaptive refinement.

Empirically, the system-level comparison shows that localization-guided VLM evidence complements detector baselines. The fair LOCO fusion reaches mean image AUROC `0.8210`, compared with PatchCore `0.7853` and EfficientAD-30 fixed-budget `0.7604`. In the QCR primary protocol, Quality-Calibrated QCR improves over naive detector-crop fusion by `+0.0096` AUROC, while adaptive consistency adds only `+0.0004` over the quality-calibrated core. These results support a conservative conclusion: the main effective component is candidate quality calibration, while adaptive consistency is a refinement rather than the main source of improvement.

---

## Contributions

The paper should state the contributions as follows:

1. **Localization-guided VLM anomaly recognition.** We formulate industrial anomaly recognition as a localization-guided visual-language reasoning problem, where detector localization evidence is converted into candidate-level VLM evidence rather than relying on full-image VLM inference.

2. **Quality-calibrated candidate reasoning.** We propose Quality-Calibrated QCR, which calibrates crop-level VLM anomaly evidence using candidate quality derived from anomaly localization. This is the main effective method component and provides the primary gain over naive detector-crop fusion.

3. **Boundary-aware consistency refinement.** We analyze detector-VLM consistency and show that fixed consistency can be unstable. Instead of using fixed Q+C fusion as the final method, we retain consistency only as a reliability-gated adaptive refinement.

4. **Strong baseline and claim-disciplined evaluation.** We compare against full-image VLM, context-aware VLM, WinCLIP, PatchCore, EfficientAD-30 fixed-budget, LOCO fusion, and QCR ablations. We also include boundary analysis and explicitly reject unsupported claims such as pixel-level segmentation SOTA or manufacturing-cause reasoning.

---

---

# 2. Related Work

### 2.1 Industrial anomaly detection and localization

Industrial anomaly detection is commonly studied under settings where only normal training examples are available. Benchmarks such as MVTec AD and VisA provide industrial inspection images with image-level and localization-oriented annotations, making them central testbeds for anomaly recognition and localization. Detector-oriented methods typically learn normal appearance and identify deviations at test time.

PatchCore represents a strong patch-feature memory-bank line of work. It stores representative nominal patch features and detects anomalies through deviations from normal patch-level statistics. Flow-based detectors such as FastFlow model feature distributions with normalizing flows and provide anomaly localization evidence. EfficientAD follows a different efficient student-teacher direction and is designed for accurate anomaly detection with low latency.

These detector methods are strong baselines and provide useful localization signals. However, their anomaly scores are not visual-language explanations, and their localization maps do not directly determine whether a candidate crop is reliable VLM evidence. Our work is therefore complementary: we use detector localization to generate candidate regions and then calibrate crop-level VLM evidence using candidate quality.

### 2.2 Vision-language anomaly detection

Vision-language models such as CLIP enable zero-shot image-text matching and have motivated prompt-based anomaly detection. WinCLIP adapts CLIP to anomaly classification and segmentation by aggregating window-level visual features and text prompts. AnomalyCLIP further studies object-agnostic prompt learning for zero-shot anomaly detection.

These methods show that language-supervised representations can support anomaly recognition. However, industrial anomalies are often small, localized, and visually subtle. Global image-text matching can be unreliable when the abnormal evidence occupies only a small region. Window-based or prompt-learning approaches address part of this problem, but they do not directly study how detector localization quality should modulate crop-level VLM evidence.

Our work should therefore not be positioned as a broad replacement for CLIP anomaly methods. Instead, it studies a narrower but practical question: given localization evidence from an anomaly detector, how can crop-level VLM evidence be made more reliable?

### 2.3 Localization-guided VLM evidence

A natural way to improve VLM-based anomaly recognition is to reduce irrelevant context by presenting localized candidate regions to the VLM. This converts detector localization evidence into localized visual-language evidence. However, localization-guided VLM reasoning is not solved by cropping alone. Candidate crops can be too small, too broad, poorly centered, or produced by a misleading detector response.

This motivates the core design of Quality-Calibrated QCR. The method does not treat every crop-level VLM score as equally trustworthy. Instead, it uses candidate quality to calibrate how strongly crop-level VLM evidence should contribute to the final anomaly score. The key contribution is therefore quality-calibrated candidate reasoning rather than simple crop extraction.

### 2.4 Reliability calibration and consistency

Naive detector-VLM fusion assumes that detector scores and VLM scores are directly comparable and equally reliable. Our ablation results show that this assumption is too weak. Candidate quality calibration provides the main improvement over naive fusion by modulating crop-level VLM evidence according to localization-derived reliability.

We also study detector-VLM consistency. Fixed consistency can produce high scores in some primary settings, but robustness analysis shows that it is not stable enough to be the final method. Consequently, the final method treats consistency only as an adaptive, conservative refinement. This distinction is important: the paper should not claim that consistency is universally beneficial or that fixed Q+C fusion is the proposed method.

### 2.5 Positioning summary

The paper is positioned between detector-only industrial anomaly detection and VLM-only anomaly reasoning. Detector-only methods provide strong anomaly localization but do not provide calibrated visual-language evidence. VLM-only methods can use textual abnormality concepts but are unreliable when abnormal regions are small or poorly localized. Quality-Calibrated QCR connects these two directions by converting detector localization into candidate-level VLM evidence and calibrating that evidence using candidate quality.

The safe positioning statement is:

```text
We propose a quality-calibrated localization-guided VLM reasoning framework for industrial anomaly recognition. Unlike detector-only methods, the framework converts localization evidence into visual-language anomaly evidence. Unlike VLM-only anomaly methods, it does not trust crop-level VLM scores blindly; instead, it calibrates them with candidate quality and uses consistency only as a conservative adaptive refinement.
```

---

# 3. Method

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

---

# 4. Experimental Setup

We evaluate industrial anomaly recognition under two complementary experimental views. First, we report a system-level strong baseline comparison over the primary categories, including full-image VLM, context-aware VLM, WinCLIP, PatchCore, EfficientAD-30 fixed-budget, and PatchCore+context fusion. Second, we evaluate the proposed Quality-Calibrated QCR family under the QCR primary protocol to isolate the effect of candidate quality calibration and adaptive consistency refinement.

Image-level AUROC is the primary metric because the paper targets image-level anomaly recognition and candidate-level reasoning. Pixel-level quantities are treated as auxiliary localization evidence and are not used to claim segmentation SOTA. For protocol fairness, LOCO fusion is used as the fair system-level result, while same-set fusion is reported only as an upper-bound diagnostic.

The system-level baselines include three groups. The first group consists of VLM-based baselines: full-image VLM, context-aware VLM, and WinCLIP under the fixed protocol used in this study. The second group consists of detector baselines: PatchCore and EfficientAD-30 fixed-budget. The third group consists of localization-guided fusion variants, including PatchCore+context VLM under LOCO and same-set settings.

EfficientAD is reported as a fixed-budget detector baseline rather than a fully optimized EfficientAD result. This distinction is important: the paper should not claim full EfficientAD defeat. Instead, EfficientAD-30 is used to test whether the proposed localization-guided VLM route remains competitive against a modern non-VLM detector under a controlled fixed budget.

---

# 5. Main Results

Table 1 reports the system-level comparison. Full-image VLM reaches mean image AUROC `0.6459`, while context-aware VLM reaches `0.7101`, giving a localization/context gain of `+0.0642`. The external WinCLIP fixed protocol obtains `0.6138`. Among detector baselines, PatchCore obtains `0.7853` and EfficientAD-30 fixed-budget obtains `0.7604`.

The fair PatchCore+context VLM LOCO fusion reaches `0.8210`, improving over PatchCore by `+0.0356` and over EfficientAD-30 fixed-budget by `+0.0606`. The same-set fusion reaches `0.8453`, but this result is an upper-bound diagnostic and should not be used as the fair deployment claim. These results support the central system-level conclusion: localization-guided VLM evidence is complementary to detector evidence, but fair evaluation must distinguish LOCO from same-set fusion.

Recommended wording:

```text
Under the fair LOCO protocol, localization-guided VLM fusion improves over the detector-only PatchCore baseline and the fixed-budget EfficientAD baseline. The same-set fusion result is reported only as a diagnostic upper bound.
```

---

# 6. Ablation Study

Table 2 reports the QCR primary-protocol ablation. Detector-only scoring obtains mean AUROC `0.9043`, crop VLM only obtains `0.9057`, and naive detector-crop fusion obtains `0.9652`. Quality-Calibrated QCR improves to `0.9750`, corresponding to a mean AUROC gain of `+0.0096` over naive fusion.

The adaptive consistency refinement obtains `0.9752`, improving over naive fusion by `+0.0100` but only improving over the quality-calibrated core by `+0.0004`. Fixed Q+C fusion obtains `0.9791` in the primary protocol and has a primary-protocol delta of `+0.0043` over quality-only, but it is not used as the final method because the robustness analysis showed that fixed consistency is not stable across protocols.

The correct interpretation is therefore not that consistency is the main source of improvement. The main effective component is candidate quality calibration. Adaptive consistency is retained only as a conservative refinement that avoids overcommitting to fixed consistency.

Recommended wording:

```text
Quality calibration provides the main ablation gain over naive detector-crop fusion. Adaptive consistency yields only a small additional refinement and should not be interpreted as the main source of improvement.
```

---

# 7. Failure and Boundary Analysis

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

---

# 8. Baseline-budget Sensitivity

To check whether EfficientAD-30 severely underestimates EfficientAD, we ran a 100-epoch sensitivity check on fruit_jelly. The image-AUROC delta from EfficientAD-30 to EfficientAD-100 is `-0.0167`. The pixel-AUROC delta is `+0.0531`.

This result supports the use of EfficientAD-30 as a fixed-budget image-level baseline in the current paper. However, because the sensitivity check is only on fruit_jelly, the paper should still avoid claiming full EfficientAD defeat. The pixel-AUROC improvement should be mentioned only as auxiliary because the paper does not claim pixel-level segmentation SOTA.

Recommended wording:

```text
We additionally run a 100-epoch EfficientAD sensitivity check on fruit_jelly. The image-level result does not indicate severe underestimation of EfficientAD-30, so we retain EfficientAD-30 as a fixed-budget baseline while avoiding claims of full EfficientAD superiority.
```

---

# 9. Limitations

The method should be interpreted as image-level anomaly recognition and candidate-level VLM evidence calibration, not as pixel-level segmentation SOTA or manufacturing-cause reasoning. EfficientAD is reported as a fixed-budget baseline, with a fruit_jelly 100-epoch sensitivity check used only as defensive evidence. AnomalyCLIP is not included in the current experiments and should be listed as a remaining external VLM anomaly baseline risk. Adaptive consistency provides only a small refinement over the quality-calibrated core, so the main method claim must remain centered on candidate quality calibration.

Risk inventory:

- `R1` **EfficientAD is still not a full four-category 100-epoch baseline.** Mitigation: Stage 17-A fruit_jelly sensitivity shows EfficientAD-100 does not improve image_AUROC over EfficientAD-30 on fruit_jelly. Keep EfficientAD as fixed-budget.
- `R2` **AnomalyCLIP is not included.** Mitigation: Avoid broad CLIP-family SOTA claims. Present WinCLIP as the fixed external VLM anomaly baseline used in this study.
- `R3` **Adaptive consistency gain over quality-only is very small.** Mitigation: Do not present adaptive consistency as the main innovation. The main method is quality calibration.
- `R4` **Quality calibration is not universally positive per category.** Mitigation: Use Stage 16-E boundary analysis. Claim reliability calibration, not universal improvement.
- `R5` **Method may look heuristic because it uses score fusion.** Mitigation: Emphasize fixed protocol, ablations, robustness checks, and boundary analysis.
- `R6` **Pixel-level claims are weak.** Mitigation: Do not claim segmentation SOTA.

---

# 10. Conclusion

We presented Quality-Calibrated QCR, a localization-guided VLM reasoning framework for industrial anomaly recognition. The main finding is that detector localization can provide useful candidate-level evidence for VLM anomaly scoring, but crop-level VLM evidence must be calibrated by candidate quality. The resulting quality-calibrated score improves over naive detector-crop fusion and provides a more reliable method core. Consistency is useful only as a conservative adaptive refinement and should not be treated as a universally beneficial fixed fusion term. The evidence chain supports a conservative but coherent conclusion: quality-calibrated localization-guided VLM reasoning is a practical way to combine industrial anomaly localization and VLM-based anomaly evidence.

---

# Appendix / Paper Assembly Notes

## A. Result-writing Restrictions

The Experiments section must follow these restrictions:

- Do not merge Panel A and Panel B into one global ranking.
- Do not use same-set fusion as the fair system-level claim.
- Do not call EfficientAD-30 a full-budget EfficientAD result.
- Do not claim consistency is universally beneficial.
- Do not describe adaptive consistency as the main performance source.
- Do not claim pixel-level segmentation SOTA.
- Do not claim manufacturing-cause reasoning.

## B. Rejected or Forbidden Claims

- Forbidden: **Fixed Q+C is the proposed final method.**  
  Replacement: Fixed consistency can peak in some settings but lacks robustness, so it is not used as the final method.
- Forbidden: **Same-set fusion is the primary deployment result.**  
  Replacement: Same-set fusion is reported only as a diagnostic upper bound.
- Forbidden: **The method achieves pixel-level segmentation SOTA.**  
  Replacement: Pixel-level/localization signals are used to generate candidate evidence for image-level anomaly recognition.

## C. Tables and Figures Still Needed

- `Table 1` **System-level strong baseline comparison** (required). Source: `results/stage16_qcru_ablation/stage16_d_paper_facing_system_baseline_table.csv`. Notes: Do not merge with QCR ablation table because protocols differ.
- `Table 2` **QCR primary-protocol ablation** (required). Source: `results/stage16_qcru_ablation/stage16_d_paper_facing_qcr_ablation_table.csv`. Notes: Quality-Calibrated QCR is the main effective core; adaptive consistency is refinement.
- `Table 3` **Claim-ready deltas** (required). Source: `results/stage16_qcru_ablation/stage16_d_paper_facing_claim_ready_deltas.csv`. Notes: Use for text evidence, not necessarily as a full paper table.
- `Table 4` **Failure and boundary summary** (required). Source: `results/stage16_qcru_ablation/stage16_e_category_boundary_summary.csv`. Notes: Important because method improvement is reliability-oriented, not huge SOTA margin.
- `Table 5` **EfficientAD-100 fruit_jelly sensitivity** (optional). Source: `results/stage17_defensive_sensitivity/stage17_a_efficientad100_vs_30_delta.csv`. Notes: Use in appendix or footnote; do not claim full EfficientAD defeat.
- `Figure 1` **Framework overview** (required). Source: `to_be_drawn`. Notes: This should be a schematic, not a result table.
- `Figure 2` **Representative boundary cases** (required). Source: `results/stage16_qcru_ablation/stage16_e_failure_boundary_case_inventory.csv`. Notes: Images must be manually inspected before paper use.

## D. Missing Items Checklist

| ID | Item | Priority | Reason | Status |
|---|---|---|---|---|
| M1 | Draw Figure 1 framework overview | high | Method needs a visual pipeline: detector localization -> candidate crop -> VLM evidence -> quality calibration -> adaptive refinement. | missing |
| M2 | Select Figure 2 representative boundary cases | high | Stage 16-E generated case inventory, but image examples must be visually inspected before paper use. | missing |
| M3 | Convert CSV tables into paper-formatted LaTeX tables | high | Paper skeleton references CSV outputs; final paper needs compact formatted tables. | missing |
| M4 | Prepare BibTeX references | high | P4 has reference inventory but not exact BibTeX entries. | missing |
| M5 | Venue template and page budget | medium | Abstract length, table count, and appendix size depend on target venue. | missing |
| M6 | Decide whether to add AnomalyCLIP or explicitly list it as limitation | medium_high | AnomalyCLIP is a likely reviewer question for VLM anomaly work. | open_decision |
| M7 | Polish English academic writing | medium | Current text is claim-safe but still draft-like. | missing |
| M8 | Write exact protocol details | medium_high | LOCO, same-set, QCR primary protocol, and EfficientAD fixed-budget must be unambiguous. | needs_detailing |

## E. Submission Readiness

| Dimension | Status | Evidence | Blocking for First Draft |
|---|---|---|---:|
| experimental_evidence_chain | mostly_closed | Stage 15 strong baselines, Stage 16 method/claim map, Stage 17 EfficientAD sensitivity are complete. | 0 |
| method_name_and_claims | closed | Quality-Calibrated QCR is locked; fixed Q+C rejected as final method. | 0 |
| paper_text_skeleton | created_by_p6 | P6 assembles P2/P3/P4/P5 into one skeleton. | 0 |
| figures | missing | Framework and boundary-case figures are not drawn/selected. | 1 |
| references | incomplete | Reference inventory exists, but BibTeX is not prepared. | 1 |
| external_vlm_baseline_risk | open | AnomalyCLIP is not included; currently handled as limitation. | 0 |
| submission_readiness | draftable_not_submission_ready | Evidence chain is coherent, but figures, BibTeX, table formatting, and final polishing remain. | 0 |

## F. Next Step

Next stage:

```text
Paper Stage P7: convert skeleton into LaTeX-style paper draft and compact tables
```

Before P7, inspect this skeleton once and remove duplicated paragraphs caused by importing earlier stage drafts.

## G. Outputs

- `docs/paper_p6/paper_p6_full_paper_skeleton.md`
- `results/paper_p6/paper_p6_section_assembly_inventory.csv`
- `results/paper_p6/paper_p6_missing_items_checklist.csv`
- `results/paper_p6/paper_p6_submission_readiness.csv`
