# Paper Stage P2: Introduction, Contributions, and Method Overview Draft

## 1. Working Title

```text
Quality-Calibrated Localization-Guided VLM Reasoning for Industrial Anomaly Recognition
```

Short method name:

```text
Quality-Calibrated QCR
```

Full variant name when adaptive consistency is mentioned:

```text
Quality-Calibrated QCR with Adaptive Consistency Refinement
```

---

## 2. Introduction Draft

Industrial anomaly recognition requires identifying subtle, localized deviations from normal product appearance. Recent vision-language models provide broad visual reasoning ability, but applying them directly to full industrial images is often unreliable because defects may occupy only a small region and may not dominate the global image semantics. In our experiments, full-image VLM inference remains weak compared with localization-guided variants, motivating a formulation that first converts anomaly localization evidence into candidate-level visual-language evidence.

Classical anomaly detectors such as PatchCore, FastFlow-style backbones, and EfficientAD provide useful localization or anomaly evidence, but their scores are not designed to express visual-language-level abnormality. Conversely, a VLM can compare localized visual evidence against textual abnormality prompts, but it is easily misled when the candidate crop is poorly localized or visually ambiguous. This creates a practical gap: detector localization and VLM reasoning are complementary, but naive score fusion does not explicitly model whether the crop-level VLM evidence should be trusted.

We address this gap with a quality-calibrated localization-guided VLM reasoning framework. The framework uses detector localization to generate candidate regions, obtains crop-level VLM anomaly evidence, and calibrates that evidence using candidate quality. The resulting method, Quality-Calibrated QCR, treats candidate quality as the main reliability mechanism. We further analyze detector-VLM consistency and find that fixed consistency is not robust enough to serve as the final method; therefore, consistency is retained only as a conservative adaptive refinement.

Empirically, the system-level comparison shows that localization-guided VLM evidence complements detector baselines. The fair LOCO fusion reaches mean image AUROC `0.8210`, compared with PatchCore `0.7853` and EfficientAD-30 fixed-budget `0.7604`. In the QCR primary protocol, Quality-Calibrated QCR improves over naive detector-crop fusion by `+0.0096` AUROC, while adaptive consistency adds only `+0.0004` over the quality-calibrated core. These results support a conservative conclusion: the main effective component is candidate quality calibration, while adaptive consistency is a refinement rather than the main source of improvement.

---

## 3. Contribution Draft

The paper should state the contributions as follows:

1. **Localization-guided VLM anomaly recognition.** We formulate industrial anomaly recognition as a localization-guided visual-language reasoning problem, where detector localization evidence is converted into candidate-level VLM evidence rather than relying on full-image VLM inference.

2. **Quality-calibrated candidate reasoning.** We propose Quality-Calibrated QCR, which calibrates crop-level VLM anomaly evidence using candidate quality derived from anomaly localization. This is the main effective method component and provides the primary gain over naive detector-crop fusion.

3. **Boundary-aware consistency refinement.** We analyze detector-VLM consistency and show that fixed consistency can be unstable. Instead of using fixed Q+C fusion as the final method, we retain consistency only as a reliability-gated adaptive refinement.

4. **Strong baseline and claim-disciplined evaluation.** We compare against full-image VLM, context-aware VLM, WinCLIP, PatchCore, EfficientAD-30 fixed-budget, LOCO fusion, and QCR ablations. We also include boundary analysis and explicitly reject unsupported claims such as pixel-level segmentation SOTA or manufacturing-cause reasoning.

---

## 4. Method Overview Draft

### 4.1 Localization-guided candidate generation

Given an input industrial image, an anomaly detector produces localization evidence and an image-level anomaly score. The localization evidence is used to generate candidate regions that are likely to contain abnormal visual patterns. This candidate-based design reduces the burden on the VLM: instead of interpreting the full image globally, the VLM evaluates localized evidence that is selected by the detector.

### 4.2 Crop-level VLM anomaly evidence

For each candidate crop, the VLM produces an abnormality score based on localized visual-language comparison. The crop-level score is denoted as `M`. This score is useful but not sufficient, because a crop may be poorly localized, too broad, too small, or visually misleading. Therefore, crop-level VLM evidence must be calibrated before it is fused with detector evidence.

### 4.3 Candidate quality calibration

We define candidate quality, denoted as `Q`, to measure whether the selected crop is a reliable carrier of anomaly evidence. The detector image-level anomaly score is denoted as `D`, and the crop-level VLM abnormality score is denoted as `M`. Naive fusion uses:

```text
S_naive = 0.5D + 0.5M
```

Quality-Calibrated QCR instead uses candidate quality to modulate the VLM term:

```text
S_quality = 0.5D + 0.5 * M * (0.5 + 0.5Q)
```

This design reflects the core assumption of the method: crop-level VLM evidence should contribute more when the candidate region is reliable, and less when the candidate region is uncertain.

### 4.4 Adaptive consistency refinement

We also examine detector-VLM consistency. Let `K` denote a high-high consistency signal between detector and VLM evidence. Fixed consistency fusion is diagnostic only because it can produce high scores in some settings but is not robust across protocols. Therefore, the final refinement uses a conservative adaptive gate:

```text
agreement = 1 - |D - M|
mutual_anomaly_evidence = min(D, M)
gate = Q * K * agreement * mutual_anomaly_evidence
S_adaptive = S_quality + 0.05 * gate
```

The adaptive term is intentionally small. It is not presented as the main source of improvement; its role is to preserve the quality-calibrated core while adding consistency only when detector evidence, VLM evidence, candidate quality, and agreement are jointly reliable.

### 4.5 Final scoring interpretation

The final method family is Quality-Calibrated QCR. When the adaptive refinement is included, the full method can be described as Quality-Calibrated QCR with adaptive consistency refinement. However, the main paper claim should remain centered on candidate quality calibration, not on consistency.

---

## 5. Evidence Anchors for Writing

Use these numbers in the paper text:

### System-level evidence

- full-image VLM mean AUROC: `0.6459`
- context-aware VLM mean AUROC: `0.7101`
- WinCLIP fixed protocol mean AUROC: `0.6138`
- EfficientAD-30 fixed-budget mean AUROC: `0.7604`
- PatchCore mean AUROC: `0.7853`
- PatchCore + context VLM, LOCO mean AUROC: `0.8210`
- PatchCore + context VLM, same-set upper bound: `0.8453`
- LOCO minus PatchCore: `+0.0356`
- LOCO minus EfficientAD-30 fixed-budget: `+0.0606`

### QCR evidence

- naive detector-crop fusion mean primary AUROC: `0.9652`
- Quality-Calibrated QCR mean primary AUROC: `0.9750`
- adaptive refinement mean primary AUROC: `0.9752`
- Quality-Calibrated QCR minus naive fusion: `+0.0096`
- adaptive refinement minus quality core: `+0.0004`
- adaptive refinement minus naive fusion: `+0.0100`

---

## 6. Required Claim Boundaries

The following boundaries must be preserved throughout the paper:

- Do not claim fixed Q+C fusion is the final method.
- Do not claim consistency is universally beneficial.
- Do not claim adaptive consistency is the main source of improvement.
- Do not claim pixel-level segmentation SOTA.
- Do not claim manufacturing-cause reasoning.
- Do not claim full EfficientAD defeat; EfficientAD is reported as a 30-epoch fixed-budget baseline.
- Do not merge same-set upper bound with LOCO fair deployment results.

---

## 7. Risk-aware Writing Notes

- `R1`: EfficientAD is still not a full four-category 100-epoch baseline. Mitigation: Stage 17-A fruit_jelly sensitivity shows EfficientAD-100 does not improve image_AUROC over EfficientAD-30 on fruit_jelly. Keep EfficientAD as fixed-budget.
- `R2`: AnomalyCLIP is not included. Mitigation: Avoid broad CLIP-family SOTA claims. Present WinCLIP as the fixed external VLM anomaly baseline used in this study.
- `R3`: Adaptive consistency gain over quality-only is very small. Mitigation: Do not present adaptive consistency as the main innovation. The main method is quality calibration.
- `R4`: Quality calibration is not universally positive per category. Mitigation: Use Stage 16-E boundary analysis. Claim reliability calibration, not universal improvement.
- `R5`: Method may look heuristic because it uses score fusion. Mitigation: Emphasize fixed protocol, ablations, robustness checks, and boundary analysis.
- `R6`: Pixel-level claims are weak. Mitigation: Do not claim segmentation SOTA.

---

## 8. Next Step

Next stage should be:

```text
Paper Stage P3: draft Experiments section and paper-facing result text
```

P3 should turn Stage 16-D and Stage 16-E tables into paper-ready experimental paragraphs.

## 9. Outputs

- `docs/paper_p2/paper_p2_intro_contributions_method_overview.md`
- `results/paper_p2/paper_p2_section_inventory.csv`
- `results/paper_p2/paper_p2_claim_usage_map.csv`
