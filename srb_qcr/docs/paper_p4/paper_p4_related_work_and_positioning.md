# Paper Stage P4: Related Work and Positioning

## 1. Purpose

This stage drafts the Related Work section and locks the paper's positioning against industrial anomaly detectors, CLIP/VLM anomaly methods, localization-guided reasoning, and reliability calibration.

The goal is not to claim broad SOTA. The goal is to state precisely where Quality-Calibrated QCR fits.

## 2. Related Work Draft

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

## 3. Reference Inventory

| Cite Key | Work | Year | Category | Positioning Use |
|---|---|---:|---|---|
| Bergmann2019MVTecAD | MVTec AD | 2019 | industrial_anomaly_dataset | Use as background benchmark context, especially for industrial inspection framing. |
| Zou2022VisA | VisA / SPot-the-Difference | 2022 | industrial_anomaly_dataset | Use as dataset context for VisA-based experiments and candidate reasoning. |
| Roth2022PatchCore | PatchCore | 2022 | industrial_anomaly_detector | Our method uses detector localization evidence and shows VLM evidence can complement detector scores. |
| Yu2021FastFlow | FastFlow | 2021 | industrial_anomaly_detector | Use as detector/localization backbone context, not as the main novelty target. |
| Batzner2024EfficientAD | EfficientAD | 2024 | industrial_anomaly_detector | Report as EfficientAD-30 fixed-budget baseline and add fruit_jelly 100-epoch sensitivity. |
| Radford2021CLIP | CLIP | 2021 | vision_language_model | Use as background for VLM anomaly reasoning and prompt-based visual evidence. |
| Jeong2023WinCLIP | WinCLIP | 2023 | clip_anomaly_detection | Use as external CLIP/VLM anomaly baseline under our fixed protocol. |
| Zhou2024AnomalyCLIP | AnomalyCLIP | 2024 | clip_anomaly_detection | Mention as important related CLIP anomaly work and as a limitation if not included experimentally. |

## 4. Positioning Map

| ID | Related Area | Prior Work Does | Our Gap | Our Position |
|---|---|---|---|---|
| RW-P1 | Industrial anomaly detection and localization | Detects and localizes anomalies using normal-only training, patch features, flows, or student-teacher signals. | Detector scores and localization maps do not directly provide reliable visual-language evidence. | We use detector localization as candidate evidence and calibrate crop-level VLM scores with candidate quality. |
| RW-P2 | CLIP / VLM anomaly detection | Uses image-text alignment, window features, prompts, or object-agnostic prompt learning for anomaly scoring. | Global or window-level VLM scores may still be unreliable for tiny or poorly localized industrial defects. | We study localization-guided VLM evidence and reliability calibration of candidate crops. |
| RW-P3 | Localization-guided reasoning | Uses localized regions to reduce irrelevant visual context. | Candidate crops are not equally reliable; naive fusion ignores whether crop evidence should be trusted. | Candidate quality explicitly modulates crop-level VLM evidence. |
| RW-P4 | Reliability calibration and consistency | Combines model scores or agreement signals. | Fixed consistency can be unstable and can hurt when detector/VLM evidence is unreliable. | Quality calibration is the stable core; adaptive consistency is a conservative refinement only. |

## 5. Forbidden Related-work Positioning

- Do not claim to be the first VLM anomaly detection method.
- Do not claim broad CLIP-family SOTA.
- Do not claim superiority over AnomalyCLIP unless AnomalyCLIP is run under a matched protocol.
- Do not claim to replace PatchCore, FastFlow, or EfficientAD.
- Do not claim pixel-level segmentation SOTA.
- Do not claim manufacturing-cause reasoning.

## 6. Related Work Section Inventory

| Section ID | Title | Main Refs | Purpose | Status |
|---|---|---|---|---|
| RW-1 | Industrial anomaly detection and localization | Bergmann2019MVTecAD; Zou2022VisA; Roth2022PatchCore; Yu2021FastFlow; Batzner2024EfficientAD | Establish detector/localization context and explain why detector evidence is useful but not sufficient for VLM reasoning. | drafted |
| RW-2 | Vision-language anomaly detection | Radford2021CLIP; Jeong2023WinCLIP; Zhou2024AnomalyCLIP | Position VLM/CLIP anomaly methods and avoid broad CLIP-family superiority claims. | drafted |
| RW-3 | Localization-guided VLM evidence | PatchCore; FastFlow; VLM crop reasoning | Explain the gap between detector localization and trustworthy VLM evidence. | drafted |
| RW-4 | Reliability calibration of candidate evidence | Quality-Calibrated QCR evidence from our ablations | Position the main method as reliability calibration, not raw score fusion. | drafted |
| RW-5 | Positioning summary | All above | State exactly how this paper differs from detector-only and CLIP-only anomaly methods. | drafted |

## 7. Next Step

Next stage:

```text
Paper Stage P5: Method section full draft
```

P5 should turn the P2 method overview into a full method section with notation, algorithm steps, and scoring formulas.

## 8. Outputs

- `docs/paper_p4/paper_p4_related_work_and_positioning.md`
- `results/paper_p4/paper_p4_reference_inventory.csv`
- `results/paper_p4/paper_p4_positioning_map.csv`
- `results/paper_p4/paper_p4_related_work_section_inventory.csv`
