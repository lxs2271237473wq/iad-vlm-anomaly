# Stage 14-A Strong Baseline Landscape

## 1. Purpose

Stage 14 is introduced because comparing only with full-image VLM or only with PatchCore is not sufficient.

The new experimental question is:

```text
Does our context-aware VLM branch remain useful when compared with successful anomaly detection directions, especially WinCLIP-style VLM anomaly detection and modern strong detectors?
```

## 2. Why Stage 14 is necessary

Stage 13-A showed that PatchCore + context-aware VLM fusion improves over PatchCore alone on ALL_PRIMARY.

However, PatchCore is a classic detector and cannot represent the whole current anomaly detection landscape. A stronger paper must compare against:

1. classical strong anomaly detectors;
2. modern efficient anomaly detectors;
3. dedicated vision-language anomaly detection methods;
4. our own PatchCore + context VLM fusion result.

## 3. Baseline Selection Table

| Method | Group | Priority | Role | Next Action |
|---|---|---|---|---|
| PatchCore | Classical anomaly detector | keep | Existing detector reference and localization source | Keep as traditional detector baseline, but not as the only strong baseline. |
| PatchCore + context VLM fusion | Our current fusion method | main_current | Current strongest internal result | Use as current method to compare against WinCLIP and later VLM baselines. |
| WinCLIP | Vision-language anomaly detection | first_external_vlm_baseline | First strong VLM anomaly detection baseline | Check Anomalib WinCLIP API, then run on AD2 primary categories if compatible. |
| AnomalyCLIP | Vision-language anomaly detection | second_external_vlm_baseline | Stronger CLIP-adaptation baseline | Add literature discussion first; reproduce if WinCLIP stage succeeds. |
| EfficientAD | Modern classical anomaly detector | modern_detector_baseline | Modern non-VLM detector baseline | Check Anomalib EfficientAD support after WinCLIP baseline. |
| FastFlow | Flow-based anomaly detector | optional_detector_baseline | Alternative front-end detector | Use after WinCLIP or EfficientAD depending on time. |
| VCP-CLIP / FADE / newer VLM-AD methods | Recent VLM anomaly detection | related_work_or_later_reproduction | Stronger recent related work | Add to related work; reproduce only if time and environment allow. |

## 4. Immediate Decision

The first external VLM anomaly detection baseline should be WinCLIP.

Reasons:

1. It is directly related to CLIP-based anomaly classification and segmentation.
2. It is implemented in Anomalib, so the reproduction cost is lower.
3. It answers the reviewer concern: why compare only against weak full-image CLIP?
4. If our PatchCore + context VLM fusion is competitive against WinCLIP, the paper becomes much stronger.

## 5. Next Step

Stage 14-B should check whether the current environment can import and instantiate WinCLIP from Anomalib.

If WinCLIP is available, Stage 14-C should run a small pilot on one AD2 primary category, preferably fruit_jelly or vial.

If WinCLIP is not available in the installed Anomalib version, Stage 14-B should document the missing dependency/API issue and decide whether to upgrade Anomalib or use an external WinCLIP implementation.