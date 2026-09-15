# Stage 8-D Claim-Evidence Mapping

## 1. Purpose

This document maps each paper-level claim to the exact experimental evidence currently available.

The goal is to prevent overclaiming and to make the experimental narrative review-ready.

## 2. Claim-Evidence Table

| ID | Claim | Evidence | Support | Paper Location | Risk if Overstated |
|---|---|---|---|---|---|
| C1 | Localization-guided anomaly crops improve VLM reasoning over full-image prompting. | MVTec AD PatchCore Top-1 improves from 0.2850 to 0.3388; VisA PatchCore AUROC improves from 0.5950 to 0.8844; VisA FastFlow AUROC improves from 0.5950 to 0.9222. | Strong | Main results | Do not claim universal improvement on all datasets/backbones without further evidence. |
| C2 | The proposed localization-to-reasoning pipeline generalizes across datasets. | The effect is observed on MVTec AD defect-type reasoning and VisA binary normal/anomaly reasoning. | Moderate-to-Strong | Dataset generalization | Tasks differ across datasets, so phrase as cross-dataset evidence rather than identical-task proof. |
| C3 | The pipeline is not tied to PatchCore and can generalize to another anomaly backbone. | On VisA, PatchCore crop_topk_ensemble reaches AUROC 0.8844; FastFlow crop_topk_ensemble reaches AUROC 0.9222. Both improve substantially over full-image AUROC 0.5950. | Strong | Backbone generalization | Only PatchCore and FastFlow are tested; do not claim all anomaly detectors. |
| C4 | Candidate localization quality is important for downstream reasoning. | PatchCore candidate coverage on VisA is 1.0000; FastFlow candidate coverage on VisA is 0.9992. Both high-coverage settings produce strong crop reasoning gains. | Moderate | Analysis / limitation | Coverage alone does not prove crop quality; include as supporting rather than sole evidence. |
| C5 | The method should not be presented as a pixel-perfect segmentation method. | PatchCore pixel F1 on VisA is 0.1814; FastFlow pixel F1 on VisA is 0.2573. Reasoning improves despite limited pixel-level F1. | Strong limitation | Limitations | Avoid claiming segmentation quality as the main contribution. |
| C6 | The current experiments validate visual anomaly-region reasoning, not full manufacturing-cause discovery. | Experiments evaluate defect-type or binary normal/anomaly reasoning from visual crops. They do not validate unknown causal mechanism discovery. | Important boundary | Limitations / discussion | Do not claim automatic discovery of all manufacturing causes. |

## 3. Recommended Writing Strategy

The main paper should emphasize C1, C2, and C3 as the core positive claims.

C4 should be discussed as analysis: localization quality and candidate coverage explain why crop reasoning works.

C5 and C6 should be included explicitly as limitations to avoid overstating the contribution.

## 4. Current Safe Main Claim

```text
Classical anomaly localization can serve as an effective bridge between industrial anomaly detectors and visual-language reasoning models.
```

## 5. Claims to Avoid

| Avoided Claim | Reason |
|---|---|
| The method solves pixel-perfect anomaly segmentation. | Pixel F1 remains limited. |
| The method discovers all unknown manufacturing causes. | Current experiments validate visual-region reasoning, not causal discovery. |
| The method works with every anomaly detector. | Only PatchCore and FastFlow are tested. |
| GT crop results are directly comparable to realistic candidate crops. | GT crops are upper-bound diagnostics only. |
