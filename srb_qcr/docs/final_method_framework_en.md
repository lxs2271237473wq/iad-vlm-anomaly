# Final Method Framework: PatchCore-guided Visual Prompt Reasoning with Manufacturing-aware Explanation

## 1. Research Problem

This project studies industrial anomaly understanding beyond conventional anomaly detection.

The goal is not only to detect whether an industrial image is abnormal, but also to provide a structured understanding process:

```text
anomaly localization -> defect type reasoning -> manufacturing-aware explanation
```

The current experiments focus on four weak MVTec AD categories:

```text
grid, screw, leather, wood
```

These categories are selected because PatchCore performs strongly at image-level anomaly detection, while pixel-level localization and fine-grained defect understanding remain challenging.

## 2. Core Motivation

Early experiments showed that simply improving segmentation masks provides limited gain.

The tested branches include:

* PatchCore baseline
* PatchCore failure analysis
* threshold diagnosis
* candidate region extraction
* SAM2 prompt-based refinement
* CLIP semantic reranking
* hand-crafted region scoring
* anomaly map calibration
* trainable anomaly map calibration
* conservative residual calibration
* defect type prompt reasoning
* visual prompt refinement
* manufacturing-aware explanation generation

The main conclusion is:

```text
PatchCore localization is useful, but mask-level refinement alone gives limited improvement.
A more promising direction is to use PatchCore anomaly regions as visual focus for defect type reasoning.
```

## 3. Final Method Pipeline

The final method pipeline is:

```text
Input industrial image
        |
        v
PatchCore anomaly map
        |
        v
Top-k anomaly candidate crops
        |
        v
CLIP-based short visual prompt defect type reasoning
        |
        v
Manufacturing knowledge retrieval
        |
        v
Structured explanation:
defect type -> visual evidence -> possible process -> possible cause
```

## 4. Module Design

### 4.1 PatchCore Anomaly Localization

PatchCore is used as the anomaly localization backbone.

Its role is not replaced. Instead, it provides anomaly maps, candidate regions, candidate crops, and anomaly scores for later semantic reasoning.

The output of this module includes:

```text
anomaly map
candidate boxes
candidate crops
candidate anomaly scores
```

### 4.2 Full-test Candidate Region Generation

Earlier candidate extraction covered only about half of the abnormal test images.

Stage 6.4 fixes this issue by regenerating candidate regions on the full test set.

Final coverage:

```text
327 / 328 abnormal images
coverage ratio = 0.9970
```

This makes crop-based reasoning close to a full-test realistic setting.

### 4.3 Short Visual Prompt Defect Reasoning

The strongest realistic fair setting is:

```text
prompt strategy = generic_label
input mode = crop_topk_ensemble
image count = 328
```

It is compared against:

```text
prompt strategy = generic_label
input mode = full_all
image count = 328
```

Main fair result:

```text
full_all Top-1: 0.2850
crop_topk_ensemble Top-1: 0.3388
Top-1 improvement: +0.0537

full_all Macro-F1: 0.1543
crop_topk_ensemble Macro-F1: 0.2206
Macro-F1 improvement: +0.0663
```

This indicates that PatchCore-guided anomaly crops improve defect type reasoning under the same data and evaluation protocol.

### 4.4 Manufacturing-aware Explanation

Manufacturing knowledge is not used as a long CLIP classification prompt.

Instead, it is used after defect type prediction to generate structured explanations.

Each explanation contains:

* predicted defect type
* top-2 defect candidates
* PatchCore candidate region
* anomaly score
* visual evidence
* defect family
* possible manufacturing process
* possible manufacturing cause
* inspection focus

Important conclusion:

```text
Manufacturing knowledge is useful for explanation, not for direct CLIP classification.
```

## 5. Main Experimental Evidence

### 5.1 Main Fair Result

| Setting                            |   Images |   Top-1 |   Top-2 | Macro-F1 |
| ---------------------------------- | -------: | ------: | ------: | -------: |
| generic_label + full_all           |      328 |  0.2850 |  0.4990 |   0.1543 |
| generic_label + crop_topk_ensemble |      328 |  0.3388 |  0.5072 |   0.2206 |
| Improvement                        | same 328 | +0.0537 | +0.0082 |  +0.0663 |

This is the cleanest main result because it uses:

```text
same dataset
same 328 images
same prompt strategy
same defect label space
same evaluation metrics
```

### 5.2 Prompt and Crop Ablation

The ablation shows:

1. Crop-based reasoning generally improves short-prompt methods.
2. `generic_label + crop_topk_ensemble` gives the best full-test Top-1 and Macro-F1.
3. `visual_ensemble` improves Top-2 but hurts Top-1 and Macro-F1.
4. Longer or more complex prompts are not always better under noisy real anomaly crops.

### 5.3 GT-crop Upper-bound Diagnosis

GT-crop results are useful diagnostic upper bounds, but they are not fair deployable results.

Best GT-crop result:

```text
category_visual + gt_crop
Top-1 = 0.3102
Macro-F1 = 0.2460
```

This shows that accurate region focus helps class-balanced defect reasoning.

However, GT masks are not available in deployment, so these results should be reported separately as upper-bound diagnostics.

### 5.4 Negative and Auxiliary Results

| Branch                            | Conclusion                                                      |
| --------------------------------- | --------------------------------------------------------------- |
| SAM2 prompt refinement            | Generic segmentation does not directly refine defect masks well |
| CLIP semantic candidate reranking | Weak positive, not enough as the main module                    |
| Hand-crafted region scoring       | Weak positive, limited gain                                     |
| Trainable anomaly map calibration | Unstable or too small gain                                      |
| Manufacturing-aware long prompts  | Not suitable for direct CLIP classification                     |

These experiments justify the final design choice.

## 6. Final Contribution Claims

### Contribution 1: PatchCore-guided anomaly crop reasoning

Instead of only improving pixel masks, PatchCore anomaly maps are used to guide vision-language defect type reasoning.

### Contribution 2: Fair full-test evaluation of crop-based defect reasoning

The project explicitly separates:

```text
full-test realistic setting
near-full candidate subset
partial candidate setting
GT-crop upper bound
```

This avoids unfair comparison across different image subsets.

### Contribution 3: Manufacturing-aware explanation layer

The method decouples classification and explanation:

```text
short visual prompts for defect type prediction
manufacturing knowledge for explanation and possible cause reasoning
```

This produces interpretable industrial anomaly understanding outputs.

## 7. Paper Structure Draft

Recommended paper structure:

```text
1. Introduction
2. Related Work
   2.1 Industrial anomaly detection
   2.2 Vision-language models for anomaly understanding
   2.3 Manufacturing knowledge and explainable inspection
3. Method
   3.1 PatchCore anomaly localization
   3.2 Full-test anomaly candidate crop generation
   3.3 Short visual prompt defect type reasoning
   3.4 Manufacturing-aware explanation generation
4. Experiments
   4.1 Dataset and evaluation protocol
   4.2 PatchCore baseline and failure analysis
   4.3 Main fair comparison
   4.4 Prompt and crop ablation
   4.5 GT-crop upper-bound analysis
   4.6 Negative and auxiliary experiments
   4.7 Manufacturing-aware explanation examples
5. Discussion
6. Conclusion
```

## 8. Current Main Result to Report

The main paper result should be:

```text
On the same 328 abnormal MVTec AD weak-category images,
generic_label + crop_topk_ensemble improves defect type reasoning over
generic_label + full_all.

Top-1: 0.2850 -> 0.3388
Macro-F1: 0.1543 -> 0.2206
```

## 9. Current Limitations

1. Defect type recognition accuracy is still modest.
2. CLIP confidence margins are small.
3. Explanations are possible-cause reasoning, not verified causal diagnosis.
4. The manufacturing knowledge base is manually structured and is not actual factory SOP.
5. Current evaluation is on MVTec AD weak categories only.

## 10. Next Recommended Stage

The next stage should be:

```text
Stage 6.9: Final paper outline and experiment narrative
```

or, if more experiments are required:

```text
Stage 7.0: Cross-dataset validation on a more complex industrial anomaly dataset
```

The immediate recommended action is to write the paper narrative first, because the current experimental chain is already coherent.
