# Stage 7: Generalization Experiments

## 1. Purpose

Stage 7 evaluates whether the current method generalizes beyond one dataset and one anomaly backbone.

The current main method is:

    anomaly backbone -> anomaly candidate crops -> short visual prompt defect reasoning -> manufacturing-aware explanation

The previous main setting used:

    MVTec AD weak categories + PatchCore

Stage 7 extends this to:

    multiple datasets + multiple anomaly backbones

The goal is to address reviewer concerns for CCF-B-level submission:

1. The method should not depend on a single dataset.
2. The method should not depend on a single anomaly detection backbone.
3. The comparison should use the same data, same evaluation protocol, and same baseline setting.
4. Positive, weak-positive, negative, and upper-bound results should be clearly separated.

## 2. Dataset Generalization

Planned datasets:

| Dataset | Role | Priority | Status |
|---|---|---:|---|
| MVTec AD | Original main dataset | Done | Completed |
| VisA | First cross-dataset validation | 1 | To do |
| MVTec LOCO AD | Structural and logical anomaly validation | 2 | To do |
| Real-IAD subset | Real-world large-scale validation | 3 | To do |

The first target is VisA because it is public, moderately sized, and has image-level and pixel-level annotations.

## 3. Baseline Generalization

Planned anomaly backbones:

| Backbone | Role | Priority |
|---|---|---:|
| PatchCore | Current main backbone | 1 |
| EfficientAD | Fast teacher-student anomaly detection baseline | 2 |
| FastFlow | Flow-based anomaly detection baseline | 3 |
| Reverse Distillation | Distillation/reconstruction anomaly baseline | 4 |
| STFPM | Optional feature matching baseline | Optional |

The initial Stage 7 experiment should first run:

| Dataset | Backbone |
|---|---|
| VisA | PatchCore |
| VisA | EfficientAD |
| VisA | FastFlow |
| VisA | Reverse Distillation |

## 4. Fair-comparison Rules

All comparisons must explicitly report:

- dataset
- category set
- image set
- anomaly backbone
- candidate source
- prompt strategy
- input mode
- number of images used
- coverage ratio
- fallback count
- skipped count
- Top-1 Accuracy
- Top-2 Accuracy
- Macro-F1
- Pixel F1 or IoU when applicable
- whether the setting is realistic, partial subset, or upper-bound

GT-crop results are upper-bound diagnostics only and must not be mixed with deployable full-test results.

## 5. Standard Comparison Groups

For each dataset and backbone, the following settings should be compared:

| Setting | Meaning | Main or Diagnostic |
|---|---|---|
| full image + prompt | VLM directly reasons on full image | baseline |
| anomaly crop + prompt | VLM reasons on top-1 anomaly crop | main comparison |
| top-k anomaly crop ensemble | VLM reasons over multiple anomaly crops | main comparison |
| GT crop | uses ground-truth mask crop | upper-bound only |
| explanation generation | adds manufacturing-aware explanation | explanation module |

The main fair comparison should be:

    same dataset
    same category set
    same image set
    same prompt strategy
    same defect label space
    same evaluation metrics

## 6. First Target: VisA

The first Stage 7 target is VisA.

Initial goals:

1. Download and inspect VisA.
2. Build a VisA manifest.
3. Select categories suitable for defect type reasoning.
4. Run PatchCore on VisA.
5. Generate anomaly candidate regions.
6. Run full image versus anomaly crop prompt reasoning.
7. Build a fair comparison table.
8. Repeat with additional anomaly backbones if the PatchCore pipeline works.

## 7. Expected Paper Value

Stage 7 directly strengthens the paper by adding:

1. Cross-dataset validation.
2. Cross-backbone validation.
3. Stronger fairness in comparison.
4. Better support for the claim that the method is not tied to MVTec AD or PatchCore.
5. A stronger experimental section for CCF-B-level review.

## 8. Immediate Next Steps

The next practical steps are:

1. Check disk space.
2. Download VisA.
3. Extract VisA.
4. Inspect VisA directory structure.
5. Build VisA dataset manifest.
6. Adapt anomaly backbone candidate extraction to VisA.
7. Run VisA + PatchCore first.
