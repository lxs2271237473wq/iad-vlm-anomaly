# Paper Stage P8: Figure Plan

## 1. Outputs

- Figure 1 schematic: `docs/paper_p8/figures/figure1_framework_schematic.svg`
- Figure 1 node inventory: `results/paper_p8/paper_p8_figure1_framework_nodes.csv`
- Figure 2 selected cases: `results/paper_p8/paper_p8_figure2_selected_boundary_cases.csv`
- Figure 2 selection plan: `docs/paper_p8/figure2_boundary_case_selection_plan.md`

## 2. Figure 1 Message

Figure 1 should communicate the full method pipeline:

```text
image -> detector localization -> candidate crops -> crop VLM score -> candidate quality calibration -> image-level anomaly score
```

It must also show:

- candidate quality is the main method core;
- adaptive consistency is a conservative refinement;
- fixed Q+C is diagnostic only.

## 3. Figure 1 Nodes

| Node | Label | Paper Message |
|---|---|---|
| N1 | Input image | The method starts from image-level industrial inspection input. |
| N2 | Detector localization | Detector evidence is used, not replaced. |
| N3 | Candidate crops | Localization evidence is converted into crop-level evidence. |
| N4 | Crop VLM scoring | VLM is applied to localized evidence rather than full image only. |
| N5 | Candidate quality | Quality is the main method core. |
| N6 | Quality-Calibrated QCR | Main final method core. |
| N7 | Adaptive refinement | Small conservative refinement only. |
| N8 | Diagnostic fixed Q+C | Shown as diagnostic branch only. |

## 4. Figure 2 Selected Boundary Cases

| Panel | Case Type | Backbone | Category | Image Key | Purpose |
|---|---|---|---|---|---|
| A | quality_helps_anomaly_boost | PatchCore | chewinggum | datasets/VisA_anomalib_1cls/chewinggum/test/anomaly/chewinggum_test_anomaly_015.JPG | Positive example: quality calibration boosts true anomaly evidence. |
| B | quality_helps_normal_suppression | PatchCore | pcb1 | datasets/VisA_anomalib_1cls/pcb1/test/good/pcb1_test_normal_0490.JPG | Positive example: quality calibration suppresses normal false-positive evidence. |
| C | quality_boundary_anomaly_suppression | PatchCore | pipe_fryum | datasets/VisA_anomalib_1cls/pipe_fryum/test/anomaly/pipe_fryum_test_anomaly_032.JPG | Boundary case: quality calibration can suppress true anomaly evidence. |
| D | quality_boundary_normal_boost | PatchCore | macaroni2 | datasets/VisA_anomalib_1cls/macaroni2/test/good/macaroni2_test_normal_0159.JPG | Boundary case: quality calibration can boost normal evidence. |
| E | fixed_consistency_boundary_normal_boost | PatchCore | pcb1 | datasets/VisA_anomalib_1cls/pcb1/test/good/pcb1_test_normal_0490.JPG | Explains why fixed Q+C can be risky. |
| F | detector_vlm_disagreement_boundary | FastFlow | pipe_fryum | datasets/VisA_anomalib_1cls/pipe_fryum/test/anomaly/pipe_fryum_test_anomaly_096.JPG | Shows detector-VLM conflict as a method boundary. |

## 5. Checklist

| ID | Item | Status | Next Action |
|---|---|---|---|
| P8-C1 | Figure 1 framework SVG generated | done | Review visual layout and convert to PDF/LaTeX figure if needed. |
| P8-C2 | Figure 2 candidate case selection generated | done | Manually inspect original images and crops before using in paper. |
| P8-C3 | Figure 2 actual image montage | not_done | Create after manual inspection confirms selected cases. |
| P8-C4 | Avoid overclaiming in figure captions | required | Caption must say boundary examples, not proof of universal behavior. |
| P8-C5 | Check image paths | required | If image_path is absent, resolve image_key to original dataset file before montage. |

## 6. Caption Drafts

### Figure 1 caption draft

```text
Overview of Quality-Calibrated QCR. Detector localization evidence is converted into candidate crops, which are scored by a VLM to obtain localized anomaly evidence. Candidate quality calibrates the crop-level VLM score and forms the main method core. Adaptive consistency is used only as a conservative refinement, while fixed Q+C fusion is retained as a diagnostic ablation.
```

### Figure 2 caption draft

```text
Representative boundary cases for quality-calibrated localization-guided VLM reasoning. Quality calibration can boost true anomaly evidence and suppress normal false positives, but it can also fail when candidate quality is misleading or detector and VLM evidence disagree. These examples illustrate method boundaries rather than universal behavior.
```

## 7. Next Step

Next stage:

```text
Paper Stage P9: BibTeX/reference preparation and citation placement
```
