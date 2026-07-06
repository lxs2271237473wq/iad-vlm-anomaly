# Table 2. QCR primary-protocol ablation

| ID | Method | Role | FastFlow | PatchCore | Mean AUROC |
| --- | --- | --- | --- | --- | --- |
| V0 | Detector only | anchor_baseline | 0.8955 | 0.9131 | 0.9043 |
| V2 | Crop VLM only | vlm_crop_baseline | 0.9269 | 0.8846 | 0.9057 |
| V3 | Naive detector-crop fusion | naive_fusion_baseline | 0.9688 | 0.9616 | 0.9652 |
| V4 | Quality-Calibrated QCR | main_effective_method_core | 0.9778 | 0.9718 | 0.9748 |
| V5 | Fixed Q+C fusion | diagnostic_not_final | 0.9842 | 0.9740 | 0.9791 |
| V6 | Quality-Calibrated QCR + adaptive consistency refinement | final_refinement_variant | 0.9783 | 0.9722 | 0.9752 |

**Note.** Quality-Calibrated QCR is the main method core. Fixed Q+C is diagnostic only.
