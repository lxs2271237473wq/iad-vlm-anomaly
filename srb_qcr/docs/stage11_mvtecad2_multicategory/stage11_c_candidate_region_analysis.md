# Stage 11-C MVTec AD 2 Candidate Region Generation

## 1. Purpose

This stage generates PatchCore anomaly-map candidate regions for selected MVTec AD 2 categories.
It prepares tight crops and context-aware crops for later VLM reasoning.

This stage reruns PatchCore fit and predict to access full anomaly maps. It does not run VLM inference and does not modify the original dataset.

## 2. Selected Categories

- sheet_metal
- vial
- fruit_jelly
- walnuts

## 3. Candidate Construction

| Item | Setting |
|---|---|
| Detector | PatchCore |
| Backbone | wide_resnet50_2 |
| Layers | layer2, layer3 |
| Candidate source | anomaly map top-percentile connected components |
| Max candidates per image | 3 |
| Tight crop | connected-component bounding box |
| Context crop | tight box expanded by 1.50 times its width/height on each side |

## 4. Output Files

- Candidate regions: `results/stage11_mvtecad2_multicategory/stage11_c_candidate_regions.csv`
- Summary: `results/stage11_mvtecad2_multicategory/stage11_c_candidate_summary.csv`
- Status: `results/stage11_mvtecad2_multicategory/stage11_c_candidate_status.csv`
- Report: `docs/stage11_mvtecad2_multicategory/stage11_c_candidate_region_analysis.md`
- Local crop root: `results/stage11_mvtecad2_multicategory/stage11_c_candidate_crops`

The crop root is a generated local artifact and should not be committed to GitHub.

## 5. Execution Status

| Category | Success | Images | Candidate rows | Time sec | Error |
|---|---:|---:|---:|---:|---|
| fruit_jelly | True | 40 | 61 | 458.4 | `` |
| sheet_metal | True | 57 | 153 | 328.8 | `` |
| vial | True | 71 | 117 | 538.6 | `` |
| walnuts | True | 75 | 171 | 923.4 | `` |

## 6. Candidate Summary

| Category | Images | Candidate rows | Coverage | Mean cand/img | Anomaly images | Top1 tight GT coverage | Top1 context GT coverage |
|---|---:|---:|---:|---:|---:|---:|---:|
| fruit_jelly | 40 | 61 | 1.0000 | 1.5250 | 30 | 0.4302 | 0.8661 |
| sheet_metal | 57 | 153 | 1.0000 | 2.6842 | 45 | 0.1132 | 0.2509 |
| vial | 71 | 117 | 1.0000 | 1.6479 | 53 | 0.4275 | 0.8016 |
| walnuts | 75 | 171 | 1.0000 | 2.2800 | 45 | 0.3606 | 0.4851 |

## 7. Interpretation

This stage checks whether PatchCore can provide usable visual candidate regions for the VLM branch.
The context crop is especially important because Stage 10-G showed that overly tight anomaly crops can hurt VLM reasoning when object-level context is removed.

## 8. Next Step

Stage 11-D should evaluate full-image versus context-aware crop VLM reasoning on these categories.