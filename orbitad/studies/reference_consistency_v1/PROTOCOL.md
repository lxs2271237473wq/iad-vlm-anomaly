# Paired-reference mechanism pilot v1

Status: protocol frozen before pilot feature extraction. No efficacy result yet.

## Scope and stopping
- A74 is the historical baseline; no historical outputs or shared model code will be overwritten.
- CPU reassessment repairs A79/A80/border FPR denominators separately; these are not new method results.
- Pilot categories: can, fabric, sheet_metal. Select 2 normal and 2 abnormal physical public instances per category by deterministic hash, keeping all illumination variants. Label stratification is allowed only for diagnostic sampling, not scoring. Select 6 validation normals independently of GT.
- Prototype extraction initially uses global448 anchors with true square-window672 encoded local features sampled at matching native positions. This is a grid-level feasibility/diagnostic pilot, not full-resolution A74 reproduction. Quantify uncovered GT regions. No-signal on unsampled tiny defects cannot falsify the full hypothesis.
- Smoke uses 2 sorted training references only for correctness/cost, never efficacy. Formal pilot uses original CLS greedy 16-image algorithm with a saved reference list. Historical A74 has no saved reference lists/feature banks, so matching must be checked rather than assumed.

## What is held fixed
Frozen DINOv2-L/14-Reg4, four layers [5,11,17,23], per-layer unit normalization, global448 and actual A74 native short-edge square-window672 views. No training head, photometric augmentation, GT-informed routing or morphology. FP32; SDPA disabled until a separate equivalence check. Record code/config/data selection hashes, exact native coordinates and crop valid support. Preserve native resize/crop geometry instead of stretching discarded pixels; comparisons within pilot use identical geometry.

## Diagnostic scores
- Legacy-style independent NN per layer and view.
- Per-view four-layer joint distance, then independent reference across views.
- Exact common paired-reference joint score (simple concatenation equivalent control).
- Shared reference-image ID with independent patch positions.
- Shuffled paired-reference joint score, fixed seed, preserving marginal bank features.
- Top-k matching support/IDs and distance gaps; different top1 IDs alone are not anomaly evidence.

All joint nearest minima must be exact chunked searches for pilot. Synthetic concatenation-equivalence and independent <= joint checks precede inference. Features are never treated as crop-reencoding equivalents.

## Analysis
GT only attached after feature extraction/scoring. Save center labels, native token footprint anomaly fractions and region coverage. Compare covered defect vs normal regions at matched baseline-score bins, split regular/shift descriptively. Define low baseline-score defect using validation-derived thresholds, not optimized test thresholds. Estimate normal matching ambiguity on held-out validation; do not self-retrieve reference training images. Aggregate/statistical resampling by physical instance with all illumination versions bound; do not interpret token counts as independent samples. With only 2 abnormal instances/category, uncertainty is large: pilot is a stop/go screen, not publication significance.

Stop before expanding if exact-vs-shuffled correspondences provide no useful evidence or normal conflict is as large as defect conflict in covered regions. If positive, predefine and compare a soft ambiguity-tolerant mechanism against concatenation, same-image matching and ordinary kNN/soft projection at identical memory budget; only then evaluate full three-category public maps. No threshold chosen on anomalous GT enters deployable model.

## Resource accounting
A74 TIFF caches exist for all three categories (162/156/114 images), but do not retain reference IDs. A10 banks exist but use another backbone/config and cannot substitute. Run an engineering smoke first and estimate cost from measured extraction/retrieval/I/O; no unmeasured runtime promises. Store probe in separate local/remote reference_consistency_v1 directories, resumable only on manifest identity match.
