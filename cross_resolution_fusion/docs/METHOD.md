# Method and module responsibilities

## Data flow

1. **A67: matched normal evidence.** SuperAD processes AD2
   `validation/good` images twice in one pass: a 448 global query and a 672
   tiled query. These maps contain no anomalous test labels and define the
   normal score distributions for both resolutions.
2. **A68: normal-only calibration audit.** Per-image q95/q99/q99.9, correlation,
   and positive tiled-score excess are extracted. This stage measures whether
   high resolution amplifies normal texture and supplies q95/q99 constants.
3. **A73: external validation.** The same 448/672 mechanism is evaluated on 14
   MVTec AD categories. Calibration uses each category's `train/good` images;
   test masks are read only for final metrics. Four fixed methods are compared:
   global, tiled, raw mean, and q99-normalized mean.
4. **A74: corrected AD2 evaluation.** All AD2 public-test component maps are
   regenerated with the same 448/672 protocol used by A67. This removes the
   A69 mismatch in which 672 test maps were calibrated by 448 normal maps.
   Global, tiled, raw mean, q99 variants, and normal-tail variants are then
   compared on AU-PRO@0.05, including defect-size bands.

## Core mechanism

The global branch preserves object context, while the tiled branch exposes
small local evidence at higher effective resolution. The fixed raw fusion is

`S_raw = 0.5 * (S_global_448 + S_tiled_672)`.

The audited calibration candidate is

`S_q99 = 0.5 * (S_global_448 / qG + S_tiled_672 / qT)`,

where qG and qT are estimated only from normal images. External validation
shows that this normalization is unnecessary and slightly harmful relative
to raw averaging. The result supports complementary cross-resolution evidence,
but does not support q99 calibration as a claimed contribution.
