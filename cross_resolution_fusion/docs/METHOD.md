# Method and module responsibilities

## Data flow

1. **A67: normal evidence.** SuperAD processes AD2 `validation/good` images
   with global and tiled queries. These maps contain no anomalous test labels
   and provide branch-specific normal score statistics.
2. **A68: normal-only calibration.** Per-image q95/q99/q99.9, correlation and
   positive tiled-score excess are extracted. Category medians supply the
   normal calibration constants used by A69.
3. **A69: AD2 development search.** The archived Global and Tiled component
   maps are combined using fixed raw, q99-normalized and normal-tail formulas.
   Evaluation covers eight AD2 categories and reports all/tiny/small/large
   condition-macro AU-PRO@0.05. q99 mean is the selected candidate.
4. **A70: paired robustness audit.** The frozen A69 q99-mean candidate is
   compared with both branches using category-level bootstrap confidence
   intervals, category wins and condition wins.
5. **A76: external evaluation on all 15 MVTec AD categories.** The frozen
   fusion is evaluated on the complete MVTec AD benchmark, cable included,
   using one evaluator for all 15 classes. This stage measures transfer rather
   than reselecting the AD2 candidate.
   Scope limit that must be stated: every MVTec AD image is square and the
   tiler uses `tile_size = min(height, width)`, so it emits a single tile equal
   to the whole image. On MVTec the 672 branch is therefore a higher-resolution
   whole-image branch, not a local view, and this stage tests a two-resolution
   ensemble of one view rather than the tiled-view hypothesis that A69/A75 test
   on AD2. Tables label this branch `Res672` for that reason. A73 is the
   earlier 14-class run and is retained as history only.
6. **A74: supplementary uniform protocol.** A separate, explicitly uniform
   global-448/tiled-672 construction is evaluated on AD2. It is an ablation of
   branch construction and is not a repair or replacement of A69.

## Core mechanism

Let `G` and `T` be the global and tiled anomaly maps used by A69, and let
`qG` and `qT` be category-level medians of per-normal-image q99 scores:

`S_q99 = 0.5 * (G / qG + T / qT)`.

Normal calibration compensates for branch-specific score scale before
combining contextual and local evidence. A69/A70 show that this improves AD2
AU-PRO over both component branches. A76 (15 MVTec AD classes) shows that the
same normalization is not universally better than raw averaging, so the paper
should present q99 calibration as an AD2-effective mechanism with measured
transfer limits.
