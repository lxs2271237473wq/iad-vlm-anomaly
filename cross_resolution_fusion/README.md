# Cross-resolution anomaly fusion

This directory isolates the reproducible A67–A74 experiment from the larger
`iad-vlm-anomaly` repository. The experiment asks whether the average of a
global and a tiled high-resolution query improves anomaly localization, and
whether normal-only q99 score calibration makes their evidence comparable.

The repository contains scripts, fixed protocol settings, audit notes, and
lightweight CSV/JSON results. Datasets, pretrained weights, SuperAD/DINOv2
third-party source trees, TIFF score maps, and logs remain server-local and
are deliberately excluded from Git.

## Main result

A69 is the primary AD2 result. Normal-q99 fusion reaches 0.602419
category-condition macro AU-PRO@0.05, versus 0.569755 for Global and 0.579753
for Tiled. A70 gives a paired category-bootstrap improvement of 0.032664 over
Global (95% CI [0.007174, 0.063030], 6/8 wins) and 0.022666 over Tiled (95% CI
[0.002874, 0.042679], 5/8 wins). The gain on tiny defects over Global is
0.039571 with 95% CI [0.010247, 0.072058].

The earlier report that treated A69 as invalid was wrong: its written
resolution/source description was incorrect, while the experiment and
archived metrics are valid. A74 remains as a supplementary uniform 448/672
configuration, not as a repair or replacement. On the 15-class MVTec AD
external evaluation (A76, cable included), raw averaging transfers better than
q99 normalization; this limits the generality claim while preserving the A69
AD2 finding. Because every MVTec AD image is square, the tiler emits a single
tile equal to the whole image there, so that branch is labelled `Res672` and
the external stage does not test the tiled-view hypothesis that A69/A75 test on
AD2. See `results/a69_primary/` and `docs/RESULTS.md` for the evidence.

## Directory layout

- `configs/`: frozen experimental protocol.
- `scripts/`: A67 normal-map generation, A68 calibration, A69/A70 primary AD2
  evaluation, A73 external validation, and A74 supplementary evaluation.
- `patches/`: the audited SuperAD changes needed to emit paired component maps.
- `docs/`: method, reproduction, results, and audit records.
- `results/`: lightweight result tables and completion/provenance markers.

## Server prerequisites

The scripts default to `/root/private_data/iad-vlm-anomaly`. Override this
with `IAD_REPO_ROOT` if the repository lives elsewhere. They expect:

- AD2 at `$IAD_REPO_ROOT/datasets/MVTec_AD_2`;
- MVTec AD at `$IAD_REPO_ROOT/datasets/MVTecAD`;
- patched SuperAD at `$IAD_REPO_ROOT/ad2_model_zoo/repos/SuperAD`;
- DINOv2 code and ViT-L/14 register weights under
  `$IAD_REPO_ROOT/ad2_model_zoo/repos/dinov2` and
  `$IAD_REPO_ROOT/ad2_model_zoo/pretrained/`.

Run the commands in `docs/REPRODUCE.md`. The code writes large component maps
to the existing server result area and only copies aggregate evidence into
this directory.
