# Cross-resolution anomaly fusion

This directory isolates the reproducible A67–A74 experiment from the larger
`iad-vlm-anomaly` repository. The experiment asks whether the average of a
global 448 query and a tiled 672 query improves anomaly localization without
training a new fusion network.

The repository contains scripts, fixed protocol settings, audit notes, and
lightweight CSV/JSON results. Datasets, pretrained weights, SuperAD/DINOv2
third-party source trees, TIFF score maps, and logs remain server-local and
are deliberately excluded from Git.

## Main result

On the 14-class MVTec AD external validation, raw cross-resolution averaging
raises macro AU-PRO@0.05 from 0.834744 (global 448) to 0.851006. The paired
category bootstrap 95% confidence interval for the improvement is
[0.010388, 0.024285], with 13 wins and 1 loss. Normal-q99 calibration does not
improve over raw averaging on this external set, so it is recorded as a
negative result rather than used as the paper's primary claim.

The corrected AD2 evaluation uses matching 448/672 inference for both normal
calibration and public test images. Raw averaging raises macro AU-PRO@0.05 from
0.525133 (global 448) to 0.584005; its paired category-bootstrap improvement is
0.058873 with 95% CI [0.025116, 0.093168] and 7/8 wins. See `results/` for the
complete category tables and `docs/RESULTS.md` for the compact interpretation.

## Directory layout

- `configs/`: frozen experimental protocol.
- `scripts/`: A67 normal-map generation, A68 calibration, A73 external
  validation, and A74 matched AD2 evaluation.
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
