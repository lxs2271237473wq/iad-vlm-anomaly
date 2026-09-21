# Audited results

All values below are macro averages across categories. The localization metric
is AU-PRO integrated over false-positive rates up to 0.05.

## AD2 matched-protocol evaluation (A74)

| Method | All | Tiny <=0.1% | Small 0.1–1% | Large >1% |
|---|---:|---:|---:|---:|
| Global 448 | 0.525133 | 0.490837 | 0.767337 | 0.645751 |
| Tiled 672 | 0.579753 | 0.561172 | 0.787014 | 0.653982 |
| Raw mean | **0.584005** | 0.561071 | **0.801357** | 0.669379 |
| q99 mean | 0.583545 | 0.560418 | 0.800333 | 0.667414 |
| Tail mean | 0.587767 | **0.565541** | 0.800262 | **0.670893** |

Raw mean improves over global 448 by 0.058873 AU-PRO, with a paired
category-bootstrap 95% CI of [0.025116, 0.093168] and 7/8 category wins. It is
only 0.004253 above tiled 672; the CI [-0.031784, 0.035520] crosses zero and the
win/loss count is 4/4. Tail mean ranks first numerically, but its +0.003762 over
raw mean is also inconclusive: CI [-0.006076, 0.018028], 3 wins and 5 losses.

These results establish that higher-resolution tiled evidence and simple
cross-resolution averaging substantially repair the weak global baseline.
They do not establish that the calibrated variants reliably beat tiled-only
inference on AD2.

## MVTec AD 14-category external evaluation (A73)

| Method | AU-PRO@0.05 | Pixel AUROC | Image AUROC |
|---|---:|---:|---:|
| Global 448 | 0.834744 | 0.979813 | **0.993930** |
| Tiled 672 | 0.848133 | 0.975655 | 0.992576 |
| Raw mean | **0.851006** | 0.980752 | 0.993524 |
| q99 mean | 0.849918 | **0.980819** | 0.993553 |

Raw mean improves AU-PRO over global 448 by 0.016262, with paired
category-bootstrap CI [0.010388, 0.024285] and 13/14 wins. Its +0.002873 over
tiled 672 is not significant: CI [-0.001356, 0.007307], 9 wins and 5 losses.
The q99 mean is 0.001088 below raw mean with CI
[-0.002495, -0.000200], losing on 12/14 categories.

## Claim boundary

The defensible finding is that global and tiled score maps are complementary,
and fixed raw averaging gives a statistically supported improvement over the
global branch on both datasets. The present evidence does not support claiming
normal-q99 calibration, tail calibration, or fusion-over-tiled as a consistent
improvement. Those variants belong in the ablation table and motivate a future
reliability-aware fusion mechanism.
