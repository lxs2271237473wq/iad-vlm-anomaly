# Audited results

All AD2 values are category-condition macro averages. The localization metric
is AU-PRO integrated over false-positive rates up to 0.05.

## Primary AD2 result (A69/A70)

| Method | All | Tiny <=0.1% | Small 0.1–1% | Large >1% |
|---|---:|---:|---:|---:|
| Global | 0.569755 | 0.543316 | 0.767137 | 0.642961 |
| Tiled | 0.579753 | 0.561172 | 0.787014 | 0.653982 |
| q99 mean | **0.602419** | **0.582886** | 0.790099 | 0.659482 |
| q99 geomean | 0.602182 | 0.582543 | 0.789412 | 0.659345 |
| q99 harmonic | 0.601633 | 0.581812 | 0.788758 | 0.659095 |
| Tail mean | 0.594986 | 0.576127 | **0.792480** | 0.658584 |

The frozen q99 mean improves over Global by 0.032664 AU-PRO, with a
category-bootstrap 95% CI of [0.007174, 0.063030] and 6/8 category wins. It
improves over Tiled by 0.022666, with CI [0.002874, 0.042679] and 5/8 wins.
For tiny defects, its improvement over Global is 0.039571, with CI
[0.010247, 0.072058] and 6/8 wins.

These are the primary AD2 results. The earlier report that excluded A69 due to
a protocol mismatch was incorrect; the error was in the report description,
not in the experiment or archived metrics.

## Supplementary uniform 448/672 evaluation (A74)

| Method | All | Tiny <=0.1% | Small 0.1–1% | Large >1% |
|---|---:|---:|---:|---:|
| Global 448 | 0.525133 | 0.490837 | 0.767337 | 0.645751 |
| Tiled 672 | 0.579753 | 0.561172 | 0.787014 | 0.653982 |
| Raw mean | 0.584005 | 0.561071 | **0.801357** | 0.669379 |
| q99 mean | 0.583545 | 0.560418 | 0.800333 | 0.667414 |
| Tail mean | **0.587767** | **0.565541** | 0.800262 | **0.670893** |

A74 is a supplementary protocol variant, not a repair or replacement for A69.
It shows that the best fusion rule depends on how the two branches are
constructed: under this explicit uniform configuration, q99 mean does not add
to raw averaging.

## MVTec AD 14-category external evaluation (A73)

| Method | AU-PRO@0.05 | Pixel AUROC | Image AUROC |
|---|---:|---:|---:|
| Global 448 | 0.834744 | 0.979813 | **0.993930** |
| Tiled 672 | 0.848133 | 0.975655 | 0.992576 |
| Raw mean | **0.851006** | 0.980752 | 0.993524 |
| q99 mean | 0.849918 | **0.980819** | 0.993553 |

Raw mean improves AU-PRO over Global by 0.016262, with paired category
bootstrap CI [0.010388, 0.024285] and 13/14 wins. q99 mean is 0.001088 below
raw mean, with CI [-0.002495, -0.000200]. This external result limits the
generality claim: A69's q99 benefit is established on AD2, while raw
cross-resolution complementarity transfers more consistently to MVTec AD.

## Claim boundary

The supported contribution is an AD2-focused normal-calibrated
cross-resolution fusion method with statistically positive A69/A70 evidence,
plus a cross-dataset analysis showing when calibration transfers and when
simple averaging is sufficient. Claims of universal q99 superiority would go
beyond the current evidence.
