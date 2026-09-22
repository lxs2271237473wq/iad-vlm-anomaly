# Audited results

All AD2 values are category-condition macro averages. The localization metric
is AU-PRO integrated over false-positive rates up to 0.05.

## Primary AD2 result and module ablation (A69/A70/A75)

| Method | All | Tiny <=0.1% | Small 0.1–1% | Large >1% |
|---|---:|---:|---:|---:|
| Global | 0.569755 | 0.543316 | 0.767137 | 0.642961 |
| Tiled | 0.579753 | 0.561172 | 0.787014 | 0.653982 |
| Raw mean | 0.601961 | 0.582495 | **0.790940** | **0.661366** |
| q99 mean | **0.602419** | **0.582886** | 0.790099 | 0.659482 |
| q99 geomean | 0.602182 | 0.582543 | 0.789412 | 0.659345 |
| q99 harmonic | 0.601633 | 0.581812 | 0.788758 | 0.659095 |
| Tail mean | 0.594986 | 0.576127 | **0.792480** | 0.658584 |

The frozen q99 mean improves over Global by 0.032664 AU-PRO, with a
category-bootstrap 95% CI of [0.007174, 0.063030] and 6/8 category wins. It
improves over Tiled by 0.022666, with CI [0.002874, 0.042679] and 5/8 wins.
For tiny defects, its improvement over Global is 0.039571, with CI
[0.010247, 0.072058] and 6/8 wins.

A75 completes the previously missing same-input ablation. Without q99
calibration, raw mean already improves over Global by 0.032206 (95% CI
[0.004758, 0.063724], 6/8 wins) and over Tiled by 0.022208 (95% CI
[0.003041, 0.041624], 6/8 wins). Adding q99 calibration to the same two maps
changes the overall score by only +0.000458, with CI [-0.001337, 0.002845]
and 3/8 category wins. Therefore the statistically supported module is the
cross-resolution fusion itself. q99 remains the highest AD2 point estimate,
but its independent contribution is not established and it should be reported
as an optional ablation rather than the source of the fusion gain.

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

## MVTec AD 15-class external evaluation (A76)

MVTec AD has 15 categories and cable is one of them, so cable is evaluated
inside the same protocol instead of being reported as a separate
single-category transfer experiment. A73 was the earlier 14-class run and is
retained as history only.

Scope limit that must be stated with this table: every MVTec AD image is
square, and the tiler uses `tile_size = min(height, width)`, so it emits a
single tile equal to the whole image. The 672 branch is therefore a
higher-resolution whole-image branch rather than a local view, and is labelled
`Res672`. This stage measures a two-resolution ensemble of one view; it does
not test the tiled-view complementarity that A69/A75 measure on AD2.

| Method | AU-PRO@0.05 | Pixel AUROC | Image AUROC |
|---|---:|---:|---:|
| Global 448 | 0.829913 | 0.979800 | 0.992960 |
| Res672 | 0.843436 | 0.975899 | 0.992034 |
| Raw mean | **0.846739** | 0.980806 | 0.993044 |
| q99 mean | 0.845696 | **0.980868** | **0.993070** |

Raw mean improves AU-PRO over Global 448 by 0.016827, with paired category
bootstrap CI [0.011136, 0.024379] and 14/15 wins. Against Res672 the
improvement is 0.003304, CI [-0.000742, 0.007482], which does not exclude zero.
q99 mean is 0.001044 below raw mean, CI [-0.002370, -0.000218]: on this
benchmark normal q99 calibration is significantly worse than plain averaging.
This external result limits the generality claim while preserving the AD2
finding.

## Claim boundary

The supported contribution is an AD2-focused cross-resolution fusion method:
the same-input A75 ablation shows statistically positive complementarity over
both component branches. Normal q99 calibration is an optional variant whose
small AD2 point improvement is not statistically resolved. The cross-dataset
analysis further shows that simple averaging transfers more consistently to
MVTec AD. Claims that q99 calibration is necessary or universally superior
would go beyond the current evidence.
