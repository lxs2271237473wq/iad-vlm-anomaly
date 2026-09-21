# Protocol audit

## Resolved issue

A69 mixed global test maps produced under a 672 reference configuration with
normal calibration maps produced under the 448/672 A67 configuration. Its
apparent AD2 gain was therefore not admissible evidence. A74 regenerated all
1,084 AD2 public-test images with matched global-448 and tiled-672 queries.
The A69 number must not appear in the main paper table.

## Evidence rules

- Normal calibration reads only AD2 `validation/good` or MVTec AD `train/good`.
- Test labels and masks are used only after all fusion formulas are fixed.
- The external A73 table reports all 14 categories and four methods.
- Claims use category-level paired bootstrap confidence intervals and
  win/loss counts, not only a macro average.
- The q99 fusion hypothesis failed against raw mean on MVTec AD and is kept as
  an ablation/negative result.

## Remaining limitations

The fusion formula has no learned parameters and the strongest external gain
is demonstrated with one detector family and one backbone. A paper should
avoid claiming universal calibration or a general detector-agnostic method
until a second backbone/detector and repeated seeds are evaluated. Image-level
metrics are near saturation on MVTec AD; AU-PRO@0.05 and defect-size analysis
carry more diagnostic value.
