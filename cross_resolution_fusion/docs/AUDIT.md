# Protocol audit

## A69 reporting correction

The previous audit incorrectly converted an error in the written resolution
and component-source description into a claim that the A69 experiment itself
was invalid. The experiment owner has confirmed that A69 was executed as
intended. The archived A69/A70 outputs are therefore the authoritative primary
AD2 evidence. No A69 metric has been changed.

The statements that A69 was a protocol mismatch, that its score must be
excluded, and that A74 repaired or replaced it are withdrawn. A74 is retained
as a supplementary evaluation under an explicitly uniform global-448/tiled-672
configuration.

## Evidence rules

- A69 normal calibration reads only AD2 `validation/good` images.
- Test labels and masks are used only for metric computation after candidate
  scores are produced.
- A70 reports category-level paired bootstrap confidence intervals and
  win/loss counts for the frozen A69 q99-mean candidate.
- A73 reports all 14 evaluated MVTec AD categories and four methods.
- Claims distinguish AD2 development evidence from external transfer evidence.

## Remaining limitations

A69 is a development-set method search over several fusion formulas. Its
condition-macro AU-PRO should be named explicitly, and the external A73 result
shows that q99 normalization does not dominate raw averaging on every dataset.
The strongest paper claim is therefore an AD2-focused normal-calibrated
cross-resolution fusion result, supported by A70 uncertainty estimates, with
A73/A74 used to delimit its transfer behavior.
