# Stage 22-C1: Paired Bootstrap Statistical Analysis

## Protocol

- bootstrap iterations: `5000`
- random seed: `20260722`
- paired resampling: `yes`
- class-stratified resampling: `yes`
- parameter reselection during bootstrap: `none`

## Results

| Protocol | Comparison | Point delta | 95% CI | P(delta>0) | Two-sided p | Interpretation |
|---|---|---:|---:|---:|---:|---|
| visa_patchcore_loco | SRB-QCR vs detector | +0.0058 | [+0.0047, +0.0070] | 1.0000 | 0.0004 | positive CI excludes zero |
| visa_patchcore_loco | Quality QCR vs naive fusion | +0.0102 | [+0.0080, +0.0124] | 1.0000 | 0.0004 | positive CI excludes zero |
| visa_patchcore_loco | Adaptive QCR vs Quality QCR | +0.0003 | [+0.0002, +0.0004] | 1.0000 | 0.0004 | positive CI excludes zero |
| visa_patchcore_loco | SRB-QCR vs Quality QCR | -0.0529 | [-0.0598, -0.0466] | 0.0000 | 0.0004 | negative CI excludes zero |
| visa_fastflow_frozen_transfer | SRB-QCR vs detector | +0.0048 | [+0.0034, +0.0063] | 1.0000 | 0.0004 | positive CI excludes zero |
| visa_fastflow_frozen_transfer | Quality QCR vs naive fusion | +0.0090 | [+0.0070, +0.0112] | 1.0000 | 0.0004 | positive CI excludes zero |
| visa_fastflow_frozen_transfer | Adaptive QCR vs Quality QCR | +0.0005 | [+0.0004, +0.0006] | 1.0000 | 0.0004 | positive CI excludes zero |
| visa_fastflow_frozen_transfer | SRB-QCR vs Quality QCR | -0.0775 | [-0.0865, -0.0683] | 0.0000 | 0.0004 | negative CI excludes zero |
| ad2_frozen_transfer | SRB-QCR vs detector | +0.0086 | [-0.0057, +0.0245] | 0.8756 | 0.2492 | CI includes zero |
| ad2_frozen_transfer | Quality QCR vs naive fusion | -0.0095 | [-0.0311, +0.0113] | 0.1818 | 0.3639 | CI includes zero |
| ad2_frozen_transfer | Adaptive QCR vs Quality QCR | +0.0003 | [-0.0000, +0.0016] | 0.4334 | 1.0000 | CI includes zero |
| ad2_frozen_transfer | SRB-QCR vs Quality QCR | -0.0252 | [-0.0859, +0.0356] | 0.2074 | 0.4151 | CI includes zero |

## Claim rule

- Use `significant improvement` only when the
  complete 95% confidence interval is above zero.
- When the interval includes zero, report the point
  estimate and interval without a significance claim.
- Offline potential call-rate results are not part
  of this statistical test.
