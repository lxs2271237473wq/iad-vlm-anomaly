# Stage 22-D6c: MVTec AD Actual Selective Runtime

## Protocol

- target: `MVTec AD`
- categories: `15`
- images: `1725`
- CLIP: `ViT-B-32/openai`
- prompt strategy: `inspection_binary`
- crop mode: `crop_topk_ensemble`
- CLIP model load excluded from timing
- text features and candidate boxes cached before timing
- image loading, crop construction, preprocessing, and CLIP image inference included
- paired repeats: `3`
- execution order alternates by repeat to reduce order bias

## Frozen SRB configuration

- `w_max = 0.35`
- `q_quantile = 0.25`
- `tau_delta = 0.75`

## Actual calls

- full VLM calls: `1725`
- selective VLM calls: `1294`
- actual calls saved: `431`
- actual call saving rate: `0.249855`

## Paired runtime results

| Repeat | Full sec | Selective sec | Saved sec | Saving rate | Speedup |
|---:|---:|---:|---:|---:|---:|
| 1 | 1787.330 | 1405.595 | 381.735 | 0.2136 | 1.2716x |
| 2 | 1781.435 | 1403.632 | 377.803 | 0.2121 | 1.2692x |
| 3 | 1750.147 | 1436.916 | 313.230 | 0.1790 | 1.2180x |

## Median runtime

- full median time: `1781.435 s`
- selective median time: `1405.595 s`
- median wall-time saving: `0.212078`
- median speedup: `1.269161x`

## Output consistency

### full
- runtime rows: `1725`
- max raw M difference: `0.0000000000`
- max normalized M difference: `0.0000000000`
- max SRB score difference: `0.0000000000`

### selective
- runtime rows: `1294`
- max raw M difference: `0.0000000000`
- max normalized M difference: `0.0000000000`
- max SRB score difference: `0.0000000000`

The measured speedup is specific to this hardware, software stack, image-crop protocol, and per-image CLIP execution path.