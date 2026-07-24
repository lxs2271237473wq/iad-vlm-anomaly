# Stage 22-C2b: Actual Selective VLM Runtime

## Scope

- dataset: `VisA`
- detector: `PatchCore`
- reasoning strategy: `inspection_binary`
- evaluation mode: `crop_topk_ensemble`
- GPU: `NVIDIA GeForce RTX 4090`
- CLIP: `ViT-B-32 / openai`
- detector and candidate generation: `excluded`
- model loading: `excluded from mode timing`

## Actual invocation

- full VLM calls: `2162`
- selective VLM calls: `890`
- actual call rate: `0.4117`
- actual calls saved: `0.5883`

## Wall-clock result

- full median time: `1650.937 s`
- selective median time: `738.287 s`
- wall-clock reduction: `0.5528`
- speedup: `2.236x`

## GPU memory

- peak allocated: `605.9 MiB`
- peak reserved: `654.0 MiB`

## Interpretation

The benchmark executes the original Stage 7
CLIP reasoning path. Samples rejected by the
SRB pre-gate are removed before the VLM loop,
so the reported call reduction is an actual
execution-level reduction rather than an
offline estimate.

The benchmark covers only the VLM reasoning
stage after detector inference and candidate
generation.
