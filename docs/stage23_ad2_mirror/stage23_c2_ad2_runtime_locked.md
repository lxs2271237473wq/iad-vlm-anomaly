# Stage 23-C2: AD2 Actual Selective Runtime

## Locked runtime claim

- Images: `243`
- VLM calls: `243 -> 182`
- Call saving rate: `25.1029%`
- Evaluated crops: `502 -> 397`
- Crop saving rate: `20.9163%`
- Full median time: `287.431 s`
- Selective median time: `240.746 s`
- Median saved time: `54.600 s`
- Median wall-time saving: `18.9470%`
- Median speedup: `1.234x`

## Interpretation boundary

The three paired GPU runs validate the actual selective execution cost under the locked crop and gate decisions.

Exact replay of the historical Stage 11 VLM margins is not claimed. The mismatch is category-dependent and is consistent with an incompletely preserved category-specific text-prompt configuration.

Accuracy claims therefore use the locked Stage 23-B1 prediction cache, whereas runtime claims use the Stage 23-C2 measured executions.