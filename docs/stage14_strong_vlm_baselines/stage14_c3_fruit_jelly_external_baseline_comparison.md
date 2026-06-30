# Stage 14-C3 fruit_jelly External Baseline Comparison

## 1. Purpose

This report compares the WinCLIP fruit_jelly pilot with PatchCore, the context-aware VLM branch, and PatchCore+context fusion.

## 2. Comparison Table

| Method | Group | AUROC | AP | Pixel AUROC | Pixel F1 | Interpretation |
|---|---|---:|---:|---:|---:|---|
| WinCLIP zero-shot | external_vlm_baseline | 0.4267 |  | 0.5308 | 0.0063 | External VLM anomaly detection baseline under default zero-shot setting. |
| PatchCore | classical_detector | 0.7167 |  |  |  | Classical detector reference. |
| full-image VLM | vlm_branch | 0.7533 |  |  |  | Full-image VLM baseline. |
| context-aware VLM | vlm_branch | 0.8567 |  |  |  | Our context-aware VLM branch before detector fusion. |
| PatchCore + context VLM, same-set | fusion_same_set | 0.8933 | 0.9711 |  |  | Upper-bound same-set fusion diagnostic. |
| PatchCore + context VLM, leave-one-category-out | fusion_loco | 0.8333 | 0.9476 |  |  | More conservative fusion result using weights selected on other categories. |

## 3. Main Observation

The default zero-shot WinCLIP pilot is weak on AD2 fruit_jelly, with image-level AUROC below 0.5.

This result should not yet be used to claim that our method generally outperforms WinCLIP, because only one category and one zero-shot configuration have been tested.

However, it does show that simply introducing a known VLM anomaly baseline is not automatically stronger on the AD2 setting.

## 4. Next Decision

Before expanding to all primary categories, the next step should test whether WinCLIP few-shot settings or class-name variants improve the fruit_jelly result.

Recommended next step: Stage 14-D should run WinCLIP k-shot/class-name sensitivity on fruit_jelly.

## 5. Output

- Comparison CSV: `results/stage14_strong_vlm_baselines/stage14_c3_fruit_jelly_external_baseline_comparison.csv`
- Report: `docs/stage14_strong_vlm_baselines/stage14_c3_fruit_jelly_external_baseline_comparison.md`