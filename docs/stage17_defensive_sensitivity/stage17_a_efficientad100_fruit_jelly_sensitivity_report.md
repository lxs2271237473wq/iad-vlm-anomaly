# Stage 17-A EfficientAD-100 Fruit Jelly Sensitivity

## 1. Purpose

This stage checks whether the Stage 15 EfficientAD-30 fixed-budget baseline severely underestimates EfficientAD.

Only `fruit_jelly` is tested at 100 epochs. This is a defensive sensitivity check, not a new main baseline sweep.

## 2. Configuration

- category: `fruit_jelly`
- max_epochs: `100`
- eval_batch_size: `64`
- num_workers: `16`
- check_val_every_n_epoch: `20`
- precision: `16-mixed`
- model_size: `small`
- train_batch_size: `1`

## 3. 100 Epoch vs 30 Epoch

| Metric | EfficientAD-30 | EfficientAD-100 | Delta 100-30 |
|---|---:|---:|---:|
| image_AUROC | 0.8433 | 0.8267 | -0.0167 |
| image_F1Score | 0.8571 | 0.8438 | -0.0134 |
| pixel_AUROC | 0.7894 | 0.8424 | +0.0531 |
| pixel_F1Score | 0.5395 | 0.5561 | +0.0166 |

## 4. Decision

- decision: `efficientad30_not_severely_underestimating`
- interpretation: EfficientAD-100 does not substantially improve image AUROC over EfficientAD-30 on fruit_jelly.
- next_action: Keep EfficientAD-30 as fixed-budget baseline and cite this sensitivity check defensively.

## 5. Paper Usage

Use this result only as baseline-budget sensitivity evidence.

Do not claim full EfficientAD defeat unless a full-budget multi-category EfficientAD sweep is run.

Safe wording:

```text
We report EfficientAD under a fixed 30-epoch budget and include a 100-epoch fruit_jelly sensitivity check to assess whether the fixed budget severely underestimates EfficientAD.
```

## 6. Outputs

- `results/stage17_defensive_sensitivity/stage17_a_efficientad100_fruit_jelly.csv`
- `results/stage17_defensive_sensitivity/stage17_a_efficientad100_vs_30_delta.csv`
- `docs/stage17_defensive_sensitivity/stage17_a_efficientad100_fruit_jelly_sensitivity_report.md`
