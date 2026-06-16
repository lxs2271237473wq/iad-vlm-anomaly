# Baseline Plan

## Stage 0 — environment check

Run `scripts/01_check_env.py` to confirm Python, CUDA, Torch, and Anomalib.

## Stage 1 — dataset check

Run `scripts/03_check_mvtec_structure.py` to confirm MVTec AD is correctly extracted.

## Stage 2 — dataset index

Run `scripts/04_make_dataset_index.py` to record image counts and defect folders.

## Stage 3 — first baselines

Run `scripts/run_mvtec_baselines.py` with `configs/first_baselines.yaml`.

First baselines:

| Baseline | Type | Why it matters |
|---|---|---|
| PatchCore | normal memory bank | Strong traditional IAD baseline |
| PaDiM | feature distribution | Classical statistical feature baseline |
| WinCLIP zero-shot | VLM | Text-semantic baseline without target normal references |
| WinCLIP 4-shot | VLM few-shot | Text + few normal samples baseline |

## Stage 4 — result table

The script saves:

```text
results/baselines_mvtec/baseline_summary.csv
```

This file is the first project master table.
