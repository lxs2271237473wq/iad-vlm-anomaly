# Datasets for First-Stage Industrial Anomaly Detection

## 1. MVTec AD — first required dataset

Purpose: first main benchmark.

Expected folder structure:

```text
./datasets/MVTecAD/
  bottle/
    train/good/
    test/good/
    test/<defect_type>/
    ground_truth/<defect_type>/
  cable/
  capsule/
  ...
```

The official MVTec page requires filling in a download form. After downloading and extracting, put the class folders under `datasets/MVTecAD`.

Recommended first categories:

```text
bottle, cable, capsule, metal_nut, screw, transistor
```

Reason: they cover transparent object, assembly object, small pharmaceutical object, metal part, small structure, and electronic component.

## 2. VisA — second benchmark

Purpose: cross-dataset / VLM few-shot benchmark.

Download with:

```bash
bash scripts/02_download_visa.sh ./datasets
```

## 3. MPDD — manufacturing-knowledge validation dataset

Purpose: metal part defect validation. Use after MVTec AD and VisA are stable.

## 4. Later datasets

- MVTec AD 2: high-difficulty real industrial scenarios.
- MVTec LOCO AD: logical and structural anomalies.
- BTAD: small real industrial dataset.
- KolektorSDD2: industrial surface defects.

Do not start with all datasets. First finish MVTec AD + VisA.
