# Stage 10-B2 MVTec AD 2 Folder Adapter Report

## 1. Purpose

This stage converts the MVTec AD 2 vial manifest into an Anomalib Folder-style dataset.
It does not train PatchCore, run FastFlow, run VLM reasoning, or modify previous experiment results.

## 2. Generated Folder Structure

`datasets/MVTec_AD_2_anomalib/vial_folder`

```text
train/good/
test/good/
test/bad/
ground_truth/bad/
```

## 3. Output Files

- `results/stage10_dataset_expansion/stage10_b2_mvtecad2_folder_mapping.csv`
- `results/stage10_dataset_expansion/stage10_b2_mvtecad2_folder_summary.csv`
- `docs/stage10_dataset_expansion/stage10_b2_mvtecad2_folder_report.md`

## 4. Summary

| Dataset | Category | Subset | Label | is_anomaly | Anomaly type | Images | Masks |
|---|---|---|---|---:|---|---:|---:|
| MVTec AD 2 | vial | test | bad | 1 | bad | 105 | 105 |
| MVTec AD 2 | vial | test | good | 0 | good | 35 | 0 |
| MVTec AD 2 | vial | train | good | 0 | good | 332 | 0 |

## 5. File Validation

- Missing target images: `0`
- Missing target masks: `0`

## 6. Optional Anomalib Folder Validation

- Attempted: `True`
- Success: `True`
- Train batch keys: `<class 'anomalib.data.dataclasses.torch.image.ImageBatch'>`
- Test batch keys: `<class 'anomalib.data.dataclasses.torch.image.ImageBatch'>`

## 7. Next Step

If `Success=True`, Stage 10-C should run the first PatchCore pilot on this Folder dataset.
If Anomalib validation fails, Stage 10-C should first patch the datamodule arguments instead of training.