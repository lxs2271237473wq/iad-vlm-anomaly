# Stage 20-C2: AD2-four AnomalyCLIP Metadata

## Data view

- source: `/root/private_data/iad-vlm-anomaly/datasets/MVTec_AD_2_anomalib_all`
- non-destructive view: `/root/private_data/anomalyclip_data/ad2four`
- metadata: `/root/private_data/anomalyclip_data/ad2four/meta.json`
- view method: symbolic links

## Dataset mapping

- dataset name: `ad2four`
- categories: `fruit_jelly`, `sheet_metal`, `vial`, `walnuts`
- dataset.py patch status: `already_patched`
- original backup: `/root/private_data/third_party/AnomalyCLIP/dataset.py.stage20_original_backup`

## Counts

| category | train good | test good | test bad | masks | test total |
|---|---:|---:|---:|---:|---:|
| fruit_jelly | 300 | 20 | 60 | 60 | 80 |
| sheet_metal | 156 | 24 | 90 | 90 | 114 |
| vial | 332 | 35 | 105 | 105 | 140 |
| walnuts | 480 | 60 | 90 | 90 | 150 |

## Validation

- test images: `484`
- normal test images: `139`
- anomalous test images: `345`
- anomaly image-mask matching: one-to-one

## Next step

Load the official Dataset class and run a one-image GPU smoke test.
