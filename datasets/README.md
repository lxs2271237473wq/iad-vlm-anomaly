# Datasets

This directory stores local datasets only. Real dataset files are not uploaded to GitHub.

Expected local structure:

datasets/
  MVTecAD/
    bottle/
    cable/
    capsule/
    carpet/
    grid/
    hazelnut/
    leather/
    metal_nut/
    pill/
    screw/
    tile/
    toothbrush/
    transistor/
    wood/
    zipper/
  VisA/
  MVTecAD2/
  MPDD/
  BTAD/

The datasets/MVTecAD/ layer must be preserved because more datasets will be added later.
# Stage 24 Experimental Evidence

This directory contains lightweight reproducibility artifacts for Stage 24.

Included:
- manifests
- prediction tables
- evaluation metrics
- source-domain evidence
- transfer results
- ROI diagnostic figures

Excluded:
- PatchCore memory banks
- dense anomaly maps
- model checkpoints
- temporary caches

Heavy artifacts can be regenerated using:
experiments/stage24_evidence/