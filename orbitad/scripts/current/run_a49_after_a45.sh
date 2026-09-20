#!/usr/bin/env bash
set -euo pipefail
ROOT=/root/private_data/iad-vlm-anomaly
while [ ! -f "$ROOT/ad2_model_zoo/results/a45_superad_reg4_public_v1/A45_INFERENCE_COMPLETE.json" ]; do
  sleep 30
done
cd "$ROOT"
/opt/conda/bin/python -u "$ROOT/work/stage25/a47_evaluate_superad_reg4_unified.py"
cd "$ROOT/srb_qcr"
/opt/conda/bin/python -u "$ROOT/work/stage25/a49_export_patchcore_maps.py"
UNIFIED_METHOD=patchcore256 \
UNIFIED_LAYOUT=flat_category_label \
UNIFIED_MAP_ROOT="$ROOT/ad2_model_zoo/results/a49_patchcore256_public_maps_v1" \
UNIFIED_OUT="$ROOT/orbitad/results/a50_patchcore256_unified_eval_v1" \
/opt/conda/bin/python -u "$ROOT/work/stage25/a47_evaluate_superad_reg4_unified.py"
