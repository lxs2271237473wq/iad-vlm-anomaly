#!/usr/bin/env bash
set -euo pipefail
ROOT=/root/private_data/iad-vlm-anomaly
while [ ! -f "$ROOT/ad2_model_zoo/results/a49_patchcore256_public_maps_v1/MAPS_COMPLETE.json" ]; do sleep 30; done
UNIFIED_METHOD=patchcore256 \
UNIFIED_LAYOUT=flat_category_label \
UNIFIED_MAP_ROOT="$ROOT/ad2_model_zoo/results/a49_patchcore256_public_maps_v1" \
UNIFIED_OUT="$ROOT/orbitad/results/a50_patchcore256_unified_eval_v1" \
/opt/conda/bin/python -u "$ROOT/work/stage25/a47_evaluate_superad_reg4_unified.py"
