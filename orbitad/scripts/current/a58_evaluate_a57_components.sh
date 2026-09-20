#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/private_data/iad-vlm-anomaly
SRC="$ROOT/ad2_model_zoo/results/a57_superad_reg4_sheetmetal_tiled_query_v1/sheet_metal/component_maps/seed=0/sheet_metal"
EVAL="$ROOT/work/stage25/a47_evaluate_superad_reg4_unified.py"

for component in global tiled; do
  MAP_ROOT="$ROOT/ad2_model_zoo/results/a58_superad_reg4_sheetmetal_${component}_component_v1"
  rm -rf "$MAP_ROOT"
  for anomaly_type in good bad; do
    dst="$MAP_ROOT/sheet_metal/anomaly_maps/seed=0/sheet_metal/test/$anomaly_type"
    mkdir -p "$dst"
    for src in "$SRC/$anomaly_type"/*_"$component".tiff; do
      name=$(basename "$src" "_${component}.tiff")
      ln "$src" "$dst/$name.tiff"
    done
  done
  printf '{"status":"complete","source":"A57 component maps"}\n' > "$MAP_ROOT/A58_COMPLETE.json"
  UNIFIED_METHOD="superad_reg4_${component}_component" \
  UNIFIED_LAYOUT=superad \
  UNIFIED_CATEGORIES=sheet_metal \
  UNIFIED_COMPLETION_MARKER=A58_COMPLETE.json \
  UNIFIED_MAP_ROOT="$MAP_ROOT" \
  UNIFIED_OUT="$ROOT/orbitad/results/a58_superad_reg4_sheetmetal_${component}_eval_v1" \
  /opt/conda/bin/python -u "$EVAL"
done
