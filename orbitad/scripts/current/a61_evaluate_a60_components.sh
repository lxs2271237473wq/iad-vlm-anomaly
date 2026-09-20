#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/private_data/iad-vlm-anomaly
RUN_ROOT="$ROOT/ad2_model_zoo/results/a60_superad_reg4_sheetmetal_true672_tiled_query_v1"
SRC="$RUN_ROOT/sheet_metal/component_maps/seed=0/sheet_metal"
EVAL="$ROOT/work/stage25/a47_evaluate_superad_reg4_unified.py"

test -f "$RUN_ROOT/A60_INFERENCE_COMPLETE.json"

for component in global tiled; do
  MAP_ROOT="$ROOT/ad2_model_zoo/results/a61_superad_reg4_sheetmetal_true672_${component}_component_v1"
  rm -rf "$MAP_ROOT"
  for anomaly_type in good bad; do
    dst="$MAP_ROOT/sheet_metal/anomaly_maps/seed=0/sheet_metal/test/$anomaly_type"
    mkdir -p "$dst"
    for src in "$SRC/$anomaly_type"/*_"$component".tiff; do
      name=$(basename "$src" "_${component}.tiff")
      ln "$src" "$dst/$name.tiff"
    done
  done
  printf '{"status":"complete","source":"A60 true-672 component maps"}\n' > "$MAP_ROOT/A61_COMPLETE.json"
  UNIFIED_METHOD="superad_reg4_true672_${component}_component" \
  UNIFIED_LAYOUT=superad \
  UNIFIED_CATEGORIES=sheet_metal \
  UNIFIED_COMPLETION_MARKER=A61_COMPLETE.json \
  UNIFIED_MAP_ROOT="$MAP_ROOT" \
  UNIFIED_OUT="$ROOT/orbitad/results/a61_superad_reg4_sheetmetal_true672_${component}_eval_v1" \
  /opt/conda/bin/python -u "$EVAL"
done

/opt/conda/bin/python - <<'PY'
import json
from pathlib import Path

root = Path('/root/private_data/iad-vlm-anomaly/orbitad/results')
paths = {
    'global_448': root / 'a61_superad_reg4_sheetmetal_true672_global_eval_v1/summary.json',
    'tiled_672': root / 'a61_superad_reg4_sheetmetal_true672_tiled_eval_v1/summary.json',
    'fused_010': root / 'a60_superad_reg4_sheetmetal_true672_tiled_query_eval_v1/summary.json',
}
payload = {name: json.loads(path.read_text()) for name, path in paths.items()}
out = root / 'a61_superad_reg4_sheetmetal_true672_component_comparison_v1'
out.mkdir(parents=True, exist_ok=True)
(out / 'summary.json').write_text(json.dumps(payload, indent=2))
(out / 'A61_COMPONENT_AUDIT_COMPLETE.json').write_text(json.dumps({'status': 'complete'}, indent=2))
print('A61_COMPONENT_AUDIT_COMPLETE')
PY
