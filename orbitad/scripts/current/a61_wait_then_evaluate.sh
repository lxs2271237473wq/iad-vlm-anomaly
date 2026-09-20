#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/private_data/iad-vlm-anomaly
MARKER="$ROOT/ad2_model_zoo/results/a60_superad_reg4_sheetmetal_true672_tiled_query_v1/A60_INFERENCE_COMPLETE.json"

while [[ ! -f "$MARKER" ]]; do
  if ! pgrep -f 'test_public.py.*a60_superad_reg4_sheetmetal_true672_tiled_query_v1' >/dev/null; then
    echo "A60 process ended before its completion marker was created" >&2
    exit 1
  fi
  sleep 30
done

bash "$ROOT/work/stage25/a61_evaluate_a60_components.sh"
