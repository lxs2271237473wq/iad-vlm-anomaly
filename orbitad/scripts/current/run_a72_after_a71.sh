#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/private_data/iad-vlm-anomaly
MARKER="$ROOT/ad2_model_zoo/results/a71_mvtec_cable_train_normal_dual_resolution_v1/A71_INFERENCE_COMPLETE.json"
LOG="$ROOT/orbitad/logs/a72_frozen_q99_mean_cable.log"

while [[ ! -f "$MARKER" ]]; do
  if ! pgrep -f 'test_public.py.*a71_mvtec_cable_train_normal_dual_resolution_v1' >/dev/null; then
    echo "A71 stopped without completion marker" >&2
    exit 1
  fi
  sleep 30
done

cd "$ROOT"
/opt/conda/bin/python -u work/stage25/a72_evaluate_frozen_q99_mean_cable.py > "$LOG" 2>&1
