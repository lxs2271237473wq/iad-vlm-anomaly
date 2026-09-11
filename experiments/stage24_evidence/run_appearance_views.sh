#!/usr/bin/env bash
set -eu
cd /root/private_data/iad-vlm-anomaly
OUT=results/stage24_evidence/20260911_g
mkdir -p "$OUT"
exec 9>"$OUT/queue.lock"
flock -n 9 || exit 73
export HF_HUB_OFFLINE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
printf '%s\n' "$$" > "$OUT/queue.pid"
trap 'code=$?; printf "%s\n" "$code" > "$OUT/queue.exit"' EXIT
/opt/conda/bin/python -u experiments/stage24_evidence/evaluate_appearance_views.py
