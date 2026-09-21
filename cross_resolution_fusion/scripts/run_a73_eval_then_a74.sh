#!/usr/bin/env bash
set -euo pipefail

ROOT="${IAD_REPO_ROOT:-/root/private_data/iad-vlm-anomaly}"
EVAL_MARKER="$ROOT/orbitad/results/a73_mvtec14_unified_eval_v1/A73_UNIFIED_EVALUATION_COMPLETE.json"
EVAL_PATTERN='[a]73_evaluate_mvtec14_unified.py'

cd "$ROOT"
echo "WATCH_A73_EVAL_START $(date -Is)"
while pgrep -f "$EVAL_PATTERN" >/dev/null; do
  sleep 30
done

if [[ ! -f "$EVAL_MARKER" ]]; then
  echo "A73_EVAL_FAILED_OR_INCOMPLETE $(date -Is)"
  exit 1
fi

echo "A73_EVAL_CONFIRMED $(date -Is)"
echo "A74_START $(date -Is)"
bash cross_resolution_fusion/scripts/run_a74_ad2_matched_dual_resolution.sh \
  > orbitad/logs/a74_ad2_matched_dual_resolution.log 2>&1
echo "A74_COMPLETE $(date -Is)"
