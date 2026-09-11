#!/usr/bin/env bash
set -eu
cd /root/private_data/iad-vlm-anomaly
OUT=results/stage24_evidence/20260911_d
exec 8>"$OUT/postprocess.lock"
flock -n 8 || exit 73
printf '%s\n' "$$" > "$OUT/postprocess.pid"
trap 'code=$?; printf "%s\n" "$code" > "$OUT/postprocess.exit"' EXIT
# Wait for the existing training job, without inspecting partial performance.
exec 9>"$OUT/queue.lock"
flock -x 9
test "$(cat "$OUT/queue.exit")" = 0
test -f "$OUT/fixed_bank/queue_complete.json"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
/opt/conda/bin/python -u experiments/stage24_evidence/crossscale_residual.py \
 --low results/stage24_evidence/20260910_a/baselines --high "$OUT/fixed_bank" --out "$OUT/residual"
/opt/conda/bin/python -u experiments/stage24_evidence/evaluate_highres.py \
 --previous results/stage24_evidence/20260911_b --high "$OUT/fixed_bank" --residual "$OUT/residual" --out "$OUT/evaluation"
