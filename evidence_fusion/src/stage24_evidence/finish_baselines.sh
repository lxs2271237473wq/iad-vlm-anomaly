#!/usr/bin/env bash
# Continue the same computation after the GPU queue releases its lock.
set -eu
cd /root/private_data/iad-vlm-anomaly
OUT=results/stage24_evidence/20260910_a
exec 8>"$OUT/postprocess.lock"
flock -n 8 || exit 73
exec 9>"$OUT/queue.lock"
printf '%s\n' "$$" > "$OUT/postprocess.pid"
flock 9
test "$(cat "$OUT/queue.exit")" = 0
test -f "$OUT/baselines/queue_complete.json"
trap 'code=$?; printf "%s\n" "$code" > "$OUT/postprocess.exit"; date -Is > "$OUT/postprocess.finished"' EXIT
/opt/conda/bin/python experiments/stage24_evidence/summarize_baselines.py --out "$OUT/baselines"
/opt/conda/bin/python -u experiments/stage24_evidence/source_fusion.py \
  --base "$OUT/baselines" --manifest "$OUT/manifest" --out "$OUT/simple_fusion"
