#!/usr/bin/env bash
set -eu
cd /root/private_data/iad-vlm-anomaly
OUT=results/stage24_evidence/20260911_c
mkdir -p "$OUT"
exec 9>"$OUT/queue.lock"
flock -n 9 || exit 73
export HF_HUB_OFFLINE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
printf '%s\n' "$$" > "$OUT/queue.pid"
date -Is > "$OUT/queue.started"
trap 'code=$?; printf "%s\n" "$code" > "$OUT/queue.exit"; date -Is > "$OUT/queue.finished"' EXIT
/opt/conda/bin/python -u experiments/stage24_evidence/scale_probe.py \
 --run results/stage24_evidence/20260910_a --out "$OUT/source_scores" \
 --clip-checkpoint /root/.cache/huggingface/hub/models--timm--vit_base_patch32_clip_224.openai/snapshots/a6f597a30f7b82c51704746581f9a4e41421e878/open_clip_model.safetensors
/opt/conda/bin/python -u experiments/stage24_evidence/evaluate_scale_probe.py \
 --previous results/stage24_evidence/20260911_b --scores "$OUT/source_scores" --out "$OUT/evaluation"
