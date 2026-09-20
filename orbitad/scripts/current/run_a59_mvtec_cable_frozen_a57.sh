#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/private_data/iad-vlm-anomaly
REPO="$ROOT/ad2_model_zoo/repos/SuperAD"
DATA="$ROOT/datasets/MVTecAD"
OUT="$ROOT/ad2_model_zoo/results/a59_superad_reg4_mvtec_cable_tiled_query_v1"

export DINOV2_LOCAL_REPO="$ROOT/ad2_model_zoo/repos/dinov2"
export DINOV2_LOCAL_WEIGHTS="$ROOT/ad2_model_zoo/pretrained/dinov2_vitl14_reg4_pretrain.pth"
export SUPERAD_SKIP_POST_EVAL=1
export SUPERAD_SAVE_OVERLAYS=0
export SUPERAD_TILED_QUERY=1
export SUPERAD_TILED_RESOLUTION=672
export SUPERAD_TILED_RESIDUAL_WEIGHT=0.10

mkdir -p "$OUT"
cd "$REPO"
/opt/conda/bin/python -u test_public.py \
  --dataset MVTec \
  --preprocess informed \
  --model_name dinov2_vitl14_reg \
  --data_root "$DATA" \
  --results_dir "$OUT/cable" \
  --objects cable \
  --resolution 448 \
  --warmup_iters 1

expected=$(find "$DATA/cable/test" -type f -name '*.png' | wc -l)
produced=$(find "$OUT/cable/anomaly_maps/seed=0/cable/test" -type f -name '*.tiff' | wc -l)
test "$produced" -eq "$expected"

/opt/conda/bin/python - <<PY
import json
from pathlib import Path
out=Path("$OUT")
(out/"A59_INFERENCE_COMPLETE.json").write_text(json.dumps({
  "status":"complete", "dataset":"MVTec AD", "category":"cable",
  "frozen_from":"A57 AD2 Sheet Metal", "global_resolution":448,
  "tiled_resolution":672, "alignment_quantile":0.995, "residual_weight":0.10,
  "preprocess":"official SuperAD informed setting for MVTec cable"
}, indent=2))
print("A59_INFERENCE_COMPLETE")
PY

/opt/conda/bin/python -u "$ROOT/work/stage25/a59_evaluate_mvtec_cable_tiled_query.py"
