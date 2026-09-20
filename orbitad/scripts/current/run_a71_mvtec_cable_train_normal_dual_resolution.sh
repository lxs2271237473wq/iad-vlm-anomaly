#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/private_data/iad-vlm-anomaly
REPO="$ROOT/ad2_model_zoo/repos/SuperAD"
DATA="$ROOT/datasets/MVTecAD"
OUT="$ROOT/ad2_model_zoo/results/a71_mvtec_cable_train_normal_dual_resolution_v1"

export DINOV2_LOCAL_REPO="$ROOT/ad2_model_zoo/repos/dinov2"
export DINOV2_LOCAL_WEIGHTS="$ROOT/ad2_model_zoo/pretrained/dinov2_vitl14_reg4_pretrain.pth"
export SUPERAD_SKIP_POST_EVAL=1
export SUPERAD_SAVE_OVERLAYS=0
export SUPERAD_TEST_SPLIT=train
export SUPERAD_TILED_QUERY=1
export SUPERAD_TILED_ONLY=0
export SUPERAD_TILED_RESOLUTION=672
export SUPERAD_TILED_RESIDUAL_WEIGHT=0.10

rm -rf "$OUT"
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

expected=$(find "$DATA/cable/train/good" -type f -name '*.png' | wc -l)
global_count=$(find "$OUT/cable/component_maps/seed=0/cable/good" -type f -name '*_global.tiff' | wc -l)
tiled_count=$(find "$OUT/cable/component_maps/seed=0/cable/good" -type f -name '*_tiled.tiff' | wc -l)
test "$global_count" -eq "$expected"
test "$tiled_count" -eq "$expected"

/opt/conda/bin/python - <<PY
import json
from pathlib import Path
out=Path("$OUT")
(out/"A71_INFERENCE_COMPLETE.json").write_text(json.dumps({
  "status":"complete", "dataset":"MVTec AD", "category":"cable",
  "selection_data":"train/good only", "reference_resolution":448,
  "candidate_query_resolutions":[448,672], "normal_images":$expected,
  "purpose":"external normal calibration for A69 frozen q99-mean fusion"
}, indent=2))
print("A71_INFERENCE_COMPLETE")
PY
