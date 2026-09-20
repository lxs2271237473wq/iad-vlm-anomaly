#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/private_data/iad-vlm-anomaly
REPO="$ROOT/ad2_model_zoo/repos/SuperAD"
DATA="$ROOT/datasets/MVTec_AD_2"
OUT="$ROOT/ad2_model_zoo/results/a67_normal_validation_dual_resolution_v1"
CATEGORIES=(can fabric fruit_jelly rice sheet_metal vial wallplugs walnuts)

export DINOV2_LOCAL_REPO="$ROOT/ad2_model_zoo/repos/dinov2"
export DINOV2_LOCAL_WEIGHTS="$ROOT/ad2_model_zoo/pretrained/dinov2_vitl14_reg4_pretrain.pth"
export SUPERAD_SKIP_POST_EVAL=1
export SUPERAD_SAVE_OVERLAYS=0
export SUPERAD_TEST_SPLIT=validation
export SUPERAD_TILED_QUERY=1
export SUPERAD_TILED_ONLY=0
export SUPERAD_TILED_RESOLUTION=672
export SUPERAD_TILED_RESIDUAL_WEIGHT=0.10

mkdir -p "$OUT"
cd "$REPO"
for category in "${CATEGORIES[@]}"; do
  category_out="$OUT/$category"
  expected=$(find "$DATA/$category/validation/good" -type f | wc -l)
  global_count=$(find "$category_out/component_maps/seed=0/$category/good" -type f -name '*_global.tiff' 2>/dev/null | wc -l || true)
  tiled_count=$(find "$category_out/component_maps/seed=0/$category/good" -type f -name '*_tiled.tiff' 2>/dev/null | wc -l || true)
  if [[ "$global_count" -eq "$expected" && "$tiled_count" -eq "$expected" && "$expected" -gt 0 ]]; then
    echo "SKIP_COMPLETE $category"
    continue
  fi
  rm -rf "$category_out"
  echo "CATEGORY_START $category $(date -Is)"
  /opt/conda/bin/python -u test_public.py \
    --model_name dinov2_vitl14_reg \
    --data_root "$DATA" \
    --results_dir "$category_out" \
    --objects "$category" \
    --resolution 448 \
    --warmup_iters 1
  global_count=$(find "$category_out/component_maps/seed=0/$category/good" -type f -name '*_global.tiff' | wc -l)
  tiled_count=$(find "$category_out/component_maps/seed=0/$category/good" -type f -name '*_tiled.tiff' | wc -l)
  test "$global_count" -eq "$expected"
  test "$tiled_count" -eq "$expected"
  echo "CATEGORY_COMPLETE $category $(date -Is)"
done

/opt/conda/bin/python - <<'PY'
import json
from pathlib import Path
root=Path('/root/private_data/iad-vlm-anomaly/ad2_model_zoo/results/a67_normal_validation_dual_resolution_v1')
(root/'A67_INFERENCE_COMPLETE.json').write_text(json.dumps({
  'status':'complete', 'selection_data':'normal validation images only',
  'reference_resolution':448, 'candidate_query_resolutions':[448,672],
  'purpose':'normal-tail inflation and cross-condition stability routing'
}, indent=2))
print('A67_INFERENCE_COMPLETE')
PY
