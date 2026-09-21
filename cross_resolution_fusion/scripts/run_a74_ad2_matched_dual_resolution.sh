#!/usr/bin/env bash
set -euo pipefail

ROOT="${IAD_REPO_ROOT:-/root/private_data/iad-vlm-anomaly}"
REPO="$ROOT/ad2_model_zoo/repos/SuperAD"
DATA="$ROOT/datasets/MVTec_AD_2"
OUT="$ROOT/ad2_model_zoo/results/a74_ad2_matched_dual_resolution_v1"
CATEGORIES=(can fabric fruit_jelly rice sheet_metal vial wallplugs walnuts)

export DINOV2_LOCAL_REPO="$ROOT/ad2_model_zoo/repos/dinov2"
export DINOV2_LOCAL_WEIGHTS="$ROOT/ad2_model_zoo/pretrained/dinov2_vitl14_reg4_pretrain.pth"
export SUPERAD_SKIP_POST_EVAL=1
export SUPERAD_SAVE_OVERLAYS=0
export SUPERAD_TILED_QUERY=1
export SUPERAD_TILED_ONLY=0
export SUPERAD_TILED_RESOLUTION=672
export SUPERAD_TILED_RESIDUAL_WEIGHT=0.10
unset SUPERAD_TEST_SPLIT || true

mkdir -p "$OUT"
cd "$REPO"

for category in "${CATEGORIES[@]}"; do
  category_out="$OUT/$category"
  expected=$(find "$DATA/$category/test_public/good" "$DATA/$category/test_public/bad" -type f | wc -l)
  component_root="$category_out/component_maps/seed=0/$category"
  global_count=$(find "$component_root" -type f -name '*_global.tiff' 2>/dev/null | wc -l || true)
  tiled_count=$(find "$component_root" -type f -name '*_tiled.tiff' 2>/dev/null | wc -l || true)
  if [[ "$global_count" -eq "$expected" && "$tiled_count" -eq "$expected" && "$expected" -gt 0 ]]; then
    echo "SKIP_COMPLETE $category $expected"
    continue
  fi

  rm -rf "$category_out"
  echo "CATEGORY_START $category $expected $(date -Is)"
  /opt/conda/bin/python -u test_public.py \
    --model_name dinov2_vitl14_reg \
    --data_root "$DATA" \
    --results_dir "$category_out" \
    --objects "$category" \
    --resolution 448 \
    --warmup_iters 1

  global_count=$(find "$component_root" -type f -name '*_global.tiff' | wc -l)
  tiled_count=$(find "$component_root" -type f -name '*_tiled.tiff' | wc -l)
  test "$global_count" -eq "$expected"
  test "$tiled_count" -eq "$expected"
  /opt/conda/bin/python - <<PY
import json
from pathlib import Path
root = Path("$OUT")
(root / "${category}_COMPLETE.json").write_text(json.dumps({
    "status": "complete",
    "category": "$category",
    "images": $expected,
    "reference_resolution": 448,
    "global_query_resolution": 448,
    "tiled_query_resolution": 672,
    "normal_calibration_source": "A67"
}, indent=2))
PY
  echo "CATEGORY_COMPLETE $category $expected $(date -Is)"
done

/opt/conda/bin/python - <<PY
import json
from pathlib import Path
root = Path("$OUT")
categories = ['can', 'fabric', 'fruit_jelly', 'rice', 'sheet_metal', 'vial', 'wallplugs', 'walnuts']
(root / 'A74_INFERENCE_COMPLETE.json').write_text(json.dumps({
    'status': 'complete',
    'categories': categories,
    'reference_resolution': 448,
    'global_query_resolution': 448,
    'tiled_query_resolution': 672,
    'normal_calibration_source': 'A67 normal validation maps with the same 448/672 protocol',
    'purpose': 'supplementary uniform global-448/tiled-672 evaluation without overwriting A69'
}, indent=2))
print('A74_INFERENCE_COMPLETE')
PY

cd "$ROOT"
/opt/conda/bin/python -u "$ROOT/cross_resolution_fusion/scripts/a74_evaluate_matched_normal_calibrated_fusion.py"
