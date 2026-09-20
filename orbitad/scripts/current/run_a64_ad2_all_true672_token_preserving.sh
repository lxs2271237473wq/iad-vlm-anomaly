#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/private_data/iad-vlm-anomaly
REPO="$ROOT/ad2_model_zoo/repos/SuperAD"
DATA="$ROOT/datasets/MVTec_AD_2"
OUT="$ROOT/ad2_model_zoo/results/a64_superad_reg4_ad2_all_true672_token_preserving_v1"
SOURCE_A60="$ROOT/ad2_model_zoo/results/a60_superad_reg4_sheetmetal_true672_tiled_query_v1/sheet_metal/component_maps/seed=0/sheet_metal"
CATEGORIES=(can fabric fruit_jelly rice sheet_metal vial wallplugs walnuts)

export DINOV2_LOCAL_REPO="$ROOT/ad2_model_zoo/repos/dinov2"
export DINOV2_LOCAL_WEIGHTS="$ROOT/ad2_model_zoo/pretrained/dinov2_vitl14_reg4_pretrain.pth"
export SUPERAD_SKIP_POST_EVAL=1
export SUPERAD_SAVE_OVERLAYS=0
export SUPERAD_TILED_QUERY=1
export SUPERAD_TILED_ONLY=1
export SUPERAD_TILED_RESOLUTION=672

mkdir -p "$OUT"
cd "$REPO"

for category in "${CATEGORIES[@]}"; do
  category_out="$OUT/$category"
  expected=$(find "$DATA/$category/test_public/good" "$DATA/$category/test_public/bad" -type f | wc -l)
  produced=$(find "$category_out/anomaly_maps/seed=0/$category/test" -type f -name '*.tiff' 2>/dev/null | wc -l || true)
  if [[ "$produced" -eq "$expected" && "$expected" -gt 0 ]]; then
    echo "SKIP_COMPLETE $category"
    continue
  fi

  if [[ "$category" == "sheet_metal" ]]; then
    rm -rf "$category_out"
    for anomaly_type in good bad; do
      dst="$category_out/anomaly_maps/seed=0/$category/test/$anomaly_type"
      mkdir -p "$dst"
      for src in "$SOURCE_A60/$anomaly_type"/*_tiled.tiff; do
        name=$(basename "$src" _tiled.tiff)
        ln "$src" "$dst/$name.tiff"
      done
    done
    produced=$(find "$category_out/anomaly_maps/seed=0/$category/test" -type f -name '*.tiff' | wc -l)
    test "$produced" -eq "$expected"
    echo "REUSED_A60 $category $produced"
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
  produced=$(find "$category_out/anomaly_maps/seed=0/$category/test" -type f -name '*.tiff' | wc -l)
  test "$produced" -eq "$expected"
  echo "CATEGORY_COMPLETE $category $(date -Is)"
done

/opt/conda/bin/python - <<'PY'
import json
from pathlib import Path
root=Path('/root/private_data/iad-vlm-anomaly/ad2_model_zoo/results/a64_superad_reg4_ad2_all_true672_token_preserving_v1')
categories=['can','fabric','fruit_jelly','rice','sheet_metal','vial','wallplugs','walnuts']
(root/'A64_INFERENCE_COMPLETE.json').write_text(json.dumps({
  'status':'complete', 'categories':categories, 'reference_resolution':448,
  'query_resolution':672, 'query_policy':'square-window tiling along the long axis; one full-image window for square images',
  'fusion':'none; tiled/local query map only', 'sheet_metal_source':'A60 tiled component maps'
}, indent=2))
print('A64_INFERENCE_COMPLETE')
PY

UNIFIED_METHOD=superad_reg4_true672_token_preserving \
UNIFIED_LAYOUT=superad \
UNIFIED_COMPLETION_MARKER=A64_INFERENCE_COMPLETE.json \
UNIFIED_MAP_ROOT="$OUT" \
UNIFIED_OUT="$ROOT/orbitad/results/a64_superad_reg4_ad2_all_true672_token_preserving_eval_v1" \
/opt/conda/bin/python -u "$ROOT/work/stage25/a47_evaluate_superad_reg4_unified.py"
