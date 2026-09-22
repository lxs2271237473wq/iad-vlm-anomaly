#!/usr/bin/env bash
set -euo pipefail

ROOT="${IAD_REPO_ROOT:-/root/private_data/iad-vlm-anomaly}"
REPO="$ROOT/ad2_model_zoo/repos/SuperAD"
DATA="$ROOT/datasets/MVTecAD"
OUT="$ROOT/ad2_model_zoo/results/a73_mvtec14_dual_resolution_v1"
# Historical 14-class run. The paper table now uses the 15-class evaluation
# (A76, cable included): run_a76_mvtec15_add_cable.sh + a76_evaluate_mvtec15_unified.py.
CATEGORIES=(grid capsule transistor zipper carpet leather tile wood bottle hazelnut metal_nut pill screw toothbrush)

export DINOV2_LOCAL_REPO="$ROOT/ad2_model_zoo/repos/dinov2"
export DINOV2_LOCAL_WEIGHTS="$ROOT/ad2_model_zoo/pretrained/dinov2_vitl14_reg4_pretrain.pth"
export SUPERAD_SKIP_POST_EVAL=1
export SUPERAD_SAVE_OVERLAYS=0
export SUPERAD_TILED_QUERY=1
export SUPERAD_TILED_ONLY=0
export SUPERAD_TILED_RESOLUTION=672
export SUPERAD_TILED_RESIDUAL_WEIGHT=0.10

mkdir -p "$OUT"
cd "$REPO"

run_split() {
  local category=$1 split=$2
  local category_out="$OUT/$split/$category"
  local source_split="$split"
  [[ "$split" == "test" ]] && source_split=test
  local expected
  expected=$(find "$DATA/$category/$source_split" -type f -name '*.png' | wc -l)
  local global_count tiled_count
  global_count=$(find "$category_out/component_maps/seed=0/$category" -type f -name '*_global.tiff' 2>/dev/null | wc -l || true)
  tiled_count=$(find "$category_out/component_maps/seed=0/$category" -type f -name '*_tiled.tiff' 2>/dev/null | wc -l || true)
  if [[ "$global_count" -eq "$expected" && "$tiled_count" -eq "$expected" && "$expected" -gt 0 ]]; then
    echo "SKIP_COMPLETE $category $split $expected"
    return
  fi
  rm -rf "$category_out"
  echo "SPLIT_START $category $split $(date -Is)"
  if [[ "$split" == "train" ]]; then
    export SUPERAD_TEST_SPLIT=train
  else
    unset SUPERAD_TEST_SPLIT
  fi
  /opt/conda/bin/python -u test_public.py \
    --dataset MVTec \
    --preprocess informed \
    --model_name dinov2_vitl14_reg \
    --data_root "$DATA" \
    --results_dir "$category_out" \
    --objects "$category" \
    --resolution 448 \
    --warmup_iters 1
  global_count=$(find "$category_out/component_maps/seed=0/$category" -type f -name '*_global.tiff' | wc -l)
  tiled_count=$(find "$category_out/component_maps/seed=0/$category" -type f -name '*_tiled.tiff' | wc -l)
  test "$global_count" -eq "$expected"
  test "$tiled_count" -eq "$expected"
  echo "SPLIT_COMPLETE $category $split $expected $(date -Is)"
}

for category in "${CATEGORIES[@]}"; do
  run_split "$category" train
  run_split "$category" test
  /opt/conda/bin/python - <<PY
import json
from pathlib import Path
root=Path("$OUT")
(root/f"${category}_COMPLETE.json").write_text(json.dumps({
  "status":"complete", "category":"$category", "normal_split":"train/good",
  "test_split":"test", "global_resolution":448, "tiled_resolution":672
}, indent=2))
PY
done

/opt/conda/bin/python - <<PY
import json
from pathlib import Path
root=Path("$OUT")
(root/"A73_INFERENCE_COMPLETE.json").write_text(json.dumps({
  "status":"complete", "dataset":"MVTec AD",
  "categories":"grid capsule transistor zipper carpet leather tile wood bottle hazelnut metal_nut pill screw toothbrush".split(),
  "normal_calibration":"train/good only", "global_resolution":448, "tiled_resolution":672
}, indent=2))
print("A73_INFERENCE_COMPLETE")
PY
