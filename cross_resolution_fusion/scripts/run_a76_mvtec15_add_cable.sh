#!/usr/bin/env bash
# A76: generate MVTec AD cable component maps under the exact A73 protocol,
# so cable can be merged into one 15-class external evaluation instead of being
# reported as a separate single-category transfer experiment.
#
# Writes only into <A73 maps>/<split>/cable/. The 14 existing categories are not
# touched. Use a new experiment number (A76) rather than overwriting A73 results.
set -euo pipefail

ROOT="${IAD_REPO_ROOT:-/root/private_data/iad-vlm-anomaly}"
REPO="$ROOT/ad2_model_zoo/repos/SuperAD"
DATA="$ROOT/datasets/MVTecAD"
OUT="$ROOT/ad2_model_zoo/results/a73_mvtec14_dual_resolution_v1"
CATEGORY=cable

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
  local split=$1
  local category_out="$OUT/$split/$CATEGORY"
  local expected global_count tiled_count
  expected=$(find "$DATA/$CATEGORY/$split" -type f -name '*.png' | wc -l)
  global_count=$(find "$category_out/component_maps/seed=0/$CATEGORY" -type f -name '*_global.tiff' 2>/dev/null | wc -l || true)
  tiled_count=$(find "$category_out/component_maps/seed=0/$CATEGORY" -type f -name '*_tiled.tiff' 2>/dev/null | wc -l || true)
  if [[ "$global_count" -eq "$expected" && "$tiled_count" -eq "$expected" && "$expected" -gt 0 ]]; then
    echo "SKIP_COMPLETE $CATEGORY $split $expected"
    return
  fi
  rm -rf "$category_out"
  echo "SPLIT_START $CATEGORY $split $(date -Is)"
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
    --objects "$CATEGORY" \
    --resolution 448 \
    --warmup_iters 1
  global_count=$(find "$category_out/component_maps/seed=0/$CATEGORY" -type f -name '*_global.tiff' | wc -l)
  tiled_count=$(find "$category_out/component_maps/seed=0/$CATEGORY" -type f -name '*_tiled.tiff' | wc -l)
  test "$global_count" -eq "$expected"
  test "$tiled_count" -eq "$expected"
  echo "SPLIT_COMPLETE $CATEGORY $split $expected $(date -Is)"
}

run_split train
run_split test

/opt/conda/bin/python - <<PY
import json
from pathlib import Path
root = Path("$OUT")
(root / "A76_CABLE_ADDENDUM.json").write_text(json.dumps({
  "status": "complete",
  "purpose": "add MVTec AD cable to the A73 map set so a single 15-class evaluation is possible",
  "category": "cable",
  "reference_resolution": 448,
  "global_resolution": 448,
  "tiled_resolution": 672,
  "preprocess": "informed",
  "normal_split": "train/good",
  "test_split": "test",
  "supersedes_for_paper_table": "a59_superad_reg4_mvtec_cable_tiled_query_v1",
  "why_regenerated": "A59 was produced on 2026-09-18 with a src/detection.py revision that was modified again on 2026-09-19; its component maps therefore cannot be certified as the same 448/672 protocol as the other 14 categories",
  "scope": "writes only into $OUT/{train,test}/cable; the 14 existing categories are untouched",
}, indent=2))
print("A76_CABLE_ADDENDUM_WRITTEN")
PY
