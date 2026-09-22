#!/usr/bin/env bash
# A77: true 2D local-view components for the "local view vs resolution" experiment.
#
# Emits, in ONE run per category (identical bank, reference list and preprocessing):
#   <stem>_global.tiff      whole image at the global resolution (448)
#   <stem>_global672.tiff   whole image at 672
#   <stem>_tiled.tiff       legacy tiled component (kept for comparison)
#   <stem>_local672.tiff    2D sliding-window local views at 672
#
# The A77 evaluator then forms five configurations from these components:
#   G448, G672, legacy-mean = 0.5*(global+tiled), T-local = local672,
#   G448+T-local = 0.5*(global+local672).
#
# All new switches default to off in src/detection.py, so other runs are unaffected.
set -euo pipefail

ROOT="${IAD_REPO_ROOT:-/root/private_data/iad-vlm-anomaly}"
REPO="$ROOT/ad2_model_zoo/repos/SuperAD"
OUT="$ROOT/ad2_model_zoo/results/a77_local_view_dual_resolution_v1"
CATEGORY="${1:?usage: run_a77_local_view.sh <category> [smoke|full]}"
MODE="${2:-full}"
DATA="${A77_DATA_ROOT:-$ROOT/datasets/MVTecAD}"
DATASET="${A77_DATASET:-MVTec}"
PREPROCESS="${A77_PREPROCESS:-informed}"
GLOBAL_RES="${A77_GLOBAL_RES:-448}"
TILED_RES="${A77_TILED_RES:-672}"

export DINOV2_LOCAL_REPO="$ROOT/ad2_model_zoo/repos/dinov2"
export DINOV2_LOCAL_WEIGHTS="$ROOT/ad2_model_zoo/pretrained/dinov2_vitl14_reg4_pretrain.pth"
export SUPERAD_SKIP_POST_EVAL=1
export SUPERAD_SAVE_OVERLAYS=0
export SUPERAD_TILED_QUERY=1
export SUPERAD_TILED_ONLY=0
export SUPERAD_TILED_RESOLUTION="$TILED_RES"
export SUPERAD_TILED_RESIDUAL_WEIGHT=0.10
export SUPERAD_EXTRA_GLOBAL_RESOLUTIONS="$TILED_RES"
export SUPERAD_LOCAL_QUERY=1
export SUPERAD_LOCAL_WINDOW_FRACTION=0.5
export SUPERAD_LOCAL_STRIDE_FRACTION=0.5
if [[ "$MODE" == "smoke" ]]; then
  export SUPERAD_MAX_TEST_IMAGES="${SUPERAD_MAX_TEST_IMAGES:-5}"
else
  unset SUPERAD_MAX_TEST_IMAGES || true
fi

mkdir -p "$OUT"
cd "$REPO"

run_split() {
  local split=$1
  local category_out="$OUT/$split/$CATEGORY"
  # MVTec AD 2 keeps its public test set in test_public/, MVTec AD in test/.
  local source_split="${A77_TEST_SPLIT:-test}"
  if [[ "$split" == "train" ]]; then
    source_split=train
  fi
  local expected
  expected=$(find "$DATA/$CATEGORY/$source_split" -type f -name '*.png' \
               -not -path '*ground_truth*' | wc -l)
  if [[ "$expected" -le 0 ]]; then
    echo "ABORT $CATEGORY $split: no images under $DATA/$CATEGORY/$source_split"
    exit 1
  fi
  if [[ "$MODE" == "smoke" ]]; then expected="$SUPERAD_MAX_TEST_IMAGES"; fi
  local global_count
  # skip only when EVERY component is present; checking a single suffix let a
  # previous run without the newer outputs masquerade as complete
  local have_all=1 count
  for suffix in global global672 tiled local672 localstd672 localnorm672 localcenter672; do
    count=$(find "$category_out/component_maps/seed=0/$CATEGORY" -type f -name "*_${suffix}.tiff" 2>/dev/null | wc -l || true)
    if [[ "$count" -ne "$expected" ]]; then
      have_all=0
      echo "INCOMPLETE $CATEGORY $split: $suffix has $count of $expected"
    fi
  done
  if [[ "$have_all" -eq 1 ]]; then
    echo "SKIP_COMPLETE $CATEGORY $split $expected"
    return
  fi
  rm -rf "$category_out"
  echo "SPLIT_START $CATEGORY $split $(date -Is)"
  if [[ "$split" == "train" ]]; then export SUPERAD_TEST_SPLIT=train; else unset SUPERAD_TEST_SPLIT || true; fi
  /opt/conda/bin/python -u test_public.py \
    --dataset "$DATASET" \
    --preprocess "$PREPROCESS" \
    --model_name dinov2_vitl14_reg \
    --data_root "$DATA" \
    --results_dir "$category_out" \
    --objects "$CATEGORY" \
    --resolution "$GLOBAL_RES" \
    --warmup_iters 1
  for suffix in local672 localstd672 localnorm672 localcenter672; do
    produced=$(find "$category_out/component_maps/seed=0/$CATEGORY" -type f -name "*_${suffix}.tiff" | wc -l)
    if [[ "$produced" -ne "$expected" ]]; then
      echo "ABORT $CATEGORY $split: $suffix produced $produced of $expected"
      exit 1
    fi
  done
  echo "SPLIT_COMPLETE $CATEGORY $split $expected/$expected (local/std/norm) $(date -Is)"
}

if [[ "$MODE" == "smoke" || "$MODE" == "test" ]]; then
  # test-only: the five A77 comparison configurations need no normal calibration
  # constants, so the train/good component maps can be skipped entirely.
  run_split test
else
  run_split train
  run_split test
fi

/opt/conda/bin/python - <<PY
import json
from pathlib import Path
out = Path("$OUT")
(out / "A77_RUN_${CATEGORY}_${MODE}.json").write_text(json.dumps({
  "status": "complete", "category": "$CATEGORY", "mode": "$MODE",
  "dataset": "$DATASET", "data_root": "$DATA", "preprocess": "$PREPROCESS",
  "global_resolution": int("$GLOBAL_RES"), "component_resolution": int("$TILED_RES"),
  "local_window_fraction_of_short_edge": 0.5, "local_stride_fraction_of_window": 0.5,
  "components": ["global", "global$TILED_RES", "tiled", "local$TILED_RES"],
  "supersedes_nothing": "new experiment number; A69/A73/A76 results are untouched",
}, indent=2))
print("A77_RUN_MARKER_WRITTEN")
PY
