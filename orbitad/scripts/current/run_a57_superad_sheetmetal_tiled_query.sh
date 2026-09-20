#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/private_data/iad-vlm-anomaly
REPO="$ROOT/ad2_model_zoo/repos/SuperAD"
DATA="$ROOT/datasets/MVTec_AD_2"
OUT="$ROOT/ad2_model_zoo/results/a57_superad_reg4_sheetmetal_tiled_query_v1"

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
  --model_name dinov2_vitl14_reg \
  --data_root "$DATA" \
  --results_dir "$OUT/sheet_metal" \
  --objects sheet_metal \
  --resolution 448 \
  --warmup_iters 1

expected=$(find "$DATA/sheet_metal/test_public/good" "$DATA/sheet_metal/test_public/bad" -type f | wc -l)
produced=$(find "$OUT/sheet_metal/anomaly_maps/seed=0/sheet_metal/test" -type f -name '*.tiff' | wc -l)
test "$produced" -eq "$expected"

/opt/conda/bin/python - <<PY
import json
from pathlib import Path
out=Path("$OUT")
(out/"A57_INFERENCE_COMPLETE.json").write_text(json.dumps({
    "status":"complete", "category":"sheet_metal", "reference_resolution":448,
    "query_mode":"global 448 plus non-overlapping square tiles at 672",
    "fusion":"0.90 global + 0.10 q99.5-aligned tiled residual",
    "control":"same backbone, selected references, augmentation, layers and kNN as A45"
}, indent=2))
print("A57_INFERENCE_COMPLETE", out)
PY

UNIFIED_METHOD=superad_reg4_tiled_query \
UNIFIED_LAYOUT=superad \
UNIFIED_CATEGORIES=sheet_metal \
UNIFIED_COMPLETION_MARKER=A57_INFERENCE_COMPLETE.json \
UNIFIED_MAP_ROOT="$OUT" \
UNIFIED_OUT="$ROOT/orbitad/results/a57_superad_reg4_sheetmetal_tiled_query_eval_v1" \
/opt/conda/bin/python -u "$ROOT/work/stage25/a47_evaluate_superad_reg4_unified.py"
