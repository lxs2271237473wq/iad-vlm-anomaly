#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/private_data/iad-vlm-anomaly
REPO="$ROOT/ad2_model_zoo/repos/SuperAD"
DATA="$ROOT/datasets/MVTec_AD_2"
RESOLUTION="${1:-672}"
OUT="$ROOT/ad2_model_zoo/results/a56_superad_reg4_sheetmetal_r${RESOLUTION}_v1"

export DINOV2_LOCAL_REPO="$ROOT/ad2_model_zoo/repos/dinov2"
export DINOV2_LOCAL_WEIGHTS="$ROOT/ad2_model_zoo/pretrained/dinov2_vitl14_reg4_pretrain.pth"
export SUPERAD_SKIP_POST_EVAL=1
export SUPERAD_SAVE_OVERLAYS=0

mkdir -p "$OUT"
cd "$REPO"
/opt/conda/bin/python -u test_public.py \
  --model_name dinov2_vitl14_reg \
  --data_root "$DATA" \
  --results_dir "$OUT/sheet_metal" \
  --objects sheet_metal \
  --resolution "$RESOLUTION" \
  --warmup_iters 1

expected=$(find "$DATA/sheet_metal/test_public/good" "$DATA/sheet_metal/test_public/bad" -type f | wc -l)
produced=$(find "$OUT/sheet_metal/anomaly_maps/seed=0/sheet_metal" -type f -name '*.tiff' | wc -l)
test "$produced" -eq "$expected"

/opt/conda/bin/python - <<PY
import json
from pathlib import Path
out=Path("$OUT")
(out/"A56_INFERENCE_COMPLETE.json").write_text(json.dumps({
    "status":"complete", "category":"sheet_metal", "resolution":int("$RESOLUTION"),
    "control":"A45 SuperAD-Reg4; resolution is the only intended variable"
}, indent=2))
print("A56_INFERENCE_COMPLETE", out)
PY

UNIFIED_METHOD="superad_reg4_r${RESOLUTION}" \
UNIFIED_LAYOUT=superad \
UNIFIED_CATEGORIES=sheet_metal \
UNIFIED_COMPLETION_MARKER=A56_INFERENCE_COMPLETE.json \
UNIFIED_MAP_ROOT="$OUT" \
UNIFIED_OUT="$ROOT/orbitad/results/a56_superad_reg4_sheetmetal_r${RESOLUTION}_eval_v1" \
/opt/conda/bin/python -u "$ROOT/work/stage25/a47_evaluate_superad_reg4_unified.py"
