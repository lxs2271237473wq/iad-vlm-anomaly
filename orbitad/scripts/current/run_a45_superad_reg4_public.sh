#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/private_data/iad-vlm-anomaly
REPO="$ROOT/ad2_model_zoo/repos/SuperAD"
DATA="$ROOT/datasets/MVTec_AD_2"
OUT="$ROOT/ad2_model_zoo/results/a45_superad_reg4_public_v1"
export DINOV2_LOCAL_REPO="$ROOT/ad2_model_zoo/repos/dinov2"
export DINOV2_LOCAL_WEIGHTS="$ROOT/ad2_model_zoo/pretrained/dinov2_vitl14_reg4_pretrain.pth"
export SUPERAD_SKIP_POST_EVAL=1
mkdir -p "$OUT"
cd "$REPO"

for category in can fabric fruit_jelly rice sheet_metal vial wallplugs walnuts; do
  category_out="$OUT/$category"
  expected=$(find "$DATA/$category/test_public/good" "$DATA/$category/test_public/bad" -type f | wc -l)
  produced=$(find "$category_out/anomaly_maps/seed=0/$category" -type f -name '*.tiff' 2>/dev/null | wc -l || true)
  if [ "$produced" -eq "$expected" ] && [ "$expected" -gt 0 ]; then
    echo "SKIP_COMPLETE $category"
    continue
  fi
  echo "CATEGORY_START $category $(date -Is)"
  /opt/conda/bin/python -u test_public.py \
    --model_name dinov2_vitl14_reg \
    --data_root "$DATA" \
    --results_dir "$category_out" \
    --objects "$category" \
    --warmup_iters 1
  produced=$(find "$category_out/anomaly_maps/seed=0/$category" -type f -name '*.tiff' | wc -l)
  test "$produced" -eq "$expected"
  echo "CATEGORY_COMPLETE $category $(date -Is)"
done

/opt/conda/bin/python - <<'PY'
import json
from pathlib import Path
root=Path('/root/private_data/iad-vlm-anomaly/ad2_model_zoo/results/a45_superad_reg4_public_v1')
categories=['can','fabric','fruit_jelly','rice','sheet_metal','vial','wallplugs','walnuts']
for category in categories:
    expected=sum(1 for p in (Path('/root/private_data/iad-vlm-anomaly/datasets/MVTec_AD_2')/category/'test_public').glob('*/*') if p.is_file() and p.parent.name != 'ground_truth')
    produced=sum(1 for p in (root/category/'anomaly_maps'/'seed=0'/category).rglob('*.tiff'))
    assert produced == expected > 0, (category, produced, expected)
(root/'A45_INFERENCE_COMPLETE.json').write_text(json.dumps({
    'status':'complete', 'categories':categories,
    'method':'SuperAD algorithm with DINOv2-L/14-Reg4 backbone',
    'official_difference':'official SuperAD uses non-register DINOv2-L/14',
}, indent=2))
print('A45_INFERENCE_COMPLETE')
PY
