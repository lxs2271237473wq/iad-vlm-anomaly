#!/usr/bin/env bash
set -euo pipefail
ROOT=/root/private_data/iad-vlm-anomaly
REPO="$ROOT/ad2_model_zoo/repos/SuperAD"
DATA="$ROOT/datasets/MVTec_AD_2"
OUT="$ROOT/ad2_model_zoo/results/a45_superad_reg4_public_v1"
export DINOV2_LOCAL_REPO="$ROOT/ad2_model_zoo/repos/dinov2"
export DINOV2_LOCAL_WEIGHTS="$ROOT/ad2_model_zoo/pretrained/dinov2_vitl14_reg4_pretrain.pth"
export SUPERAD_SKIP_POST_EVAL=1
cd "$REPO"

# Reverse tail lane stays at least three categories ahead of the primary lane.
for category in walnuts wallplugs vial; do
  category_out="$OUT/$category"
  expected=$(find "$DATA/$category/test_public/good" "$DATA/$category/test_public/bad" -type f | wc -l)
  produced=$(find "$category_out/anomaly_maps/seed=0/$category" -type f -name '*.tiff' 2>/dev/null | wc -l || true)
  if [ "$produced" -eq "$expected" ] && [ "$expected" -gt 0 ]; then
    echo "AUX_SKIP_COMPLETE $category"
    continue
  fi
  echo "AUX_CATEGORY_START $category $(date -Is)"
  /opt/conda/bin/python -u test_public.py \
    --model_name dinov2_vitl14_reg \
    --data_root "$DATA" \
    --results_dir "$category_out" \
    --objects "$category" \
    --warmup_iters 1
  produced=$(find "$category_out/anomaly_maps/seed=0/$category" -type f -name '*.tiff' | wc -l)
  test "$produced" -eq "$expected"
  echo "AUX_CATEGORY_COMPLETE $category $(date -Is)"
done
echo "A45_AUX_REVERSE_COMPLETE $(date -Is)"
