#!/usr/bin/env bash
set -euo pipefail

ROOT="/root/private_data/iad-vlm-anomaly"
DOWNLOAD_DIR="$ROOT/downloads/mvtec_ad2"
DATASET_DIR="$ROOT/datasets/MVTec_AD_2"

mkdir -p "$DOWNLOAD_DIR"
mkdir -p "$DATASET_DIR"

declare -A URLS
URLS["can"]="https://www.mydrive.ch/shares/121501/26456e2f3ef813930866f8f9b072593a/download/466651130-1743159807/can.tar.gz"
URLS["fabric"]="https://www.mydrive.ch/shares/150999/2c2026421bcf68e37e9268885d355081/download/466651519-1743162446/fabric.tar.gz"
URLS["fruit_jelly"]="https://www.mydrive.ch/shares/121503/951a46ce30a3af3787ce9671cfa8613a/download/466651800-1743164023/fruit_jelly.tar.gz"
URLS["rice"]="https://www.mydrive.ch/shares/121504/0014676292c3c44931712a54fb3bdbe8/download/466653907-1743164943/rice.tar.gz"
URLS["sheet_metal"]="https://www.mydrive.ch/shares/121505/2d8fcdc8e988456bdd18696746eda0a0/download/466654829-1743166795/sheet_metal.tar.gz"
URLS["vial"]="https://www.mydrive.ch/shares/121506/739dc6459c939fe464c0d26acc6c2d55/download/466654885-1743167505/vial.tar.gz"
URLS["wallplugs"]="https://www.mydrive.ch/shares/121507/66fe6e114b498e03be8d48c711794be7/download/466655287-1743168151/wallplugs.tar.gz"
URLS["walnuts"]="https://www.mydrive.ch/shares/121508/9fcf67e49f0dc61a9608f57ba0482356/download/466656233-1743168988/walnuts.tar.gz"

ORDER=(
  "can"
  "fabric"
  "fruit_jelly"
  "rice"
  "sheet_metal"
  "vial"
  "wallplugs"
  "walnuts"
)

echo "===== MVTec AD 2 full category download ====="
echo "download dir: $DOWNLOAD_DIR"
echo "dataset dir : $DATASET_DIR"

for category in "${ORDER[@]}"; do
  archive="$DOWNLOAD_DIR/${category}.tar.gz"
  target_dir="$DATASET_DIR/$category"
  url="${URLS[$category]}"

  echo ""
  echo "========== [$category] =========="
  df -h "$ROOT"

  if [ -d "$target_dir" ]; then
    img_count=$(find "$target_dir" -type f \( -iname "*.png" -o -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.bmp" -o -iname "*.tif" -o -iname "*.tiff" \) | wc -l)
    if [ "$img_count" -gt 0 ]; then
      echo "[SKIP] $target_dir already exists with $img_count images."
      continue
    fi
  fi

  echo "[DOWNLOAD] $category"
  wget -c --tries=20 --timeout=120 --show-progress \
    -O "$archive" \
    "$url"

  echo "[CHECK] archive type and integrity"
  ls -lh "$archive"
  if command -v file >/dev/null 2>&1; then
    file "$archive"
  else
    echo "[INFO] file command not available; skip file-type check"
  fi
  tar -tzf "$archive" >/dev/null

  echo "[EXTRACT] $category -> $DATASET_DIR"
  tar -xzf "$archive" -C "$DATASET_DIR"

  echo "[REMOVE ARCHIVE] $archive"
  rm -f "$archive"

  if [ -d "$target_dir" ]; then
    img_count=$(find "$target_dir" -type f \( -iname "*.png" -o -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.bmp" -o -iname "*.tif" -o -iname "*.tiff" \) | wc -l)
    echo "[DONE] $category images: $img_count"
  else
    echo "[WARN] expected target dir not found: $target_dir"
    find "$DATASET_DIR" -maxdepth 1 -type d | sort
  fi
done

echo ""
echo "===== final dataset tree ====="
find "$DATASET_DIR" -maxdepth 2 -type d | sort

echo ""
echo "===== final size ====="
du -sh "$DATASET_DIR" "$DOWNLOAD_DIR"
