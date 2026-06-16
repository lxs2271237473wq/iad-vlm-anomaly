#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${1:-./datasets}"
mkdir -p "${ROOT_DIR}"

# VisA is hosted as a public AWS Open Data object. No AWS account is required.
aws s3 cp --no-sign-request --region us-west-2 \
  s3://amazon-visual-anomaly/VisA_20220922.tar \
  "${ROOT_DIR}/VisA_20220922.tar"

mkdir -p "${ROOT_DIR}/VisA"
tar -xf "${ROOT_DIR}/VisA_20220922.tar" -C "${ROOT_DIR}/VisA"

echo "VisA downloaded and extracted to ${ROOT_DIR}/VisA"
