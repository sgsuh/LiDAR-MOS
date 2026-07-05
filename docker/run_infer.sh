#!/usr/bin/env bash
# Run SalsaNext LiDAR-MOS inference on the toy dataset (sequence 08, valid split).
# Intended to be executed *inside* the container:
#
#   docker compose exec inference bash docker/run_infer.sh
#
# Optional overrides via environment variables:
#   DATASET   dataset root      (default: data)
#   MODEL     pretrained model  (default: data/model_salsanext_residual_1)
#   LOG       prediction output (default: data/predictions_salsanext_residual_1_new)
#   SPLIT     train|valid|test  (default: valid)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

DATASET="${DATASET:-${REPO_ROOT}/data}"
MODEL="${MODEL:-${REPO_ROOT}/data/model_salsanext_residual_1}"
LOG="${LOG:-${REPO_ROOT}/data/predictions_salsanext_residual_1_new}"
SPLIT="${SPLIT:-valid}"

echo "=== GPU visibility check ==="
python3 -c "import torch; print('torch', torch.__version__, '| cuda available:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

echo "=== Running inference (split=${SPLIT}) ==="
cd "${REPO_ROOT}/mos_SalsaNext/train/tasks/semantic"
python3 infer.py -d "${DATASET}" -m "${MODEL}" -l "${LOG}" -s "${SPLIT}"

echo "=== Done. Predictions written to: ${LOG} ==="
