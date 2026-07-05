#!/usr/bin/env bash
# Build and run the TensorRT C++ LiDAR-MOS inference inside the `trt` container:
#
#   docker compose exec trt bash docker/run_cpp.sh
#
# Optional args:
#   $1  sequence dir   (default: data/sequences/08)
#   $2  output labels  (default: data/predictions_cpp_trt/sequences/08/predictions)
#   $3  png dir        (default: "" -> no PNGs; pass a dir to also dump range views)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

if [ ! -f work_dir/lmnet.onnx ]; then
    echo "work_dir/lmnet.onnx not found."
    echo "Generate it first: docker compose exec inference python3 tools/export_onnx.py"
    exit 1
fi

echo "=== Configuring & building (cpp/build) ==="
cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release
cmake --build cpp/build -j"$(nproc)"

SEQ_DIR="${1:-data/sequences/08}"
OUT_DIR="${2:-data/predictions_cpp_trt/sequences/08/predictions}"
PNG_DIR="${3:-}"

echo "=== Running C++ inference (seq=${SEQ_DIR}) ==="
# The C++ builds the TensorRT engine from ONNX on the first run and caches it
# at cpp/model/lmnet.engine for subsequent runs.
./cpp/build/lmnet "${SEQ_DIR}" cpp/cfg/lmnet.yaml "${OUT_DIR}" "${PNG_DIR}"
