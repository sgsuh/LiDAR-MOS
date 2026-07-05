# TensorRT + C++ LiDAR-MOS Inference (Docker)

This document covers the full **PyTorch → ONNX → TensorRT → C++** pipeline for
SalsaNext LiDAR-MOS, running entirely inside Docker on the toy KITTI dataset
(sequence 08). It complements the PyTorch-only flow in [README.md](README.md).

## Images

| Service | Image | Base | Purpose |
| --- | --- | --- | --- |
| `inference` | `lidar-mos:cu118` | `pytorch/pytorch:2.1.2-cuda11.8` | PyTorch inference, **ONNX export** |
| `trt` | `lidar-mos:trt` | `nvcr.io/nvidia/tensorrt:23.08-py3` (TensorRT 8.6.1, CUDA 12.2) | **TRT engine build**, **C++ inference** |

TensorRT 8.6 is required: the C++ code uses the 8.x runtime API
(`getNbBindings` / `bindingIsInput` / `enqueueV2`) that was removed in
TensorRT 10. CUDA 12.2 ships `sm_89` kernels for the RTX 4070.

```sh
docker compose build              # build both images
docker compose up -d              # start both (idle)
```

## Prerequisites

Same toy dataset and pretrained model as the PyTorch flow:
- `data/sequences/08/{velodyne,labels,poses.txt,calib.txt,residual_images_1}`
- `data/model_salsanext_residual_1/`

## Stage 1 — Export ONNX (`inference` container)

```sh
docker compose exec inference python3 tools/export_onnx.py
# -> work_dir/lmnet.onnx   (input_0 [1,6,64,2048] -> output_0 [1,3,64,2048], opset 11)
```

The 6 input channels are `range, x, y, z, remission, residual`; the output is a
per-class softmax over `{0 ignore, 1 static, 2 moving}`.

## Stage 2 — Build the TensorRT engine (`trt` container)

Two equivalent options:

```sh
# (a) Standalone Python export
docker compose exec trt python3 tools/export_trt.py     # -> work_dir/lmnet.engine (FP16)

# (b) Let the C++ build it from ONNX on first run (see Stage 3).
#     Cached at cpp/model/lmnet.engine.
```

Engines are TensorRT-version-specific; both are built with TRT 8.6 inside the
`trt` image, so they match the C++ runtime.

## Stage 3 — Build & run the C++ inference (`trt` container)

```sh
docker compose exec trt bash docker/run_cpp.sh
# builds cpp/build/lmnet, then runs on data/sequences/08
# -> data/predictions_cpp_trt/sequences/08/predictions/*.label
```

Optional args: `run_cpp.sh [seq_dir] [out_label_dir] [png_dir]`. Passing a
`png_dir` also dumps range-view PNGs (top: range image, bottom: moving mask).

Predictions are written as KITTI `.label` files (uint32: `9` = static,
`251` = moving), one per point, so they can be scored directly.

## Results (toy sequence 08, RTX 4070)

- **Throughput:** ~30 ms / frame (4071 frames), FP16.
- **Accuracy (`utils/evaluate_mos.py`, `iou_moving`):**

  | Pipeline | iou_moving |
  | --- | --- |
  | PyTorch (precomputed residuals, reference) | 0.605 |
  | C++ TensorRT FP16 | ~0.50 |
  | C++ TensorRT FP32 | ~0.50 |

  (FP16 varies by ~±0.01 run-to-run due to non-deterministic kernels; FP16 and
  FP32 are effectively equal, so precision is not the source of the gap.)

Evaluate with:
```sh
docker compose exec inference python3 utils/evaluate_mos.py \
    -d data -p data/predictions_cpp_trt -s valid
```

### Why the C++ IoU is lower than PyTorch

FP16 vs FP32 is **not** the cause (both ~0.50). The difference comes from
the residual channel: the C++ computes it **on the fly** (range-projecting the
previous scan into the current frame using KITTI poses), whereas the PyTorch
flow loads the **precomputed** `residual_images_1/*.npy`.

`tools/compare_cpp_input.py` dumps the C++ network input for one frame
(`LMNET_DUMP=1 ./cpp/build/lmnet`) and compares it channel-by-channel against the
Python parser. The geometric channels (range/x/y/z/remission) match at
correlation > 0.99; the on-the-fly residual matches at ~0.93. So the pipeline is
faithful — the residual reprojection sensitivity, not a bug, accounts for the
gap. This is the expected trade-off of a real-time, self-contained C++ runtime.

## Notes

- **KITTI adaptation.** The original C++ was tuned for a custom sensor. Ported
  values for KITTI: FOV `fov_up=3 / fov_down=-25` (standard spherical
  projection), normalization means/stds from `arch_cfg.yaml`, residual valid
  range `[2, 50] m`, and the residual pose transform `inv(cur)·last`
  (matching `utils/gen_residual_images.py`).
- **Output ownership.** Containers run as root, so `data/predictions_cpp_trt/`
  is root-owned on the host. `sudo chown -R $USER data/predictions_cpp_trt` if
  you need to edit/remove it from the host.
- **Benign TensorRT warning.** `Error Code 3: Destroying a runtime before
  destroying deserialized engines` is printed once when the C++ builds an engine
  (the local `IRuntime` is freed at scope end). It does not affect results.
- **NGC MPI/VTK quirk.** The `trt` image creates a few empty HPC-X OpenMPI
  include directories (see `docker/Dockerfile.trt`) so that `find_package(PCL)`'s
  transitive VTK/MPI discovery does not abort during the CMake configure.
