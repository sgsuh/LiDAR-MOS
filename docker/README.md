# LiDAR-MOS Docker (Inference)

A container environment for running the **Using SalsaNext as the baseline →
Inferring** step from the main README. All installation and execution happen
inside the container only.

## Target environment
- GPU: NVIDIA RTX 4070 (Ada Lovelace, `sm_89`)
- Host: Docker Engine + Docker Compose v2 + NVIDIA Container Toolkit (WSL2 supported)

> ⚠️ The original README uses CUDA 10.0 / PyTorch 1.5.1, but that combination
> cannot launch CUDA kernels on the RTX 4070. This image therefore uses
> **PyTorch 2.1.2 + CUDA 11.8** (which ships `sm_89` kernels). To match the
> newer PyTorch, only two lines of the source were patched:
> `_ConvTransposeMixin` import in
> `mos_SalsaNext/.../modules/adf.py` (removed in PyTorch 1.6+) and
> `np.float` in `mos_SalsaNext/train/common/laserscan.py` (removed in NumPy 1.24).

## Prerequisites (data / model)
The following must exist under `data/` (skip if already present):
- Toy dataset: `data/sequences/08/{velodyne,labels,residual_images_1,...}`
- Pretrained model: `data/model_salsanext_residual_1/`
  ([download](https://www.ipb.uni-bonn.de/html/projects/LiDAR-MOS/model_salsanext_residual_1.zip),
  `unzip` before use)

## Usage

Run from the project root (`LiDAR-MOS/`).

```sh
# 1) Build the image
docker compose build

# 2) Start the container (detached, kept idle)
docker compose up -d

# 3) Run inference on the toy dataset (seq 08, valid split)
docker compose exec inference bash docker/run_infer.sh
```

Predictions are written to `data/predictions_salsanext_residual_1_new/` on the
host (the repository is bind-mounted into the container).

### Override paths / split
```sh
docker compose exec \
  -e SPLIT=valid \
  -e DATASET=/workspace/LiDAR-MOS/data \
  -e MODEL=/workspace/LiDAR-MOS/data/model_salsanext_residual_1 \
  -e LOG=/workspace/LiDAR-MOS/data/predictions_salsanext_residual_1_new \
  inference bash docker/run_infer.sh
```

### Open a shell in the container
```sh
docker compose exec inference bash
# Inside the container you can use the original README commands directly:
cd mos_SalsaNext/train/tasks/semantic
python3 infer.py -d ../../../../data -m ../../../../data/model_salsanext_residual_1 \
    -l ../../../../data/predictions_salsanext_residual_1_new -s valid
```

### Shut down
```sh
docker compose down
```

## Verify GPU access
```sh
docker compose exec inference python3 -c \
  "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# -> True NVIDIA GeForce RTX 4070 ...
```

## TensorRT + C++ inference
For the full PyTorch → ONNX → TensorRT → C++ pipeline (real-time C++ inference
on the toy dataset), see [tensorrt_cpp.md](tensorrt_cpp.md).

## Notes
- `tensorflow` / `vispy` are not required for inference and are intentionally
  excluded from the image (they are training-logging / visualization-only
  dependencies).
- To also run evaluation (`utils/evaluate_mos.py`) or training (`train.sh`)
  inside the container, additional dependencies (e.g. `tensorflow`, `vispy`)
  must be installed.
