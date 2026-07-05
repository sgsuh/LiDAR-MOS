# LiDAR-MOS inference image.
#
# Base: PyTorch 2.1.2 + CUDA 11.8 + cuDNN 8 (Python 3.10).
# CUDA 11.8 is the minimum toolkit that ships kernels for the Ada Lovelace
# architecture (sm_89) used by the RTX 4070. The repo's original recipe
# (CUDA 10.0 / PyTorch 1.5.1) cannot launch kernels on this GPU.
FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System libraries required by OpenCV (cv2) at runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Python inference dependencies.
COPY docker/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# The repository itself is bind-mounted at run time (see docker-compose.yml),
# so nothing from the source tree is copied into the image.
WORKDIR /workspace/LiDAR-MOS

CMD ["bash"]
