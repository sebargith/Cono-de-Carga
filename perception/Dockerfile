FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV MPLCONFIGDIR=/workspace/.cache/matplotlib
ENV YOLO_CONFIG_DIR=/workspace/.cache/ultralytics

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip setuptools wheel

RUN python -m pip install \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu124

COPY requirements/docker.txt /tmp/requirements-docker.txt
RUN python -m pip install -r /tmp/requirements-docker.txt

COPY . /workspace

CMD ["bash"]

