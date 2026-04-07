# syntax=docker/dockerfile:1.6

# GPU-capable base image for Vast.ai and local NVIDIA Docker.
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV FFMPEG_BIN=/usr/bin/ffmpeg

# System dependencies + ffmpeg from Ubuntu repo (self-consistent shared libs).
RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    bzip2 \
    ca-certificates \
    curl \
    git \
    ffmpeg \
    libsndfile1 \
    tini \
    && rm -rf /var/lib/apt/lists/*

# Install micromamba (small/fast conda-compatible package manager).
# Use "latest" endpoint for reliability across mirrors.
RUN curl -L "https://micro.mamba.pm/api/micromamba/linux-64/latest" -o /tmp/micromamba.tar.bz2 \
    && tar -xjf /tmp/micromamba.tar.bz2 -C /usr/local/bin --strip-components=1 bin/micromamba \
    && rm -f /tmp/micromamba.tar.bz2

ENV MAMBA_ROOT_PREFIX=/opt/micromamba
ENV MAMBA_EXE=/usr/local/bin/micromamba

WORKDIR /workspace/audDSR

# Copy environment metadata first to maximize Docker cache reuse.
COPY environment.yml ./environment.yml

# Create conda env and install project extras.
RUN micromamba create -y -n sDSR -f environment.yml \
    && micromamba run -n sDSR pip install --no-build-isolation pyldpc gsutil \
    && micromamba clean --all --yes

# Copy project after env creation to keep rebuilds faster.
COPY . .

# Add a small startup check helper.
RUN printf '%s\n' \
    '#!/usr/bin/env bash' \
    'set -euo pipefail' \
    'echo "[env] Python: $(micromamba run -n sDSR python --version)"' \
    'echo "[env] ffmpeg: $(ffmpeg -version | head -n 1)"' \
    'echo "[env] opus encoder present? $(ffmpeg -encoders 2>/dev/null | grep -c "libopus" || true)"' \
    > /usr/local/bin/check_env && chmod +x /usr/local/bin/check_env

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["bash"]

