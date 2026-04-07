# Docker Setup (Reproducible Vast.ai Runtime)

This project can be run in a Docker container as a controlled environment
(OS libraries + Python packages + ffmpeg).

## Why use this

- Avoid host `ffmpeg` / shared-library mismatches (`libopenh264.so.*` issues).
- Keep the same runtime on local machine and Vast.ai.
- Reproduce experiments with fixed dependencies.

## Build image

From repo root:

```bash
docker build -t auddsr:latest .
```

## Run interactively (recommended)

This opens a shell in the container so you can run any Python script manually.

```bash
docker run --rm -it \
  --gpus all \
  -v "$PWD":/workspace/audDSR \
  -w /workspace/audDSR \
  auddsr:latest
```

Then inside container:

```bash
check_env
micromamba run -n sDSR python -c "import torch; print('cuda?', torch.cuda.is_available())"
micromamba run -n sDSR python -m scripts.smoke_test_payload_channels --seed 0 --ber 1e-5
```

## Run one command directly

```bash
docker run --rm -it \
  --gpus all \
  -v "$PWD":/workspace/audDSR \
  -w /workspace/audDSR \
  auddsr:latest \
  micromamba run -n sDSR python -m scripts.evaluate_awgn_jscc --help
```

## Notes for Vast.ai

- Ensure the instance supports Docker + NVIDIA runtime.
- Keep datasets/checkpoints mounted as volumes rather than copied into image.
- Reuse the same image tag for all final runs to keep experiments consistent.
- Validate startup once with:

```bash
check_env
ffmpeg -encoders | rg opus
ffmpeg -decoders | rg opus
```

## Using gsutil inside container

`gsutil` is installed into the `sDSR` environment in this Docker image.

Example:

```bash
micromamba run -n sDSR gsutil ls gs://your-bucket/path
micromamba run -n sDSR gsutil -m cp -r gs://your-bucket/dataset ./dataset
```

### Authentication

Prefer a service account key mounted as a file:

```bash
docker run --rm -it \
  --gpus all \
  -v "$PWD":/workspace/audDSR \
  -v "/path/to/gcp-key.json:/secrets/gcp-key.json:ro" \
  -e GOOGLE_APPLICATION_CREDENTIALS=/secrets/gcp-key.json \
  -w /workspace/audDSR \
  auddsr:latest
```

Inside container:

```bash
micromamba run -n sDSR gsutil ls gs://your-bucket
```

## Vast.ai startup flow

Two practical approaches:

1) **Best reproducibility**: build/push image once (e.g. Docker Hub), then select that image in Vast.ai.
2) Build on instance from repo (slower but simple).

Typical on-start commands when building on instance:

```bash
git clone <your-repo-url> /workspace/audDSR
cd /workspace/audDSR
docker build -t auddsr:latest .
docker run --rm -it --gpus all \
  -v /workspace/audDSR:/workspace/audDSR \
  -w /workspace/audDSR \
  auddsr:latest \
  micromamba run -n sDSR python -m scripts.smoke_test_payload_channels --seed 0 --ber 1e-5
```

For long experiments, run commands through `docker run ... micromamba run -n sDSR python -m ...`
without `-it`, and write outputs to mounted paths.

