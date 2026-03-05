#!/usr/bin/env bash
# Full stage1 grid: 3 embedding_dims × 7 codebook (top,bottom) configs, 20k iter, batch 256.
# After each run, uploads checkpoints to GCS with a config stamp.
# Prerequisite: sync dataset from GCS, e.g.:
#   gsutil -m cp -r gs://semcom-sdsr-training-data-1772509648/dataset ./dataset

set -euo pipefail

DATA_PATH="${DATA_PATH:-./dataset}"
CKPT_DIR="${CKPT_DIR:-./checkpoints}"
GCS_CHECKPOINTS="gs://semcom-sdsr-training-data-1772509648/checkpoints"
N_ITER=20000
BATCH_SIZE=128

# All 6 DCASE2020 Task 2 machine types for stage1
MACHINE_TYPES="fan pump slider valve ToyCar ToyTrain"

# 3 embedding dims
EMBEDDING_DIMS=(64 128 256)

# 7 codebook configs (top, bottom) with top <= bottom, bottom in {2048, 4096}
CODEBOOK_CONFIGS=(
  "512:2048"
  "512:4096"
  "1024:2048"
  "1024:4096"
  "2048:2048"
  "2048:4096"
  "4096:4096"
)

if [[ ! -d "$DATA_PATH" ]]; then
  echo "Dataset not found at $DATA_PATH. Sync from GCS first, e.g.:"
  echo "  gsutil -m cp -r gs://semcom-sdsr-training-data-1772509648/dataset $DATA_PATH"
  exit 1
fi

for emb in "${EMBEDDING_DIMS[@]}"; do
  for cfg in "${CODEBOOK_CONFIGS[@]}"; do
    top="${cfg%%:*}"
    bot="${cfg##*:}"
    stamp="emb${emb}_top${top}_bot${bot}_iter${N_ITER}_bs${BATCH_SIZE}"
    run_name="ToyCar+ToyTrain+fan+pump+slider+valve"
    stage1_dir="${CKPT_DIR}/stage1/${run_name}"

    echo "=============================================="
    echo "Running: embedding_dim=$emb top=$top bottom=$bot ($stamp)"
    echo "=============================================="

    python scripts/train.py stage1 \
      --data_path "$DATA_PATH" \
      --machine_type $MACHINE_TYPES \
      --ckpt_dir "$CKPT_DIR" \
      --n_iter "$N_ITER" \
      --batch_size "$BATCH_SIZE" \
      --num_embeddings_top "$top" \
      --num_embeddings_bot "$bot" \
      --embedding_dim "$emb"

    if [[ -d "$stage1_dir" ]]; then
      dest="${GCS_CHECKPOINTS}/${stamp}/"
      echo "Uploading $stage1_dir -> $dest"
      gsutil -m cp -r "$stage1_dir" "$dest"
    else
      echo "Warning: stage1 dir not found at $stage1_dir, skipping upload"
    fi
  done
done

echo "Grid finished. All checkpoints uploaded under $GCS_CHECKPOINTS"