#!/usr/bin/env bash
# Stage-1 grid: num_embeddings_coarse = num_embeddings_fine ∈ {512,1024,2048},
# embedding_dim_coarse = embedding_dim_fine ∈ {64,128}.
# 20k iterations, batch 256, default hidden_channels (coarse=256, fine=64).
# After each train run, full test-set reconstruction MSE (mean over all elements).
#
# Results: ./results/stage1_symmetric_codebook_grid.csv (and per-run JSON under ./results/stage1_symmetric_runs/)
#
# Prerequisite: dataset at DATA_PATH (DCASE layout).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT"

DATA_PATH="${DATA_PATH:-./dataset/dcase2020_task2_dev_dataset}"
# Root for this grid; each combo writes under ${CKPT_ROOT}/<stamp>/stage1/... so runs do not overwrite.
CKPT_ROOT="${CKPT_ROOT:-./checkpoints/stage1_symmetric_grid}"
N_ITER=20000
BATCH_SIZE=256
RESULTS_DIR="${RESULTS_DIR:-./results}"
RUNS_DIR="${RESULTS_DIR}/stage1_symmetric_runs"
CSV_OUT="${RESULTS_DIR}/stage1_symmetric_codebook_grid.csv"

MACHINE_TYPES=(fan pump slider valve ToyCar ToyConveyor)
# Must match train.py: "+".join(sorted(machine_types))
RUN_NAME="ToyCar+ToyConveyor+fan+pump+slider+valve"

CODEBOOK_SIZES=(512 1024 2048)
EMBEDDING_DIMS=(64 128)

if [[ ! -d "$DATA_PATH" ]]; then
  echo "Dataset not found at $DATA_PATH"
  exit 1
fi

mkdir -p "$RUNS_DIR"
if [[ ! -f "$CSV_OUT" ]]; then
  echo "num_embeddings,embedding_dim,n_iter,batch_size,avg_test_mse,ckpt_path,json_path" >"$CSV_OUT"
fi

for K in "${CODEBOOK_SIZES[@]}"; do
  for emb in "${EMBEDDING_DIMS[@]}"; do
    stamp="K${K}_emb${emb}_iter${N_ITER}_bs${BATCH_SIZE}"
    echo "=============================================="
    echo "Training: K_coarse=K_fine=$K  emb_coarse=emb_fine=$emb  ($stamp)"
    echo "=============================================="

    RUN_CKPT_DIR="${CKPT_ROOT}/${stamp}"
    python scripts/train.py stage1 \
      --data_path "$DATA_PATH" \
      --machine_type "${MACHINE_TYPES[@]}" \
      --ckpt_dir "$RUN_CKPT_DIR" \
      --n_iter "$N_ITER" \
      --batch_size "$BATCH_SIZE" \
      --num_embeddings_coarse "$K" \
      --num_embeddings_fine "$K" \
      --embedding_dim_coarse "$emb" \
      --embedding_dim_fine "$emb"

    CKPT_PATH="${RUN_CKPT_DIR}/stage1/${RUN_NAME}/stage1_${RUN_NAME}_best.pt"
    if [[ ! -f "$CKPT_PATH" ]]; then
      echo "ERROR: expected checkpoint missing: $CKPT_PATH"
      exit 1
    fi

    JSON_PATH="${RUNS_DIR}/${stamp}.json"
    AVG_LINE="$(python scripts/stage1_test_recon_mse.py \
      --stage1_ckpt "$CKPT_PATH" \
      --data_path "$DATA_PATH" \
      --machine_types "${MACHINE_TYPES[@]}" \
      --batch_size "$BATCH_SIZE" \
      --json_out "$JSON_PATH")"
    echo "$AVG_LINE"

    # Parse avg_mse from JSON (avoid brittle grep on scientific notation)
    AVG_MSE="$(python -c "import json; print(json.load(open('$JSON_PATH'))['avg_mse'])")"

    echo "${K},${emb},${N_ITER},${BATCH_SIZE},${AVG_MSE},\"${CKPT_PATH}\",\"${JSON_PATH}\"" >>"$CSV_OUT"
  done
done

echo "Done. Summary: $CSV_OUT"
