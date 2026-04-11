#!/usr/bin/env bash
# Train stage2 for each machine type with multiple anomaly strategies:
# load best stage1 (all types), 10k iter, batch 16.
# After each run, upload stage2 checkpoints to GCS.
# Prerequisite: dataset and stage1 best checkpoint available (e.g. from experiment_grid.sh).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT"

DATA_PATH="${DATA_PATH:-./dataset/dcase2020_task2_dev_dataset}"
CKPT_DIR="${CKPT_DIR:-./checkpoints}"
GCS_CHECKPOINTS="${GCS_CHECKPOINTS:-gs://semcom-sdsr-training-data-1772509648/checkpoints_apr09_larger_model}"
N_ITER=2500
BATCH_SIZE=16

# Single shared stage1 trained on all machine types (same run_name as experiment_grid.sh)
RUN_NAME="ToyCar+ToyConveyor+fan+pump+slider+valve"
STAGE1_DIR="${CKPT_DIR}/stage1/${RUN_NAME}"
STAGE1_BEST="${STAGE1_DIR}/stage1_${RUN_NAME}_best.pt"
# Optional: override if you use a different stage1 path or stamp
if [[ -n "${STAGE1_CKPT:-}" ]]; then
  STAGE1_BEST="$STAGE1_CKPT"
fi

# All 6 DCASE2020 Task 2 machine types for stage2
MACHINE_TYPES=(slider valve fan)

# Anomaly mask strategies to train stage2 with
ANOMALY_STRATEGIES=(both)

# Optional: space-separated machine_ids (e.g. "id_00 id_01 id_02"). When set, train/eval per (machine_type, machine_id)
# with other machine_ids of same type used as adversarial samples (mask all 1s). When unset, one run per machine_type (all IDs).
MACHINE_IDS="${MACHINE_IDS:-}"

# Stamp for this stage2 run on GCS
STAMP="stage2_10k_bs${BATCH_SIZE}"

if [[ ! -d "$DATA_PATH" ]]; then
  echo "Dataset not found at $DATA_PATH. Sync from GCS first."
  exit 1
fi

if [[ ! -f "$STAGE1_BEST" ]]; then
  echo "Stage1 best checkpoint not found at $STAGE1_BEST."
  echo "Train stage1 (all machine types) first or set STAGE1_CKPT to your checkpoint path."
  exit 1
fi

# One run per (machine_type, anomaly_strategy) (all machine_ids)
for machine_type in "${MACHINE_TYPES[@]}"; do
  for anomaly_strategy in "${ANOMALY_STRATEGIES[@]}"; do
    echo "=============================================="
    echo "Stage2: machine_type=$machine_type anomaly_strategy=$anomaly_strategy ($N_ITER iter, bs=$BATCH_SIZE)"
    echo "=============================================="

    # Isolate checkpoints/results per strategy so best checkpoints don't overwrite
    stage2_dir="${CKPT_DIR}/stage2/${machine_type}/${anomaly_strategy}"
    mkdir -p "$stage2_dir"

    python scripts/train.py stage2 \
      --data_path "$DATA_PATH" \
      --machine_type "$machine_type" \
      --ckpt_dir "$stage2_dir" \
      --stage1_ckpt "$STAGE1_BEST" \
      --n_iter "$N_ITER" \
      --batch_size "$BATCH_SIZE" \
      --anomaly_sampling "uniform" \
      --anomaly_strategy "$anomaly_strategy" \
      --val_every 500

  done
done

echo "Stage2 finished. Checkpoints uploaded under ${GCS_CHECKPOINTS}/${STAMP}/"
