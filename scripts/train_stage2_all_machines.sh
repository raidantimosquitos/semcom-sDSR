#!/usr/bin/env bash
# Train stage2 for each machine type and each DCASE dev machine_id (spectromorphic masks follow machine_type):
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
N_ITER=10000
BATCH_SIZE=16

# Single shared stage1 trained on all machine types (same run_name as experiment_grid.sh)
RUN_NAME="ToyCar+ToyConveyor+fan+pump+slider+valve"
STAGE1_DIR="${CKPT_DIR}/stage1_e128_h128_K4096/stage1/${RUN_NAME}"
STAGE1_BEST="${STAGE1_DIR}/stage1_${RUN_NAME}_final.pt"
# Optional: override if you use a different stage1 path or stamp
if [[ -n "${STAGE1_CKPT:-}" ]]; then
  STAGE1_BEST="$STAGE1_CKPT"
fi

# All 6 DCASE2020 Task 2 machine types for stage2
MACHINE_TYPES=(ToyCar ToyConveyor fan pump slider valve)

# DCASE2020 Task 2 dev subset: machine_ids differ by machine_type.
# Optional: space-separated list in MACHINE_IDS overrides the per-type defaults below for every type.
MACHINE_IDS="${MACHINE_IDS:-}"

machine_ids_for_type() {
  case "$1" in
    fan | slider | pump | valve)
      echo "id_00 id_02 id_04 id_06"
      ;;
    ToyCar)
      echo "id_01 id_02 id_03 id_04"
      ;;
    ToyConveyor)
      echo "id_01 id_02 id_03"
      ;;
    *)
      echo "train_stage2_all_machines.sh: unknown machine_type: $1" >&2
      exit 1
      ;;
  esac
}

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

# --ckpt_dir is the checkpoint root; train.py Stage2Trainer appends stage2/<machine_type>/[<machine_id>/]
for machine_type in "${MACHINE_TYPES[@]}"; do
  if [[ -n "$MACHINE_IDS" ]]; then
    read -r -a IDS <<< "$MACHINE_IDS"
  else
    read -r -a IDS <<< "$(machine_ids_for_type "$machine_type")"
  fi

  for machine_id in "${IDS[@]}"; do
    echo "=============================================="
    echo "Stage2: machine_type=$machine_type machine_id=$machine_id ($N_ITER iter, bs=$BATCH_SIZE)"
    echo "=============================================="

    python scripts/train.py stage2 \
      --data_path "$DATA_PATH" \
      --machine_type "$machine_type" \
      --machine_id "$machine_id" \
      --ckpt_dir "$CKPT_DIR" \
      --stage1_ckpt "$STAGE1_BEST" \
      --n_iter "$N_ITER" \
      --batch_size "$BATCH_SIZE" \
      --anomaly_sampling "distant" \
      --val_every 500 \
      --num_workers 32
  done
done

echo "Stage2 finished. Checkpoints uploaded under ${GCS_CHECKPOINTS}/${STAMP}/"
