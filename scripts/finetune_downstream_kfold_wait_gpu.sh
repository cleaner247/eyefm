#!/usr/bin/env bash
set -euo pipefail

GPU_ID=${GPU_ID:?set GPU_ID}
FOLD_LIST=${FOLD_LIST:?set FOLD_LIST}
MEMORY_THRESHOLD_MIB=${MEMORY_THRESHOLD_MIB:-2000}
POLL_SECONDS=${POLL_SECONDS:-300}
ROOT=${ROOT:-outputs/downstream_disease_binary_kfold_extra_seed42}
SPLIT_ROOT=${SPLIT_ROOT:-splits/downstream_disease_binary_kfold_extra_seed42}
MAKE_SPLITS=${MAKE_SPLITS:-0}
SUMMARIZE=${SUMMARIZE:-0}
SKIP_EXISTING=${SKIP_EXISTING:-1}

while true; do
  used_mib=$(nvidia-smi --id="${GPU_ID}" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
  if (( used_mib < MEMORY_THRESHOLD_MIB )); then
    date
    echo "[start] GPU ${GPU_ID} memory.used=${used_mib}MiB < ${MEMORY_THRESHOLD_MIB}MiB; launching fold(s): ${FOLD_LIST}"
    break
  fi
  date
  echo "[wait] GPU ${GPU_ID} memory.used=${used_mib}MiB >= ${MEMORY_THRESHOLD_MIB}MiB; sleeping ${POLL_SECONDS}s"
  sleep "${POLL_SECONDS}"
done

CUDA_VISIBLE_DEVICES="${GPU_ID}" \
FOLD_LIST="${FOLD_LIST}" \
MAKE_SPLITS="${MAKE_SPLITS}" \
SUMMARIZE="${SUMMARIZE}" \
SKIP_EXISTING="${SKIP_EXISTING}" \
ROOT="${ROOT}" \
SPLIT_ROOT="${SPLIT_ROOT}" \
bash scripts/finetune_downstream_kfold.sh
