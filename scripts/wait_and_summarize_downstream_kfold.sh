#!/usr/bin/env bash
set -euo pipefail

SESSION_LIST=${SESSION_LIST:?set SESSION_LIST}
DISEASE_LIST=${DISEASE_LIST:?set DISEASE_LIST}
FOLD_LIST=${FOLD_LIST:-"0 1 2 3 4"}
ROOT=${ROOT:-outputs/downstream_disease_binary_kfold_extra_seed42}
POLL_SECONDS=${POLL_SECONDS:-300}

read -r -a SESSIONS <<< "${SESSION_LIST}"
read -r -a DISEASES <<< "${DISEASE_LIST}"
read -r -a FOLDS <<< "${FOLD_LIST}"

while true; do
  running=()
  for session in "${SESSIONS[@]}"; do
    if tmux has-session -t "${session}" 2>/dev/null; then
      running+=("${session}")
    fi
  done
  if (( ${#running[@]} == 0 )); then
    date
    echo "[summarize] all watched sessions exited"
    break
  fi
  date
  echo "[wait] still running: ${running[*]}"
  sleep "${POLL_SECONDS}"
done

SUMMARY_ARGS=()
for fold in "${FOLDS[@]}"; do
  SUMMARY_ARGS+=(--fold "${fold}")
done
for disease in "${DISEASES[@]}"; do
  SUMMARY_ARGS+=(--disease "${disease}")
done

python -m eyemae.summarize_downstream \
  --output_root "${ROOT}" \
  "${SUMMARY_ARGS[@]}"
