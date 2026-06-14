#!/usr/bin/env bash
set -euo pipefail

cd /mnt/disk_sde/eyemae
mkdir -p outputs/logs
export MPLCONFIGDIR=/tmp/eyemae_matplotlib
mkdir -p "$MPLCONFIGDIR"
export TERM=xterm-256color
unset CODEX_API_KEY

log_file=/mnt/disk_sde/eyemae/outputs/logs/tmux_codex_goal.log
{
  echo "===== EyeMAE tmux Codex goal start: $(date -Is) ====="
  echo "PWD=$(pwd)"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-}"
  codex \
    -m gpt-5.5 \
    -c model_reasoning_effort='"xhigh"' \
    -c service_tier='"fast"' \
    -s workspace-write \
    -a never \
    -C /mnt/disk_sde/eyemae \
    --add-dir /mnt/disk_sde/data-260606 \
    exec \
    --skip-git-repo-check \
    - < /mnt/disk_sde/eyemae/CODEX_TMUX_HANDOFF.md
  echo "===== EyeMAE tmux Codex goal end: $(date -Is) ====="
} 2>&1 | tee -a "$log_file"
