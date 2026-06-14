#!/usr/bin/env bash
set -euo pipefail

cd /mnt/disk_sde/eyemae
export MPLCONFIGDIR=/tmp/eyemae_matplotlib
mkdir -p "$MPLCONFIGDIR" outputs/logs

STATS_PATH="outputs/area_stats_subject_heldout_seed42.json"
LOG_PATH="outputs/logs/train_base_3gpu_plan.log"
: > "$LOG_PATH"

echo "Waiting for valid area stats at ${STATS_PATH}" | tee -a "$LOG_PATH"
until python - "$STATS_PATH" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.exists() or path.stat().st_size == 0:
    raise SystemExit(1)
with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)
if "global" not in data or "subjects" not in data:
    raise SystemExit(1)
if not data.get("subjects"):
    raise SystemExit(1)
PY
do
    date -Is | tee -a "$LOG_PATH"
    sleep 60
done

echo "Starting plan-aligned 3-GPU training" | tee -a "$LOG_PATH"
CUDA_VISIBLE_DEVICES=0,1,2 torchrun --standalone --nproc_per_node=3 \
  -m eyemae.train \
  --config configs/eyemae_cnn_512_12l.yaml 2>&1 | tee -a "$LOG_PATH"
