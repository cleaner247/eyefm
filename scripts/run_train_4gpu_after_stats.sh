#!/usr/bin/env bash
set -euo pipefail

cd /mnt/disk_sde/eyemae
export MPLCONFIGDIR=/tmp/eyemae_matplotlib
mkdir -p "$MPLCONFIGDIR" outputs/logs

STATS_PATH="outputs/area_stats_subject_heldout_seed42.json"
LOG_PATH="outputs/logs/train_4gpu.log"
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
PY
do
    date -Is | tee -a "$LOG_PATH"
    sleep 60
done

echo "This script is disabled: the current plan requires 3-GPU DDP." | tee -a "$LOG_PATH"
echo "Use scripts/train_base_3gpu.sh instead." | tee -a "$LOG_PATH"
exit 2
