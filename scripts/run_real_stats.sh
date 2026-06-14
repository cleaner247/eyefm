#!/usr/bin/env bash
set -euo pipefail

cd /mnt/disk_sde/eyemae
export MPLCONFIGDIR=/tmp/eyemae_matplotlib
mkdir -p "$MPLCONFIGDIR" outputs/logs

python -u -m eyemae.make_splits --config configs/eyemae_cnn_512_12l.yaml 2>&1 | tee outputs/logs/real_make_splits.log
python -u -m eyemae.compute_area_stats --config configs/eyemae_cnn_512_12l.yaml --split pretrain_train --out outputs/area_stats_subject_heldout_seed42.json 2>&1 | tee outputs/logs/real_area_stats.log
