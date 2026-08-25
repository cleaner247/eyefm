#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

PYTHON_ENV="${PYTHON_ENV:-/home/jinfanhe/miniconda3/envs/jinf}"
TORCHRUN="$PYTHON_ENV/bin/torchrun"
CONFIG="${CONFIG:-configs/eyevq/downstream_mci_subject_mil.yaml}"
GRID_ROOT="${GRID_ROOT:-outputs/eyevq/downstream_mci_subject_mil_layers_lr1e5_e120_seed42}"
OUTPUT_DIR="$GRID_ROOT/full_finetune_enc1em5_e120"

# The four partial-unfreeze runs own GPUs 0-3. Start full fine-tuning only after
# all four validation-only results exist and their CUDA contexts are released.
while true; do
  completed="$(find "$GRID_ROOT" -mindepth 2 -maxdepth 2 -name metrics_val_best.json | wc -l)"
  if [[ "$completed" -ge 4 ]]; then
    break
  fi
  sleep 60
done
sleep 30

if [[ ! -f "$OUTPUT_DIR/metrics_val_best.json" ]]; then
  "$TORCHRUN" --standalone --nproc_per_node=4 \
    -m eyemae.eyevq.downstream.train_mil \
    --config "$CONFIG" \
    --output_dir "$OUTPUT_DIR" \
    --freeze-bottom-layers 0 \
    --unfreeze-embedding \
    --encoder-lr 1e-5 \
    --epochs 120 \
    --early-stopping-patience 121 \
    --skip-test
fi

"$PYTHON_ENV/bin/python" -m eyemae.eyevq.downstream.summarize_grid \
  --grid-root "$GRID_ROOT"
