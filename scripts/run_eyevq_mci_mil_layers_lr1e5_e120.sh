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
ENCODER_LR="1e-5"
EPOCHS="120"
PATIENCE="121"

if [[ ! -x "$TORCHRUN" ]]; then
  echo "torchrun is not executable: $TORCHRUN" >&2
  exit 1
fi

mkdir -p "$GRID_ROOT"

for unfrozen_layers in 6 8 10 12; do
  freeze_bottom_layers=$((12 - unfrozen_layers))
  output_dir="$GRID_ROOT/unfreeze${unfrozen_layers}_enc1em5_e120"

  if [[ -f "$output_dir/metrics_val_best.json" ]]; then
    echo "Skip completed: $output_dir"
    continue
  fi

  echo "Start: unfrozen_layers=$unfrozen_layers encoder_lr=$ENCODER_LR epochs=$EPOCHS"
  "$TORCHRUN" --standalone --nproc_per_node=4 \
    -m eyemae.eyevq.downstream.train_mil \
    --config "$CONFIG" \
    --output_dir "$output_dir" \
    --freeze-bottom-layers "$freeze_bottom_layers" \
    --encoder-lr "$ENCODER_LR" \
    --epochs "$EPOCHS" \
    --early-stopping-patience "$PATIENCE" \
    --skip-test
done

"$PYTHON_ENV/bin/python" -m eyemae.eyevq.downstream.summarize_grid \
  --grid-root "$GRID_ROOT"
