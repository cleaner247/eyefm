#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

PYTHON_ENV="${PYTHON_ENV:-/home/jinfanhe/miniconda3/envs/jinf}"
GRID_ROOT="${GRID_ROOT:-outputs/eyevq/downstream_mci_subject_mil_layers_lr1e5_e120_seed42}"

while true; do
  completed="$(find "$GRID_ROOT" -mindepth 2 -maxdepth 2 -name metrics_val_best.json | wc -l)"
  if [[ "$completed" -eq 5 ]]; then
    break
  fi
  sleep 60
done

# Let the fourth torchrun release its CUDA context after writing validation results.
sleep 30

"$PYTHON_ENV/bin/python" -m eyemae.eyevq.downstream.summarize_grid \
  --grid-root "$GRID_ROOT"
CUDA_VISIBLE_DEVICES=0 "$PYTHON_ENV/bin/python" -m eyemae.eyevq.downstream.evaluate_all_grid \
  --grid-root "$GRID_ROOT" --expected-runs 5
