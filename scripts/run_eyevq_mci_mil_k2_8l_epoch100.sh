#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

PYTHON_ENV="${PYTHON_ENV:-/home/jinfanhe/miniconda3/envs/jinf}"
TORCHRUN="$PYTHON_ENV/bin/torchrun"
CONFIG="${CONFIG:-configs/eyevq/downstream_mci_subject_mil.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/eyevq/downstream_mci_subject_mil_concat4cls_h32_k2_8l_lr1e5_epoch100_spg4_pat10_rawauc_es_seed42_valid_eye_mintask2_noclassquota}"

if [[ ! -x "$TORCHRUN" ]]; then
  echo "torchrun is not executable: $TORCHRUN" >&2
  exit 1
fi
if [[ -e "$OUTPUT_DIR/metrics_test.json" ]]; then
  echo "Completed result already exists: $OUTPUT_DIR/metrics_test.json"
  exit 0
fi

mkdir -p "$OUTPUT_DIR"
"$TORCHRUN" --standalone --nproc_per_node=4 \
  -m eyemae.eyevq.downstream.train_mil \
  --config "$CONFIG" \
  --output_dir "$OUTPUT_DIR" \
  --freeze-bottom-layers 4 \
  --encoder-lr 1e-5 \
  --epochs 100
