#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
START_STAGE="${START_STAGE:-1}"
TOKENIZER_CONFIG="${TOKENIZER_CONFIG:-configs/eyevq/tokenizer_joint.yaml}"
BERT_CONFIG="${BERT_CONFIG:-configs/eyevq/pretrain_joint.yaml}"
TOKENIZER_OUTPUT="${TOKENIZER_OUTPUT:-outputs/eyevq/tokenizer_joint_fsq9755_50k}"
CODE_IDS_CACHE="${CODE_IDS_CACHE:-outputs/eyevq/code_ids_joint_fsq9755_50k_v2.npz}"
BERT_OUTPUT="${BERT_OUTPUT:-outputs/eyevq/bert_joint_fsq9755_uniform25_50k}"
CACHE_BATCH_PER_GPU="${CACHE_BATCH_PER_GPU:-512}"
MASK_RATIO="${MASK_RATIO:-0.25}"
TOKENIZER_STEPS="${TOKENIZER_STEPS:-50000}"
BERT_STEPS="${BERT_STEPS:-50000}"
TOKENIZER_RESUME="${TOKENIZER_RESUME:-}"
TOKENIZER_MIN_LR="${TOKENIZER_MIN_LR:-}"
TOKENIZER_LR_DECAY_START_STEP="${TOKENIZER_LR_DECAY_START_STEP:-}"

TOKENIZER_CHECKPOINT="$TOKENIZER_OUTPUT/ckpt_final.pt"

run_distributed() {
  torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" "$@"
}

mkdir -p "$TOKENIZER_OUTPUT" "$(dirname "$CODE_IDS_CACHE")" "$BERT_OUTPUT"

if [[ ! "$START_STAGE" =~ ^[123]$ ]]; then
  echo "START_STAGE must be 1, 2, or 3; got: $START_STAGE" >&2
  exit 2
fi

if (( START_STAGE <= 1 )); then
  echo "[1/3] Train tokenizer for ${TOKENIZER_STEPS} steps"
  tokenizer_args=(
    -m eyemae.eyevq.tokenizer.train
    --config "$TOKENIZER_CONFIG"
    --output_dir "$TOKENIZER_OUTPUT"
    --max-steps "$TOKENIZER_STEPS"
  )
  if [[ -n "$TOKENIZER_RESUME" ]]; then
    tokenizer_args+=(--resume "$TOKENIZER_RESUME")
  fi
  if [[ -n "$TOKENIZER_MIN_LR" ]]; then
    tokenizer_args+=(--min-lr "$TOKENIZER_MIN_LR")
  fi
  if [[ -n "$TOKENIZER_LR_DECAY_START_STEP" ]]; then
    tokenizer_args+=(--lr-decay-start-step "$TOKENIZER_LR_DECAY_START_STEP")
  fi
  run_distributed "${tokenizer_args[@]}"
else
  echo "[1/3] Skip tokenizer training (START_STAGE=$START_STAGE)"
fi

if [[ ! -f "$TOKENIZER_CHECKPOINT" ]]; then
  echo "Tokenizer final checkpoint was not created: $TOKENIZER_CHECKPOINT" >&2
  exit 1
fi

if (( START_STAGE <= 2 )); then
  echo "[2/3] Precompute train+validation code IDs once"
  run_distributed -m eyemae.eyevq.precompute_codes \
    --config "$BERT_CONFIG" \
    --tokenizer-checkpoint "$TOKENIZER_CHECKPOINT" \
    --out "$CODE_IDS_CACHE" \
    --split all \
    --n-batch "$CACHE_BATCH_PER_GPU"
else
  echo "[2/3] Skip code-ID precomputation (START_STAGE=$START_STAGE)"
fi

if [[ ! -f "$CODE_IDS_CACHE" ]]; then
  echo "Code-ID cache was not created: $CODE_IDS_CACHE" >&2
  exit 1
fi

echo "[3/3] Train BERT for ${BERT_STEPS} steps from cached labels only"
run_distributed -m eyemae.eyevq.pretrain.train \
  --config "$BERT_CONFIG" \
  --output_dir "$BERT_OUTPUT" \
  --tokenizer-checkpoint "$TOKENIZER_CHECKPOINT" \
  --code-ids-cache "$CODE_IDS_CACHE" \
  --max-steps "$BERT_STEPS" \
  --mask-ratio "$MASK_RATIO"
