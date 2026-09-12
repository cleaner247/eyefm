#!/usr/bin/env bash
# Strict canonical EyeVQ pipeline. Use --detached to survive SSH disconnects.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON_ENV="${PYTHON_ENV:-$(python -c 'import sys; print(sys.prefix)')}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/eyevq/final}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
FINETUNE_NPROC="${FINETUNE_NPROC:-1}"

if [[ "${1:-}" == "--detached" ]]; then
  shift
  session="eyevq_final_$(date +%Y%m%d_%H%M%S)"
  absolute_output="$PROJECT_ROOT/$OUTPUT_ROOT"
  if [[ "$OUTPUT_ROOT" = /* ]]; then
    absolute_output="$OUTPUT_ROOT"
  fi
  mkdir -p "$absolute_output"
  printf -v command "cd %q && PYTHON_ENV=%q OUTPUT_ROOT=%q NPROC_PER_NODE=%q FINETUNE_NPROC=%q %q" \
    "$PROJECT_ROOT" "$PYTHON_ENV" "$OUTPUT_ROOT" "$NPROC_PER_NODE" "$FINETUNE_NPROC" "$PROJECT_ROOT/scripts/run_eyevq_final.sh"
  for argument in "$@"; do
    printf -v quoted_argument " %q" "$argument"
    command+="$quoted_argument"
  done
  printf -v quoted_log "%q" "$absolute_output/tmux.log"
  command+=" >> $quoted_log 2>&1"
  tmux new-session -d -s "$session" "$command"
  echo "Started detached tmux session: $session"
  echo "Pipeline state: $absolute_output/pipeline_state.json"
  exit 0
fi

export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$PYTHON_ENV/bin/python" -m eyemae.eyevq.pipeline \
  --output-root "$OUTPUT_ROOT" \
  --python-env "$PYTHON_ENV" \
  --nproc "$NPROC_PER_NODE" \
  --finetune-nproc "$FINETUNE_NPROC" \
  "$@"
