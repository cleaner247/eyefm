#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

K2_SESSION="${K2_SESSION:-eyevq_mci_k2_epoch100_noclassquota}"
K2_DIR="${K2_DIR:-outputs/eyevq/downstream_mci_subject_mil_concat4cls_h32_k2_8l_lr1e5_epoch100_spg4_pat10_rawauc_es_seed42_valid_eye_mintask2_noclassquota}"
K4_DIR="${K4_DIR:-outputs/eyevq/downstream_mci_subject_mil_concat4cls_h32_k4_8l_lr1e5_epoch100_spg4_pat10_rawauc_es_seed42_valid_eye_mintask4_noclassquota}"
COMPARE_DIR="${COMPARE_DIR:-outputs/eyevq/downstream_mci_k2_vs_k4_epoch100_comparison}"

echo "Waiting for K=2 final test: $K2_DIR/metrics_test.json"
while [[ ! -f "$K2_DIR/metrics_test.json" ]]; do
  if ! tmux has-session -t "$K2_SESSION" 2>/dev/null; then
    echo "K=2 session ended without metrics_test.json" >&2
    exit 1
  fi
  sleep 20
done

echo "K=2 test complete; starting K=4"
OUTPUT_DIR="$K4_DIR" bash scripts/run_eyevq_mci_mil_k4_8l_epoch100.sh

if [[ ! -f "$K4_DIR/metrics_test.json" ]]; then
  echo "K=4 ended without metrics_test.json" >&2
  exit 1
fi

echo "K=4 test complete; writing comparison"
/home/jinfanhe/miniconda3/envs/jinf/bin/python scripts/compare_eyevq_mci_k2_k4.py \
  --k2-dir "$K2_DIR" \
  --k4-dir "$K4_DIR" \
  --out-dir "$COMPARE_DIR"
echo "Comparison complete: $COMPARE_DIR/comparison.md"
