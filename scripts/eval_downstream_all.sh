#!/usr/bin/env bash
set -euo pipefail

DISEASES=("AD" "MCI" "PD相关" "偏头痛" "戒毒所" "癫痫")
CONFIGS=(
  "configs/downstream/disease_binary_scratch.yaml"
  "configs/downstream/disease_binary_linear_probe.yaml"
  "configs/downstream/disease_binary_partial.yaml"
  "configs/downstream/disease_binary_full.yaml"
)
MODES=("scratch" "pretrained_linear_probe" "pretrained_partial" "pretrained_full")
ROOT=${ROOT:-outputs/downstream_disease_binary_seed42}

for disease in "${DISEASES[@]}"; do
  for i in "${!CONFIGS[@]}"; do
    mode="${MODES[$i]}"
    python -m eyemae.evaluate_downstream \
      --config "${CONFIGS[$i]}" \
      --disease "${disease}" \
      --checkpoint "${ROOT}/${disease}/${mode}/checkpoint_best.pt" \
      --output_dir "${ROOT}/${disease}/${mode}/eval_best"
  done
done
