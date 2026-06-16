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
SKIP_EXISTING=${SKIP_EXISTING:-1}

for disease in "${DISEASES[@]}"; do
  for i in "${!CONFIGS[@]}"; do
    mode="${MODES[$i]}"
    out_dir="${ROOT}/${disease}/${mode}"
    if [[ "${SKIP_EXISTING}" == "1" && -f "${out_dir}/metrics_final.json" ]]; then
      echo "[skip] ${disease}/${mode} already has metrics_final.json"
      continue
    fi
    echo "[run] ${disease}/${mode}"
    python -m eyemae.finetune \
      --config "${CONFIGS[$i]}" \
      --disease "${disease}" \
      --mode "${mode}"
  done
done

python -m eyemae.summarize_downstream --output_root "${ROOT}"
