#!/usr/bin/env bash
set -euo pipefail

if [[ -n "${DISEASE_LIST:-}" ]]; then
  read -r -a DISEASES <<< "${DISEASE_LIST}"
else
  DISEASES=("AD" "MCI" "偏头痛" "癫痫")
fi

CONFIGS=(
  "configs/downstream/disease_binary_scratch.yaml"
  "configs/downstream/disease_binary_linear_probe.yaml"
  "configs/downstream/disease_binary_partial.yaml"
  "configs/downstream/disease_binary_full.yaml"
)
MODES=("scratch" "pretrained_linear_probe" "pretrained_partial" "pretrained_full")

CONFIG_FOR_SPLITS=${CONFIG_FOR_SPLITS:-configs/downstream/disease_binary_kfold_extra.yaml}
ROOT=${ROOT:-outputs/downstream_disease_binary_kfold_extra_seed42}
SPLIT_ROOT=${SPLIT_ROOT:-splits/downstream_disease_binary_kfold_extra_seed42}
NUM_FOLDS=${NUM_FOLDS:-5}
SKIP_EXISTING=${SKIP_EXISTING:-1}
MAKE_SPLITS=${MAKE_SPLITS:-1}
SUMMARIZE=${SUMMARIZE:-1}

if [[ -n "${FOLD_LIST:-}" ]]; then
  read -r -a FOLDS <<< "${FOLD_LIST}"
else
  FOLDS=()
  for ((fold = 0; fold < NUM_FOLDS; fold++)); do
    FOLDS+=("${fold}")
  done
fi

if [[ "${MAKE_SPLITS}" == "1" ]]; then
  SPLIT_ARGS=()
  for disease in "${DISEASES[@]}"; do
    SPLIT_ARGS+=(--disease "${disease}")
  done
  python -m eyemae.make_downstream_splits \
    --config "${CONFIG_FOR_SPLITS}" \
    --strategy subject_stratified_kfold \
    --num_folds "${NUM_FOLDS}" \
    --out_dir "${SPLIT_ROOT}" \
    "${SPLIT_ARGS[@]}"
fi

for fold in "${FOLDS[@]}"; do
  split_dir="${SPLIT_ROOT}/fold_${fold}"
  fold_root="${ROOT}/fold_${fold}"
  for disease in "${DISEASES[@]}"; do
    for i in "${!CONFIGS[@]}"; do
      mode="${MODES[$i]}"
      out_dir="${fold_root}/${disease}/${mode}"
      if [[ "${SKIP_EXISTING}" == "1" && -f "${out_dir}/metrics_final.json" ]]; then
        echo "[skip] fold_${fold}/${disease}/${mode} already has metrics_final.json"
        continue
      fi
      echo "[run] fold_${fold}/${disease}/${mode}"
      python -m eyemae.finetune \
        --config "${CONFIGS[$i]}" \
        --disease "${disease}" \
        --mode "${mode}" \
        --split_dir "${split_dir}" \
        --output_root "${fold_root}"
    done
  done
done

if [[ "${SUMMARIZE}" == "1" ]]; then
  SUMMARY_ARGS=()
  for fold in "${FOLDS[@]}"; do
    SUMMARY_ARGS+=(--fold "${fold}")
  done
  for disease in "${DISEASES[@]}"; do
    SUMMARY_ARGS+=(--disease "${disease}")
  done

  python -m eyemae.summarize_downstream \
    --output_root "${ROOT}" \
    "${SUMMARY_ARGS[@]}"
fi
