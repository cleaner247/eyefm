#!/usr/bin/env bash
# Generic launcher for v9/v10-style E2Mo DL baseline runs.
#
# This script is the template the user (and any collaborators) should use to
# train more seeds. It saves per-seed preds (per_seed_preds_dl_*.npz) which
# can then be aggregated via scripts/aggregate_per_seed.py.
#
# Usage:
#   SEEDS="100,101,102" \
#   TASKS="detox_binary ad_binary epilepsy_binary mci_binary migraine_binary" \
#   GPU=0 ./scripts/finetune_downstream_per_seed.sh
#
# Env vars (with defaults):
#   DATA_ROOT   eyemae v6 root        /home/sichaohe/eyemae_v6
#   OUT_BASE    where to write runs   /home/sichaohe/experiments/v10_runs
#   TASKS       space-sep task list   (default: 5 binary tasks)
#   SEEDS       comma-sep seeds       "42,43,44"
#   EPOCHS      50
#   T_LEN       1024
#   GPU         0
#   DRY_RUN     1 to only print commands
#
# Each (task, arch) combo writes to:
#   ${OUT_BASE}/${task}_s1/arch_${arch}/   for TimesNet + CNNTransformer
#   ${OUT_BASE}/${task}_s2/arch_${arch}/   for TCN + NST
# And per-stage merged summary to:
#   ${OUT_BASE}/${task}_s{1,2}/baseline_summary.csv
set -euo pipefail

DATA_ROOT=${DATA_ROOT:-/home/sichaohe/eyemae_v6}
OUT_BASE=${OUT_BASE:-/home/sichaohe/experiments/v10_runs}
TASKS=${TASKS:-"detox_binary ad_binary epilepsy_binary mci_binary migraine_binary"}
SEEDS=${SEEDS:-"42,43,44"}
EPOCHS=${EPOCHS:-50}
T_LEN=${T_LEN:-1024}
GPU=${GPU:-0}
DRY_RUN=${DRY_RUN:-0}

# Find the right python (server convention)
PY=${PY:-/home/sichaohe/miniconda3/envs/rl/bin/python}
EYEFM_SRC=${EYEFM_SRC:-/home/sichaohe/eye-movement-lm/eyefm/src}

if [[ ! -d "$EYEFM_SRC" ]]; then
    echo "ERROR: $EYEFM_SRC not found; set EYEFM_SRC to your eyefm src/ dir" >&2
    exit 1
fi

run_arch() {
    local task=$1
    local arch=$2
    local stage=$3
    local outdir=$OUT_BASE/${task}_${stage}/arch_${arch}
    mkdir -p "$outdir"

    local log=$OUT_BASE/logs/${task}_${stage}_$(basename $outdir)_${arch}.log
    mkdir -p "$(dirname $log)"

    local cmd=(
        "$PY" -u -m baseline.run_baseline
        --data-root "$DATA_ROOT"
        --out-dir "$outdir"
        --task "$task"
        --dl-arch "$arch"
        --dl-seeds "$SEEDS"
        --dl-epochs "$EPOCHS"
        --dl-t-len "$T_LEN"
        --dl-use-swa --dl-use-logit-adjust
        --gpu "$GPU"
        --skip-ml
    )
    echo "[$(date '+%F %T')] $task / $stage / $arch -> $outdir"
    if [[ "$DRY_RUN" == "1" ]]; then
        printf '  DRY-RUN: %q ' "${cmd[@]}"; echo
    else
        (cd "$EYEFM_SRC" && "${cmd[@]}") > "$log" 2>&1
        echo "[$(date '+%F %T')] done rc=$? log=$log"
    fi
}

merge_stage() {
    local task=$1
    local stage=$2
    shift 2
    local archs=("$@")
    local merged=$OUT_BASE/${task}_${stage}/baseline_summary.csv
    mkdir -p "$(dirname $merged)"
    {
        head -1 "$OUT_BASE/${task}_${stage}/arch_${archs[0]}/baseline_summary.csv"
        for a in "${archs[@]}"; do
            tail -n +2 "$OUT_BASE/${task}_${stage}/arch_${a}/baseline_summary.csv"
        done
    } > "$merged"
    echo "  -> merged $merged"
}

# Stage 1: TimesNet + CNNTransformer  (fast archs)
# Stage 2: TCN + NST                   (slow arch, especially NST)
for task in $TASKS; do
    run_arch "$task" TimesNet       s1_timesnet_cnntrans
    run_arch "$task" CNNTransformer s1_timesnet_cnntrans
    merge_stage "$task" s1_timesnet_cnntrans TimesNet CNNTransformer
    run_arch "$task" TCN            s2_tcn_nst
    run_arch "$task" NST            s2_tcn_nst
    merge_stage "$task" s2_tcn_nst TCN NST
done

echo "[$(date '+%F %T')] All tasks done."
