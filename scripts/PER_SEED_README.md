# Per-seed E2Mo DL baseline workflow

This workflow trains an E2Mo DL baseline (TCN / TimesNet / NST / CNNTransformer) on
eye-movement downstream tasks and reports **per-seed mean ± std** for every metric,
not just the ensemble.

## Why

The main `run_baseline.py` only saves an **ensemble** `test_preds_dl_<arch>_<task>.npz`
(3 seeds' predicted probabilities averaged → 1 threshold from val → 1 metric per task).
That is fine for "what does the deployed 3-seed model look like", but for paper
reporting we usually want each seed evaluated independently, then mean ± std.

This patch adds:
- `per_seed_preds_dl_<arch>_<task>_seed<seed>.npz` — one npz per seed, written
  by `baseline.dl_baseline` after each seed's training run.
- `scripts/aggregate_per_seed.py` — recomputes every metric on each seed's
  predictions, then prints and writes `per_seed_metrics.csv` with per-seed rows
  + a `mean` and `std` row.

## How to train more seeds (give to collaborators)

```bash
# 1. Tell the script which seeds, tasks, and which GPU to use
SEEDS="100,101,102" \
TASKS="detox_binary ad_binary epilepsy_binary mci_binary migraine_binary" \
GPU=0 \
OUT_BASE=/home/sichaohe/experiments/v10_runs_seed100 \
DATA_ROOT=/home/sichaohe/eyemae_v6 \
PY=/home/sichaohe/miniconda3/envs/rl/bin/python \
EYEFM_SRC=/home/sichaohe/eye-movement-lm/eyefm/src \
./scripts/finetune_downstream_per_seed.sh
```

What you get:
```
${OUT_BASE}/
├── detox_binary_s1_timesnet_cnntrans/
│   ├── arch_TimesNet/
│   │   ├── per_seed_preds_dl_TimesNet_detox_binary_seed100.npz
│   │   ├── per_seed_preds_dl_TimesNet_detox_binary_seed101.npz
│   │   ├── per_seed_preds_dl_TimesNet_detox_binary_seed102.npz
│   │   ├── test_preds_dl_TimesNet_detox_binary.npz   # 3-seed ensemble (unchanged)
│   │   ├── best_ckpt_dl_TimesNet_detox_binary.pt
│   │   └── baseline_summary.csv
│   ├── arch_CNNTransformer/...
│   └── baseline_summary.csv           # merged (TimesNet + CNNTransformer)
├── detox_binary_s2_tcn_nst/...
└── logs/                               # per-run stdout
```

## How to compute per-seed mean ± std for a (task, arch)

```bash
cd $EYEFM_SRC
python -m baseline.aggregate_per_seed \
    --out-dir ${OUT_BASE}/detox_binary_s1_timesnet_cnntrans/arch_TimesNet \
    --task detox_binary \
    --arch TimesNet \
    --csv-out /tmp/timesnet_detox_perseed.csv
```

Prints:
```
seed=100  auroc=0.5542  bal_acc=0.4821  ba_tuned=0.4500  auprc=0.4548
seed=101  auroc=0.4497  bal_acc=0.4237  ba_tuned=0.4400  auprc=0.3501
seed=102  auroc=0.6094  bal_acc=0.5000  ba_tuned=0.5000  auprc=0.5012
 mean: auroc=0.5378  bal_acc=0.4686  ba_tuned=0.4633  auprc=0.4354
  std: auroc=0.0800  bal_acc=0.0391  ba_tuned=0.0321  auprc=0.0761
```

Available metrics per task type:
- **Binary** (n_classes=2): `auroc`, `auprc`, `accuracy`, `balanced_accuracy`,
  `balanced_accuracy_tuned` (using each seed's own val-tuned threshold),
  `f1_macro`, `f1_weighted`, `cohen_kappa`, `auroc_ci_low/high`, `auc_mr`,
  `sensitivity`, `specificity`.
- **3-class (or more)**: `accuracy`, `balanced_accuracy`, `f1_macro`,
  `f1_weighted`, `cohen_kappa`, `auroc_macro` (OvR).

## Merging multiple seed batches (e.g. seed 42-44 + 100-102)

The 6 seeds live in different `OUT_BASE`s. To get a single mean ± std across
all 6 seeds for one (task, arch), aggregate each batch's CSV and concatenate
the per-seed rows, then compute mean ± std across all 6 rows.

```python
import csv, statistics
def load(p):
    with open(p) as f:
        return [r for r in csv.DictReader(f) if r["seed"].isdigit()]
rows = load("/tmp/timesnet_detox_perseed.csv") + load("/tmp/timesnet_detox_seed100.csv")
for k in ("auroc","balanced_accuracy","auprc","f1_macro"):
    vals = [float(r[k]) for r in rows]
    print(k, "%.4f ± %.4f" % (statistics.mean(vals), statistics.stdev(vals)))
```

## Important: use the patched `dl_baseline.py`

If you copy the patch to a fresh checkout, make sure `baseline/dl_baseline.py`
contains the per-seed npz save block (search for `per_seed_preds_dl_`). Without
it, the per-seed aggregation script will fail to find any npz files.
