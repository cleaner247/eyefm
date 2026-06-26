# EyeFM Baselines (`src/baseline/`)

Two baseline families on the v2 finetune eye-movement data, sharing a unified
runner and per-task test csv layout:

| Mode | Methods | Protocol | Output file |
|------|---------|----------|-------------|
| **ML** | Logistic Regression / Random Forest / SVM (Linear, calibrated) | 6-dim paper-features per (subject, saccade_task) row; 30-combo val grid search (LR 8 + RF 16 + SVM 6); best hp per model on val; refit on train; final test | `test_final_ml_<task>.csv` |
| **DL** | TCN / TimesNet / NST / CNNTransformer | Paper-standard hp (LR=3e-4, dropout=0.3, T_LEN=1024, 50 epoch); val-pick best ckpt by val AUROC; final test | `test_final_dl_<task>.csv` |

Combined summary across tasks: `baseline_summary.csv`.

Note on dimensionality: the ML pipeline uses 6 features per (subject,
saccade_task) row. Across the 4 saccade tasks, this gives 24-dim
coverage per subject. Each subject contributes up to 4 rows to the
training matrix; rows are kept independent (a row is its own sample),
matching the reference implementation in the original v4 HP search
experiments.

## Quick start

```bash
# Run both ML and DL baselines on both detox_binary and pd_related_5class
python -m baseline.run_baseline --all \
    --data-root /path/to/eyemae_fast_dataset_v2 \
    --out-dir outputs/baseline_2026-06-26

# ML only
python -m baseline.run_baseline --task detox_binary --skip-dl \
    --data-root /path/to/eyemae_fast_dataset_v2 \
    --out-dir outputs/baseline_ml

# DL only, restrict to NST + CNNTransformer
python -m baseline.run_baseline --task detox_binary --skip-ml \
    --dl-arch NST --dl-arch CNNTransformer \
    --data-root /path/to/eyemae_fast_dataset_v2 \
    --out-dir outputs/baseline_dl_subset
```

## Layout

```
src/baseline/
  __init__.py
  feature_extraction.py   # 6 paper-feat × 4 task = 24-dim subject-level features
  data_loader.py          # v2 packed-mmap loader: TrialDataset + subject-batches + ShardCache
  ml_baseline.py          # LR/RF/SVM 30-combo grid + val pick + refit + test
  dl_baseline.py          # 4 E2Mo-style archs + 50-epoch val-pick
  run_baseline.py         # unified CLI entry
  README.md               # this file
```

Detailed experimental protocol, hp grids, and metric definitions live in
[`docs/eyemae_baseline.md`](../../docs/eyemae_baseline.md).

## Required data layout

The `--data-root` must point at the v2 finetune root, i.e.:

```
<data-root>/
  finetune/
    detox_binary/
      train.csv, validation.csv, test.csv
      subjects.csv
      shards/shard_000000/{X_data.npy, y_frame.npy}
    pd_related_5class/
      ...
    pd_binary/   (optional)
      ...
```

The `X_data.npy` files store per-frame packed features (10 columns: left
xy+area, right xy+area, stim x/y/on, fix-on). The `y_frame.npy` files store
per-frame quality codes (2 columns: left_qc, right_qc).
