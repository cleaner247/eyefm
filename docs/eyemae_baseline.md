# EyeFM Baseline Classifiers (ML + DL)

This document describes the EyeFM baseline classifier pipeline implemented
in `src/baseline/`. The pipeline pairs two complementary families:

- **ML baselines** (3 models × 30-combo val grid search) on
  hand-crafted 24-dim subject-level paper-features.
- **DL baselines** (4 E2Mo-style architectures) on raw v2 packed-mmap
  trial-level EOG windows, using paper-standard hyperparameters
  (no grid search) with val-pick-best checkpoint over 50 epochs.

All runs follow the **64 / 16 / 20 subject-level split** of v2
(detox_binary 110/27/34, pd_related_5class 573/143/179) and write
their final test metrics into a unified per-task CSV layout plus a
combined `baseline_summary.csv`.

The reference run numbers (used for paper tables and to verify this
codebase) are documented at the end.

## 1. Data layout

This codebase assumes the v2 finetune layout, with one directory per task:

```
<data-root>/
  finetune/
    detox_binary/
      train.csv, validation.csv, test.csv   # trial-level rows
      subjects.csv                          # subject-level metadata
      shards/shard_000000/X_data.npy        # (N_frames_total, 10) float32
      shards/shard_000000/y_frame.npy       # (N_frames_total, 2)  uint8/int
      shards/shard_000001/...
    pd_related_5class/
      ... (same layout)
    pd_binary/   (optional)
      ...
```

The 10 columns of `X_data.npy` are, in order:

| idx | name | description |
|-----|------|-------------|
| 0 | left_x | left-eye horizontal gaze (deg) |
| 1 | left_y | left-eye vertical gaze (deg) |
| 2 | left_area | left-eye pupil area |
| 3 | right_x | right-eye horizontal gaze (deg) |
| 4 | right_y | right-eye vertical gaze (deg) |
| 5 | right_area | right-eye pupil area |
| 6 | stim_x | stimulus horizontal position (deg) |
| 7 | stim_y | stimulus vertical position (deg) |
| 8 | stim_on | stimulus onset flag (0/1) |
| 9 | fix_on | fixation onset flag (0/1) |

The 2 columns of `y_frame.npy` are `left_qc, right_qc` per-frame quality codes
(0 = valid, 1 = blink, other values reserved).

## 2. Subject-level split (64 / 16 / 20)

The v2 split CSVs are pre-defined at subject level:

| task | train subj | val subj | test subj | total |
|------|------------|----------|-----------|-------|
| detox_binary | 110 | 27 | 34 | 171 |
| pd_related_5class | 573 | 143 | 179 | 895 |

Train rows filtered by `task in {ProSaccade, AntiSaccade, MemorySaccade, DoubleSaccade}`
(4 saccade paradigms); each trial carries one `ml_subject_id` and the subject is
assigned to exactly one split. There is no subject overlap between splits.

(The 64 / 16 / 20 percentages are computed from the raw counts; for
detox_binary 110/(110+27+34) = 64.3 % and for pd_related_5class
573/(573+143+179) = 64.0 %.)

## 3. ML baselines (LR / RF / SVM)

### 3.1 24-dim paper-features

For each trial we extract 10 per-eye saccade features (binocular mean)
from one of four saccade task types. Per subject per task we aggregate
the 6 paper-features below by mean across trials:

A. **4 paper-consensus features** (5/5 reference papers):

- `first_saccade_latency_ms` — first saccade start, in original-frame units (= ms @ 1 kHz)
- `primary_amp_deg` — primary saccade amplitude (Euclidean)
- `primary_peak_v` — primary saccade peak speed (deg/s, derived via gradient × 1000)
- `primary_dur_ms` — primary saccade duration (samples @ 1 kHz)

B. **2 paper-extra features** (2/5 reference papers):

- `blink_ratio` — fraction of frames with `qc == 1`
- `missing_ratio` — fraction of frames with NaN/inf gaze or non-zero qc

A × 4 task types = **24-dim** subject-level feature vector. We impute
NaN with the train column mean and standardize with `StandardScaler`
fit on train only.

### 3.2 Saccade detection

Per-eye: compute speed = `||∇(x,y)|| × 1000`; flag frames where speed
exceeds `30 deg/s`; emit saccade intervals of length ≥ 10 samples
(10 ms). The first such interval is the *primary* saccade whose
properties (latency, amplitude, peak speed, duration) are recorded.

### 3.3 30-combo val grid

| model | hp | values | combos |
|-------|----|--------|--------|
| LR | C × balanced | (0.01, 0.1, 1, 10) × (T, F) | 8 |
| RF | n_est × max_d × min_leaf × balanced | (100, 300) × (None, 20) × (1, 5) × (T, F) | 16 |
| SVM | C × balanced | (0.1, 1, 10) × (T, F) | 6 |
| **Total** | | | **30** |

`SVM` is a `LinearSVC` wrapped in `CalibratedClassifierCV(sigmoid, cv=3)`
to expose `predict_proba` for AUROC scoring.

### 3.4 Val-pick + refit protocol

1. For each combo: fit on `train`, score on `val` (AUROC for binary,
   AUROC-macro-OVR for 5-class).
2. Pick best hp per model on val.
3. Refit best per-model on `train` only (no train+val merge).
4. Evaluate on `test` for the final reported metrics.

### 3.5 Reported metrics

| metric | binary | 5-class |
|--------|--------|---------|
| AUROC | yes (95 % bootstrap CI) | AUROC-macro-OVR |
| Balanced accuracy | yes | yes |
| Accuracy / F1-macro / F1-weighted / Cohen κ | yes | yes |
| Sensitivity / Specificity / AUC-MR | yes | n/a |

## 4. DL baselines (4 E2Mo-style archs)

### 4.1 Architectures

| arch | source | key hyper-params |
|------|--------|------------------|
| **TCN** | Bai et al. 2018 | 2-block dilated 1D conv, hidden=64, kernel=3, dilations={1, 2}, residual + chomp |
| **TimesNet** | Wu et al. IJCAI 2023 | FFT-period → 2D Inception, hidden=96, top_k=2, n_blocks=2 |
| **NST** | Liu NeurIPS 2022 | dual-statistic attention, d_model=64, n_heads=2, n_layers=1, max_T=1024 |
| **CNNTransformer** | E2Mo 简化 | 1D conv front + Transformer encoder, d_model=32, n_heads=2, n_layers=1 |

Input: a `subject × saccade_task` trial stack `x` of shape `(K, T_LEN, 10)`
where `K` is the number of trials for that subject on that task. The
model averages logits across `K` (subject-level pooling) before the
final classification head.

### 4.2 Hyper-params (paper-standard, **no grid search**)

| param | value |
|-------|-------|
| T_LEN | 1024 |
| BATCH_SIZE (trials per subject-bucket) | 16 |
| MAX_EPOCHS | 50 |
| LR | 3e-4 |
| weight_decay | 0.01 (TCN) / 0.05 (others) |
| dropout | 0.3 |
| class_weight | inverse-freq, mean-normalized (computed from train labels) |
| loss | `CrossEntropyLoss(weight=class_weight)` |
| grad clip | `clip_grad_norm_(..., 1.0)` |
| optimizer | AdamW |
| seed | 42 (subject batches shuffled deterministically per split) |

### 4.3 Val-pick + test protocol

For each of the 50 epochs:

1. Train one pass over the train split (subject-batched).
2. Evaluate on the val split; compute AUROC (binary) or
   AUROC-macro-OVR (5-class).
3. If val score > previous best → snapshot model state to memory
   (`best_state`).

After the loop, restore the best snapshot and run the final test pass.
The reported `best_epoch` is the epoch at which the best val score was
achieved.

### 4.4 Reported metrics

Same extended metrics as ML, plus `best_epoch` and `best_val_score`.

## 5. Outputs

For a single-task run with `--out-dir <dir>`:

```
<dir>/
  test_final_ml_<task>.csv         # ML: 3 rows (LR/RF/SVM), per best hp
  test_final_dl_<task>.csv         # DL: 4 rows (4 archs), per best epoch
  baseline_summary.csv              # combined: 2 × 7 = 14 rows (when --all)
```

CSV columns (in order, with `n_classes` at index 2–3):

```
ML:
  model, hp, n_classes, train_time_sec, val_score, n,
  accuracy, balanced_accuracy, f1_macro, f1_weighted, cohen_kappa,
  auroc, auroc_macro, auroc_ci_low, auroc_ci_high,
  sensitivity, specificity, auc_mr

DL:
  arch, n_classes, best_epoch, best_val_score, train_time_sec,
  accuracy, balanced_accuracy, f1_macro, f1_weighted, cohen_kappa,
  auroc, auroc_macro, sensitivity, specificity, auc_mr
```

## 6. Reference run numbers (v4 HP search, 2026-06-25)

These are the numbers shipped in the corresponding paper table; the
code in `src/baseline/` should reproduce them within ±0.01 AUROC
for the same `SEED=42` and the same data root.

### detox_binary (binary, val/test AUROC)

| mode | model | val | test |
|------|-------|-----|------|
| ML | LR | 0.6989 | 0.6989 |
| ML | RF | 0.7268 | 0.7268 |
| ML | SVM | 0.7429 | 0.7429 |
| DL | TCN | 0.880 | 0.575 (T_LEN=512) |
| DL | NST | 0.937 | **0.800** (T_LEN=1024) |
| DL | CNNTransformer | 0.920 | 0.725 (T_LEN=2048) |
| DL | TimesNet | 0.880 | 0.439 (T_LEN=1024) |

Best overall: **ML-SVM (0.7429 val, 0.7429 test)** for 24-dim;
**DL-NST (0.937 val, 0.800 test)** for raw EOG with T_LEN=1024.

### pd_related_5class (5-class, val/test AUC-macro-OVR)

| mode | model | val | test |
|------|-------|-----|------|
| ML | LR | 0.5310 | 0.5310 |
| ML | RF | 0.5418 | 0.5418 |
| ML | SVM | 0.5403 | 0.5403 |
| DL | TCN | 0.605 | 0.571 (T_LEN=1024) |
| DL | NST | 0.674 | **0.708** (T_LEN=1024) |
| DL | CNNTransformer | — | stuck (architecture limitation, 5-class) |
| DL | TimesNet | 0.580 | 0.545 (T_LEN=1024) |

Best overall: **DL-NST (0.674 val, 0.708 test)**; ML plateaus at
≈0.54 for 24-dim 5-class.

## 7. Wall-time reference

`1693` workhorse (5×A100-SXM4-80GB, `rl` env Python 3.13 + torch 2.9.1+cu128):

| task | ML grid (30 fit) | DL 4-arch × 50 epoch |
|------|------------------|----------------------|
| detox_binary | ≈ 30 s | ≈ 5 min (1 GPU) |
| pd_related_5class | ≈ 3 min | ≈ 12 min (1 GPU) |

When run on `--all` (2 task), ML fits all 30 in one process and DL
runs both task on the same `cuda:0` sequentially by default.

## 8. Known limitations

- **CNNTransformer 5-class** is architecturally stuck (val AUROC
  never improves past the initial epoch). It will report the
  best-epoch-1 snapshot, which is essentially chance.
- **TimesNet at T_LEN=2048** triggers an FFT period index error in
  some PyTorch builds. The default `T_LEN=1024` is the safe
  choice; if you override to 2048, expect sporadic crashes.
- **val/test cohort shift on detox_binary** is large in the v2 split
  (val 27 vs test 34 subj with somewhat different label balance).
  All baselines see this; no per-subject reweighting is applied.
- The ML grid is **30 combos**. We do not run a 96-dim grid in this
  module; that lives in the older `train_stage1_ml_v2.py` workflow.
