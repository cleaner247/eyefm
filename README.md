# EyeFM / EyeMAE

This repository contains the current EyeMAE first-version pretraining project for 1000 Hz eye-movement trials. It is the implementation following:

```text
docs/eyemae_codex_plan_engineered_clean.md
```

Concrete pretraining/downstream run versions are tracked in:

```text
docs/eyemae_experiment_registry.md
```

The current main experiment is a task-conditioned masked reconstruction model:

- per-trial 1000 Hz eye movement data
- 20 ms non-overlapping patches
- token order `S_i, L_i, R_i` for stimulus/task/time, left-eye, right-eye
- CNN tokenizer
- bidirectional Transformer encoder
- `d_model=512`, `n_layers=12`, `n_heads=8`
- no CLS, no goal, no last-stim features
- reconstruct masked `x, y, pupil area, blink`
- velocity loss enabled

## Repository Contents

Tracked files include source code, configs, scripts, tests, subject-heldout split files, and area statistics needed by the current config.

Large training outputs are intentionally not tracked:

- checkpoints under `outputs/`
- TensorBoard/event logs
- visualization PNGs
- archived intermediate outputs

The current production config references:

```text
configs/eyemae_cnn_512_12l.yaml
splits/pretrain_subject_heldout_seed42/
outputs/area_stats_subject_heldout_seed42.json
```

The real dataset itself is not included. The config currently expects the data root:

```text
/mnt/disk_sde/data-260606/extracted/cd_no_cond2_structured_20260609
```

If your machine stores the same data elsewhere, update only `data.data_dir` in the config while keeping the tracked split files and area stats unchanged for the same experiment.

## Setup

```bash
cd eyefm
pip install -e .
```

The project avoids pandas/sklearn dependencies. Main runtime dependencies are in `requirements.txt`.

## Debug / Test Workflow

Generate synthetic debug data and run tests:

```bash
python tests/fixtures/make_synthetic_npz.py --out_dir tests/fixtures/synthetic_npz --num_trials 128
pytest -q
```

Run the small debug training config:

```bash
python -m eyemae.make_splits --config configs/debug.yaml
python -m eyemae.compute_area_stats --config configs/debug.yaml --split pretrain_train --out outputs/debug_area_stats.json
python -m eyemae.train --config configs/debug.yaml
```

## Main Experiment

The main split and area statistics are already tracked. To reproduce the current pretraining run on the real data:

```bash
CUDA_VISIBLE_DEVICES=0,1,2 torchrun --standalone --nproc_per_node=3 \
  -m eyemae.train \
  --config configs/eyemae_cnn_512_12l.yaml
```

Evaluate a trained checkpoint:

```bash
python -m eyemae.evaluate \
  --config configs/eyemae_cnn_512_12l.yaml \
  --checkpoint outputs/eyemae_cnn_512_12l_patch20_stimtoken/checkpoint_best.pt \
  --split pretrain_test
```

Evaluation writes:

```text
outputs/eyemae_cnn_512_12l_patch20_stimtoken/evaluation/pretrain_test/metrics.json
outputs/eyemae_cnn_512_12l_patch20_stimtoken/evaluation/pretrain_test/metrics_by_group.json
outputs/eyemae_cnn_512_12l_patch20_stimtoken/evaluation/pretrain_test/baselines.json
```

## Current Reference Results

For the local best checkpoint from the current run:

```text
pretrain_test/masked_xy_rmse_deg = 0.468622
pretrain_test/total_loss = 0.00124888
pretrain_test/masked_blink_auc = 0.995496
previous_value baseline masked_xy_rmse_deg = 2.013649
linear_interpolation baseline masked_xy_rmse_deg = 2.739195
long_span model masked_xy_rmse_deg = 1.510749
long_span linear_interpolation baseline masked_xy_rmse_deg = 4.216968
```

The checkpoint files are not in Git. Use a release/artifact store for weights if needed.

## Downstream Fine-Tuning

The downstream disease-classification plan is in:

```text
docs/eyemae_downstream_finetune_codex_plan.md
```

Completed and active downstream run versions, including the fixed holdout run
and the PD/addiction 5-fold run, are tracked in:

```text
docs/eyemae_experiment_registry.md
```
