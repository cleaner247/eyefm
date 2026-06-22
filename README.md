# EyeFM / EyeMAE

This repository contains the current EyeMAE/EyeFM pretraining and downstream
fine-tuning code for 1000 Hz eye-movement trials.

Start from these documents:

```text
docs/pretrain_v3_plan.md
docs/downstream_v3_plan.md
docs/eyemae_fast_dataset_v2_current.md
docs/eyemae_fast_dataset_v2_report.md
```

## Current Dataset

The real dataset is not stored in Git. The current accepted local dataset is:

```text
/mnt/disk_sde/data-260606/extracted/eyemae_fast_dataset_v2
```

It is a packed-mmap dataset, not one `.npz` per trial. Current roots are:

```text
pretrain:
/mnt/disk_sde/data-260606/extracted/eyemae_fast_dataset_v2/pretrain

downstream:
/mnt/disk_sde/data-260606/extracted/eyemae_fast_dataset_v2/finetune/<task>
```

The current downstream task names are:

```text
pd_related_5class
pd_binary
epilepsy_binary
detox_binary
migraine_binary
ad_binary
mci_binary
mci_matched_binary
```

## Setup

```bash
cd eyefm
pip install -e .
```

Run tests:

```bash
pytest -q
```

## Pretraining

The current v2 pretraining config is:

```text
configs/v2_corrected_max30_oldpretrain/eyemae_cnn_512_12l_v2_clean.yaml
```

Before formal v2 pretraining, compute v2 area stats:

```bash
python -m eyemae.compute_area_stats \
  --config configs/v2_corrected_max30_oldpretrain/eyemae_cnn_512_12l_v2_clean.yaml \
  --split pretrain_train \
  --out outputs/area_stats_fast_packed_v2_clean_full_subject_seed20260622.json
```

Run v2 pretraining:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
  -m eyemae.train \
  --config configs/v2_corrected_max30_oldpretrain/eyemae_cnn_512_12l_v2_clean.yaml
```

Evaluate:

```bash
python -m eyemae.evaluate \
  --config configs/v2_corrected_max30_oldpretrain/eyemae_cnn_512_12l_v2_clean.yaml \
  --checkpoint outputs/pretrain_v2_clean/eyemae_cnn_512_12l_patch20_stimtoken/checkpoint_best.pt \
  --split pretrain_test
```

## Downstream Fine-Tuning

The current downstream queue is:

```text
configs/v2_corrected_max30_oldpretrain/queue.txt
```

It runs 8 downstream tasks times 4 modes:

```text
scratch
linear_probe
partial
full
```

Run the queue on four GPUs:

```bash
python scripts/run_downstream_v3_queue.py \
  --gpus 1,2,3,4 \
  --config-list-file configs/v2_corrected_max30_oldpretrain/queue.txt \
  --log-dir outputs/downstream_v2_corrected_max30_oldpretrain_logs/run_manual
```

The active corrected-v2 downstream run intentionally uses the v2 downstream
dataset with the existing v3 pretrained checkpoint and old v3 area stats. This
isolates the effect of data cleaning and split correction. A formal v2
pretrain-to-finetune result requires recomputing v2 area stats, rerunning v2
pretraining, then switching downstream configs to the v2 checkpoint and v2 area
stats.

Large training outputs and checkpoints are intentionally not tracked by Git.
