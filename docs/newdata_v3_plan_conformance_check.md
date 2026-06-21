# Newdata V3 Plan Conformance Check

This document records the pre-training checks required before launching the v3
new-data pipeline.

## Test Evidence

Full local test suite:

```bash
pytest -q
```

Result:

```text
35 passed in 11.72s
```

## Pretrain V3 Conformance

Plan files checked:

- `docs/pretrain_v3_plan.md`
- `configs/eyemae_cnn_512_12l.yaml`
- `src/eyemae/data.py`
- `src/eyemae/config.py`
- `src/eyemae/compute_area_stats.py`
- `src/eyemae/train.py`
- `src/eyemae/evaluate.py`

Status: ready to launch first-version pretraining.

Evidence:

- `data.format=packed_mmap`.
- `data.data_dir=/mnt/disk_sde/data-260606/extracted/eyemae_fast_dataset_v1`.
- Formal split indexes:
  - `pretrain/pretrain_train.csv`
  - `pretrain/pretrain_validation.csv`
  - `pretrain/pretrain_test.csv`
- `pretrain/pretrain_split_summary.json` and `audit_summary.json` exist.
- Subject overlap audit passed:
  - train/validation: 0
  - train/test: 0
  - validation/test: 0
- Split counts:
  - train: 730,695 trials
  - validation: 41,093 trials
  - test: 40,853 trials
- Maximum required patches: 353.
- Configured `model.max_patches`: 384.
- Area stats generated from train split only:
  - `outputs/area_stats_fast_packed_seed42.json`
  - global median: 8.339022636413574
  - global MAD: 0.22008037567138672

## GPU Pretrain Readiness

Sandbox-external GPU check passed with:

```bash
CUDA_VISIBLE_DEVICES=1,2,3,4
```

Visible to PyTorch:

```text
device_count=4
all four devices are NVIDIA A100-SXM4-80GB
```

Launch target:

```bash
CUDA_VISIBLE_DEVICES=1,2,3,4 torchrun --standalone --nproc_per_node=4 \
  -m eyemae.train --config configs/eyemae_cnn_512_12l.yaml
```

## Downstream V3 Conformance

Status: data/config/code smoke checks passed; fine-tune not launched yet because
pretraining is still running.

Implemented/checked:

- Packed downstream dataset reads `downstream/<view>/{train,validation,test}.csv`.
- Existing downstream split is used directly; no K-fold or re-split is generated.
- `split_summary.json` is checked for `no_subject_overlap == true`.
- Independent `ml_subject_id` overlap check is performed for each task.
- Six binary tasks read `health_label`.
- `pd_related_5class` maps `health_label` + `pd_disease_label` to classes 0..4.
- Subject-balanced sample weights use `ml_subject_id`.
- Binary subject metrics and multiclass subject macro metrics are implemented.
- 28 resolved configs were generated for 7 tasks x 4 modes.

Downstream config count:

```text
28
```

Downstream test evidence:

```text
pytest -q tests/test_downstream.py tests/test_downstream_packed.py
10 passed
```

After the strict binary loss fix:

```text
pytest -q tests/test_downstream.py tests/test_downstream_packed.py
11 passed in 5.90s
```

Full suite after downstream changes:

```text
pytest -q
37 passed in 12.85s
```

Full suite after the strict binary loss fix and monitor-name update:

```text
pytest -q
38 passed in 13.13s
```

Real split smoke checks:

```text
AD train subjects: 0=94, 1=123
PD相关 train subjects: 0=482, 1=202, 2=113, 3=117, 4=80
```

Additional strictness checks:

- Binary loss now uses manual `class_weight_for_label` and does not pass
  `pos_weight=` into `BCEWithLogitsLoss`.
- Generated downstream configs now use plan monitor names:
  `validation/subject_auroc` for binary tasks and
  `validation/subject_macro_auroc_ovr` for the PD 5-class task.
- The trainer resolves these monitor names to the internal validation metric
  keys used by the current logging code.
- A CPU DDP torchrun smoke attempted inside the sandbox failed because local
  TCPStore socket setup is blocked by sandbox policy. This is recorded in
  `docs/newdata_v3_exception_log.md`; actual distributed jobs should run
  outside the sandbox.

Remaining before downstream launch:

- Wait for pretrain checkpoint:
  `outputs/pretrain_v3/eyemae_cnn_512_12l_patch20_stimtoken/checkpoint_best.pt`.
- Run a short downstream smoke command with the actual checkpoint.
- Launch the 7 x 4 downstream jobs after the smoke check passes.
