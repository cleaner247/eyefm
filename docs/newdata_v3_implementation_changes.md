# Newdata V3 Implementation Changes

This document records code/config changes made to run `docs/pretrain_v3_plan.md`
and `docs/downstream_v3_plan.md` on `eyemae_fast_dataset_v1`.

## Current Scope

Implemented in this checkpoint:

- Added packed pretraining data access in `src/eyemae/data.py`.
- Updated config validation in `src/eyemae/config.py` to allow `data.format=packed_mmap`.
- Updated pretraining train/evaluate dataset construction to use `data.train_index`,
  `data.val_index`, and `data.test_index` for packed data.
- Updated area statistics in `src/eyemae/compute_area_stats.py` to read packed
  mmap shards through CSV indexes.
- Updated `configs/eyemae_cnn_512_12l.yaml` to point to:
  `/mnt/disk_sde/data-260606/extracted/eyemae_fast_dataset_v1`.
- Updated the formal pretrain output directory to:
  `outputs/pretrain_v3/eyemae_cnn_512_12l_patch20_stimtoken`.

## Pretrain V3 Details

Architecture/training-parameter parity with the previous `main` pretraining
config:

- Kept the same model architecture: 512 hidden size, 12 layers, 8 heads,
  `ffn_hidden=1536`, CNN tokenizer, 20 ms patches, `max_patches=384`,
  stim-eye triplet sequence, no CLS token, token type embeddings, RMSNorm, and
  SwiGLU.
- Kept the same mask, loss, optimizer, learning rate schedule, DDP, dynamic
  token batch, validation interval, checkpoint interval, and checkpoint monitor.
- Changed the data interface from per-trial NPZ split files to packed mmap CSV
  indexes.
- Changed pretraining split handling from locally generated txt files to the
  provided subject-heldout CSV files and split summary audit.
- Changed area-stat output path to the new packed-data file.
- Non-reader difference to keep in mind: `area.max_frames_per_subject` was
  reduced from `200000` to `5000` for faster area-stat computation on the packed
  dataset. This affects area normalization statistics but does not change the
  network architecture or optimizer/training schedule.

Packed dataset reader:

- `PackedTrialStore` lazily opens shard mmap arrays.
- `PackedPretrainDataset` reads `pretrain/*.csv` index rows.
- The authoritative trial slice is `frame_offset` plus `frame_length` from CSV.
- `X_offsets.npy` and `X_lengths.npy` are used as consistency checks.
- The internal trial dict maps packed columns as required by `pretrain_v3_plan.md`.

Split/audit behavior:

- Formal split files are not regenerated.
- `pretrain/pretrain_train.csv`, `pretrain/pretrain_validation.csv`, and
  `pretrain/pretrain_test.csv` are used directly.
- `audit_packed_pretrain_splits()` checks required audit files, subject overlap,
  duplicate `global_trial_id`, valid `task_id`, and `model.max_patches`.

Area stats:

- Generated file: `outputs/area_stats_fast_packed_seed42.json`.
- Source split: `pretrain/pretrain_train.csv`.
- The config uses per-subject sampling via `area.max_frames_per_subject=5000`
  and global reservoir size `area.max_global_frames=2000000`.

## Downstream V3 Minimal Adaptation

Implemented as a minimal compatibility branch without removing the old NPZ
downstream path:

- Added `PackedDownstreamDataset` in `src/eyemae/downstream_data.py`.
- Added binary label loading from `health_label`.
- Added `pd_related_5class` label mapping from `health_label` and
  `pd_disease_label`.
- Added packed downstream split audit in `src/eyemae/finetune.py`.
- Added multiclass classifier head support in `DownstreamClassifier`.
- Added weighted multiclass cross entropy and subject-class weights.
- Added multiclass subject aggregation and macro OVR metrics.
- Added subject/trial `weighted_f1` and `cohen_kappa` metrics for both binary
  and multiclass downstream tasks. `weighted_f1` is class-F1 weighted by true
  class support; `cohen_kappa` is computed from the same argmax/thresholded
  confusion matrix used for accuracy.
- Updated `evaluate_downstream.py` to reuse the same packed/multiclass logic.
- Generated resolved configs under `configs/downstream/` for the downstream
  first-version task/mode matrix.
- Added `tests/test_downstream_packed.py`.
- Updated binary downstream loss to follow `docs/downstream_v3_plan.md`: the
  train-subject positive class ratio is applied manually as
  `class_weight_for_label`, and is not passed as `pos_weight=` to
  `BCEWithLogitsLoss`.
- Updated downstream checkpoint monitors in generated configs to use the plan
  names `validation/subject_auroc` and
  `validation/subject_macro_auroc_ovr`; the trainer resolves these aliases to
  the internal validation metric keys.

Formal v3 downstream tasks:

- `pd_related_5class`
- `pd_binary`
- `epilepsy_binary`
- `detox_binary`
- `migraine_binary`
- `ad_binary`
- `mci_binary`
- `mci_matched_binary`

Modes:

- `linear_probe`
- `partial`
- `full`
- `scratch`

## Test Evidence

Full test suite after pretrain and downstream minimal changes:

```text
37 passed in 12.85s
```

Full test suite after the binary loss strictness fix and monitor-name update:

```text
38 passed in 13.13s
```

## Pretrain Speed Variant Prepared

Prepared but not launched yet:

- Cache-only config for first causal short test:
  `configs/eyemae_cnn_512_12l_speed_cache44.yaml`.
  - Output directory:
    `outputs/pretrain_v3_speed_cache44/eyemae_cnn_512_12l_patch20_stimtoken`.
  - Keeps all baseline settings except
    `data.max_open_shards_per_worker: 16 -> 44`.
  - This is the preferred first speed test because it isolates the shard-cache
    hypothesis.
- Config: `configs/eyemae_cnn_512_12l_speed_tokens90k_prefetch4.yaml`.
- Output directory:
  `outputs/pretrain_v3_speed_tokens90k_prefetch4/eyemae_cnn_512_12l_patch20_stimtoken`.
- Keeps the same network architecture, mask, loss, optimizer, LR schedule, and
  formal packed dataset split.
- Changes only speed-related input pipeline / batch settings:
  - `train.max_seq_tokens_per_gpu: 60000 -> 90000`
  - `train.prefetch_factor: 4`
  - `data.max_open_shards_per_worker: 16 -> 44`
  - `data.validate_offsets: true -> false`
- Added optional DataLoader `prefetch_factor` support in `src/eyemae/train.py`.
  Existing configs are unchanged unless they explicitly set this field.
- Added optional pretrain timing and validation-overhead controls in
  `src/eyemae/train.py`:
  - `train.timing_every_steps: 0` disables timing by default; a positive value
    logs `data_wait`, `to_device`, `mask`, `forward_loss`, `backward`,
    `optimizer`, `validation`, and `checkpoint` timings at that interval.
  - `eval.group_metrics_every_steps: null` preserves first-version behavior by
    computing group metrics at every validation; `0` disables them and a
    positive value computes them at that step interval.
  - `eval.visualization_every_steps: null` preserves first-version behavior by
    saving validation visualizations at every validation; `0` disables them and
    a positive value saves them at that step interval.
  - The active pretrain process was already running when these code/config
    fields were added, so this change does not affect the current baseline
    unless it is restarted from the edited config.
- Updated the formal downstream resolved configs so pretrained modes load the
  completed new-data pretrain checkpoint:
  `outputs/pretrain_v3_fast_cache44_nooffset/eyemae_cnn_512_12l_patch20_stimtoken/checkpoint_best.pt`.
  Their `pretrained.pretrain_config` now points to
  `configs/eyemae_cnn_512_12l_fast_cache44_nooffset.yaml`.
- Added `scripts/run_downstream_v3_queue.py` to run the formal downstream
  matrix with one job per GPU and per-job logs/status.
- Fixed downstream monitor logging so aliases such as
  `validation/subject_macro_auroc_ovr -> val/subject/macro_auroc_ovr` do not
  emit a false fallback warning. The fallback warning is now emitted only when
  the resolved monitor is actually NaN and the code falls back to a balanced
  accuracy metric.
- Added downstream DataLoader `prefetch_factor` support in
  `src/eyemae/finetune.py` and set the formal downstream configs to
  `prefetch_factor: 4`.
- Updated the formal downstream configs to use the packed-data speed settings
  already selected for pretrain: `data.max_open_shards_per_worker: 44` and
  `data.validate_offsets: false`. These settings reduce shard LRU churn and
  skip repeated offset consistency checks after the dataset audit has passed;
  they do not change sample contents, labels, model architecture, optimizer, or
  loss.
- Updated `scripts/run_downstream_v3_queue.py` so future queue runs refresh
  `status.json` immediately after launching a job, not only after a job
  completes.
- Added downstream queue resume/attach support to
  `scripts/run_downstream_v3_queue.py`:
  - `--resume-status` imports completed jobs from an existing status file.
  - `--attach gpu:pid:config:log` monitors an already-running job instead of
    relaunching it.
  - The supervisor filters out configs that already have `metrics_final.json`,
    so it can recover from a lost queue process without rerunning completed
    jobs.
- Updated `docs/downstream_v3_plan.md` for future accelerated fine-tune runs:
  `early_stopping_patience: 20 -> 10` and
  `min_epochs_before_early_stopping: 50 -> 0`. The already-launched
  downstream v3 baseline queue is intentionally left on its process-start
  config, so its remaining jobs should still be interpreted as the current
  baseline rather than the accelerated early-stopping variant.
- Added `scripts/prepare_pd_random_finetune.py` to create a reproducible
  subject-level stratified random PD 5-class split and matching fast fine-tune
  configs. The first generated split is
  `downstream/PD相关_random_seed20260620/`, seeded with `20260620`, preserving
  the formal PD subject split proportions while preventing subject overlap.
- Added explicit config-list support to `scripts/run_downstream_v3_queue.py`:
  - `--configs ...` runs a passed config list instead of the default matrix.
  - `--config-list-file <path>` reads a newline-delimited list, ignoring blank
    lines and `#` comments.
- Added `configs/downstream/queue_pd_random_seed20260620_fast.txt` for the
  current full downstream run. It includes PD random-split 5-class configs
  first, then PD binary configs, then the six remaining formal downstream
  tasks, each with `scratch`, `linear_probe`, `partial`, and `full`.
- Updated the current downstream fast-run configs so every queued job writes to
  `outputs/downstream_v3_fast/` and uses
  `early_stopping_patience=10`, `min_epochs_before_early_stopping=0`. This
  avoids mixing new fast-run outputs with interrupted baseline outputs under
  `outputs/downstream_v3/`.
- Added `scripts/prepare_pd_binary_finetune.py` to create a PD binary
  downstream view from `downstream/PD相关_random_seed20260620/`. It reuses the
  same subject-level random split and changes only the task semantics:
  `health_label=1` merges all four PD-related subtypes
  (`pd_disease_label=0..3`) into the positive class. The generated view is
  `downstream/PD相关_binary_random_seed20260620/`, and the generated configs are
  `configs/downstream/pd_binary_random_seed20260620_fast_{scratch,linear_probe,partial,full}.yaml`.
- Expanded the active downstream config list from 28 to 32 jobs and updated
  `scripts/audit_downstream_v3_goal.py` so the final hard audit expects
  8 tasks x 4 modes, including `pd_binary`.
- Added `scripts/wait_and_run_downstream_v3_queue.py` for this mid-run scope
  extension. It waits until the already-started queue reports no running or
  pending jobs, then starts `scripts/run_downstream_v3_queue.py` with the
  expanded 32-job config list and `--resume-status` so only missing jobs, such
  as the newly added PD binary modes, are launched.

Speed-variant test evidence:

```text
pytest -q tests/test_batching.py tests/test_downstream.py tests/test_downstream_packed.py
15 passed in 5.82s
```

Downstream speed-setting test evidence:

```text
python -m py_compile src/eyemae/finetune.py scripts/run_downstream_v3_queue.py
pytest -q tests/test_downstream.py tests/test_downstream_packed.py
11 passed in 5.89s
```

Downstream queue recovery test evidence:

```text
python -m py_compile scripts/run_downstream_v3_queue.py
python scripts/run_downstream_v3_queue.py --dry-run --gpus 1,2,3,4
pytest -q tests/test_downstream.py tests/test_downstream_packed.py
11 passed in 6.79s
```

PD random-split fast fine-tune test evidence:

```text
python -m py_compile scripts/prepare_pd_random_finetune.py scripts/run_downstream_v3_queue.py
python scripts/run_downstream_v3_queue.py --dry-run --gpus 1,2,3,4 --config-list-file configs/downstream/queue_pd_random_seed20260620_fast.txt
pytest -q tests/test_downstream.py tests/test_downstream_packed.py
11 passed in 5.54s
```

Active goal guardrail:

- Added `docs/current_goal_guardrail.md` as the durable completion contract for
  the active downstream v3 fast fine-tune goal.
- The goal must not be marked complete until the guardrail checklist passes:
  8 tasks x 4 modes complete, all 32 final metrics under
  `outputs/downstream_v3_fast/`, PD 5-class using
  `PD相关_random_seed20260620`, PD binary using
  `PD相关_binary_random_seed20260620`, other tasks using formal splits,
  subject-level validation-selected best checkpoints, and
  train/validation/test final metrics recorded.
- Tightened the guardrail after the goal update on 2026-06-20: final completion
  must also verify the first-version tests, output artifacts, and acceptance
  metrics/result table from `docs/downstream_v3_plan.md` Sections 21-24.

Downstream first-version output compatibility:

- Updated `src/eyemae/finetune.py` so newly launched downstream jobs also write
  `resolved_config.yaml`, per-split metric files such as
  `validation_metrics.json` and `test_metrics.json`, and validation alias
  prediction/confusion files required by `docs/downstream_v3_plan.md` Section
  23.
- Extended `run_summary.json` for newly launched downstream jobs with the
  Section 23 fields: `task_name`, `mode`, nested `pretraining_exposure`,
  `num_train_subjects`, `num_validation_subjects`, `num_test_subjects`,
  `label_counts`, and `subject_eye_availability_counts`.
- Updated `src/eyemae/evaluate_downstream.py` to emit the same per-split metric
  files and validation aliases when running explicit evaluation.
- Existing legacy files remain unchanged (`config.json`, `metrics_final.json`,
  `trial_predictions_val.csv`, etc.) to preserve current scripts/tests.
- The four PD random-split jobs already running at the time of this code change
  loaded the older Python code. Their completed output directories must be
  audited and, if needed, materialized to the Section 23-compatible filenames
  before final goal completion.
- `scripts/audit_downstream_v3_goal.py --materialize-plan-artifacts` also
  materializes missing Section 23 `run_summary.json` fields for already-running
  legacy outputs from the config and downstream index files.
- The same materialization step now backfills missing `weighted_f1` and
  `cohen_kappa` values in legacy `metrics_final.json` files from the saved
  train/validation/test trial and subject prediction CSVs, so jobs launched
  before this metric addition can be reported with the same final schema.

Output compatibility test evidence:

```text
python -m py_compile src/eyemae/finetune.py src/eyemae/evaluate_downstream.py
pytest -q tests/test_downstream.py::test_train_downstream_tiny_smoke
1 passed in 6.39s
python -m py_compile src/eyemae/finetune.py scripts/audit_downstream_v3_goal.py scripts/summarize_downstream_v3_results.py
python scripts/audit_downstream_v3_goal.py --allow-incomplete --report-json outputs/downstream_v3_fast_logs/run_20260620_random_pd_fast/goal_audit_latest.json
pytest -q tests/test_downstream.py::test_train_downstream_tiny_smoke
1 passed in 6.02s
```

Metric addition test evidence:

```text
python -m py_compile src/eyemae/downstream_metrics.py scripts/audit_downstream_v3_goal.py scripts/summarize_downstream_v3_results.py
pytest -q tests/test_downstream.py::test_downstream_metrics_and_subject_aggregation tests/test_downstream.py::test_downstream_multiclass_weighted_f1_and_kappa
2 passed in 1.69s
pytest -q tests/test_downstream.py tests/test_downstream_packed.py
12 passed in 5.35s
python scripts/audit_downstream_v3_goal.py --allow-incomplete --materialize-plan-artifacts --report-json outputs/downstream_v3_fast_logs/run_20260620_random_pd_fast/goal_audit_latest.json
python scripts/summarize_downstream_v3_results.py
```

Goal audit tooling:

- Added `scripts/audit_downstream_v3_goal.py` to make the final guardrail
  executable. It checks the 32-config task/mode matrix, output roots, fast
  early-stopping settings, PD random split usage, non-PD formal split usage,
  subject overlap, final metrics, Section 23 output artifacts, and queue status.
- The script supports `--allow-incomplete` for monitoring while the queue is
  still running and `--materialize-plan-artifacts` to create compatibility
  artifacts such as `resolved_config.yaml`, `validation_metrics.json`, and
  validation alias files from legacy completed outputs when needed.

Goal audit smoke evidence:

```text
python -m py_compile scripts/audit_downstream_v3_goal.py
python scripts/audit_downstream_v3_goal.py --allow-incomplete \
  --report-json outputs/downstream_v3_fast_logs/run_20260620_random_pd_fast/goal_audit_latest.json
pytest -q tests/test_downstream.py tests/test_downstream_packed.py tests/test_fast_packed_dataset.py
12 passed in 5.95s
```

First-version result table tooling:

- Added `scripts/summarize_downstream_v3_results.py` to generate the Section 24
  result table from the 32-job config list and each job's `metrics_final.json`.
- Outputs:
  - `outputs/downstream_v3_fast/summary_first_version.csv`
  - `outputs/downstream_v3_fast/summary_first_version.json`
  - `outputs/downstream_v3_fast/summary_first_version.md`
- The table records task, model/mode, pretrain exposure, encoder
  initialization, fine-tune style, subject AUROC or macro AUROC, subject AUPRC
  or macro AUPRC, balanced accuracy, F1, weighted F1, and Cohen's Kappa.
- Use `--require-complete` after all jobs finish to make missing
  `metrics_final.json` files a hard failure.

First-version result table smoke evidence:

```text
python -m py_compile scripts/summarize_downstream_v3_results.py
python scripts/summarize_downstream_v3_results.py
```

The smoke run should be repeated after the PD binary addition; the expected row
count is now 32.

Current-best PD test rerun tooling:

- Added `scripts/rerun_pd_current_best_test.py` for ad-hoc test evaluation of
  the four PD 5-class current `checkpoint_best.pt` files without touching the
  active training queue.
- The runner writes per-mode `test_metrics.json`, prediction CSVs,
  confusion matrices, and `rerun_info.json` under
  `outputs/downstream_v3_fast_current_best_test_eval/pd_related_5class_random_seed20260620/`.
- It overrides the one-off evaluation loader to `num_workers=0` by default so
  the command is predictable in the managed execution environment.

Current-best PD test rerun evidence:

```text
python -m py_compile scripts/rerun_pd_current_best_test.py
env CUDA_VISIBLE_DEVICES=0 python scripts/rerun_pd_current_best_test.py \
  --modes scratch --device cuda:0 --batch-size 64 --num-workers 0
env CUDA_VISIBLE_DEVICES=0 python scripts/rerun_pd_current_best_test.py \
  --modes linear_probe partial full --device cuda:0 --batch-size 64 --num-workers 0
```

Scratch max-epoch cap:

- On 2026-06-21, changed the active v3 fast scratch configs to
  `downstream_train.max_epochs: 30` to cap from-scratch downstream run time:
  `pd_related_5class_random_seed20260620_fast_scratch.yaml`,
  `pd_binary_random_seed20260620_fast_scratch.yaml`,
  `epilepsy_binary_scratch.yaml`, `detox_binary_scratch.yaml`,
  `migraine_binary_scratch.yaml`, `ad_binary_scratch.yaml`,
  `mci_binary_scratch.yaml`, and `mci_matched_binary_scratch.yaml`.
- `linear_probe`, `partial`, and `full` configs keep `max_epochs: 100`.
- Updated `docs/downstream_v3_plan.md` and
  `scripts/audit_downstream_v3_goal.py` so final audit expects
  `scratch=30` and non-scratch modes `=100`.
- Active scratch processes that were already launched before this edit keep
  their process-start in-memory value unless they are stopped/resumed or finish
  naturally.

Scratch cap retraction and convergence reports:

- Later on 2026-06-21, the scratch cap was retracted because a 30-epoch cap can
  unfairly underestimate scratch when the random-initialized model is still
  improving.
- Restored all 8 active v3 fast scratch configs to
  `downstream_train.max_epochs: 100` and restored
  `scripts/audit_downstream_v3_goal.py` to require `max_epochs=100` for every
  mode.
- Added early convergence checkpointing in `src/eyemae/finetune.py`:
  `checkpoint_epoch_000.pt` / `metrics_epoch_000.json` for exactly one
  training epoch, and `checkpoint_best_within30.pt` /
  `metrics_best_within30.json` for the validation-best checkpoint inside
  epochs 0-29.
- Added `scripts/summarize_downstream_v3_convergence.py` to evaluate and
  summarize the two requested convergence-speed views:
  `within30` and `epoch1`.
- Added `scripts/prepare_downstream_v3_epoch1_configs.py` to generate a
  separate 32-job epoch1 config list under `configs/downstream_epoch1/`, with
  outputs isolated under
  `outputs/downstream_v3_fast_convergence/epoch1_train/`.
- Added `scripts/lock_downstream_v3_within30_checkpoints.py` to preserve
  `checkpoint_best_within30.pt` / `metrics_best_within30.json` once a running
  job reaches epoch 29, so later epoch30+ improvements cannot overwrite the
  requested within-30-epoch comparison checkpoint.
- Updated `scripts/lock_downstream_v3_within30_checkpoints.py` so an existing
  within30 lock is recognized before checking whether the current global best
  has moved to epoch 30 or later.
- Started tmux session `eyemae_downstream_v3_within30_lockwatch`, which runs
  the lock script once per minute while the active queue is training.
- Updated `scripts/run_downstream_v3_queue.py` so a clean resume queue
  automatically passes `--resume checkpoint_last.pt` when an output directory
  has a last checkpoint but no `metrics_final.json`.
- Updated `scripts/wait_and_run_downstream_v3_queue.py` with
  `--allow-upstream-failures`, so a tail queue can still launch after the
  current upstream queue reports the known `epilepsy_binary_scratch` SIGTERM
  from the retracted 30-epoch cap experiment.
- Started a clean-resume watcher in tmux session
  `eyemae_downstream_v3_tail32_allowfail`. It waits for the current queue to
  become idle, then runs the expanded 32-job config list under
  `outputs/downstream_v3_fast_logs/run_20260620_random_pd_fast_clean_resume/`
  with `--resume-status` pointing at the original queue status.
- Added `scripts/update_downstream_epoch_dynamics_doc.py` to regenerate the
  downstream training-dynamics section in `docs/newdata_v3_training_log.md`
  from fine-tune logs. The table format is one row per epoch with columns
  `scratch`, `linear`, `partial`, and `full`, and each cell records
  `train_loss / val_subject_metric / val_subject_f1`.
- The temporary 30-epoch epilepsy scratch final metrics were archived to
  `outputs/downstream_v3_fast_convergence/within30/epilepsy_binary/scratch/`
  and removed from the main `metrics_final.json` path so the job remains
  incomplete for the 100-epoch main result.

Detox random split rerun:

- Added `scripts/prepare_detox_random_finetune.py` for the requested detox
  train/validation/test reshuffle. It creates a subject-level stratified random
  binary split from `downstream/戒毒所/{train,validation,test}.csv` while
  preserving the original split subject counts.
- Generated `downstream/戒毒所_random_seed20260621/` with seed `20260621`:
  train `109` subjects (`65` control, `44` patient), validation `27`
  subjects (`16` control, `11` patient), and test `34` subjects (`20`
  control, `14` patient). `split_summary.json` reports no subject overlap.
- Regenerated the same deterministic split and configs with
  `python scripts/prepare_detox_random_finetune.py --seed 20260621 --force`
  at `2026-06-21 03:27 CST` after the explicit rerun request. The training
  output directories were not cleared.
- Generated
  `configs/downstream/detox_binary_random_seed20260621_fast_{scratch,linear_probe,partial,full}.yaml`
  with the same downstream training policy as the active v3 fast fine-tune
  configs: `max_epochs=100`, `early_stopping_patience=10`, and
  `min_epochs_before_early_stopping=0`.
- The four detox random configs use the generated CSV index files directly and
  record the realized subject split ratios:
  train `0.6411764705882353`, validation `0.1588235294117647`, test `0.2`.
- Generated `configs/downstream/queue_detox_random_seed20260621_fast.txt`.
  Dry-run maps the four modes to GPU1-4.
- Started tmux session `eyemae_downstream_v3_detox_random_seed20260621`. It
  waits for
  `outputs/downstream_v3_fast_logs/run_20260620_random_pd_fast_clean_resume/status.json`
  to become idle, then launches the 4-job detox random queue under
  `outputs/downstream_v3_fast_logs/run_20260621_detox_random_seed20260621/`.
  This avoids competing with the active first-version and clean-resume queues.

Missing-job補跑 queue:

- Added
  `configs/downstream/queue_missing_pd_binary_epilepsy_scratch_20260621.txt`
  after auditing that the original active launcher had read the older 28-job
  queue before the four PD binary jobs were appended, and that
  `epilepsy_binary_scratch` still lacked the fair 100-epoch
  `metrics_final.json` after the retracted 30-epoch cap.
- Confirmed GPU2/GPU4 were idle via `nvidia-smi`, then started tmux session
  `eyemae_downstream_v3_missing_pd_binary_epilepsy_20260621` with:
  `python scripts/run_downstream_v3_queue.py --gpus 2,4 --config-list-file configs/downstream/queue_missing_pd_binary_epilepsy_scratch_20260621.txt --log-dir outputs/downstream_v3_fast_logs/run_20260621_missing_pd_binary_epilepsy_scratch --poll-seconds 30`.
- Initial补跑 status:
  `0 completed / 2 running / 3 pending / 0 failures`, with
  `pd_binary_random_seed20260620_fast_scratch` on GPU2 and
  `pd_binary_random_seed20260620_fast_linear_probe` on GPU4.
- Stopped stale tmux watchers `eyemae_downstream_v3_tail32` and
  `eyemae_downstream_v3_tail32_allowfail` after the explicit補跑 queue was
  launched, because they would otherwise be able to launch duplicate missing
  jobs as soon as the original main queue became idle.
- Stopped `eyemae_downstream_v3_detox_random_seed20260621` because it was
  waiting for the now-disabled clean-resume status path. The detox random
  queue remains prepared and should be launched explicitly after the
  first-version 32-job pressure clears.
- Started replacement waiting tmux session
  `eyemae_downstream_v3_detox_random_after_mci_20260621` at
  `2026-06-21 04:57 CST`. It waits for
  `outputs/downstream_v3_fast_logs/run_mci_followup_seed20260621/status.json`
  to exist and become idle, then runs
  `configs/downstream/queue_detox_random_seed20260621_fast.txt` under
  `outputs/downstream_v3_fast_logs/run_20260621_detox_random_seed20260621/`
  on GPU2/GPU4. This keeps the detox random rerun queued behind the corrected
  MCI follow-up rather than competing for the same GPUs.
- Status update at 2026-06-21 04:39 CST:
  `outputs/downstream_v3_fast_logs/run_20260620_random_pd_fast/status.json`
  is idle with `28 completed / 0 running / 0 pending / 1 known failure`
  (`epilepsy_binary_scratch rc=-15` from the retracted scratch-cap run).
  The authoritative first-version completion path is now the explicit
  missing-job queue, which still needs the four PD binary jobs and the fair
  100-epoch `epilepsy_binary_scratch` result.
- Status update at 2026-06-21 04:53 CST:
  the explicit missing-job queue is healthy and still running
  `pd_binary_random_seed20260620_fast_scratch` on GPU2 and
  `pd_binary_random_seed20260620_fast_linear_probe` on GPU4, with
  `3` jobs pending. The refreshed first-version summary still has
  `32` rows and `5` missing final metrics.
- Status update at 2026-06-21 05:09 CST:
  `pd_binary_random_seed20260620_fast_linear_probe` finished successfully.
  It selected best epoch `5`, validation subject AUROC `0.97441`, and test
  subject AUROC `0.96407`. The explicit missing-job queue is now
  `1 completed / 2 running / 2 pending / 0 failures`: `pd_binary` scratch
  remains on GPU2, `pd_binary` partial started on GPU4, and pending jobs are
  `pd_binary` full plus fair 100-epoch `epilepsy_binary_scratch`. Refreshed
  `summary_first_version.*` has `32` rows and `4` missing final metrics; the
  incomplete audit reports `errors=[]`, `warnings=4`, and
  `metrics_final_count=28`.

Epoch1 convergence queue:

- Started tmux session `eyemae_downstream_v3_epoch1_20260621` on GPU1/GPU3
  with:
  `python scripts/run_downstream_v3_queue.py --gpus 1,3 --config-list-file configs/downstream_epoch1/queue_epoch1.txt --log-dir outputs/downstream_v3_fast_logs/run_epoch1 --poll-seconds 30`.
- Initial epoch1 status:
  `0 completed / 2 running / 30 pending / 0 failures`, with
  `pd_related_5class_random_seed20260620_fast_scratch_epoch1` on GPU1 and
  `pd_related_5class_random_seed20260620_fast_linear_probe_epoch1` on GPU3.
- This queue writes isolated outputs under
  `outputs/downstream_v3_fast_convergence/epoch1_train/`, so it does not
  overwrite first-version fine-tune outputs.
- Status update at 2026-06-21 04:53 CST:
  `3 completed / 2 running / 27 pending / 0 failures`; refreshed
  `summary_epoch1.*` has `32` rows and `29` missing outputs.
- Status update at 2026-06-21 05:03 CST:
  `4 completed / 2 running / 26 pending / 0 failures`; `pd_related_5class`
  epoch1 `full` finished with validation subject macro AUROC `0.89423` and
  test subject macro AUROC `0.87461`. Refreshed `summary_epoch1.*` has
  `32` rows and `28` missing outputs.
- Status update at 2026-06-21 05:04 CST:
  `5 completed / 2 running / 25 pending / 0 failures`; `pd_binary` epoch1
  `scratch` finished with validation subject AUROC `0.93622` and test subject
  AUROC `0.93382`. Refreshed `summary_epoch1.*` has `32` rows and
  `27` missing outputs.
- Status update at 2026-06-21 05:11 CST:
  `6 completed / 2 running / 24 pending / 0 failures`; `pd_binary` epoch1
  `linear_probe` finished with validation subject AUROC `0.96129` and test
  subject AUROC `0.96171`. Refreshed `summary_epoch1.*` has `32` rows and
  `26` missing outputs.
- Status update at 2026-06-21 05:13 CST:
  `7 completed / 2 running / 23 pending / 0 failures`; `pd_binary` epoch1
  `partial` finished with validation subject AUROC `0.97854` and test subject
  AUROC `0.97885`. The queue launched `epilepsy_binary_scratch_epoch1` on
  GPU3 while `pd_binary_full_epoch1` continues on GPU1. Refreshed
  `summary_epoch1.*` has `32` rows and `25` missing outputs.
- Status update at 2026-06-21 05:21 CST:
  `9 completed / 2 running / 21 pending / 0 failures`; `pd_binary` epoch1
  `full` finished with validation subject AUROC `0.98891` and test subject
  AUROC `0.98572`, and `epilepsy_binary` epoch1 `scratch` finished with
  validation subject AUROC `0.75333` and test subject AUROC `0.79405`. The
  queue launched `epilepsy_binary_linear_probe_epoch1` on GPU1 and
  `epilepsy_binary_partial_epoch1` on GPU3. Refreshed `summary_epoch1.*` has
  `32` rows and `23` missing outputs.
- Status update at 2026-06-21 05:33 CST:
  `16 completed / 2 running / 14 pending / 0 failures`; newly completed
  epoch1 outputs include `epilepsy_binary` `linear_probe`, `partial`, and
  `full`, plus `detox_binary` all four modes. Refreshed `summary_epoch1.*` has
  `32` rows and `16` missing outputs. The queue is now running
  `migraine_binary_scratch_epoch1` on GPU3 and
  `migraine_binary_linear_probe_epoch1` on GPU1.
- Status update at 2026-06-21 05:37 CST:
  `20 completed / 2 running / 10 pending / 0 failures`; all four
  `migraine_binary` epoch1 outputs finished. Refreshed `summary_epoch1.*` has
  `32` rows and `12` missing outputs. The queue is now running
  `ad_binary_scratch_epoch1` on GPU3 and `ad_binary_linear_probe_epoch1` on
  GPU1.
- Verified waiting sessions at 2026-06-21 05:35 CST: MCI follow-up tmux is
  still waiting on the missing-job status (`1 completed / 2 running /
  2 pending`), and detox random tmux is still waiting for the MCI follow-up
  status file to be created. This is the expected dependency chain, not an
  exited queue.
- GPU/process sample at 2026-06-21 05:38 CST: GPU1 is running epoch1
  `ad_binary/linear_probe`, GPU3 is running epoch1 `ad_binary/scratch`, GPU2
  is running missing-job `pd_binary/scratch`, and GPU4 is running missing-job
  `pd_binary/partial`; sampled utilization is 95-98% on GPU1-4.
- Status update at 2026-06-21 05:40 CST:
  epoch1 is `22 completed / 2 running / 8 pending / 0 failures`; `ad_binary`
  epoch1 `scratch` and `linear_probe` finished, and `summary_epoch1.*` now has
  `32` rows with `10` missing outputs. The queue is running
  `ad_binary_partial_epoch1` on GPU1 and `ad_binary_full_epoch1` on GPU3.
  The first-version summary/audit still has four missing final metrics:
  `pd_binary` `scratch`, `partial`, `full`, and fair `epilepsy_binary/scratch`.
- Status update at 2026-06-21 05:42 CST:
  epoch1 is `23 completed / 2 running / 7 pending / 0 failures`; `ad_binary`
  epoch1 `partial` finished and `summary_epoch1.*` now has `32` rows with
  `9` missing outputs. The queue is running `ad_binary_full_epoch1` on GPU3
  and has started `mci_binary_scratch_epoch1` on GPU1.
- Status update at 2026-06-21 05:43 CST:
  epoch1 is `24 completed / 2 running / 6 pending / 0 failures`; all four
  `ad_binary` epoch1 outputs are complete, and `summary_epoch1.*` now has
  `32` rows with `8` missing outputs. The queue is running
  `mci_binary_scratch_epoch1` on GPU1 and `mci_binary_linear_probe_epoch1` on
  GPU3.
- Status update at 2026-06-21 05:45 CST:
  no new first-version final metrics landed; `summary_first_version.*` remains
  `32` rows with `4` missing outputs and the incomplete audit still reports
  `errors=[]`, `warnings=4`, `metrics_final_count=28`. Epoch1 remains
  `24 completed / 2 running / 6 pending / 0 failures`, with
  `mci_binary_scratch_epoch1` on GPU1 and `mci_binary_linear_probe_epoch1` on
  GPU3. GPU1-4 sampled utilization was 96-99%. MCI follow-up and detox random
  tmux sessions are still alive and waiting on the expected dependency chain.
- Status update at 2026-06-21 05:47 CST:
  epoch1 is `26 completed / 2 running / 4 pending / 0 failures`; old-view
  `mci_binary` epoch1 `scratch` and `linear_probe` finished, and
  `summary_epoch1.*` now has `32` rows with `6` missing outputs. The queue is
  running `mci_binary_partial_epoch1` on GPU3 and `mci_binary_full_epoch1` on
  GPU1. These are convergence-speed outputs for the original first-version MCI
  view and remain provisional because the original MCI labels were already
  flagged as invalid; corrected MCI follow-up is still queued separately.
- Status update at 2026-06-21 05:49 CST:
  no new final metrics landed after the 05:47 update. First-version remains
  `32` rows with `4` missing outputs and audit `errors=[]`, `warnings=4`,
  `metrics_final_count=28`. Epoch1 remains `26 completed / 2 running /
  4 pending / 0 failures`, with old-view `mci_binary_partial_epoch1` on GPU3
  and old-view `mci_binary_full_epoch1` on GPU1. GPU1-4 sampled utilization was
  94-99%, and the MCI follow-up / detox random waiting chain has not started
  because the missing-job queue is still active.
- Status update at 2026-06-21 05:51 CST:
  old-view `mci_binary` epoch1 `partial` finished, so `summary_epoch1.*` now
  has `32` rows with `5` missing outputs. Epoch1 status is `27 completed /
  2 running / 3 pending / 0 failures`, with old-view `mci_binary_full_epoch1`
  and old-label `mci_matched_binary_scratch_epoch1` running. These outputs are
  convergence-speed artifacts for previously flagged MCI views and remain
  provisional; corrected MCI follow-up is still waiting behind the missing-job
  queue.

GPU/process monitor at 2026-06-21 05:31 CST:

| GPU | active queue role | PID | sampled utilization |
| ---: | --- | ---: | ---: |
| 1 | epoch1 `epilepsy_binary/full` at sample time; later switched to `migraine_binary/linear_probe` | 3617440 | 97% |
| 2 | missing-job `pd_binary/scratch` | 3545934 | 96% |
| 3 | epoch1 `detox_binary/full` at sample time; later switched to `migraine_binary/scratch` | 3622295 | 97% |
| 4 | missing-job `pd_binary/partial` | 3601741 | 98% |

MCI label-corrected follow-up:

- Added `scripts/prepare_mci_followup_finetune.py`.
- Generated `downstream/MCI_original_only_no_matched/` by keeping only original
  `downstream/MCI/` rows whose `source_dataset` is not `匹配后`. This creates the
  replacement `mci_original_only_binary` task and removes 33,154 matched-copy
  rows from the invalid mixed MCI view.
- Built a raw `subject -> health_label` anchor map from those original MCI rows.
  The anchor map has 383 subjects and no label conflicts.
- Generated `downstream/MCI匹配后_random_seed20260621/` from standalone
  `downstream/MCI匹配后/`, but ignored every source `MCI匹配后` label. A row is kept
  only when its raw `subject` exists in the original MCI anchor map, and its
  `health_label` is overwritten by the original MCI subject label.
- The matched follow-up kept all 33,154 rows, dropped 0 unmapped rows, and
  changed 33,154 labels, confirming that the previous matched labels were fully
  crossed relative to original MCI labels.
- User clarified the intended rule explicitly: the matched view is not a label
  source. Only raw subjects that exist in the original MCI anchor map are
  usable; rows without that anchor must be excluded. The current generation and
  queue already follow this rule, so no training queue change was needed.
- Generated four configs for each corrected task:
  `configs/downstream/mci_original_only_binary_{scratch,linear_probe,partial,full}.yaml`
  and
  `configs/downstream/mci_matched_binary_random_seed20260621_{scratch,linear_probe,partial,full}.yaml`.
- Generated `configs/downstream/queue_mci_followup_seed20260621.txt`. Dry-run
  maps the eight jobs onto GPU2/GPU4.
- Started tmux session `eyemae_downstream_v3_mci_followup_seed20260621`. It
  waits for
  `outputs/downstream_v3_fast_logs/run_20260621_missing_pd_binary_epilepsy_scratch/status.json`
  to become idle, then launches the corrected MCI follow-up queue under
  `outputs/downstream_v3_fast_logs/run_mci_followup_seed20260621/`.
- Status update at 2026-06-21 04:53 CST:
  the MCI follow-up tmux session is still waiting for the missing-job queue;
  `outputs/downstream_v3_fast_logs/run_mci_followup_seed20260621/status.json`
  does not exist yet because no corrected MCI job has launched.
- Status update at 2026-06-21 05:56 CST:
  the user clarified the MCI correction in stricter terms. The current
  generated data matches this rule: `MCI匹配后` is only a source of sample rows,
  not a label source. Every matched row must have a raw `subject` in the
  original-MCI anchor map built after excluding `source_dataset=匹配后`; otherwise
  the row is excluded. For kept rows, `health_label` is overwritten from the
  original-MCI subject label. Verification results:
  `MCI_original_only_no_matched` has 58,279 rows and 0 anchor mismatches;
  `MCI匹配后_random_seed20260621` has 33,154 rows, `outside_anchor=0`, and
  `label_mismatch_vs_anchor=0`. The epoch1 convergence queue later completed
  `32/32` with `summary_epoch1.*` refreshed and `missing_count=0`; old
  `mci_binary` and old-label `mci_matched_binary` epoch1 outputs remain
  provisional diagnostics only and must not be used as formal MCI results. The
  corrected MCI follow-up queue has not launched yet because it is still waiting
  for the missing-job queue.
- Status update at 2026-06-21 06:00 CST:
  refreshed `summary_first_version.*` and the incomplete goal audit. The first
  version still has `32` rows with `4` missing final metrics and audit
  `errors=[]`, `warnings=4`, `metrics_final_count=28`. The missing-job queue is
  healthy with `1 completed / 2 running / 2 pending / 0 failures`: running
  `pd_binary_random_seed20260620_fast_scratch` and
  `pd_binary_random_seed20260620_fast_partial`; pending
  `pd_binary_random_seed20260620_fast_full` and fair `epilepsy_binary_scratch`.
  Latest logs show `pd_binary/scratch` at epoch 11, step 25200,
  train loss 0.29962, latest validation epoch 10 AUROC 0.96404; and
  `pd_binary/partial` at epoch 9, step 21700, train loss 0.13866, latest
  validation epoch 8 AUROC 0.99186. Corrected MCI and detox random queues have
  not launched yet because they are intentionally chained behind the missing-job
  queue. Runtime checks show the MCI follow-up tmux is repeatedly waiting on
  `completed=1 running=2 pending=2 failures=0`, while detox random is waiting
  for `run_mci_followup_seed20260621/status.json`; this confirms the queue
  chaining is active rather than accidentally skipped.
- Config conformance fix at 2026-06-21 06:06 CST:
  found that several formal downstream and epoch1 YAML configs still had
  `split.split_summary: pretrain/pretrain_split_summary.json` even though their
  `data.train_index` pointed to `downstream/<view>/train.csv`. The fine-tune
  runtime audits downstream splits from `train_index.parent/split_summary.json`,
  so completed training used the correct downstream split files; however the
  stale config field violated the plan's reproducibility contract. Mechanically
  updated 52 configs in `configs/downstream/` and `configs/downstream_epoch1/`
  so `split.split_summary` now matches `Path(data.train_index).parent /
  "split_summary.json"`. Pending `epilepsy_binary_scratch` is corrected before
  launch. Enhanced `scripts/audit_downstream_v3_goal.py` to check this field,
  and verified `bad_count=0`, `python -m py_compile scripts/audit_downstream_v3_goal.py`,
  and enhanced incomplete audit with `errors=[]`, `warnings=4`.
- Status update at 2026-06-21 06:09 CST:
  refreshed `summary_first_version.*` again and reran the enhanced incomplete
  audit. It still reports `errors=[]`, `warnings=4`, `metrics_final_count=28`,
  with no `split_summary` consistency errors. The missing-job queue remains
  healthy with `1 completed / 2 running / 2 pending / 0 failures`; GPU sampling
  shows GPU2 PID 3545934 at 98% and GPU4 PID 3601741 at 92%. Latest logs show
  `pd_binary/scratch` at epoch 12, step 27050, train loss 0.28537, latest
  validation epoch 11 AUROC 0.96923; and `pd_binary/partial` at epoch 11,
  step 24950, train loss 0.12092, latest validation epoch 10 AUROC 0.98891.
- Status update at 2026-06-21 06:11 CST:
  refreshed `summary_first_version.*` and reran the enhanced incomplete audit;
  results remain `errors=[]`, `warnings=4`, `metrics_final_count=28`. The
  missing-job queue is still healthy with `1 completed / 2 running / 2 pending /
  0 failures`. GPU sampling shows GPU2 PID 3545934 at 98% and GPU4 PID 3601741
  at 97%. Latest logs show `pd_binary/scratch` at epoch 12, step 27650, train
  loss 0.28588; `pd_binary/partial` at epoch 11, step 26050, train loss
  0.12165. Corrected MCI and detox random queues remain correctly chained
  behind the missing-job queue.
- Status update at 2026-06-21 06:13 CST:
  refreshed `summary_first_version.*` and reran the enhanced incomplete audit;
  status remains `errors=[]`, `warnings=4`, `metrics_final_count=28`. The
  missing-job queue is still `1 completed / 2 running / 2 pending / 0 failures`.
  GPU sampling again shows GPU2/GPU4 active at 98%/97%. Latest logs show
  `pd_binary/scratch` at epoch 12, step 28250, train loss 0.28502, latest
  validation epoch 11 AUROC 0.96923; and `pd_binary/partial` at epoch 12,
  step 26900, train loss 0.11867, latest validation epoch 11 AUROC 0.99219.
- Status update at 2026-06-21 06:16 CST:
  refreshed `summary_first_version.*` and reran the enhanced incomplete audit;
  status remains `errors=[]`, `warnings=4`, `metrics_final_count=28`. The
  missing-job queue is still `1 completed / 2 running / 2 pending / 0 failures`.
  GPU sampling shows GPU2/GPU4 active at 99%/95%. Latest logs show
  `pd_binary/scratch` at epoch 12, step 28650, train loss 0.28634, latest
  validation epoch 11 AUROC 0.96923; and `pd_binary/partial` at epoch 12,
  step 27950, train loss 0.11460, latest validation epoch 11 AUROC 0.99219.
- Status update at 2026-06-21 06:18 CST:
  refreshed `summary_first_version.*` and reran the enhanced incomplete audit;
  status remains `errors=[]`, `warnings=4`, `metrics_final_count=28`. The
  missing-job queue remains `1 completed / 2 running / 2 pending / 0 failures`.
  GPU sampling shows GPU2/GPU4 active at 98%/97%. Latest logs show
  `pd_binary/scratch` at epoch 13, step 29200, train loss 0.27272, latest
  validation epoch 12 AUROC 0.96831; and `pd_binary/partial` at epoch 13,
  step 28700, train loss 0.12267, latest validation epoch 12 AUROC 0.99304.
- Status update at 2026-06-21 06:22 CST:
  refreshed `summary_first_version.*` and reran the enhanced incomplete audit;
  status remains `errors=[]`, `warnings=4`, `metrics_final_count=28`. The
  missing-job queue is still `1 completed / 2 running / 2 pending / 0 failures`.
  GPU sampling shows GPU2/GPU4 active at 98%/97%. Latest logs show
  `pd_binary/scratch` at epoch 13, step 30200, train loss 0.27149, latest
  validation epoch 12 AUROC 0.96831; and `pd_binary/partial` at epoch 13,
  step 30650, train loss 0.10864, latest validation epoch 12 AUROC 0.99304.
- Status update at 2026-06-21 06:24 CST:
  refreshed `summary_first_version.*` and reran the enhanced incomplete audit;
  status remains `errors=[]`, `warnings=4`, `metrics_final_count=28`. The
  missing-job queue remains `1 completed / 2 running / 2 pending / 0 failures`.
  GPU sampling shows GPU2/GPU4 active at 96%/97%. Latest logs show
  `pd_binary/scratch` at epoch 13, step 30800, train loss 0.27098, latest
  validation epoch 12 AUROC 0.96831; and `pd_binary/partial` at epoch 14,
  step 31550, train loss 0.09777, latest validation epoch 13 AUROC 0.99180.
- Status update at 2026-06-21 06:26 CST:
  refreshed `summary_first_version.*` and reran the enhanced incomplete audit;
  status remains `errors=[]`, `warnings=4`, `metrics_final_count=28`. The
  missing-job queue is still `1 completed / 2 running / 2 pending / 0 failures`.
  GPU sampling shows GPU2/GPU4 active at 96%/96%. Latest logs show
  `pd_binary/scratch` at epoch 14, step 31200, train loss 0.24831, latest
  validation epoch 13 AUROC 0.96306; and `pd_binary/partial` at epoch 14,
  step 32500, train loss 0.09966, latest validation epoch 13 AUROC 0.99180.
- MCI label rule clarification at 2026-06-21 06:31 CST:
  verified `scripts/prepare_mci_followup_finetune.py` and the generated
  split summaries. The corrected MCI matched-row task uses `MCI匹配后` rows only
  as samples; labels are never read from `MCI匹配后`. Each matched row is kept
  only if its raw `subject` appears in the original `MCI` subject-label anchor
  after removing `source_dataset=匹配后`; otherwise it is dropped. For kept rows,
  `health_label` is overwritten from the original MCI raw-subject label. The
  generated seed-20260621 split kept 33,154 / 33,154 matched rows, dropped 0
  unmapped rows, and overwrote all 33,154 labels. The old `mci_binary` and
  old `mci_matched_binary` metrics remain invalid/provisional and are not the
  official MCI follow-up results.
- Queue execution update at 2026-06-21 06:40 CST:
  the missing-job queue was originally constrained to GPU2/GPU4 while GPU1/GPU3
  were idle. Stopped the old 2-GPU queue manager and restarted the same queue
  as `eyemae_downstream_v3_missing_pd_binary_epilepsy_4gpu_resume_20260621`
  with `--gpus 1,2,3,4`, same log/status directory, and `--resume-status`.
  Stopping the old manager also ended the two active training children, so the
  restarted queue resumed `pd_binary/scratch`, `pd_binary/partial`, and
  `epilepsy_binary/scratch` from `checkpoint_last.pt`; `pd_binary/full` started
  fresh on GPU3. Replaced the MCI follow-up and detox random waiters with
  4-GPU waiters using the same status/log paths. To preserve pre-resume logs,
  `scripts/run_downstream_v3_queue.py` now appends to existing job logs and
  writes a launch marker instead of truncating them. Verified with
  `python -m py_compile scripts/run_downstream_v3_queue.py`.

Current full test-suite evidence:

```text
pytest -q tests
38 passed in 11.21s
```

- MCI label rule tightening at 2026-06-21 07:24 CST:
  the user clarified that the MCI fine-tune must not use any label from
  `MCI匹配后`. Updated `scripts/prepare_mci_followup_finetune.py` so the matched
  follow-up still uses `MCI匹配后` rows as samples, but keeps a row only when
  raw `subject` exists in the original `MCI` anchor after removing
  `source_dataset=匹配后`. For kept rows, both `health_label` and the label-like
  metadata field `source_group` are overwritten from the original MCI subject
  label; otherwise the row is dropped.
- Regenerated the corrected MCI views/configs with
  `python scripts/prepare_mci_followup_finetune.py --force` and verified
  `python -m py_compile scripts/prepare_mci_followup_finetune.py`.
  `MCI匹配后_random_seed20260621` now has 33,154 rows, 218 subjects,
  `outside_anchor_rows=0`, `health_label_mismatch_rows=0`, and
  `source_group_mismatch_rows=0`. The corrected MCI queue has not launched yet,
  so it will read this regenerated data.
- Official downstream scope tightening at 2026-06-21 07:28 CST:
  replaced old `mci_binary` and `mci_matched_binary` entries in
  `configs/downstream/queue_pd_random_seed20260620_fast.txt` with
  `mci_original_only_binary` and `mci_matched_binary_random_seed20260621`.
  Updated `scripts/audit_downstream_v3_goal.py`, `scripts/run_downstream_v3_queue.py`,
  `docs/current_goal_guardrail.md`, and `docs/downstream_v3_plan.md` so the
  official 32-job scope no longer counts the old MCI outputs as complete.
  Refreshed `summary_first_version.*`: still 32 rows, now 12 missing final
  metrics, including the 8 corrected MCI jobs that have not launched yet.
- Audit status handling update at 2026-06-21 07:32 CST:
  `scripts/audit_downstream_v3_goal.py` now distinguishes historical queue
  failures from unresolved official jobs. A status entry in `running`,
  `pending`, or `failures` is treated as unresolved only while the corresponding
  official config still lacks `metrics_final.json`. This keeps final audit
  strict on the 32 official result artifacts while preventing the historical
  interrupted `epilepsy_binary_scratch.yaml` status from blocking completion
  after recovery writes its final metrics.
- Queue utilization update at 2026-06-21 07:37 CST:
  after `pd_binary/partial` and `epilepsy_binary/scratch` completed, GPU2/GPU4
  became idle while `pd_binary/scratch` and `pd_binary/full` continued on
  GPU1/GPU3. Stopped the MCI wait-only tmux session
  `eyemae_downstream_v3_mci_followup_seed20260621_4gpu_wait` to prevent a
  duplicate future launch, then started
  `eyemae_downstream_v3_mci_followup_seed20260621_gpu24_now` with
  `configs/downstream/queue_mci_followup_seed20260621.txt` on GPU2/GPU4 and
  log/status directory `outputs/downstream_v3_fast_logs/run_mci_followup_seed20260621`.
  This uses the corrected MCI data generated earlier and keeps GPU1-4 active
  across the combined recovery/MCI workload.
- MCI label-anchor confirmation at 2026-06-21 08:01 CST:
  re-checked the user's intended rule against `scripts/prepare_mci_followup_finetune.py`
  and the generated CSV files. The implementation already matches the intended
  contract: original `MCI` labels after removing `source_dataset=匹配后` are the
  only label authority; `MCI匹配后` contributes rows only when raw `subject`
  exists in that original-MCI anchor; its `health_label` and label-like
  `source_group` are both ignored and overwritten. Verification on
  `MCI匹配后_random_seed20260621`: 33,154 rows, 218 subjects, 0 outside-anchor
  rows, 0 `health_label` mismatches, 0 `source_group` mismatches, and 33,154
  source matched labels overwritten relative to `MCI匹配后`.
- MCI plan wording tightened at 2026-06-21 08:35 CST:
  updated `docs/downstream_v3_plan.md` so the formal task list explicitly says
  `mci_original_only_binary` removes `source_dataset=匹配后`, while
  `mci_matched_binary_random_seed20260621` keeps a matched row only if its raw
  `subject` exists in the original-MCI subject anchor and never uses the
  `MCI匹配后` source label. No generated CSV, config, or running training job was
  changed by this wording-only update.
- MCI queue GPU3 resume update at 2026-06-21 08:52 CST:
  after `pd_binary/full` completed, GPU3 was idle while the corrected MCI queue
  still had three pending jobs. The old MCI queue manager was stopped and the
  corrected MCI queue was restarted as
  `eyemae_downstream_v3_mci_followup_seed20260621_gpu234_resume_0849` with
  `--gpus 2,3,4`, the same log/status directory, and `--resume-status`.
  The two interrupted active MCI jobs were relaunched from their
  `checkpoint_last.pt` files:
  `mci_original_only_binary/full` on GPU2 and
  `mci_matched_binary_random_seed20260621/scratch` on GPU3. The newly freed
  GPU4 launched `mci_matched_binary_random_seed20260621/linear_probe`.
  Status after restart: 3 completed, 3 running, 2 pending, 0 failures.
- Corrected MCI completion and detox random launch at 2026-06-21 09:31 CST:
  the corrected MCI follow-up queue finished all eight jobs with 0 failures.
  The waiting detox-random queue then launched
  `detox_binary_random_seed20260621_fast_{scratch,linear_probe,partial,full}.yaml`
  on GPU1/GPU2/GPU3/GPU4. At launch time `pd_binary/scratch` was still running
  on GPU1, so GPU1 is shared by `pd_binary/scratch` and detox-random `scratch`;
  no OOM or runtime error was observed.
- Detox random completion at 2026-06-21 10:22 CST:
  all four `detox_binary_random_seed20260621` modes completed with 0 failures.
  The final test subject AUROCs were `scratch=0.86071`,
  `linear_probe=0.82143`, `partial=0.82500`, and `full=0.85714`. After
  detox-random `scratch` finished, GPU1 was no longer shared and only
  `pd_binary/scratch` remained running from the official first-version scope.
- Final downstream-v3 audit completion at 2026-06-21 11:08 CST:
  `pd_binary/scratch` completed as the last official first-version job.
  Updated `scripts/audit_downstream_v3_goal.py` so strict completion checks
  require 32 official config-listed `metrics_final.json` files while allowing
  extra metrics under `outputs/downstream_v3_fast` from follow-up reruns such
  as detox-random. Final strict audit reports
  `official_metrics_final_count=32`, `metrics_final_count=44`,
  `extra_metrics_final_count=12`, `errors=[]`, and `warnings=[]`. Full tests
  passed with `pytest -q tests`: 39 passed.
- MCI matched label-fixed rerun at 2026-06-21 12:14 CST:
  the `MCI匹配后` healthy/disease label direction was confirmed to be reversed
  relative to the previous original-anchor relabeling assumption. Updated
  `scripts/prepare_mci_followup_finetune.py` to support a matched-only
  follow-up with `--invert-matched-anchor-labels` and a task suffix. Generated
  `downstream/MCI匹配后_random_seed20260622_label_fixed/` with a fresh
  subject-level stratified random split (`seed=20260622`), keeping matched
  rows only when raw `subject` exists in the original-MCI anchor and setting
  final `health_label` to the inverse of that anchor label. The regenerated
  split has 33,154 rows, 218 subjects, 0 subject/split overlaps, and balanced
  subject counts: train 70/70, validation 17/17, test 22/22. Generated four
  configs named
  `configs/downstream/mci_matched_binary_random_seed20260622_label_fixed_{scratch,linear_probe,partial,full}.yaml`
  plus `configs/downstream/queue_mci_matched_binary_random_seed20260622_label_fixed.txt`.
  Launched the four-mode rerun on GPU1/GPU2/GPU3/GPU4 with log/status directory
  `outputs/downstream_v3_fast_logs/run_20260622_mci_matched_label_fixed`.
- MCI matched label-fixed completion at 2026-06-21 12:38 CST:
  all four `mci_matched_binary_random_seed20260622_label_fixed` modes completed
  with returncode 0 and 0 queue failures. Final best checkpoints were selected
  by `val/subject/auroc`: `scratch` best epoch 4, `linear_probe` epoch 3,
  `partial` epoch 7, and `full` epoch 3. Test subject AUROCs were
  `scratch=0.60537`, `linear_probe=0.53926`, `partial=0.65289`, and
  `full=0.63017`; partial is the strongest by test subject AUROC in this
  rerun. Error scan found no `Traceback`, `ERROR`, `RuntimeError`, CUDA OOM, or
  `nan` in the label-fixed logs. Targeted tests passed:
  `pytest -q tests/test_downstream.py tests/test_downstream_packed.py`: 12
  passed.
