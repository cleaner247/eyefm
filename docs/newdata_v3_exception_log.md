# Newdata V3 Exception Log

This document records environment and runtime exceptions encountered while
preparing/running the v3 new-data pretrain and downstream pipeline.

## 2026-06-19 Pretrain Preparation

### GPU Visibility Check

Command:

```bash
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
```

Observed result:

```text
NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver.
```

Additional PyTorch check:

```bash
python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.device_count())
PY
```

Observed result:

```text
torch 2.8.0+cu126
cuda available False
device count 0
```

Impact:

- This was a sandbox visibility issue. A sandbox-external GPU check was required
  before launching training.
- The user reported `nvtop` can see these GPUs as idle.

Sandbox-external check:

```bash
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
```

Observed result outside sandbox:

```text
0, NVIDIA A100-SXM4-80GB, 35730, 81920, 100
1, NVIDIA A100-SXM4-80GB, 17, 81920, 0
2, NVIDIA A100-SXM4-80GB, 17, 81920, 0
3, NVIDIA A100-SXM4-80GB, 17, 81920, 0
4, NVIDIA A100-SXM4-80GB, 17, 81920, 0
```

Sandbox-external PyTorch check:

```bash
CUDA_VISIBLE_DEVICES=1,2,3,4 python - <<'PY'
import torch
print('cuda_available', torch.cuda.is_available())
print('device_count', torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))
PY
```

Observed result:

```text
cuda_available True
device_count 4
0 NVIDIA A100-SXM4-80GB
1 NVIDIA A100-SXM4-80GB
2 NVIDIA A100-SXM4-80GB
3 NVIDIA A100-SXM4-80GB
```

### Area Stats

Command:

```bash
python -m eyemae.compute_area_stats \
  --config configs/eyemae_cnn_512_12l.yaml \
  --split train \
  --out outputs/area_stats_fast_packed_seed42.json
```

Result:

```text
global median = 8.339022636413574
global mad = 0.22008037567138672
sampled valid frames = 34155020
```

Status: completed.

### Pretrain Launch

Command:

```bash
CUDA_VISIBLE_DEVICES=1,2,3,4 PYTHONUNBUFFERED=1 torchrun --standalone --nproc_per_node=4 \
  -m eyemae.train --config configs/eyemae_cnn_512_12l.yaml
```

Initial status:

```text
Using token-based dynamic batches: max_seq_tokens_per_gpu=60000 max_trials_per_gpu=256
step=0 loss=0.31945 xy=0.02723 area=1.09385 blink=0.70527 vel=0.02917 lr=1.00e-07
```

Warnings:

```text
Grad strides do not match bucket view strides.
```

Impact:

- This is a PyTorch DDP performance warning, not a correctness failure.
- Training continued after the warning.

### GPU Utilization Note

Observed during the first pretraining minutes:

```text
GPU1-4 memory used: about 37 GB each
GPU1-4 nvidia-smi utilization sample: 77%, 98%, 99%, 53%
GPU1-4 pmon SM sample: about 63%-68% per training process
```

Interpretation:

- GPU1-4 are actively used by the four DDP ranks.
- Utilization is not expected to stay pinned at 100% because each step includes
  mmap reading, CPU preprocessing/patching, dynamic batching, host-to-device
  transfer, DDP synchronization, and logging/validation gaps.
- The initial training loss decreased from step 0 to step 150, so this is not
  currently classified as a training failure.

### Visualization Font Warning

Observed at the first validation/visualization around step 1000:

```text
UserWarning: Glyph ... missing from font(s) DejaVu Sans.
```

Impact:

- This affects rendering of Chinese text in saved visualization figure labels.
- It does not affect model training, validation metrics, checkpoints, or numeric
  outputs.
- No action taken in the first-version run because it is not a correctness
  failure.

### Sandbox CPU Torchrun Smoke Failure

Command attempted inside the managed sandbox:

```bash
CUDA_VISIBLE_DEVICES= torchrun --standalone --nproc_per_node=2 \
  -m eyemae.finetune --config /tmp/eyemae_downstream_ddp_smoke/config.yaml
```

Failure:

```text
RendezvousConnectionError: The connection to the C10d store has failed.
client socket cannot be initialized to connect to localhost:0 (errno: 1 - Operation not permitted)
```

Impact:

- This is a sandbox networking restriction on local TCPStore rendezvous.
- It is not classified as a downstream code correctness failure.
- Full non-DDP tests pass after the downstream DDP/code changes.
- Actual distributed runs should be launched outside the sandbox, as done for
  the pretraining torchrun job.

### GPU Utilization Interpretation Update

Later sample during active pretraining:

```text
GPU1: 72% util, about 37.5 GB used
GPU2: 55% util, about 37.5 GB used
GPU3: 62% util, about 37.8 GB used
GPU4: 72% util, about 37.5 GB used
```

Continuous 10-second `nvidia-smi dmon` sample after step 2400:

```text
GPU1-4 average SM utilization: about 82%, 88%, 79%, 83%
```

Interpretation:

- The selected GPUs are active and hold the expected DDP model/batch memory.
- Single-point or UI-window average utilization below 100% is expected for this first-version input
  pipeline because each step includes mmap access, CPU collation/patching/mask
  work, host-to-device transfer, DDP synchronization, and validation/logging
  pauses.
- Step 2000 validation improved to `val/total_loss=0.0036134`, so this is not
  currently treated as a training failure.

## 2026-06-20 Pretrain Completion

Status:

- The continued fast-cache/no-offset pretrain run reached `max_steps=100000`
  and exited normally.
- Final validation ran at `global_step=99999`.
- `checkpoint_best.pt`, `checkpoint_last.pt`, and
  `checkpoint_step_00099999.pt` were written successfully.

Final monitored metric:

```text
val/masked_xy_rmse_deg = 0.4121278867332725
```

Exceptions:

- No new training crash or checkpoint-write failure occurred during the final
  continuation from step 14000 to step 99999.
- The remaining GPU-utilization issue is classified as a performance
  observation, not a correctness exception. Its likely causes and optional
  future-run controls are recorded in `docs/newdata_v3_training_log.md` and
  `docs/newdata_v3_implementation_changes.md`.

## 2026-06-20 Downstream Queue Supervisor Interruption

Status:

- The original downstream queue tool session became unavailable while four
  downstream jobs were still training. The training child processes continued
  as orphaned processes with `PPID=1`.
- The affected running jobs were:
  - GPU1 PID 2799000:
    `configs/downstream/pd_related_5class_scratch.yaml`
  - GPU3 PID 2799002:
    `configs/downstream/pd_related_5class_partial.yaml`
  - GPU4 PID 2799003:
    `configs/downstream/pd_related_5class_full.yaml`
  - GPU2 PID 2911035:
    `configs/downstream/epilepsy_binary_scratch.yaml`

Impact:

- The active training jobs did not crash; their logs continued to update.
- The queue supervisor was no longer available to launch later jobs after the
  active jobs finished.

Resolution:

- Added resume/attach support to `scripts/run_downstream_v3_queue.py`.
- Started a replacement supervisor by attaching the four running PIDs and
  reusing `outputs/downstream_v3_logs/run_20260620_full/status.json`.
- The refreshed status file now records 1 completed job, 4 attached running
  jobs, and 23 pending jobs.

## 2026-06-20 Downstream Fast Queue Restart

Status:

- User requested stopping the currently running baseline fine-tune jobs and
  rerunning downstream fine-tune with:
  - a new random subject-level split for PD 5-class,
  - the faster early-stopping plan for all downstream tasks.
- The old baseline supervisor and four running child jobs were terminated with
  `SIGTERM`:
  - supervisor PID 2968763,
  - PD scratch PID 2799000,
  - PD partial PID 2799002,
  - PD full PID 2799003,
  - epilepsy scratch PID 2911035.
- GPU1-4 were verified free after termination.

Incident:

- Initial detached launches with plain `setsid`/`nohup` did not persist in the
  managed execution environment. No training results were produced from those
  failed launch attempts.
- A stale old-style queue supervisor briefly appeared and launched jobs from
  the old queue context. It was terminated before being used as the formal
  fast-run queue.

Resolution:

- Verified that `tmux` sessions persist on this host.
- Launched the formal fast-run queue in tmux session
  `eyemae_downstream_v3_fast`.
- Confirmed `status.json` for
  `outputs/downstream_v3_fast_logs/run_20260620_random_pd_fast/` recorded
  0 completed, 4 running, and 24 pending jobs.
- Confirmed the first four running jobs are the intended PD random-split
  configs:
  - `pd_related_5class_random_seed20260620_fast_scratch.yaml`
  - `pd_related_5class_random_seed20260620_fast_linear_probe.yaml`
  - `pd_related_5class_random_seed20260620_fast_partial.yaml`
  - `pd_related_5class_random_seed20260620_fast_full.yaml`

## 2026-06-20 PD Binary Scope Addition During Active Queue

Status:

- User added a downstream PD binary task after the fast downstream queue had
  already started.
- The new task treats all four PD-related subtypes as the positive class:
  `pd_disease_label=0..3 -> health_label=1`.
- Generated view:
  `data-260606/extracted/eyemae_fast_dataset_v1/downstream/PD相关_binary_random_seed20260620/`.
- Generated configs:
  - `configs/downstream/pd_binary_random_seed20260620_fast_scratch.yaml`
  - `configs/downstream/pd_binary_random_seed20260620_fast_linear_probe.yaml`
  - `configs/downstream/pd_binary_random_seed20260620_fast_partial.yaml`
  - `configs/downstream/pd_binary_random_seed20260620_fast_full.yaml`

Impact:

- `configs/downstream/queue_pd_random_seed20260620_fast.txt` now contains 32
  jobs.
- The already-running tmux supervisor had read the old 28-job config list at
  process start, so editing the text file does not automatically enqueue the
  four new PD binary jobs.

Resolution:

- Updated `docs/current_goal_guardrail.md`,
  `docs/downstream_v3_plan.md`, `docs/newdata_v3_training_log.md`, and
  `scripts/audit_downstream_v3_goal.py` to require all 32 jobs for final
  completion.
- Added `scripts/wait_and_run_downstream_v3_queue.py` so a detached tail
  runner can wait for the current queue to become idle and then run the
  expanded 32-job queue with `--resume-status`.
- Started tmux session `eyemae_downstream_v3_tail32` at 2026-06-20 22:31 CST.
  Its first log line was
  `WAIT completed=1 running=4 pending=23 failures=0`, confirming that it is
  waiting and did not launch extra training while the current queue is active.

Resolution update:

- At 2026-06-21 04:13 CST, after confirming GPU2/GPU4 were idle, started an
  explicit missing-job補跑 session
  `eyemae_downstream_v3_missing_pd_binary_epilepsy_20260621`.
- The補跑 queue file is
  `configs/downstream/queue_missing_pd_binary_epilepsy_scratch_20260621.txt`
  and contains the four PD binary configs plus the interrupted
  `epilepsy_binary_scratch` config.
- Initial status:
  `0 completed / 2 running / 3 pending / 0 failures`; this should provide the
  missing first-version outputs without waiting for the original 28-job queue
  to fully idle.
- Stopped `eyemae_downstream_v3_tail32` and
  `eyemae_downstream_v3_tail32_allowfail` after the explicit补跑 queue started,
  because leaving them active could duplicate the same missing jobs when the
  main queue becomes idle.
- Stopped the old detox random watcher
  `eyemae_downstream_v3_detox_random_seed20260621` because it was waiting on
  the disabled clean-resume status path; the detox random configs/queue remain
  prepared for explicit launch later.

## 2026-06-21 Current-Best Test Rerun CUDA Visibility

Status:

- User requested a current-best PD 5-class test rerun for the four fine-tune
  modes.
- The first non-escalated sandbox attempt failed at model transfer with:
  `RuntimeError: No CUDA GPUs are available`.

Impact:

- No training process, checkpoint, or queue status was affected.
- The failed attempt did not overwrite the test rerun outputs.

Resolution:

- Added `scripts/rerun_pd_current_best_test.py` and ran the rerun with GPU0
  visibility enabled.
- Completed the four current-best test evaluations under
  `outputs/downstream_v3_fast_current_best_test_eval/pd_related_5class_random_seed20260620/`.

## 2026-06-21 Scratch Max-Epoch Cap During Active Queue

Status:

- User requested `scratch` downstream runs to use `max_epochs=30` to reduce
  training time.
- Updated the 8 active v3 fast scratch config files to
  `downstream_train.max_epochs: 30`.
- Updated `scripts/audit_downstream_v3_goal.py` so scratch configs are audited
  against `30`, while linear probe, partial, and full configs remain audited
  against `100`.

Impact:

- Pending scratch jobs will pick up the new YAML value when they launch.
- The already-running PD 5-class scratch and epilepsy scratch processes were
  launched before this edit, so they keep their old in-memory max-epoch value
  unless stopped/resumed.
- A one-off PD scratch current-best test rerun on GPU0 was interrupted to avoid
  spending extra compute while applying the new run-time cap. It did not affect
  the main queue, checkpoints, or final output directories.

Resolution update:

- The scratch cap was retracted after user noted that scratch may not have
  converged by 30 epochs and that capping scratch would be unfair.
- Restored scratch configs and audit expectation to `max_epochs=100`.
- `epilepsy_binary_scratch` had already been stopped after epoch 29 with
  `SIGTERM`; its 30-epoch best-checkpoint final evaluation was completed, then
  archived under
  `outputs/downstream_v3_fast_convergence/within30/epilepsy_binary/scratch/`.
- The archived 30-epoch `metrics_final.json` was moved out of
  `outputs/downstream_v3_fast/epilepsy_binary/scratch_full/metrics_final.json`
  and renamed in-place to `metrics_final_epoch29_cap_retracted.json`, so the
  main job remains incomplete and can be resumed from `checkpoint_last.pt` for
  the 100-epoch main result.
- `scripts/run_downstream_v3_queue.py` now auto-resumes from
  `checkpoint_last.pt` when rerunning incomplete outputs.

## 2026-06-21 Original Tail Queue Would Abort on Known Upstream SIGTERM

Status:

- The original tail tmux session `eyemae_downstream_v3_tail32` waits for the
  active queue to become idle, but its first implementation aborts if the
  upstream status contains any failures.
- The upstream status is expected to contain the documented
  `epilepsy_binary_scratch` `rc=-15` interruption from the retracted temporary
  scratch cap.

Impact:

- Without a failure-tolerant clean resume, the four added PD binary jobs and
  the interrupted epilepsy scratch job could remain incomplete even after the
  original 28-job queue stops.

Resolution:

- Added `--allow-upstream-failures` to
  `scripts/wait_and_run_downstream_v3_queue.py`.
- Started tmux session `eyemae_downstream_v3_tail32_allowfail`; it writes
  watcher/queue logs under
  `outputs/downstream_v3_fast_logs/run_20260620_random_pd_fast_clean_resume/`.
- Later replaced the tail watcher with an explicit missing-job queue to avoid
  duplicate launches after the main queue became idle.
- Current status as of 2026-06-21 04:39 CST:
  the original main queue is idle with `28 completed / 0 running / 0 pending /
  1 known failure`; the remaining first-version outputs are handled by
  `eyemae_downstream_v3_missing_pd_binary_epilepsy_20260621`.
- Final completion audit should treat the original queue status as provenance
  for the known `epilepsy_binary_scratch rc=-15`, and use the actual
  `metrics_final.json` coverage plus the explicit missing-job queue status to
  prove all 32 first-version jobs completed.

## 2026-06-21 MCI View Label Conflict Audit

Status:

- User flagged that current MCI AUROC values are implausibly low.
- Audited `downstream/MCI/{train,validation,test}.csv` and
  `downstream/MCI匹配后/{train,validation,test}.csv`.

Finding:

- `downstream/MCI/` has severe raw-subject/file-level label conflicts:
  218 raw subjects and 878 `(subject, filename basename)` keys have both
  `health_label=0` and `health_label=1`.
- These conflicts cover 66,308 / 91,433 rows, or 72.52% of the `MCI` view.
- 123 raw subjects also appear across multiple train/validation/test splits,
  affecting 37,536 / 91,433 rows.
- `ml_subject_id` hides this because it includes source/group fields; the same
  raw subject/file can be represented as different `ml_subject_id` values with
  opposite labels.
- `downstream/MCI匹配后/` does not show internal identity conflicts by itself:
  0 raw-subject conflicts, 0 `(subject, filename basename)` conflicts, and
  0 raw-subject split overlap.
- `MCI匹配后` is exactly the `source_dataset=匹配后` block inside the combined
  `MCI` source view by raw trial key, but its split assignment differs. The
  matched block is fully contained in the original `MCI` `实验组 + 对照组`
  raw-trial union with crossed labels: `匹配后/实验组` maps to original
  `对照组`, and `匹配后/对照组` maps to original `实验组`.
- The conflict is already present in source
  `cd_speed4_hard_blink_ml_ready_subjectkey_20260619/MCI/`; packed mmap
  conversion and fine-tune loading copied the source labels rather than
  introducing the issue.

Impact:

- Current `mci_binary` results should be treated as invalid/provisional and not
  compared against other disease tasks.
- The old `mci_matched_binary` split is internally non-overlapping, but its
  source labels are crossed relative to original MCI labels. Its prior metrics
  are therefore suspect/provisional until rerun with labels anchored to original
  MCI subjects.

Resolution:

- Added `docs/newdata_v3_mci_label_audit.md`.
- Before using `mci_binary`, rebuild `downstream/MCI/` so no raw subject or
  recording basename receives both binary labels and so split assignment is
  raw-subject held out.
- For MCI matched-row follow-up, ignore `MCI匹配后` source labels and relabel
  rows only from the original MCI subject label; rows without an original MCI
  raw-subject anchor must be excluded.

Follow-up all-view audit:

- Added `docs/newdata_v3_downstream_identity_audit.md`.
- The same raw identity/file audit found:
  - `MCI` is invalid: 218 raw-subject conflicts and 878 raw recording conflicts,
    covering 66,308 / 91,433 rows.
  - `AD` and `AD匹配后` have a smaller conflict centered on raw subject
    `GaoLianYing` across four files.
  - `PD相关`, `PD相关_random_seed20260620`, and
    `PD相关_binary_random_seed20260620` have one raw-subject label conflict
    (`ZhangMingSha`, four files, 320 rows) and substantial raw-subject/file split
    overlap caused by aggregate PD subtype/control reuse.
  - `MCI匹配后`, the four PD subtype matched views, `偏头痛`, `戒毒所`,
    `戒毒所_random_seed20260621`, and `癫痫` have 0 raw-subject conflicts,
    0 raw recording conflicts, and 0 raw-subject/file split overlap by this
    audit.
- Updated `docs/eyemae_fast_packed_dataset_adaptation_plan.md` to require raw
  subject/raw recording label-conflict and split-overlap checks in addition to
  `ml_subject_id` checks.

Follow-up correction executed:

- Added `scripts/prepare_mci_followup_finetune.py`.
- Generated `downstream/MCI_original_only_no_matched/`:
  58,279 rows, 383 subjects, 33,154 matched-copy rows removed, 0 raw-subject,
  raw-file, or raw-trial label conflicts, and 0 split overlap.
- Generated `downstream/MCI匹配后_random_seed20260621/`:
  33,154 rows, 218 subjects, 0 unmapped rows dropped, and 33,154 labels
  overwritten from the original MCI subject label. The generated view has 0
  raw-subject, raw-file, or raw-trial label conflicts and 0 split overlap.
- Clarified execution rule: `MCI匹配后` rows are samples only and never provide
  labels. A matched row is kept only if its raw `subject` exists in the
  original `MCI` subject-label anchor built after excluding
  `source_dataset=匹配后`; otherwise it must be dropped. The seed-20260621
  generation kept all matched rows because every matched raw subject had an
  original MCI anchor.
- Started waiting tmux session `eyemae_downstream_v3_mci_followup_seed20260621`
  so the eight corrected MCI follow-up fine-tune jobs run after the active
  missing-job queue frees GPU2/GPU4.

## 2026-06-21 06:06 CST: downstream `split_summary` config provenance mismatch

Finding:

- Several formal downstream and epoch1 YAML configs had
  `split.split_summary: pretrain/pretrain_split_summary.json` while their
  actual `data.train_index`/`val_index`/`test_index` pointed to
  `downstream/<view>/...csv`.
- This contradicted `docs/downstream_v3_plan.md`, which requires
  `downstream/<view>/split_summary.json` to exist and pass
  `no_subject_overlap == true`.

Impact:

- This was a configuration provenance/conformance issue, not a data loading
  issue for completed fine-tune runs. The fine-tune runtime audits downstream
  packed splits from `Path(data.train_index).parent / "split_summary.json"`.
- Pending `epilepsy_binary_scratch` had the stale field and was corrected before
  launch.

Resolution:

- Mechanically updated 52 files in `configs/downstream/` and
  `configs/downstream_epoch1/` so `split.split_summary` matches the parent view
  of `data.train_index`.
- Added an explicit `split.split_summary` consistency check to
  `scripts/audit_downstream_v3_goal.py`.
- Verified `bad_count=0`, `python -m py_compile scripts/audit_downstream_v3_goal.py`,
  and enhanced incomplete audit with `errors=[]`, `warnings=4`; the remaining
  warnings are only the four expected missing final metrics.

## 2026-06-21 06:40 CST: 2-GPU missing queue manager replacement interrupted active children

Finding:

- The missing-job queue was running with `--gpus 2,4`, leaving GPU1/GPU3 idle
  while `pd_binary/full` and `epilepsy_binary/scratch` were still pending.
- To satisfy the 4-GPU execution target, the old queue manager PID was stopped
  so a new manager could own the same status/log directory and launch pending
  jobs on GPU1/GPU3.
- Although only the queue manager PID was signaled, its active child training
  processes also exited in that tmux/session context.

Impact:

- No final metric was written for the interrupted `pd_binary/scratch` or
  `pd_binary/partial` jobs before the interruption.
- Both jobs had `checkpoint_last.pt`, so the interruption did not require a
  restart from scratch.

Resolution:

- Restarted the missing-job queue as
  `eyemae_downstream_v3_missing_pd_binary_epilepsy_4gpu_resume_20260621` with
  `--gpus 1,2,3,4`, `--resume-status`, and the same status/log directory.
- Confirmed active recovery:
  - `pd_binary/scratch` resumed on GPU1 from epoch 15 checkpoint.
  - `pd_binary/partial` resumed on GPU2 from epoch 16 checkpoint.
  - `pd_binary/full` started fresh on GPU3.
  - `epilepsy_binary/scratch` resumed on GPU4 from epoch 30 checkpoint.
- Replaced MCI follow-up and detox random waiters with 4-GPU waiters using the
  same status/log paths.
- Updated `scripts/run_downstream_v3_queue.py` to append to existing job logs
  instead of truncating them when resuming into an existing log directory.

## 2026-06-21 07:24 CST: Matched MCI label metadata also needed relabeling

Finding:

- The corrected matched MCI split already used original `MCI` subject anchors
  for the training label `health_label`, but the CSV field `source_group` still
  carried the old `MCI匹配后` group text.
- The model does not train from `source_group`, but the field is label-like and
  could mislead downstream users reading the generated dataset.

Resolution:

- Updated `scripts/prepare_mci_followup_finetune.py` so matched MCI rows kept
  by the original-MCI subject anchor overwrite both `health_label` and
  `source_group`; rows without an original-MCI subject anchor are still dropped.
- Regenerated corrected MCI data/configs before the corrected MCI queue
  launched. Verification on `MCI匹配后_random_seed20260621`:
  `rows=33154`, `subjects=218`, `outside_anchor_rows=0`,
  `health_label_mismatch_rows=0`, and `source_group_mismatch_rows=0`.
