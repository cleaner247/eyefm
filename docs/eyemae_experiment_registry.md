# EyeMAE Experiment Registry

本文件记录已经完成或正在运行的训练/测试版本。目标是后续可以清楚知道：

```text
哪个版本
用了哪个 config
用了哪个 split
输出写到哪里
怎么复现
怎么汇总
当前状态是什么
```

大文件仍然不进 Git：checkpoint、TensorBoard/event log、prediction CSV、`metrics_final.json`
等训练输出默认保存在本机 `outputs/` 下。

## Common Paths

Repo root:

```bash
cd /mnt/disk_sde/eyemae
```

Real data root:

```text
/mnt/disk_sde/data-260606/extracted/cd_no_cond2_structured_20260609
```

Pretraining checkpoint used by downstream runs:

```text
outputs/eyemae_cnn_512_12l_patch20_stimtoken/checkpoint_best.pt
```

Tracked area stats:

```text
outputs/area_stats_subject_heldout_seed42.json
```

## V0 Pretraining

Purpose:

```text
EyeMAE masked reconstruction pretraining on 1000 Hz eye-movement trials.
```

Plan:

```text
docs/eyemae_codex_plan_engineered_clean.md
```

Config:

```text
configs/eyemae_cnn_512_12l.yaml
```

Tracked split:

```text
splits/pretrain_subject_heldout_seed42/
```

Command:

```bash
CUDA_VISIBLE_DEVICES=0,1,2 torchrun --standalone --nproc_per_node=3 \
  -m eyemae.train \
  --config configs/eyemae_cnn_512_12l.yaml
```

Local output:

```text
outputs/eyemae_cnn_512_12l_patch20_stimtoken/
```

Reference metrics recorded in `README.md`:

```text
pretrain_test/masked_xy_rmse_deg = 0.468622
pretrain_test/total_loss = 0.00124888
pretrain_test/masked_blink_auc = 0.995496
```

Status:

```text
Complete locally.
```

## V1 Downstream Binary Holdout

Purpose:

```text
First downstream disease binary classification run with fixed subject-level
train/val/test splits.
```

Plan:

```text
docs/eyemae_downstream_finetune_codex_plan.md
```

Core code:

```text
src/eyemae/finetune.py
src/eyemae/downstream_data.py
src/eyemae/downstream_metrics.py
src/eyemae/make_downstream_splits.py
src/eyemae/summarize_downstream.py
```

Configs:

```text
configs/downstream/disease_binary_base.yaml
configs/downstream/disease_binary_scratch.yaml
configs/downstream/disease_binary_linear_probe.yaml
configs/downstream/disease_binary_partial.yaml
configs/downstream/disease_binary_full.yaml
```

Tracked split:

```text
splits/downstream_disease_binary_seed42/
```

Local output:

```text
outputs/downstream_disease_binary_seed42/
```

Diseases:

```text
AD
MCI
PD相关
偏头痛
戒毒所
癫痫
```

Modes:

```text
scratch
pretrained_linear_probe
pretrained_partial
pretrained_full
```

Launcher:

```bash
bash scripts/finetune_downstream_all.sh
```

Resume/check launcher:

```bash
SKIP_EXISTING=1 bash scripts/finetune_downstream_all.sh
```

Summary:

```bash
python -m eyemae.summarize_downstream \
  --output_root outputs/downstream_disease_binary_seed42
```

Expected final outputs:

```text
outputs/downstream_disease_binary_seed42/summary.csv
outputs/downstream_disease_binary_seed42/summary.json
outputs/downstream_disease_binary_seed42/<disease>/<mode>/metrics_final.json
outputs/downstream_disease_binary_seed42/<disease>/<mode>/trial_predictions_<split>.csv
outputs/downstream_disease_binary_seed42/<disease>/<mode>/subject_predictions_<split>.csv
outputs/downstream_disease_binary_seed42/<disease>/<mode>/confusion_matrix_<split>.json
```

Expected count:

```text
6 diseases * 4 modes = 24 metrics_final.json files
```

Status:

```text
Complete locally: 24 / 24 metrics_final.json files.
```

Training semantics:

```text
Input sample = one trial.
Training loss = trial-level BCE.
Subject balancing = each trial weight is 1 / train_trials_of_that_subject.
Subject metric = mean trial logit per base_subject_id during evaluation.
PD subtypes = merged into binary "患病" in this version.
```

## V2 PD/Addiction 5-Fold Downstream

Purpose:

```text
K-fold robustness check for PD相关 and 戒毒所 after the fixed holdout run
showed that full fine-tuning is not consistently best.
```

Config:

```text
configs/downstream/disease_binary_kfold_pd_addiction.yaml
```

Launcher:

```text
scripts/finetune_downstream_kfold_pd_addiction.sh
```

Tracked split:

```text
splits/downstream_disease_binary_kfold_seed42/
```

Local output:

```text
outputs/downstream_disease_binary_kfold_seed42/
```

Diseases:

```text
PD相关
戒毒所
```

Modes:

```text
scratch
pretrained_linear_probe
pretrained_partial
pretrained_full
```

Fold rule:

```text
strategy = subject_stratified_kfold
num_folds = 5
seed = 42
group_by_base_subject_id = true

For fold_i:
  test = bucket_i
  val = bucket_(i + 1 mod 5)
  train = remaining 3 buckets
```

Generated subject counts:

| Disease | Fold | Train Subjects | Val Subjects | Test Subjects |
|---|---:|---:|---:|---:|
| PD相关 | 0 | 447 | 149 | 149 |
| PD相关 | 1 | 447 | 149 | 149 |
| PD相关 | 2 | 447 | 149 | 149 |
| PD相关 | 3 | 447 | 149 | 149 |
| PD相关 | 4 | 447 | 149 | 149 |
| 戒毒所 | 0 | 74 | 25 | 25 |
| 戒毒所 | 1 | 74 | 25 | 25 |
| 戒毒所 | 2 | 74 | 25 | 25 |
| 戒毒所 | 3 | 75 | 24 | 25 |
| 戒毒所 | 4 | 75 | 25 | 24 |

Generate splits:

```bash
python -m eyemae.make_downstream_splits \
  --config configs/downstream/disease_binary_kfold_pd_addiction.yaml \
  --strategy subject_stratified_kfold \
  --num_folds 5 \
  --out_dir splits/downstream_disease_binary_kfold_seed42 \
  --disease "PD相关" \
  --disease "戒毒所"
```

Run all folds sequentially:

```bash
bash scripts/finetune_downstream_kfold_pd_addiction.sh
```

Run one fold:

```bash
FOLD_LIST=0 MAKE_SPLITS=0 SUMMARIZE=0 \
  bash scripts/finetune_downstream_kfold_pd_addiction.sh
```

Current local launch, started 2026-06-16 16:07 Asia/Shanghai:

```text
eyemae_kfold_fold0 -> GPU 0 -> FOLD_LIST=0
eyemae_kfold_fold1 -> GPU 1 -> FOLD_LIST=1
eyemae_kfold_fold2 -> GPU 2 -> FOLD_LIST=2
eyemae_kfold_fold3 -> GPU 3 -> FOLD_LIST=3
eyemae_kfold_fold4 -> GPU 4 -> FOLD_LIST=4
```

Runtime interventions:

```text
2026-06-16 18:30 Asia/Shanghai:
  fold_2/PD相关/pretrained_partial hit CUDA OOM on GPU 2 after
  fold_2/PD相关/scratch and fold_2/PD相关/pretrained_linear_probe had already
  produced metrics_final.json.

  Cause observed by nvidia-smi:
    physical GPU 2 was occupied by an external CUDA training process using
    nearly the full 80 GB device memory.

  Recovery:
    restarted fold 2 with SKIP_EXISTING=1 so completed fold_2 results were
    reused; set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; moved the
    recovered fold_2 session to physical GPU 3. The recovery session name is
    eyemae_kfold_fold2_recover.

  Post-recovery check:
    fold_2/PD相关/pretrained_partial progressed past epoch=0 step=1150 by
    2026-06-16 18:43 without a new OOM. At that point GPU 3 was shared by
    fold 3 and recovered fold 2.
```

Watcher session:

```text
eyemae_kfold_summary_wait
```

The watcher waits for all five fold sessions to finish, then runs the summary
command below.

Progress monitor session:

```text
eyemae_kfold_progress_monitor
```

The progress monitor writes training status every 5 minutes until all 40
`metrics_final.json` files exist.

Training logs:

```text
outputs/downstream_disease_binary_kfold_seed42/logs/fold_0.log
outputs/downstream_disease_binary_kfold_seed42/logs/fold_1.log
outputs/downstream_disease_binary_kfold_seed42/logs/fold_2.log
outputs/downstream_disease_binary_kfold_seed42/logs/fold_3.log
outputs/downstream_disease_binary_kfold_seed42/logs/fold_4.log
outputs/downstream_disease_binary_kfold_seed42/logs/progress_monitor.log
outputs/downstream_disease_binary_kfold_seed42/logs/summary_wait.log
```

Expected count:

```text
5 folds * 2 diseases * 4 modes = 40 metrics_final.json files
```

Manual summary:

```bash
python -m eyemae.summarize_downstream \
  --output_root outputs/downstream_disease_binary_kfold_seed42 \
  --disease "PD相关" \
  --disease "戒毒所" \
  --fold 0 --fold 1 --fold 2 --fold 3 --fold 4
```

Expected summaries:

```text
outputs/downstream_disease_binary_kfold_seed42/summary.csv
outputs/downstream_disease_binary_kfold_seed42/summary.json
outputs/downstream_disease_binary_kfold_seed42/summary_aggregate.csv
outputs/downstream_disease_binary_kfold_seed42/summary_aggregate.json
```

Progress checks:

```bash
tmux ls | grep eyemae_kfold

find outputs/downstream_disease_binary_kfold_seed42 \
  -name metrics_final.json | wc -l

tail -n 40 outputs/downstream_disease_binary_kfold_seed42/logs/fold_0.log
```

Status:

```text
Completed locally as of 2026-06-17 01:35 Asia/Shanghai.
Final metrics count = 40 / 40.
The watcher generated all expected summary files:
  outputs/downstream_disease_binary_kfold_seed42/summary.csv
  outputs/downstream_disease_binary_kfold_seed42/summary.json
  outputs/downstream_disease_binary_kfold_seed42/summary_aggregate.csv
  outputs/downstream_disease_binary_kfold_seed42/summary_aggregate.json
The final recovered mode was fold_2/戒毒所/pretrained_full.
```

Live recovery note:

```text
2026-06-16 18:30 Asia/Shanghai:
  fold_2 completed PD相关/scratch and PD相关/pretrained_linear_probe, then
  hit CUDA OOM when starting PD相关/pretrained_partial on GPU 2. At that
  moment GPU 2 was occupied by a non-project Python process, so the failed
  fold_2 tmux session exited before completing the remaining modes.

2026-06-16 18:37 Asia/Shanghai:
  fold_2 was restarted in tmux session eyemae_kfold_fold2_recover with
  SKIP_EXISTING=1, MAKE_SPLITS=0, SUMMARIZE=0, FOLD_LIST=2, and
  CUDA_VISIBLE_DEVICES=3. This skips already completed fold_2 outputs and
  resumes from PD相关/pretrained_partial. The recovery appends to
  outputs/downstream_disease_binary_kfold_seed42/logs/fold_2.log, so the
  earlier OOM traceback in that file is a handled historical event.
```

## V2 Code Changes

Files added or modified:

```text
configs/downstream/disease_binary_kfold_pd_addiction.yaml
scripts/finetune_downstream_kfold_pd_addiction.sh
src/eyemae/make_downstream_splits.py
src/eyemae/finetune.py
src/eyemae/summarize_downstream.py
tests/test_downstream.py
```

Functional changes:

```text
make_downstream_splits.py:
  added subject_stratified_kfold.

finetune.py:
  added --split_dir, --output_root, and --output_dir CLI overrides.

summarize_downstream.py:
  added --fold support and aggregate mean/std summaries.

tests/test_downstream.py:
  added k-fold split validation.
```

Validation:

```bash
python -m pytest tests/test_downstream.py
```

Result:

```text
7 passed
```

## V3 Extra-Disease 5-Fold Downstream

Purpose:

```text
Extend the k-fold downstream robustness check to the remaining disease tasks:
AD, MCI, 偏头痛, and 癫痫.
```

Config:

```text
configs/downstream/disease_binary_kfold_extra.yaml
```

Launchers/helpers:

```text
scripts/finetune_downstream_kfold.sh
scripts/finetune_downstream_kfold_wait_gpu.sh
scripts/wait_and_summarize_downstream_kfold.sh
```

Tracked split:

```text
splits/downstream_disease_binary_kfold_extra_seed42/
```

Local output:

```text
outputs/downstream_disease_binary_kfold_extra_seed42/
```

Diseases:

```text
AD
MCI
偏头痛
癫痫
```

Modes:

```text
scratch
pretrained_linear_probe
pretrained_partial
pretrained_full
```

Fold rule:

```text
strategy = subject_stratified_kfold
num_folds = 5
seed = 42
group_by_base_subject_id = true

For fold_i:
  test = bucket_i
  val = bucket_(i + 1 mod 5)
  train = remaining 3 buckets
```

Generated subject counts:

| Disease | Fold | Train Subjects | Val Subjects | Test Subjects |
|---|---:|---:|---:|---:|
| AD | 0 | 81 | 27 | 28 |
| AD | 1 | 82 | 27 | 27 |
| AD | 2 | 82 | 27 | 27 |
| AD | 3 | 82 | 27 | 27 |
| AD | 4 | 81 | 28 | 27 |
| MCI | 0 | 229 | 77 | 77 |
| MCI | 1 | 229 | 77 | 77 |
| MCI | 2 | 230 | 76 | 77 |
| MCI | 3 | 231 | 76 | 76 |
| MCI | 4 | 230 | 77 | 76 |
| 偏头痛 | 0 | 108 | 36 | 38 |
| 偏头痛 | 1 | 110 | 36 | 36 |
| 偏头痛 | 2 | 110 | 36 | 36 |
| 偏头痛 | 3 | 110 | 36 | 36 |
| 偏头痛 | 4 | 108 | 38 | 36 |
| 癫痫 | 0 | 485 | 162 | 163 |
| 癫痫 | 1 | 486 | 162 | 162 |
| 癫痫 | 2 | 486 | 162 | 162 |
| 癫痫 | 3 | 487 | 161 | 162 |
| 癫痫 | 4 | 486 | 163 | 161 |

Generate splits:

```bash
python -m eyemae.make_downstream_splits \
  --config configs/downstream/disease_binary_kfold_extra.yaml \
  --strategy subject_stratified_kfold \
  --num_folds 5 \
  --out_dir splits/downstream_disease_binary_kfold_extra_seed42 \
  --disease AD \
  --disease MCI \
  --disease 偏头痛 \
  --disease 癫痫
```

Run all folds sequentially:

```bash
bash scripts/finetune_downstream_kfold.sh
```

Run one fold:

```bash
FOLD_LIST=0 MAKE_SPLITS=0 SUMMARIZE=0 \
  bash scripts/finetune_downstream_kfold.sh
```

Current local launch, started 2026-06-16 23:23 Asia/Shanghai:

```text
eyemae_kfold_extra_fold0 -> GPU 0 -> FOLD_LIST=0
eyemae_kfold_extra_fold1 -> GPU 1 -> FOLD_LIST=1
eyemae_kfold_extra_fold2 -> GPU 2 -> FOLD_LIST=2
eyemae_kfold_extra_fold3 -> GPU 4 -> FOLD_LIST=3
eyemae_kfold_extra_fold4_wait -> waits for GPU 3 memory.used < 2000 MiB, then runs FOLD_LIST=4
```

Watcher session:

```text
eyemae_kfold_extra_summary_wait
```

The watcher waits for `eyemae_kfold_extra_fold0..3` and
`eyemae_kfold_extra_fold4_wait` to exit, then runs the summary command below.

Training logs:

```text
outputs/downstream_disease_binary_kfold_extra_seed42/logs/fold_0.log
outputs/downstream_disease_binary_kfold_extra_seed42/logs/fold_1.log
outputs/downstream_disease_binary_kfold_extra_seed42/logs/fold_2.log
outputs/downstream_disease_binary_kfold_extra_seed42/logs/fold_3.log
outputs/downstream_disease_binary_kfold_extra_seed42/logs/fold_4.log
outputs/downstream_disease_binary_kfold_extra_seed42/logs/summary_wait.log
```

Expected count:

```text
5 folds * 4 diseases * 4 modes = 80 metrics_final.json files
```

Manual summary:

```bash
python -m eyemae.summarize_downstream \
  --output_root outputs/downstream_disease_binary_kfold_extra_seed42 \
  --disease AD \
  --disease MCI \
  --disease 偏头痛 \
  --disease 癫痫 \
  --fold 0 --fold 1 --fold 2 --fold 3 --fold 4
```

Expected summaries:

```text
outputs/downstream_disease_binary_kfold_extra_seed42/summary.csv
outputs/downstream_disease_binary_kfold_extra_seed42/summary.json
outputs/downstream_disease_binary_kfold_extra_seed42/summary_aggregate.csv
outputs/downstream_disease_binary_kfold_extra_seed42/summary_aggregate.json
```

Status:

```text
Complete locally.

Started: 2026-06-16 23:23 Asia/Shanghai.
Finished: 2026-06-17 10:30 Asia/Shanghai.

Final count:
  metrics_final.json = 80 / 80
  summary.csv rows = 80
  summary_aggregate.csv rows = 16

Generated summaries:
  outputs/downstream_disease_binary_kfold_extra_seed42/summary.csv
  outputs/downstream_disease_binary_kfold_extra_seed42/summary.json
  outputs/downstream_disease_binary_kfold_extra_seed42/summary_aggregate.csv
  outputs/downstream_disease_binary_kfold_extra_seed42/summary_aggregate.json

Final session state:
  eyemae_kfold_extra_fold0 exited normally
  eyemae_kfold_extra_fold1 exited normally
  eyemae_kfold_extra_fold2 exited normally
  eyemae_kfold_extra_fold3 exited normally
  eyemae_kfold_extra_fold4_wait exited normally
  eyemae_kfold_extra_summary_wait exited after writing summaries

Filtered error count:
  fold_0.log = 0
  fold_1.log = 0
  fold_2.log = 0
  fold_3.log = 0
  fold_4.log = 0
  summary_wait.log = 0

Initial launch check:
  extra final count = 0 / 80
  fold_0..3 are running AD/scratch on GPU 0/1/2/4
  fold_4 is waiting for GPU 3 to become free
  filtered error count = 0

Update, 2026-06-17 01:14 Asia/Shanghai:
  eyemae_kfold_extra_fold4_wait observed GPU 3 memory.used=17 MiB and started
  fold_4/AD/scratch. This occurred during the V2 fold_2 transition from
  戒毒所/pretrained_partial to 戒毒所/pretrained_full, so V2 fold_2 final full
  and V3 fold_4/AD/scratch briefly shared GPU 3.

Update, 2026-06-17 01:38 Asia/Shanghai:
  V2 fold_2 finished without a new OOM, so the GPU 3 overlap ended.
  Peak observed GPU 3 memory during overlap was about 64 GiB / 80 GiB.
  V3 final count = 26 / 80.
  Running V3 tasks at that point:
    fold_0/MCI/pretrained_partial
    fold_1/MCI/pretrained_full
    fold_2/MCI/pretrained_partial
    fold_3/MCI/pretrained_full
    fold_4/AD/scratch
  filtered error count = 0
```

Final 5-fold aggregate results, subject-level test metrics:

| Disease | Mode | n | AUROC mean+-std | F1 mean+-std | Accuracy mean |
|---|---|---:|---:|---:|---:|
| AD | pretrained_full | 5 | 0.8531 +- 0.0904 | 0.8941 +- 0.0467 | 0.8312 |
| AD | pretrained_linear_probe | 5 | 0.8247 +- 0.1039 | 0.8747 +- 0.0337 | 0.8159 |
| AD | pretrained_partial | 5 | 0.8619 +- 0.0775 | 0.8839 +- 0.0596 | 0.8233 |
| AD | scratch | 5 | 0.7574 +- 0.1110 | 0.8005 +- 0.0543 | 0.7201 |
| MCI | pretrained_full | 5 | 0.7898 +- 0.0399 | 0.6357 +- 0.0803 | 0.7076 |
| MCI | pretrained_linear_probe | 5 | 0.7225 +- 0.0300 | 0.5687 +- 0.1624 | 0.6868 |
| MCI | pretrained_partial | 5 | 0.7526 +- 0.0504 | 0.5590 +- 0.1349 | 0.6970 |
| MCI | scratch | 5 | 0.7200 +- 0.0357 | 0.5926 +- 0.0556 | 0.6998 |
| 偏头痛 | pretrained_full | 5 | 0.7969 +- 0.0454 | 0.7371 +- 0.0435 | 0.7690 |
| 偏头痛 | pretrained_linear_probe | 5 | 0.7725 +- 0.0229 | 0.6991 +- 0.0507 | 0.7254 |
| 偏头痛 | pretrained_partial | 5 | 0.7522 +- 0.0287 | 0.5526 +- 0.1455 | 0.6646 |
| 偏头痛 | scratch | 5 | 0.7397 +- 0.0862 | 0.5914 +- 0.0827 | 0.6602 |
| 癫痫 | pretrained_full | 5 | 0.8576 +- 0.0194 | 0.7967 +- 0.0281 | 0.7777 |
| 癫痫 | pretrained_linear_probe | 5 | 0.8374 +- 0.0233 | 0.7624 +- 0.0267 | 0.7568 |
| 癫痫 | pretrained_partial | 5 | 0.8560 +- 0.0213 | 0.7692 +- 0.0389 | 0.7604 |
| 癫痫 | scratch | 5 | 0.7947 +- 0.0342 | 0.7343 +- 0.0383 | 0.7234 |

## V2+V3 Combined 5-Fold Summary

Purpose:

```text
Provide one final local summary across all six downstream binary tasks.

The original k-fold outputs are intentionally kept in two roots:
  outputs/downstream_disease_binary_kfold_seed42
    PD相关, 戒毒所
  outputs/downstream_disease_binary_kfold_extra_seed42
    AD, MCI, 偏头痛, 癫痫
```

Combined local output:

```text
outputs/downstream_disease_binary_kfold_all_seed42/summary.csv
outputs/downstream_disease_binary_kfold_all_seed42/summary.json
outputs/downstream_disease_binary_kfold_all_seed42/summary_aggregate.csv
outputs/downstream_disease_binary_kfold_all_seed42/summary_aggregate.json
outputs/downstream_disease_binary_kfold_all_seed42/summary_aggregate_subject_test_compact.csv
outputs/downstream_disease_binary_kfold_all_seed42/summary_aggregate_subject_test_compact.json
```

Status:

```text
Generated locally on 2026-06-17 14:44 Asia/Shanghai.
summary.csv rows = 120
summary_aggregate.csv rows = 24
summary_aggregate_subject_test_compact.csv rows = 24
All aggregate rows have n = 5.
Diseases included: AD, MCI, PD相关, 偏头痛, 戒毒所, 癫痫.
Modes included: scratch, pretrained_linear_probe, pretrained_partial, pretrained_full.
```

Final combined 5-fold aggregate results, subject-level test metrics:

| Disease | Mode | n | AUROC mean+-std | F1 mean+-std | Accuracy mean |
|---|---|---:|---:|---:|---:|
| PD相关 | pretrained_full | 5 | 0.9664 +- 0.0136 | 0.9062 +- 0.0272 | 0.8993 |
| PD相关 | pretrained_linear_probe | 5 | 0.9342 +- 0.0180 | 0.8615 +- 0.0210 | 0.8564 |
| PD相关 | pretrained_partial | 5 | 0.9629 +- 0.0160 | 0.9072 +- 0.0204 | 0.8993 |
| PD相关 | scratch | 5 | 0.9321 +- 0.0184 | 0.8698 +- 0.0174 | 0.8550 |
| 戒毒所 | pretrained_full | 5 | 0.9193 +- 0.0433 | 0.8550 +- 0.0798 | 0.8233 |
| 戒毒所 | pretrained_linear_probe | 5 | 0.8745 +- 0.0877 | 0.8167 +- 0.0893 | 0.7823 |
| 戒毒所 | pretrained_partial | 5 | 0.9062 +- 0.0459 | 0.8315 +- 0.0410 | 0.8070 |
| 戒毒所 | scratch | 5 | 0.7979 +- 0.1122 | 0.7497 +- 0.0842 | 0.7177 |
| AD | pretrained_full | 5 | 0.8531 +- 0.0904 | 0.8941 +- 0.0467 | 0.8312 |
| AD | pretrained_linear_probe | 5 | 0.8247 +- 0.1039 | 0.8747 +- 0.0337 | 0.8159 |
| AD | pretrained_partial | 5 | 0.8619 +- 0.0775 | 0.8839 +- 0.0596 | 0.8233 |
| AD | scratch | 5 | 0.7574 +- 0.1110 | 0.8005 +- 0.0543 | 0.7201 |
| MCI | pretrained_full | 5 | 0.7898 +- 0.0399 | 0.6357 +- 0.0803 | 0.7076 |
| MCI | pretrained_linear_probe | 5 | 0.7225 +- 0.0300 | 0.5687 +- 0.1624 | 0.6868 |
| MCI | pretrained_partial | 5 | 0.7526 +- 0.0504 | 0.5590 +- 0.1349 | 0.6970 |
| MCI | scratch | 5 | 0.7200 +- 0.0357 | 0.5926 +- 0.0556 | 0.6998 |
| 偏头痛 | pretrained_full | 5 | 0.7969 +- 0.0454 | 0.7371 +- 0.0435 | 0.7690 |
| 偏头痛 | pretrained_linear_probe | 5 | 0.7725 +- 0.0229 | 0.6991 +- 0.0507 | 0.7254 |
| 偏头痛 | pretrained_partial | 5 | 0.7522 +- 0.0287 | 0.5526 +- 0.1455 | 0.6646 |
| 偏头痛 | scratch | 5 | 0.7397 +- 0.0862 | 0.5914 +- 0.0827 | 0.6602 |
| 癫痫 | pretrained_full | 5 | 0.8576 +- 0.0194 | 0.7967 +- 0.0281 | 0.7777 |
| 癫痫 | pretrained_linear_probe | 5 | 0.8374 +- 0.0233 | 0.7624 +- 0.0267 | 0.7568 |
| 癫痫 | pretrained_partial | 5 | 0.8560 +- 0.0213 | 0.7692 +- 0.0389 | 0.7604 |
| 癫痫 | scratch | 5 | 0.7947 +- 0.0342 | 0.7343 +- 0.0383 | 0.7234 |

## Future Run Recording Rule

For every new training or evaluation version, add an entry here with:

```text
1. Purpose
2. Config path
3. Split path
4. Output root
5. Diseases/tasks
6. Modes/model variants
7. Launch command
8. Summary command
9. Expected metrics_final.json count
10. Current status
```

Checkpoint retention rule:

```text
For every completed downstream model variant, keep both files under:
  <output_root>/fold_<k>/<disease>/<mode>/checkpoint_best.pt
  <output_root>/fold_<k>/<disease>/<mode>/checkpoint_last.pt

The registry records the output root, split root, config, launch command,
expected count, and final summary files. The checkpoint tensors remain local
under outputs/ and are intentionally not pushed to GitHub unless a separate
artifact storage or Git LFS workflow is added.
```
