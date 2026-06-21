# EyeMAE 训练流程审计：Codex 执行计划

## 0. 审计目标

本任务用于审计已经运行或正在运行的 EyeMAE 训练流程，包括 pre-train 和 fine-tune 两部分。

本审计任务只读取现有训练输出，不重新训练模型，不修改模型结构，不重新划分 split，不修改 preprocessing。

需要回答的问题：

```text
1. pre-train:
   - 指标变化是否正常？
   - 大约多少 step 收敛？
   - 下一轮 max_steps 建议设多少？
   - 是否需要 early stopping？标准是什么？
   - 是否存在过拟合、训练不稳定、指标异常？

2. fine-tune:
   - fine-tune preprocessing / normalization 是否与 pre-train 一致？
   - fine-tune 是否正确加载预训练 encoder？
   - fine-tune 是否没有使用 MAE mask？
   - 每个任务、每个模式的 best epoch 是多少？
   - pretrained 是否优于 scratch？
```

如果某些日志或指标缺失，Codex 必须写明：

```text
MISSING: <file_or_metric>
```

不要猜测不存在的数据。

---

## 1. 输入路径

### 1.1 pre-train 输入

默认：

```text
pretrain_output_dir:
  outputs/eyemae_cnn_512_l12_patch20_stimtoken
  或 outputs/eyemae_cnn_512_12l_patch20_stimtoken

pretrain_config:
  configs/eyemae_cnn_512_12l.yaml

pretrain_data_dir:
  /mnt/disk_sde/data-260606/extracted/eyemae_fast_dataset_v1

pretrain_area_stats:
  outputs/area_stats_fast_packed_seed42.json
```

可能存在：

```text
resolved_config.yaml
checkpoint_best.pt
checkpoint_last.pt
checkpoint_step_*.pt
tensorboard/
validation_history.csv
train_history.csv
metrics.jsonl
validation_metrics.json
test_metrics.json
metrics_by_group.json
visualizations/
```

如果没有 `validation_history.csv`，尝试从 TensorBoard event files 或 `metrics.jsonl` 导出。

### 1.2 fine-tune 输入

默认输出目录模式：

```text
outputs/downstream/<task_name>/<mode_output>/
```

任务：

```text
pd_related_5class
epilepsy_binary
detox_binary
migraine_binary
ad_binary
mci_binary
mci_matched_binary
```

模式：

```text
pretrained_linear_probe
pretrained_partial
pretrained_full
scratch_full
```

每个 run 可能存在：

```text
resolved_config.yaml
run_summary.json
checkpoint_best.pt
checkpoint_last.pt
tensorboard/
validation_metrics.json
test_metrics.json
trial_predictions_validation.csv
subject_predictions_validation.csv
trial_predictions_test.csv
subject_predictions_test.csv
confusion_matrix_validation.json
confusion_matrix_test.json
epoch_history.csv
```

---

## 2. 输出文件

审计结果输出到：

```text
outputs/audit/
```

必须生成：

```text
outputs/audit/audit_report.md
outputs/audit/audit_findings.json

outputs/audit/pretrain_convergence_audit.md
outputs/audit/pretrain_convergence_summary.csv
outputs/audit/pretrain_convergence_summary.json

outputs/audit/finetune_preprocess_audit.md
outputs/audit/finetune_preprocess_matrix.csv

outputs/audit/finetune_best_epoch_audit.md
outputs/audit/finetune_best_epoch_summary.csv
outputs/audit/finetune_best_epoch_summary.json

outputs/audit/downstream_result_summary.csv
outputs/audit/preprocess_sample_check.csv
```

---

## 3. pre-train 审计

### 3.1 配置检查

读取：

```text
pretrain_config
pretrain_output_dir/resolved_config.yaml, if exists
```

检查这些字段：

```yaml
data.format: packed_mmap
data.data_dir: /mnt/disk_sde/data-260606/extracted/eyemae_fast_dataset_v1
data.train_index: pretrain/pretrain_train.csv
data.val_index: pretrain/pretrain_validation.csv
data.test_index: pretrain/pretrain_test.csv

split.split_summary: pretrain/pretrain_split_summary.json

model.pretrain_style: bert_masked_reconstruction
model.sequence_format: stim_eye_triplet_no_cls
model.use_cls: false
model.use_stim_tokens: true
model.broadcast_stim_to_eye: false

patch.samples: 20
patch.stride: 20

stim.stim_dim: 4
stim.use_last_stim_xy: false
stim.use_goal_xy: false

input.content_dim: 4
input.quality_dim: 1
input.stim_dim: 4

area.stats_path: outputs/area_stats_fast_packed_seed42.json
area.stats_source: pretrain_train_only

train.max_steps
train.val_every_steps
checkpoint.monitor
checkpoint.mode
```

CRITICAL 条件：

```text
model.pretrain_style != bert_masked_reconstruction
stim order 不是 [fix_on, stim_on, stim_x_norm, stim_y_norm]
area stats 使用 validation/test 统计
使用 pretrain_val.csv
使用 pretrain/split_summary.json
data.format 不是 packed_mmap
```

### 3.2 指标历史提取

整理成：

```text
outputs/audit/pretrain_validation_history_extracted.csv
```

至少尝试提取这些列：

```csv
step,lr,
train_total_loss,train_xy_loss,train_area_loss,train_blink_loss,train_velocity_loss,
val_total_loss,val_xy_loss,val_area_loss,val_blink_loss,val_velocity_loss,
val_masked_xy_rmse_deg,val_masked_x_rmse_deg,val_masked_y_rmse_deg,
val_masked_area_mae,val_masked_blink_auc,val_masked_velocity_rmse_deg_per_ms,
val_long_span_masked_xy_rmse_deg,val_short_span_masked_xy_rmse_deg,val_random_masked_xy_rmse_deg
```

缺失列保留为空，并在报告中列出：

```text
MISSING_METRIC: <metric_name>
```

### 3.3 主指标

主指标：

```text
val_masked_xy_rmse_deg
```

fallback：

```text
val_total_loss
```

辅助指标：

```text
val_long_span_masked_xy_rmse_deg
val_masked_blink_auc
val_velocity_loss
train_total_loss vs val_total_loss gap
```

### 3.4 收敛判断规则

设：

```text
M(step) = val_masked_xy_rmse_deg
```

如果没有该指标，则：

```text
M(step) = val_total_loss
```

计算：

```text
best_step
best_value
last_step
last_value
relative_improvement_last_10_validations
relative_improvement_last_15_validations
```

定义：

```text
relative_improvement_last_10 =
  (best_before_last_10 - best_in_last_10) / best_before_last_10

relative_improvement_last_15 =
  (best_before_last_15 - best_in_last_15) / best_before_last_15
```

判断：

```text
NOT_CONVERGED:
  最近 10 或 15 个 validation 仍有 >= 0.5% 改善

NEAR_PLATEAU:
  最近 10 个 validation 改善在 0.2% - 0.5%

CONVERGED:
  最近 15 个 validation 改善 < 0.2%
  且 long_span 指标也没有明显改善

OVERFITTING:
  train_total_loss 继续下降
  但 val_masked_xy_rmse_deg 或 val_long_span_masked_xy_rmse_deg 持平/变差
```

### 3.5 建议 max_steps

输出：

```json
{
  "current_max_steps": 100000,
  "best_step": 0,
  "last_step": 0,
  "convergence_status": "CONVERGED|NEAR_PLATEAU|NOT_CONVERGED|OVERFITTING",
  "recommended_max_steps_next_run": 0,
  "rationale": "..."
}
```

建议规则：

```text
如果 best_step < 0.5 * current_max_steps 且 plateau_15:
  recommended_max_steps_next_run = ceil((best_step + 15000) / 10000) * 10000

如果 best_step 接近 last_step 且最近仍改善:
  recommended_max_steps_next_run = current_max_steps * 1.25 或 1.5

如果 OVERFITTING:
  recommended_max_steps_next_run = best_step + 5000 到 10000
  并建议使用 checkpoint_best 而非 checkpoint_last

如果 NEAR_PLATEAU:
  recommended_max_steps_next_run = last_step 或 last_step + 10000

如果 NOT_CONVERGED:
  recommended_max_steps_next_run = last_step + 20000 到 50000
```

### 3.6 推荐 pre-train early stopping

默认建议：

```yaml
pretrain_early_stopping:
  monitor: val/masked_xy_rmse_deg
  mode: min
  min_steps: 50000
  patience_validations: 15
  min_delta_relative: 0.002
```

说明：

```text
val_every_steps=1000 时，patience_validations=15 相当于约 15k steps。
min_delta_relative=0.002 表示至少需要 0.2% 相对改善才算有效改善。
```

若曲线噪声很大，建议：

```yaml
patience_validations: 20
min_delta_relative: 0.001
```

### 3.7 pre-train summary 表

生成：

```text
outputs/audit/pretrain_convergence_summary.csv
```

列：

```csv
monitor_metric,best_step,best_value,last_step,last_value,
relative_improvement_last_10,relative_improvement_last_15,
long_span_best_step,long_span_best_value,
blink_auc_best_step,blink_auc_best_value,
convergence_status,recommended_max_steps_next_run,
recommended_early_stop_min_steps,
recommended_early_stop_patience_validations,
recommended_early_stop_min_delta_relative
```

---

## 4. fine-tune preprocessing 一致性审计

### 4.1 审计对象

对 7 个任务 × 4 个模式读取：

```text
resolved_config.yaml
run_summary.json
```

如果 `resolved_config.yaml` 缺失，读取原始 config，并标记：

```text
WARNING: resolved_config.yaml missing
```

### 4.2 需要与 pre-train 一致的字段

与 pretrain resolved config 比较：

```text
normalization.x_clip_deg
normalization.y_clip_deg

area.stats_path
area.use_log1p
area.robust_zscore_by
area.clip
area.eps
area.fallback_to_global

stim.stim_dim
stim.use_fix_on
stim.use_stim_on
stim.use_stim_xy
stim.use_last_stim_xy
stim.use_goal_xy

input.content_dim
input.quality_dim
input.stim_dim
input.num_eyes

patch.samples
patch.stride

attention.min_nonmissing_frac_for_eye_token

model.sequence_format
model.pretrain_style
model.tokenizer
model.d_model
model.n_layers
model.n_heads
model.ffn_hidden
model.max_patches
model.use_cls
model.use_stim_tokens
model.broadcast_stim_to_eye
model.use_token_type_embedding
model.fusion
```

允许不同：

```text
train.*
finetune.*
pooling.*
head.*
label.*
class_weighting.*
metrics.*
pretrained.checkpoint
```

### 4.3 关键一致性要求

必须检查：

```text
fine-tune patch.samples == pretrain patch.samples == 20
fine-tune patch.stride == pretrain patch.stride == 20
fine-tune area.stats_path == pretrain area.stats_path
fine-tune stim order == [fix_on, stim_on, stim_x_norm, stim_y_norm]
fine-tune model.sequence_format == stim_eye_triplet_no_cls
fine-tune model.pretrain_style == bert_masked_reconstruction
fine-tune mae_mask 不应出现或应为 None / all False
fine-tune 不加载 reconstruction head
scratch baseline 也使用同一份 area stats
```

### 4.4 preprocessing audit 输出

生成：

```text
outputs/audit/finetune_preprocess_matrix.csv
```

列：

```csv
task_name,mode,config_path,pretrained_checkpoint,area_stats_path,
area_stats_matches_pretrain,patch_matches_pretrain,stim_config_matches_pretrain,
model_encoder_config_matches_pretrain,uses_mae_mask_in_finetune,
loads_reconstruction_head,status,notes
```

状态：

```text
PASS
WARNING
FAIL
MISSING
```

FAIL 条件：

```text
area_stats_path 与 pretrain 不同，且不是明确 ablation
patch size 不同，且不是明确 ablation
fine-tune 使用 mae_mask
fine-tune 加载 reconstruction head 作为 classification head
model encoder config 与 checkpoint 不兼容
```

---

## 5. fine-tune best epoch 审计

### 5.1 文件读取优先级

对每个 task/mode 查找：

```text
1. epoch_history.csv
2. checkpoint_best.pt metadata + validation_metrics.json
3. TensorBoard event files
4. train.log
```

如果无法找到 best epoch，记录：

```text
best_epoch = MISSING
```

### 5.2 二分类 best metric

任务：

```text
epilepsy_binary
detox_binary
migraine_binary
ad_binary
mci_binary
mci_matched_binary
```

默认：

```text
validation/subject_auroc
```

fallback：

```text
validation/subject_balanced_accuracy
```

### 5.3 五分类 best metric

任务：

```text
pd_related_5class
```

默认：

```text
validation/subject_macro_auroc_ovr
```

fallback：

```text
validation/subject_balanced_accuracy
```

### 5.4 early stopping 检查

默认：

```yaml
max_epochs: 100
min_epochs_before_early_stopping: 50
early_stopping_patience: 20
```

检查：

```text
best checkpoint 是否按 validation 主指标保存
stopped_epoch 是否小于 50
若 stopped_epoch < 50，标记 WARNING 或 FAIL
若 best_epoch 接近 100，标记 LATE_BEST
```

### 5.5 best epoch 状态

```text
HEALTHY:
  best_epoch 在 20-80 之间
  validation/test gap 合理

EARLY_BEST:
  best_epoch < 10

LATE_BEST:
  best_epoch > 0.8 * max_epochs

OVERFITTING:
  train 指标继续改善但 validation 变差

UNSTABLE:
  validation 主指标大幅震荡

MISSING:
  缺少指标或日志
```

### 5.6 输出表

生成：

```text
outputs/audit/finetune_best_epoch_summary.csv
```

列：

```csv
task_name,mode,pretrain_style,pretrained_checkpoint,best_epoch,stopped_epoch,max_epochs,
early_stop_triggered,best_metric_name,best_metric_value,test_metric_name,test_metric_value,
validation_subject_auroc,test_subject_auroc,
validation_subject_auprc,test_subject_auprc,
validation_subject_balanced_accuracy,test_subject_balanced_accuracy,
validation_subject_macro_auroc_ovr,test_subject_macro_auroc_ovr,
validation_subject_macro_f1,test_subject_macro_f1,status,notes
```

---

## 6. downstream 效果审计

### 6.1 四模式比较

每个任务比较：

```text
scratch_full
pretrained_linear_probe
pretrained_partial
pretrained_full
```

生成：

```text
outputs/audit/downstream_result_summary.csv
```

列：

```csv
task_name,metric,scratch_full,pretrained_linear_probe,pretrained_partial,pretrained_full,
best_mode_by_validation,best_mode_test_metric,
pretrained_linear_probe_minus_scratch,pretrained_full_minus_scratch,conclusion
```

### 6.2 判断预训练是否有用

```text
PRETRAIN_HELPFUL:
  pretrained_full 或 pretrained_partial 明显优于 scratch

REPRESENTATION_USEFUL:
  pretrained_linear_probe 明显优于 scratch

NO_CLEAR_GAIN:
  pretrained 与 scratch 接近

NEGATIVE_TRANSFER:
  pretrained_full 明显低于 scratch
```

阈值建议：

```text
AUROC / macro AUROC 提升 >= 0.02:
  轻度增益

AUROC / macro AUROC 提升 >= 0.05:
  明显增益

差异 < 0.01:
  接近
```

如果有 bootstrap CI，则用 CI 判断；没有 CI 时只做描述性结论。

---

## 7. split 与数据审计

### 7.1 pretrain split

检查：

```text
pretrain/pretrain_train.csv
pretrain/pretrain_validation.csv
pretrain/pretrain_test.csv
```

确认：

```text
ml_subject_id 不跨 split
global_trial_id 唯一
task_id 只包含 0/1/2/3
frame_length > 0
max_patches <= model.max_patches
```

输出：

```text
outputs/audit/pretrain_split_audit.json
```

### 7.2 downstream split

对每个 view 检查：

```text
downstream/<view>/train.csv
downstream/<view>/validation.csv
downstream/<view>/test.csv
downstream/<view>/split_summary.json
```

确认：

```text
ml_subject_id 不跨 train/validation/test
二分类任务 health_label 只包含 0/1
PD 5分类 label 映射后只包含 0..4
validation/test 至少记录类别缺失情况
```

输出：

```text
outputs/audit/downstream_split_audit.csv
```

---

## 8. preprocessing 抽样验证

实际抽样若干 trial，运行 preprocessing + patching，验证：

```text
stim_patch 顺序为 [fix_on, stim_on, stim_x_norm, stim_y_norm]
stim_on=0 时 stim_x_norm/stim_y_norm 为0
missing/blink 时 x/y/area 为0
quality[...,0] 只表示 missing
content[...,3] 表示 blink
patch.samples=20
seq_len=3*N
fine-tune forward_features 不创建 mae_mask
```

抽样要求：

```text
pretrain train/validation/test 各至少 10 个 trial
downstream 每个 task 的 train/validation/test 各至少 5 个 trial
尽量包含 source_suffix D/L/R
尽量包含 blink/missing
```

输出：

```text
outputs/audit/preprocess_sample_check.csv
```

列：

```csv
split,task_name_or_pretrain,global_trial_id,ml_subject_id,N,
stim_order_ok,stim_off_xy_zero_ok,blink_zero_xy_area_ok,missing_zero_xy_area_ok,
quality_missing_only_ok,seq_len_ok,notes
```

---

## 9. 最终报告格式

生成：

```text
outputs/audit/audit_report.md
```

结构：

```markdown
# EyeMAE Training Audit Report

## 1. Executive Summary
- Pretrain convergence: ...
- Recommended max_steps: ...
- Recommended early stopping: ...
- Fine-tune preprocessing consistency: PASS/WARNING/FAIL
- Fine-tune best epoch summary: ...
- Downstream pretrained vs scratch: ...

## 2. Pretrain Convergence
- best step
- last step
- convergence status
- curves summary
- missing metrics

## 3. Pretrain Suggested Schedule
- max_steps recommendation
- early stopping recommendation
- reasons

## 4. Fine-tune Preprocessing Consistency
- config field comparison
- area stats check
- stim order check
- MAE mask check
- reconstruction head loading check

## 5. Fine-tune Best Epochs
- table of task/mode/best_epoch/stopped_epoch/test metric
- early/late/unstable runs

## 6. Downstream Results
- 7 task summary
- pretrained vs scratch
- linear probe vs scratch
- partial/full comparison

## 7. Split and Data Audit
- pretrain subject overlap
- downstream subject overlap
- label distributions
- class missing warnings

## 8. Action Items
- critical fixes
- recommended reruns
- optional ablations
```

---

## 10. Critical / Warning 分级

### 10.1 Critical

```text
fine-tune preprocessing 与 pretrain 不一致
fine-tune 使用不同 area.stats_path 且不是明确 ablation
fine-tune 使用 MAE mask
fine-tune 加载 reconstruction head 作为 classification head
pretrain split 存在 ml_subject_id overlap
downstream split 存在 ml_subject_id overlap
stim order 不正确
pretrain 或 downstream 使用 test set 选择 checkpoint
max_patches 不足但训练静默截断
```

### 10.2 Warning

```text
best_epoch < 10
best_epoch 接近 max_epochs
validation/test gap 很大
validation 指标剧烈震荡
AUROC/AUPRC 因类别不足为 NaN
TensorBoard/epoch_history 缺失，只能从 checkpoint 推断
部分 group metrics 缺失
```

---

## 11. Codex 入口脚本

如果没有现成脚本，Codex 需要新增：

```text
src/eyemae/audit_training.py
```

该脚本只读文件，不训练。

建议命令：

```bash
python -m eyemae.audit_training \
  --pretrain_output_dir outputs/eyemae_cnn_512_l12_patch20_stimtoken \
  --pretrain_config configs/eyemae_cnn_512_12l.yaml \
  --downstream_root outputs/downstream \
  --data_dir /mnt/disk_sde/data-260606/extracted/eyemae_fast_dataset_v1 \
  --out_dir outputs/audit
```

---

## 12. 验收标准

Codex 完成后必须满足：

```text
outputs/audit/audit_report.md 存在
outputs/audit/pretrain_convergence_summary.csv 存在
outputs/audit/finetune_preprocess_matrix.csv 存在
outputs/audit/finetune_best_epoch_summary.csv 存在
outputs/audit/downstream_result_summary.csv 存在
outputs/audit/audit_findings.json 存在
```

如果日志完整：

```text
能给出 pretrain best_step、收敛状态、建议 max_steps
能给出每个 fine-tune run 的 best_epoch
能比较 scratch vs pretrained
```

如果日志不完整：

```text
明确列出 missing 文件和 missing metric
不编造 best_epoch 或收敛结论
```
