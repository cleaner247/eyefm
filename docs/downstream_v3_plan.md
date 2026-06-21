# EyeBERT-style Encoder Packed-MMap 下游微调（EyeMAE 项目）：Codex 工程实现计划

## 0. 目标

在已完成第一版 EyeBERT-style 预训练的基础上，使用已经生成好的 packed 数据集完成健康/疾病下游微调与评估。

正式数据集已经存在：

```text
/mnt/disk_sde/data-260606/extracted/eyemae_fast_dataset_v1
```

本计划不重新生成 packed 数据集，不重新划分 downstream split。Codex 本轮只需要实现：

```text
1. PackedDownstreamDataset
2. 下游现成 split / audit 检查
3. 预训练 encoder 加载
4. pooling + classification head
5. linear probe / partial / full / scratch 四种模式
6. subject-level aggregation 与 metrics
7. finetune / evaluate_downstream
```

下游任务：

```text
8个正式下游任务：
  1. PD相关：5分类
  2. PD相关：二分类，四个 PD 相关亚型全部合并为患病
  3. 癫痫：二分类
  4. 戒毒所：二分类
  5. 偏头痛：二分类
  6. AD：二分类
  7. MCI 原始样本：二分类，剔除 `source_dataset=匹配后`
  8. MCI 匹配后样本：二分类，只保留原始 MCI 中存在的 raw subject，且 label 必须来自原始 MCI subject anchor
```

每个任务必须比较：

```text
1. Scratch full train
2. Pretrained linear probe
3. Pretrained partial fine-tune
4. Pretrained full fine-tune
```

---

## 0.1 命名约定

下游默认加载的预训练 checkpoint 来自第一版：

```yaml
model:
  pretrain_style: bert_masked_reconstruction
```

也就是说，下游主线默认使用的是 BERT-style encoder。真正的 asymmetric MAE-style encoder 只来自预训练 ablation：

```yaml
model:
  pretrain_style: asymmetric_mae
```

如果后续要比较 asymmetric MAE-style 预训练 checkpoint，必须在结果表中单独标注：

```text
pretrain_style = asymmetric_mae
```

不要把第一版 BERT-style checkpoint 和 asymmetric MAE-style checkpoint 混称为同一个预训练设置。


# 1. 总体原则

## 1.1 预训练 split 和下游 split 分开

预训练 split：

```text
pretrain/pretrain_train.csv
pretrain/pretrain_validation.csv
pretrain/pretrain_test.csv
```

下游 split：

```text
downstream/<view>/train.csv
downstream/<view>/validation.csv
downstream/<view>/test.csv
```

不要复用预训练 split 作为疾病任务 split。

## 1.2 不重新生成下游 split

正式 packed 数据集已经提供下游 split：

```text
/mnt/disk_sde/data-260606/extracted/eyemae_fast_dataset_v1/downstream/<view>/train.csv
/mnt/disk_sde/data-260606/extracted/eyemae_fast_dataset_v1/downstream/<view>/validation.csv
/mnt/disk_sde/data-260606/extracted/eyemae_fast_dataset_v1/downstream/<view>/test.csv
```

fine-tune 代码必须直接读取这些 CSV。

不要实现或调用：

```text
make_downstream_splits.py
重新 stratified split
K-fold split
```

## 1.3 主指标必须 subject-level

所有主指标按：

```text
ml_subject_id
```

聚合。

trial-level metric 只能作为辅助分析。

## 1.4 下游微调不做 MAE mask

下游时：

```text
mae_mask = None 或全False
不遮住 eye token
不使用 reconstruction head
只使用 encoder hidden states 做分类
```

---

# 2. 项目结构增量

在预训练项目基础上新增：

```text
src/eyemae/
  downstream_data.py
  finetune.py
  evaluate_downstream.py
  downstream_metrics.py
  pooling.py
  checkpoint_utils.py

configs/downstream/
  <task_name>_linear_probe.yaml
  <task_name>_partial.yaml
  <task_name>_full.yaml
  <task_name>_scratch.yaml

tests/
  test_downstream_labels.py
  test_downstream_dataset.py
  test_downstream_pooling.py
  test_finetune_freeze_modes.py
  test_subject_aggregation.py
  test_downstream_metrics.py
  test_pretrained_loading.py
```

不要新增：

```text
make_downstream_splits.py
packed converter
```

下游必须复用预训练工程中的：

```text
PackedTrialStore
preprocess.py
patching.py
batching.py
model.forward_features
area stats loader
```

---

# 3. 数据接口

正式下游数据集使用：

```text
data.format = packed_mmap
data.data_dir = /mnt/disk_sde/data-260606/extracted/eyemae_fast_dataset_v1
```

根目录必须存在：

```text
dataset_manifest.json
audit_summary.json
columns.json
label_maps.json
trials.csv
subjects.csv
shards/
pretrain/
downstream/
```

下游任务直接读取 index：

```text
downstream/<view>/train.csv
downstream/<view>/validation.csv
downstream/<view>/test.csv
downstream/<view>/split_summary.json
```

每行至少包含：

```text
global_trial_id
shard_id
local_trial_index
frame_offset
frame_length
ml_subject_id
task_id
health_label
```

`pd_disease_label` 只在 `pd_related_5class` / `downstream/PD相关/` 任务中必需。PD 二分类任务可保留该列用于统计四个亚型来源，但训练标签只使用 `health_label`。其他二分类任务不要求该列存在；如果二分类 view 中存在该列，可以忽略。

推荐包含：

```text
source_suffix
trial_id
view
```

## 3.1 shard 读取

与预训练完全一致：

```python
shard_dir = data_root / "shards" / shard_id

X = np.load(shard_dir / "X_data.npy", mmap_mode="r")
Y = np.load(shard_dir / "y_frame.npy", mmap_mode="r")
offsets = np.load(shard_dir / "X_offsets.npy", mmap_mode="r")
lengths = np.load(shard_dir / "X_lengths.npy", mmap_mode="r")
```

权威切片来自 CSV：

```python
start = row["frame_offset"]
length = row["frame_length"]
end = start + length
```

offsets/lengths 只用于校验：

```python
assert start == offsets[local_trial_index]
assert length == lengths[local_trial_index]
```

## 3.2 trial dict 映射

内部 trial dict：

```python
trial = {
    "eye": FloatTensor[T, 8],
    "task_id": LongTensor[],
    "fix_on": FloatTensor[T],
    "stim": FloatTensor[T, 3],
    "subject_id": str,
    "ml_subject_id": str,
    "trial_id": str,
    "global_trial_id": str,
    "subject_eye_availability": str,
}
```

列映射：

```python
eye[:, 0] = X[:, 0]   # left_x
eye[:, 1] = X[:, 1]   # left_y
eye[:, 2] = X[:, 2]   # left_s
eye[:, 3] = Y[:, 0]   # left_qc_label

eye[:, 4] = X[:, 3]   # right_x
eye[:, 5] = X[:, 4]   # right_y
eye[:, 6] = X[:, 5]   # right_s
eye[:, 7] = Y[:, 1]   # right_qc_label

fix_on = X[:, 9]

stim[:, 0] = X[:, 8]  # stim_on
stim[:, 1] = X[:, 6]  # stim_x
stim[:, 2] = X[:, 7]  # stim_y

task_id = row["task_id"]
trial_id = row["global_trial_id"]
global_trial_id = row["global_trial_id"]
ml_subject_id = row["ml_subject_id"]
```

内部 `stim_patch` 顺序必须为：

```text
[fix_on, stim_on, stim_x_norm, stim_y_norm]
```

---

# 4. 下游任务定义

正式 8 个任务：

| task_name | view/index目录 | type | label |
|---|---|---|---|
| `pd_related_5class` | `downstream/PD相关_random_seed20260620/` | multiclass, 5 classes | `0=control`, `1=帕金森病`, `2=震颤`, `3=特发性震颤`, `4=运动障碍` |
| `pd_binary` | `downstream/PD相关_binary_random_seed20260620/` | binary | `health_label`; `0=control`, `1=任一 PD 相关亚型患病` |
| `epilepsy_binary` | `downstream/癫痫/` | binary | `health_label` |
| `detox_binary` | `downstream/戒毒所/` | binary | `health_label` |
| `migraine_binary` | `downstream/偏头痛/` | binary | `health_label` |
| `ad_binary` | `downstream/AD_dedup_rawsubject/` | binary | `health_label`; removes duplicated `AD/匹配后/实验组` rows covered by `AD组/患病` and drops the conflicting `GaoLianYing` matched-control rows |
| `mci_original_only_binary` | `downstream/MCI_original_only_no_matched/` | binary | `health_label`; only original `MCI` rows after removing `source_dataset=匹配后` |
| `mci_matched_binary_random_seed20260621` | `downstream/MCI匹配后_random_seed20260621/` | binary | `health_label`; matched rows are samples only; keep a row only when its raw `subject` exists in the original `MCI` subject anchor; ignore `MCI匹配后` source label and overwrite label from the original `MCI` anchor |

其他目录例如：

```text
AD匹配后
PD相关_帕金森病匹配后
PD相关_震颤匹配后
PD相关_特发性震颤匹配后
PD相关_运动障碍匹配后
```

本轮不作为正式 8 个任务运行，除非另开实验。

## 4.1 PD 5分类 label

```python
if health_label == 0:
    class_id = 0
else:
    class_id = int(pd_disease_label) + 1
```

映射：

```text
pd_disease_label 0 -> class_id 1 帕金森病
pd_disease_label 1 -> class_id 2 震颤
pd_disease_label 2 -> class_id 3 特发性震颤
pd_disease_label 3 -> class_id 4 运动障碍
```

## 4.2 PD 二分类 label

PD 二分类任务 `pd_binary` 使用与 PD 5 分类一致的 subject-level split。训练标签为：

```python
label = int(health_label)
```

语义：

```text
health_label 0 -> control
health_label 1 -> patient
```

其中 `health_label=1` 覆盖四个 PD 相关亚型：

```text
pd_disease_label 0 帕金森病
pd_disease_label 1 震颤
pd_disease_label 2 特发性震颤
pd_disease_label 3 运动障碍
```

`pd_disease_label` 不进入二分类 loss，只用于数据审计和结果分层分析。

校验：

```text
control rows health_label=0 时 pd_disease_label 可以为 -1/NA。
patient rows health_label=1 时 pd_disease_label 必须在 {0,1,2,3}。
最终 class_id 必须在 {0,1,2,3,4}。
```

二分类任务：

```text
label = health_label
0 = control / healthy
1 = patient / disease
```

---

# 5. split 与 audit

训练前必须检查：

```text
downstream/<view>/train.csv 存在
downstream/<view>/validation.csv 存在
downstream/<view>/test.csv 存在
downstream/<view>/split_summary.json 存在
split_summary.json 中 no_subject_overlap == true
同一个 ml_subject_id 不跨该 view 的 train/validation/test
每行包含 shard_id/local_trial_index/frame_offset/frame_length/ml_subject_id/task_id/health_label；仅 PD相关 5分类任务要求 pd_disease_label
```

如果某 split 缺少某类标签：

```text
不重新切 split。
对应 AUROC/AUPRC 返回 NaN。
metrics.json 中记录 reason。
```

---

# 6. preprocessing

下游必须复用预训练 preprocessing：

```text
x_norm = clip(x_deg, -30, 30) / 30
y_norm = clip(y_deg, -20, 20) / 20
stim order = [fix_on, stim_on, stim_x_norm, stim_y_norm]
qc label: 0 valid, 1 blink, 2 missing
20ms non-overlap patch
S_i, L_i, R_i sequence
```

area stats 默认使用预训练统计：

```yaml
area:
  stats_mode: pretrained_stats
  stats_path: outputs/area_stats_fast_packed_full_subject_seed42.json
  fallback_to_global: true
```

scratch baseline 也使用同一份 area stats，保证输入尺度公平一致。

这份 area stats 必须来自预训练 train split 的 full-subject 统计方式：扫描每个 subject 的全部 trial，`num_valid_frames` 记录全部有效 eye-area sample 数；`max_frames_per_subject` 只用于 reservoir sampling 控制 median/MAD 估计的内存，不能提前停止 subject 的 trial 扫描。不要复用当前临时训练中用 sampled/break 方式得到的 `outputs/area_stats_fast_packed_seed42.json` 做后续 ablation。

不要用：

```text
validation.csv
test.csv
```

统计 area stats。

---

# 7. subject eye availability

正式下游分组中的眼别可用性以后缀更权威。

优先读取：

```text
source_suffix
```

若没有，则从以下字段末尾解析：

```text
subject_id
ml_subject_id
```

合法：

```text
D = both eyes available
L = left-eye only
R = right-eye only
```

用途：

```text
by_eye_availability_suffix 分组指标
run_summary 统计
可视化采样
```

frame-level loss / token validity 仍使用 `y_frame` qc label。

如果 suffix 与 qc label 冲突：

```text
记录 audit warning。
分组以 suffix 为准。
模型输入仍以 qc label 为准。
```

---

# 8. DownstreamTrialDataset 输出

返回：

```python
batch = {
    "content": FloatTensor[B, N, 2, 20, 4],
    "quality": FloatTensor[B, N, 2, 20, 1],
    "stim": FloatTensor[B, N, 20, 4],
    "task_id": LongTensor[B],
    "pad_mask": BoolTensor[B, N],
    "eye_nonmissing_frac": FloatTensor[B, N, 2],
    "eye_token_valid": BoolTensor[B, N, 2],

    "subject_id": list[str],
    "ml_subject_id": list[str],
    "trial_id": list[str],
    "global_trial_id": list[str],
    "subject_eye_availability": list[str],

    "label": FloatTensor[B] or LongTensor[B],
    "sample_weight": FloatTensor[B],
}
```

如果一个 trial 没有任何 valid eye token：

```text
默认跳过该 trial。
如果无法跳过，则 sample_weight=0。
```

---

# 9. 模型与 encoder 加载

从预训练 checkpoint 加载 encoder：

```yaml
pretrained:
  checkpoint: outputs/eyemae_pretrain/checkpoint_best.pt
  pretrain_config: configs/eyemae_cnn_512_12l.yaml
  load_encoder_only: true
  strict: false
```

加载：

```text
content tokenizer
quality tokenizer
stim tokenizer
task embedding
token type embedding
time embedding
Transformer encoder
fusion norms
```

不加载：

```text
prediction head
reconstruction head
optimizer
scheduler
```

实现要求：

```text
处理 DDP checkpoint 中的 "module." prefix。
输出 missing_keys / unexpected_keys。
encoder关键权重缺失时报错。
classification head 永远随机初始化。
```

`model.forward_features(...)`：

```python
encoder_outputs = model.forward_features(
    content=batch["content"],
    quality=batch["quality"],
    stim=batch["stim"],
    task_id=batch["task_id"],
    pad_mask=batch["pad_mask"],
    eye_token_valid=batch["eye_token_valid"],
    mae_mask=None,
)
```

返回：

```python
{
  "hidden_seq": FloatTensor[B, 3*N, d_model],
  "seq_attn_pad_mask": BoolTensor[B, 3*N],
  "hidden_eye": FloatTensor[B, N, 2, d_model],
  "eye_token_valid": BoolTensor[B, N, 2],
}
```

---

# 10. pooling

默认：

```yaml
pooling:
  token_pooling: eye_mean
  include_stim_tokens: false
  trial_to_subject: mean_logits
```

`eye_mean`：

```python
trial_embedding = masked_mean(hidden_eye, eye_token_valid)
```

只 pool：

```text
valid L/R eye tokens
```

不 pool：

```text
S_i stim token
padding
all-missing eye token
```

---

# 11. classifier head

binary：

```python
LayerNorm(d_model)
Linear(d_model, hidden_dim)
GELU
Dropout(dropout)
Linear(hidden_dim, 1)
```

multiclass：

```python
Linear(hidden_dim, num_classes)
```

默认：

```yaml
head:
  dropout: 0.2
  hidden_dim: 256
```

---

# 12. 微调模式

必须实现四种模式：

## 12.1 linear_probe

```yaml
finetune:
  mode: linear_probe
  freeze_encoder: true
  unfreeze_last_n_layers: 0
  unfreeze_tokenizers: false
  lr_encoder: 0.0
  lr_head: 1.0e-3
```

只训练 classification head。

## 12.2 partial

```yaml
finetune:
  mode: partial
  freeze_encoder: false
  unfreeze_last_n_layers: 4
  unfreeze_tokenizers: false
  lr_encoder: 1.0e-5
  lr_head: 1.0e-3
```

冻结 tokenizers 和前面 encoder blocks，只解冻最后 N 层 Transformer、final norm、head。

## 12.3 full

```yaml
finetune:
  mode: full
  freeze_encoder: false
  unfreeze_last_n_layers: null
  unfreeze_tokenizers: true
  lr_encoder: 1.0e-5
  lr_head: 1.0e-3
```

训练全部 encoder + head。

## 12.4 scratch

```yaml
pretrained:
  checkpoint: null

finetune:
  mode: scratch
  lr_encoder: 1.0e-4
  lr_head: 1.0e-3
```

使用同样架构随机初始化，直接用疾病 label 训练。

---

# 13. loss

## 13.1 binary

保留当前方案，不执行额外二次加权改造。

默认：

```python
raw_loss = BCEWithLogitsLoss(reduction="none")(logits, labels)
```

不要给 `BCEWithLogitsLoss` 传入 `pos_weight=`。本计划中的 `subject_pos_weight` 只用于构造手动的 `class_weight_for_label`，避免正类被重复加权。

subject-balanced sample weight：

```python
sample_weight = 1.0 / num_train_trials_for_ml_subject
```

class weighting：

```yaml
class_weighting:
  enabled: true
  mode: subject_pos_weight
```

自动统计规则：

```python
n_pos = number of unique ml_subject_id with label == 1 in train.csv
n_neg = number of unique ml_subject_id with label == 0 in train.csv

positive_weight = n_neg / max(n_pos, 1)
negative_weight = 1.0
```

如果 `n_pos == 0` 或 `n_neg == 0`，该二分类任务的 train split 不可训练，应 raise `ValueError`。

实现：

```python
class_weight_for_label = torch.where(
    labels == 1,
    positive_weight,
    negative_weight,
)

loss_per_trial = raw_loss * class_weight_for_label
loss = (loss_per_trial * sample_weight).sum() / sample_weight.sum().clamp_min(eps)
```

`class_weight_for_label` 只能从当前任务 `train.csv` 的 train subjects 统计，不能用 validation/test，且必须按唯一 `ml_subject_id` 统计，不能按 trial 数统计。

## 13.2 multiclass

使用：

```python
CrossEntropyLoss(reduction="none")
```

同样使用 subject-balanced sample weight。

多分类 class weight 按 train subjects 统计：

```python
class_weight[c] = num_train_subjects / (num_classes * num_subjects_in_class_c)
```

配置：

```yaml
class_weighting:
  enabled: true
  mode: subject_class_weight
```

传入：

```python
CrossEntropyLoss(weight=class_weight, reduction="none")
```

注意：

```text
class_weight 只能从 train.csv 的 ml_subject_id 统计。
不要按 trial 数统计。
不要用 validation/test。
```

---

# 14. subject-level aggregation

binary：

```python
subject_logit = mean(trial_logits for same ml_subject_id)
subject_prob = sigmoid(subject_logit)
```

multiclass：

```python
subject_logits = mean(trial_logits_per_class)
subject_prob = softmax(subject_logits)
```

输出：

```text
trial_predictions.csv
subject_predictions.csv
metrics.json
confusion_matrix.json
```

trial_predictions.csv：

```csv
ml_subject_id,global_trial_id,split,label,logit,prob,task_id,view
```

subject_predictions.csv：

```csv
ml_subject_id,split,label,logit,prob,num_trials
```

对于 multiclass，保存每类 logit/prob：

```text
logit_0 ... logit_4
prob_0 ... prob_4
```

---

# 15. metrics

二分类主指标：

```text
subject_auroc
subject_auprc
subject_accuracy
subject_balanced_accuracy
subject_sensitivity
subject_specificity
subject_f1
subject_weighted_f1
subject_cohen_kappa
subject_confusion_matrix
```

PD 5分类主指标：

```text
subject_macro_auroc_ovr
subject_macro_auprc_ovr
subject_accuracy
subject_balanced_accuracy
subject_macro_f1
subject_weighted_f1
subject_cohen_kappa
subject_confusion_matrix
```

辅助：

```text
trial-level metrics
per_task metrics
by_missing_fraction_bucket
by_eye_availability_suffix
```

如果某类正负样本不足：

```text
对应 AUROC/AUPRC 返回 NaN。
macro 对非 NaN 类别求均值。
metrics.json 记录 skipped_classes。
```

---

# 16. training

`finetune.py` 支持：

```bash
python -m eyemae.finetune \
  --config configs/downstream/<task_name>_<mode>.yaml
```

后续 ablation 固定单 GPU：

```bash
CUDA_VISIBLE_DEVICES=<one_gpu_id> torchrun --standalone --nproc_per_node=1 \
  -m eyemae.finetune \
  --config configs/downstream/<task_name>_<mode>.yaml
```

训练功能：

```text
load pretrained encoder
freeze / unfreeze according to mode
AdamW param groups
bf16 autocast
gradient clipping
early stopping
checkpoint save/resume
rank0 TensorBoard logging
DDP validation all_gather trial predictions before subject aggregation
```

Early stopping：

```yaml
train:
  max_epochs: 100
  early_stopping_patience: 10
  min_epochs_before_early_stopping: 0
  metric_for_best_model: validation/subject_auroc
  mode: max
```

规则：

```text
best checkpoint 始终按最高 validation 主指标保存。
从第一个 validation epoch 开始即可判断早停。
如果连续10个 epoch 的 validation 主指标没有刷新 best checkpoint，停止。
```

说明：

```text
2026-06-21 当前 downstream_v3 fast 队列口径：
所有模式 max_epochs=100；
所有模式 early_stopping_patience=10, min_epochs_before_early_stopping=0。
scratch 不再额外 cap 到 30 epoch，以免低估随机初始化 baseline。
```

---

# 17. evaluate_downstream

CLI 模板必须支持四种模式，不要只写 full：

```bash
python -m eyemae.evaluate_downstream \
  --config configs/downstream/<task_name>_<mode>.yaml \
  --checkpoint outputs/downstream/<task_name>/<mode_output>/checkpoint_best.pt \
  --split test
```

其中：

```text
mode in {linear_probe, partial, full, scratch}
```

支持：

```text
train
validation
test
```

---

# 18. config 策略

第一版要求：

```text
每个 <task_name>_<mode>.yaml 都是完整 resolved config。
```

可以用脚本从模板生成，但交给 `finetune.py` 的配置必须完整，不依赖隐式继承。

Codex 必须为 8 个任务 × 4 种模式生成共 32 个完整 resolved config。后面的 AD 和 PD 配置只是模板示例，不代表只实现这两个任务。

8个任务 × 4模式，共 32 个 config：

```text
configs/downstream/pd_related_5class_linear_probe.yaml
configs/downstream/pd_related_5class_partial.yaml
configs/downstream/pd_related_5class_full.yaml
configs/downstream/pd_related_5class_scratch.yaml
configs/downstream/pd_binary_random_seed20260620_fast_linear_probe.yaml
configs/downstream/pd_binary_random_seed20260620_fast_partial.yaml
configs/downstream/pd_binary_random_seed20260620_fast_full.yaml
configs/downstream/pd_binary_random_seed20260620_fast_scratch.yaml
...
```

每个 config 必须有独立输出目录：

```text
outputs/downstream/<task_name>/<mode_output>/
```

---


# 19. fine-tune ablation 计划

第一版主实验只要求四种模式：

```text
scratch full train
pretrained linear probe
pretrained partial fine-tune
pretrained full fine-tune
```

后续 fine-tune ablation 必须基于同一套 downstream train/validation/test split，且一次只改一个因素。不要用 test set 选 ablation；所有选择都基于 validation 主指标。所有 ablation 都必须记录：

```text
task_name
mode
pretrain_style
ablation_name
ablation_value
validation_main_metric
test_main_metric
```

每个 fine-tune ablation 变体固定使用一张 GPU 运行，避免 GPU 数、per-step batch exposure、吞吐差异和组件差异混在一起。若要比较多个变体，可以并行占用不同物理 GPU，但每个进程只看一张卡：

```bash
CUDA_VISIBLE_DEVICES=<one_gpu_id> torchrun --standalone --nproc_per_node=1 \
  -m eyemae.finetune \
  --config configs/downstream/<task>_<mode>_<ablation>.yaml
```

除非 ablation 本身研究 batch size，否则所有变体必须使用相同的：

```text
max_seq_tokens_per_gpu
max_trials_per_gpu
bucket_by_length
grad_accum_steps
```

默认主指标：

```text
binary:
  validation/subject_auroc

pd_related_5class:
  validation/subject_macro_auroc_ovr
```

如果 AUROC 为 NaN，则 fallback 到：

```text
validation/subject_balanced_accuracy
```

---

## 19.1 Pooling ablation

目的：

```text
检查 trial embedding 的 token pooling 是否限制下游疾病分类性能。
```

默认：

```yaml
pooling:
  token_pooling: eye_mean
  include_stim_tokens: false
```

### A. eye_mean

默认 baseline。

```text
只平均 valid L/R eye token hidden states。
不 pool S_i。
不 pool padding。
不 pool all-missing eye token。
```

### B. eye_attention

添加一个可学习 attention scorer：

```python
score_i = w^T tanh(W h_i)
alpha_i = softmax(score_i over valid eye tokens)
trial_embedding = sum(alpha_i * h_i)
```

要求：

```text
只对 valid L/R eye tokens 做 softmax。
invalid eye token score 必须 mask 为 -inf。
```

配置：

```yaml
pooling:
  token_pooling: eye_attention
  include_stim_tokens: false
```

比较：

```text
subject AUROC / macro AUROC
AUPRC
balanced accuracy
macro/weighted F1
Cohen's Kappa
per-task metrics
是否过拟合 validation
```

### C. eye_plus_stim_mean

同时 pool：

```text
valid eye tokens
non-padding S_i tokens
```

配置：

```yaml
pooling:
  token_pooling: eye_plus_stim_mean
  include_stim_tokens: true
```

目的：

```text
检查 stimulus/task/time token 直接进入 trial embedding 是否有帮助。
```

### D. taskwise_eye_mean

先按 `task_id` 分组：

```text
每个 task 内部 eye_mean pooling
再 concat 或 mean 聚合 task embeddings
```

配置示例：

```yaml
pooling:
  token_pooling: taskwise_eye_mean
  taskwise_combine: mean   # or concat
```

第一轮优先级：

```text
eye_mean vs eye_attention
```

---

## 19.2 Subject aggregation ablation

目的：

```text
检查 trial logits 到 subject prediction 的聚合方式。
```

默认：

```yaml
pooling:
  trial_to_subject: mean_logits
```

### A. mean_logits

```python
subject_logit = mean(trial_logits)
subject_prob = sigmoid(subject_logit)
```

默认。

### B. mean_probs

```python
subject_prob = mean(sigmoid(trial_logits))
```

### C. median_logits

```python
subject_logit = median(trial_logits)
```

对异常 trial 更稳健。

### D. attention_over_trials

对同一 subject 的 trial embeddings 或 logits 学习 attention weight。

第一轮优先级：

```text
mean_logits vs median_logits
```

注意：

```text
所有 aggregation 只在 evaluation / validation subject-level metric 中使用。
训练阶段仍可按 trial-level loss + sample_weight 训练。
```

---

## 19.3 Unfreeze depth ablation

目的：

```text
检查 partial fine-tune 解冻深度。
```

默认：

```yaml
finetune:
  mode: partial
  unfreeze_last_n_layers: 4
```

候选：

```yaml
unfreeze_last_n_layers: 1
unfreeze_last_n_layers: 2
unfreeze_last_n_layers: 4
unfreeze_last_n_layers: 8
```

固定：

```yaml
unfreeze_tokenizers: false
lr_encoder: 1.0e-5
lr_head: 1.0e-3
```

比较：

```text
validation subject AUROC / macro AUROC
test subject AUROC / macro AUROC
overfitting gap
training stability
```

---

## 19.4 Learning rate ablation

目的：

```text
检查 encoder/head 学习率组合。
```

优先在：

```text
pretrained_partial
pretrained_full
```

上做小网格：

```yaml
lr_encoder: [1.0e-5, 3.0e-5]
lr_head: [3.0e-4, 1.0e-3]
```

固定：

```text
same seed
same split
same batch
same early stopping
```

记录：

```text
best_epoch
validation metric
test metric
训练是否发散
```

---

## 19.5 Class weighting ablation

目的：

```text
检查类别不平衡处理是否影响 subject-level 指标。
```

候选：

```yaml
class_weighting:
  enabled: true

class_weighting:
  enabled: false
```

默认：

```text
binary:
  mode = subject_pos_weight

multiclass:
  mode = subject_class_weight
```

二分类注意：

```text
不要给 BCEWithLogitsLoss 传 pos_weight。
subject_pos_weight 只用于手动 class_weight_for_label。
```

比较：

```text
AUROC
AUPRC
sensitivity
specificity
confusion matrix
```

---

## 19.6 Stim/task input ablation

目的：

```text
检查下游疾病分类是否依赖 stimulus/task 条件。
```

### A. with_stim_task

默认，使用完整 S_i：

```text
task_id
fix_on
stim_on
stim_x
stim_y
time
```

### B. no_stim_coordinates

保留：

```text
task_id
fix_on
stim_on
time
```

但置零：

```text
stim_x = 0
stim_y = 0
```

### C. eye_only

移除或置空 S_i 中的 task/stim 信息，仅保留 time 或最小 token type。

注意：

```text
eye_only 可能导致 checkpoint 结构不兼容；
第一轮不强制实现。
```

建议先做：

```text
with_stim_task vs no_stim_coordinates
```

---

## 19.7 Task-specific evaluation / taskwise head ablation

目的：

```text
检查不同眼动任务对疾病分类贡献是否不同。
```

### A. per_task_eval

不改训练，只按 task_id 分别报告 subject-level metrics：

```text
pro
anti
memory
double
```

第一版必须实现。

若某 subject 在某 task 没有 trial：

```text
该 subject 在该 task metric 中跳过。
metrics.json 记录有效subject数。
```

### B. train_single_task

每次只用一个 task 的 trial 训练和评估。

配置示例：

```yaml
task_filter:
  enabled: true
  task_id: 0
```

### C. taskwise_logits

每个 task 单独聚合 subject logits，再做 mean 或小型 fusion head：

```text
subject_logit_task0
subject_logit_task1
subject_logit_task2
subject_logit_task3
-> mean / MLP fusion
```

第一轮先做：

```text
per_task_eval
```

---

## 19.8 Pretrain checkpoint ablation

目的：

```text
检查不同预训练 checkpoint 或预训练风格对下游的影响。
```

候选：

```text
checkpoint_best
checkpoint_last
small model checkpoint
base model checkpoint
asymmetric_mae checkpoint  # 若已完成预训练 ablation K
```

结果表必须记录：

```text
pretrain_style
pretrain_checkpoint
pretrain_model_scale
```

注意：

```text
第一版默认 pretrain_style = bert_masked_reconstruction。
asymmetric_mae checkpoint 只能作为后续对照，不属于第一版默认。
```

---

## 19.9 Fine-tune ablation 优先级

建议顺序：

```text
0. 主四模式：
   scratch full
   pretrained linear_probe
   pretrained partial
   pretrained full

1. eye_mean vs eye_attention pooling

2. mean_logits vs median_logits subject aggregation

3. unfreeze_last_n_layers:
   2 vs 4 vs 8

4. lr_encoder/lr_head 小网格:
   lr_encoder 1e-5 / 3e-5
   lr_head 3e-4 / 1e-3

5. class_weighting on/off

6. per_task_eval

7. no_stim_coordinates

8. pretrain checkpoint ablation
```

不要在所有 8 个任务上同时展开全部 ablation。建议：

```text
先选 2-3 个代表任务：
  AD
  MCI
  pd_related_5class
  pd_binary

筛出有效设置后，再推广到全部8个任务。
```



# 20. config 示例

## 20.1 AD binary full

```yaml
experiment:
  name: downstream_ad_binary_pretrained_full
  output_dir: outputs/downstream/ad_binary/pretrained_full

pretrained:
  checkpoint: outputs/eyemae_pretrain/checkpoint_best.pt
  pretrain_config: configs/eyemae_cnn_512_12l.yaml
  load_encoder_only: true
  strict: false

pretraining_exposure:
  mode: all_unlabeled_or_unknown
  pretrain_subject_manifest: /mnt/disk_sde/data-260606/extracted/eyemae_fast_dataset_v1/subjects.csv

data:
  format: packed_mmap
  data_dir: /mnt/disk_sde/data-260606/extracted/eyemae_fast_dataset_v1
  train_index: downstream/AD/train.csv
  val_index: downstream/AD/validation.csv
  test_index: downstream/AD/test.csv
  subject_key: ml_subject_id
  task_column: task_id
  trial_id_column: global_trial_id
  subject_eye_availability_column: source_suffix
  sampling_rate: 1000
  mmap_mode: r
  max_open_shards_per_worker: 44
  validate_offsets: false

label:
  type: binary
  task_name: ad_binary
  view: AD
  label_column: health_label
  negative_label: 0
  positive_label: 1
  positive_name: disease
  negative_name: healthy

area:
  stats_mode: pretrained_stats
  stats_path: outputs/area_stats_fast_packed_full_subject_seed42.json
  fallback_to_global: true
  use_log1p: true
  robust_zscore_by: subject
  clip: 5
  eps: 1.0e-6

normalization:
  x_clip_deg: 30
  y_clip_deg: 20

stim:
  use_fix_on: true
  use_stim_on: true
  use_stim_xy: true
  use_last_stim_xy: false
  use_goal_xy: false
  stim_dim: 4

input:
  content_dim: 4
  quality_dim: 1
  stim_dim: 4
  num_eyes: 2

patch:
  samples: 20
  stride: 20

attention:
  min_nonmissing_frac_for_eye_token: 0.05

model:
  tokenizer: cnn
  sequence_format: stim_eye_triplet_no_cls
  pretrain_style: bert_masked_reconstruction
  d_model: 512
  n_layers: 12
  n_heads: 8
  ffn_hidden: 1536
  dropout: 0.1
  norm: rmsnorm
  activation: swiglu
  max_patches: 384
  use_cls: false
  use_token_type_embedding: true
  fusion: add_then_layernorm
  use_stim_tokens: true
  broadcast_stim_to_eye: false

pooling:
  token_pooling: eye_mean
  include_stim_tokens: false
  trial_to_subject: mean_logits

head:
  dropout: 0.2
  hidden_dim: 256

finetune:
  mode: full
  freeze_encoder: false
  unfreeze_last_n_layers: null
  unfreeze_tokenizers: true
  lr_encoder: 1.0e-5
  lr_head: 1.0e-3
  weight_decay: 0.05

class_weighting:
  enabled: true
  mode: subject_pos_weight

train:
  seed: 42
  precision: bf16
  distributed: ddp
  max_epochs: 100
  early_stopping_patience: 10
  min_epochs_before_early_stopping: 0
  metric_for_best_model: validation/subject_auroc
  mode: max

  batch_trials_per_gpu: null
  max_seq_tokens_per_gpu: 90000
  max_trials_per_gpu: 128
  bucket_by_length: true

  grad_accum_steps: 1
  grad_clip: 1.0
  num_workers: 8
  pin_memory: true
  persistent_workers: true
  prefetch_factor: 4

metrics:
  threshold: 0.5
  bootstrap_ci: false
  bootstrap_n: 1000
```

## 20.2 PD related 5-class full

```yaml
label:
  type: multiclass
  task_name: pd_related_5class
  view: PD相关
  num_classes: 5
  source_columns: [health_label, pd_disease_label]
  class_map:
    control: 0
    帕金森病: 1
    震颤: 2
    特发性震颤: 3
    运动障碍: 4

data:
  train_index: downstream/PD相关/train.csv
  val_index: downstream/PD相关/validation.csv
  test_index: downstream/PD相关/test.csv

class_weighting:
  enabled: true
  mode: subject_class_weight

train:
  metric_for_best_model: validation/subject_macro_auroc_ovr
```

---

# 21. tests

必须覆盖：

```text
packed downstream index 可读取
同一 ml_subject_id 不跨 train/validation/test
PD 5分类 label 映射正确
6个二分类 health_label 映射正确
subject_eye_availability 使用 source_suffix / suffix
stim patch 顺序为 [fix_on, stim_on, stim_x_norm, stim_y_norm]
DownstreamTrialDataset 返回 label/sample_weight/ml_subject_id/global_trial_id
eye_mean 只 pool valid eye token
linear_probe / partial / full / scratch freeze 行为正确
pretrained loading 能去除 module. prefix
reconstruction head 不加载进 classification head
subject aggregation mean_logits 正确
binary metrics 和 multiclass metrics 正确
单一类别时 AUROC/AUPRC 返回 NaN
```

---

# 22. 训练流程

## 21.1 检查数据

```bash
pytest tests/test_fast_packed_dataset.py
pytest tests/test_downstream_dataset.py
pytest tests/test_downstream_labels.py
```

## 21.2 运行四种模式

```bash
python -m eyemae.finetune --config configs/downstream/<task_name>_linear_probe.yaml
python -m eyemae.finetune --config configs/downstream/<task_name>_scratch.yaml
python -m eyemae.finetune --config configs/downstream/<task_name>_partial.yaml
python -m eyemae.finetune --config configs/downstream/<task_name>_full.yaml
```

对 8 个任务全部运行。

## 21.3 test evaluation

```bash
python -m eyemae.evaluate_downstream \
  --config configs/downstream/<task_name>_<mode>.yaml \
  --checkpoint outputs/downstream/<task_name>/<mode_output>/checkpoint_best.pt \
  --split test
```

---

# 23. 输出文件

每个实验输出：

```text
resolved_config.yaml
run_summary.json
checkpoint_last.pt
checkpoint_best.pt
tensorboard/
validation_metrics.json
test_metrics.json
trial_predictions_validation.csv
subject_predictions_validation.csv
trial_predictions_test.csv
subject_predictions_test.csv
confusion_matrix_validation.json
confusion_matrix_test.json
```

`run_summary.json` 必须记录：

```text
task_name
mode
pretraining_exposure.mode
pretrained_checkpoint
num_train_subjects
num_validation_subjects
num_test_subjects
label_counts
subject_eye_availability_counts
```

结果表必须包含：

```text
Pretrain exposure
```

---

# 24. 推荐结果表

## 24.1 主结果表：全程 best checkpoint

| Task | Model | Pretrain exposure | Encoder初始化 | 微调方式 | Subject AUROC / Macro AUROC | Subject AUPRC / Macro AUPRC | Balanced Acc | F1 |
|---|---|---|---|---|---:|---:|---:|---:|
| AD | Scratch | none | random | full |  |  |  |  |
| AD | Pretrained | all_unlabeled_or_unknown | pretrained | linear_probe |  |  |  |  |
| AD | Pretrained | all_unlabeled_or_unknown | pretrained | partial |  |  |  |  |
| AD | Pretrained | all_unlabeled_or_unknown | pretrained | full |  |  |  |  |

全程 best checkpoint 仍然用 validation subject-level 主指标选择，并在加载
best checkpoint 后报告 train / validation / test。

## 24.2 收敛速度表：epochs 0-29 内 best checkpoint

额外报告每个 task/mode 在前 30 个 epoch 内的 validation-best checkpoint
对应 test 结果：

```text
Task
Mode
best_epoch_within_30
validation subject AUROC / macro AUROC at that epoch
test subject AUROC / macro AUROC
test subject AUPRC / macro AUPRC
test balanced accuracy
test F1 / macro F1
test weighted F1
test Cohen's Kappa
checkpoint_source
```

若全程 best_epoch 本身小于 30，可直接复用主结果 checkpoint/test 指标。
若全程 best_epoch 大于等于 30，则需要使用保存的早期 checkpoint 或单独重跑
早期 checkpoint 评估；如果某个历史已完成 job 没有保留该 checkpoint，必须在
training log 中明确标注 unavailable，而不是用最终 best 代替。

## 24.3 收敛速度表：exactly 1 epoch

额外报告每个 task/mode 只训练 1 个 epoch 后的 test 结果，用于比较随机
初始化、linear probe、partial fine-tune、full fine-tune 的早期收敛速度：

```text
Task
Mode
epoch = 0
validation subject AUROC / macro AUROC
test subject AUROC / macro AUROC
test subject AUPRC / macro AUPRC
test balanced accuracy
test F1 / macro F1
test weighted F1
test Cohen's Kappa
checkpoint_source
```

该表不替代主结果表，不用于选择最终 checkpoint；它只用于展示收敛速度。

---

# 25. 不要做的事

第一版下游不要做：

```text
不要重新生成 downstream split，除非发现既有 split/label 口径错误并由实验记录明确标注为 follow-up rerun
不要实现 make_downstream_splits.py
不要把同一个 ml_subject_id 分到多个 split
不要用 test.csv 调参
不要只报告 trial-level 指标
不要在下游微调时做 MAE mask
不要加载 reconstruction head 作为 classification head
不要让 trial 数量多的 subject 主导 loss
不要用 validation.csv/test.csv 计算 class weights 或 area stats
不要用 test set 选择 checkpoint
不要只为 full 模式写 evaluate 命令
不要生成不完整的继承式 config 给 finetune.py
```

---

# 26. 第15点说明：binary class weighting 暂不改

当前计划保留：

```text
BCEWithLogitsLoss(reduction="none")
手动按 label 乘 class_weight_for_label
再乘 sample_weight
```

你暂时不执行的那个建议，是为了避免以后同时使用两种二分类加权方式：

```text
方式A：BCEWithLogitsLoss(pos_weight=...)
方式B：loss_per_trial *= class_weight_for_label
```

如果两者同时使用，就会对 positive 类重复加权。

本计划暂不改变现有二分类 loss 方案，只要求 Codex 不要额外再加 `pos_weight=` 参数。

---

# 27. MCI 匹配后 label-fixed follow-up

2026-06-21 追加修正：`MCI匹配后` 的健康/患病 label 方向确认与原始
MCI anchor 相反，因此旧的
`mci_matched_binary_random_seed20260621` 结果只作为历史/问题定位结果保留，
不作为 MCI 匹配后正式解释结果。

新的复跑任务：

```text
task: mci_matched_binary_random_seed20260622_label_fixed
view: downstream/MCI匹配后_random_seed20260622_label_fixed/
split: subject-level stratified random split, seed=20260622
label: 只保留 raw subject 存在于原始 MCI anchor 的 matched row；
       最终 health_label = 原始 MCI subject anchor label 的反向
```

该任务仍然不使用 `MCI匹配后` 的 subject identity 作为 label authority。
`MCI匹配后` 只提供样本行；是否保留由 raw `subject` 是否存在于原始 MCI 决定。
由于 matched view 的健康/患病编码方向被确认反了，最终 label 明确采用原始
MCI anchor 的反向，并重新划分 train/validation/test，而不是沿用旧
seed-20260621 split。
