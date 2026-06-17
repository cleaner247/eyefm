# EyeMAE 下游健康/疾病微调：Codex 工程实现计划

## 0. 目标

在已经完成 EyeMAE / EyeBERT-style 预训练的基础上，实现一个下游微调工程，用带健康/疾病 label 的眼动数据评估预训练 encoder 的实际效果。

预训练背景：

```text
预训练数据：
  总时长约400h
  约4000个subject
  采样频率1000Hz
  输入和预训练plan一致：
    S_i, L_i, R_i sequence
    20ms patch
    stim/task/time token
    left/right eye token
```

下游目标：

```text
输入：
  一个trial或一个subject的眼动数据
  eye + stim + task信息
  subject-level健康/疾病label

模型：
  预训练 EyeMAE encoder
  pooling head
  disease classification head

输出：
  binary:
    healthy vs disease
  可扩展：
    multi-class disease subtype
    severity regression
```

第一版下游任务默认做：

```text
binary subject-level disease classification

训练：
  trial-level training with subject-balanced sample weights

评估：
  subject-level aggregation
  主指标使用 subject-level AUROC / AUPRC / balanced accuracy
```

第一版必须比较：

```text
1. Scratch full train
2. Pretrained linear probe
3. Pretrained partial fine-tune
4. Pretrained full fine-tune
```

这样才能判断预训练是否真的对下游健康/疾病任务有帮助。

---

# 1. 总体原则

## 1.1 预训练 split 和下游 split 分开

预训练 split 用于 masked reconstruction：

```text
pretrain_train
pretrain_val
pretrain_test
```

下游 split 用于健康/疾病分类：

```text
downstream_train
downstream_val
downstream_test
```

不要直接复用预训练 split 作为疾病任务 split。

## 1.2 下游必须 subject-level split

如果疾病 label 是 subject-level，则同一个 subject 的所有 trial 必须只能出现在一个 split 中：

```text
subject A all trials -> downstream_train
subject B all trials -> downstream_val
subject C all trials -> downstream_test
```

禁止：

```text
subject A 部分trial在train，部分trial在test
```

否则会产生 subject leakage，模型可能学到个体特征而不是真正疾病特征。

## 1.3 主指标必须 subject-level

trial-level metric 可以记录，但不能作为主结论。

主指标：

```text
subject-level AUROC
subject-level AUPRC
subject-level balanced accuracy
subject-level sensitivity
subject-level specificity
subject-level F1
```

trial-level AUROC 只能作为辅助分析。

## 1.4 下游微调不做 MAE mask

预训练时：

```text
mae_mask用于masked reconstruction
```

下游微调时：

```text
mae_mask = None 或全False
不遮住eye token
不使用重建head
只使用encoder hidden states做分类
```

---

# 2. 项目结构增量

在预训练项目基础上新增：

```text
eyemae/
  configs/
    downstream/
      disease_binary_linear_probe.yaml
      disease_binary_partial.yaml
      disease_binary_full.yaml
      disease_binary_scratch.yaml

  src/
    eyemae/
      downstream_data.py
      make_downstream_splits.py
      finetune.py
      evaluate_downstream.py
      downstream_metrics.py
      pooling.py
      checkpoint_utils.py

  scripts/
    make_downstream_splits.sh
    finetune_linear_probe.sh
    finetune_partial.sh
    finetune_full.sh
    finetune_scratch.sh
    eval_downstream.sh

  tests/
    test_downstream_splits.py
    test_downstream_labels.py
    test_downstream_dataset.py
    test_downstream_pooling.py
    test_finetune_freeze_modes.py
    test_subject_aggregation.py
    test_downstream_metrics.py
    test_pretrained_loading.py
```

保留预训练工程中的：

```text
preprocess.py
patching.py
batching.py
model.py
config.py
compute_area_stats.py
```

下游数据必须复用和预训练一致的 preprocessing / patching / S-L-R sequence 逻辑。

---

# 3. 下游数据接口

## 3.1 trial npz 输入

沿用预训练数据格式：

```python
trial = {
    "eye": FloatTensor[T, 8],
    # 0 L_x_deg
    # 1 L_y_deg
    # 2 L_area
    # 3 L_label
    # 4 R_x_deg
    # 5 R_y_deg
    # 6 R_area
    # 7 R_label

    "task_id": LongTensor[],
    # 0 pro / 正向
    # 1 anti / 反向
    # 2 memory / 记忆
    # 3 double / 二次眼跳

    "fix_on": FloatTensor[T],
    "stim": FloatTensor[T, 3],
    # 0 stim_on
    # 1 stim_x_deg
    # 2 stim_y_deg

    "subject_id": str,
    "trial_id": str,
}
```

## 3.2 label CSV

第一版默认疾病 label 是 subject-level label。

推荐 label 文件：

```csv
base_subject_id,label
subj001,0
subj002,1
subj003,1
```

其中：

```text
label=0: healthy
label=1: disease
```

也支持：

```csv
subject_id,label
subj001D,0
subj002L,1
subj003R,1
```

但推荐使用：

```text
base_subject_id
```

因为 subject_id 末尾 `D/L/R` 是眼别可用性后缀，不应该改变疾病标签归属。

## 3.3 base subject id

实现：

```python
def get_base_subject_id(subject_id: str) -> str:
    if len(subject_id) > 0 and subject_id[-1] in {"D", "L", "R"}:
        return subject_id[:-1]
    return subject_id
```

下游 label join 默认使用：

```text
base_subject_id
```

如果 label_csv 使用 `subject_id`，也先转换成 `base_subject_id` 再匹配。

## 3.4 DownstreamTrialDataset 输出

实现：

```text
src/eyemae/downstream_data.py
```

返回 batch：

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
    "base_subject_id": list[str],
    "trial_id": list[str],

    "label": FloatTensor[B] or LongTensor[B],
    "sample_weight": FloatTensor[B],
}
```

binary classification：

```text
label: FloatTensor[B]
0.0 = healthy
1.0 = disease
```

multi-class classification extension：

```text
label: LongTensor[B]
```

---

# 4. 下游 split 方案

实现：

```text
src/eyemae/make_downstream_splits.py
tests/test_downstream_splits.py
```

## 4.1 默认 split

第一版默认：

```yaml
downstream_split:
  strategy: subject_stratified
  seed: 42
  train_ratio: 0.70
  val_ratio: 0.15
  test_ratio: 0.15
  stratify_by: disease_label
  group_by_base_subject_id: true
```

输出：

```text
splits/downstream_subject_seed42/
  downstream_train.txt
  downstream_val.txt
  downstream_test.txt
  downstream_split_summary.json
```

每个 txt 一行一个相对 `data.data_dir` 的 `.npz` 路径。

## 4.2 subject_stratified 规则

必须满足：

```text
同一个 base_subject_id 不跨 split
train/val/test 都尽量保持 healthy/disease 比例一致
每个 split 至少包含 healthy 和 disease，除非样本数太少
```

如果某个 split 缺少 positive 或 negative：

```text
raise warning
允许继续生成
evaluate时该split的AUROC/AUPRC返回NaN
```

## 4.3 小样本情况

如果 labeled subjects 较少，建议使用 K-fold：

```yaml
downstream_split:
  strategy: subject_stratified_kfold
  num_folds: 5
  seed: 42
  group_by_base_subject_id: true
```

第一版必须实现：

```text
subject_stratified train/val/test
```

K-fold 可作为第二阶段功能；如果实现成本不高，可以一起实现。

## 4.4 split summary

生成：

```json
{
  "strategy": "subject_stratified",
  "seed": 42,
  "num_train_subjects": 0,
  "num_val_subjects": 0,
  "num_test_subjects": 0,
  "num_train_trials": 0,
  "num_val_trials": 0,
  "num_test_trials": 0,
  "label_counts": {
    "train": {"0": 0, "1": 0},
    "val": {"0": 0, "1": 0},
    "test": {"0": 0, "1": 0}
  },
  "task_counts": {
    "train": {"0": 0, "1": 0, "2": 0, "3": 0},
    "val": {"0": 0, "1": 0, "2": 0, "3": 0},
    "test": {"0": 0, "1": 0, "2": 0, "3": 0}
  },
  "eye_availability_counts": {
    "train": {"D": 0, "L": 0, "R": 0, "unknown": 0},
    "val": {"D": 0, "L": 0, "R": 0, "unknown": 0},
    "test": {"D": 0, "L": 0, "R": 0, "unknown": 0}
  }
}
```

## 4.5 CLI

```bash
python -m eyemae.make_downstream_splits \
  --config configs/downstream/disease_binary_full.yaml
```

也支持显式参数：

```bash
python -m eyemae.make_downstream_splits \
  --data_dir /path/to/npz \
  --label_csv /path/to/disease_labels.csv \
  --out_dir splits/downstream_subject_seed42 \
  --strategy subject_stratified \
  --seed 42 \
  --train_ratio 0.70 \
  --val_ratio 0.15 \
  --test_ratio 0.15
```

---

# 5. 数据泄漏规则

## 5.1 下游 train/val/test 泄漏

必须保证：

```text
同一个 base_subject_id 不跨 downstream_train / downstream_val / downstream_test。
```

## 5.2 预训练是否见过 downstream_test subject

需要在实验配置和输出中记录：

```yaml
pretraining_exposure:
  mode: all_unlabeled_or_unknown
  pretrain_subject_manifest: null
```

可选模式：

```text
strict_unseen:
  downstream_test subjects 不出现在预训练数据中。

all_unlabeled_or_unknown:
  预训练可能包含 downstream_test subjects 的无标签眼动数据。
  微调阶段没有使用 test labels。
```

第一版不强制过滤预训练 checkpoint，只需要在 `metrics.json` 和 `run_summary.json` 中记录该设置。

如果有预训练 subject manifest，则 evaluate_downstream 可以报告：

```text
downstream_test subjects seen in pretraining: count / total
```

---

# 6. 下游 preprocessing

下游必须复用预训练 preprocessing：

```text
坐标归一化:
  x_norm = clip(x_deg, -30, 30) / 30
  y_norm = clip(y_deg, -20, 20) / 20

stim:
  fix_on, stim_on, stim_x_norm, stim_y_norm

label:
  label=0 non-blink
  label=1 blink
  label=2 missing

patch:
  20ms non-overlap

sequence:
  S_i, L_i, R_i
```

## 6.1 area statistics

推荐第一版使用预训练时的 area stats：

```yaml
area:
  stats_mode: pretrained_stats
  stats_path: outputs/area_stats_subject_heldout_seed42.json
  fallback_to_global: true
```

这样 downstream 输入分布与预训练 encoder 对齐。

如果从 scratch 训练，也默认使用同一套 area stats，保证和 pretrained 实验公平可比。

可选模式：

```yaml
area:
  stats_mode: downstream_train_stats
```

如果使用该模式：

```text
只能用 downstream_train 统计。
不能使用 downstream_val / downstream_test。
```

第一版默认：

```text
pretrained_stats
```

---

# 7. 下游模型结构

实现：

```text
src/eyemae/finetune.py
src/eyemae/pooling.py
```

## 7.1 encoder 加载

从预训练 checkpoint 加载 encoder：

```yaml
pretrained:
  checkpoint: outputs/eyemae_pretrain/checkpoint_best.pt
  pretrain_config: configs/eyemae_cnn_512_12l.yaml
  load_encoder_only: true
  strict: false
```

必须加载：

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

不需要加载：

```text
prediction head
reconstruction head
optimizer
scheduler
```

`mask_token` 可加载但下游不使用。

实现要求：

```text
能处理 DDP checkpoint 中的 "module." prefix。
输出 missing_keys / unexpected_keys 到日志。
如果 encoder 关键权重缺失，raise error。
classification head 永远随机初始化。
```

## 7.2 forward_features

预训练模型需要暴露：

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

下游时：

```text
mae_mask=None 或全False
不替换content token
不调用prediction head
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

## 7.3 pooling

第一版默认：

```yaml
pooling:
  token_pooling: eye_mean
  include_stim_tokens: false
```

实现：

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

如果一个 trial 没有任何 valid eye token：

```text
跳过该trial，或输出zero embedding并将sample_weight=0。
第一版建议跳过该trial。
```

可选 pooling：

```text
eye_mean
eye_max
eye_attention
eye_plus_stim_mean
```

第一版必须实现：

```text
eye_mean
```

## 7.4 classifier head

binary classification：

```python
head = nn.Sequential(
    nn.LayerNorm(d_model),
    nn.Linear(d_model, d_model // 2),
    nn.GELU(),
    nn.Dropout(0.2),
    nn.Linear(d_model // 2, 1),
)
```

输出：

```python
logits: FloatTensor[B]
```

multi-class extension：

```python
nn.Linear(d_model // 2, num_classes)
```

---

# 8. 微调模式

必须实现四种模式。

## 8.1 linear_probe

```yaml
finetune:
  mode: linear_probe
  freeze_encoder: true
  unfreeze_last_n_layers: 0
  lr_encoder: 0.0
  lr_head: 1.0e-3
```

规则：

```text
encoder所有参数 requires_grad=False
classification head requires_grad=True
```

目的：

```text
评估预训练表示本身是否含有疾病信息。
```

## 8.2 partial fine-tuning

```yaml
finetune:
  mode: partial
  freeze_encoder: false
  unfreeze_last_n_layers: 4
  lr_encoder: 1.0e-5
  lr_head: 1.0e-3
```

规则：

```text
默认冻结encoder
解冻最后N个Transformer blocks
解冻最后norm层
解冻classification head
```

如果 tokenizer 是否解冻不确定，第一版默认：

```yaml
finetune:
  unfreeze_tokenizers: false
```

## 8.3 full fine-tuning

```yaml
finetune:
  mode: full
  freeze_encoder: false
  unfreeze_last_n_layers: null
  lr_encoder: 1.0e-5
  lr_head: 1.0e-3
```

规则：

```text
encoder全部可训练
classification head可训练
encoder lr小于head lr
```

## 8.4 scratch baseline

```yaml
pretrained:
  checkpoint: null

finetune:
  mode: scratch
```

规则：

```text
使用同样架构
随机初始化encoder
随机初始化classification head
直接用疾病label训练
```

目的：

```text
判断预训练是否带来增益。
```

---

# 9. 下游 loss

## 9.1 binary classification

默认：

```python
loss = BCEWithLogitsLoss(reduction="none")
```

由于 label 是 subject-level，但训练样本是 trial-level，需要避免 trial 多的 subject 主导训练。

实现：

```python
sample_weight = 1.0 / num_train_trials_for_base_subject
```

训练 loss：

```python
loss_per_trial = BCEWithLogitsLoss(reduction="none")(logits, labels)

if class_weighting.enabled:
    loss_per_trial *= class_weight_for_label

loss = (loss_per_trial * sample_weight).sum() / sample_weight.sum().clamp_min(eps)
```

class weight 默认：

```yaml
class_weighting:
  enabled: true
  mode: subject_pos_weight
```

binary pos weight：

```python
pos_weight = num_negative_train_subjects / num_positive_train_subjects
```

注意：

```text
pos_weight 只能用 downstream_train subjects 计算。
不能用 val/test。
```

## 9.2 multi-class classification

使用：

```python
CrossEntropyLoss(reduction="none")
```

sample_weight 同样基于：

```text
1 / num_train_trials_for_base_subject
```

## 9.3 severity regression extension

使用：

```python
SmoothL1Loss(reduction="none")
```

第一版不是必需。

---

# 10. subject-level aggregation

实现：

```text
src/eyemae/downstream_metrics.py
```

## 10.1 binary aggregation

每个 trial 输出：

```python
trial_logit
```

同一个 subject 聚合：

```python
subject_logit = mean(trial_logits for same base_subject_id)
subject_prob = sigmoid(subject_logit)
```

默认：

```yaml
pooling:
  trial_to_subject: mean_logits
```

可选：

```text
mean_probs
median_logits
attention_over_trials
```

第一版必须实现：

```text
mean_logits
```

## 10.2 multi-class aggregation

```python
subject_logits = mean(trial_logits_per_class)
subject_prob = softmax(subject_logits)
```

## 10.3 输出文件

evaluation 必须保存：

```text
trial_predictions.csv
subject_predictions.csv
metrics.json
confusion_matrix.json
```

trial_predictions.csv：

```csv
base_subject_id,subject_id,trial_id,split,label,logit,prob,task_id
```

subject_predictions.csv：

```csv
base_subject_id,split,label,logit,prob,num_trials
```

---

# 11. 下游 metrics

## 11.1 binary classification metrics

主指标：

```text
subject_auroc
subject_auprc
subject_accuracy
subject_balanced_accuracy
subject_sensitivity
subject_specificity
subject_f1
subject_confusion_matrix
```

辅助指标：

```text
trial_auroc
trial_auprc
trial_accuracy
```

如果某个 split 中只有一个类别：

```text
AUROC/AUPRC 返回 NaN
不报错
```

threshold：

```yaml
metrics:
  threshold: 0.5
```

也可以在 val 上选择最佳 threshold，但第一版默认固定0.5。

## 11.2 grouped metrics

记录：

```text
per_task_subject_auroc
per_task_trial_auroc
by_missing_fraction_bucket
by_eye_availability_suffix
```

如果某组类别不足，返回NaN。

## 11.3 confidence intervals

第一版可选实现 bootstrap：

```yaml
metrics:
  bootstrap_ci: false
  bootstrap_n: 1000
```

如果实现，bootstrap单位必须是：

```text
subject
```

不要按trial bootstrap。

---

# 12. 下游训练脚本

实现：

```text
src/eyemae/finetune.py
```

CLI：

```bash
python -m eyemae.finetune \
  --config configs/downstream/disease_binary_full.yaml
```

resume：

```bash
python -m eyemae.finetune \
  --config configs/downstream/disease_binary_full.yaml \
  --resume outputs/downstream_disease_full/checkpoint_last.pt
```

DDP：

```bash
torchrun --standalone --nproc_per_node=3 \
  -m eyemae.finetune \
  --config configs/downstream/disease_binary_full.yaml
```

训练功能：

```text
load pretrained encoder
freeze / unfreeze according to finetune.mode
AdamW with param groups
bf16 autocast
gradient clipping
early stopping
checkpoint save/resume
rank0 TensorBoard logging
rank0 visualization / prediction saving
DDP validation all_gather predictions before subject aggregation
```

## 12.1 optimizer param groups

必须分开：

```text
encoder params:
  lr_encoder
  weight_decay

head params:
  lr_head
  weight_decay
```

linear probe：

```text
optimizer only contains head params
```

scratch/full/partial：

```text
optimizer contains trainable encoder params + head params
```

## 12.2 early stopping

配置：

```yaml
train:
  max_epochs: 100
  early_stopping_patience: 20
  min_epochs_before_early_stopping: 50
  metric_for_best_model: val/subject_auroc
  mode: max
```

规则：

```text
每个下游 fine-tune run 最多训练 100 epoch。
前 50 个 epoch 必须完整训练，不允许触发 early stopping。
由于 epoch 从 0 开始编号，跑完 epoch index 49 才算完成 50 个 epoch；从这之后才允许判断 early stopping。
如果验证集 subject-level AUROC 连续 20 个可判断 epoch 没有提升，停止训练。
best checkpoint 仍然按整个训练期间最高的 val/subject_auroc 保存；即使 best_epoch < 50，也要继续训练到至少 50 epoch 再允许停止。
```

如果 `subject_auroc` 是 NaN：

```text
fallback到 val/subject_balanced_accuracy
```

---

# 13. downstream evaluate

实现：

```text
src/eyemae/evaluate_downstream.py
```

CLI：

```bash
python -m eyemae.evaluate_downstream \
  --config configs/downstream/disease_binary_full.yaml \
  --checkpoint outputs/downstream_disease_full/checkpoint_best.pt \
  --split downstream_test
```

支持：

```text
downstream_train
downstream_val
downstream_test
```

必须：

```text
加载checkpoint
不更新参数
收集所有trial predictions
按base_subject_id聚合
输出subject-level metrics
保存trial_predictions.csv
保存subject_predictions.csv
保存metrics.json
```

---

# 14. downstream config

创建：

```text
configs/downstream/disease_binary_full.yaml
```

示例：

```yaml
experiment:
  name: downstream_disease_binary_pretrained_full
  output_dir: outputs/downstream_disease_binary_pretrained_full

pretrained:
  checkpoint: outputs/eyemae_pretrain/checkpoint_best.pt
  pretrain_config: configs/eyemae_cnn_512_12l.yaml
  load_encoder_only: true
  strict: false

pretraining_exposure:
  mode: all_unlabeled_or_unknown
  pretrain_subject_manifest: null

data:
  format: npz_per_trial
  data_dir: /path/to/npz
  label_csv: /path/to/disease_labels.csv

  downstream_train_split: splits/downstream_subject_seed42/downstream_train.txt
  downstream_val_split: splits/downstream_subject_seed42/downstream_val.txt
  downstream_test_split: splits/downstream_subject_seed42/downstream_test.txt

  sampling_rate: 1000

  npz_keys:
    eye: eye
    task_id: task_id
    fix_on: fix_on
    stim: stim
    subject_id: subject_id
    trial_id: trial_id

downstream_split:
  strategy: subject_stratified
  seed: 42
  train_ratio: 0.70
  val_ratio: 0.15
  test_ratio: 0.15
  stratify_by: disease_label
  group_by_base_subject_id: true
  out_dir: splits/downstream_subject_seed42

label:
  type: binary
  negative_label: 0
  positive_label: 1
  positive_name: disease
  negative_name: healthy

subject_suffix:
  D: both_eyes
  L: left_eye_only
  R: right_eye_only

normalization:
  x_clip_deg: 30
  y_clip_deg: 20

area:
  stats_mode: pretrained_stats
  stats_path: outputs/area_stats_subject_heldout_seed42.json
  fallback_to_global: true
  use_log1p: true
  robust_zscore_by: subject
  clip: 5
  eps: 1.0e-6

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
  max_patches: 256

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
  early_stopping_patience: 20
  min_epochs_before_early_stopping: 50
  metric_for_best_model: val/subject_auroc
  mode: max

  batch_trials_per_gpu: 64
  max_seq_tokens_per_gpu: null
  max_trials_per_gpu: null
  bucket_by_length: true

  grad_accum_steps: 1
  grad_clip: 1.0

  num_workers: 8
  pin_memory: true
  persistent_workers: true

metrics:
  threshold: 0.5
  bootstrap_ci: false
  bootstrap_n: 1000
```

另外创建三个配置，只修改关键字段：

## disease_binary_linear_probe.yaml

```yaml
experiment:
  name: downstream_disease_binary_pretrained_linear_probe
  output_dir: outputs/downstream_disease_binary_pretrained_linear_probe

finetune:
  mode: linear_probe
  freeze_encoder: true
  unfreeze_last_n_layers: 0
  unfreeze_tokenizers: false
  lr_encoder: 0.0
  lr_head: 1.0e-3
```

## disease_binary_partial.yaml

```yaml
experiment:
  name: downstream_disease_binary_pretrained_partial
  output_dir: outputs/downstream_disease_binary_pretrained_partial

finetune:
  mode: partial
  freeze_encoder: false
  unfreeze_last_n_layers: 4
  unfreeze_tokenizers: false
  lr_encoder: 1.0e-5
  lr_head: 1.0e-3
```

## disease_binary_scratch.yaml

```yaml
experiment:
  name: downstream_disease_binary_scratch_full
  output_dir: outputs/downstream_disease_binary_scratch_full

pretrained:
  checkpoint: null
  pretrain_config: null
  load_encoder_only: false

finetune:
  mode: scratch
  freeze_encoder: false
  unfreeze_last_n_layers: null
  unfreeze_tokenizers: true
  lr_encoder: 1.0e-4
  lr_head: 1.0e-3
```

---

# 15. 下游训练流程

## 15.1 生成 downstream split

```bash
python -m eyemae.make_downstream_splits \
  --config configs/downstream/disease_binary_full.yaml
```

检查：

```text
同一个 base_subject_id 不跨 split
train/val/test 健康/疾病比例合理
每个 split 都有 healthy 和 disease
```

## 15.2 linear probe

```bash
python -m eyemae.finetune \
  --config configs/downstream/disease_binary_linear_probe.yaml
```

## 15.3 partial fine-tune

```bash
python -m eyemae.finetune \
  --config configs/downstream/disease_binary_partial.yaml
```

## 15.4 full fine-tune

```bash
python -m eyemae.finetune \
  --config configs/downstream/disease_binary_full.yaml
```

## 15.5 scratch baseline

```bash
python -m eyemae.finetune \
  --config configs/downstream/disease_binary_scratch.yaml
```

## 15.6 evaluate test

```bash
python -m eyemae.evaluate_downstream \
  --config configs/downstream/disease_binary_full.yaml \
  --checkpoint outputs/downstream_disease_binary_pretrained_full/checkpoint_best.pt \
  --split downstream_test
```

对所有四个实验都跑 downstream_test evaluation。

---

# 16. tests

## 16.1 downstream split test

```bash
pytest tests/test_downstream_splits.py
```

必须验证：

```text
subject_stratified split可生成
同一base_subject_id不跨split
healthy/disease计数被统计
split_summary.json生成
```

## 16.2 downstream labels test

```bash
pytest tests/test_downstream_labels.py
```

必须验证：

```text
label_csv能按base_subject_id匹配trial
缺label的subject被跳过并warning
重复冲突label时报错
binary label只能是0/1
```

## 16.3 downstream dataset test

```bash
pytest tests/test_downstream_dataset.py
```

必须验证：

```text
DownstreamTrialDataset返回label
sample_weight正确
base_subject_id正确
复用预训练patching/preprocess
mae_mask不会在下游创建
```

## 16.4 pooling test

```bash
pytest tests/test_downstream_pooling.py
```

必须验证：

```text
eye_mean只pool valid eye token
不pool S_i
不poolpadding
不poolall-missing eye token
无valid eye token时按规则跳过或sample_weight=0
```

## 16.5 freeze modes test

```bash
pytest tests/test_finetune_freeze_modes.py
```

必须验证：

```text
linear_probe encoder全部冻结
partial只解冻最后N层和head
full全部encoder和head可训练
scratch不加载预训练checkpoint
optimizer param groups lr正确
```

## 16.6 subject aggregation test

```bash
pytest tests/test_subject_aggregation.py
```

必须验证：

```text
trial logits按base_subject_id聚合
mean_logits正确
subject_predictions.csv字段正确
```

## 16.7 downstream metrics test

```bash
pytest tests/test_downstream_metrics.py
```

必须验证：

```text
subject AUROC计算正确
只有单一类别时AUROC返回NaN
balanced accuracy/sensitivity/specificity正确
confusion matrix正确
```

## 16.8 pretrained loading test

```bash
pytest tests/test_pretrained_loading.py
```

必须验证：

```text
能加载encoder-only checkpoint
能去除module. prefix
prediction head不会加载进classification head
关键encoder权重缺失时报错
```

---

# 17. 输出文件

每个下游实验输出：

```text
outputs/{experiment.name}/
  resolved_config.yaml
  run_summary.json

  checkpoint_last.pt
  checkpoint_best.pt

  tensorboard/

  val_metrics.json
  test_metrics.json

  trial_predictions_val.csv
  subject_predictions_val.csv

  trial_predictions_test.csv
  subject_predictions_test.csv

  confusion_matrix_val.json
  confusion_matrix_test.json
```

run_summary.json 记录：

```json
{
  "pretrained_checkpoint": "...",
  "finetune_mode": "full",
  "pretraining_exposure_mode": "all_unlabeled_or_unknown",
  "num_train_subjects": 0,
  "num_val_subjects": 0,
  "num_test_subjects": 0,
  "num_train_trials": 0,
  "num_val_trials": 0,
  "num_test_trials": 0,
  "label_counts": {
    "train": {"0": 0, "1": 0},
    "val": {"0": 0, "1": 0},
    "test": {"0": 0, "1": 0}
  }
}
```

---

# 18. 推荐结果表

最终整理：

| 模型 | Encoder初始化 | 微调方式 | Subject AUROC | Subject AUPRC | Balanced Acc | Sensitivity | Specificity | F1 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Scratch | random | full train |  |  |  |  |  |  |
| Pretrained | frozen | linear probe |  |  |  |  |  |  |
| Pretrained | partial | last 4 layers |  |  |  |  |  |  |
| Pretrained | full | full fine-tune |  |  |  |  |  |  |

附加分析：

```text
trial-level AUROC
per-task subject AUROC
不同missing比例下的AUROC
D/L/R eye availability分组表现
每个split的subject数量和trial数量
```

---

# 19. 下游验收标准

代码验收：

```text
make_downstream_splits.py 可运行
同一 base_subject_id 不跨 split
DownstreamTrialDataset 能加载 label
sample_weight 正确
预训练 encoder 能正确 load
linear probe 能训练
partial fine-tune 能训练
full fine-tune 能训练
scratch baseline 能训练
evaluate_downstream.py 输出 subject-level metrics
```

结果验收：

```text
subject-level AUROC 可计算，若类别不足则NaN且不报错
subject-level confusion matrix 可生成
每个 split 的 subject 数和label分布可输出
trial-to-subject aggregation 正确
sample_weight 生效
预训练模型和scratch baseline可公平比较
```

---

# 20. 不要做的事

第一版下游不要做：

```text
不要把同一个base_subject_id分到多个split
不要用downstream_test调参
不要只报告trial-level指标
不要把trial当独立subject解释结果
不要在下游微调时做MAE mask
不要加载预训练reconstruction head作为classification head
不要让trial数量多的subject主导loss
不要用downstream_val/downstream_test计算class weights或area stats
不要用test set选择checkpoint
```

---

# 21. 第一轮实验优先级

按顺序执行：

```text
0. pytest downstream tests
1. make_downstream_splits
2. pretrained linear probe
3. scratch full train
4. pretrained partial fine-tune
5. pretrained full fine-tune
6. downstream_test evaluation for all four
7. 结果表汇总
```

判断预训练是否有用：

```text
Pretrained linear probe > Scratch:
  预训练表示本身已经带有疾病相关信息。

Pretrained partial/full > Scratch:
  预训练对下游微调有实际增益。

Pretrained ≈ Scratch:
  检查 pooling、split、label噪声、样本量、预训练任务和疾病相关性。
```

---

# 22. 实际运行记录入口

本文件是下游微调设计计划。已经完成或正在运行的训练/测试版本统一记录在：

```text
docs/eyemae_experiment_registry.md
```

其中包括：

```text
固定 holdout 下游微调版本
PD相关/戒毒所 5-fold k-fold 版本
每个版本使用的 config、split、output_root、launcher、summary 命令
当前状态和如何复查 metrics_final 数量
```
