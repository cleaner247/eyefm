# EyeBERT-style Packed-MMap 预训练（EyeMAE 项目）：Codex 工程实现计划

## 0. 目标

在已经生成好的高速 packed 数据集上，实现并运行 EyeBERT-style 自监督预训练。

正式数据集已经存在：

```text
/mnt/disk_sde/data-260606/extracted/eyemae_fast_dataset_v1
```

本计划**不再构建 packed 数据集**，不实现 converter，不重新生成 `shards/`、`pretrain/` 或 `downstream/`。Codex 本轮只需要实现：

```text
1. packed_mmap reader
2. pretrain split / audit 检查
3. area stats 统计与读取
4. preprocessing / patching / masking / model / loss
5. train / evaluate
6. DDP 与指标聚合
```

预训练目标：

```text
Task-conditioned EyeBERT-style masked reconstruction

输入：
  1000Hz trial级眼动时间序列
  每20ms一个patch
  每个patch构造一个 S_i stim/task/time token
  每个patch构造 L_i / R_i 两个 eye token
  对有效 eye token 进行 BERT-style 人工 mask
  重建被 mask 的 x, y, pupil area, blink

第一版模型：
  CNN tokenizer
  independent S_i stim/task/time token
  left/right eye-wise token
  bidirectional Transformer encoder
  d_model=512
  layers=12
  heads=8
  patch=20ms non-overlap
  no CLS
  no goal
  no last_stim
  use velocity loss
```

---

## 0.1 命名约定：第一版是 BERT-style，不是 asymmetric MAE

本项目名可以继续叫 EyeMAE，但**第一版预训练实现不是 true/asymmetric MAE**。第一版采用：

```text
model.pretrain_style = bert_masked_reconstruction
```

语义是：

```text
BERT-style masked reconstruction:
  被人工 mask 的 eye token 仍然保留在 encoder 输入序列里；
  只是该 eye token 的 content_token 替换成 learned mask_token；
  encoder 直接输出该位置 hidden state；
  reconstruction head 在该位置重建原始 [20, 4] eye patch。
```

工程中历史变量名可以继续使用：

```text
mae_mask
```

但在第一版中它的真实含义是：

```text
bert_reconstruction_mask / artificial_eye_content_mask
```

不要因为变量名叫 `mae_mask` 就实现 asymmetric MAE。真正的 MAE-style / asymmetric encoder-decoder 只属于后续 ablation：

```yaml
model:
  pretrain_style: asymmetric_mae
```

第一版 Codex 执行时必须只实现：

```yaml
model:
  pretrain_style: bert_masked_reconstruction
```


# 1. 第一版核心决定

第一版固定如下：

```text
数据：
  format = packed_mmap
  data_dir = /mnt/disk_sde/data-260606/extracted/eyemae_fast_dataset_v1
  读取已生成的 shards/ 与 pretrain/*.csv index
  不重新构建 packed dataset

输入：
  eye content = x, y, area, blink
  eye quality = missing
  stim patch order = [fix_on, stim_on, stim_x_norm, stim_y_norm]
  task_id = 4类任务

序列：
  S_0, L_0, R_0, S_1, L_1, R_1, ..., S_{N-1}, L_{N-1}, R_{N-1}

其中：
  S_i = 第 i 个patch的stimulus / task / time token
  L_i = 第 i 个patch的left-eye token
  R_i = 第 i 个patch的right-eye token

预训练方式：
  BERT-style masked reconstruction
  被MAE mask的eye token仍保留在encoder里，
  只是content_token替换成learned mask_token。
```

第一版不使用：

```text
last_stim_x/y
goal_x/y
CLS token
feature-wise tokenization
cross-attention
adaptive layer norm
decoder-only causal LM
true asymmetric MAE
session-level split
packed dataset converter
```

---

# 2. 已生成 packed 数据集格式

数据集根目录：

```text
/mnt/disk_sde/data-260606/extracted/eyemae_fast_dataset_v1
```

必须存在：

```text
dataset_manifest.json
audit_summary.json
columns.json
label_maps.json
trials.csv
subjects.csv

shards/
  shard_000000/
    X_data.npy
    y_frame.npy
    X_offsets.npy
    X_lengths.npy
    trial_index.csv
  shard_000001/
    ...

pretrain/
  pretrain_all_unique.csv
  pretrain_train.csv
  pretrain_validation.csv
  pretrain_test.csv
  pretrain_split_summary.json

downstream/
  <view>/
    train.csv
    validation.csv
    test.csv
    split_summary.json
```

注意：

```text
pretrain validation 文件名统一为 pretrain_validation.csv。
pretrain summary 文件名统一为 pretrain/pretrain_split_summary.json。
不要使用 pretrain_val.csv。
不要使用 pretrain/split_summary.json。
```

---

# 3. 项目结构

在现有工程中使用或新增：

```text
eyemae/
  pyproject.toml
  requirements.txt
  README.md

  configs/
    eyemae_cnn_512_12l.yaml
    debug.yaml

  src/
    eyemae/
      __init__.py
      config.py
      data.py
      compute_area_stats.py
      preprocess.py
      patching.py
      batching.py
      masking.py
      model.py
      losses.py
      metrics.py
      train.py
      evaluate.py
      visualize.py
      baselines.py
      utils.py

  tests/
    fixtures/
      make_synthetic_npz.py
    test_config.py
    test_fast_packed_dataset.py
    test_splits.py
    test_area_stats.py
    test_data_schema.py
    test_preprocess_missing.py
    test_patchify.py
    test_masking.py
    test_loss_gating.py
    test_model_shapes.py
    test_sequence_mapping.py
    test_attention_mask.py
    test_overfit_tiny_batch.py
```

本计划不要求实现：

```text
build_fast_packed_dataset.py
packed converter
重新生成 shards/
重新生成 pretrain split
重新生成 downstream split
```

如果仓库中已经存在 converter 文件，可以保留，但本轮 Codex 任务不应修改或依赖它。

---

# 4. 安装与 config

项目必须支持：

```bash
pip install -e .
```

所有命令必须能以模块方式运行：

```bash
python -m eyemae.train --config configs/debug.yaml
python -m eyemae.evaluate --config configs/debug.yaml --checkpoint /path/to/checkpoint.pt --split validation
python -m eyemae.compute_area_stats --config configs/eyemae_cnn_512_12l.yaml --split train
```

`debug.yaml` 和 `eyemae_cnn_512_12l.yaml` 都必须是完整配置，不依赖隐式继承。

`validate_config` 必须检查：

```text
data.format in {packed_mmap, npz_per_trial}
正式训练 data.format == packed_mmap
正式训练 data.data_dir == /mnt/disk_sde/data-260606/extracted/eyemae_fast_dataset_v1
正式训练 data.train_index == pretrain/pretrain_train.csv
正式训练 data.val_index == pretrain/pretrain_validation.csv
正式训练 data.test_index == pretrain/pretrain_test.csv
split.split_summary == pretrain/pretrain_split_summary.json
data.subject_key == ml_subject_id
data.task_column == task_id
data.trial_id_column == global_trial_id
label nonblink/blink/missing 三个值互不相同
input.content_dim == 4
input.quality_dim == 1
input.stim_dim == 4
stim.stim_dim == 4
patch.samples > 0
patch.stride == patch.samples
model.sequence_format == stim_eye_triplet_no_cls
model.pretrain_style == bert_masked_reconstruction
model.use_cls == false
model.use_stim_tokens == true
model.broadcast_stim_to_eye == false
model.max_patches > 0
attention.min_nonmissing_frac_for_eye_token in [0, 1]
mask.min_nonmissing_frac_for_mae in [0, 1]
train.precision in {bf16, fp32, fp16}
```

---

# 5. packed_mmap reader

实现位置：

```text
src/eyemae/data.py
```

核心类：

```text
PackedTrialStore
PackedPretrainDataset
```

## 5.1 shard 路径

`shard_id` 必须指向目录：

```python
shard_dir = data_root / "shards" / shard_id

X_path = shard_dir / "X_data.npy"
Y_path = shard_dir / "y_frame.npy"
offsets_path = shard_dir / "X_offsets.npy"
lengths_path = shard_dir / "X_lengths.npy"
trial_index_path = shard_dir / "trial_index.csv"
```

不要实现为：

```text
shards/shard_*.X_data.npy
```

## 5.2 index 字段

`pretrain/*.csv` 每行至少包含：

```text
global_trial_id
shard_id
local_trial_index
frame_offset
frame_length
ml_subject_id
task_id
```

推荐也读取：

```text
trial_id
source_suffix
subject_id
source_top
source_dataset
source_group
source_subtype
```

## 5.3 frame_offset / frame_length 是权威

读取 trial 时使用 CSV 中的：

```python
start = int(row["frame_offset"])
length = int(row["frame_length"])
end = start + length
```

`X_offsets.npy` / `X_lengths.npy` 只用于校验：

```python
local_idx = int(row["local_trial_index"])
assert start == int(offsets[local_idx])
assert length == int(lengths[local_idx])
```

如果不一致，抛出包含 `global_trial_id`、`shard_id`、`local_trial_index` 的 `ValueError`。

## 5.4 mmap 与 shard cache

每个 worker 懒加载 shard：

```python
np.load(path, mmap_mode="r")
```

不要在 Dataset `__init__` 中把大数组读进内存。

不要在 `__getitem__` 每次重新 `np.load`。

实现 worker 内 LRU cache：

```yaml
data:
  max_open_shards_per_worker: 44
```

如果访问的 shard 数超过该限制，移除最久未使用的 shard 句柄。

## 5.5 frame column schema

`X_data.npy` 10列：

```text
0 left_x
1 left_y
2 left_s
3 right_x
4 right_y
5 right_s
6 stimulus_x
7 stimulus_y
8 stimulus_on
9 cross_on
```

`y_frame.npy` 2列：

```text
0 left_qc_label
1 right_qc_label
```

qc label：

```text
0 = nonblink / valid
1 = blink
2 = missing
```

映射为内部 trial dict：

```python
eye[:, 0] = X[:, 0]  # left_x
eye[:, 1] = X[:, 1]  # left_y
eye[:, 2] = X[:, 2]  # left_s
eye[:, 3] = Y[:, 0]  # left_qc_label

eye[:, 4] = X[:, 3]  # right_x
eye[:, 5] = X[:, 4]  # right_y
eye[:, 6] = X[:, 5]  # right_s
eye[:, 7] = Y[:, 1]  # right_qc_label

fix_on = X[:, 9]     # cross_on

stim[:, 0] = X[:, 8] # stimulus_on
stim[:, 1] = X[:, 6] # stimulus_x
stim[:, 2] = X[:, 7] # stimulus_y

subject_id = row["ml_subject_id"]
trial_id = row["global_trial_id"]
global_trial_id = row["global_trial_id"]
task_id = row["task_id"]
```

内部 `stim_patch` 顺序必须始终为：

```text
[fix_on, stim_on, stim_x_norm, stim_y_norm]
```

## 5.6 trial_id / global_trial_id

下游和预训练统一：

```python
trial["trial_id"] = row["global_trial_id"]
trial["global_trial_id"] = row["global_trial_id"]
```

如果 index 同时有 `trial_id`，也保留：

```python
trial["source_trial_id"] = row["trial_id"]
```

模型训练和预测输出优先使用：

```text
global_trial_id
```

## 5.7 task_id 来源

正式 packed_mmap 读取时：

```text
task_id 直接来自 index CSV 的 task_id 字段。
```

不要再从旧 manifest 动态映射 task_id。

---

# 6. split 与 audit

正式预训练直接读取已生成 split：

```text
pretrain/pretrain_train.csv
pretrain/pretrain_validation.csv
pretrain/pretrain_test.csv
```

不重新划分。

必须读取并检查：

```text
pretrain/pretrain_split_summary.json
audit_summary.json
```

训练前必须验证：

```text
同一个 ml_subject_id 不跨 train/validation/test
pretrain_split_summary.json 存在
audit_summary.json 存在
global_trial_id 唯一
task_id 只允许 0/1/2/3
CSV 必需列存在
```

如果 audit 不通过，训练停止。

配置：

```yaml
split:
  strategy: provided_subject_heldout
  subject_key: ml_subject_id
  audit_required: true
  split_summary: pretrain/pretrain_split_summary.json
```

---

# 7. max_patches preflight

模型配置：

```yaml
model:
  max_patches: 384
```

训练启动前必须检查所有使用 split 的最大 patch 数：

```python
num_patches = frame_length // patch.samples
max_required_patches = max(num_patches over train/validation/test)
```

如果：

```python
max_required_patches > model.max_patches
```

则报错并停止：

```text
max_patches too small; do not silently truncate trials.
```

不要静默截断 trial。

---

# 8. label / blink / missing / eye availability

每帧 label：

```text
0 = nonblink / valid
1 = blink
2 = missing
```

处理：

```python
missing = label == 2
blink = (label == 1) & (~missing)
```

如果 `label == 2`：

```text
missing = 1
blink = 0
x/y/area输入置0
该帧不计算xy/area/blink重建loss
```

## 8.1 subject eye availability 使用后缀更权威

正式 packed 数据中，subject / source 后缀仍然作为**眼别可用性审计与分组的权威来源**。

优先读取 index 中的：

```text
source_suffix
```

若没有，则尝试从以下字段最后一位解析：

```text
subject_id
ml_subject_id
```

合法后缀：

```text
D = both eyes available
L = left-eye only
R = right-eye only
```

规则：

```text
subject_eye_availability 分组指标使用后缀。
可视化/metrics 中 single-eye vs both-eye 使用后缀。
```

frame-level missing 仍然来自 `y_frame` 的 qc label。若后缀与 qc label 冲突：

```text
训练时不静默修正原始 y_frame。
记录 audit warning。
subject_eye_availability 分组以 suffix 为准。
loss / eye_token_valid 仍以 frame-level qc label 为准。
```

如果需要强制后缀不可用眼为 missing，必须显式开启：

```yaml
data:
  enforce_suffix_eye_availability: false
```

第一版默认：

```text
false
```

这样既保留 suffix 作为权威分组依据，又不在训练时隐式改写已生成数据。

---

# 9. NaN / inf 处理

`validate_trial` 必须检查：

```text
不允许 inf
```

NaN 规则：

## 9.1 eye 通道 NaN

如果某只眼任意 `x/y/s` 出现 NaN：

```text
该眼该帧 label 强制视为 missing
该眼该帧 x/y/s 置0
记录 warning 计数
```

## 9.2 stim 通道 NaN

如果：

```text
stim_on == 0
```

则：

```text
stim_x = 0
stim_y = 0
```

如果：

```text
stim_on == 1 and (stim_x or stim_y is NaN)
```

则：

```text
raise ValueError
```

## 9.3 fix_on / stim_on NaN

如果 `fix_on` 或 `stim_on` 出现 NaN：

```text
raise ValueError
```

---

# 10. preprocessing

## 10.1 坐标归一化

坐标已经是视觉角度，不做 trial z-score。

```python
x_norm = clip(x_deg, -30, 30) / 30
y_norm = clip(y_deg, -20, 20) / 20
stim_x_norm = clip(stim_x_deg, -30, 30) / 30
stim_y_norm = clip(stim_y_deg, -20, 20) / 20
```

当：

```python
missing == True or blink == True
```

时：

```python
x_norm = 0
y_norm = 0
area_norm = 0
```

但保留：

```text
blink
missing
```

不要用 `x=0,y=0,area=0` 自动推断 missing。

## 10.2 area stats

第一版只用：

```text
pretrain/pretrain_train.csv
```

统计 area median/MAD。

不要用：

```text
pretrain/pretrain_validation.csv
pretrain/pretrain_test.csv
```

统一 area stats 文件名：

```text
outputs/area_stats_fast_packed_full_subject_seed42.json
```

pretrain 和 downstream 都默认使用这同一份 stats。

后续 ablation 必须使用旧版等价的 subject-level robust 统计方式：

```text
对 pretrain/pretrain_train.csv 中每个 subject 的全部 trial 做扫描
只纳入非 missing、非 blink、area > 0、眼睛可用的 eye-area sample
左右眼分别贡献 sample，同一帧双眼有效则计为 2 个 sample
num_valid_frames 记录该 subject 全部有效 eye-area sample 数
median/MAD 使用 reservoir sampling 控制内存，但 sampling 不得提前停止 subject 的 trial 扫描
```

也就是说，`max_frames_per_subject` 只能限制进入 median/MAD 估计的样本池大小，不能像当前这版临时实现一样在 subject sample 数达到上限后直接 `break`。当前已经在跑的预训练不回改；后续 ablation 先重新计算这份 full-subject stats，再使用它训练。

命令：

```bash
python -m eyemae.compute_area_stats \
  --config configs/eyemae_cnn_512_12l.yaml \
  --split train \
  --out outputs/area_stats_fast_packed_full_subject_seed42.json
```

配置：

```yaml
area:
  use_log1p: true
  robust_zscore_by: subject
  clip: 5
  eps: 1.0e-6
  stats_source: pretrain_train_only
  stats_path: outputs/area_stats_fast_packed_full_subject_seed42.json
  fallback_to_global: true
  max_frames_per_subject: 200000
  max_global_frames: 2000000
```

## 10.3 stimulus condition

第一版 stim feature 固定为：

```text
fix_on
stim_on
stim_x_norm
stim_y_norm
```

当 `stim_on=0`：

```text
stim_x_norm = 0
stim_y_norm = 0
```

---

# 11. patching

采样率固定：

```text
1000Hz
```

第一版：

```yaml
patch:
  samples: 20
  stride: 20
```

每个 trial：

```python
N = frame_length // 20
```

丢弃最后不足20帧的尾巴。

如果 `N == 0`：

```text
该trial跳过，并记录 warning。
```

返回 batch：

```python
batch = {
    "content": FloatTensor[B, Nmax, 2, 20, 4],
    "quality": FloatTensor[B, Nmax, 2, 20, 1],
    "stim": FloatTensor[B, Nmax, 20, 4],
    "task_id": LongTensor[B],
    "pad_mask": BoolTensor[B, Nmax],
    "eye_nonmissing_frac": FloatTensor[B, Nmax, 2],
    "eye_token_valid": BoolTensor[B, Nmax, 2],
    "subject_id": list[str],
    "ml_subject_id": list[str],
    "trial_id": list[str],
    "global_trial_id": list[str],
    "subject_eye_availability": list[str],
}
```

`stim` patch 内部顺序：

```text
0 fix_on
1 stim_on
2 stim_x_norm
3 stim_y_norm
```

---

# 12. batching

第一版支持：

```text
fixed trial batch
token-based dynamic batch
```

主训练建议使用 token-based dynamic batch。

配置：

```yaml
train:
  batch_trials_per_gpu: null
  max_seq_tokens_per_gpu: 90000
  max_trials_per_gpu: 256
  bucket_by_length: true
```

token budget 计算按 padding 后实际进入 Transformer 的 token 数：

```python
Nmax = max(N_i for trial_i in batch)
seq_tokens = 3 * Nmax * len(batch)
seq_tokens <= max_seq_tokens_per_gpu
```

`bucket_by_length: true` 时：

```text
按 frame_length / num_patches 做 length bucketing，减少 padding。
```

---

# 13. attention mask

sequence：

```text
S_0, L_0, R_0, S_1, L_1, R_1, ...
```

attention padding mask：

```python
seq_attn_pad_mask: BoolTensor[B, 3 * Nmax]

seq_attn_pad_mask[:, 0::3] = pad_mask
seq_attn_pad_mask[:, 1::3] = pad_mask | (~eye_token_valid[:, :, 0])
seq_attn_pad_mask[:, 2::3] = pad_mask | (~eye_token_valid[:, :, 1])
```

注意：

```text
MAE-masked eye token不进入 seq_attn_pad_mask。
MAE-masked eye token仍然参与attention。
```

Transformer 输出后：

```python
hidden = hidden.masked_fill(seq_attn_pad_mask[..., None], 0.0)
```

---

# 14. MAE mask

输出：

```python
mae_mask: BoolTensor[B, Nmax, 2]
```

只对应：

```text
L_i / R_i eye tokens
```

不包含：

```text
S_i stim token
```

候选：

```python
mae_eligible = (
    (~pad_mask[:, :, None])
    & eye_token_valid
    & (eye_nonmissing_frac >= mask.min_nonmissing_frac_for_mae)
)
```

配置：

```yaml
mask:
  ratio_min: 0.35
  ratio_max: 0.65
  short_span_prob: 0.70
  short_span_len_min: 2
  short_span_len_max: 8
  long_span_prob: 0.15
  long_span_len_min: 10
  long_span_len_max: 25
  single_eye_mask_prob: 0.25
  min_nonmissing_frac_for_mae: 0.50
```

如果某 trial eligible 数少于2：

```text
该trial mae_mask全False，不贡献重建loss。
不要死循环。
```

mask 实现不能使用无限 `while`；必须 sample without replacement。

---

# 15. 模型结构

必须复用现有 EyeMAE 结构：

```text
S_i, L_i, R_i sequence
CNN tokenizer
bidirectional Transformer encoder
prediction head only for eye tokens
```

配置：

```yaml
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
```

S token：

```python
S_i = stim_token_i + task_embedding + token_type_embedding["stim"] + time_embedding_i
```

Eye token：

```python
eye_token = content_token_or_mask_token + quality_token + task_embedding + token_type_embedding["left/right"] + time_embedding_i
```

prediction head 输出：

```python
pred: [B, N, 2, 20, 4]
```

---

# 16. loss

输入：

```python
pred:       [B, N, 2, 20, 4]
target:     [B, N, 2, 20, 4]
quality:    [B, N, 2, 20, 1]
mae_mask:   [B, N, 2]
pad_mask:   [B, N]
eye_token_valid: [B, N, 2]
```

展开：

```python
missing = quality[..., 0].bool()
target_blink = target[..., 3] > 0.5
nonpad = (~pad_mask)[:, :, None, None]
eye_valid = eye_token_valid[:, :, :, None]
mae_frame = mae_mask[:, :, :, None].expand_as(missing)
```

valid masks：

```python
coord_valid = mae_frame & nonpad & eye_valid & (~missing) & (~target_blink)
blink_valid = mae_frame & nonpad & eye_valid & (~missing)
```

loss：

```text
xy: SmoothL1 on x/y, coord_valid
area: SmoothL1 on area, coord_valid
blink: BCEWithLogits, blink_valid
velocity: patch内部差分，v_valid = coord_valid[...,1:] & coord_valid[...,:-1]
```

所有 loss 使用 numerator / denominator，denominator=0 时该项返回0，不产生 NaN。

如果整个 batch 总有效 denominator 为0：

```text
跳过 optimizer step，记录 warning。
```

---

# 17. metrics / baselines / visualization

validation 记录：

```text
total_loss
xy_loss
area_loss
blink_loss
velocity_loss
masked_xy_rmse_deg
masked_x_rmse_deg
masked_y_rmse_deg
masked_area_mae
masked_blink_bce
masked_blink_auc
masked_velocity_rmse_deg_per_ms
```

分组：

```text
task_id
eye
trial_length
missing_fraction
subject_eye_availability  # 来自 suffix / source_suffix
eye_token_valid_fraction
mask_type if implemented
```

AUC 边界：

```text
若正负样本不足，返回NaN，不报错。
```

baseline：

```text
previous value
linear interpolation
```

插值边界：

```text
开头无previous -> 使用next valid
结尾无next -> 使用previous valid
两边都无valid -> 跳过该段
```

visualization：

```text
rank0保存
matplotlib Agg
每次validation最多16个trial
从整个validation pass中选择代表性样本
```

---

# 18. DDP / checkpoint / evaluate

DDP：

```text
使用 LOCAL_RANK / RANK / WORLD_SIZE
torch.cuda.set_device(local_rank)
rank0 写 TensorBoard / checkpoint / visualization
validation metrics 跨rank all_reduce numerator/denominator
```

checkpoint 保存：

```text
model
optimizer
scheduler
global_step
epoch
config
area_stats_path
best_metric
rng states
```

evaluate 支持：

```bash
python -m eyemae.evaluate \
  --config configs/eyemae_cnn_512_12l.yaml \
  --checkpoint outputs/exp/checkpoint_best.pt \
  --split validation

python -m eyemae.evaluate \
  --config configs/eyemae_cnn_512_12l.yaml \
  --checkpoint outputs/exp/checkpoint_best.pt \
  --split test
```

---

# 19. 第一版主配置

创建：

```text
configs/eyemae_cnn_512_12l.yaml
```

内容：

```yaml
experiment:
  name: eyemae_cnn_512_12l_patch20_stimtoken_nolaststim_nogoal_nocls_velloss
  output_dir: outputs/eyemae_cnn_512_12l_patch20_stimtoken

data:
  format: packed_mmap
  data_dir: /mnt/disk_sde/data-260606/extracted/eyemae_fast_dataset_v1

  train_index: pretrain/pretrain_train.csv
  val_index: pretrain/pretrain_validation.csv
  test_index: pretrain/pretrain_test.csv
  subject_key: ml_subject_id
  task_column: task_id
  trial_id_column: global_trial_id
  subject_eye_availability_column: source_suffix

  mmap_mode: r
  max_open_shards_per_worker: 16
  enforce_suffix_eye_availability: false
  sampling_rate: 1000

split:
  strategy: provided_subject_heldout
  subject_key: ml_subject_id
  audit_required: true
  split_summary: pretrain/pretrain_split_summary.json

label:
  nonblink_value: 0
  blink_value: 1
  missing_value: 2

normalization:
  x_clip_deg: 30
  y_clip_deg: 20

area:
  use_log1p: true
  robust_zscore_by: subject
  clip: 5
  eps: 1.0e-6
  stats_source: pretrain_train_only
  stats_path: outputs/area_stats_fast_packed_full_subject_seed42.json
  fallback_to_global: true
  max_frames_per_subject: 200000
  max_global_frames: 2000000

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

  use_rope: false
  use_gqa: false
  use_cross_attention: false
  use_adaln: false

mask:
  ratio_min: 0.35
  ratio_max: 0.65
  short_span_prob: 0.70
  short_span_len_min: 2
  short_span_len_max: 8
  long_span_prob: 0.15
  long_span_len_min: 10
  long_span_len_max: 25
  single_eye_mask_prob: 0.25
  min_nonmissing_frac_for_mae: 0.50

loss:
  xy_weight: 1.0
  area_weight: 0.2
  blink_weight: 0.1
  velocity_weight: 0.1
  loss_only_on_mae_mask: true
  ignore_missing_for_all_losses: true
  ignore_blink_for_xy_area_loss: true
  velocity_within_patch_only: true
  no_loss_on_stim_tokens: true

train:
  seed: 42
  precision: bf16
  distributed: ddp
  optimizer: adamw
  lr: 2.0e-4
  min_lr: 2.0e-5
  betas: [0.9, 0.95]
  weight_decay: 0.05
  grad_clip: 1.0

  warmup_steps: 2000
  max_steps: 100000
  val_every_steps: 1000
  save_every_steps: 2000
  log_every_steps: 50

  batch_trials_per_gpu: null
  max_seq_tokens_per_gpu: 90000
  max_trials_per_gpu: 256
  bucket_by_length: true
  grad_accum_steps: 1

  num_workers: 8
  pin_memory: true
  persistent_workers: true
  gradient_checkpointing: false
  timing_every_steps: 0

checkpoint:
  monitor: val/masked_xy_rmse_deg
  mode: min

eval:
  seed: 1234
  fixed_mask: true
  # null means every validation, preserving the first-version behavior.
  # Set a positive interval to reduce validation overhead in future speed runs.
  group_metrics_every_steps: null
  visualization_every_steps: null
```

---

# 20. tests

必须覆盖：

```text
packed_mmap reader:
  shard_id目录读取正确
  X_data/y_frame/offsets/lengths mmap读取正确
  frame_offset/frame_length 是权威
  offsets/lengths 校验生效
  global_trial_id -> trial_id 映射正确
  task_id 直接来自 index CSV

pretrain split audit:
  pretrain_train.csv / pretrain_validation.csv / pretrain_test.csv 存在
  pretrain/pretrain_split_summary.json 存在
  ml_subject_id 不跨 split
  tests/test_splits.py 只做 audit，不重新生成 split

preprocess:
  stim patch 顺序为 [fix_on, stim_on, stim_x_norm, stim_y_norm]
  stim_on=0 时 stim_x/y 强制0
  eye NaN 转 missing
  stim_on/fix_on NaN 报错
  stim_on=1 且 stim_x/y NaN 报错
  subject_eye_availability 来源 suffix/source_suffix

max_patches:
  超过 model.max_patches 启动报错

mask/loss/model:
  S/L/R sequence mapping
  MAE mask 不包含 S token
  MAE-masked eye token attention-valid
  all-missing eye token attention-masked
  loss gating 不在 missing/blink/padding/invalid token 上算错误loss
```

---

# 21. 训练流程

## 21.1 计算正式 area stats

```bash
python -m eyemae.compute_area_stats \
  --config configs/eyemae_cnn_512_12l.yaml \
  --split train \
  --out outputs/area_stats_fast_packed_full_subject_seed42.json
```

## 21.2 检查 packed pretrain 数据

```bash
pytest tests/test_fast_packed_dataset.py
pytest tests/test_splits.py
pytest tests/test_area_stats.py
```

## 21.3 GPU 主训练

第一版正式 baseline 可以使用空闲 GPU1-4 加速：

```bash
CUDA_VISIBLE_DEVICES=1,2,3,4 torchrun --standalone --nproc_per_node=4 \
  -m eyemae.train \
  --config configs/eyemae_cnn_512_12l.yaml
```

后续 ablation 每个变体固定使用一张 GPU，以免 GPU 数改变导致 throughput、batch exposure 和 batch normless optimizer dynamics 不可比：

```bash
CUDA_VISIBLE_DEVICES=<one_gpu_id> torchrun --standalone --nproc_per_node=1 \
  -m eyemae.train \
  --config configs/eyemae_cnn_512_12l_<ablation>.yaml
```

## 21.4 validation / test evaluation

```bash
python -m eyemae.evaluate \
  --config configs/eyemae_cnn_512_12l.yaml \
  --checkpoint outputs/eyemae_cnn_512_12l_patch20_stimtoken/checkpoint_best.pt \
  --split validation

python -m eyemae.evaluate \
  --config configs/eyemae_cnn_512_12l.yaml \
  --checkpoint outputs/eyemae_cnn_512_12l_patch20_stimtoken/checkpoint_best.pt \
  --split test
```

---

# 22. 不要做的事

第一版不要做：

```text
不要重新构建 eyemae_fast_dataset_v1
不要实现 packed converter
不要重新生成 pretrain split
不要使用 pretrain_val.csv
不要使用 pretrain/split_summary.json
不要把 stim 顺序写成 [stim_x, stim_y, stim_on, fix_on]
不要用 X_offsets/X_lengths 作为权威而忽略 CSV frame_offset/frame_length
不要静默截断超过 max_patches 的 trial
不要在 stim_on=1 且 stim坐标 NaN 时置0
不要把missing当成MAE mask
不要对S_i stim token做MAE mask
不要对S_i stim token算reconstruction loss
不要把MAE-masked eye token从attention里删掉
不要用 pretrain_validation.csv / pretrain_test.csv 统计 area stats
不要实现 session-level split
```

---

# 23. 后续 ablation 计划

后续 ablation 均基于同一 provided pretrain train/validation/test split。所有实验必须固定：

```text
same data_dir
same pretrain train/validation/test csv
same area_stats_path
same random seed
same eval mask seed
same max_steps
same token budget / batch rule
same GPU count: one GPU per ablation variant
same validation protocol
same metrics
```

除非该 ablation 本身就是研究训练长度、batch 或 seed，否则一次只改一个因素。第一版 baseline 必须保持：

```yaml
model:
  pretrain_style: bert_masked_reconstruction
  sequence_format: stim_eye_triplet_no_cls
  tokenizer: cnn
  d_model: 512
  n_layers: 12
  n_heads: 8

patch:
  samples: 20
  stride: 20

stim:
  use_last_stim_xy: false
  use_goal_xy: false

loss:
  velocity_weight: 0.1
```

重要说明：

```text
第一版是 BERT-style masked reconstruction。
true/asymmetric MAE 不是第一版主线，只在 Ablation K 中做。
```

---

## 23.1 主 baseline

名称建议：

```text
pretrain_baseline_bertstyle_cnn_patch20_stimtoken_nolaststim_nogoal_velocity
```

配置核心：

```yaml
model:
  pretrain_style: bert_masked_reconstruction
  tokenizer: cnn
  sequence_format: stim_eye_triplet_no_cls
  d_model: 512
  n_layers: 12
  n_heads: 8
  ffn_hidden: 1536
  use_cls: false
  use_stim_tokens: true
  broadcast_stim_to_eye: false

patch:
  samples: 20
  stride: 20

stim:
  use_last_stim_xy: false
  use_goal_xy: false

loss:
  velocity_weight: 0.1
```

必须记录：

```text
masked_xy_rmse_deg
masked_blink_auc
long_span masked_xy_rmse_deg
previous-value baseline
linear-interpolation baseline
per-task metrics
by subject_eye_availability metrics
```

---

## 23.2 Ablation A：add_last_stim

目的：

```text
判断记忆眼跳中显式提供过去刺激位置是否帮助重建。
```

只改：

```yaml
stim:
  use_last_stim_xy: true
  stim_dim: 6
```

S_i stim feature 顺序变成：

```text
[fix_on, stim_on, stim_x_norm, stim_y_norm, last_stim_x_norm, last_stim_y_norm]
```

`last_stim_x/y` 定义：

```text
stim_on=1:
  last_stim = current stim

stim_on=0:
  last_stim = 最近一次 stim_on=1 的位置

trial开头从未出现stim:
  last_stim = 0
```

重点比较：

```text
memory task
stim消失后的patch
long_span mask
late response phase
```

不要同时加入 goal。

---

## 23.3 Ablation B：with_goal

目的：

```text
判断显式提供任务规则后的理想目标是否帮助重建。
```

只改：

```yaml
stim:
  use_goal_xy: true
```

建议与 `last_stim` 分两种：

```text
B1: with_goal, no_last_stim
B2: with_goal, with_last_stim
```

goal 构造：

```text
pro / 正向:
  goal = stim 或 last_stim

anti / 反向:
  goal = -stim 或 -last_stim
  若fixation不在原点，则 goal = 2 * fixation - stim

memory / 记忆:
  goal = remembered last_stim

double / 二次眼跳:
  第一版先用当前 stim/last_stim
  phase-aware goal 后续单独做
```

解释时必须注明：

```text
with_goal 更像“给定理想目标后的行为建模”；
no_goal 更像“从 task/stim history 中学习任务规则”。
```

---

## 23.4 Ablation C：tokenizer

目的：

```text
判断 CNN tokenizer 是否优于简单 MLP tokenizer。
```

### C1 CNN tokenizer

baseline。

### C2 MLP tokenizer

只改：

```yaml
model:
  tokenizer: mlp
```

MLP结构：

```text
content:
  flatten [20,4]
  LayerNorm
  Linear(80 -> d_model)
  GELU
  Linear(d_model -> d_model)

quality:
  flatten [20,1]
  LayerNorm
  Linear(20 -> d_model)

stim:
  flatten [20,4]
  LayerNorm
  Linear(80 -> d_model)
```

比较：

```text
masked_xy_rmse_deg
long_span masked_xy_rmse_deg
blink_auc
training throughput
GPU memory
```

---

## 23.5 Ablation D：velocity loss

目的：

```text
判断速度/一阶差分loss是否改善轨迹动态重建。
```

比较：

```yaml
loss:
  velocity_weight: 0.0

loss:
  velocity_weight: 0.1

loss:
  velocity_weight: 0.2
```

重点看：

```text
眼跳起止附近RMSE
long_span mask轨迹形状
重建轨迹是否过度平滑
velocity_rmse_deg_per_ms
```

---

## 23.6 Ablation E：mask策略

目的：

```text
确认模型不是只学插值。
```

### E1 random only

```yaml
mask:
  short_span_prob: 0.0
  long_span_prob: 0.0
```

### E2 default mixed mask

baseline。

### E3 span heavy

```yaml
mask:
  ratio_min: 0.45
  ratio_max: 0.75
  short_span_prob: 0.90
  long_span_prob: 0.30
```

重点比较：

```text
model vs previous-value baseline
model vs linear-interpolation baseline
long_span masked_xy_rmse_deg
memory task
anti task
```

若 random-only 表现很好但 span-heavy 大幅下降，说明 random mask 任务太容易。

---

## 23.7 Ablation F：patch size

目的：

```text
确认20ms patch是否合适。
```

比较：

```yaml
patch:
  samples: 10
  stride: 10

patch:
  samples: 20
  stride: 20

patch:
  samples: 40
  stride: 40
```

对应修改：

```text
Content/Quality/Stim tokenizer 的 flatten 输入长度
velocity loss patch内部长度
max_patches preflight
token budget
```

比较：

```text
训练速度
GPU显存
masked_xy_rmse_deg
blink_auc
眼跳起止附近RMSE
long_span表现
```

---

## 23.8 Ablation G：bilateral token

目的：

```text
判断左右眼分开建 token 是否优于合并成双眼 token。
```

### G1 eye-wise tokens

baseline：

```text
S0, L0, R0, S1, L1, R1, ...
```

### G2 bilateral token

```text
S0, E0, S1, E1, ...
```

E_i 包含：

```text
left x/y/area/blink
right x/y/area/blink
left missing
right missing
```

重点比较：

```text
单眼缺失trial
single-eye artificial mask
左右眼互补能力
missing fraction bucket
```

注意：

```text
bilateral token 可能更难处理单眼缺失；
需要明确 loss 仍然按 left/right frame gating。
```

---

## 23.9 Ablation H：add CLS

目的：

```text
观察 CLS token 是否帮助后续 trial-level embedding。
```

序列：

```text
CLS, S0, L0, R0, S1, L1, R1, ...
```

配置：

```yaml
model:
  use_cls: true
```

预训练：

```text
CLS 不做 reconstruction target
eye reconstruction loss 不变
```

下游：

```text
可比较 eye_mean pooling vs CLS pooling
```

---

## 23.10 Ablation I：feature-wise tokens

目的：

```text
测试是否需要像 EEG channel token 那样把不同特征拆开。
```

序列：

```text
S_i,
Lx_i, Ly_i, Larea_i, Lblink_i,
Rx_i, Ry_i, Rarea_i, Rblink_i
```

注意：

```text
seq_len 从 3N 变成 9N
attention 开销显著增加
```

第一版后期才做。

比较：

```text
masked_xy_rmse_deg
blink_auc
GPU memory
throughput
long_span表现
```

---

## 23.11 Ablation J：overlap patch

目的：

```text
测试20ms patch、10ms stride是否改善眼跳边界。
```

配置：

```yaml
patch:
  samples: 20
  stride: 10
```

重要要求：

```text
overlap patch必须用frame interval mask。
不能只随机mask单个overlap patch。
```

原因：

```text
patch_i   = 0-20ms
patch_i+1 = 10-30ms

如果只mask patch_i，但patch_i+1可见，
patch_i一半原始帧已经泄漏。
```

实现要求：

```text
如果一个原始frame被mask，
所有包含这个frame的overlap patch content都应视为masked。
```

---

## 23.12 Ablation K：asymmetric MAE-style encoder-decoder

目的：

```text
比较第一版 BERT-style masked reconstruction 和真正 MAE-style asymmetric encoder-decoder 的差异。
```

baseline：

```yaml
model:
  pretrain_style: bert_masked_reconstruction
```

ablation：

```yaml
model:
  pretrain_style: asymmetric_mae
```

true/asymmetric MAE 逻辑：

```text
encoder:
  只输入 S_i 和 visible eye tokens。
  masked eye tokens 不进入 encoder。

decoder:
  接收 encoder 输出 + masked eye placeholders。
  重建 masked eye patches。

all-missing eye tokens:
  不进入 encoder。
  不进入 decoder。
  不算 loss。
```

注意：

```text
这不是第一版主线。
不要把 asymmetric_mae 和第一版 bert_masked_reconstruction 混在一起。
```

比较：

```text
训练速度
显存
masked_xy_rmse_deg
long_span表现
downstream transfer
```

---

## 23.13 Ablation L：model scale / parameter count

目的：

```text
确认模型容量是否适合400h/约4000 subject的预训练规模。
```

比较：

```yaml
small:
  d_model: 256
  n_layers: 6
  n_heads: 4
  ffn_hidden: 768
  approx_params: 6.49M

base:
  d_model: 512
  n_layers: 12
  n_heads: 8
  ffn_hidden: 1536
  approx_params: 43.80M

larger:
  d_model: 768
  n_layers: 16
  n_heads: 12
  ffn_hidden: 2048
  approx_params: 117.82M
```

第一轮优先：

```text
small 256x6 vs base 512x12
```

如果 small 明显欠拟合或 base 明显更好，再考虑 larger。

判断不要只看 reconstruction，也要看 downstream：

```text
pretrain:
  masked_xy_rmse_deg
  long_span masked_xy_rmse_deg
  blink_auc

downstream:
  linear probe AUROC
  partial/full fine-tune AUROC
  scratch差距
```

---

## 23.14 Ablation M：Qwen-style block

目的：

```text
检查更现代的Transformer block细节是否改善表示。
```

建议顺序：

```text
M1 learned time embedding -> RoPE
M2 RoPE + GQA
M3 RoPE + GQA + QK norm
```

不要一次性全改，否则无法判断收益来源。

---

## 23.15 Ablation N：condition fusion

第一版：

```text
independent S_i token
eye token 通过 attention 读取 S_i
```

候选：

```text
N1 independent S_i token
N2 broadcast stim_token 到 L/R eye token
N3 independent S_i + broadcast stim_token
N4 concat + MLP fusion
N5 task AdaLN + independent S_i
N6 two-stream eye/stim cross-attention
```

重点比较：

```text
memory task
anti task
downstream transfer
training stability
```

---

## 23.16 Ablation O：event-aware / event-driven tokenization

第一版仍使用固定20ms patch。

后续逐步试：

```text
O1 fixed 20ms patch + event-aware mask
   更高概率mask眼跳起点、终点、高速度片段。

O2 variable-length event segment token
   fixation / saccade / blink / missing 分段后建token。

O3 event boundary tokenization without event label input
   用事件边界切token，但不把事件类别作为输入，减少泄漏。
```

这类实验工程复杂度高，应放在固定patch baseline稳定之后。

---

## 23.17 实验优先级

建议顺序：

```text
0. pytest全部通过
1. debug tiny overfit
2. 主baseline：BERT-style + CNN + 20ms + S/L/R + no_last_stim + no_goal + no_CLS + velocity
3. previous value / linear interpolation baseline
4. random mask vs span mask
5. no_velocity vs velocity
6. add_last_stim
7. MLP tokenizer vs CNN tokenizer
8. patch 10/20/40ms
9. with_goal
10. bilateral token
11. add CLS
12. feature-wise tokens
13. overlap patch
14. asymmetric MAE-style encoder-decoder
15. model scale: 256×6 small vs 512×12 base
16. 768×16 larger model
17. RoPE/GQA/Qwen-style block
18. condition fusion variants
19. event-aware / event-driven tokenization
```
