# EyeMAE 第一版预训练：Codex 工程实现计划

## 0. 目标

实现并训练一个眼动数据自监督预训练模型：

```text
Task-conditioned EyeMAE / EyeBERT-style masked reconstruction

输入：
  1000Hz trial级眼动时间序列
  每20ms一个patch
  每个patch构造一个stim/task/time token
  每个patch构造左眼、右眼两个eye token
  对有效eye token进行BERT-style人工mask
  重建被mask的 x, y, pupil area, blink

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

第一版重点不是堆大模型，而是把以下机制写干净：

```text
blink机制
missing机制
MAE人工mask机制
stim/task/time独立条件token机制
attention validity机制
loss gating机制
padding机制
pretrain split机制
area statistics机制
validation和可视化机制
DDP训练机制
```

---

# 1. 第一版核心决定

第一版固定如下：

```text
数据：
  1000Hz trial级眼动时间序列
  每20ms一个patch
  左眼、右眼分别建eye token
  每个patch额外建一个stim/task/time token

输入：
  eye content = x, y, area, blink
  eye quality = missing
  stim = stim_x, stim_y, stim_on, fix_on
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

第一版不使用：
  last_stim_x/y
  goal_x/y
  CLS token
  feature-wise tokenization
  cross-attention
  adaptive layer norm
  DeepSeek MoE
  decoder-only causal LM
  true asymmetric MAE
  session-level split
```

---

# 2. 工程里程碑

请按里程碑实现，不要一次性实现所有ablation。

## Milestone 1：最小可运行与单元测试

必须完成：

```text
pyproject.toml / editable install
config loader + validate_config
synthetic npz fixtures
data loading
preprocess
area stats loader
patching
masking
model forward
loss
shape tests
mask tests
loss gating tests
```

验收：

```bash
pip install -e .
pytest tests/test_config.py
pytest tests/test_preprocess_missing.py
pytest tests/test_patchify.py
pytest tests/test_masking.py
pytest tests/test_loss_gating.py
pytest tests/test_model_shapes.py
pytest tests/test_sequence_mapping.py
pytest tests/test_attention_mask.py
```

## Milestone 2：单GPU debug和tiny overfit

必须完成：

```text
单GPU train
checkpoint save/resume
TensorBoard logging
validation loop
visualization rank0/single-process保存
tiny overfit
```

验收：

```bash
python -m eyemae.train --config configs/debug.yaml
python -m eyemae.train --config configs/debug.yaml --overfit_trials 64
```

## Milestone 3：真实npz、split、area stats、baselines

必须完成：

```text
make_splits.py
compute_area_stats.py
pretrain_train/pretrain_val/pretrain_test split
previous value baseline
linear interpolation baseline
evaluate.py
```

验收：

```bash
python -m eyemae.make_splits --config configs/debug.yaml
python -m eyemae.compute_area_stats --config configs/debug.yaml
python -m eyemae.evaluate --config configs/debug.yaml --checkpoint /path/to/checkpoint.pt --split pretrain_val
```

## Milestone 4：3GPU DDP主训练

必须完成：

```text
DDP初始化
DistributedSampler或token batch sampler的rank划分
跨rank validation metrics all_reduce
rank0保存checkpoint、TensorBoard、可视化
```

验收：

```bash
torchrun --standalone --nproc_per_node=3 \
  -m eyemae.train \
  --config configs/eyemae_cnn_512_12l.yaml
```

---

# 3. 项目结构

创建如下目录：

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
      make_splits.py
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

  scripts/
    make_debug_split.sh
    make_pretrain_subject_split.sh
    compute_area_stats.sh
    train_debug_1gpu.sh
    train_base_3gpu.sh
    eval_checkpoint.sh
    ddp_smoke_test.sh
    export_embeddings.sh

  tests/
    fixtures/
      make_synthetic_npz.py
    test_config.py
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

第一版只需要实现主配置和debug配置。`configs/ablations/` 可以后续再加，不要求第一版创建所有ablation yaml。

---

# 4. 安装与运行方式

项目必须支持 editable install：

```bash
pip install -e .
```

`pyproject.toml` 至少包含：

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "eyemae"
version = "0.1.0"
requires-python = ">=3.10"

[tool.setuptools.packages.find]
where = ["src"]
```

所有命令必须能以模块方式运行：

```bash
python -m eyemae.train --config configs/debug.yaml
python -m eyemae.evaluate --config configs/debug.yaml --checkpoint /path/to/checkpoint.pt
python -m eyemae.make_splits --config configs/debug.yaml
python -m eyemae.compute_area_stats --config configs/debug.yaml
```

---

# 5. Config系统

实现：

```text
src/eyemae/config.py
```

需要支持：

```python
load_config(path: str) -> dict
deep_merge(base: dict, override: dict) -> dict
validate_config(cfg: dict) -> None
```

第一版采用：

```text
debug.yaml 是完整配置文件，不依赖隐式继承。
eyemae_cnn_512_12l.yaml 也是完整配置文件。
```

不要让 debug.yaml 依赖未说明的 base config，避免缺字段。

`validate_config` 必须检查：

```text
data.format == npz_per_trial
data.data_dir 存在，除非使用 synthetic fixture
data.pretrain_train_split / pretrain_val_split 存在，除非 make_splits 阶段
label nonblink/blink/missing 三个值互不相同
input.content_dim == 4
input.quality_dim == 1
input.stim_dim == stim.stim_dim
stim.stim_dim == 4 for first version
patch.samples > 0
patch.stride == patch.samples for first version
model.sequence_format == stim_eye_triplet_no_cls
model.pretrain_style == bert_masked_reconstruction
model.use_cls == false
model.use_stim_tokens == true
model.broadcast_stim_to_eye == false
attention.min_nonmissing_frac_for_eye_token in [0, 1]
mask.min_nonmissing_frac_for_mae in [0, 1]
train.precision in {bf16, fp32, fp16}
```

---

# 6. 数据接口与npz schema

第一版假设数据形式为：

```text
npz目录：每个trial一个 .npz 文件
```

每个trial读出后统一成：

```python
trial = {
    "eye": FloatTensor[T, 8],
    # dim order:
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
    # fixation只在屏幕中央出现，所以只记录0/1

    "stim": FloatTensor[T, 3],
    # 0 stim_on
    # 1 stim_x_deg
    # 2 stim_y_deg

    "subject_id": str,
    "trial_id": str,
}
```

实现位置：

```text
src/eyemae/data.py
```

## 6.1 split txt格式

所有 split 文件采用：

```text
每行一个相对 data.data_dir 的 .npz 路径
空行忽略
以 # 开头的行忽略
```

示例：

```text
subj001D/trial_0001.npz
subj001D/trial_0002.npz
subj002L/trial_0001.npz
```

`TrialDataset(data_dir, split_file, config)` 读取 split 文件并加载对应 `.npz`。

## 6.2 npz key映射

训练代码不要直接依赖 `.npz` 内部字段名。字段名映射写进config：

```yaml
data:
  npz_keys:
    eye: eye
    task_id: task_id
    fix_on: fix_on
    stim: stim
    subject_id: subject_id
    trial_id: trial_id
```

`subject_id` / `trial_id` 如果 npz 中不存在，则允许从路径或文件名解析，但必须记录 warning。解析规则：

```text
subject_id fallback:
  使用 .npz 文件所在的父目录名。
trial_id fallback:
  使用 .npz 文件名去掉后缀。
```

## 6.3 validate_npz_trial

实现：

```python
def validate_npz_trial(trial: dict, cfg: dict) -> None:
    ...
```

必须检查：

```text
eye.ndim == 2
eye.shape[1] == 8
T = eye.shape[0]
fix_on.shape == [T]
stim.shape == [T, 3]
task_id 是 scalar，且 task_id in {0,1,2,3}
left label 和 right label 只允许 {0,1,2}
eye / fix_on / stim 中不允许 inf
允许 NaN 输入，但 preprocess 中必须转为 missing
subject_id 是非空字符串
trial_id 是非空字符串
```

如果校验失败，抛出带文件路径的 `ValueError`。

---

# 7. 预训练 split 方案

这里的 split 只用于自监督预训练 masked reconstruction，不是下游疾病诊断任务的 split。

第一版只实现两种策略：

```text
trial_random
subject_heldout
```

实现位置：

```text
src/eyemae/make_splits.py
tests/test_splits.py
```

## 7.1 debug split：trial_random

用途：

```text
快速debug
tiny overfit
检查loss、mask、padding、attention、可视化是否正常
```

默认比例：

```yaml
split:
  strategy: trial_random
  seed: 42
  train_ratio: 0.98
  val_ratio: 0.02
  test_ratio: 0.00
  group_by_base_subject_id: false
```

输出目录示例：

```text
splits/debug_trial_random_seed42/
  pretrain_train.txt
  pretrain_val.txt
  pretrain_test.txt
  split_summary.json
```

即使 `test_ratio=0.00`，也创建空的 `pretrain_test.txt`。

## 7.2 正式预训练 split：subject_heldout

用途：

```text
正式预训练
ablation比较
最终自监督预训练泛化评估
```

默认比例：

```yaml
split:
  strategy: subject_heldout
  seed: 42
  train_ratio: 0.90
  val_ratio: 0.05
  test_ratio: 0.05
  group_by_base_subject_id: true
```

规则：

```text
按subject分组，而不是按trial分组。
同一个base subject的所有trial只能出现在一个split里。
```

## 7.3 base subject id

subject_id 后缀含义：

```text
D = both eyes available
L = left-eye only
R = right-eye only
```

为了避免同一个人因为 D/L/R 后缀进入不同split，正式 subject_heldout 默认使用 base subject id 分组。

实现：

```python
def get_split_subject_key(subject_id: str, group_by_base_subject_id: bool = True) -> str:
    if group_by_base_subject_id and len(subject_id) > 0 and subject_id[-1] in {"D", "L", "R"}:
        return subject_id[:-1]
    return subject_id
```

如果你确认同一个人不会同时出现 `subj001D / subj001L / subj001R`，这个规则仍然安全。

## 7.4 split summary

`make_splits.py` 输出：

```text
split_summary.json
```

结构：

```json
{
  "strategy": "subject_heldout",
  "seed": 42,
  "train_ratio": 0.90,
  "val_ratio": 0.05,
  "test_ratio": 0.05,
  "group_by_base_subject_id": true,
  "num_train_trials": 0,
  "num_val_trials": 0,
  "num_test_trials": 0,
  "num_train_subjects": 0,
  "num_val_subjects": 0,
  "num_test_subjects": 0,
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

## 7.5 make_splits CLI

支持：

```bash
python -m eyemae.make_splits \
  --config configs/debug.yaml
```

也支持显式参数覆盖：

```bash
python -m eyemae.make_splits \
  --data_dir /path/to/npz \
  --out_dir splits/pretrain_subject_heldout_seed42 \
  --strategy subject_heldout \
  --seed 42 \
  --train_ratio 0.90 \
  --val_ratio 0.05 \
  --test_ratio 0.05 \
  --group_by_base_subject_id
```

`make_splits.py` 必须：

```text
扫描 data_dir 下所有 .npz
读取每个trial的 task_id、subject_id
生成 split txt
生成 split_summary.json
检查比例和任务分布
```

如果 val/test 缺少某个 task_id，不强制失败，但要在 summary 和日志中 warning。

---

# 8. label / blink / missing 语义

第一版明确使用：

```text
label = 0: non-blink
label = 1: blink
label = 2: missing
```

配置：

```yaml
label:
  nonblink_value: 0
  blink_value: 1
  missing_value: 2
```

每只眼单独处理：

```python
missing = label == missing_value
blink = (label == blink_value) & (~missing)
```

如果 `label == 2`，则：

```text
missing = 1
blink = 0
x/y/area输入置0
该帧不计算xy/area/blink重建loss
```

subject后缀提供单眼/双眼信息：

```text
subject_id 后缀 D: 双眼都有
subject_id 后缀 L: 只有左眼，右眼整条trial missing
subject_id 后缀 R: 只有右眼，左眼整条trial missing
```

实现：

```python
def parse_subject_eye_availability(subject_id: str):
    if len(subject_id) == 0:
        raise ValueError("empty subject_id")

    suffix = subject_id[-1]

    if suffix == "D":
        return {"left_available": True, "right_available": True, "suffix": "D"}
    elif suffix == "L":
        return {"left_available": True, "right_available": False, "suffix": "L"}
    elif suffix == "R":
        return {"left_available": False, "right_available": True, "suffix": "R"}
    else:
        raise ValueError(f"Unknown subject suffix: {subject_id}")
```

最终missing定义：

```python
left_missing = left_label == missing_value
right_missing = right_label == missing_value

if subject_suffix == "L":
    right_missing[:] = True

if subject_suffix == "R":
    left_missing[:] = True
```

注意：

```text
missing不是MAE mask。
blink不是MAE mask。
padding不是MAE mask。
attention mask也不是MAE mask。
```

四者区别：

```text
blink:
  生理/行为状态，是content的一部分，可以预测。

missing:
  数据采集状态，是quality condition。
  missing位置不作为重建目标。

MAE mask:
  训练时人工遮住有效eye content，让模型预测。

padding:
  batch对齐补出来的假patch，不是trial真实内容。

attention mask:
  告诉Transformer哪些token不应该参与attention。
  padding token和all-missing eye token应该被attention mask掉。
  MAE-masked eye token不应该被attention mask掉。
```

---

# 9. 预处理

实现位置：

```text
src/eyemae/preprocess.py
```

## 9.1 坐标归一化

坐标已经是视觉角度，不做trial z-score。

使用固定尺度：

```python
x_norm = clip(x_deg, -30, 30) / 30
y_norm = clip(y_deg, -20, 20) / 20
stim_x_norm = clip(stim_x_deg, -30, 30) / 30
stim_y_norm = clip(stim_y_deg, -20, 20) / 20
```

配置：

```yaml
normalization:
  x_clip_deg: 30
  y_clip_deg: 20
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

```python
blink
missing
```

不要用 `x=0,y=0,area=0` 自动推断 missing。第一版missing只相信：

```text
label=2
subject suffix L/R
```

模型输入 content 和重建 target 都使用 preprocess 后的 content。xy/area loss 会被 valid mask gating，因此 missing/blink 处的0不会用于 xy/area监督。blink target 保留为0/1。

---

## 9.2 area / pupil面积归一化

area使用 robust z-score。

valid frames定义：

```python
valid = (missing == 0) & (blink == 0) & (area > 0)
```

统计：

```python
u = log1p(area)

median = median(u over valid frames)
mad = median(abs(u - median)) + eps

area_norm = (u - median) / (1.4826 * mad)
area_norm = clip(area_norm, -5, 5)
```

第一版只用 `pretrain_train_split` 统计 area median/MAD，不用 `pretrain_val_split` 或 `pretrain_test_split`。

实现：

```text
src/eyemae/compute_area_stats.py
```

命令：

```bash
python -m eyemae.compute_area_stats \
  --config configs/eyemae_cnn_512_12l.yaml \
  --split pretrain_train \
  --out outputs/area_stats.json
```

训练时只读取 `area.stats_path`，不要在 DataLoader worker 中重复统计。

配置：

```yaml
area:
  use_log1p: true
  robust_zscore_by: subject
  clip: 5
  eps: 1.0e-6
  stats_source: pretrain_train_only
  stats_path: outputs/area_stats.json
  fallback_to_global: true
```

如果某个subject在 stats 中不存在：

```text
fallback到global median/MAD。
```

如果 subject MAD < eps：

```text
fallback到global MAD。
```

保存统计文件：

```json
{
  "global": {"median": 0.0, "mad": 1.0, "num_valid_frames": 0},
  "subjects": {
    "subj001D": {"median": 0.0, "mad": 1.0, "num_valid_frames": 0}
  }
}
```

---

## 9.3 stimulus condition

第一版stim feature固定为4维：

```text
fix_on
stim_on
stim_x_norm
stim_y_norm
```

不使用：

```text
last_stim_x_norm
last_stim_y_norm
goal_x_norm
goal_y_norm
```

当stim不出现：

```python
stim_on = 0
stim_x_norm = 0
stim_y_norm = 0
```

即使原始npz里 `stim_on=0` 时 `stim_x/stim_y` 不是0，preprocess 也要强制置0。

配置：

```yaml
stim:
  use_fix_on: true
  use_stim_on: true
  use_stim_xy: true
  use_last_stim_xy: false
  use_goal_xy: false
  stim_dim: 4
```

说明：

```text
记忆眼跳任务中，第一版不显式提供last_stim。
因为模型是双向Transformer encoder，后面的eye token可以通过attention读到前面patch的S_i stim token。
后续通过add_last_stim ablation验证显式记忆位置是否有帮助。
```

---

# 10. patching

实现位置：

```text
src/eyemae/patching.py
```

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

即：

```text
20ms non-overlap patch
```

每个trial：

```python
N = T // 20
丢弃最后不足20帧的尾巴
```

如果：

```text
N == 0
```

则该 trial 不能用于训练，Dataset 应跳过或 raise warning。

构造：

```python
left_content_patch:  [N, 20, 4]
right_content_patch: [N, 20, 4]

# content dim:
# 0 x_norm
# 1 y_norm
# 2 area_norm
# 3 blink

left_quality_patch:  [N, 20, 1]
right_quality_patch: [N, 20, 1]

# quality dim:
# 0 missing

stim_patch: [N, 20, 4]

# stim dim:
# 0 fix_on
# 1 stim_on
# 2 stim_x_norm
# 3 stim_y_norm
```

batch collate时padding到batch内最大N。

返回batch：

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
    "trial_id": list[str],
}
```

眼别维度：

```text
eye=0 left
eye=1 right
```

其中：

```python
eye_nonmissing_frac[b, n, e] = 1.0 - mean(quality[b, n, e, :, 0])
```

第一版建议：

```yaml
attention:
  min_nonmissing_frac_for_eye_token: 0.05
```

即：

```python
eye_token_valid = (~pad_mask[:, :, None]) & (
    eye_nonmissing_frac >= min_nonmissing_frac_for_eye_token
)
```

含义：

```text
只要一个eye patch里至少有少量非missing帧，就允许它作为eye token参与attention。
纯missing eye token不参与attention，不作为MAE候选，不算loss。
```

---

# 11. batching

实现位置：

```text
src/eyemae/batching.py
```

第一版需要支持两种batch方式：

```text
fixed trial batch
token-based dynamic batch
```

主训练建议使用 token-based dynamic batch。

配置：

```yaml
train:
  batch_trials_per_gpu: null
  max_seq_tokens_per_gpu: 60000
  max_trials_per_gpu: 256
  bucket_by_length: true
```

定义：

```python
seq_len_i = 3 * N_i
Nmax = max(N_i for trial_i in batch)
3 * Nmax * len(batch) <= max_seq_tokens_per_gpu
len(batch) <= max_trials_per_gpu
```

也就是说 token budget 按 batch padding 后真正送入 Transformer 的 S/L/R token 数控制。
`bucket_by_length: true` 时先在随机窗口内按 trial 长度排序，再组 batch，以减少 padding 浪费。

debug 可以使用固定小batch：

```yaml
train:
  batch_trials_per_gpu: 16
  max_seq_tokens_per_gpu: null
  max_trials_per_gpu: null
```

如果实现动态batch复杂，第一版至少必须保证：

```text
固定batch不会因为长trial OOM。
遇到OOM时可以通过配置降低 batch_trials_per_gpu。
```

---

# 12. padding 与 attention validity

trial长度不同，例如：

```text
trial A: 1s -> 50 patches
trial B: 4s -> 200 patches
```

同一个batch需要补齐到：

```text
Nmax = max(N_i in batch)
```

定义：

```python
pad_mask[b, n] = True   # padding，假patch
pad_mask[b, n] = False  # 真实patch
```

sequence是：

```text
S_0, L_0, R_0, S_1, L_1, R_1, ..., S_{N-1}, L_{N-1}, R_{N-1}
```

因此attention padding mask需要扩展成：

```python
seq_attn_pad_mask: BoolTensor[B, 3 * Nmax]
```

索引规则：

```python
S_i index = 3 * i
L_i index = 3 * i + 1
R_i index = 3 * i + 2
```

valid定义：

```python
S_i_valid = ~pad_mask[b, i]

L_i_valid = (~pad_mask[b, i]) & eye_token_valid[b, i, 0]
R_i_valid = (~pad_mask[b, i]) & eye_token_valid[b, i, 1]
```

attention mask定义：

```python
seq_attn_pad_mask[b, 3*i + 0] = not S_i_valid
seq_attn_pad_mask[b, 3*i + 1] = not L_i_valid
seq_attn_pad_mask[b, 3*i + 2] = not R_i_valid
```

注意：

```text
MAE-masked eye token不应该被attention mask掉。
MAE-masked eye token仍然参与attention，
只是content_token换成learned mask_token。
```

只有以下token应该被attention mask掉：

```text
batch padding产生的S/L/R token
all-missing或低于nonmissing阈值的L/R eye token
```

Transformer输出后，必须把 invalid token hidden 清零，避免后续误用：

```python
hidden = hidden.masked_fill(seq_attn_pad_mask[..., None], 0.0)
```

---

# 13. MAE mask策略

实现位置：

```text
src/eyemae/masking.py
```

输出：

```python
mae_mask: BoolTensor[B, Nmax, 2]
```

含义：

```text
True = 这个trial、这个patch、这只眼的content被人工mask
False = content可见
```

注意：

```text
mae_mask只对应 L_i / R_i eye tokens。
mae_mask不包含 S_i stim token。
```

第一版mask配置：

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

暂时不实现：

```text
random_ratio
both_eye_mask_prob
artificial_monocular_dropout_prob
```

避免第一版逻辑过复杂。

## 13.1 span mask定义

span mask就是连续遮住一段eye patches。

例如：

```text
N = 100
start = 30
span_len = 6

mask left eye patches:
  L_30, L_31, L_32, L_33, L_34, L_35

或mask right eye patches:
  R_30, R_31, R_32, R_33, R_34, R_35

或左右眼同时mask：
  L/R_30 ... L/R_35
```

由于每个patch是20ms：

```text
short span: 2-8 patches = 40-160ms
long span: 10-25 patches = 200-500ms
```

目的：

```text
防止模型只学线性插值。
迫使模型利用任务、刺激、前后轨迹和左右眼关系。
```

## 13.2 MAE候选集合

对每个eye-patch token计算：

```python
eye_nonmissing_frac = 1.0 - quality[..., 0].float().mean(dim=-1)
# shape: [B, N, 2]
```

MAE候选：

```python
mae_eligible = (
    (~pad_mask[:, :, None])
    & eye_token_valid
    & (eye_nonmissing_frac >= min_nonmissing_frac_for_mae)
)
```

第一版推荐：

```yaml
mask:
  min_nonmissing_frac_for_mae: 0.50
```

含义：

```text
20帧里至少10帧不是missing，才允许作为MAE重建目标。
```

不要在第一版加入额外的 `min_coord_valid_frac_for_mae` 或 blink-only 细化规则。

## 13.3 mask实现逻辑

对每个trial `b`：

```python
eligible_positions = where(mae_eligible[b] == True)
num_eligible_eye_tokens = len(eligible_positions)
```

如果：

```text
num_eligible_eye_tokens < 2
```

则：

```text
该trial的mae_mask全False。
该trial在loss中不会贡献重建loss。
不要死循环，不要报错。
```

否则：

```python
target_ratio = uniform(ratio_min, ratio_max)
target_num_masked = int(target_ratio * num_eligible_eye_tokens)
target_num_masked = max(1, target_num_masked)
target_num_masked = min(target_num_masked, num_eligible_eye_tokens - 1)

mae_mask[b] = False
```

选择眼别模式：

```python
with probability single_eye_mask_prob:
    selected_eyes = [left] or [right]
otherwise:
    selected_eyes = [left, right]
```

如果 selected_eyes 下 eligible token 少于2个，则自动 fallback 到左右眼 eligible 全集合。

先做short span：

```python
if random() < short_span_prob:
    sample start
    sample span_len between 2 and 8
    mask selected eye/eyes over start:start+span_len
    但只允许mask mae_eligible=True 的eye token
```

再做long span：

```python
if random() < long_span_prob:
    sample start
    sample span_len between 10 and 25
    mask selected eye/eyes over start:start+span_len
    但只允许mask mae_eligible=True 的eye token
```

然后用随机eligible eye-patch token补足到目标mask数量，不要使用可能死循环的 `while`：

```python
remaining = eligible_positions that are not already masked
num_to_sample = min(target_num_masked - current_masked, len(remaining))
sample num_to_sample positions without replacement
```

强制规则：

```python
mae_mask[b, pad_mask[b], :] = False
mae_mask[b] &= mae_eligible[b]
```

保证至少有一个可见的eligible eye token：

```python
visible_eligible = mae_eligible[b] & (~mae_mask[b])
```

如果没有 visible eligible，则随机 unmask 一个 masked eligible token。

## 13.4 mask_type

为了validation分组统计，可以返回：

```python
mask_type: IntTensor[B, Nmax, 2]
```

定义：

```text
0 = not masked
1 = random
2 = short_span
3 = long_span
```

如果实现 `mask_type`，同一个位置被多种mask覆盖时优先级：

```text
long_span > short_span > random
```

`mask_type` 第一版不是必须字段；如果没有实现，metrics中跳过按mask_type分组。

---

# 14. 模型结构

实现位置：

```text
src/eyemae/model.py
```

## 14.1 序列格式

第一版不加CLS。

序列格式：

```text
S_0, L_0, R_0, S_1, L_1, R_1, ..., S_{N-1}, L_{N-1}, R_{N-1}
```

长度：

```text
seq_len = 3 * N
```

每个token对应：

```text
S_i:
  第 i 个patch的stimulus / task / time token。
  只作为条件输入。
  不被MAE mask。
  不做prediction head。
  不算reconstruction loss。

L_i:
  左眼第 i 个patch的eye token。
  可以被MAE mask。
  可以算reconstruction loss。

R_i:
  右眼第 i 个patch的eye token。
  可以被MAE mask。
  可以算reconstruction loss。
```

第一版不要做：

```text
CLS token
feature-wise tokens: Lx_i, Ly_i, Larea_i, Lblink_i ...
bilateral token: E_i = left+right
```

---

## 14.2 token组成

### S_i token

```python
S_i = stim_token_i
    + task_embedding
    + token_type_embedding["stim"]
    + time_embedding_i

S_i = fusion_layernorm(S_i)
```

### L_i / R_i eye token

```python
eye_token = content_token_or_mask_token
          + quality_token
          + task_embedding
          + token_type_embedding["left" or "right"]
          + time_embedding_i

eye_token = fusion_layernorm(eye_token)
```

解释：

```text
content_token:
  来自 x/y/area/blink。
  被MAE mask时替换成learned mask token。

quality_token:
  来自missing。
  不被MAE mask。

stim_token:
  单独形成S_i，不broadcast加到L_i/R_i。
  eye token通过attention读取S_i里的刺激信息。

task_embedding:
  4类任务embedding。
  S_i、L_i、R_i都加入task embedding。

token_type_embedding:
  3类：stim / left / right。
  用来告诉模型这个token是什么类型。

time_embedding:
  patch index embedding。
  S_i、L_i、R_i共享同一个patch index i。
```

第一版fusion使用：

```text
additive fusion + LayerNorm
```

不做：

```text
cross-attention
adaptive layer norm
concat + MLP fusion
```

---

## 14.3 tokenizer

第一版用 CNN tokenizer。

输入shape：

```python
content: [B, N, 2, 20, 4]
quality: [B, N, 2, 20, 1]
stim:    [B, N, 20, 4]
```

### Content tokenizer

左右眼共享同一个content tokenizer。

```text
Conv1d(4 -> 64, kernel=5, padding=2)
GELU
Conv1d(64 -> 128, kernel=3, padding=1)
GELU
flatten
Linear(128*20 -> d_model)
```

输入输出：

```python
[B, N, 2, 20, 4] -> [B, N, 2, d_model]
```

### Quality tokenizer

左右眼共享同一个quality tokenizer。

```text
Conv1d(1 -> 32, kernel=3, padding=1)
GELU
flatten
Linear(32*20 -> d_model)
```

输入输出：

```python
[B, N, 2, 20, 1] -> [B, N, 2, d_model]
```

### Stim tokenizer

```text
Conv1d(4 -> 64, kernel=3, padding=1)
GELU
flatten
Linear(64*20 -> d_model)
```

输入输出：

```python
[B, N, 20, 4] -> [B, N, d_model]
```

注意：

```text
stim_token不broadcast到左右眼。
stim_token用于构造独立的S_i token。
```

### Task embedding

```python
task_embedding = nn.Embedding(4, d_model)
```

shape：

```python
[B] -> [B, 1, 1, d_model]
```

对S/L/R都broadcast。

### Token type embedding

```python
token_type_embedding = nn.Embedding(3, d_model)
```

定义：

```text
0 = stim
1 = left
2 = right
```

### Time embedding

```python
time_embedding = nn.Embedding(max_patches, d_model)
```

shape：

```python
[N] -> [1, N, d_model]
```

注意：

```text
S_i、L_i、R_i都使用同一个time_embedding[i]。
不要把sequence index 3*i, 3*i+1, 3*i+2当作真实时间。
```

---

## 14.4 mask token

当：

```python
mae_mask[b, n, e] == True
```

时：

```python
content_token[b, n, e] = learned_mask_token
```

但仍保留：

```python
quality_token
task_embedding
token_type_embedding
time_embedding
```

注意：

```text
MAE-masked eye token仍然参与attention。
不要把MAE-masked eye token放进seq_attn_pad_mask。
```

---

## 14.5 sequence assembly

构造：

```python
S_tokens: [B, N, d_model]
L_tokens: [B, N, d_model]
R_tokens: [B, N, d_model]
```

组装成：

```python
seq = torch.empty(B, 3*N, d_model)

seq[:, 0::3, :] = S_tokens
seq[:, 1::3, :] = L_tokens
seq[:, 2::3, :] = R_tokens
```

attention mask：

```python
seq_attn_pad_mask: [B, 3*N]

seq_attn_pad_mask[:, 0::3] = pad_mask
seq_attn_pad_mask[:, 1::3] = pad_mask | (~eye_token_valid[:, :, 0])
seq_attn_pad_mask[:, 2::3] = pad_mask | (~eye_token_valid[:, :, 1])
```

---

## 14.6 Transformer encoder

第一版配置：

```yaml
model:
  tokenizer: cnn
  sequence_format: stim_eye_triplet_no_cls

  d_model: 512
  n_layers: 12
  n_heads: 8
  ffn_hidden: 1536
  dropout: 0.1

  norm: rmsnorm
  activation: swiglu

  max_patches: 384
  pretrain_style: bert_masked_reconstruction
  use_cls: false
  use_token_type_embedding: true
  fusion: add_then_layernorm

  use_stim_tokens: true
  broadcast_stim_to_eye: false

  use_rope: false
  use_gqa: false
  use_cross_attention: false
  use_adaln: false
```

实现：

```text
pre-norm Transformer encoder
RMSNorm
MultiheadAttention
RMSNorm
SwiGLU FFN
residual connection
dropout
```

不用causal mask。

这是：

```text
bidirectional Transformer encoder
```

Transformer block 使用 PyTorch `nn.MultiheadAttention(batch_first=True)` 时，传入：

```python
key_padding_mask=seq_attn_pad_mask
```

---

## 14.7 prediction head

只对eye tokens预测。

```text
不对S_i做prediction head。
不对S_i算reconstruction loss。
```

从hidden sequence取回eye positions：

```python
hidden_L = hidden_seq[:, 1::3, :]  # [B, N, d_model]
hidden_R = hidden_seq[:, 2::3, :]  # [B, N, d_model]

hidden_eye = torch.stack([hidden_L, hidden_R], dim=2)
# [B, N, 2, d_model]
```

prediction head：

```text
Linear(d_model -> d_model)
GELU
Linear(d_model -> 20*4)
```

输出：

```python
pred: [B, N, 2, 20, 4]
```

4维为：

```text
0 x_norm
1 y_norm
2 area_norm
3 blink_logit
```

---

# 15. loss设计

实现位置：

```text
src/eyemae/losses.py
```

输入：

```python
pred:       FloatTensor[B, N, 2, 20, 4]
target:     FloatTensor[B, N, 2, 20, 4]
quality:    FloatTensor[B, N, 2, 20, 1]
mae_mask:   BoolTensor[B, N, 2]
pad_mask:   BoolTensor[B, N]
eye_token_valid: BoolTensor[B, N, 2]
```

target content维度：

```text
0 x_norm
1 y_norm
2 area_norm
3 blink
```

quality维度：

```text
0 missing
```

注意：

```text
loss只对L/R eye tokens算。
S_i stim token没有target，没有prediction head，不算loss。
```

---

## 15.1 mask展开

```python
missing = quality[..., 0].bool()        # [B,N,2,20]
target_blink = target[..., 3] > 0.5     # [B,N,2,20]

nonpad = (~pad_mask)[:, :, None, None]  # [B,N,1,1]
eye_valid = eye_token_valid[:, :, :, None]  # [B,N,2,1]

mae_frame = mae_mask[:, :, :, None]     # [B,N,2,1]
mae_frame = mae_frame.expand_as(missing)
```

---

## 15.2 masked mean

所有loss都必须使用 numerator / denominator 形式，避免NaN：

```python
def masked_mean(loss_tensor, mask, eps=1e-8):
    mask = mask.to(loss_tensor.dtype)
    numerator = (loss_tensor * mask).sum()
    denominator = mask.sum()
    value = numerator / denominator.clamp_min(eps)
    return value, numerator.detach(), denominator.detach()
```

如果某项denominator为0，该项loss返回0，并记录 denominator=0。

总loss中每项仍然乘对应weight。

---

## 15.3 坐标loss

坐标只在以下位置计算：

```python
coord_valid = (
    mae_frame
    & nonpad
    & eye_valid
    & (~missing)
    & (~target_blink)
)
```

对：

```text
x, y
```

使用SmoothL1 / Huber loss。

```yaml
loss:
  xy_weight: 1.0
```

---

## 15.4 area loss

area使用同样valid mask：

```python
area_valid = coord_valid
```

对：

```text
area
```

使用SmoothL1 / Huber loss。

```yaml
loss:
  area_weight: 0.2
```

---

## 15.5 blink loss

blink在非missing且被MAE mask的位置计算：

```python
blink_valid = (
    mae_frame
    & nonpad
    & eye_valid
    & (~missing)
)
```

对：

```text
blink_logit
```

使用：

```python
BCEWithLogitsLoss
```

```yaml
loss:
  blink_weight: 0.1
```

---

## 15.6 velocity loss

第一版先只在**patch内部连续帧**计算速度loss，不跨patch边界。

对x/y计算一阶差分：

```python
v_pred = pred[..., 1:, 0:2] - pred[..., :-1, 0:2]
v_true = target[..., 1:, 0:2] - target[..., :-1, 0:2]
```

valid条件：

```python
v_valid = coord_valid[..., 1:] & coord_valid[..., :-1]
```

不要跨：

```text
blink边界
missing边界
padding
未mask区域
无效eye token
```

计算：

```python
velocity_loss = SmoothL1(v_pred, v_true, mask=v_valid)
```

配置：

```yaml
loss:
  velocity_weight: 0.1
```

---

## 15.7 总loss

```python
loss = 1.0 * xy_loss \
     + 0.2 * area_loss \
     + 0.1 * blink_loss \
     + 0.1 * velocity_loss
```

所有重建loss都只在：

```text
MAE人工masked的有效eye token区域
```

上计算。

missing处不计算任何重建loss。

blink处：

```text
不计算x/y/area loss
计算blink loss
```

stim token处：

```text
不计算任何重建loss
```

如果整个batch的总有效denominator为0：

```text
跳过optimizer step。
记录warning。
继续下一个batch。
```

---

# 16. metrics

实现位置：

```text
src/eyemae/metrics.py
```

每次validation记录：

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

坐标从归一化单位还原到视觉角度：

```python
x_deg = x_norm * 30
y_deg = y_norm * 20
```

分组统计：

```text
task_id
left/right eye
trial length bucket
missing fraction bucket
single-eye subject vs both-eye subject
eye_token_valid fraction bucket
```

如果实现了 `mask_type`，再增加：

```text
mask_type: random / short_span / long_span
```

注意：

```text
metrics只统计有效eye token和有效帧。
不统计S_i stim token。
不统计padding。
不统计missing。
xy/area不统计blink。
```

AUC边界：

```text
如果validation子集里blink正负样本不同时存在，blink_auc返回NaN，不报错。
```

DDP validation：

```text
所有numerator和denominator必须跨rank all_reduce(sum)后再计算最终指标。
不要简单平均每个rank的loss。
```

---

# 17. baselines

实现位置：

```text
src/eyemae/baselines.py
```

用于判断模型是否只是插值器。

## 17.1 previous value baseline

对被mask的x/y：

```text
用mask前最近一个可见点填充
```

计算同样的：

```text
masked_xy_rmse_deg
```

## 17.2 linear interpolation baseline

对被mask的x/y：

```text
用mask前后最近可见点线性插值
```

边界规则：

```text
mask段前后都有可见点：
  线性插值

mask段在开头，没有previous：
  使用next valid填充

mask段在结尾，没有next：
  使用previous valid填充

两边都没有valid：
  跳过该段，不计入baseline指标
```

baseline 必须遵守和模型相同的valid mask：

```text
只在MAE-masked有效eye token上评估。
missing和blink处不参与xy baseline指标。
stim token不参与baseline。
padding不参与baseline。
```

模型至少应优于：

```text
previous value baseline
```

并且在long span上接近或优于：

```text
linear interpolation baseline
```

---

# 18. visualize

实现位置：

```text
src/eyemae/visualize.py
```

每次validation保存若干图：

```text
true vs pred x over time
true vs pred y over time
blink true vs pred probability
MAE mask spans
missing spans
invalid eye token spans
task_id
fix_on
stim_on
stim_x
stim_y
```

第一版不画：

```text
last_stim_x/y
goal_x/y
```

因为第一版没有这些输入。

工程要求：

```text
只在rank0保存图。
使用matplotlib Agg backend。
每次validation最多保存16个trial。
保存路径：
  outputs/{experiment.name}/visualizations/step_{global_step}/...
```

每个checkpoint至少保存：

```text
10个随机trial
每个task至少2个trial
单眼缺失trial若存在，也保存
long span mask trial至少2个
包含all-missing eye token但stim有效的trial至少2个
```

---

# 19. 模型训练

实现位置：

```text
src/eyemae/train.py
```

支持：

```bash
python -m eyemae.train --config configs/debug.yaml
```

以及：

```bash
torchrun --standalone --nproc_per_node=3 \
  -m eyemae.train \
  --config configs/eyemae_cnn_512_12l.yaml
```

## 19.1 DDP要求

DDP实现要求：

```text
使用 torchrun 环境变量 LOCAL_RANK / RANK / WORLD_SIZE。
每个进程 torch.cuda.set_device(local_rank)。
每个进程创建一个模型副本。
train DataLoader 使用 DistributedSampler 或兼容的rank-aware token batch sampler。
如果使用 DistributedSampler，每个epoch调用 sampler.set_epoch(epoch)。
只有 rank0 写 TensorBoard、保存checkpoint、保存可视化。
validation metrics 跨rank all_reduce。
训练期 validation 同时保存整体 metrics 和 group metrics。
```

precision：

```text
bf16:
  使用 torch.autocast(device_type="cuda", dtype=torch.bfloat16)
  不需要 GradScaler。

fp16:
  使用 GradScaler。

fp32:
  不使用autocast。
```

## 19.2 checkpoint

checkpoint保存：

```text
model state_dict
optimizer state_dict
scheduler state_dict
global_step
epoch
config
area_stats_path
best_metric
rng states
```

只由rank0保存。

命名：

```text
checkpoint_last.pt
checkpoint_best.pt
checkpoint_step_00010000.pt
```

best checkpoint 监控：

```yaml
checkpoint:
  monitor: val/masked_xy_rmse_deg
  mode: min
```

如果 `masked_xy_rmse_deg` 不可用，则fallback到：

```yaml
checkpoint:
  monitor: val/total_loss
  mode: min
```

## 19.3 CLI

训练：

```bash
python -m eyemae.train \
  --config configs/debug.yaml
```

resume：

```bash
python -m eyemae.train \
  --config configs/debug.yaml \
  --resume outputs/debug/checkpoint_last.pt
```

DDP：

```bash
torchrun --standalone --nproc_per_node=3 \
  -m eyemae.train \
  --config configs/eyemae_cnn_512_12l.yaml
```

---

# 20. evaluate.py

实现位置：

```text
src/eyemae/evaluate.py
```

CLI：

```bash
python -m eyemae.evaluate \
  --config configs/eyemae_cnn_512_12l.yaml \
  --checkpoint outputs/exp/checkpoint_best.pt \
  --split pretrain_val
```

支持：

```text
pretrain_train
pretrain_val
pretrain_test
```

最终测试：

```bash
python -m eyemae.evaluate \
  --config configs/eyemae_cnn_512_12l.yaml \
  --checkpoint outputs/exp/checkpoint_best.pt \
  --split pretrain_test
```

evaluate 必须：

```text
不更新模型参数
使用相同area_stats
使用相同mask策略或固定eval mask seed
输出metrics.json
输出metrics_by_group.json
可选保存visualization
```

训练期 validation 也使用同一套分组口径：

```text
task_id
eye
trial_length
missing_fraction
subject_eye_availability
eye_token_valid_fraction
mask_type
```

训练期 visualization 从整个 validation pass 中选择代表性样本，而不是只保存第一个batch：

```text
各task样本
long_span mask样本
单眼可用subject样本
all-missing eye token但stim_on存在的样本
随机reservoir样本
```

eval mask 建议：

```yaml
eval:
  seed: 1234
  fixed_mask: true
```

---

# 21. 第一版主配置

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
  format: npz_per_trial
  data_dir: /path/to/npz

  pretrain_train_split: splits/pretrain_subject_heldout_seed42/pretrain_train.txt
  pretrain_val_split: splits/pretrain_subject_heldout_seed42/pretrain_val.txt
  pretrain_test_split: splits/pretrain_subject_heldout_seed42/pretrain_test.txt

  sampling_rate: 1000

  npz_keys:
    eye: eye
    task_id: task_id
    fix_on: fix_on
    stim: stim
    subject_id: subject_id
    trial_id: trial_id

split:
  strategy: subject_heldout
  seed: 42
  train_ratio: 0.90
  val_ratio: 0.05
  test_ratio: 0.05
  group_by_base_subject_id: true
  out_dir: splits/pretrain_subject_heldout_seed42

label:
  nonblink_value: 0
  blink_value: 1
  missing_value: 2

subject_suffix:
  D: both_eyes
  L: left_eye_only
  R: right_eye_only

normalization:
  x_clip_deg: 30
  y_clip_deg: 20

area:
  use_log1p: true
  robust_zscore_by: subject
  clip: 5
  eps: 1.0e-6
  stats_source: pretrain_train_only
  stats_path: outputs/area_stats_subject_heldout_seed42.json
  fallback_to_global: true

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
  max_seq_tokens_per_gpu: 60000
  max_trials_per_gpu: 256
  bucket_by_length: true
  grad_accum_steps: 1

  num_workers: 8
  pin_memory: true
  persistent_workers: true
  gradient_checkpointing: false

checkpoint:
  monitor: val/masked_xy_rmse_deg
  mode: min

eval:
  seed: 1234
  fixed_mask: true
```

---

# 22. debug配置

创建：

```text
configs/debug.yaml
```

内容：

```yaml
experiment:
  name: debug_eyemae_stimtoken_tiny
  output_dir: outputs/debug_eyemae_stimtoken_tiny

data:
  format: npz_per_trial
  data_dir: tests/fixtures/synthetic_npz

  pretrain_train_split: splits/debug_trial_random_seed42/pretrain_train.txt
  pretrain_val_split: splits/debug_trial_random_seed42/pretrain_val.txt
  pretrain_test_split: splits/debug_trial_random_seed42/pretrain_test.txt

  sampling_rate: 1000

  npz_keys:
    eye: eye
    task_id: task_id
    fix_on: fix_on
    stim: stim
    subject_id: subject_id
    trial_id: trial_id

split:
  strategy: trial_random
  seed: 42
  train_ratio: 0.98
  val_ratio: 0.02
  test_ratio: 0.00
  group_by_base_subject_id: false
  out_dir: splits/debug_trial_random_seed42

label:
  nonblink_value: 0
  blink_value: 1
  missing_value: 2

subject_suffix:
  D: both_eyes
  L: left_eye_only
  R: right_eye_only

normalization:
  x_clip_deg: 30
  y_clip_deg: 20

area:
  use_log1p: true
  robust_zscore_by: subject
  clip: 5
  eps: 1.0e-6
  stats_source: pretrain_train_only
  stats_path: outputs/debug_area_stats.json
  fallback_to_global: true

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

  d_model: 256
  n_layers: 4
  n_heads: 4
  ffn_hidden: 768
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
  distributed: none
  optimizer: adamw
  lr: 3.0e-4
  min_lr: 3.0e-5
  betas: [0.9, 0.95]
  weight_decay: 0.05
  grad_clip: 1.0

  warmup_steps: 100
  max_steps: 2000
  val_every_steps: 100
  save_every_steps: 500
  log_every_steps: 10

  batch_trials_per_gpu: 16
  max_seq_tokens_per_gpu: null
  max_trials_per_gpu: null
  bucket_by_length: false
  grad_accum_steps: 1

  num_workers: 2
  pin_memory: true
  persistent_workers: false
  gradient_checkpointing: false

checkpoint:
  monitor: val/masked_xy_rmse_deg
  mode: min

eval:
  seed: 1234
  fixed_mask: true

debug:
  overfit_trials: 64
```

---

# 23. tests必须覆盖

## 23.1 config test

```bash
pytest tests/test_config.py
```

必须验证：

```text
debug.yaml 是完整配置
eyemae_cnn_512_12l.yaml 是完整配置
validate_config能检查required fields
stim_dim == input.stim_dim
content_dim == 4
quality_dim == 1
sequence_format == stim_eye_triplet_no_cls
pretrain_style == bert_masked_reconstruction
```

## 23.2 splits test

```bash
pytest tests/test_splits.py
```

必须验证：

```text
trial_random 能生成 pretrain_train/val/test txt
subject_heldout 中同一个 base subject 不跨 split
D/L/R suffix 去掉后分组
split_summary.json 生成
task_counts 被统计
eye_availability_counts 被统计
```

## 23.3 area stats test

```bash
pytest tests/test_area_stats.py
```

必须验证：

```text
只用 pretrain_train split 统计
val/test 不参与统计
subject stats 和 global stats 都存在
MAD=0 时不产生NaN
subject缺失时 fallback到global
```

## 23.4 data schema test

```bash
pytest tests/test_data_schema.py
```

必须验证：

```text
eye shape错误时报错
fix_on长度错误时报错
stim shape错误时报错
task_id非法时报错
label包含非0/1/2时报错
split txt 空行和#注释被忽略
```

## 23.5 shape test

```bash
pytest tests/test_model_shapes.py
```

必须检查：

```text
content              [B,N,2,20,4]
quality              [B,N,2,20,1]
stim                 [B,N,20,4]
mae_mask             [B,N,2]
pad_mask             [B,N]
eye_nonmissing_frac  [B,N,2]
eye_token_valid      [B,N,2]
seq_attn_pad_mask    [B,3*N]
pred                 [B,N,2,20,4]
```

## 23.6 sequence mapping test

```bash
pytest tests/test_sequence_mapping.py
```

必须验证：

```text
S_i index = 3*i
L_i index = 3*i + 1
R_i index = 3*i + 2

hidden_seq[:, 1::3] 正确映射回 left eye
hidden_seq[:, 2::3] 正确映射回 right eye

prediction head只处理L/R eye token。
S token不进入prediction head。
```

## 23.7 attention mask test

```bash
pytest tests/test_attention_mask.py
```

必须验证：

```text
padding patch:
  S/L/R 全部attention-masked

非padding patch:
  S_i 总是attention-valid

all-missing left eye patch:
  L_i attention-masked
  L_i 不作为MAE候选

all-missing right eye patch:
  R_i attention-masked
  R_i 不作为MAE候选

MAE-masked但有效的eye token:
  attention-valid
  content_token替换为mask_token

encoder输出后invalid hidden被置0
```

## 23.8 missing / blink preprocess test

```bash
pytest tests/test_preprocess_missing.py
```

必须覆盖：

```text
1. label=2 -> missing=1, blink=0, x/y/area置0
2. subject suffix L -> right eye missing=1 for all frames
3. subject suffix R -> left eye missing=1 for all frames
4. subject suffix D -> 不额外加subject-level missing
5. label=1 -> blink=1, missing=0, x/y/area置0
6. label=0 且 x=0,y=0,area>0 -> 正常中心注视，不是missing
7. label=0 且 x=0,y=0,area=0 -> 不自动当missing，除非label=2或subject suffix说明该眼缺失
```

## 23.9 patchify test

```bash
pytest tests/test_patchify.py
```

覆盖：

```text
T=1000 -> N=50 when patch=20
T=1025 -> N=51，丢弃最后5帧
T<20 -> trial无有效patch
left/right content shape正确
quality shape正确
stim shape正确
eye_nonmissing_frac正确
eye_token_valid正确
```

## 23.10 masking test

```bash
pytest tests/test_masking.py
```

覆盖：

```text
mae_mask shape = [B,N,2]
mae_mask不包含stim维度
padding位置不被MAE mask
all-missing eye token不被MAE mask
低于min_nonmissing_frac_for_mae的eye token不被MAE mask
至少一个eligible eye token可见
mask ratio基于eligible eye tokens计算
single_eye_mask_prob能产生只mask单眼的trial
span mask能产生连续mask段
eligible过少时不死循环
```

## 23.11 loss gating test

```bash
pytest tests/test_loss_gating.py
```

必须验证：

```text
stim token不产生任何重建loss

missing处不产生xy loss
missing处不产生area loss
missing处不产生blink loss

blink处不产生xy loss
blink处不产生area loss
blink处产生blink loss

未被MAE mask的位置不产生重建loss
padding位置不产生任何loss
invalid eye token不产生任何loss

velocity loss不跨blink/missing边界
velocity loss不跨未mask区域
velocity loss不跨invalid eye token

denominator=0时loss为0且不NaN
```

## 23.12 tiny overfit test

```bash
pytest tests/test_overfit_tiny_batch.py
```

或脚本：

```bash
python -m eyemae.train \
  --config configs/debug.yaml \
  --overfit_trials 64
```

期待：

```text
total_loss明显下降
xy_loss明显下降
blink_loss下降
无NaN
```

---

# 24. 训练流程

## 24.1 生成 synthetic fixtures

```bash
python tests/fixtures/make_synthetic_npz.py \
  --out_dir tests/fixtures/synthetic_npz \
  --num_trials 128
```

synthetic数据必须覆盖：

```text
D/L/R subject suffix
四类 task_id
正常帧
blink帧
missing帧
短trial
长trial
stim_on/stim_off
```

## 24.2 生成debug split

```bash
python -m eyemae.make_splits \
  --config configs/debug.yaml
```

## 24.3 计算debug area stats

```bash
python -m eyemae.compute_area_stats \
  --config configs/debug.yaml \
  --split pretrain_train \
  --out outputs/debug_area_stats.json
```

## 24.4 单GPU debug

```bash
python -m eyemae.train \
  --config configs/debug.yaml
```

确认：

```text
shape正确
S/L/R sequence mapping正确
seq_attn_pad_mask正确
loss无NaN
loss能下降
可视化正常保存
checkpoint正常保存
```

## 24.5 tiny overfit

```bash
python -m eyemae.train \
  --config configs/debug.yaml \
  --overfit_trials 64
```

期待：

```text
模型能明显过拟合小数据
xy_loss下降
blink_loss下降
```

如果tiny overfit失败，不要上全量训练。

## 24.6 生成正式pretrain split

```bash
python -m eyemae.make_splits \
  --config configs/eyemae_cnn_512_12l.yaml
```

## 24.7 计算正式area stats

```bash
python -m eyemae.compute_area_stats \
  --config configs/eyemae_cnn_512_12l.yaml \
  --split pretrain_train \
  --out outputs/area_stats_subject_heldout_seed42.json
```

## 24.8 3GPU主训练

```bash
torchrun --standalone --nproc_per_node=3 \
  -m eyemae.train \
  --config configs/eyemae_cnn_512_12l.yaml
```

## 24.9 最终pretrain_test评估

```bash
python -m eyemae.evaluate \
  --config configs/eyemae_cnn_512_12l.yaml \
  --checkpoint outputs/eyemae_cnn_512_12l_patch20_stimtoken/checkpoint_best.pt \
  --split pretrain_test
```

---

# 25. 第一版验收标准

代码完成标准：

```text
pip install -e . 成功
pytest全部通过
make_splits.py 能生成 debug split 和 subject-heldout split
split_summary.json 正确
compute_area_stats.py 只用pretrain_train统计
1GPU debug能跑
tiny overfit能下降
3GPU DDP能跑
checkpoint能保存和resume
validation能输出主要metrics
visualization能保存true/pred轨迹图
previous value和linear interpolation baseline能比较
evaluate.py能在pretrain_test上运行
```

核心验收项：

```text
S/L/R sequence mapping正确
S token不被MAE mask
S token不算reconstruction loss
all-missing eye token不参与attention
all-missing eye token不作为MAE候选
MAE-masked eye token仍参与attention
mask ratio基于eligible eye tokens计算
同一个base subject不跨split
area stats不使用pretrain_val/pretrain_test
DDP validation metrics跨rank正确聚合
```

训练结果最低可接受标准：

```text
无NaN
blink_auc高于随机，若无正负样本则返回NaN且不报错
masked_xy_rmse_deg优于previous-value baseline
long_span_mask下接近或优于linear interpolation baseline
单眼缺失trial不报错
missing/blink/loss gating正确
stim token独立后memory task不明显崩坏
```

---

# 26. 后续 ablation 计划

所有ablation都基于第一版主配置，只改一个因素。

固定：

```text
same pretrain_train/pretrain_val/pretrain_test split
same random seed
same max_steps
same validation protocol
same metrics
```

重要实验后续再跑多seed：

```text
seed 42, 43, 44
```

第一轮先用：

```text
seed 42
```

## Ablation A：add_last_stim

目的：

```text
判断记忆眼跳中是否需要显式提供过去刺激位置。
```

配置：

```yaml
stim:
  use_last_stim_xy: true
  stim_dim: 6
```

S_i stim feature变成：

```text
fix_on
stim_on
stim_x
stim_y
last_stim_x
last_stim_y
```

比较重点：

```text
memory task
stim消失后的long span mask
late response phase
```

## Ablation B：with_goal

目的：

```text
判断提供任务规则后的理想目标是否提高重建。
```

配置：

```yaml
stim:
  use_goal_xy: true
```

goal构造：

```text
正向：goal = stim 或 last_stim
反向：goal = -stim 或 -last_stim
记忆：goal = remembered last_stim
二次眼跳：第一版先用当前stim/last_stim，复杂版本之后做phase-aware goal
```

## Ablation C：tokenizer

比较：

```text
CNN tokenizer
MLP tokenizer
```

MLP结构：

```text
flatten [20,4]
LayerNorm
Linear(80 -> d_model)
GELU
Linear(d_model -> d_model)
```

## Ablation D：velocity loss

比较：

```yaml
velocity_weight: 0.0
velocity_weight: 0.1
velocity_weight: 0.2
```

## Ablation E：mask策略

比较：

```text
random only
default mixed mask
span heavy
```

span heavy：

```yaml
mask:
  ratio_min: 0.45
  ratio_max: 0.75
  short_span_prob: 0.90
  long_span_prob: 0.30
```

## Ablation F：patch size

比较：

```yaml
patch.samples: 10
patch.samples: 20
patch.samples: 40
```

对应 stride 与 samples 相同。

## Ablation G：bilateral token

比较：

```text
S0, L0, R0, S1, L1, R1, ...
S0, E0, S1, E1, ...
```

## Ablation H：add CLS

序列：

```text
CLS, S0, L0, R0, S1, L1, R1, ...
```

MAE重建仍然只预测eye tokens。

## Ablation I：feature-wise tokens

序列从：

```text
S_i, L_i, R_i
```

变成：

```text
S_i,
Lx_i, Ly_i, Larea_i, Lblink_i,
Rx_i, Ry_i, Rarea_i, Rblink_i
```

注意：

```text
seq_len从3N变成9N
attention开销显著增加
```

## Ablation J：overlap patch

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
如果一个原始frame被mask，所有包含这个frame的overlap patch content都应视为masked。
```

## Ablation K：true asymmetric MAE

比较：

```yaml
model:
  pretrain_style: bert_masked_reconstruction
```

与：

```yaml
model:
  pretrain_style: asymmetric_mae
```

true MAE逻辑：

```text
encoder:
  只输入S_i和visible eye tokens。
  masked eye tokens不进入encoder。

decoder:
  接收encoder输出 + masked eye placeholders。
  重建masked eye patches。

all-missing eye tokens:
  不进入encoder。
  不进入decoder。
  不算loss。
```

## Ablation L：model scale

比较：

```yaml
debug:
  d_model: 256
  n_layers: 4
  n_heads: 4

base:
  d_model: 512
  n_layers: 12
  n_heads: 8

larger:
  d_model: 768
  n_layers: 16
  n_heads: 12
  ffn_hidden: 2048
```

## Ablation M：Qwen-style block

建议顺序：

```text
M1 RoPE
M2 RoPE + GQA
M3 RoPE + GQA + QK norm
```

不要一次性全改。

## Ablation N：condition fusion

第一版使用：

```text
S_i independent stim token
eye token通过attention读取S_i
```

后续可比较：

```text
N1 independent S_i token
N2 broadcast stim_token到L/R eye token
N3 independent S_i + broadcast stim_token
N4 concat + MLP fusion
N5 task AdaLN + independent S_i
N6 two-stream eye/stim cross-attention
```

## Ablation O：event-aware / event-driven tokenization

第一版仍使用固定20ms patch。

后续可逐步试：

```text
O1 fixed 20ms patch + event-aware mask
O2 variable-length event segment token
O3 event boundary tokenization without event label input
```

---

# 27. 实验优先级

按这个顺序执行：

```text
0. pytest全部通过
1. debug tiny overfit
2. 主baseline：CNN + 20ms + S/L/R sequence + no last_stim + no goal + no CLS + velocity
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
14. true asymmetric MAE
15. 768×16 larger model
16. RoPE/GQA/Qwen-style block
17. condition fusion variants
18. event-aware / event-driven tokenization
```

第一轮最重要的6个结果：

```text
1. 主baseline是否能稳定训练
2. tiny overfit是否成功
3. S/L/R sequence mapping是否正确
4. 是否明显优于previous-value baseline
5. span mask下是否接近或优于linear interpolation
6. memory task中add_last_stim是否有帮助
```

---

# 28. 不要做的事

第一版不要做：

```text
不要实现decoder-only causal LM
不要实现DeepSeek MoE
不要实现复杂multi-token prediction
不要加入CLS
不要加入last_stim
不要加入goal
不要用feature-wise tokens
不要把missing当成MAE mask
不要在missing处用0坐标算loss
不要在blink处用0坐标算xy/area loss
不要对S_i stim token做MAE mask
不要对S_i stim token算reconstruction loss
不要把all-missing eye token作为MAE候选
不要让all-missing eye token作为key/value污染attention
不要把MAE-masked eye token从attention里删掉
不要在overlap patch里随机mask单个patch造成原始frame泄漏
不要用pretrain_val/pretrain_test统计area归一化参数
不要实现session-level split
```

第一版的主线就是：

```text
content:
  x, y, area, blink

quality:
  missing

condition:
  S_i = task_id, fix_on, stim_on, stim_x, stim_y, time

sequence:
  S_i, L_i, R_i

MAE:
  只人工mask有效L/R eye content
  不mask S_i

target:
  重建masked eye content

attention:
  S_i只要不是padding就有效
  all-missing L/R eye token无效
  MAE-masked L/R eye token仍有效

loss:
  S_i不算
  missing处不算
  blink处只算blink
  非missing且非blink处算x/y/area/velocity
```

这份计划可以直接作为 Codex 的任务书使用。
