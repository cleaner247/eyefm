# EyeMAE 高速 packed 数据集适配计划

## 0. 目标

把 EyeMAE 的 pretrain 和 downstream fine-tune 从“每个 trial 一个 `.npz` 文件”的读取方式，迁移到更适合 PyTorch DataLoader 随机访问的 packed 数据格式。

核心目标：

```text
减少每个 epoch 的小文件 open/read 次数
避免每个 trial 单独解压 NPZ
保留 pretrain/fine-tune 所需的全部 metadata
保证 subject-level split、去重、标签和 task 信息可审计
让下游各疾病 view 可以共享同一份底层眼动数组，不重复存储
```

当前新数据集：

```text
/mnt/disk_sde/data-260606/extracted/cd_speed4_hard_blink_ml_ready_subjectkey_20260619
```

已经解决了“小文件太多”的一部分问题，但它仍然是每个 view/split 一个大 `.npz`。`.npz` 是 zip 容器，不适合作为最终训练主格式：`np.load(..., mmap_mode="r")` 不能对 zip 内部数组做真正 memmap，worker 随机访问时容易变成整块数组解压或大块内存缓存。

因此推荐把训练主格式升级为：

```text
mmap-friendly shard store:
  大数组使用 .npy
  metadata 使用 csv/json，必要时另存 parquet
  view/split 只保存轻量 index，不重复保存 X_data
```

这里不是“一个 trial 一个文件”。一个 shard 是很多 trial 合并后的连续大数组，通常可以放几千到几万个 trial；`trial_index.csv` 只是轻量 metadata，一行定位一个 trial 在大数组里的 offset/length。训练时 DataLoader 打开的文件数约等于 shard 数，而不是 trial 数。

---

## 1. 当前文档和代码需要的数据语义

## 1.1 pretrain 需要的信息

模型前向真正需要的是帧级信号和 task：

```text
eye:     (T, 8)
         left_x, left_y, left_s, left_qc_label,
         right_x, right_y, right_s, right_qc_label

fix_on:  (T,)

stim:    (T, 3)
         stim_on, stim_x, stim_y

task_id: scalar in {0,1,2,3}
```

训练工程还必须保留：

```text
subject_id:
  用于 subject-level pretrain split
  用于 area stats 只从 pretrain_train 统计
  用于审计 downstream_test subject 是否在 pretrain 中出现过

trial_id:
  用于稳定定位样本
  用于去重
  用于异常样本追踪、可视化、复现实验
```

pretrain 不使用疾病 label 作为训练目标，但可以保留 label metadata 供审计。

## 1.2 fine-tune 需要的信息

downstream fine-tune 需要：

```text
trial 输入:
  和 pretrain 完全一致的 eye/fix_on/stim/task_id preprocessing

trial label:
  health_label
  可选 pd_disease_label

subject key:
  用于 subject-level train/val/test split
  用于 subject-level AUROC/AUPRC 聚合

view 信息:
  AD
  MCI
  PD相关
  偏头痛
  戒毒所
  癫痫
  AD匹配后
  MCI匹配后
  PD相关_帕金森病匹配后
  PD相关_震颤匹配后
  PD相关_特发性震颤匹配后
  PD相关_运动障碍匹配后
```

---

## 2. 推荐数据集组织方式

推荐把数据分成两层：

```text
1. storage layer:
   只存唯一底层 trial 的连续帧数组。
   所有 pretrain 和 downstream view 共享它。

2. index/view layer:
   只存哪些 trial 属于哪个 split/view/task/label。
   不复制 X_data。
```

目录结构：

```text
eyemae_fast_dataset_v1/
  dataset_manifest.json
  columns.json
  label_maps.json
  audit_summary.json
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
      X_data.npy
      y_frame.npy
      X_offsets.npy
      X_lengths.npy
      trial_index.csv
    ...

  pretrain/
    pretrain_all_unique.csv
    pretrain_train.csv
    pretrain_val.csv
    pretrain_test.csv
    pretrain_split_summary.json

  downstream/
    AD/
      train.csv
      validation.csv
      test.csv
      split_summary.json
    MCI/
      train.csv
      validation.csv
      test.csv
      split_summary.json
    PD相关/
      train.csv
      validation.csv
      test.csv
      split_summary.json
    PD相关_帕金森病匹配后/
      train.csv
      validation.csv
      test.csv
      split_summary.json
    ...

```

说明：

```text
shards/ 是唯一真实帧数据。
pretrain/ 和 downstream/ 只保存轻量 index。
同一个 trial 如果属于多个 view，只在 shards/ 里存一次。
不生成一个 trial 一个 npz/csv/json 文件。
```

---

## 3. storage layer 格式

每个 shard 保存一批完整 trial。不要让同一个 trial 横跨两个 shard。

shard 粒度推荐：

```text
不要一个 trial 一个 shard/file。
不要全数据一个超大 shard。
推荐每个 shard 包含很多 trial，目标大小约 1-4 GiB。
```

原因：

```text
一个 trial 一个 shard:
  文件数量仍然接近 trial 数。
  虽然 .npy 比 .npz 少了解压，但 open/stat/seek 元数据开销仍然很大。
  多 worker 随机采样时仍然容易让 CPU/文件系统成为瓶颈。

全数据一个超大 shard:
  文件打开次数最低，但写入、校验、重做、损坏恢复成本高。
  多 worker 在一个巨大文件上随机 page fault，局部性差。
  后续追加或只重建部分数据不方便。
  train/val/test 或 downstream view 的调试不方便。

1-4 GiB 多 trial shard:
  文件数量从几十万降到几十或几百个。
  每个 worker 可以缓存已打开 shard 的 mmap 句柄。
  OS page cache 和 readahead 更容易发挥作用。
  单个 shard 重建成本可控。
  可以按 subject 或 source 组织，兼顾随机训练和审计。
```

默认建议：

```text
shard_target_size_gib: 2
shard_min_size_gib: 1
shard_max_size_gib: 4
trial_must_not_cross_shard: true
```

如果单个 subject 的 trial 总量很大，也不要强行把一个 subject 全部放进同一个 shard；subject-level split 由 index 控制，不依赖物理 shard 边界。物理 shard 的第一目标是训练读取速度和可维护性。

```text
shards/shard_000000/X_data.npy
  shape: (total_frames_in_shard, 10)
  dtype: float32

shards/shard_000000/y_frame.npy
  shape: (total_frames_in_shard, 2)
  dtype: int8

shards/shard_000000/X_offsets.npy
  shape: (n_trials_in_shard,)
  dtype: int64

shards/shard_000000/X_lengths.npy
  shape: (n_trials_in_shard,)
  dtype: int32

shards/shard_000000/trial_index.csv
  one row per local trial
```

`X_data` 列定义：

| 列号 | 名称 | 说明 |
| --- | --- | --- |
| 0 | `left_x` | 左眼 x |
| 1 | `left_y` | 左眼 y |
| 2 | `left_s` | 左眼 pupil/area/size |
| 3 | `right_x` | 右眼 x |
| 4 | `right_y` | 右眼 y |
| 5 | `right_s` | 右眼 pupil/area/size |
| 6 | `stimulus_x` | 刺激 x |
| 7 | `stimulus_y` | 刺激 y |
| 8 | `stimulus_on` | 刺激呈现开关 |
| 9 | `cross_on` | 注视点/十字开关，进入模型时映射为 `fix_on` |

`y_frame` 列定义：

| 列号 | 名称 | 取值 |
| --- | --- | --- |
| 0 | `left_qc_label` | `0=VALID`, `1=BLINK`, `2=MISSING` |
| 1 | `right_qc_label` | `0=VALID`, `1=BLINK`, `2=MISSING` |

从 shard 恢复一个 trial：

```python
X = np.load("X_data.npy", mmap_mode="r")
Y = np.load("y_frame.npy", mmap_mode="r")
offsets = np.load("X_offsets.npy", mmap_mode="r")
lengths = np.load("X_lengths.npy", mmap_mode="r")

start = offsets[local_trial_index]
end = start + lengths[local_trial_index]

x_trial = X[start:end]
y_frame_trial = Y[start:end]
```

映射到当前 EyeMAE 内部 trial dict：

```text
eye[:, 0] = X[:, 0]  left_x
eye[:, 1] = X[:, 1]  left_y
eye[:, 2] = X[:, 2]  left_s
eye[:, 3] = y_frame[:, 0] left_qc_label

eye[:, 4] = X[:, 3]  right_x
eye[:, 5] = X[:, 4]  right_y
eye[:, 6] = X[:, 5]  right_s
eye[:, 7] = y_frame[:, 1] right_qc_label

fix_on = X[:, 9]

stim[:, 0] = X[:, 8]  stimulus_on
stim[:, 1] = X[:, 6]  stimulus_x
stim[:, 2] = X[:, 7]  stimulus_y
```

---

## 4. trial-level metadata

全局 `trials.csv` 和每个 shard 的 `trial_index.csv` 至少包含以下列。

必需定位列：

```text
global_trial_id
shard_id
local_trial_index
frame_offset
frame_length
```

必需 pretrain 列：

```text
ml_subject_id
trial_id
task
task_id
source_file_uid
original_trial_index
direction
```

推荐 `trial_id`：

```text
trial_id = source_file_uid + "|" + original_trial_index + "|" + direction + "|" + task
```

如果 `source_file_uid` 不稳定，使用：

```text
trial_id = source_top + "|" + source_dataset + "|" + source_group + "|" + source_subtype + "|" + subject + "|" + source_stem + "|" + original_trial_index + "|" + direction + "|" + task
```

必需 subject/source 列：

```text
source_top
source_dataset
source_group
source_subtype
subject
ml_subject_id
source_suffix
```

必需质量控制列：

```text
left_final_keep
right_final_keep
left_final_reject
right_final_reject
left_blink_points
left_missing_points
right_blink_points
right_missing_points
n_samples
```

下游标签列可以放在 view index 中，也可以在全局 trials 中保留默认来源标签：

```text
health_label
pd_disease_label
```

但要注意：疾病二分类 label 是 view-specific 的。比如同一个 control trial 可以在多个疾病 view 中作为 `health_label=0`。因此最终训练时应以 downstream view index 中的 label 为准。

---

## 5. subject-level metadata

`subjects.csv` 至少包含：

```text
ml_subject_id
source_top
source_dataset
source_group
source_subtype
subject
num_trials
num_frames
available_tasks
has_left_eye
has_right_eye
```

下游可选列：

```text
AD_label
MCI_label
PD_related_label
PD_subtype
migraine_label
detox_label
epilepsy_label
```

注意：

```text
所有 split 必须基于 ml_subject_id。
禁止使用 raw subject 字符串做 split。
```

---

## 6. pretrain index 设计

## 6.1 pretrain_all_unique

pretrain 应使用去重后的底层 trial index：

```text
pretrain/pretrain_all_unique.csv
```

每行至少包含：

```text
global_trial_id
shard_id
local_trial_index
ml_subject_id
task_id
frame_length
```

去重 key：

```text
source_file_uid + original_trial_index + direction + task
```

如果源数据重新导出导致 `source_file_uid` 变化，则用更长的 source path key。

## 6.2 pretrain control 覆盖范围

如果目标是复现或增强当前 EyeMAE pretrain，新的 packed 训练集必须使用同一套 speed4 hard blink 清洗版本。不要再从旧 `cd_no_cond2_structured_20260609` 补数据，因为它的清理算法和当前新数据集不同。

pretrain 数据池推荐来源：

```text
主体:
  cd_speed4_hard_blink_ml_ready_subjectkey_20260619
  12 个 downstream view 去重后的底层 trial

额外 control-only:
  cd_speed4_hard_blink_fixed_pd_20260618/analysis_subject_unique/剩余对照组
  当前核对为 251,448 个 .npz trial
```

当前 `cd_speed4_hard_blink_ml_ready_subjectkey_20260619` 是 downstream view 数据。它的健康样本已经在各疾病 view 内体现，例如 AD 的健康样本来自 `AD/匹配后/对照组/...`，PD 相关健康样本来自 `PD相关/<subtype>匹配后/对照组/...`。因此 fine-tune 的对照组不是缺失的。

pretrain 需要额外审计的是：20260618 `analysis_subject_unique/剩余对照组` 中哪些 control-only trial 没有进入 12 个 downstream view，并把这些同清洗版本的 trial 纳入无标签预训练池。不能把 12 个 downstream view 简单拼起来作为完整 pretrain 数据池，因为这些 view 之间有重叠，也可能没有覆盖全部 control-only trial。

`analysis_subject_unique/共享匹配对照组` 不作为默认额外补充来源；共享匹配对照原则上已经被当前 downstream view 的健康样本覆盖。只有覆盖审计发现当前 downstream 健康样本缺失某些共享匹配对照 trial 时，才单独处理。

因此推荐生成一个专门的 pretrain storage/index：

```text
pretrain_all_unique:
  从 20260619 downstream views + 20260618 analysis_subject_unique/剩余对照组 生成
  使用同一套 speed4 hard blink 清洗版本
  先对 12 个 downstream view 去重
  再补入剩余对照组里未覆盖的 control-only trial
  按 global_trial_id 去重
  审计当前 downstream 健康样本和 20260618 剩余对照组之间的覆盖关系
```

## 6.3 pretrain split

pretrain split 文件：

```text
pretrain_train.csv
pretrain_val.csv
pretrain_test.csv
```

split 规则：

```text
硬约束：同一个 ml_subject_id 不跨 split。
split 的基本单位是 subject，不是 trial。
每个 subject 通常完成四个 task，因此不要为了 task 分布把同一个 subject 的 trial 拆到不同 split。
task/source 分布只作为 split summary 的审计统计；如果极端不均衡，只能通过 subject 级重抽样或换 seed 处理。
area stats 只用 pretrain_train
```

推荐 split summary：

```json
{
  "strategy": "subject_heldout",
  "seed": 42,
  "subject_key": "ml_subject_id",
  "num_train_trials": 0,
  "num_val_trials": 0,
  "num_test_trials": 0,
  "num_train_subjects": 0,
  "num_val_subjects": 0,
  "num_test_subjects": 0,
  "task_counts": {},
  "source_top_counts": {},
  "no_subject_overlap": true
}
```

---

## 7. downstream fine-tune index 设计

每个 downstream view 单独建目录：

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
ml_subject_id
task_id
frame_length
health_label
pd_disease_label
view
```

`pd_disease_label` 规则：

```text
PD相关:
  y_trial = [health_label, pd_disease_label]
  pd_disease_label:
    -1 control/NA
     0 帕金森病
     1 震颤
     2 特发性震颤
     3 运动障碍

PD相关_帕金森病匹配后:
PD相关_震颤匹配后:
PD相关_特发性震颤匹配后:
PD相关_运动障碍匹配后:
  y_trial = [health_label]
  subtype 由 view 名确定
```

fine-tune split 规则：

```text
每个 view 独立 subject-level split
同一个 ml_subject_id 不跨该 view 的 train/validation/test
主指标按 ml_subject_id 聚合
trial-level metric 只作为辅助
```

本计划暂不包含 K-fold 和 train/test-only 分支，先保持 train/validation/test 三 split。

---

## 8. DataLoader 适配方案

新增 dataset 类型：

```text
data.format: packed_mmap
data.npz_schema: cd_speed4_hard_blink_mmap_v1
```

核心类：

```text
PackedTrialStore
PackedPretrainDataset
PackedDownstreamDataset
```

读取流程：

```text
1. Dataset 初始化时只读 index csv，不打开大数组。
2. 每个 worker 第一次访问某个 shard 时，用 np.load(..., mmap_mode="r") 懒打开 X_data/y_frame/offsets/lengths。
3. __getitem__ 根据 shard_id/local_trial_index 切片出一个 trial。
4. 组装成当前 preprocess_trial 需要的 trial dict。
5. 后续 preprocess/patchify/collate 尽量复用原代码。
```

worker 内 shard cache：

```python
class PackedTrialStore:
    def __init__(self, root):
        self.root = Path(root)
        self._cache = {}

    def get_shard(self, shard_id):
        if shard_id not in self._cache:
            p = self.root / "shards" / shard_id
            self._cache[shard_id] = {
                "X": np.load(p / "X_data.npy", mmap_mode="r"),
                "Y": np.load(p / "y_frame.npy", mmap_mode="r"),
                "offsets": np.load(p / "X_offsets.npy", mmap_mode="r"),
                "lengths": np.load(p / "X_lengths.npy", mmap_mode="r"),
            }
        return self._cache[shard_id]
```

重要实现要求：

```text
不要在 Dataset __init__ 里把大数组全部读进内存
不要在 __getitem__ 每次重新 np.load
不要用 zip npz 作为训练主格式
不要把 matched view 的 X_data 重复存储
```

---

## 9. batch 和采样策略

packed 数据只解决 IO，不自动解决 batch padding 浪费。

推荐保留两种 batch：

```text
fixed trial batch:
  每个 batch 固定 N 个 trial
  简单稳定
  但长短 trial 混在一起时 padding 浪费较大

token-based dynamic batch:
  按 patch/token 数动态组成 batch
  控制每个 batch 的总 token 数
  更适合不同 trial 长度差异大的数据
```

为了加速，建议 index 里保留：

```text
frame_length
num_patches = frame_length // patch_samples
```

这样 sampler 不需要打开数组就能做 length bucketing 或 token-based batching。

pretrain 推荐：

```text
先实现 fixed trial batch + length bucketing
再实现 token-based dynamic batch
```

fine-tune 推荐：

```text
保留 subject-balanced sample_weight
可以按 subject 或 trial 做 sampler
metric 聚合必须按 subject
```

---

## 10. area stats 和 preprocessing

如果重新用 packed 数据 pretrain：

```text
area stats 必须从 packed pretrain_train 重新统计
不能继续默认使用 cd_no_cond2_structured_20260609 的 stats
```

原因：

```text
新数据来自 speed4 hard blink 清洗版本
blink/missing/area 分布可能变化
pretrain/fine-tune 输入分布应保持一致
```

推荐输出：

```text
outputs/area_stats_packed_pretrain_subject_heldout_seed42.json
```

fine-tune 默认使用同一份 packed pretrain area stats。

---

## 11. 必须保留的审计文件

每次生成 packed 数据集，都必须输出：

```text
dataset_manifest.json
audit_summary.json
pretrain/pretrain_split_summary.json
downstream/<view>/split_summary.json
```

审计项：

```text
array shape 一致
X_offsets + X_lengths 不越界
manifest 行数等于 trial 数
global_trial_id 唯一
trial_id 去重结果可追踪
ml_subject_id 非空
pretrain train/val/test subject 无重叠
downstream 每个 view train/val/test subject 无重叠
health_label 只允许 0/1
PD相关 pd_disease_label 只允许 -1/0/1/2/3
task_id 只允许 0/1/2/3
每个 split 的 task/source/label/subject 统计写入 summary
同一 downstream view 内，raw subject 不能同时拥有 health_label=0 和 1
同一 downstream view 内，raw recording 不能同时拥有 health_label=0 和 1
同一 downstream view 的 train/validation/test 之间，raw subject 不能重叠
同一 downstream view 的 train/validation/test 之间，raw recording 不能重叠
```

其中 raw subject 至少应包含 `source_top + subject`；raw recording 至少应包含
`source_top + subject + source_stem/source_csv basename`，或使用稳定的
`source_file_uid + original_trial_index + direction + task`。`ml_subject_id`
不能作为唯一身份审计键，因为它可能包含 `source_dataset/source_group`，从而把同一真实
subject 在原始组、匹配后组、实验组、对照组中拆成多个身份，掩盖 label 冲突和 split
泄漏。

速度审计：

```text
记录 DataLoader throughput:
  trials/sec
  patches/sec
  frames/sec
  GPU utilization
  data_time / step_time

目标：
  GPU 不应长期等待 CPU IO
  每个 epoch open 文件数约等于 worker 访问过的 shard 数，而不是 trial 数
```

---

## 12. 从当前 20260619 数据迁移的步骤

## 12.1 第一阶段：让当前 downstream view 可训练

输入：

```text
cd_speed4_hard_blink_ml_ready_subjectkey_20260619/<view>/{train,validation,test}.npz
cd_speed4_hard_blink_ml_ready_subjectkey_20260619/<view>/manifest_*.csv
```

转换：

```text
1. 读取每个 view/split 的 packed NPZ。
2. 按 source_file_uid + original_trial_index + direction + task 生成 global_trial_id。
3. 去重并写入统一 shards。
4. 生成 downstream/<view>/<split>.csv。
5. 保留当前 view 的 health_label/pd_disease_label。
```

结果：

```text
可以跑所有 downstream fine-tune view。
但 pretrain 仍然需要额外生成 pretrain_all_unique，因为当前 12 个 downstream view 之间有重叠，且不保证覆盖 20260618 analysis_subject_unique/剩余对照组 里的全部 control-only trial。
```

## 12.2 第二阶段：生成完整 pretrain_all

输入应来自同一套 speed4 hard blink 清洗结果，而不是旧 `cd_no_cond2_structured_20260609`：

```text
主体:
  cd_speed4_hard_blink_ml_ready_subjectkey_20260619 的 12 个 downstream view 去重 trial

额外 control-only:
  cd_speed4_hard_blink_fixed_pd_20260618/analysis_subject_unique/剩余对照组
```

输出：

```text
pretrain/pretrain_all_unique.csv
pretrain/pretrain_train.csv
pretrain/pretrain_val.csv
pretrain/pretrain_test.csv
```

这样才能让新 packed pretrain 在数据覆盖上不弱于旧 per-trial pretrain。

## 12.3 第三阶段：统一 pretrain 和 fine-tune

最终目标：

```text
同一套 shards/
同一套 trials.csv / subjects.csv
pretrain 和 downstream 只用不同 index
```

这样不会重复存储大数组，也不会出现 pretrain/fine-tune 使用不同清洗版本的问题。

---

## 13. 配置文件建议

新增 pretrain config：

```yaml
data:
  format: packed_mmap
  data_dir: /mnt/disk_sde/data-260606/extracted/eyemae_fast_dataset_v1
  train_index: pretrain/pretrain_train.csv
  val_index: pretrain/pretrain_val.csv
  test_index: pretrain/pretrain_test.csv
  sampling_rate: 1000
  subject_key: ml_subject_id
  task_column: task_id
  trial_id_column: trial_id
```

新增 downstream config：

```yaml
data:
  format: packed_mmap
  data_dir: /mnt/disk_sde/data-260606/extracted/eyemae_fast_dataset_v1
  sampling_rate: 1000

downstream:
  view: PD相关_帕金森病匹配后
  train_index: downstream/PD相关_帕金森病匹配后/train.csv
  val_index: downstream/PD相关_帕金森病匹配后/validation.csv
  test_index: downstream/PD相关_帕金森病匹配后/test.csv
  label_column: health_label
  subject_key: ml_subject_id
```

兼容策略：

```text
旧 data.format=npz_per_trial 继续保留。
新 data.format=packed_mmap 走新 Dataset。
model/preprocess/patchify/loss 尽量不改。
```

---

## 14. 当前新数据集还缺什么

当前 `cd_speed4_hard_blink_ml_ready_subjectkey_20260619` 已有：

```text
task 信息在 manifest 中
ml_subject_id
health_label
PD相关的 pd_disease_label
train/validation/test split
X_data/y_frame/y_trial packed arrays
```

还不满足最终适配目标的点：

```text
1. 训练主格式仍是大 NPZ，不是真正 mmap-friendly。
2. view/split 重复存储 X_data，matched view 和 overall view 有重叠。
3. 没有独立 pretrain_all_unique index。
4. 当前 12 个 view 已包含各自 fine-tune 所需健康对照，但不保证覆盖 20260618 analysis_subject_unique/剩余对照组 里的完整 control-only pretrain 数据池。
5. task_id 没有放进 NPZ 数组，必须从 manifest 映射。
6. 当前 EyeMAE loader 还不支持 packed index。
7. area stats 需要基于 packed pretrain_train 重算。
8. 需要补充统一 audit，确保 pretrain 和 downstream 使用同一 subject key、同一清洗版本、同一底层 trial id。
```

---

## 15. 推荐实施顺序

第一步：

```text
实现 packed_mmap converter
把当前 20260619 downstream view 转成统一 shards + downstream index
实现 PackedDownstreamDataset
先验证 fine-tune 能跑，且结果和当前 view label 对齐
```

第二步：

```text
从 20260619 downstream views + 20260618 analysis_subject_unique/剩余对照组 生成 pretrain_all_unique
审计并纳入同清洗版本的剩余对照组/control-only trial
实现 PackedPretrainDataset
重算 area stats
启动 packed pretrain
```

第三步：

```text
做 DataLoader benchmark
比较旧 per-trial NPZ 和新 packed_mmap:
  data_time
  step_time
  GPU utilization
  epoch time
```

第四步：

```text
把 packed_mmap 作为后续正式数据格式
旧 npz_per_trial 只保留为兼容路径
```
