# EyeMAE Fast Dataset v2 Report

Date: 2026-06-23

This materialized v2 dataset fixes the AD, PD, and MCI label/identity issues
before reporting dataset statistics. On 2026-06-23, the packed arrays were kept
unchanged while CSV/JSON metadata was updated to the corrected `ml_subject_id`
identity rule. The source is:

```text
/mnt/disk_sde/data-260606/extracted/cd_speed4_hard_blink_fixed_pd_20260618/matched_groups_full_BACKUP_intermediate_20260618
```

The generated dataset is:

```text
/mnt/disk_sde/data-260606/extracted/eyemae_fast_dataset_v2
```

## Format

The dataset is not converted back to one-trial `.npz` files. It is stored as
packed mmap shards. Each root has:

```text
shards/shard_xxxxxx/X_data.npy
shards/shard_xxxxxx/y_frame.npy
shards/shard_xxxxxx/X_offsets.npy
shards/shard_xxxxxx/X_lengths.npy
```

`X_data.npy` stores many concatenated trials as `[T, 10]`; `y_frame.npy` stores
left/right frame QC labels as `[T, 2]`. CSV indexes map each trial to
`shard_id`, `frame_offset`, and `frame_length`.

## Cleaning Rules Applied Before Statistics

| Area | Rule |
| --- | --- |
| AD | Drop all `AD/匹配后/实验组`; those rows duplicate `AD/AD组/患病`. Drop `AD/匹配后/对照组/GaoLianYing`; GaoLianYing remains only as AD patient. |
| PD | Drop all `PD相关/*匹配后/实验组`. Deduplicate matched controls by `name + age + education`; if repeated, keep the canonical matched-control source. |
| PD named cases | `ZhangMingSha` remains only as healthy control. `LiuWenFang` and `ZhangMingLin` do not enter PD downstream tasks because their PD rows are matched experimental rows and are dropped. |
| MCI original | Use only `MCI/实验组` and `MCI/对照组`; exclude all `MCI/匹配后`. |
| MCI matched | Separate task. Use only `MCI/匹配后`; source matched labels are reversed: `实验组 -> health_label=0`, `对照组 -> health_label=1`. |
| Epilepsy/migraine/detox | Preserve the v1 identity-level train/validation/test split. |

Identity key is parsed as `name + age + education_code` from `source_stem`, not
only raw folder name. The education code is the filename field two positions
before age, such as `XX/CZ/GZ/ZZ/DZ/BK/SS/BS/WM`. `FangDeXiu|age:77`,
`LuXingQiong|age:61`, and `雷妮莎|age:23` are intentionally kept as
`edu:MIXED` so those metadata-inconsistent subjects stay merged as requested.

## Pretrain

| split | trials | identities | frames |
| --- | ---: | ---: | ---: |
| train | 580,190 | 3,829 | 1,226,487,793 |
| validation | 32,401 | 213 | 69,333,342 |
| test | 32,387 | 213 | 68,668,716 |

Total: 644,978 trials, 4,255 identities, 1,364,489,851 frames, 27 shards.

Pretrain drops:

| dropped row type | rows |
| --- | ---: |
| AD matched control GaoLianYing | 133 |
| AD matched experimental | 13,016 |
| duplicate noncanonical control identity | 207,915 |
| MCI matched rows for pretrain | 33,154 |
| PD matched experimental | 50,957 |

Pretrain train/validation/test identity overlap is zero.

Control deduplication check: after cleaning, pretrain has 458,863 control rows
from 2,997 identities, and 0 control identities appear under multiple source
subjects. Not all non-master controls are covered by `对照组数据汇总` under the
strict `name + age + education` identity rule, so v2 keeps non-duplicated
non-master controls:

| remaining non-master control source | rows |
| --- | ---: |
| `MCI/对照组/对照组` | 35,362 |
| `偏头痛/对照组/对照组` | 10,614 |
| `戒毒所/对照组原始数据/对照组` | 3,115 |
| `癫痫/对照组/对照组` | 20,812 |

## Downstream

| task | trials | subjects | split policy | label type |
| --- | ---: | ---: | --- | --- |
| `pd_related_5class` | 127,863 | 895 | identity-stratified random 64/16/20 | 5-class |
| `pd_binary` | 127,863 | 895 | same identity assignment as PD 5-class | binary |
| `epilepsy_binary` | 139,908 | 891 | preserve v1 identity split | binary |
| `detox_binary` | 26,338 | 171 | preserve v1 identity split | binary |
| `migraine_binary` | 32,130 | 209 | preserve v1 identity split | binary |
| `ad_binary` | 37,808 | 252 | identity-stratified random 64/16/20 | binary |
| `mci_binary` | 58,279 | 383 | identity-stratified random 64/16/20 | binary |
| `mci_matched_binary` | 33,154 | 218 | identity-stratified random 64/16/20 | binary |

All 8 downstream tasks have zero `ml_subject_id` overlap across
train/validation/test and zero subject-level label conflicts.

For epilepsy, migraine, and detox, v2 preserves the v1 split at the trial key
level: `(relative_source_path, original_trial_index, direction, task)` has
0 missing rows and 0 split mismatches against v1 for all three tasks. The v2
subject counts can differ from v1 because v2 uses `name + age + education` as
the identity key instead of the older source-path-derived `ml_subject_id`.

## AD/PD/MCI Checks

| check | result |
| --- | --- |
| AD GaoLianYing | 133 rows retained, all `AD/AD组/患病`, `health_label=1`; no matched-control rows remain. |
| PD ZhangMingSha | 160 rows retained, all `帕金森匹配后/对照组`, `health_label=0`. |
| PD LiuWenFang | 0 rows in corrected PD downstream tasks. |
| PD ZhangMingLin | 0 rows in corrected PD downstream tasks. |
| MCI original | `MCI/对照组 -> health_label=0`: 35,362 rows; `MCI/实验组 -> health_label=1`: 22,917 rows. |
| MCI matched | `MCI/匹配后/实验组 -> health_label=0`: 16,720 rows; `MCI/匹配后/对照组 -> health_label=1`: 16,434 rows. |

## Generated Files

| file | purpose |
| --- | --- |
| `scripts/build_eyemae_fast_dataset_v2.py` | Rebuilds v2 packed dataset from raw source with the cleaning rules above. |
| `/mnt/disk_sde/data-260606/extracted/eyemae_fast_dataset_v2/README.md` | Dataset-level user README. |
| `/mnt/disk_sde/data-260606/extracted/eyemae_fast_dataset_v2/v2_build_summary.json` | Full machine-readable build summary. |
| `/mnt/disk_sde/data-260606/extracted/eyemae_fast_dataset_v2/pretrain/audit_summary.json` | Pretrain audit and split summary. |
| `/mnt/disk_sde/data-260606/extracted/eyemae_fast_dataset_v2/finetune/<task>/split_summary.json` | Per-task downstream split summary. |
