# Newdata v3 Old-Baseline Dataset Audit

Audit time: 2026-06-21

Old baseline:

- `/mnt/disk_sde/data-260606/extracted/cd_no_cond2_structured_20260609`

Current fast dataset:

- `/mnt/disk_sde/data-260606/extracted/eyemae_fast_dataset_v1`

Reason:

- The current fast dataset had label and raw-identity issues in some downstream
  views.
- The old cleaned dataset used a different cleaning algorithm, but its label
  information is treated here as the historical label baseline.
- This audit checks whether the current problems already existed in the old
  dataset, and whether the current new dataset has other label/identity issues
  when compared against that old baseline.

## Method

The old and new datasets do not use stable comparable `source_file_uid` values.
For example, the same source CSV can receive different UID numbers across the
old per-trial NPZ dataset and the new packed dataset. Therefore this audit did
not compare rows by `source_file_uid`.

Two comparison keys were used:

- exact trial key: `(relative source CSV path, original_trial_index, direction)`;
- raw subject key: `subject`, after stripping dataset/group-specific identity
  wrappers.

Old labels were derived as:

- old disease folders: `group=健康 -> 0`, `group=患病 -> 1`;
- old `共享匹配对照组` and `剩余对照组`: control label `0`.

The old healthy control pools were included as baseline anchors:

| old control pool | rows | subjects | label |
| --- | ---: | ---: | --- |
| `共享匹配对照组` | 80,740 | 200 | 0 |
| `剩余对照组` | 251,448 | 1,652 | 0 |

## Old Dataset Baseline

The six old disease folders do not show raw-subject, source-file, or exact-trial
label conflicts under the old labels.

| old disease folder | rows | subjects | label 0 rows | label 1 rows | raw-subject label conflicts | file label conflicts | exact-trial label conflicts |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `PD相关` | 128,711 | 746 | 73,326 | 55,385 | 0 | 0 | 0 |
| `AD` | 20,226 | 136 | 4,872 | 15,354 | 0 | 0 | 0 |
| `MCI` | 58,279 | 383 | 35,362 | 22,917 | 0 | 0 | 0 |
| `偏头痛` | 28,092 | 182 | 15,379 | 12,713 | 0 | 0 | 0 |
| `戒毒所` | 18,955 | 124 | 8,366 | 10,589 | 0 | 0 | 0 |
| `癫痫` | 127,695 | 812 | 58,538 | 69,157 | 0 | 0 | 0 |

Interpretation: the old dataset itself does not contain the raw label conflict
patterns that were later observed in the current combined/new views. The
current issues are therefore not explained as pre-existing contradictions in
`cd_no_cond2_structured_20260609`.

Narrow check for the specific AD/PD question:

| old scope | rows | raw subjects | subjects with both label 0 and label 1 |
| --- | ---: | ---: | ---: |
| `AD` only | 20,226 | 136 | 0 |
| `PD相关` only | 128,711 | 746 | 0 |
| `AD` + old global healthy pools | 352,414 | 1,988 | 0 |
| `PD相关` + old global healthy pools | 460,899 | 2,598 | 0 |

So the current AD/PD pattern where one raw person appears as both healthy and
patient is not present in the old data under the same raw-subject check.

## Current Official Views vs Old Baseline

`direct found` is the fraction of current rows that could be aligned to old
data by exact trial key. A low value does not by itself mean a label error,
because the current dataset can contain new cleaning output or rows from the
newer healthy/control pools. `direct label mismatch` is the stricter check among
rows that were exactly aligned.

| current view | rows | direct found | direct label mismatch rows | subject-baseline found | subject label mismatch rows | current raw conflict keys / rows | current raw split-overlap keys / rows | status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `PD相关_random_seed20260620` | 220,427 | 76.82% | 0 | 99.94% | 467 | 1 / 320 | 321 / 100,249 | unresolved |
| `PD相关_binary_random_seed20260620` | 220,427 | 76.82% | 0 | 99.94% | 467 | 1 / 320 | 321 / 100,249 | unresolved |
| `癫痫` | 139,908 | 99.22% | 0 | 99.22% | 0 | 0 / 0 | 0 / 0 | clean |
| `戒毒所` | 26,338 | 100.00% | 0 | 100.00% | 0 | 0 / 0 | 0 / 0 | clean |
| `偏头痛` | 32,130 | 100.00% | 0 | 100.00% | 0 | 0 / 0 | 0 / 0 | clean |
| `AD` | 50,957 | 74.20% | 0 | 100.00% | 133 | 1 / 399 | 54 / 16,069 | unresolved |
| `MCI_original_only_no_matched` | 58,279 | 100.00% | 0 | 100.00% | 0 | 0 / 0 | 0 / 0 | clean |
| `MCI匹配后_random_seed20260622_label_fixed` | 33,154 | 0.00% | 0 | 100.00% | 0 | 0 / 0 | 0 / 0 | clean by old-subject anchor |

Main findings:

- Exact trial rows that can be matched back to old data have zero label
  mismatches in all current official views.
- `癫痫`, `戒毒所`, `偏头痛`, `MCI_original_only_no_matched`, and
  `MCI匹配后_random_seed20260622_label_fixed` are clean under this old-baseline
  audit.
- `PD相关_random_seed20260620`, `PD相关_binary_random_seed20260620`, and `AD`
  still have raw-subject/raw-file risks that are not present in the old
  disease-folder baseline.
- The new dataset's internal `ml_subject_id` split checks are not sufficient
  for identity safety, because `ml_subject_id` includes source/dataset/group
  fields. A raw person can therefore be hidden behind multiple `ml_subject_id`
  values.

## Issue Status by Task

### MCI

The old `MCI` folder is clean: 58,279 rows, 383 subjects, 0 raw-subject conflicts,
0 file conflicts, and 0 exact-trial conflicts.

The invalid current combined `downstream/MCI/` problem is not inherited from the
old `MCI` folder. It is caused by mixing original `MCI/实验组` + `MCI/对照组`
rows with `MCI/匹配后` rows that reuse the same raw subjects/recordings under a
crossed label convention:

| view | rows | raw conflict keys / rows | raw split-overlap keys / rows | subject mismatches vs old anchor | interpretation |
| --- | ---: | ---: | ---: | ---: | --- |
| `MCI` | 91,433 | 218 / 66,308 | 123 / 37,536 | 33,154 | invalid legacy combined view |
| `MCI匹配后` | 33,154 | 0 / 0 | 0 / 0 | 33,154 | internally clean but label direction disagrees with old original-MCI anchor |
| `MCI匹配后_random_seed20260622_label_fixed` | 33,154 | 0 / 0 | 0 / 0 | 0 | current official label-fixed view |

Conclusion: the current official MCI replacement views are clean by the old
baseline. The old invalid `MCI` and old standalone `MCI匹配后` views should remain
excluded from official reports.

### AD

The old `AD` folder is clean: 20,226 rows, 136 subjects, 0 raw-subject conflicts,
0 file conflicts, and 0 exact-trial conflicts.

The current `AD` view still has a new raw identity problem:

- conflicting raw subject example: `GaoLianYing`;
- raw conflict: 1 key / 399 rows;
- file conflict: 4 keys;
- raw split overlap: 54 raw subjects / 16,069 rows;
- subject-label mismatch vs old baseline: 133 rows.

Example pattern: `GaoLianYing` is an old AD patient anchor, but appears in the
current new `AD/匹配后/对照组/...` rows with `health_label=0`. This is not a
conflict that existed in the old AD baseline.

`GaoLianYing` source trace:

| dataset/view | split | source path pattern | health label | rows |
| --- | --- | --- | ---: | ---: |
| old `AD` | old per-subject manifest | `AD/AD组/2021-04-27/GaoLianYing/...` | 1 | 133 |
| source ML-ready `AD/manifest_train.csv` | train | `AD/匹配后/对照组/GaoLianYing/...` | 0 | 133 |
| source ML-ready `AD/manifest_validation.csv` | validation | `AD/AD组/2021-04-27/GaoLianYing/...` | 1 | 133 |
| source ML-ready `AD/manifest_test.csv` | test | `AD/匹配后/实验组/GaoLianYing/...` | 1 | 133 |
| packed fast `downstream/AD/train.csv` | train | `AD/匹配后/对照组/GaoLianYing/...` | 0 | 133 |
| packed fast `downstream/AD/validation.csv` | validation | `AD/AD组/2021-04-27/GaoLianYing/...` | 1 | 133 |
| packed fast `downstream/AD/test.csv` | test | `AD/匹配后/实验组/GaoLianYing/...` | 1 | 133 |

The label-0 rows therefore come from the source ML-ready AD matched-control
block, not from the packed mmap conversion. The same four source recording
filenames also appear under old/current patient paths, so this should be treated
as a source view construction problem.

Raw unpacked source confirmation:

- `/mnt/disk_sde/data-260606/extracted/unpacked/AD/匹配后/实验组/GaoLianYing/`
- `/mnt/disk_sde/data-260606/extracted/unpacked/AD/匹配后/对照组/GaoLianYing/`

Both directories exist. They contain the same seven CSV filenames:

- `20210427_161206_GaoLianYing_F_ZZ_YH_80_N_ProSaccade_D.csv`
- `20210427_162338_GaoLianYing_F_ZZ_YH_80_N_AntiSaccade_D.csv`
- `20210427_163552_GaoLianYing_F_ZZ_YH_80_N_MemorySaccade_D.csv`
- `20210427_164612_GaoLianYing_F_ZZ_YH_80_N_DoubleSaccade_D.csv`
- `20210427_164756_GaoLianYing_F_ZZ_YH_80_N_VisionSaccade_D.csv`
- `20210427_164813_GaoLianYing_F_ZZ_YH_80_N_VisionSaccade_D.csv`
- `20210427_164920_GaoLianYing_F_ZZ_YH_80_N_HorizonSaccade_D.csv`

All seven experiment/control file pairs have identical file sizes and identical
SHA256 hashes. So `GaoLianYing` is duplicated in the raw `AD/匹配后` source
itself, not merely duplicated by the ML-ready or packed-dataset conversion.

Follow-up generated view:

- `downstream/AD_dedup_rawsubject/`
- source rows: 50,957
- removed duplicated `AD/匹配后/实验组` rows: 13,016
- removed conflicting `AD/匹配后/对照组/GaoLianYing` rows: 133
- kept rows: 37,808
- containment check: all 13,016 `匹配后/实验组` rows are contained in
  `AD组/患病` by raw trial key
- final audit: 0 raw-subject/file/trial label conflicts and 0 raw-subject/file/trial
  split overlap

Conclusion: use `AD_dedup_rawsubject` for future AD downstream runs. The
original `AD` view remains as a historical high-risk view and should not be used
for final AD reporting.

### PD

The old `PD相关` folder is clean under binary health labels: 128,711 rows, 746
subjects, 0 raw-subject conflicts, 0 file conflicts, and 0 exact-trial conflicts.

The current aggregate PD views still have raw identity risks:

- conflicting raw subject example: `ZhangMingSha`;
- raw conflict: 1 key / 320 rows;
- file conflict: 4 keys;
- raw split overlap: 321 raw subjects / 100,249 rows;
- subject-label mismatch vs old baseline: 467 rows;
- rows with 5-class subtype mismatch vs old subject subtype anchor: 24,745.

The exact-trial check found no binary label mismatch on rows that aligned back
to old data. The risk is instead at the aggregate/raw-subject level: the same
raw person can appear under different matched/original source branches, and the
current split is clean by `ml_subject_id` but not by raw subject.

The true current 0/1 raw-label conflict is `ZhangMingSha`:

| source branch | source group | label | rows in aggregate PD |
| --- | --- | ---: | ---: |
| `PD相关/帕金森匹配后/对照组/ZhangMingSha/...` | control | 0 | 160 |
| `PD相关/震颤匹配后/实验组/ZhangMingSha/...` | patient | 1 | 160 |

In the raw unpacked source, both directories exist and contain the same eight
CSV filenames. Each experiment/control file pair has identical size and
identical SHA256 hash. This is analogous to the AD `GaoLianYing` problem, but in
PD it affects only this one raw subject in the current aggregate PD view.

There are also three subject-baseline mismatches against old data:

| subject | old baseline label | current PD branch | current label | rows |
| --- | ---: | --- | ---: | ---: |
| `ZhangMingSha` | 0 | `震颤匹配后/实验组` | 1 | 160 |
| `LiuWenFang` | 0 | `震颤匹配后/实验组` | 1 | 159 |
| `ZhangMingLin` | 0 | `震颤匹配后/实验组` | 1 | 148 |

Raw source folders for the two old-control mismatches:

| subject | raw PD patient-side folder | raw global-control folder | identical file pairs |
| --- | --- | --- | ---: |
| `LiuWenFang` | `unpacked/PD相关/震颤匹配后/实验组/LiuWenFang/` | `unpacked/对照组数据汇总/LiuWenFang/` | 10 / 10 |
| `ZhangMingLin` | `unpacked/PD相关/震颤匹配后/实验组/ZhangMingLin/` | `unpacked/对照组数据汇总/ZhangMingLin/` | 11 / 11 |

The listed pairs have identical filenames, file sizes, and SHA256 hashes. This
means these two raw subjects are duplicated between the raw PD tremor matched
patient branch and the global control pool.

The large split-overlap problem is different from the one-subject 0/1 conflict:
many control subjects are reused across multiple PD matched-control branches.
For example, one raw control subject can appear as `帕金森匹配后/对照组` in one
split and `震颤匹配后/对照组` or `运动障碍匹配后/对照组` in another split. Because
the split key includes source fields in `ml_subject_id`, those repeated raw
subjects are not caught by the nominal subject-heldout check.

Conclusion: the PD issue is not explained by label contradictions in old
`PD相关`. Rebuild the aggregate PD 5-class and PD binary views with raw-subject
identity as the split/removal key, or report the clean subtype-specific matched
binary views separately.

### Epilepsy, Detox, Migraine

These current official views are clean against the old baseline:

| view | rows | direct found | direct label mismatches | subject label mismatches | raw conflict | raw split overlap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `癫痫` | 139,908 | 99.22% | 0 | 0 | 0 | 0 |
| `戒毒所` | 26,338 | 100.00% | 0 | 0 | 0 | 0 |
| `偏头痛` | 32,130 | 100.00% | 0 | 0 | 0 | 0 |

`戒毒所_random_seed20260621` is also clean by this audit, but it remains an
excluded sensitivity run rather than the official result view.

## Other Problems Found

1. `source_file_uid` is not stable across old and new datasets. Any future
   cross-dataset audit should use `relative_source_path`, `original_trial_index`,
   and `direction`, plus raw `subject`, not UID alone.
2. `ml_subject_id` is useful for loader bookkeeping but should not be the only
   train/validation/test split identity key. It encodes source/group fields and
   can hide the same raw person appearing in multiple views or folders.
3. Direct exact-trial coverage is below 100% for PD, AD, epilepsy, and matched
   MCI. This is expected when the new dataset contains rows not present under the
   same old relative path, but those rows still need raw-subject baseline checks.
4. The current official dataset is usable for `癫痫`, `戒毒所`, `偏头痛`, and the
   two fixed MCI tasks. It is not yet fully clean for aggregate PD and AD.

## Required Follow-Up

Before using all downstream numbers as final:

1. Rebuild `AD` using raw subject/file conflict removal and raw-subject split.
2. Rebuild aggregate `PD相关_random_seed20260620` and
   `PD相关_binary_random_seed20260620` using raw-subject identity, or explicitly
   exclude/report them as unresolved.
3. Keep legacy `MCI`, old `MCI匹配后`, and superseded MCI views out of official
   summaries.
4. Add raw-subject/raw-file conflict and split-overlap audits to the dataset
   generation pipeline, in addition to existing `ml_subject_id` checks.
