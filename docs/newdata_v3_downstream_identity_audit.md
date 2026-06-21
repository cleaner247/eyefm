# Newdata v3 Downstream Identity/Label Audit

Audit time: 2026-06-21 04:08 CST

Reason:

- `mci_binary` results were implausibly close to random.
- The first MCI audit found severe raw-subject/file-level label conflicts.
- This follow-up applies the same audit to all downstream views.

## Why The Existing Plan Missed This

`docs/eyemae_fast_packed_dataset_adaptation_plan.md` intentionally made phase 1
copy the current 20260619 downstream views:

> 保留当前 view 的 health_label/pd_disease_label。

The plan also treated labels as view-specific and required downstream split
overlap checks by `ml_subject_id`. In the current dataset,
`ml_subject_id = source_top|source_dataset|source_group|source_subtype|subject`.

That key hides conflicts where the same raw person or the same recording file
appears under different source/group folders. For example, the same raw subject
can become distinct identities when it appears in `实验组`, `对照组`, or `匹配后`.

The missing audit keys were:

- raw `subject`
- recording identity, approximated as `(subject, filename basename)`
- raw-subject/raw-file split overlap within each downstream view
- raw-subject/raw-file label conflicts within each downstream view

## All View Summary

Legend:

- `raw conflicts`: raw subjects with both `health_label=0` and `1`.
- `file conflicts`: `(subject, filename basename)` keys with both labels.
- `raw split overlap`: raw subjects appearing in multiple train/validation/test
  splits within the same view.
- `file split overlap`: `(subject, filename basename)` keys appearing in
  multiple splits.
- `ml conflicts/overlap`: the same checks using `ml_subject_id`; these remain
  zero because the current key encodes source/group.

| view | rows | ml subjects | raw subjects | labels | raw conflicts keys/rows | raw split overlap keys/rows | file conflicts keys/rows | file split overlap keys/rows | ml conflicts/overlap |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| `AD` | 50,957 | 341 | 252 | 0=22,587, 1=28,370 | 1 / 399 | 54 / 16,069 | 4 / 399 | 230 / 16,069 | 0 / 0 |
| `AD匹配后` | 35,603 | 236 | 235 | 0=22,587, 1=13,016 | 1 / 266 | 1 / 266 | 4 / 266 | 4 / 266 | 0 / 0 |
| `MCI` | 91,433 | 601 | 383 | 0=51,796, 1=39,637 | 218 / 66,308 | 123 / 37,536 | 878 / 66,308 | 496 / 37,536 | 0 / 0 |
| `MCI匹配后` | 33,154 | 218 | 218 | 0=16,434, 1=16,720 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| `PD相关` | 220,427 | 1,552 | 896 | 0=114,085, 1=106,342 | 1 / 320 | 311 / 98,896 | 4 / 320 | 1,200 / 98,671 | 0 / 0 |
| `PD相关_random_seed20260620` | 220,427 | 1,552 | 896 | 0=114,085, 1=106,342 | 1 / 320 | 321 / 100,249 | 4 / 320 | 1,217 / 100,064 | 0 / 0 |
| `PD相关_binary_random_seed20260620` | 220,427 | 1,552 | 896 | 0=114,085, 1=106,342 | 1 / 320 | 321 / 100,249 | 4 / 320 | 1,217 / 100,064 | 0 / 0 |
| `PD相关_帕金森病匹配后` | 64,585 | 456 | 456 | 0=45,940, 1=18,645 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| `PD相关_特发性震颤匹配后` | 39,201 | 261 | 261 | 0=26,552, 1=12,649 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| `PD相关_运动障碍匹配后` | 24,601 | 174 | 174 | 0=17,294, 1=7,307 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| `PD相关_震颤匹配后` | 36,655 | 245 | 245 | 0=24,299, 1=12,356 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| `偏头痛` | 32,130 | 208 | 208 | 0=19,417, 1=12,713 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| `戒毒所` | 26,338 | 170 | 170 | 0=15,749, 1=10,589 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| `戒毒所_random_seed20260621` | 26,338 | 170 | 170 | 0=15,749, 1=10,589 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| `癫痫` | 139,908 | 889 | 889 | 0=70,751, 1=69,157 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |

## Interpretation

### Invalid Or High Risk

`MCI` is invalid as a disease binary task:

- 72.52% of rows are in raw-subject/file label conflicts.
- The same raw person/file can be both `health_label=0` and `1`.
- This explains why `mci_binary` results are near random and unstable.

`PD相关_random_seed20260620` and `PD相关_binary_random_seed20260620` are high
risk:

- Label conflict is small: one raw subject, four files, 320 rows.
- Raw-subject split overlap is large: 321 raw subjects, about 100k rows.
- This is likely shared controls and/or shared raw people across PD subtype
  matched sources.
- Current PD-related 5-class and PD binary results can be inflated or otherwise
  biased by raw identity leakage.

`AD` is also high risk, but much smaller than MCI/PD:

- One raw subject (`GaoLianYing`) has both labels across four files.
- The overall `AD` view also has raw-subject split overlap due to original vs
  matched copies.
- `AD匹配后` still has the `GaoLianYing` conflict, so it is not fully clean.

### Clean By This Audit

These views showed zero raw-subject conflicts, zero file conflicts, and zero
raw-subject/file split overlap:

- `MCI匹配后`
- `PD相关_帕金森病匹配后`
- `PD相关_特发性震颤匹配后`
- `PD相关_运动障碍匹配后`
- `PD相关_震颤匹配后`
- `偏头痛`
- `戒毒所`
- `戒毒所_random_seed20260621`
- `癫痫`

## Recommended Fix

For any task used as a formal downstream benchmark, rebuild its split using a
raw identity key, not only `ml_subject_id`.

Minimum required rules:

1. Define raw subject identity as at least `(source_top, subject)` for non-PD
   tasks and carefully review PD because controls are reused across subtypes.
2. Define raw recording identity as `(source_top, subject, filename basename)`
   or a stable source file UID plus task/trial/direction if reliable.
3. Within a downstream view, fail the dataset if any raw subject or raw
   recording has both `health_label=0` and `health_label=1`.
4. Within train/validation/test, split by raw subject identity so no raw
   subject or raw recording appears in more than one split.
5. For PD-related aggregate tasks, either:
   - build the aggregate task from clean per-subtype matched views and split
     after merging by raw subject; or
   - train/report the four clean subtype matched binary tasks separately.

Until rebuilt, treat `mci_binary`, PD aggregate random tasks, and AD aggregate
results as provisional.
