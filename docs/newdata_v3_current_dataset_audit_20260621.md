# Newdata v3 Current Dataset Audit

Audit time: 2026-06-21

Scope:

- Current official/reportable downstream summary rows:
  `outputs/downstream_v3_fast/all_downstream_best_epoch1_within30_summary_20260621/summary_all_completed_best_epoch1_within30.json`
- Fast packed dataset root:
  `/mnt/disk_sde/data-260606/extracted/eyemae_fast_dataset_v1/downstream/`
- Checks were run directly on each view's `train.csv`, `validation.csv`, and
  `test.csv`.

Audit checks:

- `ml_subject_id` label conflict and split overlap.
- Raw `subject` label conflict and split overlap.
- Raw file label conflict and split overlap, using `(subject, basename(relative_source_path))`.
- For label-fixed matched MCI, verify every row has an original-MCI anchor and
  `health_label == 1 - original_anchor_label`.

## Current Official Views

| view | rows | raw subjects | raw label conflict keys / rows | raw split overlap keys / rows | file label conflict keys / rows | file split overlap keys / rows | status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `PD相关_random_seed20260620` | 220,427 | 896 | 1 / 320 | 321 / 100,249 | 4 / 320 | 1,217 / 100,064 | unresolved high risk |
| `PD相关_binary_random_seed20260620` | 220,427 | 896 | 1 / 320 | 321 / 100,249 | 4 / 320 | 1,217 / 100,064 | unresolved high risk |
| `癫痫` | 139,908 | 889 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | clean |
| `戒毒所` | 26,338 | 170 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | clean |
| `偏头痛` | 32,130 | 208 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | clean |
| `AD` | 50,957 | 252 | 1 / 399 | 54 / 16,069 | 4 / 399 | 230 / 16,069 | unresolved high risk |
| `MCI_original_only_no_matched` | 58,279 | 383 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | clean |
| `MCI匹配后_random_seed20260622_label_fixed` | 33,154 | 218 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | clean |

All eight official views have zero `ml_subject_id` label conflicts and zero
`ml_subject_id` split overlap. This is not sufficient for identity safety
because `ml_subject_id` includes source/group fields.

## Previous Issue Status

| issue | current status |
| --- | --- |
| Combined `MCI` view mixes original and matched rows with crossed labels. | Not cleaned in-place. The bad `downstream/MCI/` view still exists and remains invalid, but it is excluded from the current official report. |
| Old `MCI匹配后` label direction was wrong for the intended matched-MCI task. | Handled by generating `downstream/MCI匹配后_random_seed20260622_label_fixed/` and using that in the current report. |
| MCI matched rows must be anchored to original MCI subjects. | Handled. Current label-fixed matched-MCI view has 33,154 rows, 0 missing original anchors, and 0 rows whose label violates the required inverse-anchor rule. |
| `detox_binary_random_seed20260621` is a random-resplit sensitivity run. | Clean as a dataset view, but excluded from the current official report by request. |
| `AD` had a small raw label conflict and raw identity leakage. | Not handled. Current official `AD` view still has one conflicting raw subject (`GaoLianYing`) and raw split overlap. |
| PD aggregate tasks had small raw label conflict but large raw identity leakage. | Not handled. Current official PD 5-class and PD binary views still have the same raw conflict and large raw split overlap. |

## Legacy / Excluded Views

| view | audit result |
| --- | --- |
| `MCI` | Still invalid: 218 raw-subject conflict keys / 66,308 rows; 123 raw-subject split-overlap keys / 37,536 rows. |
| `MCI匹配后` | Internally has 0 raw conflicts and 0 raw split overlap, but source label direction is not the intended final matched-MCI label authority. |
| `MCI匹配后_random_seed20260621` | Internally clean, but superseded by `MCI匹配后_random_seed20260622_label_fixed` because the label direction was corrected. |
| `AD匹配后` | Not clean: one raw-subject label conflict (`GaoLianYing`) affecting 266 rows, and corresponding split overlap. |
| `戒毒所_random_seed20260621` | Clean, but excluded from the official report as a sensitivity run. |

## Interpretation

The previous MCI-specific problem has been handled for the current official
report by replacing the invalid views with `MCI_original_only_no_matched` and
`MCI匹配后_random_seed20260622_label_fixed`.

The dataset is not fully cleaned overall. The current official report still
contains `AD`, `PD相关_random_seed20260620`, and
`PD相关_binary_random_seed20260620`, and these views still have raw
subject/file-level problems that `ml_subject_id` split checks do not catch.

Recommended next step before treating all downstream numbers as final:

1. Rebuild AD with raw-subject/raw-file conflict removal and raw-subject split.
2. Rebuild PD aggregate 5-class and PD binary by splitting on raw identity after
   merging, or report the four clean PD subtype matched binary views separately.
3. Keep old `MCI`, old `MCI匹配后`, and `MCI匹配后_random_seed20260621` out of
   official summaries.
4. Keep `detox_binary_random_seed20260621` out of official summaries unless it
   is explicitly presented as a sensitivity run.
