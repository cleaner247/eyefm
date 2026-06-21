# Newdata v3 MCI Label Audit

Audit time: 2026-06-21 03:58 CST

Reason:

- `mci_binary` downstream results were close to random despite other tasks
  behaving more plausibly.
- User asked whether the MCI labels may be wrong.

## Checked Artifacts

Configs:

- `configs/downstream/mci_binary_{scratch,linear_probe,partial,full}.yaml`
- `configs/downstream/mci_matched_binary_{scratch,linear_probe,partial,full}.yaml`

Indexes:

- `downstream/MCI/{train,validation,test}.csv`
- `downstream/MCI匹配后/{train,validation,test}.csv`

Model outputs checked for label-flip symptoms:

- `outputs/downstream_v3_fast/mci_binary/scratch_full/metrics_final.json`
- `outputs/downstream_v3_fast/mci_binary/pretrained_linear_probe/metrics_final.json`
- `outputs/downstream_v3_fast/mci_binary/pretrained_partial/metrics_final.json`
- `outputs/downstream_v3_fast/mci_matched_binary/scratch_full/metrics_final.json`

## Config Label Mapping

Both MCI tasks use the intended binary label config:

| task | view | label column | negative | positive |
| --- | --- | --- | ---: | ---: |
| `mci_binary` | `MCI` | `health_label` | 0 | 1 |
| `mci_matched_binary` | `MCI匹配后` | `health_label` | 0 | 1 |

The config itself is not inverted.

## Critical Finding: `MCI` View Is Invalid

The `MCI` view has widespread conflicts when identity is checked by raw
`subject`, filename basename, or `(subject, basename)`.

| check key | conflict keys | conflict rows | percent of rows |
| --- | ---: | ---: | ---: |
| raw `subject` | 218 | 66,308 / 91,433 | 72.52% |
| filename basename | 878 | 66,308 / 91,433 | 72.52% |
| `(subject, basename)` | 878 | 66,308 / 91,433 | 72.52% |

Rows affected by split and label:

| split | label 0 rows | label 1 rows |
| --- | ---: | ---: |
| train | 21,385 | 20,886 |
| validation | 5,729 | 5,799 |
| test | 6,040 | 6,469 |

There is also raw-subject leakage across train/validation/test:

| check | count |
| --- | ---: |
| raw subjects appearing in multiple splits | 123 |
| rows involving these overlapped raw subjects | 37,536 / 91,433 = 41.05% |

`ml_subject_id` does not reveal this because it includes dataset/group fields.
For example, the same raw subject can appear as:

- `MCI|实验组|实验组|MCI|CBYOS1225` with `health_label=1`
- `MCI|匹配后|对照组|MCI|CBYOS1225` with `health_label=0`

Some entries also share the same recording filename basename while receiving
opposite labels, e.g. the same `CBYOS1225` recording basename appears under
both `MCI/实验组/...` and `MCI/匹配后/对照组/...`.

Conclusion: current `downstream/MCI/` mixes original and matched folders in a
way that duplicates many raw subjects/recordings with opposite labels. Its
metrics should not be interpreted as a valid MCI disease result.

## Matched Containment Check

The standalone `MCI匹配后` view is exactly the `source_dataset=匹配后` block inside
the combined `MCI` source view when split assignment is ignored:

| comparison | result |
| --- | ---: |
| `MCI` matched block rows | 33,154 |
| standalone `MCI匹配后` rows | 33,154 |
| raw trial keys only in `MCI` matched block | 0 |
| raw trial keys only in standalone `MCI匹配后` | 0 |

However, the train/validation/test assignment differs between the two views, so
the row identities match but the split labels are not identical.

The matched block is also fully contained in the original `MCI` `实验组` +
`对照组` raw-trial union. The containment is crossed:

| matched block | raw trial containment in original folders |
| --- | --- |
| `匹配后/实验组`, `health_label=1`, 16,720 rows / 109 subjects | 16,720 / 16,720 raw trials are in original `对照组`; 0 are in original `实验组` |
| `匹配后/对照组`, `health_label=0`, 16,434 rows / 109 subjects | 16,434 / 16,434 raw trials are in original `实验组`; 0 are in original `对照组` |

This explains the large conflict: the combined `MCI` view contains the original
folders and the matched folders together, but the matched folders reuse the same
raw recordings with opposite disease labels relative to the corresponding
original folders.

The issue is already present in the source ML-ready view:

- source root:
  `/mnt/disk_sde/data-260606/extracted/cd_speed4_hard_blink_ml_ready_subjectkey_20260619`
- source `MCI` manifest rows: 91,433
- source raw-subject label conflicts: 218
- source `MCI匹配后` raw-subject label conflicts: 0

So this is not introduced by the packed mmap conversion or by the fine-tune
loader. The older source audit used subject key
`source_top|source_dataset|source_group|source_subtype|subject`; that key hides
raw-subject conflicts because the same person/file receives different keys when
it appears under `实验组`, `对照组`, or `匹配后`.

## `MCI匹配后` View Has No Internal Overlap But Uses Crossed Labels

The matched-row view did not show internal duplicate-label conflicts:

| check key | conflict keys | conflict rows | percent of rows |
| --- | ---: | ---: | ---: |
| raw `subject` | 0 | 0 / 33,154 | 0.00% |
| filename basename | 0 | 0 / 33,154 | 0.00% |
| `(subject, basename)` | 0 | 0 / 33,154 | 0.00% |

It also has no raw-subject overlap across train/validation/test.

However, this is not enough to treat its labels as correct. The containment
check above shows that every matched raw trial maps back to the original MCI
folders with the opposite label convention:

| matched source label status | rows |
| --- | ---: |
| rows kept after requiring a matching original MCI raw subject | 33,154 |
| rows dropped because no original MCI subject label was available | 0 |
| rows whose `health_label` changes when overwritten by original MCI subject label | 33,154 |

Therefore the old `mci_matched_binary` results are also provisional. The
matched-row follow-up task must ignore `MCI匹配后` source labels, keep rows only
when their raw `subject` exists in the original MCI anchor map, and use the
original MCI subject label as the only label source.

## Label-Flip Check

The completed MCI metrics do not look like a simple global positive/negative
flip. Inverting prediction scores would not consistently improve the test
AUROC:

| output | val subject AUROC | inverted val | test subject AUROC | inverted test |
| --- | ---: | ---: | ---: | ---: |
| `mci_binary/scratch` | 0.47619 | 0.52381 | 0.56292 | 0.43708 |
| `mci_binary/linear_probe` | 0.41975 | 0.58025 | 0.54907 | 0.45093 |
| `mci_binary/partial` | 0.38889 | 0.61111 | 0.55501 | 0.44499 |
| `mci_matched_binary/scratch` | 0.52595 | 0.47405 | 0.55992 | 0.44008 |

## Recommended Action

1. Treat current `mci_binary` results as invalid/provisional.
2. Treat current `mci_matched_binary` results as suspect/provisional because
   the matched-row source labels are crossed relative to original MCI labels.
3. Rebuild `downstream/MCI/` before using it:
   - split by raw `subject`, not by `ml_subject_id` that encodes group/source;
   - do not mix original and matched copies of the same raw subject/file in one
     binary view;
   - enforce no `(subject, filename basename)` can have both labels;
   - audit raw-subject split overlap before training.
4. Rerun MCI matched-row follow-up using labels anchored to the original MCI
   subject label, not the `MCI匹配后` source label; drop rows without an original
   MCI raw-subject anchor.
5. After rebuilding, rerun both MCI four-mode matrices. Until then, do not
   compare old MCI metrics against other disease tasks.

## Follow-Up Generated Views

Execution rule clarified after the MCI audit:

- `MCI匹配后` rows are never allowed to provide labels.
- For a matched row, keep it only if its raw `subject` exists in the original
  `MCI` label anchor built after excluding `source_dataset=匹配后`.
- Assign the matched row the original `MCI` subject label. If no original-MCI
  subject anchor exists, drop the row.
- In the generated seed-20260621 view, all 33,154 matched rows had an original
  MCI subject anchor, so `dropped_unmapped_rows=0`; all 33,154 source labels
  were overwritten.

| item | `mci_original_only_binary` | `mci_matched_binary_random_seed20260621` |
| --- | --- | --- |
| generated view | `downstream/MCI_original_only_no_matched/` | `downstream/MCI匹配后_random_seed20260621/` |
| label source | original `MCI` rows only | original `MCI` subject label overwrites matched source label |
| split policy | original split preserved after removing matched rows | subject-level stratified random split, seed `20260621` |
| rows | 58,279 | 33,154 |
| subjects | 383 | 218 |
| removed/unmapped rows | removed 33,154 matched rows | dropped 0 unmatched rows |
| overwritten labels | n/a | 33,154 |
| raw label conflicts | 0 | 0 |
| raw split overlap | 0 | 0 |
| config queue | `configs/downstream/queue_mci_followup_seed20260621.txt` | `configs/downstream/queue_mci_followup_seed20260621.txt` |

## Clarification: Label Authority for Matched MCI Rows

Confirmed at 2026-06-21 06:31 CST:

- The old combined `MCI` fine-tune view was invalid because it contained both
  original `MCI` rows and `source_dataset=匹配后` copies; many of those matched
  rows reused the same raw subjects/recordings with crossed labels.
- The corrected `mci_original_only_binary` task uses only original `MCI` rows
  after removing `source_dataset=匹配后`.
- The corrected `mci_matched_binary_random_seed20260621` task may use matched
  rows as input samples, but it must not use matched-row labels. A row is kept
  only when its raw `subject` exists in the original `MCI` subject-label anchor
  built after removing `source_dataset=匹配后`; otherwise the row is dropped.
- For every kept matched row, `health_label` is overwritten by the original
  `MCI` raw-subject label. In the generated seed-20260621 view this kept all
  33,154 rows, dropped 0 rows, and overwrote all 33,154 labels.
- After the stricter clarification at 2026-06-21 07:24 CST, the regenerated
  matched view also overwrites `source_group` to match the original `MCI`
  subject label. This prevents the CSV metadata from preserving the crossed
  matched-label text even though training uses `health_label`.
- Therefore official follow-up MCI reporting must use
  `mci_original_only_binary` and `mci_matched_binary_random_seed20260621`, not
  the earlier `mci_binary` / `mci_matched_binary` outputs.

Latest verification after regeneration:

| check | value |
| --- | --- |
| rows | 33,154 |
| raw subjects | 218 |
| rows outside original MCI anchor | 0 |
| `health_label` mismatches vs original MCI anchor | 0 |
| `source_group` mismatches vs original MCI anchor | 0 |

## Label-Fixed Matched-MCI Rerun

Updated at 2026-06-21 12:14 CST:

The `MCI匹配后` healthy/disease direction was confirmed to be reversed relative
to the previous seed-20260621 original-anchor relabeling. Therefore
`mci_matched_binary_random_seed20260621` should be treated as historical and
superseded for matched-MCI interpretation.

The new rerun is:

| item | value |
| --- | --- |
| task | `mci_matched_binary_random_seed20260622_label_fixed` |
| generated view | `downstream/MCI匹配后_random_seed20260622_label_fixed/` |
| split policy | fresh subject-level stratified random split, `seed=20260622` |
| label rule | keep rows only when raw `subject` exists in original MCI; final `health_label` is the inverse of the original-MCI anchor label |
| rows | 33,154 |
| raw subjects | 218 |
| dropped unmatched rows | 0 |
| subject overlap | 0 across train/validation/test |
| label conflicts | 0 raw-subject, raw-file, and raw-trial conflicts |
| subject split | train 140 (70/70), validation 34 (17/17), test 44 (22/22) |

Final label-fixed fine-tune results:

| mode | best epoch | val subject AUROC | test subject AUROC | test balanced accuracy | test weighted F1 | test Cohen's kappa |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| scratch | 4 | `0.86851` | `0.60537` | `0.61364` | `0.61183` | `0.22727` |
| linear_probe | 3 | `0.67820` | `0.53926` | `0.50000` | `0.44368` | `0.00000` |
| partial | 7 | `0.79585` | `0.65289` | `0.56818` | `0.56617` | `0.13636` |
| full | 3 | `0.76471` | `0.63017` | `0.61364` | `0.61344` | `0.22727` |

This rerun supersedes `mci_matched_binary_random_seed20260621` for matched-MCI
interpretation because it uses the confirmed label-fixed direction and a new
subject-level random train/validation/test split.
