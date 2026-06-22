# EyeMAE Fast Dataset v2 Current

This is the canonical pointer for the currently accepted v2 fast dataset.

Dataset path:

```text
/mnt/disk_sde/data-260606/extracted/eyemae_fast_dataset_v2
```

Primary dataset documentation:

```text
/mnt/disk_sde/data-260606/extracted/eyemae_fast_dataset_v2/README.md
```

Machine-readable summary:

```text
/mnt/disk_sde/data-260606/extracted/eyemae_fast_dataset_v2/v2_build_summary.json
```

Use the materialized CSV indexes and packed mmap shards under this directory as
the source of truth for current pretraining and downstream fine-tuning. Older
audit notes in `eyemae/docs/` are historical records used to explain the data
decisions; they are not the canonical dataset specification.

Current `ml_subject_id` rule:

```text
name|age:<age>|edu:<education_code>
```

The education code is parsed from the filename field two positions before age.
`FangDeXiu|age:77`, `LuXingQiong|age:61`, and `雷妮莎|age:23` are intentionally
kept merged as `edu:MIXED`.

Current task roots:

```text
pretrain:
/mnt/disk_sde/data-260606/extracted/eyemae_fast_dataset_v2/pretrain

pretrain index files:
/mnt/disk_sde/data-260606/extracted/eyemae_fast_dataset_v2/pretrain/pretrain/pretrain_train.csv
/mnt/disk_sde/data-260606/extracted/eyemae_fast_dataset_v2/pretrain/pretrain/pretrain_validation.csv
/mnt/disk_sde/data-260606/extracted/eyemae_fast_dataset_v2/pretrain/pretrain/pretrain_test.csv

downstream:
/mnt/disk_sde/data-260606/extracted/eyemae_fast_dataset_v2/finetune/ad_binary
/mnt/disk_sde/data-260606/extracted/eyemae_fast_dataset_v2/finetune/detox_binary
/mnt/disk_sde/data-260606/extracted/eyemae_fast_dataset_v2/finetune/epilepsy_binary
/mnt/disk_sde/data-260606/extracted/eyemae_fast_dataset_v2/finetune/mci_binary
/mnt/disk_sde/data-260606/extracted/eyemae_fast_dataset_v2/finetune/mci_matched_binary
/mnt/disk_sde/data-260606/extracted/eyemae_fast_dataset_v2/finetune/migraine_binary
/mnt/disk_sde/data-260606/extracted/eyemae_fast_dataset_v2/finetune/pd_binary
/mnt/disk_sde/data-260606/extracted/eyemae_fast_dataset_v2/finetune/pd_related_5class
```
