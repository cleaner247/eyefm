# EyeMAE Downstream Epoch Dynamics

This note records the epoch-level training dynamics for the completed V3
extra-disease 5-fold downstream run.

Source run:

```text
outputs/downstream_disease_binary_kfold_extra_seed42/
```

Source logs:

```text
outputs/downstream_disease_binary_kfold_extra_seed42/logs/fold_0.log
outputs/downstream_disease_binary_kfold_extra_seed42/logs/fold_1.log
outputs/downstream_disease_binary_kfold_extra_seed42/logs/fold_2.log
outputs/downstream_disease_binary_kfold_extra_seed42/logs/fold_3.log
outputs/downstream_disease_binary_kfold_extra_seed42/logs/fold_4.log
```

Generated local CSV artifacts:

```text
outputs/downstream_disease_binary_kfold_extra_seed42/epoch_curves.csv
outputs/downstream_disease_binary_kfold_extra_seed42/epoch_dynamics_by_run.csv
outputs/downstream_disease_binary_kfold_extra_seed42/epoch_dynamics_by_task_mode.csv
```

CSV contents:

```text
epoch_curves.csv:
  one row per validation epoch
  columns = fold, disease, mode, epoch, train_loss, val_subject_auroc, val_subject_f1
  rows = 1467

epoch_dynamics_by_run.csv:
  one row per fold/disease/mode run
  rows = 80

epoch_dynamics_by_task_mode.csv:
  one row per disease/mode averaged across 5 folds
  rows = 16
```

## Training Rule

The downstream config uses:

```text
max_epochs = 100
early_stopping_patience = 10
monitor = val/subject/auroc
```

Epoch is zero-indexed. Therefore:

```text
best_epoch = 0
stop_epoch = 10
actual epochs trained = 11
```

This is why the actual epoch range starts around 11 even when the best
checkpoint is from epoch 0 or epoch 1.

## Overall Dynamics

Across all 80 V3 runs:

```text
actual epochs trained: min = 11, max = 53, mean = 18.34
best_epoch:            min = 0,  max = 42, mean = 7.34
```

No run reached `max_epochs=100`; every run stopped through early stopping.

## Mean Dynamics By Disease And Mode

Values below are averaged across the 5 folds for each disease/mode.

`first` is epoch 0. `best` is the epoch with highest validation subject AUROC.
`final` is the last epoch before early stopping. Test metrics are evaluated
after reloading `checkpoint_best.pt`, not `checkpoint_last.pt`.

| Disease | Mode | Epochs Trained | Best Epoch | Train Loss first->best->final | Val AUROC first->best->final | Val F1 first->best->final | Test AUROC | Test F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| AD | scratch | 14.0 | 3.0 | 0.3050->0.2476->0.1063 | 0.6544->0.8299->0.7531 | 0.6316->0.8562->0.8597 | 0.7574 | 0.8005 |
| AD | pretrained_linear_probe | 16.4 | 5.4 | 0.2518->0.2053->0.1364 | 0.8193->0.8417->0.7712 | 0.8540->0.8557->0.8587 | 0.8247 | 0.8747 |
| AD | pretrained_partial | 19.2 | 8.2 | 0.2265->0.0994->0.0227 | 0.8696->0.9193->0.8664 | 0.8847->0.8961->0.9138 | 0.8619 | 0.8839 |
| AD | pretrained_full | 13.4 | 2.4 | 0.1996->0.1316->0.0098 | 0.8912->0.9147->0.8481 | 0.8890->0.9001->0.8999 | 0.8531 | 0.8941 |
| MCI | scratch | 20.4 | 9.4 | 0.8284->0.6695->0.5373 | 0.6511->0.7623->0.7164 | 0.4480->0.6287->0.5867 | 0.7200 | 0.5926 |
| MCI | pretrained_linear_probe | 23.4 | 12.4 | 0.7903->0.6924->0.6518 | 0.6979->0.7659->0.7297 | 0.5123->0.6297->0.6091 | 0.7225 | 0.5687 |
| MCI | pretrained_partial | 12.4 | 1.4 | 0.7610->0.6803->0.3391 | 0.7575->0.7865->0.7106 | 0.5830->0.5799->0.5887 | 0.7526 | 0.5590 |
| MCI | pretrained_full | 11.2 | 0.2 | 0.7374->0.7155->0.1560 | 0.7929->0.8047->0.7207 | 0.6573->0.6449->0.5609 | 0.7898 | 0.6357 |
| 偏头痛 | scratch | 31.2 | 20.2 | 0.7685->0.3657->0.2467 | 0.5257->0.7946->0.7574 | 0.4238->0.6139->0.6113 | 0.7397 | 0.5914 |
| 偏头痛 | pretrained_linear_probe | 24.4 | 13.4 | 0.7161->0.4939->0.4393 | 0.6276->0.7954->0.7418 | 0.4280->0.7325->0.6875 | 0.7725 | 0.6991 |
| 偏头痛 | pretrained_partial | 15.8 | 4.8 | 0.6578->0.4213->0.1723 | 0.7683->0.8482->0.7985 | 0.6321->0.7018->0.6692 | 0.7522 | 0.5526 |
| 偏头痛 | pretrained_full | 17.6 | 6.6 | 0.5987->0.2087->0.0426 | 0.7910->0.8769->0.8435 | 0.7230->0.7862->0.7610 | 0.7969 | 0.7371 |
| 癫痫 | scratch | 20.0 | 9.0 | 0.5864->0.4347->0.3225 | 0.7902->0.8192->0.7900 | 0.6825->0.7588->0.7497 | 0.7947 | 0.7343 |
| 癫痫 | pretrained_linear_probe | 25.2 | 14.2 | 0.5264->0.4522->0.4384 | 0.8217->0.8463->0.8309 | 0.7155->0.7696->0.7397 | 0.8374 | 0.7624 |
| 癫痫 | pretrained_partial | 15.0 | 4.0 | 0.4873->0.3753->0.2041 | 0.8596->0.8679->0.8407 | 0.7214->0.7752->0.7978 | 0.8560 | 0.7692 |
| 癫痫 | pretrained_full | 13.8 | 2.8 | 0.4606->0.3187->0.0752 | 0.8653->0.8720->0.8418 | 0.7464->0.7959->0.8051 | 0.8576 | 0.7967 |

## Concrete Early-Best Examples

These runs have `best_epoch` equal to 0 or 1 but still trained about 11-12
epochs because patience remained active.

| Run | Actual Epochs | Best Epoch | Train Loss first->final | Val AUROC first->best->final | Test AUROC |
|---|---:|---:|---:|---:|---:|
| fold_0 AD pretrained_linear_probe | 11 | 0 | 0.2445->0.1501 | 0.8095->0.8095->0.7778 | 0.9252 |
| fold_0 AD scratch | 12 | 1 | 0.3009->0.1019 | 0.7222->0.8571->0.6825 | 0.8503 |
| fold_0 MCI pretrained_full | 11 | 0 | 0.7579->0.1847 | 0.8401->0.8401->0.7314 | 0.7826 |
| fold_0 MCI pretrained_partial | 11 | 0 | 0.7795->0.4062 | 0.8170->0.8170->0.7090 | 0.7377 |
| fold_1 AD pretrained_full | 11 | 0 | 0.1954->0.0095 | 0.9444->0.9444->0.8730 | 0.8889 |
| fold_1 AD pretrained_partial | 11 | 0 | 0.2211->0.0352 | 0.9286->0.9286->0.8492 | 0.8810 |
| fold_2 癫痫 pretrained_full | 12 | 1 | 0.4625->0.0983 | 0.8823->0.8934->0.8757 | 0.8454 |
| fold_3 癫痫 pretrained_partial | 11 | 0 | 0.4755->0.2204 | 0.8202->0.8202->0.7821 | 0.8771 |
| fold_4 癫痫 pretrained_full | 12 | 1 | 0.5956->0.0805 | 0.8536->0.8788->0.8357 | 0.8316 |

## Interpretation

The dominant pattern is:

```text
train_loss usually keeps decreasing
validation AUROC rises early, then plateaus or drops
early stopping selects the best validation AUROC checkpoint
final test metrics are computed after reloading checkpoint_best.pt
```

So decreasing train loss alone is not evidence that more epochs would help.
For many pretrained full/partial runs, the loss continues to drop sharply after
the best validation AUROC has already occurred, which is a typical overfitting
pattern on small downstream datasets.

The current evidence does not indicate that the 100-epoch cap is too small:
all 80 runs stopped by early stopping before reaching the cap. If we want a
more conservative check, the next controlled experiment should change only
`early_stopping_patience` from 10 to 20 while keeping the same splits and seeds.
