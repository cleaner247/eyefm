# Newdata V3 Training Log

This document records necessary training status for the v3 new-data pipeline.

## 2026-06-19 Pretrain V3

Plan:

- `docs/pretrain_v3_plan.md`
- first-version EyeBERT-style masked reconstruction
- dataset: `/mnt/disk_sde/data-260606/extracted/eyemae_fast_dataset_v1`
- GPUs: physical GPU1,2,3,4 through `CUDA_VISIBLE_DEVICES=1,2,3,4`

Pre-launch checks:

- Full test suite passed: `35 passed in 11.72s`.
- Packed pretrain audit passed:
  - train: 730,695 trials
  - validation: 41,093 trials
  - test: 40,853 trials
  - subject overlap: 0 for all split pairs
  - max required patches: 353
  - configured `model.max_patches`: 384
- Area stats completed:
  - file: `outputs/area_stats_fast_packed_seed42.json`
  - global median: 8.339022636413574
  - global MAD: 0.22008037567138672

Launch command:

```bash
CUDA_VISIBLE_DEVICES=1,2,3,4 PYTHONUNBUFFERED=1 torchrun --standalone --nproc_per_node=4 \
  -m eyemae.train --config configs/eyemae_cnn_512_12l.yaml
```

Initial progress:

| step | total_loss | xy_loss | area_loss | blink_loss | velocity_loss | lr |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.31945 | 0.02723 | 1.09385 | 0.70527 | 0.02917 | 1.00e-07 |
| 50 | 0.28293 | 0.02606 | 0.95800 | 0.64202 | 0.01070 | 5.10e-06 |
| 100 | 0.25003 | 0.01878 | 0.92667 | 0.45432 | 0.00484 | 1.01e-05 |
| 150 | 0.23154 | 0.01216 | 0.96411 | 0.26294 | 0.00268 | 1.51e-05 |
| 200 | 0.14767 | 0.02357 | 0.49378 | 0.24971 | 0.00375 | 2.01e-05 |
| 250 | 0.14235 | 0.01893 | 0.53283 | 0.16629 | 0.00224 | 2.51e-05 |
| 300 | 0.08382 | 0.01211 | 0.30910 | 0.09652 | 0.00237 | 3.01e-05 |
| 350 | 0.08689 | 0.02135 | 0.25375 | 0.14625 | 0.00166 | 3.51e-05 |
| 400 | 0.06983 | 0.01554 | 0.20693 | 0.12755 | 0.00152 | 4.01e-05 |
| 450 | 0.06284 | 0.01182 | 0.21505 | 0.07875 | 0.00138 | 4.51e-05 |
| 500 | 0.08326 | 0.02242 | 0.23687 | 0.13350 | 0.00120 | 5.01e-05 |
| 550 | 0.06979 | 0.01383 | 0.21381 | 0.13079 | 0.00112 | 5.51e-05 |
| 600 | 0.05256 | 0.01154 | 0.16626 | 0.07671 | 0.00095 | 6.01e-05 |
| 650 | 0.07247 | 0.02089 | 0.18272 | 0.14948 | 0.00091 | 6.51e-05 |
| 700 | 0.05177 | 0.01681 | 0.11554 | 0.11764 | 0.00093 | 7.01e-05 |
| 750 | 0.03363 | 0.01160 | 0.07437 | 0.07037 | 0.00116 | 7.51e-05 |
| 800 | 0.05292 | 0.02102 | 0.08960 | 0.13895 | 0.00087 | 8.01e-05 |
| 850 | 0.03898 | 0.01296 | 0.05162 | 0.15615 | 0.00082 | 8.51e-05 |
| 900 | 0.03002 | 0.01061 | 0.06003 | 0.07324 | 0.00077 | 9.01e-05 |
| 950 | 0.04383 | 0.02081 | 0.05504 | 0.11919 | 0.00094 | 9.51e-05 |
| 1000 | 0.03337 | 0.01293 | 0.04330 | 0.11696 | 0.00084 | 1.00e-04 |
| 1050 | 0.02906 | 0.00915 | 0.05692 | 0.08444 | 0.00081 | 1.05e-04 |
| 1100 | 0.04284 | 0.01528 | 0.08724 | 0.10028 | 0.00090 | 1.10e-04 |
| 1150 | 0.03014 | 0.00667 | 0.06192 | 0.11002 | 0.00087 | 1.15e-04 |
| 1200 | 0.02132 | 0.00385 | 0.04021 | 0.09337 | 0.00089 | 1.20e-04 |
| 1250 | 0.02569 | 0.00626 | 0.05827 | 0.07675 | 0.00101 | 1.25e-04 |
| 1300 | 0.02489 | 0.00350 | 0.05219 | 0.10874 | 0.00079 | 1.30e-04 |
| 1350 | 0.01171 | 0.00197 | 0.03424 | 0.02813 | 0.00074 | 1.35e-04 |
| 1400 | 0.01807 | 0.00313 | 0.04822 | 0.05224 | 0.00071 | 1.40e-04 |
| 1450 | 0.01358 | 0.00250 | 0.03913 | 0.03205 | 0.00056 | 1.45e-04 |
| 1500 | 0.01198 | 0.00127 | 0.04366 | 0.01930 | 0.00055 | 1.50e-04 |
| 1550 | 0.01427 | 0.00195 | 0.05041 | 0.02178 | 0.00061 | 1.55e-04 |
| 1600 | 0.01296 | 0.00181 | 0.03976 | 0.03147 | 0.00045 | 1.60e-04 |
| 1650 | 0.00957 | 0.00096 | 0.03062 | 0.02436 | 0.00044 | 1.65e-04 |
| 1700 | 0.00982 | 0.00115 | 0.02946 | 0.02732 | 0.00042 | 1.70e-04 |
| 1750 | 0.01237 | 0.00173 | 0.04003 | 0.02593 | 0.00039 | 1.75e-04 |
| 1800 | 0.00710 | 0.00090 | 0.02572 | 0.01026 | 0.00035 | 1.80e-04 |
| 1850 | 0.01265 | 0.00144 | 0.03789 | 0.03595 | 0.00040 | 1.85e-04 |
| 1900 | 0.01183 | 0.00157 | 0.03733 | 0.02758 | 0.00036 | 1.90e-04 |
| 1950 | 0.00826 | 0.00069 | 0.02946 | 0.01645 | 0.00032 | 1.95e-04 |
| 2000 | 0.00879 | 0.00083 | 0.03093 | 0.01741 | 0.00029 | 2.00e-04 |
| 2050 | 0.01051 | 0.00132 | 0.03481 | 0.02200 | 0.00029 | 2.00e-04 |
| 2100 | 0.00848 | 0.00101 | 0.03190 | 0.01063 | 0.00028 | 2.00e-04 |
| 2150 | 0.01100 | 0.00089 | 0.03566 | 0.02948 | 0.00029 | 2.00e-04 |
| 2200 | 0.00791 | 0.00092 | 0.02477 | 0.02011 | 0.00027 | 2.00e-04 |
| 2250 | 0.00994 | 0.00097 | 0.03275 | 0.02398 | 0.00024 | 2.00e-04 |
| 2300 | 0.00887 | 0.00085 | 0.02890 | 0.02214 | 0.00026 | 2.00e-04 |
| 2350 | 0.00769 | 0.00079 | 0.02655 | 0.01561 | 0.00027 | 2.00e-04 |
| 2400 | 0.00871 | 0.00094 | 0.03271 | 0.01201 | 0.00024 | 2.00e-04 |
| 2450 | 0.01069 | 0.00117 | 0.03542 | 0.02408 | 0.00028 | 2.00e-04 |
| 2500 | 0.00742 | 0.00056 | 0.02626 | 0.01587 | 0.00023 | 2.00e-04 |
| 2550 | 0.00954 | 0.00068 | 0.03615 | 0.01615 | 0.00018 | 2.00e-04 |
| 2600 | 0.00981 | 0.00070 | 0.03673 | 0.01742 | 0.00017 | 2.00e-04 |
| 2650 | 0.01034 | 0.00067 | 0.03918 | 0.01807 | 0.00020 | 2.00e-04 |
| 2700 | 0.00884 | 0.00084 | 0.03201 | 0.01591 | 0.00015 | 2.00e-04 |
| 2750 | 0.00980 | 0.00051 | 0.03722 | 0.01831 | 0.00016 | 2.00e-04 |
| 2800 | 0.00800 | 0.00053 | 0.02790 | 0.01877 | 0.00017 | 2.00e-04 |
| 2850 | 0.00857 | 0.00067 | 0.03059 | 0.01774 | 0.00013 | 2.00e-04 |
| 2900 | 0.00658 | 0.00052 | 0.02311 | 0.01424 | 0.00014 | 2.00e-04 |
| 2950 | 0.00700 | 0.00047 | 0.02380 | 0.01762 | 0.00014 | 2.00e-04 |
| 3000 | 0.00891 | 0.00059 | 0.03544 | 0.01220 | 0.00011 | 2.00e-04 |
| 3050 | 0.01069 | 0.00074 | 0.04001 | 0.01932 | 0.00012 | 2.00e-04 |
| 3100 | 0.00551 | 0.00042 | 0.01975 | 0.01124 | 0.00011 | 2.00e-04 |
| 3150 | 0.00789 | 0.00051 | 0.02973 | 0.01426 | 0.00010 | 2.00e-04 |
| 3200 | 0.01102 | 0.00059 | 0.04449 | 0.01517 | 0.00011 | 2.00e-04 |
| 3250 | 0.00816 | 0.00053 | 0.02768 | 0.02080 | 0.00012 | 2.00e-04 |
| 3300 | 0.00631 | 0.00028 | 0.02243 | 0.01533 | 0.00009 | 2.00e-04 |
| 3350 | 0.00638 | 0.00041 | 0.02315 | 0.01324 | 0.00009 | 2.00e-04 |
| 3400 | 0.00800 | 0.00051 | 0.02407 | 0.02664 | 0.00010 | 2.00e-04 |
| 3450 | 0.00858 | 0.00052 | 0.02961 | 0.02130 | 0.00009 | 2.00e-04 |
| 3500 | 0.00869 | 0.00066 | 0.03082 | 0.01861 | 0.00008 | 2.00e-04 |
| 3550 | 0.00867 | 0.00049 | 0.02440 | 0.03291 | 0.00009 | 2.00e-04 |
| 3600 | 0.00534 | 0.00034 | 0.01882 | 0.01229 | 0.00008 | 2.00e-04 |
| 3650 | 0.00762 | 0.00064 | 0.02862 | 0.01250 | 0.00008 | 2.00e-04 |
| 3700 | 0.00883 | 0.00049 | 0.03139 | 0.02051 | 0.00009 | 2.00e-04 |
| 3750 | 0.00590 | 0.00034 | 0.02124 | 0.01303 | 0.00007 | 2.00e-04 |
| 3800 | 0.00720 | 0.00038 | 0.02243 | 0.02332 | 0.00006 | 2.00e-04 |
| 3850 | 0.00765 | 0.00065 | 0.02618 | 0.01760 | 0.00009 | 2.00e-04 |
| 3900 | 0.00677 | 0.00030 | 0.02017 | 0.02429 | 0.00007 | 2.00e-04 |
| 3950 | 0.00742 | 0.00046 | 0.02768 | 0.01419 | 0.00006 | 2.00e-04 |
| 4000 | 0.00809 | 0.00055 | 0.02501 | 0.02525 | 0.00008 | 2.00e-04 |
| 4050 | 0.00700 | 0.00027 | 0.02804 | 0.01119 | 0.00006 | 2.00e-04 |
| 4100 | 0.00946 | 0.00049 | 0.03897 | 0.01171 | 0.00006 | 2.00e-04 |
| 4150 | 0.00915 | 0.00043 | 0.03281 | 0.02154 | 0.00007 | 2.00e-04 |
| 4200 | 0.00557 | 0.00036 | 0.01922 | 0.01366 | 0.00006 | 2.00e-04 |
| 4250 | 0.00803 | 0.00039 | 0.03178 | 0.01272 | 0.00005 | 2.00e-04 |
| 4300 | 0.00865 | 0.00041 | 0.03587 | 0.01057 | 0.00005 | 2.00e-04 |
| 4350 | 0.00775 | 0.00046 | 0.02678 | 0.01921 | 0.00006 | 2.00e-04 |
| 4400 | 0.00558 | 0.00039 | 0.02044 | 0.01094 | 0.00006 | 2.00e-04 |
| 4450 | 0.00666 | 0.00032 | 0.02597 | 0.01138 | 0.00005 | 2.00e-04 |
| 4500 | 0.00673 | 0.00040 | 0.02569 | 0.01184 | 0.00005 | 2.00e-04 |
| 4550 | 0.00591 | 0.00025 | 0.02125 | 0.01403 | 0.00005 | 2.00e-04 |
| 4600 | 0.00584 | 0.00036 | 0.02419 | 0.00634 | 0.00005 | 2.00e-04 |
| 4650 | 0.01163 | 0.00045 | 0.03843 | 0.03488 | 0.00006 | 2.00e-04 |
| 4700 | 0.00620 | 0.00034 | 0.02366 | 0.01116 | 0.00005 | 2.00e-04 |
| 4750 | 0.00780 | 0.00025 | 0.02938 | 0.01668 | 0.00004 | 2.00e-04 |
| 4800 | 0.00776 | 0.00064 | 0.02638 | 0.01837 | 0.00005 | 2.00e-04 |
| 4850 | 0.00519 | 0.00025 | 0.02073 | 0.00797 | 0.00004 | 2.00e-04 |
| 4900 | 0.00703 | 0.00033 | 0.02620 | 0.01451 | 0.00004 | 2.00e-04 |
| 4950 | 0.00660 | 0.00045 | 0.02388 | 0.01364 | 0.00005 | 2.00e-04 |
| 5000 | 0.00515 | 0.00026 | 0.01931 | 0.01027 | 0.00004 | 2.00e-04 |
| 5050 | 0.00641 | 0.00027 | 0.02377 | 0.01383 | 0.00004 | 2.00e-04 |
| 5100 | 0.00853 | 0.00044 | 0.03055 | 0.01977 | 0.00004 | 2.00e-04 |
| 5150 | 0.00599 | 0.00044 | 0.01912 | 0.01726 | 0.00005 | 2.00e-04 |
| 5200 | 0.00689 | 0.00041 | 0.02782 | 0.00917 | 0.00005 | 2.00e-04 |
| 5250 | 0.00802 | 0.00051 | 0.02977 | 0.01553 | 0.00005 | 2.00e-04 |
| 5300 | 0.00572 | 0.00031 | 0.01952 | 0.01504 | 0.00004 | 1.99e-04 |
| 5350 | 0.00798 | 0.00049 | 0.02683 | 0.02124 | 0.00005 | 1.99e-04 |
| 5400 | 0.00595 | 0.00028 | 0.02370 | 0.00927 | 0.00004 | 1.99e-04 |
| 5450 | 0.00814 | 0.00041 | 0.03055 | 0.01610 | 0.00004 | 1.99e-04 |
| 5500 | 0.00597 | 0.00034 | 0.02395 | 0.00834 | 0.00004 | 1.99e-04 |
| 5550 | 0.00596 | 0.00020 | 0.02451 | 0.00851 | 0.00003 | 1.99e-04 |
| 5600 | 0.00725 | 0.00037 | 0.02594 | 0.01688 | 0.00004 | 1.99e-04 |
| 5650 | 0.00654 | 0.00032 | 0.02360 | 0.01495 | 0.00004 | 1.99e-04 |
| 5700 | 0.00675 | 0.00025 | 0.02933 | 0.00631 | 0.00004 | 1.99e-04 |
| 5750 | 0.00750 | 0.00039 | 0.02758 | 0.01586 | 0.00004 | 1.99e-04 |
| 5800 | 0.00604 | 0.00028 | 0.02383 | 0.00991 | 0.00004 | 1.99e-04 |
| 5850 | 0.00662 | 0.00026 | 0.02743 | 0.00870 | 0.00003 | 1.99e-04 |
| 5900 | 0.00906 | 0.00041 | 0.03133 | 0.02379 | 0.00004 | 1.99e-04 |
| 5950 | 0.00633 | 0.00026 | 0.02289 | 0.01493 | 0.00003 | 1.99e-04 |
| 6000 | 0.00541 | 0.00029 | 0.02313 | 0.00498 | 0.00003 | 1.99e-04 |
| 6050 | 0.00801 | 0.00042 | 0.02875 | 0.01829 | 0.00004 | 1.99e-04 |
| 6100 | 0.00562 | 0.00027 | 0.02079 | 0.01188 | 0.00003 | 1.99e-04 |
| 6150 | 0.00877 | 0.00022 | 0.03984 | 0.00580 | 0.00003 | 1.99e-04 |
| 6200 | 0.00890 | 0.00069 | 0.03300 | 0.01614 | 0.00003 | 1.99e-04 |
| 6250 | 0.00637 | 0.00056 | 0.02191 | 0.01425 | 0.00003 | 1.99e-04 |
| 6300 | 0.00499 | 0.00027 | 0.01926 | 0.00871 | 0.00003 | 1.99e-04 |
| 6350 | 0.00839 | 0.00062 | 0.03069 | 0.01628 | 0.00003 | 1.99e-04 |
| 6400 | 0.00641 | 0.00027 | 0.02205 | 0.01734 | 0.00003 | 1.99e-04 |
| 6450 | 0.00947 | 0.00035 | 0.04193 | 0.00737 | 0.00004 | 1.99e-04 |
| 6500 | 0.00971 | 0.00027 | 0.03545 | 0.02345 | 0.00003 | 1.99e-04 |
| 6550 | 0.00930 | 0.00033 | 0.03844 | 0.01274 | 0.00003 | 1.99e-04 |
| 6600 | 0.00612 | 0.00025 | 0.02140 | 0.01579 | 0.00003 | 1.99e-04 |
| 6650 | 0.00633 | 0.00023 | 0.02852 | 0.00394 | 0.00003 | 1.99e-04 |
| 6700 | 0.00701 | 0.00040 | 0.02855 | 0.00901 | 0.00003 | 1.99e-04 |
| 6750 | 0.00671 | 0.00041 | 0.02515 | 0.01268 | 0.00003 | 1.99e-04 |
| 6800 | 0.00564 | 0.00019 | 0.02316 | 0.00811 | 0.00003 | 1.99e-04 |
| 6850 | 0.00938 | 0.00059 | 0.03644 | 0.01507 | 0.00003 | 1.99e-04 |
| 6900 | 0.00629 | 0.00028 | 0.02486 | 0.01035 | 0.00003 | 1.99e-04 |
| 6950 | 0.00719 | 0.00025 | 0.02593 | 0.01754 | 0.00003 | 1.99e-04 |
| 7000 | 0.00752 | 0.00046 | 0.02666 | 0.01729 | 0.00003 | 1.99e-04 |
| 8000 | 0.00640 | 0.00035 | 0.02209 | 0.01628 | 0.00002 | 1.98e-04 |
| 9000 | 0.00510 | 0.00019 | 0.02102 | 0.00704 | 0.00002 | 1.98e-04 |
| 10000 | 0.00557 | 0.00027 | 0.02275 | 0.00739 | 0.00002 | 1.97e-04 |
| 11000 | 0.00641 | 0.00032 | 0.02355 | 0.01383 | 0.00002 | 1.96e-04 |
| 12000 | 0.00570 | 0.00012 | 0.02475 | 0.00624 | 0.00002 | 1.95e-04 |

First validation:

| step | val/total_loss | val/xy_loss | val/area_loss | val/blink_loss | val/masked_xy_rmse_deg | val/masked_blink_auc |
| --- | --- | --- | --- | --- | --- | --- |
| 1000 | 0.031843 | 0.014603 | 0.019299 | 0.13336 | 5.9347 | 0.60771 |
| 2000 | 0.0036134 | 0.00068785 | 0.0054029 | 0.018351 | 1.3126 | 0.99165 |
| 3000 | 0.0032872 | 0.00040234 | 0.0071251 | 0.014558 | 0.9805 | 0.99561 |
| 4000 | 0.0024752 | 0.00032254 | 0.0042774 | 0.012941 | 0.89171 | 0.99621 |
| 5000 | 0.0023912 | 0.00027687 | 0.0041647 | 0.012792 | 0.81036 | 0.99535 |
| 6000 | 0.0023938 | 0.00024702 | 0.0048186 | 0.011812 | 0.78344 | 0.99604 |
| 7000 | 0.0023903 | 0.00023551 | 0.0050226 | 0.011484 | 0.75006 | 0.99693 |
| 8000 | 0.0023293 | 0.00058314 | 0.0030247 | 0.011398 | 1.2444 | 0.99616 |
| 9000 | 0.0020599 | 0.00023890 | 0.0035164 | 0.011164 | 0.74803 | 0.99686 |
| 10000 | 0.0022418 | 0.00030693 | 0.0041011 | 0.011132 | 0.87926 | 0.99696 |
| 11000 | 0.0020739 | 0.00018330 | 0.0040016 | 0.010890 | 0.65741 | 0.99722 |
| 12000 | 0.0020652 | 0.00016999 | 0.0039261 | 0.011088 | 0.63137 | 0.99654 |
| 13000 baseline | 0.0021663 | 0.00016407 | 0.0045351 | 0.010942 | 0.62693 | 0.99643 |
| 14000 baseline | 0.0018667 | 0.00017643 | 0.0030290 | 0.010833 | 0.63965 | 0.99636 |
| 14000 fast resume | 0.0019653 | 0.00016074 | 0.0035313 | 0.010972 | 0.62668 | 0.99622 |
| 15000 fast | 0.0020908 | 0.00018184 | 0.0042124 | 0.010654 | 0.66413 | 0.99704 |
| 16000 fast | 0.0019390 | 0.00021784 | 0.0033445 | 0.010510 | 0.74897 | 0.99713 |
| 17000 fast | 0.0022690 | 0.00035392 | 0.0041035 | 0.010932 | 0.93182 | 0.99600 |
| 18000 fast | 0.0019475 | 0.00014112 | 0.0036693 | 0.010716 | 0.58533 | 0.99625 |
| 19000 fast | 0.0020581 | 0.00015909 | 0.0041490 | 0.010682 | 0.63513 | 0.99643 |
| 20000 fast | 0.0019146 | 0.00014314 | 0.0036854 | 0.010334 | 0.58885 | 0.99702 |
| 21000 fast | 0.0020020 | 0.00014189 | 0.0041962 | 0.010201 | 0.57984 | 0.99731 |
| 22000 fast | 0.0020132 | 0.00018179 | 0.0041049 | 0.010095 | 0.66417 | 0.99759 |
| 23000 fast | 0.0020580 | 0.00012590 | 0.0045075 | 0.010297 | 0.54986 | 0.99601 |
| 24000 fast | 0.0020264 | 0.00020961 | 0.0040587 | 0.010041 | 0.71133 | 0.99716 |
| 25000 fast | 0.0018363 | 0.00012077 | 0.0035055 | 0.010136 | 0.53829 | 0.99669 |
| 26000 fast | 0.0019201 | 0.00012971 | 0.0037845 | 0.010326 | 0.55683 | 0.99637 |
| 27000 fast | 0.0016775 | 0.00012713 | 0.0027916 | 0.0099119 | 0.54748 | 0.99715 |
| 28000 fast | 0.0019214 | 0.00011980 | 0.0040085 | 0.0099913 | 0.53751 | 0.99713 |

GPU check during run:

```text
GPU1-4 memory used: about 37 GB each
GPU1-4 nvidia-smi utilization sample: 77%, 98%, 99%, 53%
GPU1-4 pmon SM sample: about 63%-68% per training process
Later sample: GPU1-4 memory used about 37.5 GB each, nvidia-smi utilization 72%, 55%, 62%, 72%.
10-second dmon sample after step 2400: GPU1-4 average SM about 82%, 88%, 79%, 83%.
```

Current status:

- Pretrain process is running.
- Initial loss is decreasing.
- No correctness failure observed.
- Step 7000 validation is nearly tied with step 6000 on total loss and improves the checkpoint monitor `val/masked_xy_rmse_deg`.
- Current GPU utilization is active but not fully saturated, likely due to mmap/CPU preprocessing, host-to-device transfer, DDP synchronization, and validation/logging gaps.
- `checkpoint_best.pt`, `checkpoint_last.pt`, and `checkpoint_step_00004000.pt` were written at step 4000.
- `checkpoint_best.pt` was updated at step 5000.
- `checkpoint_best.pt`, `checkpoint_last.pt`, and `checkpoint_step_00006000.pt` were written at step 6000. The best checkpoint update is expected because the pretrain monitor is `val/masked_xy_rmse_deg`.
- `checkpoint_best.pt` was updated at step 7000.
- Step 8000 validation had an xy RMSE rebound, but step 9000 returned to the
  previous best range and slightly improved the monitor to 0.74803.
- Step 10000 did not improve the monitor; `checkpoint_last.pt` and
  `checkpoint_step_00010000.pt` were written.
- Step 11000 improved the monitor to `val/masked_xy_rmse_deg=0.65741`, and
  `checkpoint_best.pt` was updated.
- Step 12000 improved the monitor again to `val/masked_xy_rmse_deg=0.63137`;
  `checkpoint_best.pt`, `checkpoint_last.pt`, and
  `checkpoint_step_00012000.pt` were written.
- Step 13000 baseline validation improved the monitor to
  `val/masked_xy_rmse_deg=0.62693`; `checkpoint_best.pt` was updated.
- Step 14000 baseline validation did not improve the monitor
  (`val/masked_xy_rmse_deg=0.63965`). `checkpoint_last.pt` and
  `checkpoint_step_00014000.pt` were written, then the baseline process was
  stopped after checkpoint verification to run speed tests.
- The official fast continuation resumed from the baseline step-14000
  checkpoint with `configs/eyemae_cnn_512_12l_fast_cache44_nooffset.yaml`.
  Its resume validation at step 14000 gave
  `val/masked_xy_rmse_deg=0.62668`, slightly better than the baseline best, so
  the fast output directory wrote its own `checkpoint_best.pt`.
- Step 15000 fast validation did not improve the monitor
  (`val/masked_xy_rmse_deg=0.66413`). Step 15000 is not a configured save
  boundary; the next numbered checkpoint is expected at step 16000.
- Step 16000 fast validation did not improve the monitor
  (`val/masked_xy_rmse_deg=0.74897`). `checkpoint_last.pt` and
  `checkpoint_step_00016000.pt` were written in the fast output directory.
- Step 17000 fast validation did not improve the monitor
  (`val/masked_xy_rmse_deg=0.93182`). Step 17000 is not a configured save
  boundary.
- Step 18000 fast validation improved the monitor to
  `val/masked_xy_rmse_deg=0.58533`. `checkpoint_best.pt`,
  `checkpoint_last.pt`, and `checkpoint_step_00018000.pt` were written in the
  fast output directory; checkpoint metadata confirms `global_step=18000` and
  `best_metric=0.5853256726976972`.
- Step 19000 fast validation did not improve the monitor
  (`val/masked_xy_rmse_deg=0.63513`). Step 19000 is not a configured save
  boundary.
- Step 20000 fast validation nearly matched but did not improve the monitor
  (`val/masked_xy_rmse_deg=0.58885` vs best 0.58533 from step 18000).
  `checkpoint_last.pt` and `checkpoint_step_00020000.pt` were written with
  `global_step=20000`; checkpoint metadata confirms `best_metric` remains
  0.5853256726976972.
- Step 21000 fast validation improved the monitor to
  `val/masked_xy_rmse_deg=0.57984`. `checkpoint_best.pt` was updated with
  `global_step=21000` and `best_metric=0.5798368446537829`; `checkpoint_last.pt`
  remains at step 20000 because step 21000 is not a configured save boundary.
- Step 22000 fast validation did not improve the monitor
  (`val/masked_xy_rmse_deg=0.66417`). `checkpoint_last.pt` and
  `checkpoint_step_00022000.pt` were written with `global_step=22000`;
  checkpoint metadata confirms `best_metric` remains 0.5798368446537829 from
  step 21000.
- Step 23000 fast validation improved the monitor to
  `val/masked_xy_rmse_deg=0.54986`. `checkpoint_best.pt` was updated with
  `global_step=23000` and `best_metric=0.5498602441629539`; `checkpoint_last.pt`
  remains at step 22000 because step 23000 is not a configured save boundary.
- Step 24000 fast validation did not improve the monitor
  (`val/masked_xy_rmse_deg=0.71133`). `checkpoint_last.pt` and
  `checkpoint_step_00024000.pt` were written with `global_step=24000`;
  checkpoint metadata confirms `best_metric` remains 0.5498602441629539 from
  step 23000.
- Step 25000 fast validation improved the monitor to
  `val/masked_xy_rmse_deg=0.53829`. `checkpoint_best.pt` was updated with
  `global_step=25000` and `best_metric=0.5382878322163417`; `checkpoint_last.pt`
  remains at step 24000 because step 25000 is not a configured save boundary.
- Step 26000 fast validation did not improve the monitor
  (`val/masked_xy_rmse_deg=0.55683`). `checkpoint_last.pt` and
  `checkpoint_step_00026000.pt` were written with `global_step=26000`;
  checkpoint metadata confirms `best_metric` remains 0.5382878322163417 from
  step 25000.
- Step 27000 fast validation did not improve the checkpoint monitor
  (`val/masked_xy_rmse_deg=0.54748` vs best 0.53829 from step 25000), but it
  produced the lowest recorded `val/total_loss` so far in the fast run
  (`0.0016775`). Step 27000 is not a configured save boundary.
- Step 28000 fast validation improved the monitor slightly to
  `val/masked_xy_rmse_deg=0.53751`. `checkpoint_best.pt`,
  `checkpoint_last.pt`, and `checkpoint_step_00028000.pt` were written with
  `global_step=28000`; checkpoint metadata confirms
  `best_metric=0.5375056882294036`.
- Step 29000 fast validation improved the monitor to
  `val/masked_xy_rmse_deg=0.52117`. `checkpoint_best.pt` was updated with
  `global_step=29000` and `best_metric=0.521166803331739`; `checkpoint_last.pt`
  remains at step 28000 because step 29000 is not a configured save boundary.
- Step 30000 fast validation did not improve the monitor
  (`val/masked_xy_rmse_deg=0.53011` vs best 0.52117 from step 29000).
  `checkpoint_last.pt` and `checkpoint_step_00030000.pt` were written with
  `global_step=30000`; checkpoint metadata confirms `checkpoint_best.pt`
  remains at step 29000 and `best_metric=0.521166803331739`.
- Step 31000 fast validation did not improve the monitor
  (`val/masked_xy_rmse_deg=0.64758` vs best 0.52117 from step 29000).
  Step 31000 is not a configured save boundary.
- Step 32000 fast validation improved the monitor to
  `val/masked_xy_rmse_deg=0.50918`. `checkpoint_best.pt`,
  `checkpoint_last.pt`, and `checkpoint_step_00032000.pt` were written with
  `global_step=32000`; checkpoint metadata confirms
  `best_metric=0.5091842693642684`.
- Step 33000 fast validation improved the monitor to
  `val/masked_xy_rmse_deg=0.50394`. `checkpoint_best.pt` was updated with
  `global_step=33000` and `best_metric=0.503944417627027`; `checkpoint_last.pt`
  remains at step 32000 because step 33000 is not a configured save boundary.
- Step 34000 fast validation improved the monitor to
  `val/masked_xy_rmse_deg=0.50060`. `checkpoint_best.pt`,
  `checkpoint_last.pt`, and `checkpoint_step_00034000.pt` were written with
  `global_step=34000`; checkpoint metadata confirms
  `best_metric=0.500602271154861`.
- Step 35000 fast validation improved the monitor to
  `val/masked_xy_rmse_deg=0.48818`. `checkpoint_best.pt` was updated with
  `global_step=35000` and `best_metric=0.4881798027732034`; `checkpoint_last.pt`
  remains at step 34000 because step 35000 is not a configured save boundary.
- Step 36000 fast validation improved the monitor to
  `val/masked_xy_rmse_deg=0.48038`. `checkpoint_best.pt`,
  `checkpoint_last.pt`, and `checkpoint_step_00036000.pt` were written with
  `global_step=36000`; checkpoint metadata confirms
  `best_metric=0.4803788958799253`.
- Step 37000 fast validation did not improve the monitor
  (`val/masked_xy_rmse_deg=0.48049` vs best 0.48038 from step 36000).
  Step 37000 is not a configured save boundary.
- Step 38000 fast validation did not improve the monitor
  (`val/masked_xy_rmse_deg=0.48747` vs best 0.48038 from step 36000).
  `checkpoint_last.pt` and `checkpoint_step_00038000.pt` were written with
  `global_step=38000`; checkpoint metadata confirms `checkpoint_best.pt`
  remains at step 36000 and `best_metric=0.4803788958799253`.
- Step 39000 fast validation improved the monitor to
  `val/masked_xy_rmse_deg=0.47656`. `checkpoint_best.pt` was updated with
  `global_step=39000` and `best_metric=0.47655682739845046`;
  `checkpoint_last.pt` remains at step 38000 because step 39000 is not a
  configured save boundary.
- Step 40000 fast validation did not improve the monitor
  (`val/masked_xy_rmse_deg=0.48482` vs best 0.47656 from step 39000).
  `checkpoint_last.pt` and `checkpoint_step_00040000.pt` were written with
  `global_step=40000`; checkpoint metadata confirms `checkpoint_best.pt`
  remains at step 39000 and `best_metric=0.47655682739845046`.
- Step 41000 fast validation improved the monitor to
  `val/masked_xy_rmse_deg=0.47362`. `checkpoint_best.pt` was updated with
  `global_step=41000` and `best_metric=0.47361976267421324`;
  `checkpoint_last.pt` remains at step 40000 because step 41000 is not a
  configured save boundary.
- Step 42000 fast validation did not improve the monitor
  (`val/masked_xy_rmse_deg=0.48367` vs best 0.47362 from step 41000).
  `checkpoint_last.pt` and `checkpoint_step_00042000.pt` were written with
  `global_step=42000`; checkpoint metadata confirms `checkpoint_best.pt`
  remains at step 41000 and `best_metric=0.47361976267421324`.
- Step 43000 fast validation did not improve the monitor
  (`val/masked_xy_rmse_deg=0.48512` vs best 0.47362 from step 41000).
  Step 43000 is not a configured save boundary.
- Step 44000 fast validation improved the monitor to
  `val/masked_xy_rmse_deg=0.47168`. `checkpoint_best.pt`,
  `checkpoint_last.pt`, and `checkpoint_step_00044000.pt` were written with
  `global_step=44000`; checkpoint metadata confirms
  `best_metric=0.47168129456706787`.
- Step 45000 fast validation improved the monitor to
  `val/masked_xy_rmse_deg=0.46646`. `checkpoint_best.pt` was updated with
  `global_step=45000` and `best_metric=0.4664561255544593`;
  `checkpoint_last.pt` remains at step 44000 because step 45000 is not a
  configured save boundary.
- Step 46000 fast validation did not improve the monitor
  (`val/masked_xy_rmse_deg=0.46856` vs best 0.46646 from step 45000).
  `checkpoint_last.pt` and `checkpoint_step_00046000.pt` were written with
  `global_step=46000`; checkpoint metadata confirms `checkpoint_best.pt`
  remains at step 45000 and `best_metric=0.4664561255544593`.
- Step 47000 fast validation improved the monitor to
  `val/masked_xy_rmse_deg=0.46364`. `checkpoint_best.pt` was updated with
  `global_step=47000` and `best_metric=0.4636437405263664`;
  `checkpoint_last.pt` remains at step 46000 because step 47000 is not a
  configured save boundary.
- Step 48000 fast validation did not improve the monitor
  (`val/masked_xy_rmse_deg=0.46951` vs best 0.46364 from step 47000).
  `checkpoint_last.pt` and `checkpoint_step_00048000.pt` were written with
  `global_step=48000`; checkpoint metadata confirms `checkpoint_best.pt`
  remains at step 47000 and `best_metric=0.4636437405263664`.
- Step 49000 fast validation did not improve the monitor
  (`val/masked_xy_rmse_deg=0.59199` vs best 0.46364 from step 47000).
  Step 49000 is not a configured save boundary. The validation total loss
  remained in the usual range (`val/total_loss=0.0018159`), so this was logged
  as validation metric volatility rather than a training exception.
- Step 50000 fast validation did not improve the monitor
  (`val/masked_xy_rmse_deg=0.46528` vs best 0.46364 from step 47000).
  `checkpoint_last.pt` and `checkpoint_step_00050000.pt` were written with
  `global_step=50000`; checkpoint metadata confirms `checkpoint_best.pt`
  remains at step 47000 and `best_metric=0.4636437405263664`.
- Step 51000 fast validation improved the monitor to
  `val/masked_xy_rmse_deg=0.45937`. `checkpoint_best.pt` was updated with
  `global_step=51000` and `best_metric=0.4593708082136576`;
  `checkpoint_last.pt` remains at step 50000 because step 51000 is not a
  configured save boundary.
- Step 52000 fast validation improved the monitor to
  `val/masked_xy_rmse_deg=0.45290`. `checkpoint_best.pt`,
  `checkpoint_last.pt`, and `checkpoint_step_00052000.pt` were written with
  `global_step=52000`; checkpoint metadata confirms
  `best_metric=0.45290411536624803`.
- Step 53000 fast validation did not improve the monitor
  (`val/masked_xy_rmse_deg=0.45393` vs best 0.45290 from step 52000).
  Step 53000 is not a configured save boundary.
- Step 54000 fast validation did not improve the monitor
  (`val/masked_xy_rmse_deg=0.46954` vs best 0.45290 from step 52000).
  `checkpoint_last.pt` and `checkpoint_step_00054000.pt` were written with
  `global_step=54000`; checkpoint metadata confirms `checkpoint_best.pt`
  remains at step 52000 and `best_metric=0.45290411536624803`.
- Step 55000 fast validation improved the monitor to
  `val/masked_xy_rmse_deg=0.45067`. `checkpoint_best.pt` was updated with
  `global_step=55000` and `best_metric=0.4506728483649554`;
  `checkpoint_last.pt` remains at step 54000 because step 55000 is not a
  configured save boundary.
- Step 56000 fast validation did not improve the monitor
  (`val/masked_xy_rmse_deg=0.45376` vs best 0.45067 from step 55000).
  `checkpoint_last.pt` and `checkpoint_step_00056000.pt` were written with
  `global_step=56000`; checkpoint metadata confirms `checkpoint_best.pt`
  remains at step 55000 and `best_metric=0.4506728483649554`.
- Step 57000 fast validation improved the monitor to
  `val/masked_xy_rmse_deg=0.44419`. `checkpoint_best.pt` was updated with
  `global_step=57000` and `best_metric=0.4441866776057503`;
  `checkpoint_last.pt` remains at step 56000 because step 57000 is not a
  configured save boundary.
- Step 58000 fast validation did not improve the monitor
  (`val/masked_xy_rmse_deg=0.44915` vs best 0.44419 from step 57000).
  `checkpoint_last.pt` and `checkpoint_step_00058000.pt` were written with
  `global_step=58000`; checkpoint metadata confirms `checkpoint_best.pt`
  remains at step 57000 and `best_metric=0.4441866776057503`.
- Step 59000 fast validation did not improve the monitor
  (`val/masked_xy_rmse_deg=0.44550` vs best 0.44419 from step 57000).
  Step 59000 is not a configured save boundary.
- Step 60000 fast validation did not improve the monitor
  (`val/masked_xy_rmse_deg=0.44916` vs best 0.44419 from step 57000).
  `checkpoint_last.pt` and `checkpoint_step_00060000.pt` were written with
  `global_step=60000`; checkpoint metadata confirms `checkpoint_best.pt`
  remains at step 57000 and `best_metric=0.4441866776057503`.
- Step 61000 fast validation improved the monitor to
  `val/masked_xy_rmse_deg=0.44327`. `checkpoint_best.pt` was updated with
  `global_step=61000` and `best_metric=0.44327108575186525`;
  `checkpoint_last.pt` remains at step 60000 because step 61000 is not a
  configured save boundary.
- Step 62000 fast validation improved the monitor to
  `val/masked_xy_rmse_deg=0.44037`. `checkpoint_best.pt`,
  `checkpoint_last.pt`, and `checkpoint_step_00062000.pt` were written with
  `global_step=62000`; checkpoint metadata confirms
  `best_metric=0.44036921221363`.
- Step 63000 fast validation did not improve the monitor
  (`val/masked_xy_rmse_deg=0.44198` vs best 0.44037 from step 62000).
  Step 63000 is not a configured save boundary.
- Step 64000 fast validation did not improve the monitor
  (`val/masked_xy_rmse_deg=0.44462` vs best 0.44037 from step 62000).
  `checkpoint_last.pt` and `checkpoint_step_00064000.pt` were written with
  `global_step=64000`; checkpoint metadata confirms `checkpoint_best.pt`
  remains at step 62000 and `best_metric=0.44036921221363`.
- Step 65000 fast validation improved the monitor to
  `val/masked_xy_rmse_deg=0.43985`. `checkpoint_best.pt` was updated with
  `global_step=65000` and `best_metric=0.43985166430558026`;
  `checkpoint_last.pt` remains at step 64000 because step 65000 is not a
  configured save boundary.
- Step 66000 fast validation improved the monitor to
  `val/masked_xy_rmse_deg=0.43441`, below the previous old-data run best
  monitor of about 0.43636. `checkpoint_best.pt`, `checkpoint_last.pt`, and
  `checkpoint_step_00066000.pt` were written with `global_step=66000`;
  checkpoint metadata confirms `best_metric=0.43440763003367994`.
- Step 67000 fast validation did not improve the monitor
  (`val/masked_xy_rmse_deg=0.43967` vs best 0.43441 from step 66000).
  Step 67000 is not a configured save boundary.
- Step 68000 fast validation improved the monitor to
  `val/masked_xy_rmse_deg=0.43251`. `checkpoint_best.pt`,
  `checkpoint_last.pt`, and `checkpoint_step_00068000.pt` were written with
  `global_step=68000`; checkpoint metadata confirms
  `best_metric=0.4325077391941313`.
- Step 69000 fast validation improved the monitor to
  `val/masked_xy_rmse_deg=0.43121`. `checkpoint_best.pt` was updated with
  `global_step=69000` and `best_metric=0.43120560290797455`;
  `checkpoint_last.pt` remains at step 68000 because step 69000 is not a
  configured save boundary.
- Step 70000 fast validation did not improve the monitor
  (`val/masked_xy_rmse_deg=0.43406` vs best 0.43121 from step 69000).
  `checkpoint_last.pt` and `checkpoint_step_00070000.pt` were written with
  `global_step=70000`; checkpoint metadata confirms `checkpoint_best.pt`
  remains at step 69000 and `best_metric=0.43120560290797455`.
- Step 71000 fast validation did not improve the monitor
  (`val/masked_xy_rmse_deg=0.43250` vs best 0.43121 from step 69000).
  Step 71000 is not a configured save boundary.
- Step 72000 fast validation improved the monitor to
  `val/masked_xy_rmse_deg=0.42732`. `checkpoint_best.pt`,
  `checkpoint_last.pt`, and `checkpoint_step_00072000.pt` were written with
  `global_step=72000`; checkpoint metadata confirms
  `best_metric=0.4273208825787566`.
- Step 73000 fast validation did not improve the monitor
  (`val/masked_xy_rmse_deg=0.43057` vs best 0.42732 from step 72000).
  Step 73000 is not a configured save boundary.
- Step 74000 fast validation improved the monitor to
  `val/masked_xy_rmse_deg=0.42689`. `checkpoint_best.pt`,
  `checkpoint_last.pt`, and `checkpoint_step_00074000.pt` were written with
  `global_step=74000`; checkpoint metadata confirms
  `best_metric=0.42689136959628693`.
- Step 75000 fast validation did not improve the monitor
  (`val/masked_xy_rmse_deg=0.42809` vs best 0.42689 from step 74000).
  Step 75000 is not a configured save boundary.
- Step 76000 fast validation did not improve the monitor
  (`val/masked_xy_rmse_deg=0.42981` vs best 0.42689 from step 74000).
  `checkpoint_last.pt` and `checkpoint_step_00076000.pt` were written with
  `global_step=76000`; checkpoint metadata confirms `checkpoint_best.pt`
  remains at step 74000 and `best_metric=0.42689136959628693`.
- Step 77000 fast validation did not improve the monitor
  (`val/masked_xy_rmse_deg=0.42761` vs best 0.42689 from step 74000).
  Step 77000 is not a configured save boundary.
- Step 78000 fast validation improved the monitor to
  `val/masked_xy_rmse_deg=0.42285`. `checkpoint_best.pt`,
  `checkpoint_last.pt`, and `checkpoint_step_00078000.pt` were written with
  `global_step=78000`; checkpoint metadata confirms
  `best_metric=0.4228492972117707`.
- Step 79000 fast validation did not improve the monitor
  (`val/masked_xy_rmse_deg=0.42338` vs best 0.42285 from step 78000).
  Step 79000 is not a configured save boundary.
- Step 80000 fast validation did not improve the monitor
  (`val/masked_xy_rmse_deg=0.42310` vs best 0.42285 from step 78000).
  `checkpoint_last.pt` and `checkpoint_step_00080000.pt` were written with
  `global_step=80000`; checkpoint metadata confirms `checkpoint_best.pt`
  remains at step 78000 and `best_metric=0.4228492972117707`.
- Step 81000 fast validation did not improve the monitor
  (`val/masked_xy_rmse_deg=0.42499` vs best 0.42285 from step 78000).
  Step 81000 is not a configured save boundary.
- Step 82000 fast validation improved the monitor to
  `val/masked_xy_rmse_deg=0.41946`. `checkpoint_best.pt`,
  `checkpoint_last.pt`, and `checkpoint_step_00082000.pt` were written with
  `global_step=82000`; checkpoint metadata confirms
  `best_metric=0.4194625832098894`.
- Step 83000 fast validation did not improve the monitor
  (`val/masked_xy_rmse_deg=0.42067` vs best 0.41946 from step 82000).
  Step 83000 is not a configured save boundary.
- Step 84000 fast validation did not improve the monitor
  (`val/masked_xy_rmse_deg=0.42338` vs best 0.41946 from step 82000).
  `checkpoint_last.pt` and `checkpoint_step_00084000.pt` were written with
  `global_step=84000`; checkpoint metadata confirms `checkpoint_best.pt`
  remains at step 82000 and `best_metric=0.4194625832098894`.
- Step 85000 fast validation did not improve the monitor
  (`val/masked_xy_rmse_deg=0.42260` vs best 0.41946 from step 82000).
  Step 85000 is not a configured save boundary.
- Step 86000 fast validation did not improve the monitor
  (`val/masked_xy_rmse_deg=0.41997` vs best 0.41946 from step 82000).
  `checkpoint_last.pt` and `checkpoint_step_00086000.pt` were written with
  `global_step=86000`; checkpoint metadata confirms `checkpoint_best.pt`
  remains at step 82000 and `best_metric=0.4194625832098894`.
- Step 87000 fast validation did not improve the monitor
  (`val/masked_xy_rmse_deg=0.41988` vs best 0.41946 from step 82000).
  Step 87000 is not a configured save boundary.
- Step 88000 fast validation improved the monitor to
  `val/masked_xy_rmse_deg=0.41737`. `checkpoint_best.pt`,
  `checkpoint_last.pt`, and `checkpoint_step_00088000.pt` were written with
  `global_step=88000`; checkpoint metadata confirms
  `best_metric=0.41736846912527636`.
- Step 89000 fast validation did not improve the monitor
  (`val/masked_xy_rmse_deg=0.41759` vs best 0.41737 from step 88000).
  Step 89000 is not a configured save boundary.
- Step 90000 fast validation did not improve the monitor
  (`val/masked_xy_rmse_deg=0.41800` vs best 0.41737 from step 88000).
  `checkpoint_last.pt` and `checkpoint_step_00090000.pt` were written with
  `global_step=90000`; checkpoint metadata confirms `checkpoint_best.pt`
  remains at step 88000 and `best_metric=0.41736846912527636`.
- Step 91000 fast validation improved the monitor to
  `val/masked_xy_rmse_deg=0.41624`. `checkpoint_best.pt` was written with
  `global_step=91000`; this is not a configured numbered-save boundary, so
  `checkpoint_last.pt` remains at step 90000. Checkpoint metadata confirms
  `best_metric=0.41624130955210276`.
- Step 92000 fast validation improved the monitor to
  `val/masked_xy_rmse_deg=0.41573`. `checkpoint_best.pt`,
  `checkpoint_last.pt`, and `checkpoint_step_00092000.pt` were written with
  `global_step=92000`; checkpoint metadata confirms
  `best_metric=0.41572902001293505`.
- Step 93000 fast validation did not improve the monitor
  (`val/masked_xy_rmse_deg=0.41744` vs best 0.41573 from step 92000).
  Step 93000 is not a configured save boundary.
- Step 94000 fast validation improved the monitor to
  `val/masked_xy_rmse_deg=0.41226`. `checkpoint_best.pt`,
  `checkpoint_last.pt`, and `checkpoint_step_00094000.pt` were written with
  `global_step=94000`; checkpoint metadata confirms
  `best_metric=0.4122596177676634`.
- Step 95000 fast validation did not improve the monitor
  (`val/masked_xy_rmse_deg=0.41476` vs best 0.41226 from step 94000).
  Step 95000 is not a configured save boundary.
- Step 96000 fast validation did not improve the monitor
  (`val/masked_xy_rmse_deg=0.41342` vs best 0.41226 from step 94000).
  `checkpoint_last.pt` and `checkpoint_step_00096000.pt` were written with
  `global_step=96000`; checkpoint metadata confirms `checkpoint_best.pt`
  remains at step 94000 and `best_metric=0.4122596177676634`.
- Step 97000 fast validation did not improve the monitor
  (`val/masked_xy_rmse_deg=0.41607` vs best 0.41226 from step 94000).
  Step 97000 is not a configured save boundary.
- Step 98000 fast validation did not improve the monitor
  (`val/masked_xy_rmse_deg=0.41452` vs best 0.41226 from step 94000).
  `checkpoint_last.pt` and `checkpoint_step_00098000.pt` were written with
  `global_step=98000`; checkpoint metadata confirms `checkpoint_best.pt`
  remains at step 94000 and `best_metric=0.4122596177676634`.
- Step 99000 fast validation did not improve the monitor
  (`val/masked_xy_rmse_deg=0.41315` vs best 0.41226 from step 94000).
  Step 99000 is not a configured save boundary.
- Final step 99999 validation improved the monitor to
  `val/masked_xy_rmse_deg=0.41213`. The pretrain process exited normally after
  reaching `max_steps=100000`. `checkpoint_best.pt`, `checkpoint_last.pt`, and
  `checkpoint_step_00099999.pt` were written with `global_step=99999`;
  checkpoint metadata confirms `best_metric=0.4121278867332725`. Final
  `metrics_last.json` also reports `val/total_loss=0.0018227987501365885`,
  `val/masked_area_mae=0.04033342540395032`,
  `val/masked_blink_auc=0.9977654373775667`, and
  `val/masked_velocity_rmse_deg_per_ms=0.0036721392535344457`.
- GPU utilization investigation during the 84000-91000 window found two main
  contributors to low average utilization. First, every 1000-step validation
  takes about 83-85 seconds, including full validation metrics, group metrics,
  visualization, and metric JSON writes. Second, ordinary training still has
  CPU/DataLoader work despite packed mmap: mmap slicing, array stacking,
  validation, preprocessing, patchification, and padding/collation happen before
  GPU compute. Checkpoint writes are present but much smaller than validation
  overhead. Optional future-run controls were added after this process had
  already started (`train.timing_every_steps`,
  `eval.group_metrics_every_steps`, `eval.visualization_every_steps`), so the
  active baseline is unaffected unless restarted from the edited config.

Speed comparison with previous `main` old-data run:

- Previous run `outputs/eyemae_cnn_512_12l_patch20_stimtoken`:
  training-only intervals around 0.216 s/step, about 4.6 steps/s.
- Current packed run:
  training-only intervals around 0.519 s/step, about 1.93 steps/s.
- The raw seconds/step comparison is not apples-to-apples. The old
  plan-aligned run was launched as 3-GPU training and logged
  `effective max_trials_per_gpu=64 below configured upper bound=256`.
  The current packed run is 4-GPU training and uses the configured
  `max_trials_per_gpu=256` without the old 64-trial cap.
- With token dynamic batching, the sampler budgets
  `3 * batch_len * max_patches_in_batch` tokens per GPU. Simulated effective
  load is about 22.5k padded tokens/GPU/step for the old 64-trial cap versus
  about 59.3k padded tokens/GPU/step for the current 256-trial cap.
- Estimated global load is therefore about 67.6k padded tokens/step for the
  old 3-GPU run and about 237k padded tokens/step for the current 4-GPU run.
  The new run has about 3.5x more padded tokens per step, while step time is
  about 2.3-2.4x longer. In token/s, the packed run is not slower.
- Trial length is not the main explanation: old sampled train trials average
  about 105 patches/trial, and new packed train trials average about
  106 patches/trial.
- Because `max_steps=100000` is unchanged, the larger effective batch also
  changes total training exposure. Approximate trial exposure is about
  192 trials/step for the old 3-GPU 64-trial-capped run and about
  680 trials/step for the current 4-GPU run. With old/new train split sizes of
  about 643k/731k trials, this corresponds to roughly 30 old-data epochs versus
  roughly 90+ new-data epochs if both are run for 100k steps. This must be
  considered when comparing convergence or final metrics.

Speed short tests from baseline step 14000:

- Baseline checkpoint:
  `outputs/pretrain_v3/eyemae_cnn_512_12l_patch20_stimtoken/checkpoint_step_00014000.pt`.
- `configs/eyemae_cnn_512_12l_speed_cache44.yaml`:
  median train-only speed 0.5189 s/step for steps 14050-15000.
- `configs/eyemae_cnn_512_12l_speed_cache44_nooffset.yaml`:
  median train-only speed 0.5013 s/step for steps 14050-15000.
- `configs/eyemae_cnn_512_12l_speed_cache44_nooffset_prefetch4.yaml`:
  median train-only speed 0.5102 s/step for steps 14050-15000.
- Fastest tested variant is `cache44_nooffset`.
- Official continuation uses
  `configs/eyemae_cnn_512_12l_fast_cache44_nooffset.yaml`, which keeps
  `max_seq_tokens_per_gpu=60000` and `max_trials_per_gpu=256`, sets
  `max_open_shards_per_worker=44`, disables per-trial offset validation, and
  resumes from the baseline step-14000 checkpoint with the normal
  `max_steps=100000` schedule.

Packed data-pipeline investigation:

- The new packed dataset has 44 shards, about 69 GB.
- Baseline keeps only 16 open shards per worker.
- Sampled token batches touch many shards: median about 39 shards per batch,
  p95 about 42.
- Simulated worker LRU cache with 16 shards gives high shard miss/evict churn;
  increasing cache to 44 should nearly eliminate shard open/evict misses.
- Cache/no-offset changes improve speed modestly, but they do not explain the
  full old/new step-time gap. The dominant reason for the apparent 2x step
  slowdown is the larger effective per-step batch in the current run.
- Packed mmap still does not eliminate all CPU work: each trial is sliced from
  mmap, copied/stacked into eye/stim arrays, validated, preprocessed, and
  patchified. Shard-aware batching or precomputed patch tensors are the next
  likely speed improvements if token/s is still insufficient.

Subagent effect comparison through step 9000:

- New run speed is about 0.49x of the old run by validation/checkpoint wall time.
- New run best `val/masked_xy_rmse_deg` through step 9000 is 0.74803 vs old
  0.7630, slightly better.
- New run best velocity RMSE through step 9000 is also slightly better.
- New run total loss and area MAE are worse, but those metrics are less directly
  comparable because the dataset cleaning, area stats, and blink distribution
  changed.

## Downstream Launch Preparation

- Updated the formal downstream configs to use the completed new-data pretrain
  checkpoint:
  `outputs/pretrain_v3_fast_cache44_nooffset/eyemae_cnn_512_12l_patch20_stimtoken/checkpoint_best.pt`.
  The corresponding `pretrained.pretrain_config` now points to
  `configs/eyemae_cnn_512_12l_fast_cache44_nooffset.yaml`.
- Kept downstream `area.stats_path` at
  `outputs/area_stats_fast_packed_seed42.json` for this pipeline run, matching
  the area normalization actually used by the completed pretrain baseline.
- Audited all 28 formal downstream configs
  (`7 tasks x {scratch, linear_probe, partial, full}`): config files, pretrain
  checkpoint paths, area stats path, train/validation/test CSVs, split summaries,
  and subject-overlap checks passed.
- GPU smoke test passed on `configs/downstream/ad_binary_linear_probe.yaml`
  with `max_epochs=1`, `max_train_batches=1`, and `max_eval_batches=1`.
  The smoke loaded the new pretrain checkpoint and completed train/val/test
  evaluation. AUROC was NaN in the smoke because the capped evaluation batches
  contained only one class, which is expected for this smoke setting.
- Formal downstream queue launched with:
  `python scripts/run_downstream_v3_queue.py --gpus 1,2,3,4 --log-dir outputs/downstream_v3_logs/run_20260620_full --poll-seconds 15`.
  The queue runs 28 jobs (`7 tasks x 4 modes`) with one job per GPU and
  per-job logs under `outputs/downstream_v3_logs/run_20260620_full/`.
- Initial queue status: the four `pd_related_5class` jobs
  (`scratch`, `linear_probe`, `partial`, `full`) started on GPU1-4 and entered
  training normally. GPU utilization during the first PD epoch sampled around
  97-99% on GPU1-4. The four jobs wrote `param_counts.json`; train losses were
  decreasing in early epoch logs.

Superseded baseline downstream run:

The earlier queue under `outputs/downstream_v3/` was stopped and replaced by
the current fast queue under `outputs/downstream_v3_fast/`. It is kept here only
as historical context and is not completion evidence for the active goal.

| item | value |
| --- | --- |
| old queue root | `outputs/downstream_v3/` |
| stopped reason | PD was re-split randomly at subject level, and the downstream plan switched to the faster early-stopping policy |
| completed before stop | `pd_related_5class/pretrained_linear_probe` |
| completed test metric | macro AUROC OVR `0.8680341326630808`, accuracy `0.7363344051446945`, macro F1 `0.6113239750594727` |
| queue recovery note | an earlier supervisor replacement happened at 2026-06-20 18:54, but that queue was later superseded by the current `downstream_v3_fast` queue |

Old baseline PD 5-class epoch matrix:

Cell format: `train_loss / val_macro_auroc_ovr / val_macro_f1`. `-` means
that mode did not reach validation for that epoch before the old queue was
stopped or completed.

| epoch | scratch | linear | partial | full |
| ---: | --- | --- | --- | --- |
| 0 | `1.51639 / 0.77265 / 0.33858` | `1.29404 / 0.85180 / 0.48930` | `1.22848 / 0.87056 / 0.51318` | `1.12091 / 0.89594 / 0.61356` |
| 1 | `1.39360 / 0.78882 / 0.37250` | `1.17050 / 0.88032 / 0.55668` | `1.02612 / 0.89454 / 0.61630` | `0.82149 / 0.91664 / 0.68923` |
| 2 | `1.30501 / 0.80771 / 0.44070` | `1.10566 / 0.88386 / 0.60725` | `0.90463 / 0.89923 / 0.67651` | `0.63514 / 0.91904 / 0.72366` |
| 3 | `1.22723 / 0.82316 / 0.49441` | `1.06073 / 0.88549 / 0.54844` | `0.81650 / 0.90117 / 0.69796` | `0.50816 / 0.93223 / 0.73441` |
| 4 | `1.15946 / 0.84124 / 0.50234` | `1.03378 / 0.89450 / 0.57609` | `0.74656 / 0.91068 / 0.66017` | `0.41166 / 0.93473 / 0.78355` |
| 5 | `1.09344 / 0.84973 / 0.54293` | `1.01161 / 0.90299 / 0.64658` | `0.68856 / 0.91760 / 0.72259` | `0.34312 / 0.94105 / 0.76713` |
| 6 | `1.03133 / 0.85492 / 0.64720` | `0.99255 / 0.90315 / 0.67813` | `0.63505 / 0.91712 / 0.73159` | `0.28775 / 0.94456 / 0.77511` |
| 7 | `0.96955 / 0.86790 / 0.58232` | `0.98034 / 0.90067 / 0.65191` | `0.59032 / 0.91942 / 0.71572` | `0.24112 / 0.94450 / 0.74236` |
| 8 | `0.90983 / 0.88261 / 0.62551` | `0.96704 / 0.90712 / 0.67226` | `0.55385 / 0.92251 / 0.71401` | `0.20689 / 0.94772 / 0.74177` |
| 9 | `0.84860 / 0.88368 / 0.68331` | `0.96042 / 0.90182 / 0.64978` | `0.51898 / 0.91798 / 0.73145` | `0.17699 / 0.95005 / 0.79129` |
| 10 | `0.79466 / 0.87908 / 0.66012` | `0.95129 / 0.90522 / 0.65578` | `0.48671 / 0.92069 / 0.71662` | `0.14979 / 0.94699 / 0.78359` |
| 11 | `0.73994 / 0.88694 / 0.69817` | `0.94504 / 0.90060 / 0.66241` | `0.45847 / 0.92958 / 0.74382` | `0.12934 / 0.94965 / 0.75874` |
| 12 | `0.68749 / 0.90414 / 0.70773` | `0.93824 / 0.90717 / 0.69743` | `0.42837 / 0.92830 / 0.77122` | `0.11641 / 0.95088 / 0.79180` |
| 13 | `0.63543 / 0.89499 / 0.70338` | `0.92879 / 0.90795 / 0.66145` | `0.40835 / 0.92833 / 0.72412` | `0.10466 / 0.94744 / 0.74915` |
| 14 | `0.58851 / 0.88314 / 0.71679` | `0.92877 / 0.90248 / 0.63386` | `0.38253 / 0.93219 / 0.74431` | `0.09280 / 0.95815 / 0.78605` |
| 15 | `0.53905 / 0.90374 / 0.73016` | `0.92288 / 0.90794 / 0.64910` | `0.36282 / 0.92994 / 0.72504` | `0.08391 / 0.95406 / 0.78971` |
| 16 | `0.50036 / 0.90059 / 0.73400` | `0.91670 / 0.90207 / 0.65002` | `0.34172 / 0.93113 / 0.74007` | `0.07746 / 0.95570 / 0.75833` |
| 17 | `0.45883 / 0.91153 / 0.72076` | `0.91863 / 0.90300 / 0.68164` | `0.32600 / 0.93063 / 0.72012` | `0.06961 / 0.95435 / 0.74127` |
| 18 | `0.41917 / 0.90680 / 0.73630` | `0.90674 / 0.90798 / 0.63964` | `0.30834 / 0.93158 / 0.74059` | `0.06604 / 0.95343 / 0.74964` |
| 19 | `0.38929 / 0.90774 / 0.71922` | `0.91007 / 0.91024 / 0.63890` | `0.29345 / 0.93780 / 0.75732` | `0.05984 / 0.95474 / 0.77073` |
| 20 | `0.35906 / 0.90820 / 0.75004` | `0.90477 / 0.90604 / 0.64775` | `0.27867 / 0.93104 / 0.75770` | `0.05784 / 0.95130 / 0.75450` |
| 21 | `0.33099 / 0.91465 / 0.72821` | `0.90515 / 0.90464 / 0.68807` | `0.26732 / 0.93509 / 0.73923` | `0.05509 / 0.95671 / 0.75502` |
| 22 | `0.30733 / 0.91711 / 0.73330` | `0.90483 / 0.91344 / 0.68274` | `0.25647 / 0.94036 / 0.77023` | `0.04915 / 0.95826 / 0.73647` |
| 23 | `0.28623 / 0.91226 / 0.74670` | `0.89756 / 0.90932 / 0.69181` | `0.23946 / 0.93655 / 0.72961` | `0.05030 / 0.95449 / 0.74720` |
| 24 | `0.26619 / 0.92540 / 0.76027` | `0.89830 / 0.90395 / 0.66280` | `0.22763 / 0.93896 / 0.75127` | `0.04577 / 0.95761 / 0.75289` |
| 25 | `0.24519 / 0.91515 / 0.72079` | `0.89509 / 0.90579 / 0.67320` | `0.22232 / 0.93840 / 0.74432` | `0.04574 / 0.96351 / 0.77400` |
| 26 | `0.23714 / 0.92770 / 0.75325` | `0.89415 / 0.90418 / 0.67713` | `0.21056 / 0.93957 / 0.76403` | `0.04003 / 0.95954 / 0.73598` |
| 27 | `0.21790 / 0.93502 / 0.76536` | `0.89179 / 0.90960 / 0.67248` | `0.19953 / 0.94325 / 0.74749` | `0.03919 / 0.95791 / 0.72356` |
| 28 | `0.20743 / 0.92169 / 0.74228` | `0.89092 / 0.90552 / 0.65086` | `0.19649 / 0.94260 / 0.74235` | `0.03615 / 0.95779 / 0.74877` |
| 29 | `0.19706 / 0.92873 / 0.73989` | `0.88808 / 0.91322 / 0.63715` | `0.18396 / 0.93753 / 0.73831` | `0.03972 / 0.95882 / 0.73941` |
| 30 | `0.18866 / 0.92284 / 0.73965` | `0.89001 / 0.90710 / 0.64700` | `0.18027 / 0.94053 / 0.73910` | `0.03775 / 0.95897 / 0.77138` |
| 31 | `0.17349 / 0.92523 / 0.76448` | `0.88573 / 0.91233 / 0.68719` | `0.16902 / 0.94138 / 0.73650` | `0.03459 / 0.96084 / 0.75396` |
| 32 | `0.16764 / 0.93034 / 0.75288` | `0.88541 / 0.90961 / 0.67539` | `0.16400 / 0.94327 / 0.75997` | `0.03481 / 0.96120 / 0.76756` |
| 33 | `0.15748 / 0.92600 / 0.73658` | `0.87989 / 0.90829 / 0.63188` | `0.15683 / 0.94499 / 0.73777` | `0.03296 / 0.95810 / 0.75336` |
| 34 | `0.15096 / 0.92595 / 0.73665` | `0.88149 / 0.91254 / 0.70635` | `0.14976 / 0.94415 / 0.76481` | `0.03291 / 0.95983 / 0.72473` |
| 35 | `0.14743 / 0.93851 / 0.73854` | `0.88214 / 0.90751 / 0.66398` | `0.14402 / 0.94414 / 0.76002` | `0.03130 / 0.95346 / 0.73148` |
| 36 | - | `0.87836 / 0.91626 / 0.68790` | `0.13940 / 0.94591 / 0.73045` | - |
| 37 | - | `0.87549 / 0.90952 / 0.69196` | `0.13353 / 0.94136 / 0.76840` | - |
| 38 | - | `0.88050 / 0.91433 / 0.65456` | `0.12772 / 0.95016 / 0.74198` | - |
| 39 | - | `0.87647 / 0.91417 / 0.69769` | `0.12256 / 0.94893 / 0.75619` | - |
| 40 | - | `0.87662 / 0.90798 / 0.63535` | `0.11986 / 0.94372 / 0.73076` | - |
| 41 | - | `0.87442 / 0.91228 / 0.68436` | `0.11482 / 0.94575 / 0.74889` | - |
| 42 | - | `0.87430 / 0.91071 / 0.69202` | `0.10997 / 0.95024 / 0.76922` | - |
| 43 | - | `0.87013 / 0.91271 / 0.64251` | `0.10510 / 0.94805 / 0.71035` | - |
| 44 | - | `0.87298 / 0.91105 / 0.68697` | `0.10128 / 0.94952 / 0.74948` | - |
| 45 | - | `0.87013 / 0.91554 / 0.67347` | `0.09698 / 0.94825 / 0.74356` | - |
| 46 | - | `0.86870 / 0.91063 / 0.66709` | `0.09464 / 0.95325 / 0.73802` | - |
| 47 | - | `0.86782 / 0.91371 / 0.68256` | `0.09233 / 0.94658 / 0.71794` | - |
| 48 | - | `0.86656 / 0.91248 / 0.67304` | `0.08840 / 0.94790 / 0.74235` | - |
| 49 | - | `0.86978 / 0.91470 / 0.65690` | `0.08582 / 0.95041 / 0.73900` | - |
| 50 | - | `0.86791 / 0.91130 / 0.66310` | `0.08232 / 0.95037 / 0.73913` | - |
| 51 | - | `0.86604 / 0.91560 / 0.70200` | `0.08260 / 0.95574 / 0.76511` | - |
| 52 | - | `0.86522 / 0.91304 / 0.69752` | `0.07924 / 0.94949 / 0.74612` | - |
| 53 | - | `0.86368 / 0.90771 / 0.68288` | `0.07734 / 0.95422 / 0.73566` | - |
| 54 | - | `0.86494 / 0.91019 / 0.69435` | `0.07192 / 0.95054 / 0.73274` | - |
| 55 | - | `0.86414 / 0.91042 / 0.65937` | `0.07170 / 0.95337 / 0.72774` | - |
| 56 | - | `0.86716 / 0.90836 / 0.67533` | `0.06726 / 0.95259 / 0.73538` | - |
| 57 | - | - | `0.06436 / 0.95399 / 0.73520` | - |
| 58 | - | - | `0.06101 / 0.95373 / 0.72922` | - |
| 59 | - | - | `0.06313 / 0.95341 / 0.73527` | - |
| 60 | - | - | `0.06068 / 0.94964 / 0.73336` | - |
| 61 | - | - | `0.05657 / 0.95637 / 0.75457` | - |
| 62 | - | - | `0.05659 / 0.94736 / 0.73915` | - |
| 63 | - | - | `0.05563 / 0.95024 / 0.73674` | - |
| 64 | - | - | `0.05247 / 0.94711 / 0.72157` | - |

Old baseline best validation macro AUROC:

| mode | best epoch | best val macro AUROC OVR | val macro F1 at best |
| --- | ---: | ---: | ---: |
| `scratch` | 35 | `0.93851` | `0.73854` |
| `linear` | 36 | `0.91626` | `0.68790` |
| `partial` | 61 | `0.95637` | `0.75457` |
| `full` | 25 | `0.96351` | `0.77400` |

Old baseline best-checkpoint test rerun:

Reran the four old baseline `checkpoint_best.pt` files on the original formal
PD test split with `python -m eyemae.evaluate_downstream --split test`.
Outputs are under
`outputs/downstream_v3_old_baseline_best_test_eval/pd_related_5class/`.

| mode | best epoch | val subject AUROC | test subject AUROC | test subject AUPRC | test subject acc | test subject bal acc | test subject F1 | test trial AUROC | test trial acc | test trial F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `scratch` | 35 | `0.93851` | `0.91525` | `0.76024` | `0.77170` | `0.61226` | `0.66983` | `0.84362` | `0.70309` | `0.59932` |
| `linear_probe` | 36 | `0.91626` | `0.86803` | `0.61646` | `0.73633` | `0.60748` | `0.61132` | `0.80913` | `0.61186` | `0.46685` |
| `partial` | 61 | `0.95637` | `0.92343` | `0.77181` | `0.81029` | `0.68161` | `0.70290` | `0.87630` | `0.76213` | `0.64064` |
| `full` | 25 | `0.96351` | `0.92743` | `0.77997` | `0.79421` | `0.65996` | `0.69633` | `0.88535` | `0.76464` | `0.64124` |

PD random-split fast fine-tune setup:

| time CST | item | detail |
| --- | --- | --- |
| 2026-06-20 20:09 | split generated | `downstream/PD相关_random_seed20260620/` from all rows in `downstream/PD相关/{train,validation,test}.csv`; subject-level stratified random split, seed `20260620` |
| 2026-06-20 20:09 | split ratios | Preserved formal PD subject proportions: train `0.6404639175`, validation `0.1591494845`, test `0.2003865979` |
| 2026-06-20 20:09 | split size | train `994` subjects / `141206` trials; validation `247` subjects / `34726` trials; test `311` subjects / `44495` trials |
| 2026-06-20 20:09 | subject overlap | `train-validation=0`, `train-test=0`, `validation-test=0`; all five PD classes present in every split |
| 2026-06-20 20:09 | configs generated | `configs/downstream/pd_related_5class_random_seed20260620_fast_{scratch,linear_probe,partial,full}.yaml` |
| 2026-06-20 20:09 | fast early stopping | generated configs use `max_epochs=100`, `early_stopping_patience=10`, `min_epochs_before_early_stopping=0` |
| 2026-06-20 20:09 | pre-launch validation | `py_compile` passed for `scripts/prepare_pd_random_finetune.py` and `scripts/run_downstream_v3_queue.py`; queue dry-run mapped the four configs to GPU1-4; packed downstream dataset audit/summary passed |
| 2026-06-20 20:09 | tests | `pytest -q tests/test_downstream.py tests/test_downstream_packed.py` -> `11 passed in 5.54s` |
| 2026-06-20 20:20 | full fast queue launched | old baseline queue was stopped; launched `tmux` session `eyemae_downstream_v3_fast` using `configs/downstream/queue_pd_random_seed20260620_fast.txt` |
| 2026-06-20 20:20 | queue status | `0` completed, `4` running, `24` pending; first running jobs are PD random split `scratch`, `linear_probe`, `partial`, `full` |
| 2026-06-20 20:20 | GPU sample | GPU1-4 utilization `92%, 91%, 97%, 95%`; logs under `outputs/downstream_v3_fast_logs/run_20260620_random_pd_fast/` |
| 2026-06-20 22:27 | PD binary scope added | Generated `downstream/PD相关_binary_random_seed20260620/` from the PD random split. It reuses the same subjects/splits and treats all four PD-related subtypes as `health_label=1`. |
| 2026-06-20 22:27 | PD binary size | train `994` subjects / `141206` trials; validation `247` subjects / `34726` trials; test `311` subjects / `44495` trials; subject overlap remains zero. |
| 2026-06-20 22:27 | PD binary configs | Added `configs/downstream/pd_binary_random_seed20260620_fast_{scratch,linear_probe,partial,full}.yaml`; all use binary `health_label`, `validation/subject_auroc`, and the same fast early-stopping policy. |
| 2026-06-20 22:27 | queue list extended | `configs/downstream/queue_pd_random_seed20260620_fast.txt` now has 32 jobs. The already-running supervisor started before this addition, so the four PD binary jobs must be picked up by a resume/tail queue after the current in-memory queue finishes or is safely reattached. |
| 2026-06-20 22:27 | tail runner prepared | Added `scripts/wait_and_run_downstream_v3_queue.py`; it waits for the current status to become idle, then launches the expanded 32-job queue with `--resume-status` so only missing jobs run. |
| 2026-06-20 22:31 | tail watcher launched | Started tmux session `eyemae_downstream_v3_tail32`; log file `outputs/downstream_v3_fast_logs/run_20260620_random_pd_fast/tail32.log` currently shows `WAIT completed=1 running=4 pending=23 failures=0`, so it is waiting and has not launched extra training. |

Downstream V3 fast queue:

| item | value |
| --- | --- |
| queue list | `configs/downstream/queue_pd_random_seed20260620_fast.txt` |
| log dir | `outputs/downstream_v3_fast_logs/run_20260620_random_pd_fast/` |
| output root | `outputs/downstream_v3_fast/` |
| tmux session | `eyemae_downstream_v3_fast` |
| tail tmux session | `eyemae_downstream_v3_tail32` waits for the current queue to become idle, then resumes the expanded 32-job queue |
| first jobs | `pd_related_5class_random_seed20260620_fast_{scratch,linear_probe,partial,full}.yaml` |
| remaining jobs | `pd_binary_random_seed20260620_fast_{scratch,linear_probe,partial,full}.yaml`, plus formal split tasks for `epilepsy_binary`, `detox_binary`, `migraine_binary`, `ad_binary`, `mci_binary`, `mci_matched_binary`, each with `scratch`, `linear_probe`, `partial`, `full` |
| early stopping | `max_epochs=100`, `early_stopping_patience=10`, `min_epochs_before_early_stopping=0` |

Goal acceptance criteria for this fast queue:

Durable goal guardrail: see `docs/current_goal_guardrail.md`. If the run is
resumed after a long wait or context compaction, this file is the completion
contract. The goal must not be marked complete until that checklist passes.
On 2026-06-20 the active goal was tightened again: completion must include the
full first-version fine-tune requirements in `docs/downstream_v3_plan.md`,
including tests and acceptance metrics, not only successful queue execution.

| requirement source | acceptance criterion |
| --- | --- |
| `docs/downstream_v3_plan.md` first-version downstream scope | complete all 8 downstream tasks: PD-related 5-class, PD binary, plus `epilepsy_binary`, `detox_binary`, `migraine_binary`, `ad_binary`, `mci_binary`, `mci_matched_binary` |
| `docs/downstream_v3_plan.md` first-version mode matrix | complete all 4 modes per task: `scratch`, `linear_probe`, `partial`, `full`, for 32 total jobs |
| `docs/downstream_v3_plan.md` Sections 21-24 | run/record available tests, verify output artifacts, and report first-version acceptance metrics/result table |
| PD split override requested on 2026-06-20 | use `downstream/PD相关_random_seed20260620/{train,validation,test}.csv` for PD 5-class, with subject-level stratified random split and no subject overlap |
| PD binary scope added on 2026-06-20 | use `downstream/PD相关_binary_random_seed20260620/{train,validation,test}.csv`; label `1` merges the four PD-related subtypes into one patient class |
| Other six tasks | use the provided formal downstream split directories from the packed dataset |
| Main metric | select best checkpoints by validation subject-level main metric; report train/validation/test final metrics after loading each best checkpoint |
| Faster early stopping | use `max_epochs=100`, `early_stopping_patience=10`, `min_epochs_before_early_stopping=0` for the current fast queue |
| Documentation | keep `newdata_v3_training_log.md`, `newdata_v3_implementation_changes.md`, and `newdata_v3_exception_log.md` updated through completion |

Downstream V3 fast queue progress:

Latest health check at 2026-06-21 00:27 CST: `4` completed, `4` running,
`20` pending, `0` failed, `4` final metric files. GPU1-4 utilization sample:
`97%, 96%, 96%, 98%`. No hard error pattern was found in the queue logs. The
latest incomplete audit recognizes all `32` expected jobs and reports only
missing-output warnings for unfinished jobs. The first-version summary command
currently produces `32` rows with `missing_count=28`.

Completed fast jobs so far:

| task | mode | best epoch | best val subject AUROC | test subject AUROC | test subject AUPRC | test subject acc | test subject bal acc | test subject F1 | test subject weighted F1 | test subject Cohen Kappa | output |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `pd_related_5class_random_seed20260620` | `linear_probe` | 19 | `0.90030` | `0.87229` | `0.63372` | `0.72026` | `0.61310` | `0.60879` | `0.71934` | `0.59590` | `outputs/downstream_v3_fast/pd_related_5class_random_seed20260620/pretrained_linear_probe/metrics_final.json` |
| `pd_related_5class_random_seed20260620` | `partial` | 20 | `0.93984` | `0.91077` | `0.75323` | `0.80707` | `0.68295` | `0.71083` | `0.80082` | `0.71226` | `outputs/downstream_v3_fast/pd_related_5class_random_seed20260620/pretrained_partial/metrics_final.json` |
| `pd_related_5class_random_seed20260620` | `full` | 13 | `0.93883` | `0.93061` | `0.79426` | `0.80064` | `0.66366` | `0.70066` | `0.78703` | `0.69651` | `outputs/downstream_v3_fast/pd_related_5class_random_seed20260620/pretrained_full/metrics_final.json` |
| `epilepsy_binary` | `linear_probe` | 0 | `0.82864` | `0.86376` | `0.80285` | `0.80791` | `0.80809` | `0.81319` | `0.80773` | `0.61595` | `outputs/downstream_v3_fast/epilepsy_binary/pretrained_linear_probe/metrics_final.json` |

PD random-split current epoch matrix:

Cell format: `train_loss / val_macro_auroc_ovr / val_macro_f1`. `-` means
that mode has not reached validation for that epoch yet.

| epoch | scratch | linear | partial | full |
| ---: | --- | --- | --- | --- |
| 0 | `1.54569 / 0.77348 / 0.44228` | `1.30164 / 0.84930 / 0.46242` | `1.23717 / 0.86769 / 0.51726` | `1.13323 / 0.89350 / 0.54915` |
| 1 | `1.41154 / 0.77155 / 0.39957` | `1.17752 / 0.85837 / 0.49466` | `1.03788 / 0.87886 / 0.54721` | `0.83407 / 0.89039 / 0.58548` |
| 2 | `1.34148 / 0.83117 / 0.34730` | `1.12038 / 0.88286 / 0.49345` | `0.92789 / 0.89499 / 0.55992` | `0.65930 / 0.91334 / 0.64550` |
| 3 | `1.28064 / 0.82825 / 0.44355` | `1.07508 / 0.87202 / 0.56866` | `0.83840 / 0.90174 / 0.62776` | `0.53944 / 0.91525 / 0.64659` |
| 4 | `1.22301 / 0.82003 / 0.44981` | `1.04264 / 0.87822 / 0.56124` | `0.76475 / 0.90343 / 0.62691` | `0.43170 / 0.91990 / 0.64968` |
| 5 | `1.16808 / 0.84791 / 0.48491` | `1.02039 / 0.89020 / 0.55990` | `0.70294 / 0.91334 / 0.68067` | `0.35789 / 0.93037 / 0.69974` |
| 6 | `1.11062 / 0.86158 / 0.52211` | `1.00230 / 0.87754 / 0.59411` | `0.65315 / 0.91237 / 0.68283` | `0.30061 / 0.92362 / 0.67250` |
| 7 | `1.05497 / 0.86661 / 0.55496` | `0.98518 / 0.88292 / 0.56661` | `0.61211 / 0.92061 / 0.69778` | `0.25052 / 0.92219 / 0.69311` |
| 8 | `0.99322 / 0.88147 / 0.59711` | `0.98206 / 0.88388 / 0.60794` | `0.56809 / 0.91739 / 0.68432` | `0.21353 / 0.92790 / 0.68073` |
| 9 | `0.93823 / 0.87980 / 0.60984` | `0.96722 / 0.88571 / 0.59177` | `0.52710 / 0.92289 / 0.69405` | `0.18242 / 0.93025 / 0.67978` |
| 10 | `0.87440 / 0.89460 / 0.62045` | `0.95865 / 0.88421 / 0.58126` | `0.49630 / 0.92359 / 0.71795` | `0.15716 / 0.93133 / 0.68731` |
| 11 | `0.81482 / 0.90307 / 0.67335` | `0.95111 / 0.88769 / 0.64055` | `0.46922 / 0.93053 / 0.71870` | `0.13436 / 0.93754 / 0.69551` |
| 12 | `0.75626 / 0.89465 / 0.68102` | `0.94329 / 0.89470 / 0.62868` | `0.44578 / 0.92643 / 0.75009` | `0.12068 / 0.93562 / 0.68561` |
| 13 | `0.70100 / 0.89828 / 0.64402` | `0.93699 / 0.88682 / 0.59537` | `0.42233 / 0.92561 / 0.73201` | `0.10558 / 0.93883 / 0.70117` |
| 14 | `0.64750 / 0.90552 / 0.68929` | `0.92948 / 0.89394 / 0.64202` | `0.39195 / 0.92564 / 0.73142` | `0.09426 / 0.93609 / 0.68697` |
| 15 | `0.59390 / 0.90979 / 0.68070` | `0.92461 / 0.88948 / 0.61223` | `0.37983 / 0.93228 / 0.69550` | `0.08808 / 0.93388 / 0.69508` |
| 16 | `0.54337 / 0.89765 / 0.67096` | `0.92850 / 0.88982 / 0.60383` | `0.34866 / 0.92920 / 0.72620` | `0.07635 / 0.93194 / 0.67023` |
| 17 | `0.49553 / 0.90742 / 0.68026` | `0.91853 / 0.89071 / 0.58307` | `0.34289 / 0.93208 / 0.70917` | `0.07497 / 0.93617 / 0.68375` |
| 18 | `0.45626 / 0.91340 / 0.67222` | `0.91652 / 0.88878 / 0.56162` | `0.31361 / 0.92944 / 0.73116` | `0.06736 / 0.93099 / 0.69986` |
| 19 | `0.42077 / 0.91349 / 0.66887` | `0.91918 / 0.90030 / 0.59627` | `0.29607 / 0.93672 / 0.70454` | `0.06201 / 0.93468 / 0.68840` |
| 20 | `0.38445 / 0.92117 / 0.65751` | `0.90832 / 0.89611 / 0.64681` | `0.28508 / 0.93984 / 0.73646` | `0.05457 / 0.93387 / 0.68627` |
| 21 | `0.35468 / 0.91780 / 0.68566` | `0.90844 / 0.88026 / 0.63004` | `0.27398 / 0.93151 / 0.70312` | `0.05419 / 0.93373 / 0.69703` |
| 22 | `0.32596 / 0.91730 / 0.67913` | `0.91089 / 0.88834 / 0.59723` | `0.25634 / 0.93529 / 0.74525` | `0.05108 / 0.93738 / 0.68298` |
| 23 | `0.30284 / 0.92280 / 0.68583` | `0.90620 / 0.88824 / 0.59870` | `0.24415 / 0.93602 / 0.72810` | `0.04988 / 0.93785 / 0.68187` |
| 24 | `0.27646 / 0.92693 / 0.67012` | `0.90527 / 0.88671 / 0.62700` | `0.23992 / 0.93722 / 0.72226` | - |
| 25 | `0.25835 / 0.92556 / 0.64923` | `0.90561 / 0.87461 / 0.60008` | `0.22200 / 0.93576 / 0.71158` | - |
| 26 | - | `0.90552 / 0.88879 / 0.59526` | `0.21296 / 0.93325 / 0.67963` | - |
| 27 | - | `0.89997 / 0.88900 / 0.61858` | `0.20233 / 0.93644 / 0.68833` | - |
| 28 | - | `0.89512 / 0.88618 / 0.58744` | `0.19530 / 0.93820 / 0.72108` | - |
| 29 | - | `0.89662 / 0.88971 / 0.63781` | `0.18233 / 0.93979 / 0.72738` | - |
| 30 | - | - | `0.17884 / 0.93590 / 0.71509` | - |

Best validation macro AUROC so far:

| mode | best epoch | best val macro AUROC OVR | val macro F1 at best |
| --- | ---: | ---: | ---: |
| `scratch` | 24 | `0.92693` | `0.67012` |
| `linear` | 19 | `0.90030` | `0.59627` |
| `partial` | 20 | `0.93984` | `0.73646` |
| `full` | 13 | `0.93883` | `0.70117` |

PD random-split current-best test rerun artifact:

Interim test evaluations were requested on 2026-06-20 and rerun through
2026-06-21 00:06 CST using then-current checkpoints. The detailed test table is
omitted from this log under the current reporting policy; the retained artifact
is only used where the 30-epoch report explicitly needs an under-30 source.
Outputs are under
`outputs/downstream_v3_fast_current_best_test_eval/pd_related_5class_random_seed20260620/`.

Rerun note: the first attempt from the managed sandbox could not access CUDA,
so `scripts/rerun_pd_current_best_test.py` was run with GPU visibility enabled
on GPU0, `batch_size=64`, and `num_workers=0`.

Epilepsy binary current epoch matrix:

Cell format: `train_loss / val_auroc / val_subject_f1`. `-` means that mode
has not reached validation for that epoch yet.

| epoch | scratch | linear | partial | full |
| ---: | --- | --- | --- | --- |
| 0 | `0.59619 / 0.74609 / 0.60163` | `0.51171 / 0.82864 / 0.71942` | `0.47879 / 0.83020 / 0.75524` | `0.42773 / 0.82962 / 0.72857` |
| 1 | `0.53702 / 0.75196 / 0.60465` | `0.47065 / 0.81925 / 0.69697` | `0.41329 / 0.84155 / 0.71642` | `0.32772 / 0.84820 / 0.73913` |
| 2 | `0.52157 / 0.74883 / 0.57377` | `0.45475 / 0.80614 / 0.65079` | `0.38005 / 0.82864 / 0.68750` | `0.27225 / 0.83783 / 0.69697` |
| 3 | `0.50771 / 0.76682 / 0.63636` | `0.44130 / 0.81592 / 0.72059` | `0.35348 / 0.83920 / 0.71942` | - |
| 4 | `0.49635 / 0.76624 / 0.62400` | `0.42723 / 0.81103 / 0.70677` | `0.32984 / 0.83744 / 0.73759` | - |
| 5 | `0.48430 / 0.76721 / 0.66154` | `0.41874 / 0.81651 / 0.68182` | `0.31404 / 0.84174 / 0.72340` | - |
| 6 | `0.47096 / 0.77700 / 0.67647` | `0.41094 / 0.82140 / 0.72993` | `0.29753 / 0.84527 / 0.74286` | - |
| 7 | `0.45825 / 0.77406 / 0.59016` | `0.40398 / 0.81984 / 0.68702` | `0.28493 / 0.83744 / 0.69291` | - |
| 8 | `0.44321 / 0.76917 / 0.65672` | `0.39969 / 0.82062 / 0.73611` | `0.27246 / 0.84096 / 0.70677` | - |
| 9 | `0.43167 / 0.77210 / 0.63636` | `0.39544 / 0.81495 / 0.74648` | `0.26172 / 0.84605 / 0.69231` | - |
| 10 | `0.41702 / 0.76917 / 0.58537` | `0.39134 / 0.81710 / 0.72464` | `0.25237 / 0.84272 / 0.73134` | - |
| 11 | `0.40058 / 0.77250 / 0.68182` | - | `0.24020 / 0.83901 / 0.71533` | - |
| 12 | `0.38855 / 0.77523 / 0.64567` | - | `0.23154 / 0.84272 / 0.71111` | - |
| 13 | `0.37287 / 0.78013 / 0.63077` | - | `0.22212 / 0.85739 / 0.77143` | - |
| 14 | `0.35774 / 0.77152 / 0.59677` | - | `0.21524 / 0.85270 / 0.76259` | - |
| 15 | `0.34211 / 0.77817 / 0.58333` | - | `0.20583 / 0.85270 / 0.74074` | - |
| 16 | `0.32988 / 0.78169 / 0.60504` | - | - | - |
| 17 | `0.31723 / 0.77660 / 0.56140` | - | - | - |
| 18 | `0.29884 / 0.78032 / 0.60163` | - | - | - |
| 19 | `0.28434 / 0.79049 / 0.63934` | - | - | - |
| 20 | `0.27094 / 0.76937 / 0.61290` | - | - | - |
| 21 | `0.25727 / 0.76937 / 0.67606` | - | - | - |
| 22 | `0.24503 / 0.78091 / 0.65152` | - | - | - |
| 23 | `0.23144 / 0.76937 / 0.66142` | - | - | - |
| 24 | `0.22033 / 0.77387 / 0.63492` | - | - | - |
| 25 | `0.20637 / 0.79441 / 0.71942` | - | - | - |
| 26 | running | - | - | - |

Best validation AUROC so far:

| mode | best epoch | best val AUROC | val F1 at best |
| --- | ---: | ---: | ---: |
| `scratch` | 25 | `0.79441` | `0.71942` |
| `linear` | 0 | `0.82864` | `0.71942` |
| `partial` | 13 | `0.85739` | `0.77143` |
| `full` | 1 | `0.84820` | `0.73913` |

`epilepsy_binary/full` started on GPU4 after PD full completed and reached
epoch 2 with `0.27225 / 0.83783 / 0.69697`; its current best remained epoch 1.
`epilepsy_binary/partial` reached epoch 15 with
`0.20583 / 0.85270 / 0.74074`; its current best remained epoch 13.

`epilepsy_binary/linear_probe` triggered fast early stopping at epoch 10 with
`epochs_without_improve=10`; final metrics were exported, and the queue then
started `epilepsy_binary_partial.yaml` on GPU3.

Downstream first-version output compatibility note:

| time CST | change | verification | active-run caveat |
| --- | --- | --- | --- |
| 2026-06-20 20:33 | Added plan-compatible output files to `finetune.py` / `evaluate_downstream.py`: `resolved_config.yaml`, per-split metrics such as `validation_metrics.json`, and validation alias prediction/confusion files. | `python -m py_compile src/eyemae/finetune.py src/eyemae/evaluate_downstream.py`; `pytest -q tests/test_downstream.py::test_train_downstream_tiny_smoke` passed. | The four currently running PD jobs loaded the old code, so their output dirs must be audited/materialized before final completion if these Section 23 files are missing. |
| 2026-06-20 20:34 | Added executable final audit script `scripts/audit_downstream_v3_goal.py`; current run with `--allow-incomplete` produced `goal_audit_latest.json` with no structural errors and 28 expected incomplete-output warnings under the then-current 28-job scope. | `python -m py_compile scripts/audit_downstream_v3_goal.py`; `pytest -q tests/test_downstream.py tests/test_downstream_packed.py tests/test_fast_packed_dataset.py` passed (`12 passed in 5.95s`). | After the 22:27 PD binary addition, the final completion run must be executed without `--allow-incomplete` after all 32 jobs finish. |
| 2026-06-20 20:41 | Added Section 24 result-table generator `scripts/summarize_downstream_v3_results.py`, writing CSV/JSON/Markdown under `outputs/downstream_v3_fast/summary_first_version.*`. | `python -m py_compile scripts/summarize_downstream_v3_results.py`; the pre-PD-binary smoke run produced 28 rows with `missing_count=28`. | After the 22:27 PD binary addition, final run must be repeated with `--require-complete` once all 32 `metrics_final.json` files exist. |
| 2026-06-20 20:48 | Extended `run_summary.json` for future jobs with Section 23 fields and upgraded `audit_downstream_v3_goal.py --materialize-plan-artifacts` to backfill those fields for legacy completed outputs. | `python -m py_compile src/eyemae/finetune.py scripts/audit_downstream_v3_goal.py scripts/summarize_downstream_v3_results.py`; `python scripts/audit_downstream_v3_goal.py --allow-incomplete ...`; `pytest -q tests/test_downstream.py::test_train_downstream_tiny_smoke` passed. | Active PD jobs loaded old code, so final audit should run with `--materialize-plan-artifacts` before the hard audit. |
| 2026-06-20 20:50 | Ran current full test suite as interim Section 21 evidence. | `pytest -q tests` passed (`38 passed in 11.21s`). | Final completion should rerun tests after all downstream jobs finish and after any materialization step. |
| 2026-06-20 20:59 | Re-ran incomplete-state goal audit after the run-summary/materialization changes. | `python scripts/audit_downstream_v3_goal.py --allow-incomplete --report-json outputs/downstream_v3_fast_logs/run_20260620_random_pd_fast/goal_audit_latest.json` exited 0. | Audit had no structural errors under the then-current 28-job scope. After the 22:27 PD binary addition, final audit must require 32 jobs. |
| 2026-06-20 21:50 | Reinforced the active-goal scope lock in `docs/current_goal_guardrail.md`. | The guardrail now explicitly requires final materialization audit, hard audit, complete result-table generation, and `pytest -q tests` before any goal-complete action. | Future resume/context-compaction should start from the guardrail and must not treat queue completion alone as sufficient. |
| 2026-06-20 22:27 | Added PD binary first-version task and expanded audit/summary config list to 32 jobs. | `scripts/prepare_pd_binary_finetune.py` generated the binary view and configs; dry-run listed 32 configs; `scripts/summarize_downstream_v3_results.py` produced 32 rows with `missing_count=31`; incomplete audit exited 0 and reported no structural errors; `scripts/wait_and_run_downstream_v3_queue.py` compiled and exposes the intended CLI; `pytest -q tests/test_downstream.py tests/test_downstream_packed.py` passed (`11 passed in 5.98s`). | The current tmux supervisor read the old 28-job list at process start. The PD binary jobs must be launched by a resume/tail queue once this is safe, and final audit must require all 32 jobs. |
| 2026-06-20 22:42 | Re-ran incomplete-state audit under the expanded 32-job scope while the queue was active. | `python scripts/audit_downstream_v3_goal.py --allow-incomplete --materialize-plan-artifacts --report-json outputs/downstream_v3_fast_logs/run_20260620_random_pd_fast/goal_audit_latest.json` exited 0; `python scripts/summarize_downstream_v3_results.py` produced 32 rows with `missing_count=31`. | This is interim evidence only. Final completion still requires all 32 `metrics_final.json` files, hard audit without `--allow-incomplete`, summary with `--require-complete`, and `pytest -q tests`. |
| 2026-06-20 23:15 | Added `weighted_f1` and `cohen_kappa` to downstream metrics and the first-version summary table. | `python -m py_compile src/eyemae/downstream_metrics.py scripts/audit_downstream_v3_goal.py scripts/summarize_downstream_v3_results.py`; targeted metric tests passed; `pytest -q tests/test_downstream.py tests/test_downstream_packed.py` passed (`12 passed in 5.35s`); materialization backfilled the two completed PD 5-class jobs and `python scripts/summarize_downstream_v3_results.py` produced 32 rows with `missing_count=30`. | Active jobs that started before this code change may finish without these fields in their raw `metrics_final.json`; final materialization audit will backfill from saved prediction CSVs before the hard audit. |

Downstream epoch dynamics:

Cell format: `train_loss / val_subject_metric / val_subject_f1`. Empty or
not-yet-run cells are `-`. For binary tasks, `val_subject_metric` is AUROC; for
the PD 5-class task, it is macro AUROC OVR.

`pd_related_5class_random_seed20260620`

| epoch | scratch | linear | partial | full |
| ---: | --- | --- | --- | --- |
| 0 | 1.546 / 0.773 / 0.442 | 1.302 / 0.849 / 0.462 | 1.237 / 0.868 / 0.517 | 1.133 / 0.893 / 0.549 |
| 1 | 1.412 / 0.772 / 0.400 | 1.178 / 0.858 / 0.495 | 1.038 / 0.879 / 0.547 | 0.834 / 0.890 / 0.585 |
| 2 | 1.341 / 0.831 / 0.347 | 1.120 / 0.883 / 0.493 | 0.928 / 0.895 / 0.560 | 0.659 / 0.913 / 0.645 |
| 3 | 1.281 / 0.828 / 0.444 | 1.075 / 0.872 / 0.569 | 0.838 / 0.902 / 0.628 | 0.539 / 0.915 / 0.647 |
| 4 | 1.223 / 0.820 / 0.450 | 1.043 / 0.878 / 0.561 | 0.765 / 0.903 / 0.627 | 0.432 / 0.920 / 0.650 |
| 5 | 1.168 / 0.848 / 0.485 | 1.020 / 0.890 / 0.560 | 0.703 / 0.913 / 0.681 | 0.358 / 0.930 / 0.700 |
| 6 | 1.111 / 0.862 / 0.522 | 1.002 / 0.878 / 0.594 | 0.653 / 0.912 / 0.683 | 0.301 / 0.924 / 0.672 |
| 7 | 1.055 / 0.867 / 0.555 | 0.985 / 0.883 / 0.567 | 0.612 / 0.921 / 0.698 | 0.251 / 0.922 / 0.693 |
| 8 | 0.993 / 0.881 / 0.597 | 0.982 / 0.884 / 0.608 | 0.568 / 0.917 / 0.684 | 0.214 / 0.928 / 0.681 |
| 9 | 0.938 / 0.880 / 0.610 | 0.967 / 0.886 / 0.592 | 0.527 / 0.923 / 0.694 | 0.182 / 0.930 / 0.680 |
| 10 | 0.874 / 0.895 / 0.620 | 0.959 / 0.884 / 0.581 | 0.496 / 0.924 / 0.718 | 0.157 / 0.931 / 0.687 |
| 11 | 0.815 / 0.903 / 0.673 | 0.951 / 0.888 / 0.641 | 0.469 / 0.931 / 0.719 | 0.134 / 0.938 / 0.696 |
| 12 | 0.756 / 0.895 / 0.681 | 0.943 / 0.895 / 0.629 | 0.446 / 0.926 / 0.750 | 0.121 / 0.936 / 0.686 |
| 13 | 0.701 / 0.898 / 0.644 | 0.937 / 0.887 / 0.595 | 0.422 / 0.926 / 0.732 | 0.106 / 0.939 / 0.701 |
| 14 | 0.647 / 0.906 / 0.689 | 0.929 / 0.894 / 0.642 | 0.392 / 0.926 / 0.731 | 0.094 / 0.936 / 0.687 |
| 15 | 0.594 / 0.910 / 0.681 | 0.925 / 0.889 / 0.612 | 0.380 / 0.932 / 0.696 | 0.088 / 0.934 / 0.695 |
| 16 | 0.543 / 0.898 / 0.671 | 0.928 / 0.890 / 0.604 | 0.349 / 0.929 / 0.726 | 0.076 / 0.932 / 0.670 |
| 17 | 0.496 / 0.907 / 0.680 | 0.919 / 0.891 / 0.583 | 0.343 / 0.932 / 0.709 | 0.075 / 0.936 / 0.684 |
| 18 | 0.456 / 0.913 / 0.672 | 0.917 / 0.889 / 0.562 | 0.314 / 0.929 / 0.731 | 0.067 / 0.931 / 0.700 |
| 19 | 0.421 / 0.913 / 0.669 | 0.919 / 0.900 / 0.596 | 0.296 / 0.937 / 0.705 | 0.062 / 0.935 / 0.688 |
| 20 | 0.384 / 0.921 / 0.658 | 0.908 / 0.896 / 0.647 | 0.285 / 0.940 / 0.736 | 0.055 / 0.934 / 0.686 |
| 21 | 0.355 / 0.918 / 0.686 | 0.908 / 0.880 / 0.630 | 0.274 / 0.932 / 0.703 | 0.054 / 0.934 / 0.697 |
| 22 | 0.326 / 0.917 / 0.679 | 0.911 / 0.888 / 0.597 | 0.256 / 0.935 / 0.745 | 0.051 / 0.937 / 0.683 |
| 23 | 0.303 / 0.923 / 0.686 | 0.906 / 0.888 / 0.599 | 0.244 / 0.936 / 0.728 | 0.050 / 0.938 / 0.682 |
| 24 | 0.276 / 0.927 / 0.670 | 0.905 / 0.887 / 0.627 | 0.240 / 0.937 / 0.722 | - |
| 25 | 0.258 / 0.926 / 0.649 | 0.906 / 0.875 / 0.600 | 0.222 / 0.936 / 0.712 | - |
| 26 | 0.239 / 0.925 / 0.689 | 0.906 / 0.889 / 0.595 | 0.213 / 0.933 / 0.680 | - |
| 27 | 0.224 / 0.924 / 0.676 | 0.900 / 0.889 / 0.619 | 0.202 / 0.936 / 0.688 | - |
| 28 | 0.212 / 0.922 / 0.677 | 0.895 / 0.886 / 0.587 | 0.195 / 0.938 / 0.721 | - |
| 29 | 0.200 / 0.925 / 0.670 | 0.897 / 0.890 / 0.638 | 0.182 / 0.940 / 0.727 | - |
| 30 | 0.185 / 0.920 / 0.660 | - | 0.179 / 0.936 / 0.715 | - |
| 31 | 0.176 / 0.927 / 0.687 | - | - | - |
| 32 | 0.167 / 0.926 / 0.673 | - | - | - |
| 33 | 0.161 / 0.934 / 0.670 | - | - | - |
| 34 | 0.156 / 0.929 / 0.671 | - | - | - |
| 35 | 0.142 / 0.927 / 0.683 | - | - | - |
| 36 | 0.138 / 0.928 / 0.696 | - | - | - |
| 37 | 0.132 / 0.932 / 0.670 | - | - | - |
| 38 | 0.129 / 0.931 / 0.675 | - | - | - |
| 39 | 0.124 / 0.929 / 0.675 | - | - | - |
| 40 | 0.119 / 0.933 / 0.666 | - | - | - |
| 41 | 0.118 / 0.940 / 0.679 | - | - | - |
| 42 | 0.115 / 0.936 / 0.689 | - | - | - |
| 43 | 0.105 / 0.930 / 0.660 | - | - | - |
| 44 | 0.104 / 0.931 / 0.682 | - | - | - |
| 45 | 0.102 / 0.933 / 0.675 | - | - | - |
| 46 | 0.098 / 0.933 / 0.682 | - | - | - |
| 47 | 0.097 / 0.933 / 0.678 | - | - | - |
| 48 | 0.094 / 0.935 / 0.671 | - | - | - |
| 49 | 0.092 / 0.938 / 0.674 | - | - | - |
| 50 | 0.089 / 0.933 / 0.675 | - | - | - |
| 51 | 0.089 / 0.932 / 0.660 | - | - | - |

`epilepsy_binary`

| epoch | scratch | linear | partial | full |
| ---: | --- | --- | --- | --- |
| 0 | 0.596 / 0.746 / 0.602 | 0.512 / 0.829 / 0.719 | 0.479 / 0.830 / 0.755 | 0.428 / 0.830 / 0.729 |
| 1 | 0.537 / 0.752 / 0.605 | 0.471 / 0.819 / 0.697 | 0.413 / 0.842 / 0.716 | 0.328 / 0.848 / 0.739 |
| 2 | 0.522 / 0.749 / 0.574 | 0.455 / 0.806 / 0.651 | 0.380 / 0.829 / 0.688 | 0.272 / 0.838 / 0.697 |
| 3 | 0.508 / 0.767 / 0.636 | 0.441 / 0.816 / 0.721 | 0.353 / 0.839 / 0.719 | 0.227 / 0.848 / 0.778 |
| 4 | 0.496 / 0.766 / 0.624 | 0.427 / 0.811 / 0.707 | 0.330 / 0.837 / 0.738 | 0.188 / 0.857 / 0.783 |
| 5 | 0.484 / 0.767 / 0.662 | 0.419 / 0.817 / 0.682 | 0.314 / 0.842 / 0.723 | 0.159 / 0.862 / 0.771 |
| 6 | 0.471 / 0.777 / 0.676 | 0.411 / 0.821 / 0.730 | 0.298 / 0.845 / 0.743 | 0.134 / 0.857 / 0.761 |
| 7 | 0.458 / 0.774 / 0.590 | 0.404 / 0.820 / 0.687 | 0.285 / 0.837 / 0.693 | 0.114 / 0.851 / 0.757 |
| 8 | 0.443 / 0.769 / 0.657 | 0.400 / 0.821 / 0.736 | 0.272 / 0.841 / 0.707 | 0.097 / 0.852 / 0.745 |
| 9 | 0.432 / 0.772 / 0.636 | 0.395 / 0.815 / 0.746 | 0.262 / 0.846 / 0.692 | 0.084 / 0.854 / 0.783 |
| 10 | 0.417 / 0.769 / 0.585 | 0.391 / 0.817 / 0.725 | 0.252 / 0.843 / 0.731 | 0.071 / 0.856 / 0.772 |
| 11 | 0.401 / 0.772 / 0.682 | - | 0.240 / 0.839 / 0.715 | 0.063 / 0.852 / 0.763 |
| 12 | 0.389 / 0.775 / 0.646 | - | 0.232 / 0.843 / 0.711 | 0.053 / 0.853 / 0.775 |
| 13 | 0.373 / 0.780 / 0.631 | - | 0.222 / 0.857 / 0.771 | 0.049 / 0.856 / 0.729 |
| 14 | 0.358 / 0.772 / 0.597 | - | 0.215 / 0.853 / 0.763 | 0.044 / 0.858 / 0.786 |
| 15 | 0.342 / 0.778 / 0.583 | - | 0.206 / 0.853 / 0.741 | 0.041 / 0.848 / 0.794 |
| 16 | 0.330 / 0.782 / 0.605 | - | 0.200 / 0.853 / 0.708 | - |
| 17 | 0.317 / 0.777 / 0.561 | - | 0.193 / 0.857 / 0.727 | - |
| 18 | 0.299 / 0.780 / 0.602 | - | 0.185 / 0.861 / 0.763 | - |
| 19 | 0.284 / 0.790 / 0.639 | - | 0.176 / 0.852 / 0.698 | - |
| 20 | 0.271 / 0.769 / 0.613 | - | 0.170 / 0.855 / 0.688 | - |
| 21 | 0.257 / 0.769 / 0.676 | - | 0.167 / 0.849 / 0.745 | - |
| 22 | 0.245 / 0.781 / 0.652 | - | 0.158 / 0.863 / 0.771 | - |
| 23 | 0.231 / 0.769 / 0.661 | - | 0.152 / 0.853 / 0.741 | - |
| 24 | 0.220 / 0.774 / 0.635 | - | 0.146 / 0.856 / 0.775 | - |
| 25 | 0.206 / 0.794 / 0.719 | - | 0.142 / 0.855 / 0.727 | - |
| 26 | 0.196 / 0.785 / 0.699 | - | 0.137 / 0.858 / 0.772 | - |
| 27 | 0.183 / 0.779 / 0.623 | - | 0.133 / 0.850 / 0.741 | - |
| 28 | 0.174 / 0.793 / 0.702 | - | 0.128 / 0.851 / 0.712 | - |
| 29 | 0.164 / 0.781 / 0.709 | - | 0.123 / 0.859 / 0.775 | - |
| 30 | - | - | 0.117 / 0.863 / 0.754 | - |
| 31 | - | - | 0.116 / 0.854 / 0.767 | - |
| 32 | - | - | 0.110 / 0.854 / 0.748 | - |
| 33 | - | - | 0.105 / 0.861 / 0.761 | - |
| 34 | - | - | 0.101 / 0.863 / 0.795 | - |
| 35 | - | - | 0.096 / 0.864 / 0.729 | - |
| 36 | - | - | 0.092 / 0.851 / 0.776 | - |
| 37 | - | - | 0.090 / 0.864 / 0.783 | - |
| 38 | - | - | 0.086 / 0.857 / 0.769 | - |
| 39 | - | - | 0.081 / 0.855 / 0.786 | - |
| 40 | - | - | 0.081 / 0.855 / 0.734 | - |
| 41 | - | - | 0.077 / 0.853 / 0.745 | - |
| 42 | - | - | 0.075 / 0.859 / 0.731 | - |
| 43 | - | - | 0.074 / 0.862 / 0.778 | - |
| 44 | - | - | 0.070 / 0.855 / 0.769 | - |
| 45 | - | - | 0.066 / 0.868 / 0.808 | - |
| 46 | - | - | 0.064 / 0.847 / 0.726 | - |
| 47 | - | - | 0.062 / 0.851 / 0.768 | - |
| 48 | - | - | 0.061 / 0.849 / 0.757 | - |
| 49 | - | - | 0.059 / 0.854 / 0.771 | - |
| 50 | - | - | 0.054 / 0.861 / 0.757 | - |
| 51 | - | - | 0.055 / 0.862 / 0.772 | - |
| 52 | - | - | 0.053 / 0.856 / 0.766 | - |
| 53 | - | - | 0.047 / 0.861 / 0.778 | - |
| 54 | - | - | 0.048 / 0.859 / 0.757 | - |
| 55 | - | - | 0.048 / 0.859 / 0.775 | - |

`detox_binary`

| epoch | scratch | linear | partial | full |
| ---: | --- | --- | --- | --- |
| 0 | 0.680 / 0.676 / 0.583 | 0.478 / 0.801 / 0.815 | 0.453 / 0.807 / 0.815 | 0.414 / 0.903 / 0.846 |
| 1 | 0.626 / 0.949 / 0.857 | 0.426 / 0.881 / 0.815 | 0.380 / 0.898 / 0.815 | 0.300 / 0.943 / 0.846 |
| 2 | 0.561 / 0.932 / 0.733 | 0.407 / 0.841 / 0.815 | 0.347 / 0.864 / 0.846 | 0.220 / 0.955 / 0.870 |
| 3 | 0.528 / 0.898 / 0.815 | 0.394 / 0.824 / 0.815 | 0.321 / 0.892 / 0.846 | 0.163 / 0.966 / 0.909 |
| 4 | 0.499 / 0.898 / 0.846 | 0.375 / 0.818 / 0.815 | 0.288 / 0.864 / 0.846 | 0.120 / 0.983 / 0.909 |
| 5 | 0.482 / 0.966 / 0.880 | 0.364 / 0.858 / 0.815 | 0.261 / 0.875 / 0.846 | 0.092 / 0.932 / 0.762 |
| 6 | 0.460 / 0.903 / 0.815 | 0.357 / 0.830 / 0.815 | 0.243 / 0.892 / 0.846 | 0.070 / 0.983 / 0.909 |
| 7 | 0.436 / 0.938 / 0.846 | 0.344 / 0.847 / 0.815 | 0.223 / 0.915 / 0.846 | 0.059 / 0.989 / 0.917 |
| 8 | 0.420 / 0.869 / 0.846 | 0.332 / 0.795 / 0.815 | 0.204 / 0.886 / 0.846 | 0.044 / 0.983 / 0.952 |
| 9 | 0.406 / 0.864 / 0.727 | 0.324 / 0.812 / 0.846 | 0.192 / 0.920 / 0.846 | 0.041 / 0.989 / 0.957 |
| 10 | 0.377 / 0.915 / 0.818 | 0.314 / 0.835 / 0.846 | 0.180 / 0.909 / 0.917 | 0.032 / 0.960 / 0.857 |
| 11 | 0.369 / 0.989 / 0.909 | 0.309 / 0.835 / 0.769 | 0.165 / 0.915 / 0.833 | 0.031 / 0.977 / 0.833 |
| 12 | 0.353 / 0.932 / 0.870 | - | 0.155 / 0.903 / 0.846 | 0.024 / 0.966 / 0.870 |
| 13 | 0.339 / 0.858 / 0.880 | - | 0.142 / 0.915 / 0.870 | 0.026 / 0.977 / 0.952 |
| 14 | 0.324 / 0.858 / 0.800 | - | 0.133 / 0.915 / 0.833 | 0.022 / 0.955 / 0.800 |
| 15 | 0.305 / 0.886 / 0.833 | - | 0.127 / 0.920 / 0.917 | 0.022 / 0.989 / 0.917 |
| 16 | 0.292 / 0.864 / 0.846 | - | 0.115 / 0.915 / 0.917 | 0.022 / 0.983 / 0.857 |
| 17 | 0.275 / 0.858 / 0.833 | - | 0.109 / 0.920 / 0.917 | 0.016 / 0.983 / 0.909 |
| 18 | 0.265 / 0.864 / 0.667 | - | 0.102 / 0.915 / 0.818 | - |
| 19 | 0.257 / 0.875 / 0.786 | - | 0.098 / 0.903 / 0.870 | - |
| 20 | 0.246 / 0.915 / 0.846 | - | - | - |
| 21 | 0.231 / 0.881 / 0.833 | - | - | - |

`migraine_binary`

| epoch | scratch | linear | partial | full |
| ---: | --- | --- | --- | --- |
| 0 | 0.695 / 0.473 / 0.276 | 0.597 / 0.688 / 0.571 | 0.564 / 0.742 / 0.645 | 0.472 / 0.877 / 0.710 |
| 1 | 0.669 / 0.565 / 0.516 | 0.516 / 0.808 / 0.667 | 0.427 / 0.877 / 0.667 | 0.282 / 0.885 / 0.710 |
| 2 | 0.628 / 0.608 / 0.429 | 0.472 / 0.819 / 0.688 | 0.352 / 0.873 / 0.714 | 0.190 / 0.912 / 0.741 |
| 3 | 0.598 / 0.677 / 0.571 | 0.445 / 0.842 / 0.688 | 0.306 / 0.842 / 0.690 | 0.142 / 0.865 / 0.750 |
| 4 | 0.567 / 0.688 / 0.552 | 0.418 / 0.823 / 0.710 | 0.265 / 0.846 / 0.667 | 0.102 / 0.896 / 0.800 |
| 5 | 0.545 / 0.700 / 0.571 | 0.401 / 0.850 / 0.688 | 0.236 / 0.846 / 0.692 | 0.079 / 0.915 / 0.759 |
| 6 | 0.528 / 0.658 / 0.435 | 0.387 / 0.877 / 0.686 | 0.210 / 0.846 / 0.714 | 0.071 / 0.877 / 0.759 |
| 7 | 0.512 / 0.704 / 0.519 | 0.373 / 0.854 / 0.688 | 0.195 / 0.900 / 0.690 | 0.056 / 0.869 / 0.759 |
| 8 | 0.486 / 0.758 / 0.690 | 0.357 / 0.912 / 0.686 | 0.172 / 0.838 / 0.714 | 0.044 / 0.862 / 0.759 |
| 9 | 0.466 / 0.746 / 0.538 | 0.348 / 0.877 / 0.727 | 0.161 / 0.923 / 0.774 | 0.047 / 0.892 / 0.759 |
| 10 | 0.453 / 0.731 / 0.643 | 0.335 / 0.873 / 0.667 | 0.145 / 0.935 / 0.786 | 0.031 / 0.904 / 0.759 |
| 11 | 0.431 / 0.727 / 0.647 | 0.324 / 0.858 / 0.727 | 0.132 / 0.904 / 0.759 | 0.034 / 0.931 / 0.759 |
| 12 | 0.410 / 0.819 / 0.621 | 0.314 / 0.877 / 0.688 | 0.124 / 0.935 / 0.786 | 0.025 / 0.877 / 0.759 |
| 13 | 0.394 / 0.808 / 0.647 | 0.315 / 0.869 / 0.686 | 0.116 / 0.946 / 0.828 | 0.025 / 0.865 / 0.769 |
| 14 | 0.369 / 0.804 / 0.615 | 0.298 / 0.854 / 0.690 | 0.102 / 0.950 / 0.815 | 0.022 / 0.892 / 0.741 |
| 15 | 0.353 / 0.792 / 0.643 | 0.295 / 0.896 / 0.727 | 0.094 / 0.935 / 0.828 | 0.017 / 0.873 / 0.710 |
| 16 | 0.333 / 0.788 / 0.552 | 0.287 / 0.881 / 0.706 | 0.087 / 0.938 / 0.828 | 0.019 / 0.885 / 0.710 |
| 17 | 0.319 / 0.796 / 0.621 | 0.280 / 0.854 / 0.706 | 0.081 / 0.935 / 0.774 | 0.018 / 0.900 / 0.759 |
| 18 | 0.302 / 0.796 / 0.640 | 0.275 / 0.888 / 0.706 | 0.080 / 0.946 / 0.786 | 0.018 / 0.885 / 0.774 |
| 19 | 0.288 / 0.815 / 0.593 | - | 0.068 / 0.954 / 0.828 | 0.014 / 0.873 / 0.800 |
| 20 | 0.269 / 0.781 / 0.552 | - | 0.068 / 0.931 / 0.786 | 0.015 / 0.877 / 0.786 |
| 21 | 0.256 / 0.804 / 0.609 | - | 0.061 / 0.938 / 0.786 | 0.016 / 0.896 / 0.786 |
| 22 | 0.248 / 0.777 / 0.571 | - | 0.058 / 0.942 / 0.828 | - |
| 23 | - | - | 0.060 / 0.946 / 0.786 | - |
| 24 | - | - | 0.051 / 0.938 / 0.828 | - |
| 25 | - | - | 0.047 / 0.927 / 0.774 | - |
| 26 | - | - | 0.045 / 0.935 / 0.857 | - |
| 27 | - | - | 0.046 / 0.935 / 0.828 | - |
| 28 | - | - | 0.040 / 0.923 / 0.800 | - |
| 29 | - | - | 0.038 / 0.938 / 0.828 | - |

`ad_binary`

| epoch | scratch | linear | partial | full |
| ---: | --- | --- | --- | --- |
| 0 | 0.605 / 0.961 / 0.853 | 0.429 / 0.927 / 0.877 | 0.401 / 0.949 / 0.912 | 0.326 / 0.969 / 0.949 |
| 1 | 0.498 / 0.957 / 0.875 | 0.369 / 0.942 / 0.871 | 0.302 / 0.958 / 0.903 | 0.173 / 0.976 / 0.933 |
| 2 | 0.464 / 0.948 / 0.889 | 0.339 / 0.937 / 0.881 | 0.244 / 0.961 / 0.933 | 0.119 / 0.958 / 0.949 |
| 3 | 0.426 / 0.946 / 0.867 | 0.316 / 0.940 / 0.912 | 0.203 / 0.957 / 0.931 | 0.088 / 0.957 / 0.949 |
| 4 | 0.396 / 0.962 / 0.918 | 0.295 / 0.956 / 0.931 | 0.173 / 0.964 / 0.949 | 0.067 / 0.961 / 0.949 |
| 5 | 0.378 / 0.964 / 0.881 | 0.279 / 0.960 / 0.949 | 0.151 / 0.980 / 0.949 | 0.055 / 0.958 / 0.949 |
| 6 | 0.356 / 0.958 / 0.915 | 0.263 / 0.958 / 0.949 | 0.133 / 0.976 / 0.949 | 0.044 / 0.969 / 0.949 |
| 7 | 0.334 / 0.978 / 0.949 | 0.254 / 0.958 / 0.949 | 0.124 / 0.968 / 0.933 | 0.039 / 0.953 / 0.933 |
| 8 | 0.315 / 0.956 / 0.931 | 0.240 / 0.954 / 0.931 | 0.108 / 0.962 / 0.949 | 0.032 / 0.957 / 0.949 |
| 9 | 0.293 / 0.966 / 0.933 | 0.234 / 0.945 / 0.949 | 0.099 / 0.956 / 0.949 | 0.027 / 0.964 / 0.933 |
| 10 | 0.272 / 0.972 / 0.931 | 0.228 / 0.953 / 0.949 | 0.092 / 0.964 / 0.949 | 0.027 / 0.954 / 0.949 |
| 11 | 0.250 / 0.956 / 0.931 | 0.218 / 0.949 / 0.949 | 0.083 / 0.962 / 0.949 | 0.024 / 0.974 / 0.949 |
| 12 | 0.233 / 0.964 / 0.949 | 0.216 / 0.956 / 0.949 | 0.078 / 0.968 / 0.949 | - |
| 13 | 0.223 / 0.968 / 0.893 | 0.205 / 0.950 / 0.949 | 0.070 / 0.964 / 0.949 | - |
| 14 | 0.208 / 0.950 / 0.918 | 0.199 / 0.953 / 0.949 | 0.066 / 0.964 / 0.949 | - |
| 15 | 0.196 / 0.956 / 0.949 | 0.198 / 0.950 / 0.949 | 0.062 / 0.962 / 0.949 | - |
| 16 | 0.185 / 0.954 / 0.918 | - | - | - |
| 17 | 0.170 / 0.961 / 0.918 | - | - | - |

`mci_binary`

| epoch | scratch | linear | partial | full |
| ---: | --- | --- | --- | --- |
| 0 | 0.693 / 0.459 / 0.424 | 0.687 / 0.420 / 0.368 | 0.685 / 0.389 / 0.374 | 0.670 / 0.361 / 0.349 |
| 1 | 0.690 / 0.463 / 0.447 | 0.676 / 0.395 / 0.209 | 0.665 / 0.381 / 0.194 | 0.624 / 0.304 / 0.380 |
| 2 | 0.690 / 0.473 / 0.410 | 0.670 / 0.403 / 0.213 | 0.650 / 0.380 / 0.216 | 0.581 / 0.283 / 0.295 |
| 3 | 0.689 / 0.476 / 0.500 | 0.666 / 0.361 / 0.433 | 0.639 / 0.357 / 0.424 | 0.544 / 0.253 / 0.190 |
| 4 | 0.687 / 0.439 / 0.274 | 0.660 / 0.334 / 0.330 | 0.625 / 0.326 / 0.279 | 0.510 / 0.207 / 0.207 |
| 5 | 0.685 / 0.454 / 0.320 | 0.656 / 0.346 / 0.364 | 0.614 / 0.333 / 0.295 | 0.482 / 0.200 / 0.222 |
| 6 | 0.683 / 0.448 / 0.414 | 0.652 / 0.333 / 0.265 | 0.604 / 0.321 / 0.225 | 0.458 / 0.190 / 0.154 |
| 7 | 0.681 / 0.429 / 0.347 | 0.649 / 0.336 / 0.253 | 0.594 / 0.290 / 0.259 | 0.437 / 0.180 / 0.150 |
| 8 | 0.679 / 0.407 / 0.316 | 0.646 / 0.329 / 0.282 | 0.582 / 0.262 / 0.250 | 0.418 / 0.164 / 0.146 |
| 9 | 0.676 / 0.384 / 0.341 | 0.643 / 0.329 / 0.234 | 0.573 / 0.244 / 0.205 | 0.400 / 0.153 / 0.103 |
| 10 | 0.673 / 0.396 / 0.282 | 0.640 / 0.351 / 0.215 | 0.565 / 0.263 / 0.244 | 0.384 / 0.161 / 0.127 |
| 11 | 0.669 / 0.388 / 0.472 | - | - | - |
| 12 | 0.667 / 0.391 / 0.484 | - | - | - |
| 13 | 0.664 / 0.355 / 0.219 | - | - | - |

`mci_matched_binary`

| epoch | scratch | linear | partial | full |
| ---: | --- | --- | --- | --- |
| 0 | 0.691 / 0.526 / 0.095 | 0.647 / 0.599 / 0.629 | 0.637 / 0.623 / 0.562 | 0.592 / 0.609 / 0.667 |
| 1 | 0.678 / 0.457 / 0.622 | 0.597 / 0.592 / 0.698 | 0.560 / 0.612 / 0.667 | 0.445 / 0.571 / 0.500 |
| 2 | 0.662 / 0.429 / 0.429 | 0.568 / 0.557 / 0.438 | 0.506 / 0.567 / 0.500 | 0.343 / 0.519 / 0.400 |
| 3 | 0.640 / 0.405 / 0.345 | 0.544 / 0.609 / 0.467 | 0.458 / 0.637 / 0.606 | 0.271 / 0.606 / 0.514 |
| 4 | 0.615 / 0.429 / 0.250 | 0.525 / 0.540 / 0.345 | 0.416 / 0.557 / 0.467 | 0.211 / 0.533 / 0.500 |
| 5 | 0.593 / 0.405 / 0.444 | 0.508 / 0.550 / 0.615 | 0.382 / 0.561 / 0.500 | 0.169 / 0.516 / 0.429 |
| 6 | 0.581 / 0.426 / 0.526 | 0.492 / 0.526 / 0.514 | 0.349 / 0.588 / 0.529 | 0.134 / 0.491 / 0.471 |
| 7 | 0.565 / 0.453 / 0.526 | 0.484 / 0.522 / 0.605 | 0.318 / 0.561 / 0.556 | 0.109 / 0.543 / 0.462 |
| 8 | 0.546 / 0.422 / 0.444 | 0.467 / 0.602 / 0.684 | 0.294 / 0.623 / 0.564 | 0.091 / 0.505 / 0.452 |
| 9 | 0.528 / 0.436 / 0.605 | 0.458 / 0.557 / 0.579 | 0.270 / 0.616 / 0.585 | 0.074 / 0.519 / 0.500 |
| 10 | 0.518 / 0.426 / 0.323 | 0.448 / 0.581 / 0.516 | 0.256 / 0.595 / 0.552 | 0.066 / 0.550 / 0.529 |
| 11 | - | 0.440 / 0.561 / 0.545 | 0.236 / 0.626 / 0.552 | - |
| 12 | - | 0.436 / 0.599 / 0.533 | 0.223 / 0.543 / 0.529 | - |
| 13 | - | 0.426 / 0.557 / 0.579 | 0.204 / 0.595 / 0.474 | - |

Downstream run guards and open gates:

| item | current state | completion condition |
| --- | --- | --- |
| Main queue | Idle: 28 completed, 0 running, 0 pending, and 1 known failure (`epilepsy_binary_scratch rc=-15` from the retracted scratch-cap interruption) in `outputs/downstream_v3_fast_logs/run_20260620_random_pd_fast/status.json`. | The explicit missing-job补跑 queue writes the fair `epilepsy_binary_scratch` final result. |
| Missing-job补跑 queue | tmux session `eyemae_downstream_v3_missing_pd_binary_epilepsy_20260621` is active on GPU2/GPU4 using `configs/downstream/queue_missing_pd_binary_epilepsy_scratch_20260621.txt`; current status: 1 completed, 2 running, 2 pending. | Remaining PD binary `scratch`, `partial`, `full` plus `epilepsy_binary_scratch` finish with final `metrics_final.json`. |
| Clean resume queue | Stopped stale tmux sessions `eyemae_downstream_v3_tail32` and `eyemae_downstream_v3_tail32_allowfail` after starting the explicit missing-job補跑 queue, to avoid duplicate launches when the main queue becomes idle. | Explicit補跑 status accounts for all 32 first-version jobs with no unresolved failures. |
| Known interruption | `epilepsy_binary/scratch` has a documented `rc=-15` from the retracted scratch cap experiment; it is included in the explicit missing-job補跑 queue and should resume from `checkpoint_last.pt`. | The 100-epoch fair-result job writes `metrics_final.json`; the archived epoch-29 cap result remains separate. |
| within30 checkpoint locking | `eyemae_downstream_v3_within30_lockwatch` scans once per minute and writes `within30_lockwatch_latest.json`. | All available epochs 0-29 best checkpoints are locked or represented by final best checkpoints within epoch 0-29. |
| Epoch1 convergence queue | tmux session `eyemae_downstream_v3_epoch1_20260621` is active on GPU1/GPU3 with `configs/downstream_epoch1/queue_epoch1.txt`; current status: 9 completed, 2 running, 21 pending. | All 32 epoch1 jobs finish, then summarize with `summary_epoch1.*`. |
| MCI label-corrected follow-up queue | tmux session `eyemae_downstream_v3_mci_followup_seed20260621` is waiting for `outputs/downstream_v3_fast_logs/run_20260621_missing_pd_binary_epilepsy_scratch/status.json` to become idle, then runs `configs/downstream/queue_mci_followup_seed20260621.txt` on GPU2/GPU4. | Eight jobs finish for `mci_original_only_binary` and `mci_matched_binary_random_seed20260621`, each with `scratch`, `linear_probe`, `partial`, and `full`. |
| Final summary | Refreshed `summary_first_version.*`: 32 rows and currently 4 missing final metrics. | `scripts/summarize_downstream_v3_results.py --require-complete` exits 0. |
| Active incomplete audit | Refreshed `scripts/audit_downstream_v3_goal.py --allow-incomplete --materialize-plan-artifacts`: `errors=[]`, `warnings=4`, and `metrics_final_count=28`. | Final materialized audit and hard audit pass without `--allow-incomplete`. |
| Convergence summaries | Refreshed `summary_epoch1.*`: 32 rows, 23 missing; refreshed `summary_within30.*`: 32 rows, 31 missing. | Epoch1 queue and within30 checkpoint/test artifacts finish, then rerun with completeness requirements. |
| Detox random split rerun | Regenerated at `2026-06-21 03:27 CST`: `downstream/戒毒所_random_seed20260621/`, configs `detox_binary_random_seed20260621_fast_{scratch,linear_probe,partial,full}.yaml`, and queue `configs/downstream/queue_detox_random_seed20260621_fast.txt`. tmux session `eyemae_downstream_v3_detox_random_after_mci_20260621` now waits for `run_mci_followup_seed20260621/status.json` to finish, then runs the four detox random jobs on GPU2/GPU4. | Summarize separately from the formal-split detox result after the waiting queue completes. |

Recently completed result:

| task | mode | best epoch | val subject metric | test subject AUROC | test subject AUPRC | test balanced acc | test F1 | weighted F1 | kappa | test trial AUROC | TP/TN/FP/FN |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `pd_binary` | `linear_probe` | 5 | `0.97441` | `0.96407` | n/a | `0.93326` | `0.93248` | `0.93248` | `0.86506` | `0.92255` | n/a |
| `pd_related_5class` | `scratch` | 41 | `0.94036` | `0.92753` | `0.79112` | `0.63348` | `0.69918` | `0.76396` | `0.65397` | `0.84122` | n/a |
| `mci_matched_binary` old matched-label view, provisional | `full` | 0 | `0.60900` | `0.75000` | `0.76802` | `0.72727` | `0.75000` | `0.72500` | `0.45455` | `0.65407` | 18/14/8/4 |
| `mci_matched_binary` old matched-label view, provisional | `linear_probe` | 3 | `0.60900` | `0.65083` | `0.65959` | `0.59091` | `0.50000` | `0.57692` | `0.18182` | `0.61323` | 9/17/5/13 |
| `mci_binary` old mixed view, invalid/provisional | `scratch` | 3 | `0.47619` | `0.56292` | `0.51665` | `0.53337` | `0.52101` | `0.52549` | `0.06455` | `0.54301` | 31/32/36/21 |
| `mci_matched_binary` old matched-label view, provisional | `scratch` | 0 | `0.52595` | `0.55992` | `0.61288` | `0.52273` | `0.16000` | `0.41333` | `0.04545` | `0.55965` | 2/21/1/20 |
| `mci_binary` old mixed view, invalid/provisional | `partial` | 0 | `0.38889` | `0.55501` | `0.46713` | `0.58993` | `0.55046` | `0.59324` | `0.17785` | `0.53318` | 30/41/27/22 |

Active job snapshot at 2026-06-21 05:23 CST:

| task | mode | latest step epoch / loss | latest validation epoch / train loss / subject metric / F1 |
| --- | --- | --- | --- |
| `pd_binary` | `scratch` | epoch 7 / `0.36012` | epoch 6 / `0.37709` / `0.95072` / `0.89844` |
| `pd_binary` | `linear_probe` | completed | best epoch 5 / val `0.97441`; test subject AUROC `0.96407` |
| `pd_binary` | `partial` | epoch 2 / `0.24997` | epoch 1 / `0.28734` / `0.98491` / `0.90756` |
| `pd_related_5class` epoch1 | `full` | completed | epoch 0 / `1.13325` / `0.89423` / `0.53803`; test subject macro AUROC `0.87461` |
| `pd_binary` epoch1 | `scratch` | completed | epoch 0 / `0.55861` / `0.93622` / `0.86614`; test subject AUROC `0.93382` |
| `pd_binary` epoch1 | `linear_probe` | completed | epoch 0 / `0.39926` / `0.96129` / `0.86531`; test subject AUROC `0.96171` |
| `pd_binary` epoch1 | `partial` | completed | epoch 0 / `0.35887` / `0.97854` / `0.90763`; test subject AUROC `0.97885` |
| `pd_binary` epoch1 | `full` | completed | epoch 0 / `0.29689` / `0.98891` / `0.92562`; test subject AUROC `0.98572` |
| `epilepsy_binary` epoch1 | `scratch` | completed | epoch 0 / val `0.75333`; test subject AUROC `0.79405` |
| `epilepsy_binary` epoch1 | `linear_probe` | epoch 0 / `0.51261` | epoch 0 / `0.51171` / `0.82864` / `0.71942` |
| `epilepsy_binary` epoch1 | `partial` | epoch 0 / `0.48948` | not yet reached |

Downstream performance observations:

| topic | observation | action/status |
| --- | --- | --- |
| MCI data quality | `downstream/MCI/` has severe raw-subject/file-level label conflicts: 218 raw subjects and 878 `(subject, filename basename)` keys receive both `health_label=0` and `1`, covering 66,308 / 91,433 rows. `downstream/MCI匹配后/` has 0 such conflicts. | Added `docs/newdata_v3_mci_label_audit.md`; treat current `mci_binary` results as invalid/provisional until `downstream/MCI/` is rebuilt. |
| MCI follow-up correction | Generated `MCI_original_only_no_matched` by keeping only original `MCI` `实验组/对照组` rows, and generated `MCI匹配后_random_seed20260621` by keeping matched rows only when their raw `subject` exists in original `MCI` and overwriting `health_label` from the original MCI subject label. | Both generated views have 0 raw-subject/file/trial label conflicts and 0 split overlap; the matched random view changed all 33,154 source labels, so prior matched-label results should be treated as suspect/provisional. |
| All-view identity audit | `AD`/`AD匹配后` have a small raw-subject/file label conflict; `PD相关*` aggregate views have one small label conflict but large raw-subject/file split overlap from aggregate subtype/control reuse. `MCI匹配后`, four PD subtype matched views, `偏头痛`, `戒毒所`, `戒毒所_random_seed20260621`, and `癫痫` pass raw-subject/file conflict and split-overlap checks. | Added `docs/newdata_v3_downstream_identity_audit.md`; treat MCI original view as invalid and PD aggregate/AD aggregate as high-risk/provisional until rebuilt or explicitly justified. |
| GPU utilization | Read-only audits did not find all GPU1-4 idle during normal train steps; sampled windows were mostly 93-100%. | Continue using one downstream job per GPU. |
| Utilization dips | Dips are expected at epoch/job boundaries during validation, subject-level metric computation, checkpointing, and final train/val/test prediction export. | Logged as performance behavior, not a correctness exception. |
| CPU/input pipeline | Packed mmap still does worker slicing, copying/stacking, preprocessing, patchification, padding/collation, and H2D transfer. | Future speed work can consider shard-aware batching or precomputed patch tensors. |
| Result-neutral speed config | Later jobs use `max_open_shards_per_worker=44`, `validate_offsets=false`, and `prefetch_factor=4`; `src/eyemae/finetune.py` passes `prefetch_factor` to `DataLoader`. | Jobs already running when edited keep their process-start config/code. |
| PD 5-class curve plot | Generated `outputs/downstream_v3_reports/pd_related_5class_finetune_epoch_curves_current.png` and CSV source `outputs/downstream_v3_reports/pd_related_5class_finetune_epoch_curves_current.csv`. | Plot covers completed epoch summaries available at generation time; running modes are truncated at their latest completed validation epoch. |
| Faster early stopping | Current v3 downstream configs use `early_stopping_patience=10` and `min_epochs_before_early_stopping=0`. | Applies to all 32 first-version jobs. |
| Scratch max epoch cap | The temporary `scratch max_epochs=30` cap was retracted because it could unfairly underestimate scratch convergence. | All active v3 fast configs and the final audit require `max_epochs=100` for every mode. |
| Convergence-speed reports | Final reporting must include best checkpoint within epochs 0-29 and exactly-one-epoch test results. | `summarize_downstream_v3_convergence.py` scaffolds `within30` and `epoch1`; full evaluation is still pending. |
| Detox best-test rerun | Reran `detox_binary` `scratch`, `linear_probe`, `partial`, and `full` `checkpoint_best.pt` on the test split with GPU0. | Results are recorded in `outputs/downstream_v3_fast/detox_binary/detox_sanity_check.md`; formal `full` later finished with the same epoch-7 best checkpoint. |

Detox random split details:

| item | value |
| --- | --- |
| source view | `downstream/戒毒所/` |
| target view | `downstream/戒毒所_random_seed20260621/` |
| split policy | subject-level stratified random split by `health_label`; preserves original subject counts |
| train subjects | 109 subjects: 65 control, 44 patient |
| validation subjects | 27 subjects: 16 control, 11 patient |
| test subjects | 34 subjects: 20 control, 14 patient |
| subject overlap | 0 for train/validation, train/test, and validation/test |
| output root | `outputs/downstream_v3_fast/detox_binary_random_seed20260621/` |

MCI label-corrected follow-up data:

Clarified MCI label authority: `MCI匹配后` provides rows only, not labels. The
corrected matched view keeps a row only when its raw `subject` exists in the
original `MCI` subject-label anchor after excluding `source_dataset=匹配后`; rows
without that anchor are excluded. In the current seed-20260621 generated view,
all 33,154 matched rows had an original MCI anchor, so 0 were dropped and all
33,154 source labels were overwritten.

| item | `mci_original_only_binary` | `mci_matched_binary_random_seed20260621` |
| --- | --- | --- |
| generated view | `downstream/MCI_original_only_no_matched/` | `downstream/MCI匹配后_random_seed20260621/` |
| label source | original `MCI` rows only; `source_dataset=匹配后` labels ignored | original `MCI` subject label overwrites `MCI匹配后` source label |
| split policy | preserve original MCI split after removing matched rows | subject-level stratified random split, seed `20260621` |
| rows | 58,279 | 33,154 |
| train subjects | 245 subjects: 145 control, 100 MCI | 140 subjects: 70 control, 70 MCI |
| validation subjects | 59 subjects: 37 control, 22 MCI | 34 subjects: 17 control, 17 MCI |
| test subjects | 79 subjects: 48 control, 31 MCI | 44 subjects: 22 control, 22 MCI |
| removed/unmapped rows | removed 33,154 matched rows | dropped 0 unmapped rows |
| overwritten labels | n/a | 33,154 / 33,154 rows |
| raw label conflicts | 0 raw-subject, 0 raw-file, 0 raw-trial | 0 raw-subject, 0 raw-file, 0 raw-trial |
| split overlap | 0 `ml_subject_id`, 0 raw-subject, 0 raw-file | 0 `ml_subject_id`, 0 raw-subject, 0 raw-file |
| config queue | `configs/downstream/queue_mci_followup_seed20260621.txt` | `configs/downstream/queue_mci_followup_seed20260621.txt` |
| training session | `eyemae_downstream_v3_mci_followup_seed20260621` waits for the missing-job queue, then runs on GPU2/GPU4 | same |

Monitor update at 2026-06-21 05:33 CST:

| queue/artifact | current state | next condition |
| --- | --- | --- |
| first-version summary | `summary_first_version.*` refreshed: 32 rows, 4 missing final metrics. | Finish `pd_binary` scratch/partial/full and fair `epilepsy_binary` scratch. |
| first-version audit | `errors=[]`, `warnings=4`, `metrics_final_count=28`. | Run hard audit without `--allow-incomplete` after the four missing metrics land. |
| missing-job queue | 1 completed, 2 running, 2 pending, 0 failures. | Running: `pd_binary/scratch` on GPU2 and `pd_binary/partial` on GPU4; pending: `pd_binary/full`, `epilepsy_binary/scratch`. |
| epoch1 convergence queue | 16 completed, 2 running, 14 pending, 0 failures. | Running: `migraine_binary/scratch` on GPU3 and `migraine_binary/linear_probe` on GPU1. |
| epoch1 summary | `summary_epoch1.*` refreshed: 32 rows, 16 missing outputs. | Continue until all 32 one-epoch convergence tests finish. |
| MCI follow-up | `run_mci_followup_seed20260621/status.json` not created yet. | Waiting for missing-job queue to become idle, then eight corrected MCI jobs run on GPU2/GPU4. |
| detox random rerun | `run_20260621_detox_random_seed20260621/status.json` not created yet. | Waiting for corrected MCI follow-up completion. |

GPU/process sample at 2026-06-21 05:31 CST:

| GPU | queue role | PID | sampled GPU utilization | memory |
| ---: | --- | ---: | ---: | ---: |
| 1 | epoch1 `epilepsy_binary/full` at sample time; later switched to `migraine_binary/linear_probe` | 3617440 | 97% | 39,561 / 81,920 MiB |
| 2 | missing-job `pd_binary/scratch` | 3545934 | 96% | 36,723 / 81,920 MiB |
| 3 | epoch1 `detox_binary/full` at sample time; later switched to `migraine_binary/scratch` | 3622295 | 97% | 37,481 / 81,920 MiB |
| 4 | missing-job `pd_binary/partial` | 3601741 | 98% | 12,681 / 81,920 MiB |

New epoch1 results since the previous monitor snapshot:

| task | mode | best epoch | val subject metric | test subject AUROC | test balanced acc | test F1 | weighted F1 | kappa |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `epilepsy_binary` | `linear_probe` | 0 | `0.82864` | `0.86376` | `0.80809` | `0.81319` | `0.80773` | `0.61595` |
| `epilepsy_binary` | `partial` | 0 | `0.83020` | `0.87232` | `0.81933` | `0.82222` | `0.81914` | `0.63850` |
| `epilepsy_binary` | `full` | 0 | `0.83020` | `0.88802` | `0.83625` | `0.83799` | `0.83613` | `0.67237` |
| `detox_binary` | `scratch` | 0 | `0.67614` | `0.57857` | `0.65000` | `0.56000` | `0.66834` | `0.30996` |
| `detox_binary` | `linear_probe` | 0 | `0.80114` | `0.96429` | `0.90000` | `0.87500` | `0.88317` | `0.76712` |
| `detox_binary` | `partial` | 0 | `0.80682` | `0.96786` | `0.87500` | `0.84848` | `0.85358` | `0.71186` |
| `detox_binary` | `full` | 0 | `0.90909` | `0.96429` | `0.86429` | `0.83871` | `0.85410` | `0.70588` |

Active job snapshot at 2026-06-21 05:32 CST:

| task | mode | latest step epoch / loss | latest validation epoch / train loss / subject metric / F1 |
| --- | --- | --- | --- |
| `pd_binary` | `scratch` | epoch 8 / `0.34597` | epoch 6 / `0.37709` / `0.95072` / `0.89844` |
| `pd_binary` | `partial` | epoch 4 / `0.20451` | epoch 3 / `0.22149` / `0.98839` / `0.93496` |
| `epilepsy_binary` epoch1 | `full` | completed | epoch 0 / `0.42778` / `0.83020` / `0.71942`; test subject AUROC `0.88802` |
| `detox_binary` epoch1 | `full` | completed | epoch 0 / `0.41375` / `0.90909` / `0.84615`; test subject AUROC `0.96429` |
| `migraine_binary` epoch1 | `scratch` | epoch 0 / `0.70181` | not yet reached |
| `migraine_binary` epoch1 | `linear_probe` | launched | not yet reached |

Monitor update at 2026-06-21 05:37 CST:

| queue/artifact | current state | next condition |
| --- | --- | --- |
| first-version summary | `summary_first_version.*` refreshed: 32 rows, 4 missing final metrics. | Finish `pd_binary` scratch/partial/full and fair `epilepsy_binary` scratch. |
| first-version audit | `errors=[]`, `warnings=4`, `metrics_final_count=28`. | Run hard audit without `--allow-incomplete` after the four missing metrics land. |
| missing-job queue | 1 completed, 2 running, 2 pending, 0 failures. | Running: `pd_binary/scratch` on GPU2 and `pd_binary/partial` on GPU4; pending: `pd_binary/full`, `epilepsy_binary/scratch`. |
| epoch1 convergence queue | 20 completed, 2 running, 10 pending, 0 failures. | Running: `ad_binary/scratch` on GPU3 and `ad_binary/linear_probe` on GPU1. |
| epoch1 summary | `summary_epoch1.*` refreshed: 32 rows, 12 missing outputs. | Continue until all 32 one-epoch convergence tests finish. |
| MCI follow-up | tmux session is alive and polling the missing-job status. | Starts when the missing-job queue becomes idle. |
| detox random rerun | tmux session is alive and waiting for `run_mci_followup_seed20260621/status.json`. | Starts after corrected MCI follow-up completes. |

New epoch1 results since the 05:33 monitor snapshot:

| task | mode | best epoch | val subject metric | test subject AUROC | test balanced acc | test F1 | weighted F1 | kappa |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `migraine_binary` | `scratch` | 0 | `0.46154` | `0.57750` | `0.67750` | `0.65116` | `0.62935` | `0.31591` |
| `migraine_binary` | `linear_probe` | 0 | `0.68846` | `0.82500` | `0.74625` | `0.70270` | `0.73493` | `0.46627` |
| `migraine_binary` | `partial` | 0 | `0.74231` | `0.86750` | `0.77750` | `0.73684` | `0.75872` | `0.51991` |
| `migraine_binary` | `full` | 0 | `0.87692` | `0.93000` | `0.88875` | `0.85714` | `0.87938` | `0.75212` |

Active job snapshot at 2026-06-21 05:37 CST:

| task | mode | latest step epoch / loss | latest validation epoch / train loss / subject metric / F1 |
| --- | --- | --- | --- |
| `pd_binary` | `scratch` | epoch 8 / `0.34707` | epoch 6 / `0.37709` / `0.95072` / `0.89844` |
| `pd_binary` | `partial` | epoch 4 / `0.20117` | epoch 3 / `0.22149` / `0.98839` / `0.93496` |
| `migraine_binary` epoch1 | `scratch` | completed | epoch 0 / `0.69665` / `0.46154` / `0.42424`; test subject AUROC `0.57750` |
| `migraine_binary` epoch1 | `linear_probe` | completed | epoch 0 / `0.59650` / `0.68846` / `0.57143`; test subject AUROC `0.82500` |
| `migraine_binary` epoch1 | `partial` | completed | epoch 0 / `0.56432` / `0.74231` / `0.64516`; test subject AUROC `0.86750` |
| `migraine_binary` epoch1 | `full` | completed | epoch 0 / `0.47160` / `0.87692` / `0.70968`; test subject AUROC `0.93000` |
| `ad_binary` epoch1 | `scratch` | launched | not yet reached |
| `ad_binary` epoch1 | `linear_probe` | launched | not yet reached |

GPU/process sample at 2026-06-21 05:38 CST:

| GPU | queue role | PID | sampled GPU utilization | memory |
| ---: | --- | ---: | ---: | ---: |
| 1 | epoch1 `ad_binary/linear_probe` | 3627767 | 97% | 3,141 / 81,920 MiB |
| 2 | missing-job `pd_binary/scratch` | 3545934 | 98% | 36,723 / 81,920 MiB |
| 3 | epoch1 `ad_binary/scratch` | 3627280 | 95% | 33,441 / 81,920 MiB |
| 4 | missing-job `pd_binary/partial` | 3601741 | 97% | 12,681 / 81,920 MiB |

Monitor update at 2026-06-21 05:40 CST:

| queue/artifact | current state | next condition |
| --- | --- | --- |
| first-version summary | `summary_first_version.*` refreshed: 32 rows, 4 missing final metrics. | Finish `pd_binary` scratch/partial/full and fair `epilepsy_binary` scratch. |
| first-version audit | `errors=[]`, `warnings=4`, `metrics_final_count=28`. | Run hard audit without `--allow-incomplete` after the four missing metrics land. |
| missing-job queue | 1 completed, 2 running, 2 pending, 0 failures. | Running: `pd_binary/scratch` on GPU2 and `pd_binary/partial` on GPU4; pending: `pd_binary/full`, `epilepsy_binary/scratch`. |
| epoch1 convergence queue | 22 completed, 2 running, 8 pending, 0 failures. | Running: `ad_binary/partial` on GPU1 and `ad_binary/full` on GPU3. |
| epoch1 summary | `summary_epoch1.*` refreshed: 32 rows, 10 missing outputs. | Continue until all 32 one-epoch convergence tests finish. |
| MCI follow-up | tmux session is alive and polling the missing-job status. | Starts when the missing-job queue becomes idle. |
| detox random rerun | tmux session is alive and waiting for corrected MCI follow-up. | Starts after corrected MCI follow-up completes. |

New epoch1 results since the 05:37 monitor snapshot:

| task | mode | best epoch | val subject metric | test subject AUROC | test balanced acc | test F1 | weighted F1 | kappa |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ad_binary` | `scratch` | 0 | `0.95296` | `0.93761` | `0.81923` | `0.85000` | `0.82526` | `0.64341` |
| `ad_binary` | `linear_probe` | 0 | `0.92742` | `0.93846` | `0.88590` | `0.89474` | `0.88440` | `0.76590` |

Active job snapshot at 2026-06-21 05:40 CST:

| task | mode | latest step epoch / loss | latest validation epoch / train loss / subject metric / F1 |
| --- | --- | --- | --- |
| `pd_binary` | `scratch` | epoch 9 / `0.33233` | epoch 6 / `0.37709` / `0.95072` / `0.89844` |
| `pd_binary` | `partial` | epoch 5 / `0.18407` | epoch 3 / `0.22149` / `0.98839` / `0.93496` |
| `ad_binary` epoch1 | `scratch` | completed | epoch 0 / `0.60423` / `0.95296` / `0.87500`; test subject AUROC `0.93761` |
| `ad_binary` epoch1 | `linear_probe` | completed | epoch 0 / `0.42894` / `0.92742` / `0.87719`; test subject AUROC `0.93846` |
| `ad_binary` epoch1 | `partial` | epoch 0 / `0.43436` | not yet reached |
| `ad_binary` epoch1 | `full` | launched | not yet reached |

Monitor update at 2026-06-21 05:42 CST:

| queue/artifact | current state | next condition |
| --- | --- | --- |
| first-version summary | `summary_first_version.*` refreshed: 32 rows, 4 missing final metrics. | Finish `pd_binary` scratch/partial/full and fair `epilepsy_binary` scratch. |
| missing-job queue | 1 completed, 2 running, 2 pending, 0 failures. | Running: `pd_binary/scratch` on GPU2 and `pd_binary/partial` on GPU4. |
| epoch1 convergence queue | 23 completed, 2 running, 7 pending, 0 failures. | Running: `ad_binary/full` on GPU3 and `mci_binary/scratch` on GPU1. |
| epoch1 summary | `summary_epoch1.*` refreshed: 32 rows, 9 missing outputs. | Continue until all 32 one-epoch convergence tests finish. |
| MCI follow-up | status file not created yet; tmux waiting chain is alive. | Starts when the missing-job queue becomes idle. |

New epoch1 result since the 05:40 monitor snapshot:

| task | mode | best epoch | val subject metric | test subject AUROC | test balanced acc | test F1 | weighted F1 | kappa |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ad_binary` | `partial` | 0 | `0.94892` | `0.95043` | `0.88590` | `0.89474` | `0.88440` | `0.76590` |

Active job snapshot at 2026-06-21 05:42 CST:

| task | mode | latest step epoch / loss | latest validation epoch / train loss / subject metric / F1 |
| --- | --- | --- | --- |
| `pd_binary` | `scratch` | epoch 9 / `0.33233` | epoch 6 / `0.37709` / `0.95072` / `0.89844` |
| `pd_binary` | `partial` | epoch 5 / `0.18407` | epoch 3 / `0.22149` / `0.98839` / `0.93496` |
| `ad_binary` epoch1 | `partial` | completed | epoch 0 / `0.40067` / `0.94892` / `0.91228`; test subject AUROC `0.95043` |
| `ad_binary` epoch1 | `full` | epoch 0 / `0.34549` | not yet reached |
| `mci_binary` epoch1 | `scratch` | epoch 0 / `0.70883` | not yet reached |

Monitor update at 2026-06-21 05:43 CST:

| queue/artifact | current state | next condition |
| --- | --- | --- |
| first-version summary | `summary_first_version.*` refreshed: 32 rows, 4 missing final metrics. | Finish `pd_binary` scratch/partial/full and fair `epilepsy_binary` scratch. |
| missing-job queue | 1 completed, 2 running, 2 pending, 0 failures. | Running: `pd_binary/scratch` on GPU2 and `pd_binary/partial` on GPU4. |
| epoch1 convergence queue | 24 completed, 2 running, 6 pending, 0 failures. | Running: `mci_binary/scratch` on GPU1 and `mci_binary/linear_probe` on GPU3. |
| epoch1 summary | `summary_epoch1.*` refreshed: 32 rows, 8 missing outputs. | Continue until all 32 one-epoch convergence tests finish. |
| MCI follow-up | status file not created yet; tmux waiting chain is alive. | Starts when the missing-job queue becomes idle. |

New epoch1 result since the 05:42 monitor snapshot:

| task | mode | best epoch | val subject metric | test subject AUROC | test balanced acc | test F1 | weighted F1 | kappa |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ad_binary` | `full` | 0 | `0.96909` | `0.96923` | `0.92821` | `0.93506` | `0.92766` | `0.85313` |

Active job snapshot at 2026-06-21 05:43 CST:

| task | mode | latest step epoch / loss | latest validation epoch / train loss / subject metric / F1 |
| --- | --- | --- | --- |
| `pd_binary` | `scratch` | epoch 9 / `0.33233` | epoch 6 / `0.37709` / `0.95072` / `0.89844` |
| `pd_binary` | `partial` | epoch 5 / `0.18407` | epoch 3 / `0.22149` / `0.98839` / `0.93496` |
| `ad_binary` epoch1 | `full` | completed | epoch 0 / validation `0.96909`; test subject AUROC `0.96923` |
| `mci_binary` epoch1 | `scratch` | epoch 0 / `0.70883` | not yet reached |
| `mci_binary` epoch1 | `linear_probe` | launched | not yet reached |

Monitor update at 2026-06-21 05:45 CST:

| queue/artifact | current state | next condition |
| --- | --- | --- |
| first-version summary | `summary_first_version.*` refreshed: 32 rows, 4 missing final metrics. | Finish `pd_binary` scratch/partial/full and fair `epilepsy_binary` scratch. |
| first-version audit | `errors=[]`, `warnings=4`, `metrics_final_count=28`. | Run hard audit without `--allow-incomplete` after the four missing metrics land. |
| missing-job queue | 1 completed, 2 running, 2 pending, 0 failures. | Running: `pd_binary/scratch` on GPU2 and `pd_binary/partial` on GPU4; pending: `pd_binary/full`, `epilepsy_binary/scratch`. |
| epoch1 convergence queue | 24 completed, 2 running, 6 pending, 0 failures. | Running: `mci_binary/scratch` on GPU1 and `mci_binary/linear_probe` on GPU3. |
| epoch1 summary | `summary_epoch1.*` refreshed: 32 rows, 8 missing outputs. | Continue until all 32 one-epoch convergence tests finish. |
| MCI follow-up | tmux session is alive and polling missing-job status `1 completed / 2 running / 2 pending`. | Starts when the missing-job queue becomes idle. |
| detox random rerun | tmux session is alive and waiting for corrected MCI follow-up status. | Starts after corrected MCI follow-up completes. |

GPU/process sample at 2026-06-21 05:45 CST:

| GPU | queue role | PID | sampled GPU utilization | memory |
| ---: | --- | ---: | ---: | ---: |
| 1 | epoch1 `mci_binary/scratch` | 3631792 | 98% | 28,929 / 81,920 MiB |
| 2 | missing-job `pd_binary/scratch` | 3545934 | 99% | 36,723 / 81,920 MiB |
| 3 | epoch1 `mci_binary/linear_probe` | 3633425 | 96% | 2,887 / 81,920 MiB |
| 4 | missing-job `pd_binary/partial` | 3601741 | 98% | 12,681 / 81,920 MiB |

Active job snapshot at 2026-06-21 05:45 CST:

| task | mode | latest step epoch / loss | latest validation epoch / train loss / subject metric / F1 |
| --- | --- | --- | --- |
| `pd_binary` | `scratch` | epoch 9 / `0.33030` | epoch 6 / `0.37709` / `0.95072` / `0.89844` |
| `pd_binary` | `partial` | epoch 6 / `0.17099` | epoch 3 / `0.22149` / `0.98839` / `0.93496` |
| `mci_binary` epoch1 | `scratch` | epoch 0 / `0.69370` | not yet reached |
| `mci_binary` epoch1 | `linear_probe` | epoch 0 / `0.68682` | not yet reached |

Monitor update at 2026-06-21 05:47 CST:

| queue/artifact | current state | next condition |
| --- | --- | --- |
| first-version summary | `summary_first_version.*` refreshed: 32 rows, 4 missing final metrics. | Finish `pd_binary` scratch/partial/full and fair `epilepsy_binary` scratch. |
| missing-job queue | 1 completed, 2 running, 2 pending, 0 failures. | Running: `pd_binary/scratch` on GPU2 and `pd_binary/partial` on GPU4. |
| epoch1 convergence queue | 26 completed, 2 running, 4 pending, 0 failures. | Running: `mci_binary/partial` on GPU3 and `mci_binary/full` on GPU1. |
| epoch1 summary | `summary_epoch1.*` refreshed: 32 rows, 6 missing outputs. | Continue until all 32 one-epoch convergence tests finish. |
| MCI follow-up | corrected MCI queue has not started yet. | Starts when the missing-job queue becomes idle; this is separate from old-view `mci_binary` epoch1. |

New epoch1 results since the 05:45 monitor snapshot:

| task | mode | best epoch | val subject metric | test subject AUROC | test balanced acc | test F1 | weighted F1 | kappa |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `mci_binary` old view | `scratch` | 0 | `0.50705` | `0.55557` | `0.55317` | `0.52632` | `0.55188` | `0.10398` |
| `mci_binary` old view | `linear_probe` | 0 | `0.41975` | `0.54907` | `0.56109` | `0.50943` | `0.56752` | `0.12162` |

Active job snapshot at 2026-06-21 05:47 CST:

| task | mode | latest step epoch / loss | latest validation epoch / train loss / subject metric / F1 |
| --- | --- | --- | --- |
| `pd_binary` | `scratch` | epoch 9 / `0.33030` | epoch 6 / `0.37709` / `0.95072` / `0.89844` |
| `pd_binary` | `partial` | epoch 6 / `0.17099` | epoch 3 / `0.22149` / `0.98839` / `0.93496` |
| `mci_binary` epoch1 old view | `scratch` | completed | epoch 0 / validation `0.50705`; test subject AUROC `0.55557` |
| `mci_binary` epoch1 old view | `linear_probe` | completed | epoch 0 / validation `0.41975`; test subject AUROC `0.54907` |
| `mci_binary` epoch1 old view | `partial` | launched | not yet reached |
| `mci_binary` epoch1 old view | `full` | launched | not yet reached |

Monitor update at 2026-06-21 05:49 CST:

| queue/artifact | current state | next condition |
| --- | --- | --- |
| first-version summary | `summary_first_version.*` refreshed: 32 rows, 4 missing final metrics. | Finish `pd_binary` scratch/partial/full and fair `epilepsy_binary` scratch. |
| first-version audit | `errors=[]`, `warnings=4`, `metrics_final_count=28`. | Run hard audit without `--allow-incomplete` after the four missing metrics land. |
| missing-job queue | 1 completed, 2 running, 2 pending, 0 failures. | Running: `pd_binary/scratch` on GPU2 and `pd_binary/partial` on GPU4. |
| epoch1 convergence queue | 26 completed, 2 running, 4 pending, 0 failures. | Running old-view `mci_binary/partial` on GPU3 and old-view `mci_binary/full` on GPU1. |
| epoch1 summary | `summary_epoch1.*` refreshed: 32 rows, 6 missing outputs. | Continue until all 32 one-epoch convergence tests finish. |
| MCI follow-up | corrected MCI queue has not started yet. | Starts when the missing-job queue becomes idle. |
| detox random rerun | detox random queue has not started yet. | Starts after corrected MCI follow-up completes. |

GPU/process sample at 2026-06-21 05:49 CST:

| GPU | queue role | PID | sampled GPU utilization | memory |
| ---: | --- | ---: | ---: | ---: |
| 1 | epoch1 old-view `mci_binary/full` | 3638055 | 99% | 28,929 / 81,920 MiB |
| 2 | missing-job `pd_binary/scratch` | 3545934 | 98% | 36,723 / 81,920 MiB |
| 3 | epoch1 old-view `mci_binary/partial` | 3637603 | 94% | 10,115 / 81,920 MiB |
| 4 | missing-job `pd_binary/partial` | 3601741 | 97% | 12,681 / 81,920 MiB |

Active job snapshot at 2026-06-21 05:49 CST:

| task | mode | latest step epoch / loss | latest validation epoch / train loss / subject metric / F1 |
| --- | --- | --- | --- |
| `pd_binary` | `scratch` | epoch 10 / `0.30956` | epoch 6 / `0.37709` / `0.95072` / `0.89844` |
| `pd_binary` | `partial` | epoch 7 / `0.15848` | epoch 3 / `0.22149` / `0.98839` / `0.93496` |
| `mci_binary` epoch1 old view | `partial` | epoch 0 / `0.68463` | not yet reached |
| `mci_binary` epoch1 old view | `full` | epoch 0 / `0.68684` | not yet reached |

Monitor update at 2026-06-21 05:51 CST:

| queue/artifact | current state | next condition |
| --- | --- | --- |
| first-version summary | `summary_first_version.*` refreshed: 32 rows, 4 missing final metrics. | Finish `pd_binary` scratch/partial/full and fair `epilepsy_binary` scratch. |
| first-version audit | `errors=[]`, `warnings=4`, `metrics_final_count=28`. | Run hard audit without `--allow-incomplete` after the four missing metrics land. |
| missing-job queue | 1 completed, 2 running, 2 pending, 0 failures. | Running: `pd_binary/scratch` on GPU2 and `pd_binary/partial` on GPU4. |
| epoch1 convergence queue | 27 completed, 2 running, 3 pending, 0 failures. | Running old-view `mci_binary/full` on GPU1 and old-label `mci_matched_binary/scratch` on GPU3. |
| epoch1 summary | `summary_epoch1.*` refreshed: 32 rows, 5 missing outputs. | Continue until all 32 one-epoch convergence tests finish. |
| MCI follow-up | corrected MCI queue has not started yet. | Starts when the missing-job queue becomes idle. |

New epoch1 result since the 05:49 monitor snapshot:

| task | mode | best epoch | val subject metric | test subject AUROC | test balanced acc | test F1 | weighted F1 | kappa |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `mci_binary` old view | `partial` | 0 | `0.39021` | `0.55416` | `0.58032` | `0.53704` | `0.58474` | `0.15919` |

Active job snapshot at 2026-06-21 05:51 CST:

| task | mode | latest step epoch / loss | latest validation epoch / train loss / subject metric / F1 |
| --- | --- | --- | --- |
| `pd_binary` | `scratch` | epoch 10 / `0.31457` | epoch 6 / `0.37709` / `0.95072` / `0.89844` |
| `pd_binary` | `partial` | epoch 7 / `0.15759` | epoch 7 / `0.15759` / `0.99121` / `0.94862` |
| `mci_binary` epoch1 old view | `partial` | completed | epoch 0 / validation `0.39021`; test subject AUROC `0.55416` |
| `mci_binary` epoch1 old view | `full` | epoch 0 / `0.67212` | not yet reached |
| `mci_matched_binary` epoch1 old-label view | `scratch` | epoch 0 / `0.69356` | not yet reached |

Monitor update at 2026-06-21 05:56 CST:

| queue/artifact | current state | next condition |
| --- | --- | --- |
| corrected MCI rule | `MCI匹配后` rows are sample rows only, not label authority. Keep a matched row only if raw `subject` exists in the original `MCI` anchor after excluding `source_dataset=匹配后`; overwrite `health_label` from that original anchor. | Use only `mci_original_only_binary` and `mci_matched_binary_random_seed20260621` for formal MCI follow-up interpretation. |
| corrected MCI data audit | `MCI_original_only_no_matched`: 58,279 rows, 383 subjects, 0 anchor mismatches. `MCI匹配后_random_seed20260621`: 33,154 rows, 218 subjects, `outside_anchor=0`, `label_mismatch_vs_anchor=0`; all source matched labels are ignored. | Corrected MCI queue is ready but not launched yet. |
| invalid/provisional MCI views | Old `mci_binary` mixes original and matched copies; old `mci_matched_binary` uses crossed matched-source labels. | Existing old-view epoch1 rows below are convergence diagnostics only, not official MCI disease results. |
| missing-job queue | 1 completed, 2 running, 2 pending, 0 failures. | Corrected MCI follow-up starts after this queue becomes idle. |
| epoch1 convergence queue | 32 completed, 0 running, 0 pending, 0 failures. | Full epoch1 convergence queue is complete. |
| epoch1 summary | `summary_epoch1.*` refreshed: 32 rows, 0 missing outputs. | Keep old MCI rows separated from corrected follow-up. |

Old/provisional MCI epoch1 results currently present in `summary_epoch1.csv`:

| task | mode | checkpoint epoch | val main metric | test main metric | test balanced acc | test F1 | weighted F1 | kappa | status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `mci_binary` old mixed view | `scratch` | 0 | `0.50705` | `0.55557` | `0.55317` | `0.52632` | `0.55188` | `0.10398` | provisional/invalid labels |
| `mci_binary` old mixed view | `linear_probe` | 0 | `0.41975` | `0.54907` | `0.56109` | `0.50943` | `0.56752` | `0.12162` | provisional/invalid labels |
| `mci_binary` old mixed view | `partial` | 0 | `0.39021` | `0.55416` | `0.58032` | `0.53704` | `0.58474` | `0.15919` | provisional/invalid labels |
| `mci_binary` old mixed view | `full` | 0 | `0.36155` | `0.48402` | `0.52941` | `0.48148` | `0.53490` | `0.05830` | provisional/invalid labels |
| `mci_matched_binary` old matched-label view | `scratch` | 0 | `0.53287` | `0.59711` | `0.52273` | `0.08696` | `0.38194` | `0.04545` | provisional/crossed labels |
| `mci_matched_binary` old matched-label view | `linear_probe` | 0 | `0.59862` | `0.69008` | `0.61364` | `0.65306` | `0.60858` | `0.22727` | provisional/crossed labels |
| `mci_matched_binary` old matched-label view | `partial` | 0 | `0.62630` | `0.72727` | `0.61364` | `0.62222` | `0.61344` | `0.22727` | provisional/crossed labels |
| `mci_matched_binary` old matched-label view | `full` | 0 | `0.60900` | `0.75000` | `0.72727` | `0.75000` | `0.72500` | `0.45455` | provisional/crossed labels |

Monitor update at 2026-06-21 06:00 CST:

| queue/artifact | current state | next condition |
| --- | --- | --- |
| first-version summary | `summary_first_version.*` refreshed: 32 rows, 4 missing final metrics. | Finish `pd_binary` scratch/partial/full and fair `epilepsy_binary` scratch. |
| first-version audit | `errors=[]`, `warnings=4`, `metrics_final_count=28`. | Run hard audit without `--allow-incomplete` after the four missing metrics land. |
| missing-job queue | 1 completed, 2 running, 2 pending, 0 failures. | Running: `pd_binary/scratch` on GPU2 and `pd_binary/partial` on GPU4; pending: `pd_binary/full`, `epilepsy_binary/scratch`. |
| corrected MCI follow-up | status file still absent; waiting tmux has not launched corrected jobs. | Starts when the missing-job queue becomes idle. |
| detox random rerun | status file still absent. | Starts after corrected MCI follow-up completes. |
| epoch1 convergence | complete: 32 completed, 0 missing outputs. | No action except keeping old MCI rows marked provisional. |

Active job snapshot at 2026-06-21 06:00 CST:

| task | mode | latest step epoch / loss | latest validation epoch / train loss / subject metric / F1 |
| --- | --- | --- | --- |
| `pd_binary` | `scratch` | epoch 11 / `0.29962` | epoch 10 / `0.31532` / `0.96404` / `0.90347` |
| `pd_binary` | `partial` | epoch 9 / `0.13866` | epoch 8 / `0.14673` / `0.99186` / `0.94862` |

Runtime check at 2026-06-21 06:00 CST:

| item | observation | interpretation |
| --- | --- | --- |
| GPU2 | PID 3545934, 36,723 MiB, sampled utilization 98%. | `pd_binary/scratch` is actively training. |
| GPU4 | PID 3601741, 12,681 MiB, sampled utilization 3%. | `pd_binary/partial` process exists; low sample likely caught an evaluation/data-loading interval because the log is still advancing. |
| GPU1/GPU3 | no active downstream job after epoch1 completion. | Available until the chained queues need them; current missing-job queue is configured on GPU2/GPU4. |
| MCI follow-up tmux | repeated `WAIT completed=1 running=2 pending=2 failures=0`. | Correctly waiting for the missing-job queue to finish. |
| detox random tmux | repeated wait for `run_mci_followup_seed20260621/status.json`. | Correctly waiting for corrected MCI follow-up to launch and finish. |
| within30 summary | `summary_within30.*` remains at 32 rows with 31 missing outputs. | Additional within-30 checkpoint testing still needs completion after stable first-version checkpoints are available. |

Monitor update at 2026-06-21 06:06 CST:

| item | current state | note |
| --- | --- | --- |
| missing-job queue | still 1 completed, 2 running, 2 pending, 0 failures. | Running `pd_binary/scratch` and `pd_binary/partial`; pending `pd_binary/full` and `epilepsy_binary/scratch`. |
| active `pd_binary/scratch` | latest epoch 11 step 26000 train loss `0.29900`; latest validation remains epoch 10 AUROC `0.96404`, F1 `0.90347`. | No final metrics yet. |
| active `pd_binary/partial` | latest epoch 10 step 22900 train loss `0.12992`; latest validation epoch 9 AUROC `0.99055`, F1 `0.96124`. | No final metrics yet. |
| first-version summary/audit | `summary_first_version.*`: 32 rows, 4 missing; enhanced audit: `errors=[]`, `warnings=4`, `metrics_final_count=28`. | Warnings are exactly the four missing final metrics. |
| `split_summary` config provenance | Fixed 52 downstream/epoch1 YAML configs whose `split.split_summary` still pointed to `pretrain/pretrain_split_summary.json`; all now match the parent view of `data.train_index`. | Training code audits downstream splits from `train_index.parent/split_summary.json`, so completed metrics were not redirected to pretrain split files; this is a reproducibility/config-conformance fix. |
| audit hardening | Added an explicit check in `scripts/audit_downstream_v3_goal.py` that `split.split_summary == Path(train_index).parent / "split_summary.json"`. | Future goal audits will catch this mismatch automatically. |

Active job snapshot at 2026-06-21 06:07 CST:

| task | mode | latest step epoch / loss | latest validation epoch / train loss / subject metric / F1 |
| --- | --- | --- | --- |
| `pd_binary` | `scratch` | epoch 12 / `0.28347` | epoch 11 / `0.30043` / `0.96923` / `0.91603` |
| `pd_binary` | `partial` | epoch 11 / `0.12234` | epoch 10 / `0.12890` / `0.98891` / `0.94400` |

Monitor update at 2026-06-21 06:09 CST:

| queue/artifact | current state | next condition |
| --- | --- | --- |
| first-version summary | `summary_first_version.*` refreshed: 32 rows, 4 missing final metrics. | Finish `pd_binary` scratch/partial/full and fair `epilepsy_binary` scratch. |
| enhanced first-version audit | `errors=[]`, `warnings=4`, `metrics_final_count=28`; the `split_summary` consistency check reports no errors. | Run hard audit without `--allow-incomplete` after the four missing metrics land. |
| missing-job queue | 1 completed, 2 running, 2 pending, 0 failures. | Running `pd_binary/scratch` and `pd_binary/partial`; pending `pd_binary/full`, `epilepsy_binary/scratch`. |
| corrected MCI follow-up | tmux is still repeating `WAIT completed=1 running=2 pending=2 failures=0`. | Starts after the missing-job queue becomes idle. |
| detox random rerun | tmux is still waiting for `run_mci_followup_seed20260621/status.json`. | Starts after corrected MCI follow-up completes. |
| GPU sample | GPU2 PID 3545934 at 98%; GPU4 PID 3601741 at 92%. | Both active PD binary jobs are genuinely running. |

Active job snapshot at 2026-06-21 06:08 CST:

| task | mode | latest step epoch / loss | latest validation epoch / train loss / subject metric / F1 |
| --- | --- | --- | --- |
| `pd_binary` | `scratch` | epoch 12 / `0.28537` | epoch 11 / `0.30043` / `0.96923` / `0.91603` |
| `pd_binary` | `partial` | epoch 11 / `0.12092` | epoch 10 / `0.12890` / `0.98891` / `0.94400` |

Monitor update at 2026-06-21 06:11 CST:

| queue/artifact | current state | next condition |
| --- | --- | --- |
| first-version summary | `summary_first_version.*` refreshed: 32 rows, 4 missing final metrics. | Finish `pd_binary` scratch/partial/full and fair `epilepsy_binary` scratch. |
| enhanced first-version audit | `errors=[]`, `warnings=4`, `metrics_final_count=28`. | Run hard audit without `--allow-incomplete` after the four missing metrics land. |
| missing-job queue | 1 completed, 2 running, 2 pending, 0 failures. | Running `pd_binary/scratch` and `pd_binary/partial`; pending `pd_binary/full`, `epilepsy_binary/scratch`. |
| corrected MCI follow-up | tmux is still repeating `WAIT completed=1 running=2 pending=2 failures=0`. | Starts after the missing-job queue becomes idle. |
| detox random rerun | tmux is still waiting for `run_mci_followup_seed20260621/status.json`. | Starts after corrected MCI follow-up completes. |
| GPU sample | GPU2 PID 3545934 at 98%; GPU4 PID 3601741 at 97%. | Both active PD binary jobs are still running. |

Active job snapshot at 2026-06-21 06:10 CST:

| task | mode | latest step epoch / loss | latest validation epoch / train loss / subject metric / F1 |
| --- | --- | --- | --- |
| `pd_binary` | `scratch` | epoch 12 / `0.28588` | epoch 11 / `0.30043` / `0.96923` / `0.91603` |
| `pd_binary` | `partial` | epoch 11 / `0.12165` | epoch 10 / `0.12890` / `0.98891` / `0.94400` |

Monitor update at 2026-06-21 06:13 CST:

| queue/artifact | current state | next condition |
| --- | --- | --- |
| first-version summary | `summary_first_version.*` refreshed: 32 rows, 4 missing final metrics. | Finish `pd_binary` scratch/partial/full and fair `epilepsy_binary` scratch. |
| enhanced first-version audit | `errors=[]`, `warnings=4`, `metrics_final_count=28`. | Run hard audit without `--allow-incomplete` after the four missing metrics land. |
| missing-job queue | 1 completed, 2 running, 2 pending, 0 failures. | Running `pd_binary/scratch` and `pd_binary/partial`; pending `pd_binary/full`, `epilepsy_binary/scratch`. |
| corrected MCI follow-up | tmux is still repeating `WAIT completed=1 running=2 pending=2 failures=0`. | Starts after the missing-job queue becomes idle. |
| detox random rerun | tmux is still waiting for `run_mci_followup_seed20260621/status.json`. | Starts after corrected MCI follow-up completes. |
| GPU sample | GPU2 PID 3545934 at 98%; GPU4 PID 3601741 at 97%. | Both active PD binary jobs are still running. |

Active job snapshot at 2026-06-21 06:13 CST:

| task | mode | latest step epoch / loss | latest validation epoch / train loss / subject metric / F1 |
| --- | --- | --- | --- |
| `pd_binary` | `scratch` | epoch 12 / `0.28502` | epoch 11 / `0.30043` / `0.96923` / `0.91603` |
| `pd_binary` | `partial` | epoch 12 / `0.11867` | epoch 11 / `0.12232` / `0.99219` / `0.96525` |

Monitor update at 2026-06-21 06:16 CST:

| queue/artifact | current state | next condition |
| --- | --- | --- |
| first-version summary | `summary_first_version.*` refreshed: 32 rows, 4 missing final metrics. | Finish `pd_binary` scratch/partial/full and fair `epilepsy_binary` scratch. |
| enhanced first-version audit | `errors=[]`, `warnings=4`, `metrics_final_count=28`. | Run hard audit without `--allow-incomplete` after the four missing metrics land. |
| missing-job queue | 1 completed, 2 running, 2 pending, 0 failures. | Running `pd_binary/scratch` and `pd_binary/partial`; pending `pd_binary/full`, `epilepsy_binary/scratch`. |
| corrected MCI follow-up | tmux is still repeating `WAIT completed=1 running=2 pending=2 failures=0`. | Starts after the missing-job queue becomes idle. |
| detox random rerun | tmux is still waiting for `run_mci_followup_seed20260621/status.json`. | Starts after corrected MCI follow-up completes. |
| GPU sample | GPU2 PID 3545934 at 99%; GPU4 PID 3601741 at 95%. | Both active PD binary jobs are still running. |

Active job snapshot at 2026-06-21 06:15 CST:

| task | mode | latest step epoch / loss | latest validation epoch / train loss / subject metric / F1 |
| --- | --- | --- | --- |
| `pd_binary` | `scratch` | epoch 12 / `0.28634` | epoch 11 / `0.30043` / `0.96923` / `0.91603` |
| `pd_binary` | `partial` | epoch 12 / `0.11460` | epoch 11 / `0.12232` / `0.99219` / `0.96525` |

Monitor update at 2026-06-21 06:18 CST:

| queue/artifact | current state | next condition |
| --- | --- | --- |
| first-version summary | `summary_first_version.*` refreshed: 32 rows, 4 missing final metrics. | Finish `pd_binary` scratch/partial/full and fair `epilepsy_binary` scratch. |
| enhanced first-version audit | `errors=[]`, `warnings=4`, `metrics_final_count=28`. | Run hard audit without `--allow-incomplete` after the four missing metrics land. |
| missing-job queue | 1 completed, 2 running, 2 pending, 0 failures. | Running `pd_binary/scratch` and `pd_binary/partial`; pending `pd_binary/full`, `epilepsy_binary/scratch`. |
| corrected MCI follow-up | tmux is still repeating `WAIT completed=1 running=2 pending=2 failures=0`. | Starts after the missing-job queue becomes idle. |
| detox random rerun | tmux is still waiting for `run_mci_followup_seed20260621/status.json`. | Starts after corrected MCI follow-up completes. |
| GPU sample | GPU2 PID 3545934 at 98%; GPU4 PID 3601741 at 97%. | Both active PD binary jobs are still running. |

Active job snapshot at 2026-06-21 06:17 CST:

| task | mode | latest step epoch / loss | latest validation epoch / train loss / subject metric / F1 |
| --- | --- | --- | --- |
| `pd_binary` | `scratch` | epoch 13 / `0.27272` | epoch 12 / `0.28596` / `0.96831` / `0.91667` |
| `pd_binary` | `partial` | epoch 13 / `0.12267` | epoch 12 / `0.11404` / `0.99304` / `0.96525` |

Monitor update at 2026-06-21 06:22 CST:

| queue/artifact | current state | next condition |
| --- | --- | --- |
| first-version summary | `summary_first_version.*` refreshed: 32 rows, 4 missing final metrics. | Finish `pd_binary` scratch/partial/full and fair `epilepsy_binary` scratch. |
| enhanced first-version audit | `errors=[]`, `warnings=4`, `metrics_final_count=28`. | Run hard audit without `--allow-incomplete` after the four missing metrics land. |
| missing-job queue | 1 completed, 2 running, 2 pending, 0 failures. | Running `pd_binary/scratch` and `pd_binary/partial`; pending `pd_binary/full`, `epilepsy_binary/scratch`. |
| corrected MCI follow-up | tmux is still repeating `WAIT completed=1 running=2 pending=2 failures=0`. | Starts after the missing-job queue becomes idle. |
| detox random rerun | tmux is still waiting for `run_mci_followup_seed20260621/status.json`. | Starts after corrected MCI follow-up completes. |
| GPU sample | GPU2 PID 3545934 at 98%; GPU4 PID 3601741 at 97%. | Both active PD binary jobs are still running. |

Active job snapshot at 2026-06-21 06:21 CST:

| task | mode | latest step epoch / loss | latest validation epoch / train loss / subject metric / F1 |
| --- | --- | --- | --- |
| `pd_binary` | `scratch` | epoch 13 / `0.27149` | epoch 12 / `0.28596` / `0.96831` / `0.91667` |
| `pd_binary` | `partial` | epoch 13 / `0.10864` | epoch 12 / `0.11404` / `0.99304` / `0.96525` |

Monitor update at 2026-06-21 06:24 CST:

| queue/artifact | current state | next condition |
| --- | --- | --- |
| first-version summary | `summary_first_version.*` refreshed: 32 rows, 4 missing final metrics. | Finish `pd_binary` scratch/partial/full and fair `epilepsy_binary` scratch. |
| enhanced first-version audit | `errors=[]`, `warnings=4`, `metrics_final_count=28`. | Run hard audit without `--allow-incomplete` after the four missing metrics land. |
| missing-job queue | 1 completed, 2 running, 2 pending, 0 failures. | Running `pd_binary/scratch` and `pd_binary/partial`; pending `pd_binary/full`, `epilepsy_binary/scratch`. |
| corrected MCI follow-up | tmux is still repeating `WAIT completed=1 running=2 pending=2 failures=0`. | Starts after the missing-job queue becomes idle. |
| detox random rerun | tmux is still waiting for `run_mci_followup_seed20260621/status.json`. | Starts after corrected MCI follow-up completes. |
| GPU sample | GPU2 PID 3545934 at 96%; GPU4 PID 3601741 at 97%. | Both active PD binary jobs are still running. |

Active job snapshot at 2026-06-21 06:23 CST:

| task | mode | latest step epoch / loss | latest validation epoch / train loss / subject metric / F1 |
| --- | --- | --- | --- |
| `pd_binary` | `scratch` | epoch 13 / `0.27098` | epoch 12 / `0.28596` / `0.96831` / `0.91667` |
| `pd_binary` | `partial` | epoch 14 / `0.09777` | epoch 13 / `0.10863` / `0.99180` / `0.93600` |

Monitor update at 2026-06-21 06:26 CST:

| queue/artifact | current state | next condition |
| --- | --- | --- |
| first-version summary | `summary_first_version.*` refreshed: 32 rows, 4 missing final metrics. | Finish `pd_binary` scratch/partial/full and fair `epilepsy_binary` scratch. |
| enhanced first-version audit | `errors=[]`, `warnings=4`, `metrics_final_count=28`. | Run hard audit without `--allow-incomplete` after the four missing metrics land. |
| missing-job queue | 1 completed, 2 running, 2 pending, 0 failures. | Running `pd_binary/scratch` and `pd_binary/partial`; pending `pd_binary/full`, `epilepsy_binary/scratch`. |
| corrected MCI follow-up | tmux is still repeating `WAIT completed=1 running=2 pending=2 failures=0`. | Starts after the missing-job queue becomes idle. |
| detox random rerun | tmux is still waiting for `run_mci_followup_seed20260621/status.json`. | Starts after corrected MCI follow-up completes. |
| GPU sample | GPU2 PID 3545934 at 96%; GPU4 PID 3601741 at 96%. | Both active PD binary jobs are still running. |

Active job snapshot at 2026-06-21 06:25 CST:

| task | mode | latest step epoch / loss | latest validation epoch / train loss / subject metric / F1 |
| --- | --- | --- | --- |
| `pd_binary` | `scratch` | epoch 14 / `0.24831` | epoch 13 / `0.27170` / `0.96306` / `0.91473` |
| `pd_binary` | `partial` | epoch 14 / `0.09966` | epoch 13 / `0.10863` / `0.99180` / `0.93600` |

MCI label-rule clarification at 2026-06-21 06:31 CST:

| item | confirmed status |
| --- | --- |
| old `mci_binary` | Invalid/provisional because the source `MCI` view mixed original MCI rows with `source_dataset=匹配后` copies that can carry crossed labels for the same raw subject/recording. |
| corrected original MCI task | `mci_original_only_binary` uses `downstream/MCI_original_only_no_matched/`, where 33,154 matched-copy rows were removed. |
| corrected matched MCI task | `mci_matched_binary_random_seed20260621` uses `MCI匹配后` rows only as samples; source matched labels are ignored. |
| label authority | Keep a matched row only if its raw `subject` exists in the original `MCI` subject-label anchor after removing `source_dataset=匹配后`; overwrite `health_label` with that original MCI label. |
| generated seed-20260621 matched split | 33,154 rows kept, 0 dropped for missing original MCI subject anchor, 33,154 labels overwritten. |

Monitor update at 2026-06-21 06:33 CST:

| queue/artifact | current state | next condition |
| --- | --- | --- |
| first-version summary | `summary_first_version.*` refreshed: 32 rows, 4 missing final metrics. | Finish `pd_binary` scratch/partial/full and fair `epilepsy_binary` scratch. |
| enhanced first-version audit | `errors=[]`, `warnings=4`, `metrics_final_count=28`. | Run hard audit without `--allow-incomplete` after the four missing metrics land. |
| missing-job queue | 1 completed, 2 running, 2 pending, 0 failures. | Running `pd_binary/scratch` and `pd_binary/partial`; pending `pd_binary/full`, `epilepsy_binary/scratch`. |
| corrected MCI follow-up | Still waiting behind missing-job queue. | Starts after the missing-job queue becomes idle. |
| detox random rerun | Still waiting behind corrected MCI follow-up. | Starts after corrected MCI follow-up completes. |

Active job snapshot at 2026-06-21 06:33 CST:

| task | mode | latest step epoch / loss | latest validation epoch / train loss / subject metric / F1 |
| --- | --- | --- | --- |
| `pd_binary` | `scratch` | epoch 14 / `0.25951` | epoch 13 / `0.27170` / `0.96306` / `0.91473` |
| `pd_binary` | `partial` | epoch 15 / `0.09511` | epoch 15 / `0.09510` / `0.99278` / `0.96094` |

Operational update at 2026-06-21 06:40 CST:

| item | state |
| --- | --- |
| reason | The missing-job queue was still running only on GPU2/GPU4 while GPU1/GPU3 were idle, contradicting the 4-GPU execution target. |
| action | Stopped the old 2-GPU queue manager and started `eyemae_downstream_v3_missing_pd_binary_epilepsy_4gpu_resume_20260621` with `--gpus 1,2,3,4`, same status/log dir, and `--resume-status`. |
| incident | Stopping the old manager also ended the two active child training processes in that session. |
| recovery | Restarted the queue immediately from `checkpoint_last.pt` where available; logs are now appended instead of overwritten. |
| downstream waiters | Replaced old 2-GPU MCI/detox waiters with 4-GPU waiters using the same status/log paths. |
| first-version summary/audit | `summary_first_version.*` refreshed: 32 rows, 4 missing final metrics; enhanced audit `errors=[]`, `warnings=4`, `metrics_final_count=28`. |

4-GPU recovery queue at 2026-06-21 06:40 CST:

| task | mode | GPU | PID | resume source | latest observed progress |
| --- | --- | ---: | ---: | --- | --- |
| `pd_binary` | `scratch` | 1 | 3678842 | `checkpoint_last.pt` from epoch 15 | epoch 15 / step 400 / train loss `0.23638`; latest val epoch 14 AUROC `0.97054`, F1 `0.92` |
| `pd_binary` | `partial` | 2 | 3678843 | `checkpoint_last.pt` from epoch 16 | epoch 16 / step 750 / train loss `0.08881`; latest val epoch 15 AUROC `0.99278`, F1 `0.96094` |
| `pd_binary` | `full` | 3 | 3678844 | no previous checkpoint; fresh start | epoch 0 / step 400 / train loss `0.38879` |
| `epilepsy_binary` | `scratch` | 4 | 3678845 | `checkpoint_last.pt` from epoch 30 | epoch 30 / step 400 / train loss `0.15293` |

Monitor update at 2026-06-21 06:42 CST:

| item | current state |
| --- | --- |
| 4-GPU recovery status | 1 completed, 4 running, 0 pending, 0 failures. |
| GPU sample | GPU1/GPU2/GPU3/GPU4 utilization approximately 98%/96%/97%/93%; GPU0 is unrelated. |
| waiters | MCI follow-up and detox random waiters are both running with `--gpus 1,2,3,4`. |

Active job snapshot at 2026-06-21 06:42 CST:

| task | mode | latest step epoch / loss | latest validation epoch / train loss / subject metric / F1 |
| --- | --- | --- | --- |
| `pd_binary` | `scratch` | epoch 15 / `0.24205` | epoch 14 / `0.25952` / `0.97054` / `0.92` |
| `pd_binary` | `partial` | epoch 16 / `0.08894` | epoch 15 / `0.09510` / `0.99278` / `0.96094` |
| `pd_binary` | `full` | epoch 0 / `0.33927` | n/a |
| `epilepsy_binary` | `scratch` | epoch 30 / `0.15383` | n/a |

Monitor update at 2026-06-21 06:45 CST:

| item | current state |
| --- | --- |
| 4-GPU recovery status | 1 completed, 4 running, 0 pending, 0 failures. |
| first-version summary | `summary_first_version.*` refreshed: 32 rows, 4 missing final metrics. |
| enhanced first-version audit | `errors=[]`, `warnings=4`, `metrics_final_count=28`; warnings are the four currently running补齐 jobs. |
| GPU sample | GPU1/GPU2/GPU3/GPU4 utilization approximately 92%/95%/99%/93%; GPU0 is unrelated. |
| corrected MCI follow-up | 4-GPU waiter still sees upstream `completed=1 running=4 pending=0 failures=0`. |
| detox random rerun | 4-GPU waiter still waits for `run_mci_followup_seed20260621/status.json`. |

Active job snapshot at 2026-06-21 06:44 CST:

| task | mode | latest step epoch / loss | latest validation epoch / train loss / subject metric / F1 |
| --- | --- | --- | --- |
| `pd_binary` | `scratch` | epoch 15 / `0.24384` | epoch 14 / `0.25952` / `0.97054` / `0.92` |
| `pd_binary` | `partial` | epoch 17 / `0.07903` | epoch 16 / `0.08925` / `0.99081` / `0.96183` |
| `pd_binary` | `full` | epoch 0 / `0.31825` | n/a |
| `epilepsy_binary` | `scratch` | epoch 31 / `0.13289` | epoch 30 / `0.15493` / `0.78071` / `0.70073` |

Monitor update at 2026-06-21 06:47 CST:

| item | current state |
| --- | --- |
| 4-GPU recovery status | 1 completed, 4 running, 0 pending, 0 failures. |
| GPU sample | GPU1/GPU2/GPU3/GPU4 utilization approximately 96%/95%/90%/96%; GPU0 is unrelated. |
| corrected MCI follow-up | Not started yet; waits for the missing-job queue to finish. |
| detox random rerun | Not started yet; waits for corrected MCI follow-up status. |

Active job snapshot at 2026-06-21 06:47 CST:

| task | mode | latest step epoch / loss | latest validation epoch / train loss / subject metric / F1 |
| --- | --- | --- | --- |
| `pd_binary` | `scratch` | epoch 15 / `0.24687` | epoch 14 / `0.25952` / `0.97054` / `0.92` |
| `pd_binary` | `partial` | epoch 17 / `0.08261` | epoch 16 / `0.08925` / `0.99081` / `0.96183` |
| `pd_binary` | `full` | epoch 0 / `0.29829` | n/a |
| `epilepsy_binary` | `scratch` | epoch 31 / `0.14563` | epoch 30 / `0.15493` / `0.78071` / `0.70073` |

Monitor update at 2026-06-21 06:49 CST:

| item | current state |
| --- | --- |
| 4-GPU recovery status | 1 completed, 4 running, 0 pending, 0 failures. |
| first-version summary | `summary_first_version.*` refreshed: 32 rows, 4 missing final metrics. |
| enhanced first-version audit | `errors=[]`, `warnings=4`, `metrics_final_count=28`; warnings are exactly `pd_binary` scratch/partial/full and `epilepsy_binary` scratch. |
| GPU sample | GPU1/GPU2/GPU3/GPU4 utilization approximately 96%/96%/97%/94%; GPU0 is unrelated. |
| corrected MCI follow-up | Not started; 4-GPU waiter still sees upstream `completed=1 running=4 pending=0 failures=0`. |
| detox random rerun | Not started; 4-GPU waiter still waits for corrected MCI follow-up status. |

Active job snapshot at 2026-06-21 06:48 CST:

| task | mode | latest step epoch / loss | latest validation epoch / train loss / subject metric / F1 |
| --- | --- | --- | --- |
| `pd_binary` | `scratch` | epoch 15 / `0.24712` | epoch 15 / `0.24712` / `0.97329` / `0.90909` |
| `pd_binary` | `partial` | epoch 17 / `0.08277` | epoch 16 / `0.08925` / `0.99081` / `0.96183` |
| `pd_binary` | `full` | epoch 1 / `0.22408` | epoch 0 / `0.29696` / `0.98858` / `0.92116` |
| `epilepsy_binary` | `scratch` | epoch 31 / `0.14611` | epoch 30 / `0.15493` / `0.78071` / `0.70073` |

Monitor update at 2026-06-21 06:51 CST:

| item | current state |
| --- | --- |
| 4-GPU recovery status | 1 completed, 4 running, 0 pending, 0 failures. |
| GPU sample | GPU1/GPU2/GPU3/GPU4 utilization approximately 96%/96%/98%/96%; GPU0 is unrelated. |
| corrected MCI follow-up | Not started; upstream missing-job queue is still running. |
| detox random rerun | Not started; waits for corrected MCI follow-up. |

Active job snapshot at 2026-06-21 06:51 CST:

| task | mode | latest step epoch / loss | latest validation epoch / train loss / subject metric / F1 |
| --- | --- | --- | --- |
| `pd_binary` | `scratch` | epoch 16 / `0.23010` | epoch 15 / `0.24712` / `0.97329` / `0.90909` |
| `pd_binary` | `partial` | epoch 18 / `0.07808` | epoch 17 / `0.08252` / `0.98806` / `0.952` |
| `pd_binary` | `full` | epoch 1 / `0.21417` | epoch 0 / `0.29696` / `0.98858` / `0.92116` |
| `epilepsy_binary` | `scratch` | epoch 32 / `0.12991` | epoch 31 / `0.14913` / `0.77034` / `0.66667` |

Monitor update at 2026-06-21 06:53 CST:

| item | current state |
| --- | --- |
| 4-GPU recovery status | 1 completed, 4 running, 0 pending, 0 failures. |
| first-version summary | `summary_first_version.*` refreshed: 32 rows, 4 missing final metrics. |
| enhanced first-version audit | `errors=[]`, `warnings=4`, `metrics_final_count=28`; warnings remain exactly `pd_binary` scratch/partial/full and `epilepsy_binary` scratch. |
| GPU sample | GPU1/GPU2/GPU3/GPU4 utilization approximately 96%/95%/97%/95%; GPU0 is unrelated. |
| corrected MCI follow-up | Not started; 4-GPU waiter still sees upstream `completed=1 running=4 pending=0 failures=0`. |
| detox random rerun | Not started; 4-GPU waiter still waits for corrected MCI follow-up status. |

Active job snapshot at 2026-06-21 06:52 CST:

| task | mode | latest step epoch / loss | latest validation epoch / train loss / subject metric / F1 |
| --- | --- | --- | --- |
| `pd_binary` | `scratch` | epoch 16 / `0.23254` | epoch 15 / `0.24712` / `0.97329` / `0.90909` |
| `pd_binary` | `partial` | epoch 18 / `0.07826` | epoch 17 / `0.08252` / `0.98806` / `0.952` |
| `pd_binary` | `full` | epoch 1 / `0.20983` | epoch 0 / `0.29696` / `0.98858` / `0.92116` |
| `epilepsy_binary` | `scratch` | epoch 32 / `0.13538` | epoch 31 / `0.14913` / `0.77034` / `0.66667` |

Monitor update at 2026-06-21 06:55 CST:

| item | current state |
| --- | --- |
| 4-GPU recovery status | 1 completed, 4 running, 0 pending, 0 failures. |
| GPU sample | GPU1/GPU2/GPU3/GPU4 utilization approximately 97%/95%/97%/98%; GPU0 is unrelated. |
| corrected MCI follow-up | Not started; upstream missing-job queue is still running. |
| detox random rerun | Not started; waits for corrected MCI follow-up. |

Active job snapshot at 2026-06-21 06:55 CST:

| task | mode | latest step epoch / loss | latest validation epoch / train loss / subject metric / F1 |
| --- | --- | --- | --- |
| `pd_binary` | `scratch` | epoch 16 / `0.23726` | epoch 15 / `0.24712` / `0.97329` / `0.90909` |
| `pd_binary` | `partial` | epoch 19 / `0.07467` | epoch 18 / `0.07925` / `0.99193` / `0.95455` |
| `pd_binary` | `full` | epoch 1 / `0.20199` | epoch 0 / `0.29696` / `0.98858` / `0.92116` |
| `epilepsy_binary` | `scratch` | epoch 33 / `0.12406` | epoch 32 / `0.14001` / `0.78443` / `0.66667` |

Monitor update at 2026-06-21 06:57 CST:

| item | current state |
| --- | --- |
| 4-GPU recovery status | 1 completed, 4 running, 0 pending, 0 failures. |
| first-version summary | `summary_first_version.*` refreshed: 32 rows, 4 missing final metrics. |
| enhanced first-version audit | `errors=[]`, `warnings=4`, `metrics_final_count=28`; warnings remain exactly `pd_binary` scratch/partial/full and `epilepsy_binary` scratch. |
| GPU sample | GPU1/GPU2/GPU3/GPU4 utilization approximately 95%/94%/98%/96%; GPU0 is unrelated. |
| corrected MCI follow-up | Not started; 4-GPU waiter still sees upstream `completed=1 running=4 pending=0 failures=0`. |
| detox random rerun | Not started; 4-GPU waiter still waits for corrected MCI follow-up status. |

Active job snapshot at 2026-06-21 06:56 CST:

| task | mode | latest step epoch / loss | latest validation epoch / train loss / subject metric / F1 |
| --- | --- | --- | --- |
| `pd_binary` | `scratch` | epoch 16 / `0.23710` | epoch 15 / `0.24712` / `0.97329` / `0.90909` |
| `pd_binary` | `partial` | epoch 19 / `0.07483` | epoch 18 / `0.07925` / `0.99193` / `0.95455` |
| `pd_binary` | `full` | epoch 1 / `0.19890` | epoch 0 / `0.29696` / `0.98858` / `0.92116` |
| `epilepsy_binary` | `scratch` | epoch 33 / `0.12725` | epoch 32 / `0.14001` / `0.78443` / `0.66667` |

Monitor update at 2026-06-21 06:59 CST:

| item | current state |
| --- | --- |
| 4-GPU recovery status | 1 completed, 4 running, 0 pending, 0 failures. |
| GPU sample | GPU1/GPU2/GPU3/GPU4 utilization approximately 98%/95%/96%/97%; GPU0 is unrelated. |
| corrected MCI follow-up | Not started; upstream missing-job queue is still running. |
| detox random rerun | Not started; waits for corrected MCI follow-up. |

Active job snapshot at 2026-06-21 06:59 CST:

| task | mode | latest step epoch / loss | latest validation epoch / train loss / subject metric / F1 |
| --- | --- | --- | --- |
| `pd_binary` | `scratch` | epoch 17 / `0.21802` | epoch 16 / `0.23705` / `0.96706` / `0.91538` |
| `pd_binary` | `partial` | epoch 20 / `0.06202` | epoch 19 / `0.07446` / `0.98885` / `0.94488` |
| `pd_binary` | `full` | epoch 2 / `0.15364` | epoch 1 / `0.19779` / `0.9956` / `0.9551` |
| `epilepsy_binary` | `scratch` | epoch 33 / `0.13297` | epoch 32 / `0.14001` / `0.78443` / `0.66667` |

Monitor update at 2026-06-21 07:01 CST:

| item | current state |
| --- | --- |
| 4-GPU recovery status | 1 completed, 4 running, 0 pending, 0 failures. |
| first-version summary | `summary_first_version.*` refreshed: 32 rows, 4 missing final metrics. |
| enhanced first-version audit | `errors=[]`, `warnings=4`, `metrics_final_count=28`; warnings remain exactly `pd_binary` scratch/partial/full and `epilepsy_binary` scratch. |
| GPU sample | GPU1/GPU2/GPU3/GPU4 utilization approximately 95%/97%/98%/96%; GPU0 is unrelated. |
| corrected MCI follow-up | Not started; 4-GPU waiter still sees upstream `completed=1 running=4 pending=0 failures=0`. |
| detox random rerun | Not started; 4-GPU waiter still waits for corrected MCI follow-up status. |

Active job snapshot at 2026-06-21 07:00 CST:

| task | mode | latest step epoch / loss | latest validation epoch / train loss / subject metric / F1 |
| --- | --- | --- | --- |
| `pd_binary` | `scratch` | epoch 17 / `0.22053` | epoch 16 / `0.23705` / `0.96706` / `0.91538` |
| `pd_binary` | `partial` | epoch 20 / `0.06828` | epoch 19 / `0.07446` / `0.98885` / `0.94488` |
| `pd_binary` | `full` | epoch 2 / `0.15368` | epoch 1 / `0.19779` / `0.9956` / `0.9551` |
| `epilepsy_binary` | `scratch` | epoch 33 / `0.13434` | epoch 33 / `0.13426` / `0.79401` / `0.64567` |

Monitor update at 2026-06-21 07:03 CST:

| item | current state |
| --- | --- |
| 4-GPU recovery status | 1 completed, 4 running, 0 pending, 0 failures. |
| GPU sample | GPU1/GPU2/GPU3/GPU4 utilization approximately 95%/98%/97%/97%; GPU0 is unrelated. |
| corrected MCI follow-up | Not started; upstream missing-job queue is still running. |
| detox random rerun | Not started; waits for corrected MCI follow-up. |

Active job snapshot at 2026-06-21 07:03 CST:

| task | mode | latest step epoch / loss | latest validation epoch / train loss / subject metric / F1 |
| --- | --- | --- | --- |
| `pd_binary` | `scratch` | epoch 17 / `0.22417` | epoch 16 / `0.23705` / `0.96706` / `0.91538` |
| `pd_binary` | `partial` | epoch 20 / `0.06957` | epoch 19 / `0.07446` / `0.98885` / `0.94488` |
| `pd_binary` | `full` | epoch 2 / `0.15076` | epoch 1 / `0.19779` / `0.9956` / `0.9551` |
| `epilepsy_binary` | `scratch` | epoch 34 / `0.12056` | epoch 33 / `0.13426` / `0.79401` / `0.64567` |

Monitor update at 2026-06-21 07:04 CST:

| item | current state |
| --- | --- |
| 4-GPU recovery status | 1 completed, 4 running, 0 pending, 0 failures. |
| first-version summary | `summary_first_version.*` refreshed: 32 rows, 4 missing final metrics. |
| enhanced first-version audit | `errors=[]`, `warnings=4`, `metrics_final_count=28`; warnings remain exactly `pd_binary` scratch/partial/full and `epilepsy_binary` scratch. |
| GPU sample | GPU1/GPU2/GPU3/GPU4 utilization approximately 96%/97%/97%/95%; GPU0 is unrelated. |
| corrected MCI follow-up | Not started; 4-GPU waiter still sees upstream `completed=1 running=4 pending=0 failures=0`. |
| detox random rerun | Not started; 4-GPU waiter still waits for corrected MCI follow-up status. |

Active job snapshot at 2026-06-21 07:04 CST:

| task | mode | latest step epoch / loss | latest validation epoch / train loss / subject metric / F1 |
| --- | --- | --- | --- |
| `pd_binary` | `scratch` | epoch 17 / `0.22405` | epoch 16 / `0.23705` / `0.96706` / `0.91538` |
| `pd_binary` | `partial` | epoch 21 / `0.06870` | epoch 20 / `0.06982` / `0.99134` / `0.93927` |
| `pd_binary` | `full` | epoch 2 / `0.14880` | epoch 1 / `0.19779` / `0.9956` / `0.9551` |
| `epilepsy_binary` | `scratch` | epoch 34 / `0.12421` | epoch 33 / `0.13426` / `0.79401` / `0.64567` |

Monitor update at 2026-06-21 07:07 CST:

| item | current state |
| --- | --- |
| 4-GPU recovery status | 1 completed, 4 running, 0 pending, 0 failures. |
| GPU sample | GPU1/GPU2/GPU3/GPU4 utilization approximately 96%/97%/97%/95%; GPU0 is unrelated. |
| corrected MCI follow-up | Not started; upstream missing-job queue is still running. |
| detox random rerun | Not started; waits for corrected MCI follow-up. |

Active job snapshot at 2026-06-21 07:07 CST:

| task | mode | latest step epoch / loss | latest validation epoch / train loss / subject metric / F1 |
| --- | --- | --- | --- |
| `pd_binary` | `scratch` | epoch 18 / `0.20605` | epoch 17 / `0.22544` / `0.96903` / `0.9127` |
| `pd_binary` | `partial` | epoch 21 / `0.06515` | epoch 20 / `0.06982` / `0.99134` / `0.93927` |
| `pd_binary` | `full` | epoch 3 / `0.11501` | epoch 2 / `0.14742` / `0.99698` / `0.96875` |
| `epilepsy_binary` | `scratch` | epoch 35 / `0.11577` | epoch 34 / `0.12578` / `0.77347` / `0.65152` |

Monitor update at 2026-06-21 07:08 CST:

| item | current state |
| --- | --- |
| 4-GPU recovery status | 1 completed, 4 running, 0 pending, 0 failures. |
| first-version summary | `summary_first_version.*` refreshed: 32 rows, 4 missing final metrics. |
| enhanced first-version audit | `errors=[]`, `warnings=4`, `metrics_final_count=28`; warnings remain exactly `pd_binary` scratch/partial/full and `epilepsy_binary` scratch. |
| GPU sample | GPU1/GPU2/GPU3/GPU4 utilization approximately 98%/96%/99%/96%; GPU0 is unrelated. |
| corrected MCI follow-up | Not started; 4-GPU waiter still sees upstream `completed=1 running=4 pending=0 failures=0`. |
| detox random rerun | Not started; 4-GPU waiter still waits for corrected MCI follow-up status. |

Active job snapshot at 2026-06-21 07:08 CST:

| task | mode | latest step epoch / loss | latest validation epoch / train loss / subject metric / F1 |
| --- | --- | --- | --- |
| `pd_binary` | `scratch` | epoch 18 / `0.20499` | epoch 17 / `0.22544` / `0.96903` / `0.9127` |
| `pd_binary` | `partial` | epoch 21 / `0.06487` | epoch 20 / `0.06982` / `0.99134` / `0.93927` |
| `pd_binary` | `full` | epoch 3 / `0.11425` | epoch 2 / `0.14742` / `0.99698` / `0.96875` |
| `epilepsy_binary` | `scratch` | epoch 35 / `0.11850` | epoch 34 / `0.12578` / `0.77347` / `0.65152` |

Monitor update at 2026-06-21 07:11 CST:

| item | current state |
| --- | --- |
| 4-GPU recovery status | 1 completed, 4 running, 0 pending, 0 failures. |
| GPU sample | GPU1/GPU2/GPU3/GPU4 utilization approximately 98%/96%/99%/89%; GPU0 is unrelated. |
| corrected MCI follow-up | Not started; upstream missing-job queue is still running. |
| detox random rerun | Not started; waits for corrected MCI follow-up. |

Active job snapshot at 2026-06-21 07:11 CST:

| task | mode | latest step epoch / loss | latest validation epoch / train loss / subject metric / F1 |
| --- | --- | --- | --- |
| `pd_binary` | `scratch` | epoch 18 / `0.20966` | epoch 17 / `0.22544` / `0.96903` / `0.9127` |
| `pd_binary` | `partial` | epoch 22 / `0.06130` | epoch 21 / `0.06487` / `0.98917` / `0.95312` |
| `pd_binary` | `full` | epoch 3 / `0.11427` | epoch 2 / `0.14742` / `0.99698` / `0.96875` |
| `epilepsy_binary` | `scratch` | epoch 36 / `0.09733` | epoch 35 / `0.12373` / `0.77915` / `0.64516` |

Monitor update at 2026-06-21 07:12 CST:

| item | current state |
| --- | --- |
| 4-GPU recovery status | 1 completed, 4 running, 0 pending, 0 failures. |
| first-version summary | `summary_first_version.*` refreshed: 32 rows, 4 missing final metrics. |
| enhanced first-version audit | `errors=[]`, `warnings=4`, `metrics_final_count=28`; warnings remain exactly `pd_binary` scratch/partial/full and `epilepsy_binary` scratch. |
| GPU sample | GPU1/GPU2/GPU3/GPU4 utilization approximately 96%/97%/99%/94%; GPU0 is unrelated. |
| corrected MCI follow-up | Not started; 4-GPU waiter still sees upstream `completed=1 running=4 pending=0 failures=0`. |
| detox random rerun | Not started; 4-GPU waiter still waits for corrected MCI follow-up status. |

Active job snapshot at 2026-06-21 07:12 CST:

| task | mode | latest step epoch / loss | latest validation epoch / train loss / subject metric / F1 |
| --- | --- | --- | --- |
| `pd_binary` | `scratch` | epoch 18 / `0.20942` | epoch 17 / `0.22544` / `0.96903` / `0.9127` |
| `pd_binary` | `partial` | epoch 22 / `0.06178` | epoch 21 / `0.06487` / `0.98917` / `0.95312` |
| `pd_binary` | `full` | epoch 3 / `0.11429` | epoch 2 / `0.14742` / `0.99698` / `0.96875` |
| `epilepsy_binary` | `scratch` | epoch 36 / `0.11230` | epoch 35 / `0.12373` / `0.77915` / `0.64516` |

Monitor update at 2026-06-21 07:14 CST:

| item | current state |
| --- | --- |
| 4-GPU recovery status | 1 completed, 4 running, 0 pending, 0 failures. |
| first-version summary | `summary_first_version.*` refreshed: 32 rows, 4 missing final metrics. |
| enhanced first-version audit | `errors=[]`, `warnings=4`, `metrics_final_count=28`; warnings remain exactly `pd_binary` scratch/partial/full and `epilepsy_binary` scratch. |
| GPU sample | GPU1/GPU2/GPU3/GPU4 utilization approximately 98%/91%/98%/98%; GPU0 is unrelated. |
| corrected MCI follow-up | Not started; 4-GPU waiter still sees upstream `completed=1 running=4 pending=0 failures=0`. |
| detox random rerun | Not started; 4-GPU waiter still waits for corrected MCI follow-up status. |

Active job snapshot at 2026-06-21 07:14 CST:

| task | mode | latest step epoch / loss | latest validation epoch / train loss / subject metric / F1 |
| --- | --- | --- | --- |
| `pd_binary` | `scratch` | epoch 18 / `0.21057` | epoch 17 / `0.22544` / `0.96903` / `0.9127` |
| `pd_binary` | `partial` | epoch 22 / `0.06159` | epoch 21 / `0.06487` / `0.98917` / `0.95312` |
| `pd_binary` | `full` | epoch 3 / `0.11329` | epoch 2 / `0.14742` / `0.99698` / `0.96875` |
| `epilepsy_binary` | `scratch` | epoch 36 / `0.11547` | epoch 35 / `0.12373` / `0.77915` / `0.64516` |

Monitor update at 2026-06-21 07:16 CST:

| item | current state |
| --- | --- |
| 4-GPU recovery status | 1 completed, 4 running, 0 pending, 0 failures. |
| first-version summary | `summary_first_version.*` refreshed: 32 rows, 4 missing final metrics. |
| enhanced first-version audit | `errors=[]`, `warnings=4`, `metrics_final_count=28`; warnings remain exactly `pd_binary` scratch/partial/full and `epilepsy_binary` scratch. |
| GPU sample | GPU1/GPU2/GPU3/GPU4 utilization approximately 86%/93%/95%/97%; GPU0 is unrelated. |
| corrected MCI follow-up | Not started; upstream missing-job queue is still running. |
| detox random rerun | Not started; waits for corrected MCI follow-up. |

Active job snapshot at 2026-06-21 07:15 CST:

| task | mode | latest step epoch / loss | latest validation epoch / train loss / subject metric / F1 |
| --- | --- | --- | --- |
| `pd_binary` | `scratch` | epoch 18 / `0.21193` | epoch 17 / `0.22544` / `0.96903` / `0.9127` |
| `pd_binary` | `partial` | epoch 23 / `0.05380` | epoch 22 / `0.06176` / `0.99108` / `0.95652` |
| `pd_binary` | `full` | epoch 4 / `0.07521` | epoch 3 / `0.11300` / `0.99364` / `0.94606` |
| `epilepsy_binary` | `scratch` | epoch 36 / `0.11764` | epoch 35 / `0.12373` / `0.77915` / `0.64516` |

Monitor update at 2026-06-21 07:17 CST:

| item | current state |
| --- | --- |
| 4-GPU recovery status | 1 completed, 4 running, 0 pending, 0 failures. |
| first-version summary | `summary_first_version.*` refreshed: 32 rows, 4 missing final metrics. |
| enhanced first-version audit | `errors=[]`, `warnings=4`, `metrics_final_count=28`; warnings remain exactly `pd_binary` scratch/partial/full and `epilepsy_binary` scratch. |
| GPU sample | GPU1/GPU2/GPU3/GPU4 utilization approximately 98%/95%/97%/95%; GPU0 is unrelated. |
| corrected MCI follow-up | Not started; upstream missing-job queue is still running. |
| detox random rerun | Not started; waits for corrected MCI follow-up. |

Active job snapshot at 2026-06-21 07:17 CST:

| task | mode | latest step epoch / loss | latest validation epoch / train loss / subject metric / F1 |
| --- | --- | --- | --- |
| `pd_binary` | `scratch` | epoch 19 / `0.19812` | epoch 18 / `0.21223` / `0.9647` / `0.91057` |
| `pd_binary` | `partial` | epoch 23 / `0.05597` | epoch 22 / `0.06176` / `0.99108` / `0.95652` |
| `pd_binary` | `full` | epoch 4 / `0.09124` | epoch 3 / `0.11300` / `0.99364` / `0.94606` |
| `epilepsy_binary` | `scratch` | epoch 37 / `0.09701` | epoch 36 / `0.11852` / `0.78169` / `0.64567` |

Monitor update at 2026-06-21 07:19 CST:

| item | current state |
| --- | --- |
| 4-GPU recovery status | 1 completed, 4 running, 0 pending, 0 failures. |
| first-version summary | `summary_first_version.*` refreshed: 32 rows, 4 missing final metrics. |
| enhanced first-version audit | `errors=[]`, `warnings=4`, `metrics_final_count=28`; warnings remain exactly `pd_binary` scratch/partial/full and `epilepsy_binary` scratch. |
| GPU sample | GPU1/GPU2/GPU3/GPU4 utilization approximately 96%/97%/97%/97%; GPU0 is unrelated. |
| corrected MCI follow-up | Not started; upstream missing-job queue is still running. |
| detox random rerun | Not started; waits for corrected MCI follow-up. |

Active job snapshot at 2026-06-21 07:19 CST:

| task | mode | latest step epoch / loss | latest validation epoch / train loss / subject metric / F1 |
| --- | --- | --- | --- |
| `pd_binary` | `scratch` | epoch 19 / `0.19972` | epoch 18 / `0.21223` / `0.9647` / `0.91057` |
| `pd_binary` | `partial` | epoch 23 / `0.05655` | epoch 22 / `0.06176` / `0.99108` / `0.95652` |
| `pd_binary` | `full` | epoch 4 / `0.09076` | epoch 3 / `0.11300` / `0.99364` / `0.94606` |
| `epilepsy_binary` | `scratch` | epoch 37 / `0.10546` | epoch 36 / `0.11852` / `0.78169` / `0.64567` |

MCI correction update at 2026-06-21 07:24 CST:

| item | result |
| --- | --- |
| corrected rule | `MCI匹配后` rows are samples only. Keep a row only if its raw `subject` exists in the original `MCI` anchor after excluding `source_dataset=匹配后`; otherwise drop it. |
| label authority | `health_label` is overwritten from the original `MCI` subject label. `MCI匹配后` source labels are never used. |
| metadata fix | `source_group` is also overwritten to match the original `MCI` subject label, so the generated CSV no longer carries crossed label-like group text. |
| regeneration command | `python scripts/prepare_mci_followup_finetune.py --force` |
| script check | `python -m py_compile scripts/prepare_mci_followup_finetune.py` passed. |
| corrected matched rows | 33,154 rows, 218 subjects. |
| anchor audit | `outside_anchor_rows=0`, `health_label_mismatch_rows=0`, `source_group_mismatch_rows=0`. |
| queue impact | Corrected MCI queue had not launched yet, so no MCI fine-tune job used the pre-metadata-fix CSV. |

Monitor update at 2026-06-21 07:25 CST:

| item | current state |
| --- | --- |
| 4-GPU recovery status | 1 completed, 4 running, 0 pending, 0 failures. |
| first-version summary | `summary_first_version.*` refreshed: 32 rows, 4 missing final metrics. |
| enhanced first-version audit | `errors=[]`, `warnings=4`, `metrics_final_count=28`; warnings remain `pd_binary` scratch/partial/full and `epilepsy_binary` scratch. |
| GPU sample | unavailable from the current sandboxed shell; training logs and queue status remain readable. |
| corrected MCI follow-up | Not started; upstream missing-job queue is still running. |
| detox random rerun | Not started; waits for corrected MCI follow-up. |

Active job snapshot at 2026-06-21 07:25 CST:

| task | mode | latest step epoch / loss | latest validation epoch / train loss / subject metric / F1 |
| --- | --- | --- | --- |
| `pd_binary` | `scratch` | epoch 19 / `0.20057` | epoch 18 / `0.21223` / `0.9647` / `0.91057` |
| `pd_binary` | `partial` | epoch 25 / `0.04515` | epoch 24 / `0.05495` / `0.98852` / `0.94488` |
| `pd_binary` | `full` | epoch 5 / `0.06871` | epoch 4 / `0.08965` / `0.99633` / `0.95082` |
| `epilepsy_binary` | `scratch` | epoch 38 / `0.10526` | epoch 37 / `0.11084` / `0.76487` / `0.65185` |

Official scope correction at 2026-06-21 07:28 CST:

| item | current state |
| --- | --- |
| formal MCI task names | Replaced old `mci_binary` / `mci_matched_binary` with `mci_original_only_binary` / `mci_matched_binary_random_seed20260621` in the official first-version config list and audit. |
| reason | Old MCI outputs mix or use crossed matched labels; they remain provisional diagnostics only and must not count toward goal completion. |
| summary refresh | `summary_first_version.*` refreshed after the task-list replacement: 32 rows, 12 missing final metrics. |
| audit refresh | `errors=[]`, `warnings=12`, `metrics_final_count=28`. |
| warning interpretation | 4 warnings are the still-running PD binary/epilepsy recovery jobs; 8 warnings are the not-yet-launched corrected MCI follow-up jobs. |

Monitor update at 2026-06-21 07:28 CST:

| item | current state |
| --- | --- |
| 4-GPU recovery status | 1 completed, 4 running, 0 pending, 0 failures. |
| corrected MCI follow-up | Not started; upstream missing-job queue is still running. |
| detox random rerun | Not started; waits for corrected MCI follow-up. |

Active job snapshot at 2026-06-21 07:28 CST:

| task | mode | latest step epoch / loss | latest validation epoch / train loss / subject metric / F1 |
| --- | --- | --- | --- |
| `pd_binary` | `scratch` | epoch 20 / `0.18367` | epoch 19 / `0.20063` / `0.96509` / `0.89879` |
| `pd_binary` | `partial` | epoch 25 / `0.05199` | epoch 24 / `0.05495` / `0.98852` / `0.94488` |
| `pd_binary` | `full` | epoch 5 / `0.07253` | epoch 4 / `0.08965` / `0.99633` / `0.95082` |
| `epilepsy_binary` | `scratch` | epoch 39 / `0.09409` | epoch 38 / `0.10915` / `0.78013` / `0.73333` |

Monitor update at 2026-06-21 07:32 CST:

| item | current state |
| --- | --- |
| 4-GPU recovery status | 1 completed, 4 running, 0 pending, 0 failures. |
| first-version summary | `summary_first_version.*` refreshed: 32 rows, 12 missing final metrics. |
| enhanced first-version audit | `errors=[]`, `warnings=12`, `metrics_final_count=28`; status unresolved counts are `running=0`, `pending=0`, `failures=1`. |
| unresolved historical failure | `epilepsy_binary_scratch.yaml`; recovery job is still running and will resolve this only after `metrics_final.json` is written. |
| GPU sample | GPU1/GPU2/GPU3/GPU4 utilization approximately 98%/97%/98%/96%; GPU0 is unrelated. |
| corrected MCI follow-up | Not started; upstream missing-job queue is still running. |
| detox random rerun | Not started; waits for corrected MCI follow-up. |

Active job snapshot at 2026-06-21 07:32 CST:

| task | mode | latest step epoch / loss | latest validation epoch / train loss / subject metric / F1 |
| --- | --- | --- | --- |
| `pd_binary` | `scratch` | epoch 20 / `0.18918` | epoch 19 / `0.20063` / `0.96509` / `0.89879` |
| `pd_binary` | `partial` | epoch 25 / `0.05183` | epoch 25 / `0.05175` / `0.9914` / `0.95312` |
| `pd_binary` | `full` | epoch 5 / `0.07264` | epoch 4 / `0.08965` / `0.99633` / `0.95082` |
| `epilepsy_binary` | `scratch` | epoch 39 / `0.10302` | epoch 38 / `0.10915` / `0.78013` / `0.73333` |

Completion update at 2026-06-21 07:39 CST:

| item | current state |
| --- | --- |
| completed since last update | `pd_binary/partial`, `epilepsy_binary/scratch` |
| 4-GPU recovery status | 3 completed, 2 running, 0 pending, 0 failures. |
| first-version summary | `summary_first_version.*` refreshed: 32 rows, 10 missing final metrics. |
| enhanced first-version audit | `errors=[]`, `warnings=10`, `metrics_final_count=30`, `unresolved_failures=0`. |
| remaining formal missing jobs | `pd_binary/scratch`, `pd_binary/full`, and the 8 corrected MCI jobs. |
| MCI scheduling change | Stopped the MCI wait-only session and started corrected MCI follow-up immediately on free GPU2/GPU4. |
| active sessions | Recovery queue runs `pd_binary/scratch` on GPU1 and `pd_binary/full` on GPU3; MCI queue runs `mci_original_only_binary/scratch` on GPU2 and `mci_original_only_binary/linear_probe` on GPU4. |
| GPU sample | GPU1/GPU2/GPU3/GPU4 utilization approximately 94%/98%/97%/94%; GPU0 is unrelated. |
| detox random rerun | Not started; waits for corrected MCI follow-up status to become idle. |

New completed test metrics:

| task | mode | best epoch | validation subject AUROC | test subject AUROC | balanced acc | F1 | weighted F1 | Cohen's Kappa |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `pd_binary` | `partial` | 12 | `0.99304` | `0.97562` | `0.97132` | `0.97161` | `0.97107` | `0.94211` |
| `epilepsy_binary` | `scratch` | 25 | `0.79441` | `0.82891` | `0.79099` | `0.79096` | `0.79096` | `0.58193` |

Active job snapshot at 2026-06-21 07:39 CST:

| task | mode | latest step epoch / loss | latest validation epoch / train loss / subject metric / F1 |
| --- | --- | --- | --- |
| `pd_binary` | `scratch` | epoch 21 / `0.18172` | epoch 20 / `0.19037` / `0.96798` / `0.90323` |
| `pd_binary` | `full` | epoch 6 / `0.06176` | epoch 5 / `0.07243` / `0.99259` / `0.95276` |
| `mci_original_only_binary` | `scratch` | launched on GPU2 | not yet reached |
| `mci_original_only_binary` | `linear_probe` | launched on GPU4 | not yet reached |

Monitor update at 2026-06-21 07:41 CST:

| item | current state |
| --- | --- |
| recovery queue | 3 completed, 2 running, 0 pending, 0 failures. Running: `pd_binary/scratch` on GPU1 and `pd_binary/full` on GPU3. |
| corrected MCI queue | 0 completed, 2 running, 6 pending, 0 failures. Running: `mci_original_only_binary/scratch` on GPU2 and `mci_original_only_binary/linear_probe` on GPU4. |
| detox random rerun | Not started; waits for corrected MCI status to become idle. |
| first-version summary | `summary_first_version.*` refreshed: 32 rows, 10 missing final metrics. |
| enhanced first-version audit | `errors=[]`, `warnings=10`, `metrics_final_count=30`, `unresolved_failures=0`. |
| GPU sample | GPU1/GPU2/GPU3/GPU4 utilization approximately 96%/97%/98%/96%; GPU0 is unrelated. |

Active job snapshot at 2026-06-21 07:41 CST:

| task | mode | latest step epoch / loss | latest validation epoch / train loss / subject metric / F1 |
| --- | --- | --- | --- |
| `pd_binary` | `scratch` | epoch 21 / `0.18128` | epoch 20 / `0.19037` / `0.96798` / `0.90323` |
| `pd_binary` | `full` | epoch 6 / `0.05967` | epoch 5 / `0.07243` / `0.99259` / `0.95276` |
| `mci_original_only_binary` | `scratch` | epoch 1 / `0.67217` | epoch 0 / `0.68463` / `0.63636` / `0.51064` |
| `mci_original_only_binary` | `linear_probe` | epoch 3 / `0.58351` | epoch 2 / `0.60013` / `0.75799` / `0.48649` |

Monitor update at 2026-06-21 07:43 CST:

| item | current state |
| --- | --- |
| recovery queue | 3 completed, 2 running, 0 pending, 0 failures. Running: `pd_binary/scratch` on GPU1 and `pd_binary/full` on GPU3. |
| corrected MCI queue | 0 completed, 2 running, 6 pending, 0 failures. Running: `mci_original_only_binary/scratch` on GPU2 and `mci_original_only_binary/linear_probe` on GPU4. |
| detox random rerun | Not started; waits for corrected MCI status to become idle. |
| first-version summary | `summary_first_version.*` refreshed: 32 rows, 10 missing final metrics. |
| enhanced first-version audit | `errors=[]`, `warnings=10`, `metrics_final_count=30`, `unresolved_failures=0`. |
| GPU sample | GPU1/GPU2/GPU3/GPU4 utilization approximately 95%/96%/97%/92%; GPU0 is unrelated. |
| error scan | Current PD and MCI logs show no `Traceback`, `ERROR`, or `RuntimeError`. |

Active job snapshot at 2026-06-21 07:43 CST:

| task | mode | latest step epoch / loss | latest validation epoch / train loss / subject metric / F1 |
| --- | --- | --- | --- |
| `pd_binary` | `scratch` | epoch 21 / `0.18216` | epoch 20 / `0.19037` / `0.96798` / `0.90323` |
| `pd_binary` | `full` | epoch 6 / `0.05965` | epoch 5 / `0.07243` / `0.99259` / `0.95276` |
| `mci_original_only_binary` | `scratch` | epoch 2 / `0.65720` | epoch 1 / `0.66214` / `0.66216` / `0.61017` |
| `mci_original_only_binary` | `linear_probe` | epoch 6 / `0.55848` | epoch 5 / `0.55868` / `0.76658` / `0.66667` |

Monitor update at 2026-06-21 07:47 CST:

| item | current state |
| --- | --- |
| recovery queue | 3 completed, 2 running, 0 pending, 0 failures. Running: `pd_binary/scratch` on GPU1 and `pd_binary/full` on GPU3. |
| corrected MCI queue | 0 completed, 2 running, 6 pending, 0 failures. Running: `mci_original_only_binary/scratch` on GPU2 and `mci_original_only_binary/linear_probe` on GPU4. |
| detox random rerun | Not started; waits for corrected MCI status to become idle. |
| first-version summary | `summary_first_version.*` refreshed: 32 rows, 10 missing final metrics. |
| enhanced first-version audit | `errors=[]`, `warnings=10`, `metrics_final_count=30`, `unresolved_failures=0`. |
| GPU sample | GPU1/GPU2/GPU3/GPU4 utilization approximately 96%/98%/98%/98%; GPU0 is unrelated. |
| error scan | Current PD and MCI logs show no `Traceback`, `ERROR`, or `RuntimeError`. |

Active job snapshot at 2026-06-21 07:47 CST:

| task | mode | latest step epoch / loss | latest validation epoch / train loss / subject metric / F1 |
| --- | --- | --- | --- |
| `pd_binary` | `scratch` | epoch 22 / `0.17130` | epoch 21 / `0.18367` / `0.97159` / `0.912` |
| `pd_binary` | `full` | epoch 7 / `0.04621` | epoch 6 / `0.05982` / `0.99547` / `0.96327` |
| `mci_original_only_binary` | `scratch` | epoch 3 / `0.63673` | epoch 2 / `0.64735` / `0.6683` / `0.5` |
| `mci_original_only_binary` | `linear_probe` | epoch 10 / `0.51603` | epoch 9 / `0.52329` / `0.76536` / `0.57778` |

Monitor update at 2026-06-21 07:50 CST:

| item | current state |
| --- | --- |
| recovery queue | 3 completed, 2 running, 0 pending, 0 failures. Running: `pd_binary/scratch` on GPU1 and `pd_binary/full` on GPU3. |
| corrected MCI queue | 0 completed, 2 running, 6 pending, 0 failures. Running: `mci_original_only_binary/scratch` on GPU2 and `mci_original_only_binary/linear_probe` on GPU4. |
| detox random rerun | Not started; waits for corrected MCI status to become idle. |
| first-version summary | `summary_first_version.*` refreshed: 32 rows, 10 missing final metrics. |
| enhanced first-version audit | `errors=[]`, `warnings=10`, `metrics_final_count=30`, `unresolved_failures=0`. |
| GPU sample | GPU1/GPU2/GPU3/GPU4 utilization approximately 95%/97%/99%/96%; GPU0 is unrelated. |
| error scan | Current PD and MCI logs show no `Traceback`, `ERROR`, or `RuntimeError`. |

Active job snapshot at 2026-06-21 07:50 CST:

| task | mode | latest step epoch / loss | latest validation epoch / train loss / subject metric / F1 |
| --- | --- | --- | --- |
| `pd_binary` | `scratch` | epoch 22 / `0.17360` | epoch 21 / `0.18367` / `0.97159` / `0.912` |
| `pd_binary` | `full` | epoch 7 / `0.04861` | epoch 6 / `0.05982` / `0.99547` / `0.96327` |
| `mci_original_only_binary` | `scratch` | epoch 4 / `0.62984` | epoch 3 / `0.63545` / `0.76044` / `0.66667` |
| `mci_original_only_binary` | `linear_probe` | epoch 13 / `0.49214` | epoch 12 / `0.50467` / `0.78993` / `0.55556` |

Monitor update at 2026-06-21 07:52 CST:

| item | current state |
| --- | --- |
| recovery queue | 3 completed, 2 running, 0 pending, 0 failures. Running: `pd_binary/scratch` on GPU1 and `pd_binary/full` on GPU3. |
| corrected MCI queue | 0 completed, 2 running, 6 pending, 0 failures. Running: `mci_original_only_binary/scratch` on GPU2 and `mci_original_only_binary/linear_probe` on GPU4. |
| detox random rerun | Not started; waits for corrected MCI status to become idle. |
| first-version summary | `summary_first_version.*` refreshed: 32 rows, 10 missing final metrics. |
| enhanced first-version audit | `errors=[]`, `warnings=10`, `metrics_final_count=30`, `unresolved_failures=0`. |
| GPU sample | GPU1/GPU2/GPU3/GPU4 utilization approximately 96%/96%/98%/95%; GPU0 is unrelated. |
| error scan | Current PD and MCI logs show no `Traceback`, `ERROR`, or `RuntimeError`. |

Active job snapshot at 2026-06-21 07:52 CST:

| task | mode | latest step epoch / loss | latest validation epoch / train loss / subject metric / F1 |
| --- | --- | --- | --- |
| `pd_binary` | `scratch` | epoch 22 / `0.17605` | epoch 21 / `0.18367` / `0.97159` / `0.912` |
| `pd_binary` | `full` | epoch 7 / `0.04791` | epoch 6 / `0.05982` / `0.99547` / `0.96327` |
| `mci_original_only_binary` | `scratch` | epoch 5 / `0.61799` | epoch 4 / `0.62740` / `0.75553` / `0.61818` |
| `mci_original_only_binary` | `linear_probe` | epoch 15 / `0.48638` | epoch 14 / `0.49265` / `0.76781` / `0.47059` |

Monitor update at 2026-06-21 07:54 CST:

| item | current state |
| --- | --- |
| recovery queue | 3 completed, 2 running, 0 pending, 0 failures. Running: `pd_binary/scratch` on GPU1 and `pd_binary/full` on GPU3. |
| corrected MCI queue | 0 completed, 2 running, 6 pending, 0 failures. Running: `mci_original_only_binary/scratch` on GPU2 and `mci_original_only_binary/linear_probe` on GPU4. |
| detox random rerun | Not started; waits for corrected MCI status to become idle. |
| first-version summary | `summary_first_version.*` refreshed: 32 rows, 10 missing final metrics. |
| enhanced first-version audit | `errors=[]`, `warnings=10`, `metrics_final_count=30`, `unresolved_failures=0`. |
| GPU sample | GPU1/GPU2/GPU3/GPU4 utilization approximately 89%/96%/98%/95%; GPU0 is unrelated. |
| error scan | Current PD and MCI logs show no `Traceback`, `ERROR`, or `RuntimeError`. |

Active job snapshot at 2026-06-21 07:54 CST:

| task | mode | latest step epoch / loss | latest validation epoch / train loss / subject metric / F1 |
| --- | --- | --- | --- |
| `pd_binary` | `scratch` | epoch 22 / `0.17687` | epoch 22 / `0.17681` / `0.97205` / `0.93281` |
| `pd_binary` | `full` | epoch 8 / `0.04148` | epoch 7 / `0.04841` / `0.9956` / `0.95652` |
| `mci_original_only_binary` | `scratch` | epoch 6 / `0.61651` | epoch 5 / `0.61807` / `0.72359` / `0.59649` |
| `mci_original_only_binary` | `linear_probe` | epoch 18 / `0.45325` | epoch 17 / `0.47770` / `0.77764` / `0.59091` |

Monitor update at 2026-06-21 07:56 CST:

| item | current state |
| --- | --- |
| recovery queue | 3 completed, 2 running, 0 pending, 0 failures. Running: `pd_binary/scratch` on GPU1 and `pd_binary/full` on GPU3. |
| corrected MCI queue | 0 completed, 2 running, 6 pending, 0 failures. Running: `mci_original_only_binary/scratch` on GPU2 and `mci_original_only_binary/linear_probe` on GPU4. |
| detox random rerun | Not started; waits for corrected MCI status to become idle. |
| first-version summary | `summary_first_version.*` refreshed: 32 rows, 10 missing final metrics. |
| enhanced first-version audit | `errors=[]`, `warnings=10`, `metrics_final_count=30`, `unresolved_failures=0`. |
| GPU sample | GPU1/GPU2/GPU3/GPU4 utilization approximately 96%/98%/97%/96%; GPU0 is unrelated. |
| error scan | Current PD and MCI logs show no `Traceback`, `ERROR`, or `RuntimeError`. |

Active job snapshot at 2026-06-21 07:56 CST:

| task | mode | latest step epoch / loss | latest validation epoch / train loss / subject metric / F1 |
| --- | --- | --- | --- |
| `pd_binary` | `scratch` | epoch 23 / `0.15784` | epoch 22 / `0.17681` / `0.97205` / `0.93281` |
| `pd_binary` | `full` | epoch 8 / `0.04177` | epoch 7 / `0.04841` / `0.9956` / `0.95652` |
| `mci_original_only_binary` | `scratch` | epoch 7 / `0.59122` | epoch 6 / `0.60670` / `0.73464` / `0.625` |
| `mci_original_only_binary` | `linear_probe` | epoch 20 / `0.45829` | epoch 19 / `0.46886` / `0.81327` / `0.66667` |

MCI label-anchor confirmation at 2026-06-21 08:01 CST:

| item | verified result |
| --- | --- |
| intended rule | Use only original `MCI` raw `subject` values as the label authority. `MCI匹配后` rows may be samples, but their `health_label` and label-like `source_group` must not be trusted. |
| original MCI anchor | Built after removing `source_dataset=匹配后`: 58,279 rows, 383 raw subjects, 0 raw-subject label conflicts. |
| cleaned original task | `MCI_original_only_no_matched`: 58,279 rows, 383 subjects, 33,154 matched-copy rows removed, 0 label/split conflicts. |
| matched follow-up task | `MCI匹配后_random_seed20260621`: 33,154 rows, 218 subjects; each row is kept only because its raw `subject` exists in the original-MCI anchor. |
| matched-row label use | 0 rows use a mismatched label; 0 rows have mismatched `source_group`; 33,154 / 33,154 rows had labels overwritten relative to source `MCI匹配后`. |
| queue scope | Official queue uses `mci_original_only_binary` and `mci_matched_binary_random_seed20260621`; old `mci_binary` and old `mci_matched_binary` remain invalid/provisional. |

Monitor update at 2026-06-21 08:01 CST:

| item | current state |
| --- | --- |
| recovery queue | 3 completed, 2 running, 0 pending, 0 failures. Running: `pd_binary/scratch` on GPU1 and `pd_binary/full` on GPU3. |
| corrected MCI queue | 1 completed, 2 running, 5 pending, 0 failures. Completed: `mci_original_only_binary/linear_probe`. Running: `mci_original_only_binary/scratch` on GPU2 and `mci_original_only_binary/partial` on GPU4. |
| detox random rerun | Not started; waits for corrected MCI status to become idle. |
| first-version audit | `errors=[]`; remaining official missing finals are `pd_binary` scratch/full plus 7 corrected MCI jobs. |
| corrected MCI linear result | validation subject AUROC `0.81327`; final-best test metrics omitted by current reporting policy. |

Monitor update at 2026-06-21 08:03 CST:

| item | current state |
| --- | --- |
| recovery queue | 3 completed, 2 running, 0 pending, 0 failures. Running: `pd_binary/scratch` on GPU1 and `pd_binary/full` on GPU3. |
| corrected MCI queue | 1 completed, 2 running, 5 pending, 0 failures. Running: `mci_original_only_binary/scratch` on GPU2 and `mci_original_only_binary/partial` on GPU4. |
| detox random rerun | Not started; waits for corrected MCI status to become idle. |
| first-version summary | `summary_first_version.*` refreshed: 32 rows, 9 missing final metrics. |
| first-version missing finals | `pd_binary` scratch/full; `mci_original_only_binary` scratch/partial/full; all 4 `mci_matched_binary_random_seed20260621` modes. |
| GPU sample | GPU1/GPU2/GPU3/GPU4 utilization approximately 97%/97%/98%/94%; GPU0 is unrelated to this queue. |
| error scan | Current PD and MCI logs show no `Traceback`, `ERROR`, or `RuntimeError`. |

Active job snapshot at 2026-06-21 08:03 CST:

| task | mode | latest step epoch / loss | latest validation epoch / train loss / subject metric / F1 |
| --- | --- | --- | --- |
| `pd_binary` | `scratch` | epoch 23 / `0.16440` | epoch 23 / `0.16446` / `0.97178` / `0.90688` |
| `pd_binary` | `full` | epoch 9 / `0.03351` | epoch 8 / `0.04278` / `0.99409` / `0.95312` |
| `mci_original_only_binary` | `scratch` | epoch 9 / `0.56725` | epoch 9 / `0.56758` / `0.68305` / `0.5614` |
| `mci_original_only_binary` | `partial` | epoch 2 / `0.53196` | epoch 2 / `0.53202` / `0.77273` / `0.55556` |

Monitor update at 2026-06-21 08:05 CST:

| item | current state |
| --- | --- |
| recovery queue | 3 completed, 2 running, 0 pending, 0 failures. Running: `pd_binary/scratch` on GPU1 and `pd_binary/full` on GPU3. |
| corrected MCI queue | 1 completed, 2 running, 5 pending, 0 failures. Running: `mci_original_only_binary/scratch` on GPU2 and `mci_original_only_binary/partial` on GPU4. |
| detox random rerun | Not started; waits for corrected MCI status to become idle. |
| first-version summary | `summary_first_version.*` refreshed: 32 rows, 9 missing final metrics. |
| enhanced first-version audit | `errors=[]`, `warnings=9`, `metrics_final_count=31`, `unresolved_failures=0`. |
| first-version missing finals | `pd_binary` scratch/full; `mci_original_only_binary` scratch/partial/full; all 4 `mci_matched_binary_random_seed20260621` modes. |
| GPU sample | GPU1/GPU2/GPU3/GPU4 utilization approximately 96%/98%/98%/94%; GPU0 is unrelated to this queue. |
| error scan | Current PD and MCI logs show no `Traceback`, `ERROR`, or `RuntimeError`. |

Active job snapshot at 2026-06-21 08:05 CST:

| task | mode | latest step epoch / loss | latest validation epoch / train loss / subject metric / F1 |
| --- | --- | --- | --- |
| `pd_binary` | `scratch` | epoch 24 / `0.15318` | epoch 23 / `0.16446` / `0.97178` / `0.90688` |
| `pd_binary` | `full` | epoch 9 / `0.03557` | epoch 8 / `0.04278` / `0.99409` / `0.95312` |
| `mci_original_only_binary` | `scratch` | epoch 10 / `0.55275` | epoch 9 / `0.56758` / `0.68305` / `0.5614` |
| `mci_original_only_binary` | `partial` | epoch 4 / `0.47344` | epoch 3 / `0.50094` / `0.78747` / `0.61905` |

Monitor update at 2026-06-21 08:07 CST:

| item | current state |
| --- | --- |
| recovery queue | 3 completed, 2 running, 0 pending, 0 failures. Running: `pd_binary/scratch` on GPU1 and `pd_binary/full` on GPU3. |
| corrected MCI queue | 1 completed, 2 running, 5 pending, 0 failures. Running: `mci_original_only_binary/scratch` on GPU2 and `mci_original_only_binary/partial` on GPU4. |
| detox random rerun | Not started; waits for corrected MCI status to become idle. |
| first-version summary | `summary_first_version.*` refreshed: 32 rows, 9 missing final metrics. |
| enhanced first-version audit | `errors=[]`, `warnings=9`, `metrics_final_count=31`, `unresolved_failures=0`. |
| first-version missing finals | `pd_binary` scratch/full; `mci_original_only_binary` scratch/partial/full; all 4 `mci_matched_binary_random_seed20260621` modes. |
| GPU sample | GPU1/GPU2/GPU3/GPU4 utilization approximately 96%/97%/98%/96%; GPU0 is unrelated to this queue. |
| error scan | Current PD and MCI logs show no `Traceback`, `ERROR`, or `RuntimeError`. |

Active job snapshot at 2026-06-21 08:07 CST:

| task | mode | latest step epoch / loss | latest validation epoch / train loss / subject metric / F1 |
| --- | --- | --- | --- |
| `pd_binary` | `scratch` | epoch 24 / `0.15292` | epoch 23 / `0.16446` / `0.97178` / `0.90688` |
| `pd_binary` | `full` | epoch 9 / `0.03668` | epoch 8 / `0.04278` / `0.99409` / `0.95312` |
| `mci_original_only_binary` | `scratch` | epoch 11 / `0.53163` | epoch 10 / `0.55166` / `0.66093` / `0.52` |
| `mci_original_only_binary` | `partial` | epoch 5 / `0.44545` | epoch 5 / `0.44545` / `0.77027` / `0.625` |

Monitor update at 2026-06-21 08:10 CST:

| item | current state |
| --- | --- |
| recovery queue | 3 completed, 2 running, 0 pending, 0 failures. Running: `pd_binary/scratch` on GPU1 and `pd_binary/full` on GPU3. |
| corrected MCI queue | 1 completed, 2 running, 5 pending, 0 failures. Running: `mci_original_only_binary/scratch` on GPU2 and `mci_original_only_binary/partial` on GPU4. |
| detox random rerun | Not started; waits for corrected MCI status to become idle. |
| first-version summary | `summary_first_version.*` refreshed: 32 rows, 9 missing final metrics. |
| enhanced first-version audit | `errors=[]`, `warnings=9`, `metrics_final_count=31`, `unresolved_failures=0`. |
| first-version missing finals | `pd_binary` scratch/full; `mci_original_only_binary` scratch/partial/full; all 4 `mci_matched_binary_random_seed20260621` modes. |
| GPU sample | GPU1/GPU2/GPU3/GPU4 utilization approximately 98%/97%/98%/88%; GPU0 is unrelated to this queue. |
| error scan | Current PD and MCI logs show no `Traceback`, `ERROR`, or `RuntimeError`. |

Active job snapshot at 2026-06-21 08:10 CST:

| task | mode | latest step epoch / loss | latest validation epoch / train loss / subject metric / F1 |
| --- | --- | --- | --- |
| `pd_binary` | `scratch` | epoch 24 / `0.15364` | epoch 23 / `0.16446` / `0.97178` / `0.90688` |
| `pd_binary` | `full` | epoch 9 / `0.03678` | epoch 8 / `0.04278` / `0.99409` / `0.95312` |
| `mci_original_only_binary` | `scratch` | epoch 12 / `0.52077` | epoch 11 / `0.53506` / `0.68182` / `0.55319` |
| `mci_original_only_binary` | `partial` | epoch 7 / `0.40148` | epoch 6 / `0.41851` / `0.7543` / `0.58333` |

Monitor update at 2026-06-21 08:12 CST:

| item | current state |
| --- | --- |
| recovery queue | 3 completed, 2 running, 0 pending, 0 failures. Running: `pd_binary/scratch` on GPU1 and `pd_binary/full` on GPU3. |
| corrected MCI queue | 1 completed, 2 running, 5 pending, 0 failures. Running: `mci_original_only_binary/scratch` on GPU2 and `mci_original_only_binary/partial` on GPU4. |
| detox random rerun | Not started; waits for corrected MCI status to become idle. |
| first-version summary | `summary_first_version.*` refreshed: 32 rows, 9 missing final metrics. |
| enhanced first-version audit | `errors=[]`, `warnings=9`, `metrics_final_count=31`, `unresolved_failures=0`. |
| first-version missing finals | `pd_binary` scratch/full; `mci_original_only_binary` scratch/partial/full; all 4 `mci_matched_binary_random_seed20260621` modes. |
| GPU sample | GPU1/GPU2/GPU3/GPU4 utilization approximately 95%/99%/97%/98%; GPU0 is unrelated to this queue. |
| error scan | Current PD and MCI logs show no `Traceback`, `ERROR`, or `RuntimeError`. |

Active job snapshot at 2026-06-21 08:12 CST:

| task | mode | latest step epoch / loss | latest validation epoch / train loss / subject metric / F1 |
| --- | --- | --- | --- |
| `pd_binary` | `scratch` | epoch 24 / `0.15512` | epoch 23 / `0.16446` / `0.97178` / `0.90688` |
| `pd_binary` | `full` | epoch 10 / `0.02376` | epoch 9 / `0.03678` / `0.99521` / `0.97656` |
| `mci_original_only_binary` | `scratch` | epoch 13 / `0.50863` | epoch 12 / `0.52301` / `0.67076` / `0.30303` |
| `mci_original_only_binary` | `partial` | epoch 8 / `0.37726` | epoch 8 / `0.37726` / `0.75553` / `0.58333` |

Monitor update at 2026-06-21 08:17 CST:

| item | current state |
| --- | --- |
| recovery queue | 3 completed, 2 running, 0 pending, 0 failures. Running: `pd_binary/scratch` on GPU1 and `pd_binary/full` on GPU3. |
| corrected MCI queue | 2 completed, 2 running, 4 pending, 0 failures. Completed now includes `mci_original_only_binary/scratch`; running: `mci_original_only_binary/partial` on GPU4 and `mci_original_only_binary/full` on GPU2. |
| detox random rerun | Not started; waits for corrected MCI status to become idle. |
| first-version summary | `summary_first_version.*` refreshed: 32 rows, 8 missing final metrics. |
| enhanced first-version audit | `errors=[]`, `warnings=8`, `metrics_final_count=32`, `unresolved_failures=0`. |
| first-version missing finals | `pd_binary` scratch/full; `mci_original_only_binary` partial/full; all 4 `mci_matched_binary_random_seed20260621` modes. |
| new final result | `mci_original_only_binary/scratch`: validation subject AUROC `0.76044`; final-best test metrics omitted by current reporting policy. |
| GPU sample | GPU1/GPU2/GPU3/GPU4 utilization approximately 95%/97%/96%/97%; GPU0 is unrelated to this queue. |
| error scan | Current PD and MCI logs show no `Traceback`, `ERROR`, or `RuntimeError`. |

Active job snapshot at 2026-06-21 08:17 CST:

| task | mode | latest step epoch / loss | latest validation epoch / train loss / subject metric / F1 |
| --- | --- | --- | --- |
| `pd_binary` | `scratch` | epoch 25 / `0.14205` | epoch 24 / `0.15546` / `0.97185` / `0.92063` |
| `pd_binary` | `full` | epoch 10 / `0.03054` | epoch 9 / `0.03678` / `0.99521` / `0.97656` |
| `mci_original_only_binary` | `partial` | epoch 12 / `0.30475` | epoch 11 / `0.32348` / `0.7371` / `0.47368` |
| `mci_original_only_binary` | `full` | epoch 0 / `0.60818` | not reached yet |

Monitor update at 2026-06-21 08:19 CST:

| item | current state |
| --- | --- |
| recovery queue | 3 completed, 2 running, 0 pending, 0 failures. Running: `pd_binary/scratch` on GPU1 and `pd_binary/full` on GPU3. |
| corrected MCI queue | 2 completed, 2 running, 4 pending, 0 failures. Running: `mci_original_only_binary/partial` on GPU4 and `mci_original_only_binary/full` on GPU2. |
| detox random rerun | Not started; waits for corrected MCI status to become idle. |
| first-version summary | `summary_first_version.*` refreshed: 32 rows, 8 missing final metrics. |
| enhanced first-version audit | `errors=[]`, `warnings=8`, `metrics_final_count=32`, `unresolved_failures=0`. |
| first-version missing finals | `pd_binary` scratch/full; `mci_original_only_binary` partial/full; all 4 `mci_matched_binary_random_seed20260621` modes. |
| GPU sample | GPU1/GPU2/GPU3/GPU4 utilization approximately 98%/97%/97%/96%; GPU0 is unrelated to this queue. |
| error scan | Current PD and MCI logs show no `Traceback`, `ERROR`, or `RuntimeError`. |

Active job snapshot at 2026-06-21 08:19 CST:

| task | mode | latest step epoch / loss | latest validation epoch / train loss / subject metric / F1 |
| --- | --- | --- | --- |
| `pd_binary` | `scratch` | epoch 25 / `0.14533` | epoch 24 / `0.15546` / `0.97185` / `0.92063` |
| `pd_binary` | `full` | epoch 10 / `0.03129` | epoch 9 / `0.03678` / `0.99521` / `0.97656` |
| `mci_original_only_binary` | `partial` | epoch 13 / `0.28590` | epoch 12 / `0.30715` / `0.74201` / `0.48649` |
| `mci_original_only_binary` | `full` | epoch 1 / `0.48185` | epoch 0 / `0.58256` / `0.74693` / `0.57778` |

Monitor update at 2026-06-21 08:21 CST:

| item | current state |
| --- | --- |
| recovery queue | 3 completed, 2 running, 0 pending, 0 failures. Running: `pd_binary/scratch` on GPU1 and `pd_binary/full` on GPU3. |
| corrected MCI queue | 3 completed, 2 running, 3 pending, 0 failures. Completed now includes `mci_original_only_binary/partial`; running: `mci_original_only_binary/full` on GPU2 and `mci_matched_binary_random_seed20260621/scratch` on GPU4. |
| detox random rerun | Not started; waits for corrected MCI status to become idle. |
| first-version summary | `summary_first_version.*` refreshed: 32 rows, 7 missing final metrics. |
| enhanced first-version audit | `errors=[]`, `warnings=7`, `metrics_final_count=33`, `unresolved_failures=0`. |
| first-version missing finals | `pd_binary` scratch/full; `mci_original_only_binary` full; `mci_matched_binary_random_seed20260621` scratch/linear_probe/partial/full. |
| new final result | `mci_original_only_binary/partial`: validation subject AUROC `0.78747`; final-best test metrics omitted by current reporting policy. |
| GPU sample | GPU1/GPU2/GPU3/GPU4 utilization approximately 98%/99%/97%/97%; GPU0 is unrelated to this queue. |
| error scan | Current PD and MCI logs show no `Traceback`, `ERROR`, or `RuntimeError`. |

Active job snapshot at 2026-06-21 08:21 CST:

| task | mode | latest step epoch / loss | latest validation epoch / train loss / subject metric / F1 |
| --- | --- | --- | --- |
| `pd_binary` | `scratch` | epoch 25 / `0.14742` | epoch 24 / `0.15546` / `0.97185` / `0.92063` |
| `pd_binary` | `full` | epoch 11 / `0.02764` | epoch 10 / `0.03142` / `0.99501` / `0.9551` |
| `mci_original_only_binary` | `full` | epoch 2 / `0.40515` | epoch 1 / `0.46504` / `0.74693` / `0.45` |
| `mci_matched_binary_random_seed20260621` | `scratch` | epoch 0 / `0.70503` | not reached yet |

Monitor update at 2026-06-21 08:23 CST:

| item | current state |
| --- | --- |
| recovery queue | 3 completed, 2 running, 0 pending, 0 failures. Running: `pd_binary/scratch` on GPU1 and `pd_binary/full` on GPU3. |
| corrected MCI queue | 3 completed, 2 running, 3 pending, 0 failures. Running: `mci_original_only_binary/full` on GPU2 and `mci_matched_binary_random_seed20260621/scratch` on GPU4. |
| detox random rerun | Not started; waits for corrected MCI status to become idle. |
| first-version summary | `summary_first_version.*` refreshed: 32 rows, 7 missing final metrics. |
| enhanced first-version audit | `errors=[]`, `warnings=7`, `metrics_final_count=33`, `unresolved_failures=0`. |
| first-version missing finals | `pd_binary` scratch/full; `mci_original_only_binary` full; `mci_matched_binary_random_seed20260621` scratch/linear_probe/partial/full. |
| GPU sample | GPU1/GPU2/GPU3/GPU4 utilization approximately 97%/99%/95%/97%; GPU0 is unrelated to this queue. |
| error scan | Current PD and MCI logs show no `Traceback`, `ERROR`, or `RuntimeError`. |

Active job snapshot at 2026-06-21 08:23 CST:

| task | mode | latest step epoch / loss | latest validation epoch / train loss / subject metric / F1 |
| --- | --- | --- | --- |
| `pd_binary` | `scratch` | epoch 26 / `0.13039` | epoch 25 / `0.14769` / `0.97454` / `0.90909` |
| `pd_binary` | `full` | epoch 11 / `0.02688` | epoch 10 / `0.03142` / `0.99501` / `0.9551` |
| `mci_original_only_binary` | `full` | epoch 2 / `0.38563` | epoch 2 / `0.38563` / `0.73219` / `0.4878` |
| `mci_matched_binary_random_seed20260621` | `scratch` | epoch 1 / `0.65406` | epoch 1 / `0.65255` / `0.53633` / `0.37037` |

Monitor update at 2026-06-21 08:25 CST:

| item | current state |
| --- | --- |
| recovery queue | 3 completed, 2 running, 0 pending, 0 failures. Running: `pd_binary/scratch` on GPU1 and `pd_binary/full` on GPU3. |
| corrected MCI queue | 3 completed, 2 running, 3 pending, 0 failures. Running: `mci_original_only_binary/full` on GPU2 and `mci_matched_binary_random_seed20260621/scratch` on GPU4. |
| detox random rerun | Not started; waits for corrected MCI status to become idle. |
| first-version summary | `summary_first_version.*` refreshed: 32 rows, 7 missing final metrics. |
| enhanced first-version audit | `errors=[]`, `warnings=7`, `metrics_final_count=33`, `unresolved_failures=0`. |
| first-version missing finals | `pd_binary` scratch/full; `mci_original_only_binary` full; `mci_matched_binary_random_seed20260621` scratch/linear_probe/partial/full. |
| GPU sample | GPU1/GPU2/GPU3/GPU4 utilization approximately 97%/98%/97%/99%; GPU0 is unrelated to this queue. |
| error scan | Current PD and MCI logs show no `Traceback`, `ERROR`, or `RuntimeError`. |

Active job snapshot at 2026-06-21 08:25 CST:

| task | mode | latest step epoch / loss | latest validation epoch / train loss / subject metric / F1 |
| --- | --- | --- | --- |
| `pd_binary` | `scratch` | epoch 26 / `0.13487` | epoch 25 / `0.14769` / `0.97454` / `0.90909` |
| `pd_binary` | `full` | epoch 11 / `0.02868` | epoch 10 / `0.03142` / `0.99501` / `0.9551` |
| `mci_original_only_binary` | `full` | epoch 3 / `0.32135` | epoch 3 / `0.32177` / `0.74201` / `0.45` |
| `mci_matched_binary_random_seed20260621` | `scratch` | epoch 3 / `0.60575` | epoch 2 / `0.63174` / `0.50865` / `0.46667` |

Monitor update at 2026-06-21 08:28 CST:

| item | current state |
| --- | --- |
| recovery queue | 3 completed, 2 running, 0 pending, 0 failures. Running: `pd_binary/scratch` on GPU1 and `pd_binary/full` on GPU3. |
| corrected MCI queue | 3 completed, 2 running, 3 pending, 0 failures. Running: `mci_original_only_binary/full` on GPU2 and `mci_matched_binary_random_seed20260621/scratch` on GPU4. |
| detox random rerun | Not started; waits for corrected MCI status to become idle. |
| first-version summary | `summary_first_version.*` refreshed: 32 rows, 7 missing final metrics. |
| enhanced first-version audit | `errors=[]`, `warnings=7`, `metrics_final_count=33`, `unresolved_failures=0`. |
| first-version missing finals | `pd_binary` scratch/full; `mci_original_only_binary` full; `mci_matched_binary_random_seed20260621` scratch/linear_probe/partial/full. |
| GPU sample | GPU1/GPU2/GPU3/GPU4 utilization approximately 96%/98%/99%/97%; GPU0 is unrelated to this queue. |
| error scan | Current PD and MCI logs show no `Traceback`, `ERROR`, or `RuntimeError`. |

Active job snapshot at 2026-06-21 08:28 CST:

| task | mode | latest step epoch / loss | latest validation epoch / train loss / subject metric / F1 |
| --- | --- | --- | --- |
| `pd_binary` | `scratch` | epoch 26 / `0.13687` | epoch 25 / `0.14769` / `0.97454` / `0.90909` |
| `pd_binary` | `full` | epoch 11 / `0.02896` | epoch 10 / `0.03142` / `0.99501` / `0.9551` |
| `mci_original_only_binary` | `full` | epoch 5 / `0.21992` | epoch 4 / `0.26397` / `0.74693` / `0.61224` |
| `mci_matched_binary_random_seed20260621` | `scratch` | epoch 5 / `0.56048` | epoch 4 / `0.57535` / `0.54325` / `0.51429` |

Monitor update at 2026-06-21 08:31 CST:

| item | current state |
| --- | --- |
| recovery queue | 3 completed, 2 running, 0 pending, 0 failures. Running: `pd_binary/scratch` on GPU1 and `pd_binary/full` on GPU3. |
| corrected MCI queue | 3 completed, 2 running, 3 pending, 0 failures. Running: `mci_original_only_binary/full` on GPU2 and `mci_matched_binary_random_seed20260621/scratch` on GPU4. |
| detox random rerun | Not started; waits for corrected MCI status to become idle. |
| first-version summary | `summary_first_version.*` refreshed: 32 rows, 7 missing final metrics. |
| enhanced first-version audit | `errors=[]`, `warnings=7`, `metrics_final_count=33`, `unresolved_failures=0`. |
| first-version missing finals | `pd_binary` scratch/full; `mci_original_only_binary` full; `mci_matched_binary_random_seed20260621` scratch/linear_probe/partial/full. |
| GPU sample | GPU1/GPU2/GPU3/GPU4 utilization approximately 98%/97%/98%/98%; GPU0 is unrelated to this queue. |
| error scan | Current PD and MCI logs show no `Traceback`, `ERROR`, or `RuntimeError`. |

Active job snapshot at 2026-06-21 08:31 CST:

| task | mode | latest step epoch / loss | latest validation epoch / train loss / subject metric / F1 |
| --- | --- | --- | --- |
| `pd_binary` | `scratch` | epoch 26 / `0.13876` | epoch 25 / `0.14769` / `0.97454` / `0.90909` |
| `pd_binary` | `full` | epoch 12 / `0.02668` | epoch 11 / `0.02903` / `0.99508` / `0.9685` |
| `mci_original_only_binary` | `full` | epoch 5 / `0.21711` | epoch 5 / `0.21697` / `0.75061` / `0.56522` |
| `mci_matched_binary_random_seed20260621` | `scratch` | epoch 6 / `0.53657` | epoch 5 / `0.55452` / `0.57093` / `0.61538` |

MCI label-anchor clarification at 2026-06-21 08:35 CST:

| item | status |
| --- | --- |
| user-intended MCI rule | Use original `MCI` raw subjects as the only label authority. `MCI匹配后` rows may contribute samples only when their raw `subject` exists in the original-MCI anchor; `MCI匹配后` source labels must never be used. |
| generated data verification | `MCI_original_only_no_matched`: 58,279 rows, 383 raw subjects, 0 matched-source rows, 0 label mismatches vs original anchor. `MCI匹配后_random_seed20260621`: 33,154 rows, 218 raw subjects, 0 outside-anchor subjects, 0 label mismatches vs original anchor. |
| source label conflict evidence | Original standalone `MCI匹配后` has 33,154 / 33,154 rows whose source label differs from the original-MCI anchor, so the corrected matched task overwrites all matched labels. |
| code/config impact | Existing corrected configs already point to `mci_original_only_binary` and `mci_matched_binary_random_seed20260621`; this check changed only `docs/downstream_v3_plan.md` wording, not the running jobs. |

Monitor update at 2026-06-21 08:39 CST:

| item | current state |
| --- | --- |
| recovery queue | 3 completed, 2 running, 0 pending, 0 failures. Running: `pd_binary/scratch` on GPU1 and `pd_binary/full` on GPU3. |
| corrected MCI queue | 3 completed, 2 running, 3 pending, 0 failures. Running: `mci_original_only_binary/full` on GPU2 and `mci_matched_binary_random_seed20260621/scratch` on GPU4. |
| detox random rerun | Not started; waits for corrected MCI status to become idle. |
| first-version summary | `summary_first_version.*` refreshed: 32 rows, 7 missing final metrics. |
| enhanced first-version audit | `errors=[]`, `warnings=7`, `metrics_final_count=33`, `unresolved_failures=0`. |
| first-version missing finals | `pd_binary` scratch/full; `mci_original_only_binary` full; `mci_matched_binary_random_seed20260621` scratch/linear_probe/partial/full. |
| GPU sample | GPU1/GPU2/GPU3/GPU4 utilization approximately 98%/96%/94%/97%; GPU0 is unrelated to this queue. |
| error scan | Current PD and MCI logs show no `Traceback`, `ERROR`, or `RuntimeError`. |

Active job snapshot at 2026-06-21 08:39 CST:

| task | mode | latest step epoch / loss | latest validation epoch / train loss / subject metric / F1 |
| --- | --- | --- | --- |
| `pd_binary` | `scratch` | epoch 27 / `0.13157` | epoch 26 / `0.13963` / `0.97133` / `0.90196` |
| `pd_binary` | `full` | epoch 12 / `0.02666` | epoch 11 / `0.02903` / `0.99508` / `0.9685` |
| `mci_original_only_binary` | `full` | epoch 8 / `0.12378` | epoch 8 / `0.12425` / `0.71007` / `0.46154` |
| `mci_matched_binary_random_seed20260621` | `scratch` | epoch 11 / `0.45023` | epoch 10 / `0.46411` / `0.6436` / `0.61905` |

Monitor update at 2026-06-21 08:43 CST:

| item | current state |
| --- | --- |
| recovery queue | 3 completed, 2 running, 0 pending, 0 failures. Running: `pd_binary/scratch` on GPU1 and `pd_binary/full` on GPU3. |
| corrected MCI queue | 3 completed, 2 running, 3 pending, 0 failures. Running: `mci_original_only_binary/full` on GPU2 and `mci_matched_binary_random_seed20260621/scratch` on GPU4. |
| detox random rerun | Not started; waits for corrected MCI status to become idle. |
| first-version missing finals | unchanged: `pd_binary` scratch/full; `mci_original_only_binary` full; `mci_matched_binary_random_seed20260621` scratch/linear_probe/partial/full. |
| GPU sample | GPU1/GPU2/GPU3/GPU4 utilization approximately 99%/100%/95%/96%; GPU0 is unrelated to this queue. |
| error scan | Current PD and MCI logs show no `Traceback`, `ERROR`, or `RuntimeError`. |

Active job snapshot at 2026-06-21 08:43 CST:

| task | mode | latest step epoch / loss | latest validation epoch / train loss / subject metric / F1 |
| --- | --- | --- | --- |
| `pd_binary` | `scratch` | epoch 28 / `0.12266` | epoch 27 / `0.13254` / `0.97185` / `0.91051` |
| `pd_binary` | `full` | epoch 12 / `0.02690` | epoch 12 / `0.02685` / `0.99587` / `0.96552` |
| `mci_original_only_binary` | `full` | epoch 10 / `0.07655` | epoch 9 / `0.09934` / `0.74447` / `0.4878` |
| `mci_matched_binary_random_seed20260621` | `scratch` | epoch 14 / `0.39533` | epoch 13 / `0.42243` / `0.70934` / `0.65116` |

Monitor update at 2026-06-21 08:52 CST:

| item | current state |
| --- | --- |
| new final result | `pd_binary/full`: validation subject AUROC `0.99698`; final-best test metrics omitted by current reporting policy. |
| first-version summary | `summary_first_version.*` refreshed: 32 rows, 6 missing final metrics. |
| enhanced first-version audit | `errors=[]`, `warnings=6`, `metrics_final_count=34`, `unresolved_failures=0`. |
| remaining first-version missing finals | `pd_binary` scratch; `mci_original_only_binary` full; `mci_matched_binary_random_seed20260621` scratch/linear_probe/partial/full. |
| recovery queue | 4 completed, 1 running, 0 pending, 0 failures. Running: `pd_binary/scratch` on GPU1. |
| corrected MCI queue restart | Old GPU2/GPU4-only MCI manager was replaced with `eyemae_downstream_v3_mci_followup_seed20260621_gpu234_resume_0849` using GPU2/GPU3/GPU4 and the same log/status directory. |
| corrected MCI queue after restart | 3 completed, 3 running, 2 pending, 0 failures. Running: `mci_original_only_binary/full` on GPU2 from `checkpoint_last.pt`, `mci_matched_binary_random_seed20260621/scratch` on GPU3 from `checkpoint_last.pt`, and `mci_matched_binary_random_seed20260621/linear_probe` on GPU4. |
| detox random rerun | Not started; waits for corrected MCI status to become idle. |
| GPU sample | GPU1/GPU2/GPU3/GPU4 utilization approximately 96%/98%/96%/74%; GPU0 is unrelated to this queue. |

Monitor update at 2026-06-21 08:55 CST:

| item | current state |
| --- | --- |
| recovery queue | 4 completed, 1 running, 0 pending, 0 failures. Running: `pd_binary/scratch` on GPU1. |
| corrected MCI queue | 3 completed, 3 running, 2 pending, 0 failures. Running: `mci_original_only_binary/full` on GPU2, `mci_matched_binary_random_seed20260621/scratch` on GPU3, and `mci_matched_binary_random_seed20260621/linear_probe` on GPU4. |
| restart verification | Both resumed MCI jobs and the newly launched matched linear-probe job are writing normal epoch logs after the GPU2/GPU3/GPU4 restart. |
| remaining first-version missing finals | unchanged: `pd_binary` scratch; `mci_original_only_binary` full; `mci_matched_binary_random_seed20260621` scratch/linear_probe/partial/full. |
| GPU sample | GPU1/GPU2/GPU3/GPU4 utilization approximately 97%/98%/99%/94%; GPU0 is unrelated to this queue. |
| error scan | Current PD and MCI logs show no `Traceback`, `ERROR`, or `RuntimeError`. |

Active job snapshot at 2026-06-21 08:55 CST:

| task | mode | latest step epoch / loss | latest validation epoch / train loss / subject metric / F1 |
| --- | --- | --- | --- |
| `pd_binary` | `scratch` | epoch 29 / `0.11755` | epoch 28 / `0.12929` / `0.97513` / `0.90347` |
| `mci_original_only_binary` | `full` | epoch 14 / `0.04725` | epoch 13 / `0.05401` / `0.7285` / `0.47619` |
| `mci_matched_binary_random_seed20260621` | `scratch` | epoch 21 / `0.32369` | epoch 20 / `0.33767` / `0.63322` / `0.66667` |
| `mci_matched_binary_random_seed20260621` | `linear_probe` | epoch 6 / `0.57298` | epoch 5 / `0.57942` / `0.609` / `0.5` |

Monitor update at 2026-06-21 09:01 CST:

| item | current state |
| --- | --- |
| recovery queue | 4 completed, 1 running, 0 pending, 0 failures. Running: `pd_binary/scratch` on GPU1. |
| corrected MCI queue | 3 completed, 3 running, 2 pending, 0 failures. Running: `mci_original_only_binary/full` on GPU2, `mci_matched_binary_random_seed20260621/scratch` on GPU3, and `mci_matched_binary_random_seed20260621/linear_probe` on GPU4. |
| remaining first-version missing finals | unchanged: `pd_binary` scratch; `mci_original_only_binary` full; `mci_matched_binary_random_seed20260621` scratch/linear_probe/partial/full. |
| GPU sample | GPU1/GPU2/GPU3/GPU4 utilization approximately 99%/99%/97%/97%; GPU0 is unrelated to this queue. |
| error scan | Current PD and MCI logs show no `Traceback`, `ERROR`, or `RuntimeError`. |

Active job snapshot at 2026-06-21 09:01 CST:

| task | mode | latest step epoch / loss | latest validation epoch / train loss / subject metric / F1 |
| --- | --- | --- | --- |
| `pd_binary` | `scratch` | epoch 30 / `0.11414` | epoch 29 / `0.12674` / `0.97861` / `0.9243` |
| `mci_original_only_binary` | `full` | epoch 16 / `0.03564` | epoch 15 / `0.04223` / `0.74201` / `0.5` |
| `mci_matched_binary_random_seed20260621` | `scratch` | epoch 24 / `0.30202` | epoch 24 / `0.30281` / `0.67128` / `0.66667` |
| `mci_matched_binary_random_seed20260621` | `linear_probe` | epoch 17 / `0.50961` | epoch 16 / `0.50745` / `0.54325` / `0.48485` |

Monitor update at 2026-06-21 09:09 CST:

| item | current state |
| --- | --- |
| new final result | `mci_matched_binary_random_seed20260621/scratch`: validation subject AUROC `0.70934`; final-best test metrics omitted by current reporting policy. |
| first-version summary | `summary_first_version.*` refreshed: 32 rows, 5 missing final metrics. |
| enhanced first-version audit | `errors=[]`, `warnings=5`, `metrics_final_count=35`, `unresolved_failures=0`. |
| remaining first-version missing finals | `pd_binary` scratch; `mci_original_only_binary` full; `mci_matched_binary_random_seed20260621` linear_probe/partial/full. |
| recovery queue | 4 completed, 1 running, 0 pending, 0 failures. Running: `pd_binary/scratch` on GPU1. |
| corrected MCI queue | 4 completed, 3 running, 1 pending, 0 failures. Running: `mci_original_only_binary/full` on GPU2, `mci_matched_binary_random_seed20260621/linear_probe` on GPU4, and `mci_matched_binary_random_seed20260621/partial` on GPU3. |
| detox random rerun | Not started; waits for corrected MCI status to become idle. |
| GPU sample | GPU1/GPU2/GPU3/GPU4 utilization approximately 98%/97%/98%/98%; GPU0 is unrelated to this queue. |
| error scan | Current PD and MCI logs show no `Traceback`, `ERROR`, or `RuntimeError`. |

Active job snapshot at 2026-06-21 09:09 CST:

| task | mode | latest step epoch / loss | latest validation epoch / train loss / subject metric / F1 |
| --- | --- | --- | --- |
| `pd_binary` | `scratch` | epoch 30 / `0.12246` | epoch 29 / `0.12674` / `0.97861` / `0.9243` |
| `mci_original_only_binary` | `full` | epoch 18 / `0.02886` | epoch 17 / `0.03714` / `0.73096` / `0.4878` |
| `mci_matched_binary_random_seed20260621` | `linear_probe` | epoch 29 / `0.46411` | epoch 28 / `0.46673` / `0.62284` / `0.57895` |
| `mci_matched_binary_random_seed20260621` | `partial` | launched on GPU3 | not reached yet |

Monitor update at 2026-06-21 09:16 CST:

| item | current state |
| --- | --- |
| new final result | `mci_matched_binary_random_seed20260621/linear_probe`: validation subject AUROC `0.63668`; final-best test metrics omitted by current reporting policy. |
| first-version summary | `summary_first_version.*` refreshed: 32 rows, 4 missing final metrics. |
| enhanced first-version audit | `errors=[]`, `warnings=4`, `metrics_final_count=36`, `unresolved_failures=0`. |
| remaining first-version missing finals | `pd_binary` scratch; `mci_original_only_binary` full; `mci_matched_binary_random_seed20260621` partial/full. |
| recovery queue | 4 completed, 1 running, 0 pending, 0 failures. Running: `pd_binary/scratch` on GPU1. |
| corrected MCI queue | 5 completed, 3 running, 0 pending, 0 failures. Running: `mci_original_only_binary/full` on GPU2, `mci_matched_binary_random_seed20260621/partial` on GPU3, and `mci_matched_binary_random_seed20260621/full` on GPU4. |
| detox random rerun | Not started; waits for corrected MCI status to become idle. |
| GPU sample | GPU1/GPU2/GPU3/GPU4 utilization approximately 96%/96%/4%/98%; GPU3 low sample likely caught validation or IO in `mci_matched/partial`; GPU0 is unrelated to this queue. |
| error scan | Current PD and MCI logs show no `Traceback`, `ERROR`, or `RuntimeError`. |

Active job snapshot at 2026-06-21 09:16 CST:

| task | mode | latest step epoch / loss | latest validation epoch / train loss / subject metric / F1 |
| --- | --- | --- | --- |
| `pd_binary` | `scratch` | epoch 31 / `0.11703` | epoch 30 / `0.12403` / `0.9769` / `0.92607` |
| `mci_original_only_binary` | `full` | epoch 22 / `0.02227` | epoch 21 / `0.02469` / `0.73833` / `0.55319` |
| `mci_matched_binary_random_seed20260621` | `partial` | epoch 8 / `0.36307` | epoch 8 / `0.36349` / `0.61246` / `0.62857` |
| `mci_matched_binary_random_seed20260621` | `full` | epoch 3 / `0.32877` | epoch 2 / `0.38919` / `0.61938` / `0.63158` |

Monitor update at 2026-06-21 09:23 CST:

| item | current state |
| --- | --- |
| recovery queue | 4 completed, 1 running, 0 pending, 0 failures. Running: `pd_binary/scratch` on GPU1. |
| corrected MCI queue | 5 completed, 3 running, 0 pending, 0 failures. Running: `mci_original_only_binary/full` on GPU2, `mci_matched_binary_random_seed20260621/partial` on GPU3, and `mci_matched_binary_random_seed20260621/full` on GPU4. |
| remaining first-version missing finals | unchanged: `pd_binary` scratch; `mci_original_only_binary` full; `mci_matched_binary_random_seed20260621` partial/full. |
| GPU sample | GPU1/GPU2/GPU3/GPU4 utilization approximately 96%/96%/98%/98%; GPU0 is unrelated to this queue. |
| error scan | Current PD and MCI logs show no `Traceback`, `ERROR`, or `RuntimeError`. |

Active job snapshot at 2026-06-21 09:23 CST:

| task | mode | latest step epoch / loss | latest validation epoch / train loss / subject metric / F1 |
| --- | --- | --- | --- |
| `pd_binary` | `scratch` | epoch 32 / `0.10208` | epoch 31 / `0.11724` / `0.97493` / `0.91566` |
| `mci_original_only_binary` | `full` | epoch 24 / `0.02390` | epoch 23 / `0.02431` / `0.74693` / `0.5` |
| `mci_matched_binary_random_seed20260621` | `partial` | epoch 16 / `0.24377` | epoch 15 / `0.26030` / `0.65398` / `0.7` |
| `mci_matched_binary_random_seed20260621` | `full` | epoch 8 / `0.14188` | epoch 7 / `0.16591` / `0.59862` / `0.66667` |

Monitor update at 2026-06-21 09:31 CST:

| item | current state |
| --- | --- |
| new final result | `mci_original_only_binary/full`: validation subject AUROC `0.75184`; final-best test metrics omitted by current reporting policy. |
| new final result | `mci_matched_binary_random_seed20260621/partial`: validation subject AUROC `0.67474`; final-best test metrics omitted by current reporting policy. |
| new final result | `mci_matched_binary_random_seed20260621/full`: validation subject AUROC `0.66436`; final-best test metrics omitted by current reporting policy. |
| first-version summary | `summary_first_version.*` refreshed: 32 rows, 1 missing final metric. |
| enhanced first-version audit | `errors=[]`, `warnings=1`, `metrics_final_count=39`, `unresolved_failures=0`. |
| remaining first-version missing final | `pd_binary/scratch` only. |
| corrected MCI queue | 8 completed, 0 running, 0 pending, 0 failures. |
| recovery queue | 4 completed, 1 running, 0 pending, 0 failures. Running: `pd_binary/scratch` on GPU1. |
| detox random rerun | Started automatically after corrected MCI became idle. Running `scratch`, `linear_probe`, `partial`, and `full` on GPU1/GPU2/GPU3/GPU4 respectively. GPU1 is shared with `pd_binary/scratch`; no OOM or error observed. |
| GPU sample | GPU1/GPU2/GPU3/GPU4 utilization approximately 100%/93%/96%/98%; GPU0 is unrelated to this queue. |
| error scan | Current PD, MCI, and detox-random logs show no `Traceback`, `ERROR`, or `RuntimeError`. |

Monitor update at 2026-06-21 09:38 CST:

| item | current state |
| --- | --- |
| first-version official status | 31 / 32 final metrics complete; only `pd_binary/scratch` remains missing. |
| recovery queue | 4 completed, 1 running, 0 pending, 0 failures. Running: `pd_binary/scratch` on GPU1. |
| detox random rerun | 2 completed, 2 running, 0 pending, 0 failures for the new detox-random jobs. Completed: `linear_probe`, `partial`; running: `scratch` on GPU1 and `full` on GPU4. |
| detox random final result | `linear_probe`: final-best test metrics omitted by current reporting policy. |
| detox random final result | `partial`: final-best test metrics omitted by current reporting policy. |
| GPU sample | GPU1/GPU2/GPU3/GPU4 utilization approximately 100%/0%/0%/97%; GPU1 is shared by `pd_binary/scratch` and detox-random `scratch`; GPU0 is unrelated to this queue. |
| error scan | Current PD and detox-random logs show no `Traceback`, `ERROR`, or `RuntimeError`. |

Active job snapshot at 2026-06-21 09:38 CST:

| task | mode | latest step epoch / loss | latest validation epoch / train loss / subject metric / F1 |
| --- | --- | --- | --- |
| `pd_binary` | `scratch` | epoch 33 / `0.10119` | epoch 32 / `0.10925` / `0.97618` / `0.9127` |
| `detox_binary_random_seed20260621` | `scratch` | epoch 3 / `0.55472` | epoch 2 / `0.56106` / `0.79545` / `0.61538` |
| `detox_binary_random_seed20260621` | `full` | epoch 6 / `0.05331` | epoch 6 / `0.05349` / `0.84091` / `0.63636` |

Monitor update at 2026-06-21 09:44 CST:

| item | current state |
| --- | --- |
| first-version official status | 31 / 32 final metrics complete; only `pd_binary/scratch` remains missing. |
| recovery queue | 4 completed, 1 running, 0 pending, 0 failures. Running: `pd_binary/scratch` on GPU1. |
| detox random rerun | 3 completed, 1 running, 0 pending, 0 failures for the new detox-random jobs. Completed: `linear_probe`, `partial`, `full`; running: `scratch` on GPU1. |
| detox random final result | `full`: final-best test metrics omitted by current reporting policy. |
| GPU sample | GPU1 utilization approximately 100% with 73,347 MiB used; GPU2/GPU3/GPU4 are idle. GPU1 is shared by `pd_binary/scratch` and detox-random `scratch`; GPU0 is unrelated to this queue. |
| error scan | Current PD and detox-random logs show no `Traceback`, `ERROR`, or `RuntimeError`. |

Active job snapshot at 2026-06-21 09:44 CST:

| task | mode | latest step epoch / loss | latest validation epoch / train loss / subject metric / F1 |
| --- | --- | --- | --- |
| `pd_binary` | `scratch` | epoch 33 / `0.10524` | epoch 32 / `0.10925` / `0.97618` / `0.9127` |
| `detox_binary_random_seed20260621` | `scratch` | epoch 5 / `0.46620` | epoch 4 / `0.48792` / `0.84659` / `0.69231` |

Monitor update at 2026-06-21 09:49 CST:

| item | current state |
| --- | --- |
| first-version official status | 31 / 32 final metrics complete; only `pd_binary/scratch` remains missing. |
| recovery queue | 4 completed, 1 running, 0 pending, 0 failures. Running: `pd_binary/scratch` on GPU1. |
| detox random rerun | 3 completed, 1 running, 0 pending, 0 failures. Running: `scratch` on GPU1. |
| GPU sample | GPU1 utilization approximately 99% with 73,347 MiB used; GPU2/GPU3/GPU4 are idle. GPU1 is shared by `pd_binary/scratch` and detox-random `scratch`; GPU0 is unrelated to this queue. |
| error scan | Current PD and detox-random logs show no `Traceback`, `ERROR`, or `RuntimeError`. |

Active job snapshot at 2026-06-21 09:49 CST:

| task | mode | latest step epoch / loss | latest validation epoch / train loss / subject metric / F1 |
| --- | --- | --- | --- |
| `pd_binary` | `scratch` | epoch 34 / `0.09634` | epoch 33 / `0.10623` / `0.97454` / `0.91129` |
| `detox_binary_random_seed20260621` | `scratch` | epoch 8 / `0.38374` | epoch 7 / `0.41519` / `0.88068` / `0.66667` |

Monitor update at 2026-06-21 09:56 CST:

| item | current state |
| --- | --- |
| first-version official status | 31 / 32 final metrics complete; only `pd_binary/scratch` remains missing. |
| recovery queue | 4 completed, 1 running, 0 pending, 0 failures. Running: `pd_binary/scratch` on GPU1. |
| detox random rerun | 3 completed, 1 running, 0 pending, 0 failures. Running: `scratch` on GPU1. |
| GPU sample | GPU1 utilization approximately 100% with 73,347 MiB used; GPU2/GPU3/GPU4 are idle. GPU1 is shared by `pd_binary/scratch` and detox-random `scratch`; GPU0 is unrelated to this queue. |
| error scan | Current PD and detox-random logs show no `Traceback`, `ERROR`, or `RuntimeError`. |

Active job snapshot at 2026-06-21 09:56 CST:

| task | mode | latest step epoch / loss | latest validation epoch / train loss / subject metric / F1 |
| --- | --- | --- | --- |
| `pd_binary` | `scratch` | epoch 34 / `0.09909` | epoch 33 / `0.10623` / `0.97454` / `0.91129` |
| `detox_binary_random_seed20260621` | `scratch` | epoch 10 / `0.34925` | epoch 9 / `0.36942` / `0.91477` / `0.73684` |

Monitor update at 2026-06-21 10:02 CST:

| item | current state |
| --- | --- |
| first-version official status | 31 / 32 final metrics complete; only `pd_binary/scratch` remains missing. |
| recovery queue | 4 completed, 1 running, 0 pending, 0 failures. Running: `pd_binary/scratch` on GPU1. |
| detox random rerun | 3 completed, 1 running, 0 pending, 0 failures. Running: `scratch` on GPU1. |
| GPU sample | GPU1 utilization approximately 100% with 73,347 MiB used; GPU2/GPU3/GPU4 are idle. GPU1 is shared by `pd_binary/scratch` and detox-random `scratch`; GPU0 is unrelated to this queue. |
| error scan | Current PD and detox-random logs show no `Traceback`, `ERROR`, or `RuntimeError`. |

Active job snapshot at 2026-06-21 10:02 CST:

| task | mode | latest step epoch / loss | latest validation epoch / train loss / subject metric / F1 |
| --- | --- | --- | --- |
| `pd_binary` | `scratch` | epoch 34 / `0.10156` | epoch 33 / `0.10623` / `0.97454` / `0.91129` |
| `detox_binary_random_seed20260621` | `scratch` | epoch 13 / `0.30040` | epoch 12 / `0.31956` / `0.80114` / `0.63158` |

Monitor update at 2026-06-21 10:08 CST:

| item | current state |
| --- | --- |
| first-version official status | 31 / 32 final metrics complete; only `pd_binary/scratch` remains missing. |
| recovery queue | 4 completed, 1 running, 0 pending, 0 failures. Running: `pd_binary/scratch` on GPU1. |
| detox random rerun | 3 completed, 1 running, 0 pending, 0 failures. Running: `scratch` on GPU1. |
| GPU sample | GPU1 utilization approximately 100% with 73,347 MiB used; GPU2/GPU3/GPU4 are idle. GPU1 is shared by `pd_binary/scratch` and detox-random `scratch`; GPU0 is unrelated to this queue. |
| error scan | Current PD and detox-random logs show no `Traceback`, `ERROR`, or `RuntimeError`. |

Active job snapshot at 2026-06-21 10:08 CST:

| task | mode | latest step epoch / loss | latest validation epoch / train loss / subject metric / F1 |
| --- | --- | --- | --- |
| `pd_binary` | `scratch` | epoch 34 / `0.10305` | epoch 33 / `0.10623` / `0.97454` / `0.91129` |
| `detox_binary_random_seed20260621` | `scratch` | epoch 16 / `0.24804` | epoch 15 / `0.27589` / `0.875` / `0.72` |

Monitor update at 2026-06-21 10:13 CST:

| item | current state |
| --- | --- |
| first-version official status | 31 / 32 final metrics complete; only `pd_binary/scratch` remains missing. |
| recovery queue | 4 completed, 1 running, 0 pending, 0 failures. Running: `pd_binary/scratch` on GPU1. |
| detox random rerun | 3 completed, 1 running, 0 pending, 0 failures. Running: `scratch` on GPU1. |
| GPU sample | GPU1 utilization approximately 100% with 73,347 MiB used; GPU2/GPU3/GPU4 are idle. GPU1 is shared by `pd_binary/scratch` and detox-random `scratch`; GPU0 is unrelated to this queue. |
| error scan | Current PD and detox-random logs show no `Traceback`, `ERROR`, or `RuntimeError`. |

Active job snapshot at 2026-06-21 10:13 CST:

| task | mode | latest step epoch / loss | latest validation epoch / train loss / subject metric / F1 |
| --- | --- | --- | --- |
| `pd_binary` | `scratch` | epoch 35 / `0.09272` | epoch 34 / `0.10290` / `0.97073` / `0.91603` |
| `detox_binary_random_seed20260621` | `scratch` | epoch 18 / `0.23628` | epoch 17 / `0.25265` / `0.85795` / `0.63636` |

Monitor update at 2026-06-21 10:22 CST:

| item | current state |
| --- | --- |
| first-version official status | 31 / 32 final metrics complete; only `pd_binary/scratch` remains missing. |
| recovery queue | 4 completed, 1 running, 0 pending, 0 failures. Running: `pd_binary/scratch` on GPU1. |
| detox random rerun | 4 completed, 0 running, 0 pending, 0 failures. GPU1 is no longer shared by detox-random. |
| GPU sample | GPU1 utilization approximately 94% with 37,047 MiB used; GPU2/GPU3/GPU4 are idle. GPU0 is unrelated to this queue. |
| error scan | Current PD and detox-random logs show no `Traceback`, `ERROR`, or `RuntimeError`. |

Detox random final results at 2026-06-21 10:22 CST:

| mode | best epoch | val subject AUROC | test subject AUROC | test balanced accuracy | test weighted F1 | test Cohen's kappa |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| scratch | 9 | `0.91477` | `0.86071` | `0.78214` | `0.79282` | `0.57040` |
| linear_probe | 1 | `0.91477` | `0.82143` | `0.82857` | `0.82477` | `0.64336` |
| partial | 0 | `0.90341` | `0.82500` | `0.70714` | `0.70795` | `0.40559` |
| full | 0 | `0.88068` | `0.85714` | `0.75714` | `0.76471` | `0.51429` |

Active job snapshot at 2026-06-21 10:22 CST:

| task | mode | latest step epoch / loss | latest validation epoch / train loss / subject metric / F1 |
| --- | --- | --- | --- |
| `pd_binary` | `scratch` | epoch 35 / `0.10165` | epoch 34 / `0.10290` / `0.97073` / `0.91603` |

Monitor update at 2026-06-21 10:28 CST:

| item | current state |
| --- | --- |
| first-version official status | 31 / 32 final metrics complete; only `pd_binary/scratch` remains missing. |
| recovery queue | 4 completed, 1 running, 0 pending, 0 failures. Running: `pd_binary/scratch` on GPU1. |
| detox random rerun | 4 completed, 0 running, 0 pending, 0 failures. |
| GPU sample | GPU1 utilization approximately 96% with 37,047 MiB used; GPU2/GPU3/GPU4 are idle. GPU0 is unrelated to this queue. |
| error scan | Current PD log shows no `Traceback`, `ERROR`, or `RuntimeError`. |

Active job snapshot at 2026-06-21 10:28 CST:

| task | mode | latest step epoch / loss | latest validation epoch / train loss / subject metric / F1 |
| --- | --- | --- | --- |
| `pd_binary` | `scratch` | epoch 36 / `0.09211` | epoch 34 / `0.10290` / `0.97073` / `0.91603` |

Monitor update at 2026-06-21 10:34 CST:

| item | current state |
| --- | --- |
| first-version official status | 31 / 32 final metrics complete; only `pd_binary/scratch` remains missing. |
| recovery queue | 4 completed, 1 running, 0 pending, 0 failures. Running: `pd_binary/scratch` on GPU1. |
| GPU sample | GPU1 utilization approximately 95% with 37,047 MiB used; GPU2/GPU3/GPU4 are idle. GPU0 is unrelated to this queue. |
| error scan | Current PD log shows no `Traceback`, `ERROR`, or `RuntimeError`. |

Active job snapshot at 2026-06-21 10:34 CST:

| task | mode | latest step epoch / loss | latest validation epoch / train loss / subject metric / F1 |
| --- | --- | --- | --- |
| `pd_binary` | `scratch` | epoch 37 / `0.08357` | epoch 36 / `0.09510` / `0.97493` / `0.90837` |

Monitor update at 2026-06-21 10:45 CST:

| item | current state |
| --- | --- |
| first-version official status | 31 / 32 final metrics complete; only `pd_binary/scratch` remains missing. |
| recovery queue | 4 completed, 1 running, 0 pending, 0 failures. Running: `pd_binary/scratch` on GPU1. |
| GPU sample | GPU1 utilization approximately 96% with 37,047 MiB used; GPU2/GPU3/GPU4 are idle. GPU0 is unrelated to this queue. |
| error scan | Current PD log shows no `Traceback`, `ERROR`, or `RuntimeError`. |

Active job snapshot at 2026-06-21 10:45 CST:

| task | mode | latest step epoch / loss | latest validation epoch / train loss / subject metric / F1 |
| --- | --- | --- | --- |
| `pd_binary` | `scratch` | epoch 38 / `0.08427` | epoch 36 / `0.09510` / `0.97493` / `0.90837` |

Final completion update at 2026-06-21 11:08 CST:

| item | final state |
| --- | --- |
| final official result | `pd_binary/scratch`: validation subject AUROC `0.97861`; final-best test metrics omitted by current reporting policy. |
| first-version summary | `summary_first_version.*` refreshed: 32 rows, 0 missing final metrics. |
| strict goal audit | `official_metrics_final_count=32`, `metrics_final_count=44`, `extra_metrics_final_count=12`, `errors=[]`, `warnings=[]`. Extra metrics are from follow-up reruns such as detox-random and are allowed by official-scope audit. |
| recovery queue | 5 completed, 0 running, 0 pending, 0 failures. |
| corrected MCI queue | 8 completed, 0 running, 0 pending, 0 failures. |
| detox random rerun | 4/4 new detox-random jobs completed, 0 running, 0 pending, 0 failures. |
| process check | No host `eyemae.finetune`, `run_downstream_v3_queue.py`, or `wait_and_run_downstream_v3_queue.py` process remains for these queues. |
| GPU sample | GPU1/GPU2/GPU3/GPU4 are idle for this workload; GPU0 remains occupied by an unrelated process. |
| tests | `pytest -q tests`: 39 passed. |

MCI matched label-fixed rerun launch at 2026-06-21 12:14 CST:

| item | current state |
| --- | --- |
| reason | `MCI匹配后` healthy/disease label direction was confirmed to be reversed relative to the previous original-anchor relabeling assumption. The old `mci_matched_binary_random_seed20260621` results are retained as historical/provisional, not as the final matched-MCI interpretation. |
| new task | `mci_matched_binary_random_seed20260622_label_fixed` |
| new view | `downstream/MCI匹配后_random_seed20260622_label_fixed/` |
| split policy | fresh subject-level stratified random split, `seed=20260622`; old seed-20260621 train/validation/test assignment is not reused. |
| label policy | keep a matched row only if raw `subject` exists in the original-MCI anchor; final `health_label` is the inverse of that original-MCI anchor label. |
| data audit | 33,154 rows, 218 raw subjects, 0 unmapped rows, 0 subject/split overlaps, 0 raw-subject/file/trial label conflicts. |
| subject split | train 140 subjects (70/70), validation 34 subjects (17/17), test 44 subjects (22/22). |
| trial split | train 21,454 rows (`0:10477`, `1:10977`), validation 5,201 rows (`0:2586`, `1:2615`), test 6,499 rows (`0:3371`, `1:3128`). |
| configs | `configs/downstream/mci_matched_binary_random_seed20260622_label_fixed_{scratch,linear_probe,partial,full}.yaml` |
| queue | `outputs/downstream_v3_fast_logs/run_20260622_mci_matched_label_fixed/status.json`: 4 running, 0 pending, 0 completed at launch. |
| GPU assignment | scratch on GPU1, linear_probe on GPU2, partial on GPU3, full on GPU4. |
| tests | `python -m py_compile scripts/prepare_mci_followup_finetune.py` passed; `pytest -q tests/test_downstream.py tests/test_downstream_packed.py`: 12 passed. |

Active label-fixed MCI snapshot at 2026-06-21 12:15 CST:

| task | mode | latest step epoch / loss | latest validation epoch / train loss / subject metric / F1 |
| --- | --- | --- | --- |
| `mci_matched_binary_random_seed20260622_label_fixed` | `scratch` | epoch 0 / `0.69782` | not reached yet |
| `mci_matched_binary_random_seed20260622_label_fixed` | `linear_probe` | epoch 1 / `0.61391` | epoch 0 / `0.64683` / `0.54325` / `0.25000` |
| `mci_matched_binary_random_seed20260622_label_fixed` | `partial` | epoch 0 / `0.63975` | epoch 0 / `0.63704` / `0.52595` / `0.48276` |
| `mci_matched_binary_random_seed20260622_label_fixed` | `full` | epoch 0 / `0.63083` | not reached yet |

MCI matched label-fixed final results at 2026-06-21 12:38 CST:

| mode | best epoch | val subject AUROC | test subject AUROC | test balanced accuracy | test weighted F1 | test Cohen's kappa | test trial AUROC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| scratch | 4 | `0.86851` | `0.60537` | `0.61364` | `0.61183` | `0.22727` | `0.55800` |
| linear_probe | 3 | `0.67820` | `0.53926` | `0.50000` | `0.44368` | `0.00000` | `0.50787` |
| partial | 7 | `0.79585` | `0.65289` | `0.56818` | `0.56617` | `0.13636` | `0.57536` |
| full | 3 | `0.76471` | `0.63017` | `0.61364` | `0.61344` | `0.22727` | `0.56331` |

Final label-fixed MCI run status:

| item | final state |
| --- | --- |
| queue | `outputs/downstream_v3_fast_logs/run_20260622_mci_matched_label_fixed/status.json`: 4 completed, 0 running, 0 pending, 0 failures. |
| early stopping | linear_probe stopped at epoch 13, partial at epoch 17, full at epoch 13, scratch at epoch 14. |
| best validation choice | checkpoint selected by `val/subject/auroc`; scratch has highest validation AUROC, partial has highest test subject AUROC. |
| error scan | No `Traceback`, `ERROR`, `RuntimeError`, `CUDA out of memory`, or `nan` found in the label-fixed run logs. |
| tests | `python -m py_compile scripts/prepare_mci_followup_finetune.py scripts/run_downstream_v3_queue.py` passed; `pytest -q tests/test_downstream.py tests/test_downstream_packed.py`: 12 passed. |

All-completed convergence/test summary at 2026-06-21 12:58 CST:

| item | final state |
| --- | --- |
| summary artifacts | `outputs/downstream_v3_fast/all_downstream_best_epoch1_within30_summary_20260621/summary_all_completed_best_epoch1_within30.{md,csv,json}` |
| scope | 32 reportable downstream mode outputs are included. Known bad-label or superseded MCI rows and the detox random-resplit sensitivity run are filtered out of the reporting table. |
| main metric | Binary tasks use subject-level AUROC; multiclass tasks use subject-level macro AUROC OVR. The table also reports 30-epoch balanced accuracy, weighted F1, and Cohen's kappa where available. Overall `best_epoch` and overall final-best test metrics are intentionally not reported. |
| epoch1 policy | Epoch1 reports validation metrics only. No epoch1 test metric is used in this summary. |
| 30-epoch policy | The 30-epoch report uses the best tested checkpoint within epochs 0-29 when available. `pd_related_5class/scratch` uses the previously recorded under-30 test result at epoch 20 from `outputs/downstream_v3_fast_current_best_test_eval/pd_related_5class_random_seed20260620/scratch/metrics.json`. `epilepsy_binary/partial` uses the approved epoch 45 exception. |
| completeness | `epoch1_val_main` and `test_main_30ep` have no missing values across the 32 reported rows. |
