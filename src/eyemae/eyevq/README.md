# EyeVQ core

The maintained entry points are `tokenizer.train`, `precompute_codes`,
`pretrain.train`, `downstream.train_mil`, `downstream.evaluate_mil` and `pipeline`.
See the repository README and `configs/eyevq/final/` for the 12-layer reference.

`artifacts.py` owns data/config/checkpoint identity checks. `manual_features.py`
loads precomputed tokenizer supervision. `downstream/runtime.py` contains shared
distributed setup and LR schedules, independent of legacy model implementations.
