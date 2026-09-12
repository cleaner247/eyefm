# EyeFM / EyeVQ

Eye-movement representation learning: FSQ tokenizer → paired masked-code BERT
pretraining → subject-level multi-instance fine-tuning. The Python package keeps
the historical `eyemae` name for checkpoint/import compatibility; legacy EyeMAE
training and external comparison methods are not included.

## Installation

Python 3.10+ and PyTorch are required. From the repository root:

```bash
python -m pip install -e '.[test]'
python -m pytest tests -q
```

## Twelve-layer reference

Only one model-size recipe is published in `configs/eyevq/final/`:

| Stage | Reference setting |
| --- | --- |
| Tokenizer | 12 × 384 encoder, 8 heads, FFN 1152; 3-layer decoder; tanh FSQ [9,7,5,5]; 40K steps |
| BERT | 12 × 384, 8 heads, FFN 1152; factorized CE; 20K steps |
| Masking | Independent short (15 blocks, lengths 1–3) and long (5 blocks, lengths 4–6) views; valid paired eyes only |
| Fine-tuning | All embeddings/layers trainable; K16 per task; 4 subjects/GPU; LR 5e-5 → 5e-6 |
| MCI / other binary tasks | 10 epochs; subject-weighted BCE |
| PD3 | 15 epochs; two sigmoid heads; equal head losses with within-head subject inverse-frequency weights |

The tokenizer uses 40-sample non-overlapping patches, stimulus-isolated attention
(stimulus cannot read CLS or eyes), per-subject/per-eye area normalization and
38-dimensional manual-feature supervision. BERT duplicates each batch into
independently masked short/long views; `max_trials_per_gpu: 64` is before that
duplication. Factorized CE is summed across FSQ dimensions and reduced per trial.

## Data and execution

Clinical data, subject split manifests, normalization statistics, manual-feature
caches, checkpoints and experiment results are external inputs, not shipped.
The checked-in recipe pins the existing V6 dataset and derived artifacts. Its
absolute paths describe the reference machine. On another machine, provision
the matching inputs and update paths consistently in the YAMLs and artifact
manifest, then recompute the manifest SHA256 in `recipe.yaml`. Do not bypass hash
checks or reuse checkpoints after changing data/configuration.

Run from the repository root. The default pipeline runs MCI and PD3 for seeds
42/43/44. Pretraining uses four GPUs; fine-tuning uses one GPU to preserve the
reference batch size. `CUDA_VISIBLE_DEVICES` selects the devices.

```bash
bash scripts/run_eyevq_final.sh --preflight-only
bash scripts/run_eyevq_final.sh
# Optional detached execution (requires tmux):
bash scripts/run_eyevq_final.sh --detached
```

`PYTHON_ENV` defaults to the current Python environment. `OUTPUT_ROOT`,
`NPROC_PER_NODE` and `FINETUNE_NPROC` override the output location and process
counts. Changing GPU count changes global batch size and is not the exact
reference protocol. Only trusted local checkpoints/caches should be loaded.

Individual stages are also available:

```bash
python -m eyemae.eyevq.tokenizer.train --config configs/eyevq/final/tokenizer.yaml --output_dir outputs/eyevq/final/tokenizer
python -m eyemae.eyevq.precompute_codes --help
python -m eyemae.eyevq.pretrain.train --config configs/eyevq/final/bert.yaml --output_dir outputs/eyevq/final/bert
python -m eyemae.eyevq.downstream.train_mil --config configs/eyevq/final/mci.yaml --output_dir outputs/eyevq/final/downstream/mci/seed42
python -m eyemae.eyevq.downstream.evaluate_mil --help
```

Binary task configs are provided for MCI, AD, Detox, epilepsy and migraine;
PD3 uses the Scheme-B label mapping documented in its config. Run the other
binary tasks through `train_mil` with their corresponding config.

## Evaluation and reproducibility

Subject-level train/validation/test separation is audited before fine-tuning.
Checkpoint selection uses validation metrics; periodic test evaluation is off
in the published configs. The pipeline reports each seed separately, without
automatic cross-seed ensembling. Historical test sets were observed during
development: this is an internal exploratory protocol, not an untouched external
test. Unlabeled target-subject pretraining overlap and transductive per-subject
normalization are explicitly declared.

Empirical quality thresholds are advisory; missing/non-finite diagnostics and
artifact identity mismatches remain fatal. See `docs/core_audit.md` for cleanup
scope and verification limits.
