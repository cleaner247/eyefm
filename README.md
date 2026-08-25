# EyeVQ / EyeFM

The canonical project path is `eyemae.eyevq`: discrete binocular tokenization,
masked contextual pretraining, and subject-level multiple-instance fine-tuning.
Legacy EyeMAE experiments remain reproducible, but they are not production
defaults.

## Canonical pipeline

```text
V6 packed trials
  -> valid-eye filtering + per-subject/per-eye median-MAD area normalization
  -> non-overlapping 40-sample stimulus/left/right patches
  -> 12-layer joint stimulus-isolated tokenizer
  -> tanh FSQ [9,7,5,5] and quantized reconstruction
  -> SHA-bound offline code-ID cache
  -> 12-layer paired-span factorized-code BERT
  -> strict four-task subject MIL for MCI and PD5
```

Stimulus queries read stimulus tokens only. They never read CLS or L/R; CLS and
eye queries may read stimulus and eye tokens. BERT masks L/R together only when
both patches contain at least 85% nonmissing frames.

The single source of truth is:

- `configs/eyevq/final/recipe.yaml`: data contract, evidence status and gates.
- `configs/eyevq/final/tokenizer.yaml`: tokenizer model and optimization.
- `configs/eyevq/final/bert.yaml`: contextual pretraining.
- `configs/eyevq/final/mci.yaml`: MCI K16 subject adaptation.
- `configs/eyevq/final/pd5.yaml`: PD5 K16 subject adaptation.

V6 is the operational default dataset. It repacks the V5 payload without adding
another filter, preserves the established splits and source frame labels, and
therefore inherits the guarded 75-Hz filtering already present in V5. Its area
statistics and manual features were recomputed from V6 and are SHA-pinned.

The currently running V6 job also tests a different BERT masking candidate. Its
resolved configuration and logs live under its output directory; that mask
candidate does not replace the default model geometry until validation is
complete.

## Run

Install and verify:

```bash
python -m pip install -e '.[test]'
PYTHONPATH=src pytest -q
python -m compileall -q src/eyemae
```

Read-only preflight:

```bash
scripts/run_eyevq_final.sh --preflight-only
```

Detached four-GPU run:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
  OUTPUT_ROOT=outputs/eyevq/final \
  scripts/run_eyevq_final.sh --detached
```

The pipeline fails closed when the dataset contract, source hash, upstream
checkpoint/cache identity, architecture invariants, or validation quality gate
does not match. Checkpoints and large training outputs are never stored in Git.

## Documentation

- Architecture and operational contract: `src/eyemae/eyevq/README.md`
- Authoritative code map and terminology: `docs/code_structure.md`
- Data lineage and experiment summary: `docs/eyevq_project_report_20260825.md`
- Paper draft: `docs/iclr_eyevq_paper.md`

All reported test results are internal exploratory results because the test set
was observed during model development. Automated selection and early stopping
use validation data only.
