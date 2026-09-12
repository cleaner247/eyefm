# Core release audit — 2026-09-12

## Scope

Retained EyeVQ tokenizer, code-cache creation, masked BERT pretraining,
subject-level MIL training/evaluation, data preprocessing, metrics, artifact
validation and their regression tests. Removed external comparison methods,
legacy EyeMAE training, grid/search runners, historical data builders, obsolete
experiment configs and report-generation scripts from the current source tree.
Internal model compatibility interfaces remain covered by tests; no alternative
model-size or comparison-method training recipe is published.

Only the 12-layer/384-dimension reference is in `configs/eyevq/final/`, with
six downstream task configs (MCI, PD3, AD, Detox, epilepsy, migraine). The small
preprocessing YAML under `tests/fixtures/` is a synthetic unit-test fixture,
not a training recipe.

Training outputs, subject split manifests and area-statistics payloads are no
longer Git-tracked. Existing local datasets/results were not deleted. Removed
source/config/script files were backed up outside the repository before changes.
This is a normal cleanup commit, not a rewrite of existing Git history.

## Corrections

- Replaced stale 50K/single-span BERT config with the completed reference's
  20K/dual-scale 15+5 configuration and 64 unique trials per GPU.
- Restored full-encoder/embedding fine-tuning, 10 binary-task epochs and
  15 PD3 epochs with hierarchical sigmoid heads.
- Separated pretraining and fine-tuning process counts (defaults 4 and 1);
  included each stage's world size in its run identity.
- Extracted distributed setup/LR helpers from the retired trial-level trainer.
- Removed automatic ensemble reporting that assumed a three-logit PD3 softmax.
- Final checkpoint reuse now checks the configured terminal step, not just
  the config/data identity. Downstream identity includes the test index.
- Fixed absolute output paths in detached launch and removed the hardcoded
  Python-environment default. Removed unused plotting/TensorBoard dependencies.
- Empirical quality thresholds are advisory; missing/non-finite metrics,
  changed source/configuration and artifact identity mismatches still fail.

## Shared-artifact manifest migration

The old external manifest recorded area-statistics SHA256
`d02e2672ba9a39f58790272dea87ffa20ead757034fa23952dd3d57ce272eff1`,
which does not match the current file. The completed reference BERT 20K
checkpoint records
`3024309973260e7cace2d2105840ea667261e60d7d91fecfcd2e3591bd01d160`,
which exactly matches the current 1,602,540-byte file. The published artifact
manifest now pins this verified BERT-reference version. Manual-feature cache,
statistics and task weights retain their original, verified hashes. Dataset
manifest and build-audit hashes were also checked. External payloads were not
rewritten to make the check pass.

The older tokenizer checkpoint records a different area-statistics file hash;
the cleaned recipe must not be described as a bitwise reproduction of that old
tokenizer run. A new full run uses the newly pinned inputs throughout, and old
checkpoint identities are deliberately rejected for automatic reuse.

## Verification

- CPU suite: 163 passed; the full-model CUDA test was skipped in the base
  environment, whose CUDA 13 build is incompatible with this machine's driver.
- In the existing CUDA-compatible training environment (PyTorch 2.5.1/cu121),
  the full 12-layer tokenizer and dual-scale BERT passed bf16 forward/backward
  checks at 128 patches, with finite losses and gradients.
- Full 12-layer K16 binary and hierarchical-PD3 MIL models passed CUDA bf16
  forward/backward checks with three synthetic subjects and finite gradients.
- All 40 core Python modules imported; Python compilation, shell syntax,
  whitespace checks and the real-data pipeline preflight passed.
- A wheel was built and inspected: no retired baseline/training/search modules.

Use an environment compatible with your GPU driver and set `PYTHON_ENV` when
invoking the launcher. Editable installation from the repository is the supported
one-command workflow; the pipeline needs the repository's configs and source
identity files.

These checks do not prove the absence of every bug. No new 40K+20K training run,
full downstream replication, multi-GPU end-to-end training, or fresh external
clinical validation was performed for this cleanup. Existing model results are
not new validation evidence for the edited code. Historical test observations
and unlabeled/transductive data overlap are disclosed in the README.
