# EyeVQ code map

This document defines which files are authoritative. It prevents a historical
ablation or a currently running candidate from being mistaken for the supported
training recipe.

## Supported surface

- `src/eyemae/eyevq/`: model, data, optimization, training and evaluation code.
- `configs/eyevq/final/`: the only supported default configurations.
- `configs/eyevq/final/recipe.yaml`: dataset identity, architecture contract and
  validation gates.
- `scripts/run_eyevq_final.sh`: the only supported one-command launcher.
- `eyevq-pipeline`: the installed equivalent of the launcher.
- `tests/test_eyevq_*.py`: executable architecture and data invariants.

The pipeline resolves every configuration into the output directory, hashes the
executable source and dependencies, and refuses to mix stages from different
identities.

## Reproducibility records

Other files under `configs/eyevq/`, the older launch scripts, and ablation reports
exist to reproduce historical comparisons. They are not defaults and must not be
used to infer the current architecture. New candidates stay local until their
validation-only comparison is complete; promotion means updating all files under
`configs/eyevq/final/`, their tests and the paper together.

## Default data, stable geometry and active candidate

- Default dataset: V6 with V6-derived area statistics and manual features. V6
  applies no new filter while repacking V5 and inherits V5's guarded 75-Hz
  signal filtering.
- Stable model geometry: factorized BERT, paired uniform spans 1--5, mask ratio
  0.60, MCI/PD5 K16.
- Active masking candidate on 2026-08-25: paired symmetric spans 1--6, mask ratio
  0.50 on the same V6 dataset.

The active masking candidate does not replace the stable geometry until
downstream validation is complete. Test metrics never make that decision.

## Terms

- **K16**: for each subject and task, sample 16 distinct training trials without
  replacement. It is not a batch-size multiplier and does not create 16 subject
  losses.
- **Paired mask**: left and right eye tokens at one time patch are masked together.
- **Valid-only mask ratio**: the ratio is computed over time patches where both
  eyes are nonpadding and meet the nonmissing threshold.
- **Symmetric span distribution**: middle span lengths are more likely than the
  shortest and longest lengths. It is an experimental masking choice, not the
  stable uniform-span geometry.
