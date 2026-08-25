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

## Stable reference and active candidate

- Stable reference: V4, factorized BERT, paired uniform spans 1--5, mask ratio
  0.60, MCI/PD5 K16.
- Active local candidate on 2026-08-25: V6, paired symmetric spans 1--6, mask
  ratio 0.50. V6 applies no new filter while repacking V5, but therefore inherits
  the signal filtering already present in V5.

The active candidate does not replace the stable reference until downstream
validation is complete. Test metrics never make that promotion decision.

## Terms

- **K16**: for each subject and task, sample 16 distinct training trials without
  replacement. It is not a batch-size multiplier and does not create 16 subject
  losses.
- **Paired mask**: left and right eye tokens at one time patch are masked together.
- **Valid-only mask ratio**: the ratio is computed over time patches where both
  eyes are nonpadding and meet the nonmissing threshold.
- **Symmetric span distribution**: middle span lengths are more likely than the
  shortest and longest lengths. It is an experimental V6 choice, not the stable
  uniform-span reference.
