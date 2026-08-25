# EyeVQ architecture contract

`eyemae.eyevq` is the only formal implementation. The resolved run recipe is
stored under `configs/eyevq/final/`; experimental YAML and scripts are evidence,
not defaults.

## Information flow

```text
stimulus + left eye + right eye
  -> 40-sample non-overlapping patches
  -> joint masked self-attention tokenizer
  -> FSQ[9,7,5,5] codes and quantized eye reconstruction
  -> offline tokenizer-SHA-bound code cache
  -> paired valid-only span-masked BERT
  -> per-trial CLS
  -> shared trial head -> task mean -> subject mean
```

The sequence is `[CLS,S0,L0,R0,...]`. Stimulus is exogenous: stimulus queries
read only valid stimulus keys and cannot read CLS or L/R. CLS and eye queries
may read all valid stimulus/eye keys. Setting `stim_attend_cls=true` would allow
eye information to return to stimulus through CLS after one layer and is
therefore forbidden by the formal pipeline.

The tokenizer uses a 12x384 encoder, 3x384 decoder, eight heads, FFN 1152 and
dropout zero. Its parameter-free tanh FSQ has levels `[9,7,5,5]` (1,575 joint
codes); commitment loss is identically zero and is disabled in the formal
configuration. XY/area use SmoothL1, blink uses BCE, velocity is disabled, and
the two count features are excluded from the 38-D auxiliary loss.

BERT has 12x384 layers and four categorical prediction heads (9/7/5/5). A time
patch is eligible only when both eyes are nonpadding and at least 85% nonmissing.
Both eyes are masked together. The formal span is 1--5 patches, uniform by span
count, at 60% of jointly eligible time patches. CE is summed over FSQ dimensions,
then averaged within each trial and finally over globally valid DDP trials.
Tokenizer inference is never called during BERT training.

Downstream labels belong to subjects. Embeddings and the bottom four BERT layers
are frozen; the top eight layers are adapted at `1e-5`. Each trial uses
`LayerNorm(CLS384) + fold-safe demographics16 -> MLP(400,128,C)`. Trial logits
are averaged inside each task and the four task logits are averaged uniformly.
MCI and PD5 both use K16 training bags: each task contributes 16 distinct trials
sampled without replacement. Validation/test use all available trials after the
explicit four-trial minimum.

## Reproducibility and safety

The one-command pipeline:

- hashes the complete executable Python/configuration surface;
- embeds that hash in every resolved configuration and checkpoint identity;
- rejects source changes between stages;
- selects resume checkpoints by numeric step;
- binds the code cache to tokenizer SHA256 and preprocessing contract;
- audits dataset manifests and build records against `recipe.yaml`;
- checks tokenizer/BERT validation quality before downstream training;
- keeps DDP subjects rank-disjoint and restores validation-best checkpoints
  before test evaluation.

EMA-VQ, iFSQ, lightweight reconstruction, task-weight learning, Cartesian CLS
products, partial-task mask embeddings, consistency loss, mixup, layer-wise LR
decay and span-length embeddings are not part of the formal method.
