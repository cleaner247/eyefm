# EyeVQ: Stimulus-Conditioned Discrete Pretraining for Subject-Level Eye-Movement Phenotyping

Anonymous ICLR submission — working manuscript, 25 August 2026

## Abstract

Eye movements are long, irregular, binocular time series whose clinically
relevant signal is weak relative to subject identity, stimulus dynamics and
missingness. We introduce **EyeVQ**, a three-stage representation-learning
framework that separates local measurement modeling, contextual prediction and
subject-level inference. First, a stimulus-conditioned tokenizer compresses
left/right eye patches with finite scalar quantization (FSQ) while reconstructing
the eye signal and predicting trial descriptors through the discrete bottleneck.
Second, an encoder predicts masked FSQ digits from visible stimulus and binocular
context. Its asymmetric attention graph prevents stimulus tokens from reading eye
tokens, paired masking prevents inter-eye copying, and per-trial loss reduction
prevents long trials from dominating. Third, a low-capacity multiple-instance
classifier averages evidence across trials and four oculomotor tasks. On the
current internal subject-disjoint splits, the validation-selected unified model
obtains three-seed mean MCI validation/test AUROC of 0.906/0.875 and PD5
macro-AUROC of 0.863/0.881. A task-favoring span geometry raises MCI to
0.914/0.891 but reduces PD5 validation AUROC, exposing a genuine transfer
trade-off. The latest tokenizer uses 1,571 of 1,575 codes with validation
perplexity 788, showing that the bottleneck remains expressive without code
collapse. Extensive
audits isolate the contributions of correct binocular ordering, validity-aware
masking, factorized prediction and subject-level aggregation. Results are
internal exploratory estimates because the test split was observed during method
development; an untouched external cohort remains required.

## 1. Introduction

Clinical eye-movement datasets differ from conventional sequence benchmarks in
four ways. A subject contributes many correlated trials; trials are collected
under different tasks; a visual stimulus explains a large part of the motion;
and missingness is structured by eye, time and recording quality. Treating every
trial as an independent labeled example leaks subject frequency into the loss.
Treating left and right eyes as interchangeable channels enables trivial
cross-eye reconstruction. Finally, reconstructing raw trajectories alone spends
capacity on smooth local variation rather than transferable behavior.

EyeVQ resolves these issues with a deliberately small set of structural choices:

1. **A discrete binocular bottleneck.** A joint Transformer encodes stimulus and
   eye patches, but only time-major left/right states are quantized using
   FSQ `[9,7,5,5]`.
2. **Directionally isolated stimulus conditioning.** Stimulus queries read only
   stimulus keys--never CLS or eye keys--while eye and CLS queries may read
   stimulus and eye context.
3. **Validity-aware paired span prediction.** BERT masks both eyes at a selected
   time patch and predicts four FSQ digits with four categorical heads.
4. **Subject-level multiple-instance inference.** Training samples trials within
   subjects, computes one subject loss, and partitions subjects—not trials—over
   distributed ranks.

The design is intentionally conservative. We exclude dynamic masking, learned
task weights, cross-task residual networks, Cartesian trial products and large
demographic projections because they increased degrees of freedom without a
stable validation gain.

## 2. Data and preprocessing

### 2.1 Trial representation

Each trial contains a four-channel stimulus stream
`[fixation_on, stimulus_on, x, y]` and two four-channel eye streams
`[x, y, area, blink]`. Missing samples are represented by a separate quality
mask rather than a numeric sentinel. A trial is removed only when both eyes are
unusable. The remaining signal is divided into non-overlapping 40-sample patches
and padded to at most 128 patches.

### 2.2 Normalization

Horizontal and vertical coordinates use fixed geometric normalization. Pupil
area is normalized separately for each subject and eye:

$$
\tilde a_{s,e,t}=\operatorname{clip}\left(
\frac{a_{s,e,t}-\operatorname{median}(a_{s,e})}
{1.4826\max(\operatorname{MAD}(a_{s,e}),32)},-5,5\right).
$$

Statistics use unlabeled samples and never use disease labels. Left and right
eyes are normalized separately because their acquisition scale and valid-frame
distribution differ. `log1p` is not applied: robust centering, MAD scaling and
clipping already constrain outliers without compressing low-area differences.

### 2.3 Split and validity policy

All downstream splits are subject-disjoint and are audited for both subject and
trial identity overlap. Tokenizer, BERT and downstream stages discard trials in
which both eyes are invalid. BERT masking additionally requires both eyes at a
time patch to have a nonmissing fraction of at least 0.85. Padding, invalid eyes,
stimulus and CLS tokens are never prediction targets.

## 3. Method

### 3.1 Joint tokenizer

Separate convolutional patch embedders map stimulus and eye patches to 384
dimensions; left and right eyes share the eye embedder. Type, time-position and
quality embeddings are added before forming

$$
[\mathrm{CLS},S_0,L_0,R_0,\ldots,S_{N-1},L_{N-1},R_{N-1}].
$$

The encoder contains 12 pre-norm Transformer layers, eight heads and a 1,152-unit
SwiGLU feed-forward block. The structural attention matrix is asymmetric:

- a stimulus query attends only valid stimulus keys;
- an eye or CLS query attends every valid stimulus/eye key.

Consequently the stimulus representation cannot contain an eye-derived shortcut,
but the eye representation remains conditioned on the known experimental input.
This is one masked self-attention graph, not a cross-attention module.

Eye states are stacked in the invariant time-major order
`[L0,R0,L1,R1,...]`, projected to four scalars and quantized by standard tanh
FSQ with levels `[9,7,5,5]`. The Cartesian code space contains
`9×7×5×5=1,575` joint codes. A three-layer Transformer decoder receives only
quantized eye states. Its patch outputs reconstruct both eyes and its decoder CLS
predicts 38 trial descriptors, ensuring that the auxiliary semantic path cannot
bypass quantization.

### 3.2 Tokenizer objective

For valid samples, coordinate, area and blink losses are

$$
\mathcal L_{eye}=\mathcal L_{xy}
+0.1\mathcal L_{area}+0.1\mathcal L_{blink}.
$$

Coordinates, area and continuous trial descriptors use Smooth L1; binary trial
descriptors and blink use binary cross entropy. Count descriptors are excluded
because they largely encode trial duration and missingness. Explicit velocity
loss is zero-weighted: velocity is a deterministic difference of coordinates and
previous experiments found that emphasizing it duplicated the coordinate
objective and destabilized early optimization.

Standard FSQ has no learned embedding table and therefore no commitment term.
The complete loss is

$$
\mathcal L_{tok}=\mathcal L_{eye}
+0.0015\left(0.25\mathcal L_{feat}^{bin}
+1.5\mathcal L_{feat}^{cont}\right).
$$

Small feature weight prevents an easily memorized trial head from overwhelming
patch reconstruction while preserving a semantic pressure on the codes.

### 3.3 Offline label materialization

After tokenizer training, code IDs are computed once and stored with trial IDs,
the tokenizer SHA256, target-generation contract, FSQ levels and the declared
`time_major_lr_v1` layout. A sidecar manifest additionally binds the complete
cache SHA256 and byte size. BERT never calls the tokenizer. A direct BERT launch
rejects a mismatch; the canonical pipeline quarantines the stale artifact and
recomputes it, preventing silent use of stale or left/right-misaligned labels.

### 3.4 Masked discrete pretraining

BERT reuses the patch embedding structure and has 12 layers, width 384, eight
heads and FFN width 1,152. At each selected time index, both left and right eye
patches are replaced by mask content while retaining position, eye type and
quality embeddings. Visible stimulus tokens remain available. Paired masking
prevents one eye from directly revealing the synchronized target of the other.

Instead of a single 1,575-way classifier, four heads predict FSQ digits of sizes
9, 7, 5 and 5. For trial `i`, masked valid positions `M_i`, and digit `d`,

$$
\mathcal L_i=\frac{1}{|M_i|}\sum_{t\in M_i}\sum_{d=1}^4
\operatorname{CE}(h_d(z_{i,t}),q^d_{i,t}),\qquad
\mathcal L_{BERT}=\frac{1}{|I|}\sum_{i\in I}\mathcal L_i.
$$

The inner mean gives every supervised trial equal weight regardless of length.
Distributed training all-reduces numerators and denominators so that this is the
true global-trial mean. Our current validated reference masks 60% of eligible
time patches with uniformly sampled span lengths 1–5. A controlled ablation over
random masks, span 1–6/1–8, and length-event versus token-balanced distributions
is in progress; only validation metrics will select the final geometry.

### 3.5 Subject-level downstream inference

For each subject and each of four tasks (pro-, anti-, memory- and double-saccade),
the final MCI protocol samples 16 distinct trials without replacement. We denote
this protocol **K16**, where K is simply the number of trials sampled from each
task for one subject. The baseline K4 therefore uses four trials per task. The
historical implementation split K16 into four disjoint groups of four trials
(`4xK4`), but because the prediction head is shared, group consistency has zero
weight, and all logits are averaged, it is mathematically identical to directly
averaging the 16 trial logits. Every trial produces a BERT CLS
representation `c`. We concatenate `LayerNorm(c)` with a 16-dimensional,
training-fold-fitted demographic vector containing z-scored age, an age-missing
indicator, sex one-hot features and education one-hot features. A shared
`400→128→C` MLP with GELU and dropout 0.3 produces trial logits.

Logits are averaged first over the 16 trials within each task and then uniformly
over the four tasks. Each subject contributes one supervised loss, so K16 reduces
sampling variance without multiplying subject weight.
Validation and test use all valid trials. This estimator is permutation invariant,
contains no learned task weights, and has far fewer subject-specific parameters
than concatenating every trial representation. Subjects with fewer than four
valid trials in any task are excluded in the strict reference protocol.

The embedding and bottom four BERT layers are frozen; the top eight layers and
head use learning rate `1e-5`. MCI uses subject-level positive weighting and a
fixed threshold of 0.5. PD5 uses subject inverse-frequency class weights and
argmax prediction. Raw validation AUROC selects checkpoints; the test split is
evaluated only after restoring the validation-best checkpoint.

## 4. Optimization and reliability

Tokenizer training uses four GPUs, 128 trials/GPU, bf16, AdamW
`(β1,β2)=(0.9,0.95)`, 2,000 warmup steps and cosine decay from `3e-4` to
`3e-5` for 40K steps. BERT uses the same per-GPU batch and LR schedule for the
validation-selected 50K reference; 80K runs measure the effect of additional
optimization. Matrix
weights receive decay (0.01 tokenizer, 0.05 BERT/downstream); bias, normalization
and all one-dimensional parameters receive no decay.

Every checkpoint embeds the resolved configuration hash and SHA256 identities
for its dataset manifest, split indices, normalization statistics and upstream
artifacts. Resume selects the largest numeric step, not lexicographic filename
order, and is rejected unless the complete identity matches. Deliberate schedule
extension requires an explicit identity-mismatch override and is recorded as a
new experiment. DDP tests ensure that forward
passes use the wrapped module. Sampler audits verify rank-disjoint subjects and
no within-epoch repetition. Early-stopping patience resets after every new best,
and both natural completion and early stopping restore the same best checkpoint
before test evaluation.

### 4.1 Final training recipe and parameter selection

Table 1 gives the single reproducible recipe used by the canonical pipeline.
Values are selected by validation evidence across both diseases; test metrics
are reported but never used to rank a candidate.

| Component | Selected setting | Reason for selection |
|---|---|---|
| Data | V4 packed internal reference; inherited trial identities and splits; per-subject/per-eye median-MAD | V4 has the latest completed comparable validation evidence for both tasks. Current V5/V6 payloads are identical and inherit an additional guarded 75-Hz filtering stage, so they remain a provenance-limited experiment rather than the paper default. |
| Patch | 40 samples, stride 40, maximum 128 | Preserves short ocular events without the sequence/memory cost of overlapping patches. |
| Quantizer | tanh FSQ `[9,7,5,5]` | 1,575 states maintain high usage; `[9,7,7,5,5]` improved one MCI validation run but reduced PD5 validation/balanced accuracy and makes MLM substantially sparser. |
| Tokenizer optimizer | 40K, 128/GPU, `3e-4→3e-5`, 2K warmup | At 3K, `3e-4` improved both eye and feature validation losses over `2e-4`; `5e-4` gave no gain and worse code usage. Forty thousand steps balances reconstruction improvement against late auxiliary-feature overfit. |
| BERT target | four FSQ-digit CE heads | Provides a useful gradient when one scalar digit is wrong and avoids a sparse 1,575-way exact-only target. |
| BERT mask | paired span, ratio 0.60, uniform length 1–5 | Prevents cross-eye copying, mixes micro- and mesoscale occlusion, and is the strongest completed validation-selected factorized run across MCI/PD5. |
| BERT optimizer | 50K, 128/GPU, `3e-4→3e-5`, 2K warmup | Fifty thousand steps is the strongest completed validation reference; 80K lowers MLM CE but has not produced a consistent disease-validation gain. |
| Adaptation | freeze embedding/bottom four; train top eight | Top six under-adapts; top ten gave only a marginal validation increase with worse stability; full unfreezing was unstable. |
| Subject head | shared `400→128→C`, dropout 0.3, K16 for final MCI | Hidden 128 and dropout 0.3 led PD5 validation and balanced accuracy; K16 improved MCI over K4 without changing subject loss weight. |

The physical pretraining batch is deliberately 128/GPU (global 512). Historical
256/GPU runs processed approximately 2,660 rather than 1,840–2,100 trials/s,
but each step was slower (about 0.38 versus 0.21–0.27 s) and an unchanged step
budget doubled sample exposure. At equal exposure, 256/GPU would use 40K rather
than 80K updates and save only about 20–25% wall time while halving optimizer
updates. Since no controlled equal-exposure experiment shows better downstream
validation, global 512 is retained for statistical efficiency and comparability;
larger batches are a throughput ablation, not the default model.

## 5. Results

### 5.1 Tokenizer

The latest 40K tokenizer, whose stimulus tokens also do not read CLS, achieves
validation `L_eye=0.00154` and `L_feat=0.15677`. Of 1,575 possible codes, 1,571
are active; code perplexity is 787.6 and top-1 frequency is 1.37%. All four
scalar dimensions are active. These
statistics reject both complete and marginal code collapse while leaving enough
compression to define a nontrivial contextual prediction target.

### 5.2 Downstream results

All entries report the mean and population standard deviation over seeds
42/43/44. Test results are shown for completeness but were historically observed
and must not be interpreted as untouched external evaluation.

| Mask / predictor | Steps | MCI Val AUROC | MCI Test AUROC | PD5 Val macro-AUROC | PD5 Test macro-AUROC |
|---|---:|---:|---:|---:|---:|
| span 1–5, uniform, factorized | 50K | .906±.004 | .875±.008 | **.863±.006** | **.881±.003** |
| span 1–6, symmetric, joint-code predictor | 50K | **.914±.007** | **.891±.004** | .859±.009 | .870±.003 |
| span 2–6, ratio .50, joint-code predictor | 40K, B256 | .892±.003 | .888±.007 | .862±.002 | .868±.005 |

The 1–5 factorized model is the validation-selected unified reference because it
has the best standardized joint validation score across MCI and PD5. The 1–6
symmetric model is preferable only if MCI is the sole target. The 1–8 model
improves masked-token validation accuracy throughout 80K but does not uniformly
improve disease validation, demonstrating that lower pretraining CE is not a
sufficient model-selection criterion. This supports validation-only downstream
screening of mask strategies.

For the 1–8 model, MCI test balanced accuracy is `.821±.019`. PD5 test balanced
accuracy is `.434±.007` despite macro-AUROC `.866`, revealing a remaining
calibration/subtype-confusion bottleneck rather than absence of ranking signal.

## 6. Why the architecture is effective

**The quantizer removes an unnecessarily hard output problem.** Predicting four
small categorical variables preserves FSQ geometry and supplies gradients even
when only one digit is wrong; joint-code exact accuracy alone treats nearby and
distant errors identically.

**The attention graph represents the experimental causal direction.** Stimulus
drives eye motion, but recorded eyes cannot alter the stimulus. Preventing
stimulus queries from reading both eyes and CLS blocks the direct shortcut and
the two-hop eye-to-CLS-to-stimulus shortcut while letting eye tokens explain
motion using the known input.

**Paired masking makes binocular context honest.** Independent masking permits a
nearly synchronized contralateral eye to reveal the target. Paired time masks
force inference from temporal and stimulus context instead.

**Loss reduction matches the statistical unit.** The downstream label belongs to
a subject, not a patch or trial. Equalizing first over masked tokens within a
trial and finally over subjects avoids weighting long recordings or prolific
subjects more heavily.

**Low-capacity aggregation is appropriate for small cohorts.** Shared heads and
uniform means impose task/trial exchangeability where supported by the protocol.
Historical high-capacity fusion variants produced less stable validation gains,
whereas the simple estimator reaches high three-seed MCI AUROC with small
variance.

## 7. Ablations and remaining experiments

The completed MCI aggregation ablation favors K16 over K4: three-seed mean
validation AUROC increases from 0.8852 to 0.8923 and test AUROC from 0.8784 to
0.8879, while test standard deviation decreases from 0.0136 to 0.0075.
Consistency weights 0.02 and 0.05 leave seed-42 validation AUROC unchanged and
are therefore rejected. A new end-to-end run removes stimulus-to-CLS attention
in the tokenizer and uses factorized CE with 60% symmetric spans in [1,6]. Its
tokenizer is complete and non-collapsed. BERT reaches 50K with validation
loss/accuracy/perplexity 2.4158/0.3537/11.20, but its downstream evaluation is
not yet complete, so it is not used to revise the selected model.

The final paper should retain only axis-isolated comparisons:

1. joint-code versus factorized FSQ prediction;
2. random versus paired span masking;
3. span upper bound 5/6/8;
4. uniform span-event versus equal masked-token contribution;
5. BERT checkpoints 20K/40K/60K/80K with the same downstream probe;
6. frozen, top-6 and top-8 downstream adaptation;
7. eye-only, demographics-only and combined inference.

Candidate masks should first train 10–20K and use cached-CLS frozen probes. Only
the two strongest validation candidates continue to 40K, and only the winner to
80K. This successive-halving protocol reduces compute and avoids repeatedly
consulting the test set.

## 8. Limitations

The current evaluation is from one internal collection. Test labels were observed
during extensive development, so reported test values are exploratory. Area
normalization uses unlabeled per-subject statistics, which is valid for offline
phenotyping but requires sufficient samples from a new subject. Strict four-task
filtering excludes incomplete participants and may induce selection bias. PD5
contains only 34 dyskinesia training subjects and five validation subjects;
macro-AUROC is consequently more reliable than per-class F1 but still has wide
uncertainty. The present model learns association, not a causal disease marker.

## 9. Reproducibility statement

The canonical configurations are in `configs/eyevq/final/`, and
`scripts/run_eyevq_final.sh` executes tokenizer training, SHA-bound cache
materialization, BERT pretraining and three-seed MCI/PD5 evaluation. The package
contains invariant tests for left/right ordering, attention direction, invalid
mask exclusion, cache identity, AdamW grouping, DDP routing, subject split
overlap, sampler uniqueness, gradient reachability and best-checkpoint testing.
The complete run is launched with

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
  OUTPUT_ROOT=outputs/eyevq/final \
  scripts/run_eyevq_final.sh --detached
```

`pipeline_state.json` is atomically updated after every stage. A failed command
stops the pipeline. Identity-mismatched artifacts are renamed with a `.stale_*`
suffix rather than overwritten, and `final_summary.json` contains all three
seeds without using test metrics for selection. The pipeline also writes a
three-seed mean-logit ensemble for validation and test; it is a prespecified
derived result and never changes which checkpoints are selected.

## References

- Bao, H. et al. *BEiT: BERT Pre-Training of Image Transformers*. ICLR, 2022.
- Devlin, J. et al. *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*. NAACL, 2019.
- Mentzer, F. et al. *Finite Scalar Quantization: VQ-VAE Made Simple*. ICLR, 2024.
- Assran, M. et al. *Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture*. CVPR, 2023.
- He, K. et al. *Masked Autoencoders Are Scalable Vision Learners*. CVPR, 2022.
- van den Oord, A. et al. *Neural Discrete Representation Learning*. NeurIPS, 2017.
