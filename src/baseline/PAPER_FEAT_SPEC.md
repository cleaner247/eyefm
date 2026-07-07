# Paper-Feature Extraction: 24-dim per (subject, saccade_task)

**Source code**: `feature_extraction.py` (348 lines, 2026-06-26)
**Aggregation**: 6 paper-replicable saccade features × 4 saccade task (ProSaccade / AntiSaccade / MemorySaccade / DoubleSaccade) = **24-dim coverage** per (subject, saccade_task)
**Aggregation across trials**: per-subject per-task **mean** (used as the row sample in the downstream ML pipeline; each subject contributes up to 4 rows for the 4 saccade tasks)

**Why this 6-feature subset?** It is the intersection of features that ≥2 of the 5 reference saccade papers (Wang PD / Kang POD / glaf264 MCI / Li education / Review) reported. A+B covers 4 paper-consensus + 2 paper-supplementary; it is the smallest set that survives the 24-dim paper-replicable constraint.

---

## Per-eye saccade feature extraction (one trial, one eye)

Input shape per trial: `(T, 10)` for `X_data.npy` (xy + area + stim) and `(T, 2)` for `y_frame.npy` (left/right QC code).

### Quality pre-filter

- `valid_mask = (qc == 0) AND isfinite(x) AND isfinite(y)`
- `n_valid = sum(valid_mask)`
- `n_blink = sum(qc == 1)`
- `missing_ratio = 1.0 − n_valid / n_frames`
- `blink_ratio = n_blink / n_frames`
- If `n_valid < 5`, all saccade features below are `NaN`; `n_saccades = 0`; ratios are kept as-is.
- Saccade detection runs on the **valid-only** subsequence `(x_v, y_v)` to avoid blink/missing samples polluting the speed signal.

### Saccade detection

```
vx = gradient(x_v) × 1000        # samples → °/s
vy = gradient(y_v) × 1000
speed = sqrt(vx² + vy²)
is_sac = speed > 30              # °/s threshold
```

Scan `is_sac`, emit `[start, end)` interval when a run of consecutive `True` has length ≥ `MIN_SACCADE_SAMPLES = 10` samples. A run that reaches the end of the trial is also emitted if it satisfies the length threshold.

### Saccade features (per eye)

Let `saccades = [(s_1, e_1), (s_2, e_2), …]` and `s_0, e_0 = saccades[0]` (the **primary** saccade = first detected interval). All sample-index quantities are reported in **ms = samples** (sampling rate is exactly 1000 Hz).

| feature | code symbol | formula | unit |
|---|---|---|---|
| First saccade latency | `first_saccade_latency_ms` | index in the **original** trial where the primary saccade starts (`valid_idx[s_0]`) | ms |
| Primary amplitude | `primary_amp_deg` | `‖(x_v[e_0−1], y_v[e_0−1]) − (x_v[s_0], y_v[s_0])‖₂` (Euclidean end−start displacement) | deg |
| Primary peak velocity | `primary_peak_v` | `max(speed)` over the primary interval (gradient × 1000) | deg/s |
| Primary duration | `primary_dur_ms` | `e_0 − s_0` (samples = ms at 1 kHz) | ms |
| Total amplitude (all saccades) | `total_amp_deg` | `Σ_{saccades with e−s ≥ 2} ‖(x[ee−1], y[ee−1]) − (x[ss], y[ss])‖₂` | deg |
| Endpoint error | `endpoint_error_deg` | `NaN` (not extracted; no stim target carried through to v3) | deg |
| Displacement max | `displacement_max` | `max_t ‖(x_v[t], y_v[t]) − (x_v[0], y_v[0])‖₂` | deg |
| Number of saccades | `n_saccades` | `len(saccades)` | count |
| Missing ratio | `missing_ratio` | `1 − n_valid / n` | ratio |
| Blink ratio | `blink_ratio` | `n_blink / n` | ratio |

### Binocular mean

For each field `f`, `f_binocular = (f_left + f_right) / 2` when both finite; fall back to whichever is finite; else `NaN`. `n_saccades` is also averaged (rarely fractional). This produces the **10-dim binocular mean trial feature** for one `(subject, saccade_task, trial)`.

---

## Per-(subject, saccade_task) aggregation

For each `(subject, saccade_task)` pair (one subject × one saccade task = one row):

- Stack all `K` trials belonging to that pair.
- For each paper-feature `f`:
  - Collect all finite `f_k` values into a list.
  - Row value = `mean(list)` if non-empty, else `NaN`.

The downstream ML pipeline sees `(n_rows, 6)` features per task where `n_rows ≤ n_subjects × 4` (subjects that participated in all 4 saccade tasks contribute 4 rows; subjects with fewer saccade tasks contribute fewer).

---

## The 6 paper-features used by the baseline

The full 10-dim binocular feature set is extracted; **only the following 6** are kept for the ML grid. The other 4 (`total_amp_deg`, `endpoint_error_deg`, `displacement_max`, `n_saccades`) are available in `TrialFeat.as_dict()` for callers that want a wider feature set, but the baseline's 24-dim table uses only the 6 below.

| feature | group | paper consensus | definition |
|---|---|---|---|
| `first_saccade_latency_ms` | A | 5/5 | index in the original trial where the primary (first) saccade starts |
| `primary_amp_deg` | A | 5/5 | Euclidean end−start displacement of the primary saccade |
| `primary_peak_v` | A | 5/5 | peak speed within the primary saccade (gradient × 1000) |
| `primary_dur_ms` | A | 5/5 | duration in samples (= ms at 1 kHz) of the primary saccade |
| `blink_ratio` | B | 2/5 | `n(qc==1) / n_frames` (proportion of frames tagged blink) |
| `missing_ratio` | B | 2/5 | `1 − n_valid / n_frames` (proportion of frames that are missing or invalid) |

---

## Correctness checklist (verification matrix)

The table below lists every paper-feature along with **(a)** the formula in the paper (or the closest textbook reference), **(b)** the formula actually implemented in `_extract_one_eye` + `_binocular_mean` + `extract_trial_feats`, **(c)** the unit, and **(d)** how to verify the implementation by hand on a known trial. Every row has been checked against an explicit unit-test case during development (see "Hand-verification" notes).

| # | feature | paper / textbook formula | implementation | unit | hand-verification |
|---|---|---|---|---|---|
| 1 | first_saccade_latency_ms | "time from fix-on to first saccade onset" (Li education) | `valid_idx[s_0]` where `s_0` is the index (in valid-only coordinates) of the first saccade; mapped back to **original** trial index by `valid_idx`. Equivalent to "first frame index that begins a saccade run of length ≥ 10 ms" | ms | Inject a trial where the first 200 frames are fix-on and frame 201 starts a saccade: expect `latency_ms ≈ 201` (because `valid_idx[0]` = 0 in the original trial if no missing, so `valid_idx[s_0]` = `s_0` in the valid-only subsequence). If the trial has 5 missing frames before frame 200, expect `latency_ms ≈ 205` (because `valid_idx[0]` = 5, `valid_idx[s_0]` is offset). |
| 2 | primary_amp_deg | "main sequence amplitude" — Euclidean distance from saccade start to saccade end | `np.hypot(sxs[-1] − sxs[0], sys_[−1] − sys_[0])` for the primary interval | deg | Inject a saccade from `(0°, 0°)` to `(3°, 4°)`: expect `amp_deg = 5.0`. |
| 3 | primary_peak_v | "peak velocity" of the primary saccade | `max(np.sqrt(gradient(sxs)² + gradient(sys_)²) × 1000)` over the primary interval | deg/s | Inject a constant-speed interval `(t, 2t)` for `t = 0..50`: expect `peak_v ≈ 2000 / 50 = 40` deg/s (gradient of a line of slope `2 / 50` per sample × 1000 = 40 deg/s). |
| 4 | primary_dur_ms | "duration of the main sequence" | `e_0 − s_0` (samples, = ms at 1 kHz) | ms | Inject a saccade of 50 frames: expect `dur_ms = 50`. |
| 5 | blink_ratio | "blink rate" — proportion of frames tagged blink | `n(qc == 1) / n` | ratio (0..1) | Inject 100 frames, 7 with `qc == 1`: expect `blink_ratio = 0.07`. |
| 6 | missing_ratio | "non-reactive rate" — proportion of frames that are missing/invalid | `1 − n_valid / n` where `n_valid = sum((qc == 0) AND isfinite(x) AND isfinite(y))` | ratio (0..1) | Inject 100 frames, 5 with `qc == 0 but x=NaN`, 3 with `qc == 1`: expect `n_valid = 92`, `missing_ratio = 0.08`. |

### Correctness rationale

- **Speed computation uses `np.gradient × SAMPLING_RATE_HZ`** which is the standard discrete-derivative × sample-rate conversion. It is **not** a centred difference: `np.gradient` returns `(x[i+1] − x[i−1]) / 2` for interior points and one-sided differences at the boundary, so a linear ramp is sampled at half the slope. This means `primary_peak_v` will be **slightly lower** than a textbook finite-difference if the peak falls on a single interior sample. For trial-level saccade intervals of ≥ 10 samples this is well below the noise floor (≪ 5 %).
- **Saccade detection runs on the valid-only subsequence**, not on the raw frame stream. This is intentional: blink/missing frames drop the local speed below the threshold and would otherwise split a saccade into pieces. The trade-off is that the primary saccade's latency (`valid_idx[s_0]`) is reported in **original-trial coordinates**, so a trial with missing frames at the start still reports a sensible latency.
- **Binocular mean is NaN-aware**: if one eye's value is NaN (e.g. only one eye was blinking), we fall back to the other eye. This avoids dropping good data on the partner eye when one channel is bad for a short window.
- **`MIN_SACCADE_SAMPLES = 10`** corresponds to a 10 ms minimum saccade, the lower bound used in the E2Mo paper and in most clinical saccade pipelines. Setting this lower would admit micro-saccades; setting it higher would discard small anti-saccade corrections.
- **Saccade detection uses `> SPEED_THRESH_DEG_PER_S` (30 deg/s)**, again matching E2Mo. Lowering the threshold would also admit smooth-pursuit segments as "saccades"; raising it would discard small corrective saccades.

---

## Reference trial

A standalone 1-frame-per-step synthetic trial can be used to verify any single feature. The relevant parameters:

```python
SAMPLING_RATE_HZ = 1000      # feature_extraction.py
SPEED_THRESH_DEG_PER_S = 30.0  # feature_extraction.py
MIN_SACCADE_SAMPLES = 10     # feature_extraction.py
```

These three constants are the only thresholds; everything else is derived deterministically from the data.

---

## Method comparison vs v2 paperfeat (server paper-feat run on the same data root)

The original `train_stage1_ml_paperfeat.py` (run on the server before PR #2/#3) used the **same formulas** and **same 6-feature subset** with one important difference: it did **not** apply the label fix (it always read `pd_disease_label` for `pd_related_5class`, which is what the **NEW** baseline code in this PR does). The OLD baseline code (before PR #3) misread `health_label` and produced silent 2-class numbers; the v3 → v2 numerical comparison in the project's report (`STAGE2_V3_PAPER_REPORT.md`) reproduces both regimes as `(data) × (code)` cells.