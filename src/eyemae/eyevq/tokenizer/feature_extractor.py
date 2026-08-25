"""
Trial-level saccade feature extraction for EyeVQ tokenizer auxiliary loss.

Extracts 6 continuous metrics per trial (transformed for better distribution):
  spatial_error_deg  → log1p(°)           (multiplicative error → additive)
  reaction_time_ms   → log(ms)            (log-normal RT)
  mean_velocity_deg_s → / sqrt(amplitude) (velocity efficiency, removes main sequence)
  peak_velocity_deg_s → / sqrt(amplitude) (peak efficiency)
  saccadic_gain      → raw ratio          (natural [0,2], no transform)
  first_amplitude_deg → raw (°)           (mild skew, robust-z sufficient)

Based on velocity-based saccade detection with savgol smoothing.
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass, field

try:
    from scipy.signal import savgol_filter
    HAS_SAVGOL = True
except ImportError:
    HAS_SAVGOL = False


# ──────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────

@dataclass
class SaccadeEvent:
    onset_idx: int       # start frame index
    offset_idx: int      # end frame index
    onset_ms: float      # start time (ms)
    offset_ms: float     # end time (ms)
    amplitude_deg: float # Euclidean displacement
    peak_velocity_deg_s: float
    mean_velocity_deg_s: float
    start_xy: np.ndarray  # [2]
    end_xy: np.ndarray    # [2]
    direction_deg: float   # direction of movement

@dataclass
class TrialMetrics:
    # continuous (6)
    spatial_error_deg: float = np.nan
    reaction_time_ms: float = np.nan
    mean_velocity_deg_s: float = np.nan
    peak_velocity_deg_s: float = np.nan
    saccadic_gain: float = np.nan
    first_amplitude_deg: float = np.nan
    # validity (6)
    metric_valid: np.ndarray = field(default_factory=lambda: np.ones(6, dtype=bool))

    def to_dict(self) -> dict:
        return {
            "spatial_error_deg": self.spatial_error_deg,
            "reaction_time_ms": self.reaction_time_ms,
            "mean_velocity_deg_s": self.mean_velocity_deg_s,
            "peak_velocity_deg_s": self.peak_velocity_deg_s,
            "saccadic_gain": self.saccadic_gain,
            "first_amplitude_deg": self.first_amplitude_deg,
            "metric_valid": self.metric_valid,
        }


# ──────────────────────────────────────────────
# Gaze preprocessing
# ──────────────────────────────────────────────

def cyclopean_gaze(
    left_xy: np.ndarray,    # [T, 2]
    right_xy: np.ndarray,   # [T, 2]
    left_label: np.ndarray,  # [T] 0=valid, 1=blink, 2=missing
    right_label: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Construct cyclopean gaze: mean of both eyes when available.

    Returns:
        gaze: [T, 2] in degrees
        valid: [T] bool
    """
    left_valid = (left_label == 0)  # only non-blink, non-missing
    right_valid = (right_label == 0)

    both = left_valid & right_valid
    left_only = left_valid & ~right_valid
    right_only = ~left_valid & right_valid

    gaze = np.zeros((len(left_xy), 2), dtype=np.float32)
    gaze[both] = (left_xy[both] + right_xy[both]) / 2.0
    gaze[left_only] = left_xy[left_only]
    gaze[right_only] = right_xy[right_only]

    valid = left_valid | right_valid
    return gaze, valid


# ──────────────────────────────────────────────
# Velocity computation
# ──────────────────────────────────────────────

def _masked_gradient(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Gradient computed only within contiguous valid runs (masked).

    Mirrors ``np.gradient``: central difference in the interior of a run,
    one-sided difference at run boundaries. Invalid frames never participate
    (their output is 0), so a blink/missing gap can never produce an
    artificial slope across it.
    """
    values = np.asarray(values, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    n = len(values)
    out = np.zeros(n, dtype=np.float64)
    i = 0
    while i < n:
        if not valid[i]:
            i += 1
            continue
        j = i
        while j < n and valid[j]:
            j += 1
        run_len = j - i
        if run_len == 1:
            out[i] = 0.0
        elif run_len == 2:
            delta = values[i + 1] - values[i]
            out[i] = delta
            out[i + 1] = delta
        else:
            out[i + 1 : j - 1] = (
                values[i + 2 : j] - values[i : j - 2]
            ) / 2.0
            out[i] = values[i + 1] - values[i]
            out[j - 1] = values[j - 1] - values[j - 2]
        i = j
    return out


def compute_velocity(
    gaze: np.ndarray,       # [T, 2] degrees
    valid: np.ndarray,      # [T] bool
    sampling_rate: int = 1000,
    smooth_window_ms: int = 7,
    smooth_polyorder: int = 2,
) -> np.ndarray:
    """Compute instantaneous angular velocity (deg/s).

    Invalid frames are masked out and never linearly interpolated. Smoothing
    (savgol) and the derivative are applied only within each contiguous valid
    run, so blink/missing gaps cannot produce artificial velocity. Invalid
    positions are forced to 0 so saccade detection never searches inside them.
    """
    gaze = np.asarray(gaze, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    n = len(gaze)

    if not HAS_SAVGOL:
        # Fallback: masked central difference
        dx = _masked_gradient(gaze[:, 0], valid)
        dy = _masked_gradient(gaze[:, 1], valid)
        vel = np.sqrt(dx**2 + dy**2) * sampling_rate
        vel[~valid] = 0.0
        return vel.astype(np.float32)

    window = int(smooth_window_ms * sampling_rate / 1000)
    if window % 2 == 0:
        window += 1
    window = max(window, 3)

    # Smooth each contiguous valid run independently (never across invalid
    # frames); invalid positions stay NaN.
    smooth = np.full((n, 2), np.nan, dtype=np.float64)
    for dim in range(2):
        series = gaze[:, dim].copy()
        series[~valid] = np.nan
        i = 0
        while i < n:
            if not np.isfinite(series[i]):
                i += 1
                continue
            j = i
            while j < n and np.isfinite(series[j]):
                j += 1
            run = series[i:j]
            if len(run) >= window:
                try:
                    smooth[i:j, dim] = savgol_filter(
                        run, window, smooth_polyorder
                    )
                except Exception:
                    smooth[i:j, dim] = run
            else:
                smooth[i:j, dim] = run
            i = j

    dx = _masked_gradient(smooth[:, 0], valid)
    dy = _masked_gradient(smooth[:, 1], valid)
    vel = np.sqrt(dx**2 + dy**2) * sampling_rate
    vel[~valid] = 0.0
    return vel.astype(np.float32)


# ──────────────────────────────────────────────
# Saccade detection
# ──────────────────────────────────────────────

def detect_saccades(
    vel: np.ndarray,          # [T] deg/s
    valid: np.ndarray,        # [T] bool
    sampling_rate: int = 1000,
    min_velocity_thresh: float = 30.0,
    adaptive_mad_multiplier: float = 6.0,
    min_duration_ms: int = 8,
    max_internal_gap_ms: int = 5,
    min_amplitude_deg: float = 1.0,
    gaze: np.ndarray | None = None,
) -> list[SaccadeEvent]:
    """Velocity-based saccade detection with adaptive threshold.

    Uses median + MAD * multiplier as velocity floor when valid data exists,
    falling back to min_velocity_thresh.
    """
    # Adaptive baseline: median of valid low-velocity segments
    if valid.sum() > 50:
        vel_valid = vel[valid]
        vel_median = float(np.median(vel_valid))
        vel_mad = float(np.median(np.abs(vel_valid - vel_median))) * 1.4826
        vel_floor = max(min_velocity_thresh, vel_median + adaptive_mad_multiplier * vel_mad)
    else:
        vel_floor = min_velocity_thresh

    min_frames = max(1, int(min_duration_ms * sampling_rate / 1000))
    max_gap_frames = max(1, int(max_internal_gap_ms * sampling_rate / 1000))

    # Binary saccade indicator. Invalid frames are masked out (velocity is 0
    # there from compute_velocity), so saccades are never searched inside them
    # and a max_gap merge can never bridge across an invalid frame.
    is_saccade = (vel > vel_floor) & valid

    # Find contiguous segments
    events = []
    i = 0
    while i < len(is_saccade):
        if not is_saccade[i]:
            i += 1
            continue
        start = i
        while i < len(is_saccade) and is_saccade[i]:
            i += 1
        end = i - 1

        # Merge close segments
        gap_end = end
        for j in range(end + 1, min(len(is_saccade), end + max_gap_frames + 1)):
            if is_saccade[j]:
                gap_end = j
                break
        if gap_end > end:
            gap_len = gap_end - end
            if gap_len <= max_gap_frames:
                end = gap_end
                i = end + 1
                while i < len(is_saccade) and is_saccade[i]:
                    i += 1
                end = i - 1

        duration_frames = end - start + 1
        if duration_frames < min_frames:
            continue

        onset_ms = start * 1000.0 / sampling_rate
        offset_ms = (end + 1) * 1000.0 / sampling_rate

        if gaze is not None:
            start_xy = gaze[start]
            end_xy = gaze[min(end, len(gaze) - 1)]
            amplitude = float(np.sqrt(np.sum((end_xy - start_xy) ** 2)))
            direction = float(np.degrees(np.arctan2(end_xy[1] - start_xy[1], end_xy[0] - start_xy[0])))
        else:
            amplitude = 0.0
            start_xy = np.zeros(2)
            end_xy = np.zeros(2)
            direction = 0.0

        if amplitude < min_amplitude_deg:
            continue

        peak_vel = float(vel[start:end+1].max())
        mean_vel = float(vel[start:end+1].mean())

        events.append(SaccadeEvent(
            onset_idx=start, offset_idx=end,
            onset_ms=onset_ms, offset_ms=offset_ms,
            amplitude_deg=amplitude,
            peak_velocity_deg_s=peak_vel,
            mean_velocity_deg_s=mean_vel,
            start_xy=start_xy.copy(), end_xy=end_xy.copy(),
            direction_deg=direction,
        ))

    return events


# ──────────────────────────────────────────────
# Primary saccade identification (task-agnostic)
# ──────────────────────────────────────────────

def find_primary_saccade(
    saccades: list[SaccadeEvent],
    go_cue_ms: float,
    rt_min_ms: float = 80.0,
    rt_max_ms: float = 1500.0,
) -> Optional[SaccadeEvent]:
    """Identify the primary saccade: first responsive saccade after go cue.

    Task-agnostic: always the first saccade within RT window.
    No direction filtering — the saccade is what the subject actually made.
    """
    responsive = [
        s for s in saccades
        if rt_min_ms <= (s.onset_ms - go_cue_ms) <= rt_max_ms
    ]
    if not responsive:
        return None
    return responsive[0]


# ──────────────────────────────────────────────
# Goal position computation
# ──────────────────────────────────────────────

def compute_goal_position(
    stim_x_deg: float,
    stim_y_deg: float,
) -> np.ndarray:
    """Goal = stimulus position at go cue (task-agnostic).

    All tasks share the same definition: the stimulus is the target position.
    The auxiliary loss predicts continuous saccade metrics;
    task-specific correctness is handled implicitly by the metric values
    (e.g., spatial_error naturally penalizes saccades away from goal).
    """
    return np.array([stim_x_deg, stim_y_deg], dtype=np.float32)


# ──────────────────────────────────────────────
# Cue detection
# ──────────────────────────────────────────────

def detect_go_cue(
    stim_on: np.ndarray,   # [T]
    fix_on: np.ndarray,    # [T] (kept for API compatibility)
    task_id: int,          # (unused — task-agnostic extraction)
    sampling_rate: int = 1000,
) -> Tuple[float, bool]:
    """Detect go cue onset: first stim_on 0→1 transition (universal).

    All 4 tasks (Pro/Anti/Memory/Double) begin with stimulus presentation.
    The first stimulus onset is the universal "go" signal.
    For Memory, this captures the encoding phase onset;
    the memory-guided saccade after delay is a secondary event.
    """
    stim_bin = stim_on > 0.5
    transitions = np.where(np.diff(stim_bin.astype(int)) == 1)[0]
    if len(transitions) > 0:
        cue_frame = transitions[0] + 1
        return cue_frame * 1000.0 / sampling_rate, True
    return 0.0, False


def detect_memory_delay_start(
    stim_on: np.ndarray,
    sampling_rate: int = 1000,
) -> Tuple[float, bool]:
    """For Memory task: start of delay period (stimulus offset)."""
    stim_bin = stim_on > 0.5
    transitions = np.where(np.diff(stim_bin.astype(int)) == -1)[0]
    if len(transitions) > 0:
        return (transitions[0] + 1) * 1000.0 / sampling_rate, True
    return 0.0, False


# ──────────────────────────────────────────────
# Metric transforms (applied after raw extraction)
# ──────────────────────────────────────────────

def _apply_metric_transforms(m: TrialMetrics) -> None:
    """Apply distribution-improving transforms to raw metrics in-place.

    Transforms:
      spatial_error_deg  → log1p       (multiplicative error → additive log-space)
      reaction_time_ms   → log         (RT is classically log-normal)
      peak_velocity_deg_s → log1p      (compress right-skewed velocity)
      mean_velocity_deg_s → log1p      (same)
      saccadic_gain      → unchanged   (already a natural ratio [0,~2])
      first_amplitude_deg → unchanged  (near-linear with stimulus, robust-z fine)
    """
    eps = 1e-8

    # spatial_error: log1p (handles near-zero values gracefully)
    if m.metric_valid[0]:
        m.spatial_error_deg = np.log1p(max(m.spatial_error_deg, 0.0))

    # reaction_time: log (RT > 0, p01=93ms so always safe)
    if m.metric_valid[1]:
        m.reaction_time_ms = np.log(max(m.reaction_time_ms, eps))

    # velocity: no transform (VQ quantization bottleneck dominates, log1p doesn't help)
    # saccadic_gain, first_amplitude: unchanged


# ──────────────────────────────────────────────
# Main extraction function
# ──────────────────────────────────────────────

def extract_trial_metrics(
    eye: np.ndarray,          # [T, 8] raw eye data
    stim: np.ndarray,         # [T, 3] stim_on, stim_x, stim_y
    fix_on: np.ndarray,       # [T]
    task_id: int,
    cfg: dict | None = None,
) -> TrialMetrics:
    """Extract 8 trial-level saccade metrics from raw trial data.

    Args:
        eye: [T, 8] columns: lx,ly,larea,llabel, rx,ry,rarea,rlabel
        stim: [T, 3] columns: stim_on, stim_x(deg), stim_y(deg)
        fix_on: [T] fixation flag
        task_id: 0=Pro, 1=Anti, 2=Memory, 3=Double
        cfg: optional config dict with thresholds

    Returns:
        TrialMetrics with computed values and validity masks.
    """
    # Default config
    sampling_rate = 1000
    vel_thresh = 30.0
    min_dur_ms = 8
    max_gap_ms = 5
    min_amp_deg = 0.5  # lowered from 1.0 for Memory/Double tasks
    rt_min = 80.0
    rt_max = 1500.0  # wide window for all tasks
    smooth_win = 7
    smooth_order = 2

    if cfg is not None:
        ed = cfg.get("event_detector", {})
        sampling_rate = int(ed.get("sampling_rate", 1000))
        vel_thresh = float(ed.get("velocity", {}).get("min_threshold_deg_s", 30.0))
        min_dur_ms = int(ed.get("saccade", {}).get("min_duration_ms", 8))
        max_gap_ms = int(ed.get("saccade", {}).get("max_internal_gap_ms", 5))
        min_amp_deg = float(ed.get("saccade", {}).get("min_amplitude_deg", 1.0))
        rt_min = float(ed.get("reaction_time", {}).get("min_ms", 80.0))
        rt_max = float(ed.get("reaction_time", {}).get("max_ms", 1000.0))
        smooth_win = int(ed.get("smoothing", {}).get("window_ms", 7))
        smooth_order = int(ed.get("smoothing", {}).get("polyorder", 2))

    result = TrialMetrics()

    # ── 1. Extract gaze ──
    left_xy = eye[:, 0:2].astype(np.float64)
    right_xy = eye[:, 4:6].astype(np.float64)
    left_label = eye[:, 3].astype(np.int64)
    right_label = eye[:, 7].astype(np.int64)

    gaze, gaze_valid = cyclopean_gaze(left_xy, right_xy, left_label, right_label)
    if gaze_valid.sum() < 10:
        result.metric_valid[:] = False
        return result

    # ── 2. Compute velocity ──
    vel = compute_velocity(gaze, gaze_valid, sampling_rate, smooth_win, smooth_order)

    # ── 3. Detect saccades ──
    saccades = detect_saccades(
        vel, gaze_valid, sampling_rate,
        min_velocity_thresh=vel_thresh,
        min_duration_ms=min_dur_ms,
        max_internal_gap_ms=max_gap_ms,
        min_amplitude_deg=min_amp_deg,
        gaze=gaze,
    )

    # ── 4. Detect go cue ──
    stim_on = stim[:, 0]
    stim_x = stim[:, 1]
    stim_y = stim[:, 2]
    go_cue_ms, go_valid = detect_go_cue(stim_on, fix_on, task_id, sampling_rate)

    if not go_valid:
        result.metric_valid[:] = False
        return result

    # ── 6. Get stimulus position at go cue ──
    cue_frame = int(go_cue_ms * sampling_rate / 1000)
    cue_frame = min(cue_frame, len(stim_x) - 1)
    stim_pos = np.array([stim_x[cue_frame], stim_y[cue_frame]], dtype=np.float64)

    # ── 7. Compute goal (task-agnostic: goal = stimulus position) ──
    goal = compute_goal_position(stim_pos[0], stim_pos[1])

    # ── 8. Find primary saccade (task-agnostic: first responsive) ──
    primary = find_primary_saccade(saccades, go_cue_ms, rt_min, rt_max)

    if primary is None:
        result.metric_valid[:] = False
        return result

    # ── 9. Compute continuous metrics (uniform across tasks) ──
    result.spatial_error_deg = float(np.sqrt(np.sum((primary.end_xy - goal) ** 2)))
    result.reaction_time_ms = float(primary.onset_ms - go_cue_ms)
    result.mean_velocity_deg_s = primary.mean_velocity_deg_s
    result.peak_velocity_deg_s = primary.peak_velocity_deg_s
    result.first_amplitude_deg = primary.amplitude_deg

    # Saccadic gain: saccade_amplitude / distance(go_cue_gaze, goal)
    go_gaze = gaze[min(cue_frame, len(gaze) - 1)]
    target_ecc = float(np.sqrt(np.sum((goal - go_gaze) ** 2)))
    if target_ecc > 0.5:
        result.saccadic_gain = primary.amplitude_deg / target_ecc
    else:
        result.metric_valid[4] = False

    # ── 10. Apply distribution-improving transforms ──
    _apply_metric_transforms(result)

    return result


# ──────────────────────────────────────────────
# Batch extraction
# ──────────────────────────────────────────────

def extract_trial_metrics_batch(
    trials: list[dict],
    cfg: dict | None = None,
    verbose: bool = False,
) -> list[TrialMetrics]:
    """Extract metrics for a batch of trials.

    Args:
        trials: list of trial dicts from PackedTrialStore.read_trial()
        cfg: optional config
        verbose: print progress

    Returns:
        list of TrialMetrics
    """
    results = []
    for i, trial in enumerate(trials):
        try:
            eye = trial["eye"]
            stim = trial["stim"]
            fix_on = trial["fix_on"]
            task_id = int(trial["task_id"])
        except KeyError as e:
            if verbose:
                print(f"  Trial {i}: missing key {e}, skipping")
            results.append(TrialMetrics())
            results[-1].metric_valid[:] = False
            continue

        metrics = extract_trial_metrics(eye, stim, fix_on, task_id, cfg)
        results.append(metrics)

        if verbose and (i + 1) % 100 == 0:
            valid_count = sum(1 for m in results if m.metric_valid[0])
            print(f"  {i+1}/{len(trials)}: {valid_count} with valid primary saccade")

    return results


def metrics_to_arrays(metrics_list: list[TrialMetrics]) -> dict[str, np.ndarray]:
    """Convert TrialMetrics list to numpy arrays for training.

    Returns:
        dict with:
          continuous: [N, 6] float32
          metric_valid: [N, 6] bool
    """
    N = len(metrics_list)
    continuous = np.zeros((N, 6), dtype=np.float32)
    valid = np.ones((N, 6), dtype=bool)

    for i, m in enumerate(metrics_list):
        continuous[i] = [
            m.spatial_error_deg if not np.isnan(m.spatial_error_deg) else 0.0,
            m.reaction_time_ms if not np.isnan(m.reaction_time_ms) else 0.0,
            m.mean_velocity_deg_s if not np.isnan(m.mean_velocity_deg_s) else 0.0,
            m.peak_velocity_deg_s if not np.isnan(m.peak_velocity_deg_s) else 0.0,
            m.saccadic_gain if not np.isnan(m.saccadic_gain) else 0.0,
            m.first_amplitude_deg if not np.isnan(m.first_amplitude_deg) else 0.0,
        ]
        valid[i] = m.metric_valid

    return {
        "continuous": continuous,
        "metric_valid": valid,
    }
