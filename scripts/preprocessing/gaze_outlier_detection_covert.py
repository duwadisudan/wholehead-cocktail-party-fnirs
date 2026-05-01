"""
Gaze outlier detection — covert condition (Neon eye tracker).

Reads the aligned physio TSVs and events TSVs for the covert task,
interpolates missing values, lowpass-filters at 2 Hz, converts pixel gaze
to visual degrees via the arctan pinhole model, baseline-corrects to the
left-trial pre-cue mean, epochs around stimulus onset, and flags trials
whose sub-window gaze exceeds the outlier threshold. Outputs per-run CSV
of outlier trial indices and a diagnostic figure.

Author: Sudan Duwadi <sudan@bu.edu>
Notes: Code refactoring, documentation, and commenting were AI-assisted;
       all scientific decisions and accountability remain with the author.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.interpolate import interp1d

# CONFIGURATION

SUBJECTS = [
'37','38','42','45','48','49','50','51'
]

# RUNS TO PROCESS
# Each entry: (task_name_in_snirf, run_number, neon_folder_name)
# If the SNIRF or Neon folder doesn't exist for a subject, that run is skipped.
RUNS = [
    ("overtcontrol",   "01", "overtcontrol_run-01"),
    ("overtcontrol",   "02", "overtcontrol_run-02"),
    ("overtcontrol",   "03", "overtcontrol_run-03")
]

SNIRF_BASE = r"U:\eng_research_hrc_binauralhearinglab\Sudan\Labs\Sen Lab\Research_projects\Whole_Head_Cocktail_party\Cocktail_party_whole_head_master_data"
OUT_DIR    = os.path.join(SNIRF_BASE, "derivatives", "gaze_pruning_neon")
MASTER_DIR = os.path.join(SNIRF_BASE, "derivatives", "gaze_pruning_master")

# Signal processing
SAMPLING_RATE   = 100       # Hz — resample to uniform grid
LP_CUTOFF       = 2.0       # Hz lowpass
LP_ORDER        = 3

# Epoch parameters (seconds relative to onset)
PRE_CUE         = 2.0       # seconds before onset to include (for baseline)
POST_CUE        = 15.0      # seconds after onset

# Outlier detection parameters — task-specific
OUTLIER_PARAMS = {
    "overt": {
        "expected_left":  -30,
        "expected_right":  30,
        "tol":             20,
        "analysis_start_s": 1.0,   # seconds post-onset (after saccade lands)
        "analysis_end_s":   4.0,   # seconds post-onset (steady fixation)
        "check_sign":      True,   # mean of left must be <0, right must be >0
    },
    "covert": {
        "expected_left":   0,
        "expected_right":  0,
        "tol":             20,
        "analysis_start_s": -2.0,  # from epoch start (2s before onset)
        "analysis_end_s":   3.0,
        "check_sign":      False,
    },
    "control": {
        "expected_left":  -30,
        "expected_right":  30,
        "tol":             20,
        "analysis_start_s": 1.0,
        "analysis_end_s":   4.0,
        "check_sign":      True,
    },
}
WINDOW_SEC = 1.0  # sub-window size for outlier check (seconds)

# Neon scene camera specs (fixed for all recordings)
NEON_WIDTH   = 1600   # px
NEON_HEIGHT  = 1200   # px
NEON_FOV_X   = 103.0  # degrees horizontal
NEON_FOV_Y   = 77.0   # degrees vertical

# HELPERS

def pixel_to_gaze_angles(x_px, y_px=None,
                         width=NEON_WIDTH, height=NEON_HEIGHT,
                         fov_x_deg=NEON_FOV_X, fov_y_deg=NEON_FOV_Y):
    """Convert Neon scene-camera pixel coords to gaze angle in degrees.

    Uses the **pinhole camera model** with FOV-derived focal lengths:
        f_x = (width / 2) / tan(fov_x / 2)
        θ_x = arctan((x_px − cx) / f_x)

    Parameters
    ----------
    x_px : float or array-like
        Horizontal gaze pixel coordinate(s).  Origin at top-left.
    y_px : float, array-like or None
        Vertical gaze pixel coordinate(s).  If None only θ_x is computed.
    width, height : int
        Scene camera resolution in pixels (default 1600×1200 for Neon).
    fov_x_deg, fov_y_deg : float
        Horizontal / vertical field-of-view in degrees (103° / 77° for Neon).

    Returns
    -------
    theta_x_deg : ndarray  — horizontal angle (+ = right, − = left)
    theta_y_deg : ndarray or None  — vertical angle (+ = down, − = up)
    gaze_vector : tuple(ndarray, ndarray, ndarray) or None
        Unit gaze direction (gx, gy, gz) in camera coords, only when y_px given.

    Sign convention (image origin top-left):
        positive θ_x → gaze right
        negative θ_x → gaze left
        positive θ_y → gaze down
        negative θ_y → gaze up
    """
    x_px = np.asarray(x_px, dtype=np.float64)

    cx = width / 2.0
    cy = height / 2.0

    fov_x_rad = np.deg2rad(fov_x_deg)
    fov_y_rad = np.deg2rad(fov_y_deg)

    f_x = (width / 2.0) / np.tan(fov_x_rad / 2.0)
    f_y = (height / 2.0) / np.tan(fov_y_rad / 2.0)

    dx = x_px - cx
    theta_x_rad = np.arctan(dx / f_x)
    theta_x_deg = np.rad2deg(theta_x_rad)

    if y_px is None:
        return theta_x_deg, None, None

    y_px = np.asarray(y_px, dtype=np.float64)
    dy = y_px - cy
    theta_y_rad = np.arctan(dy / f_y)
    theta_y_deg = np.rad2deg(theta_y_rad)

    # Unit gaze vector in camera coordinates
    gx = np.tan(theta_x_rad)
    gy = np.tan(theta_y_rad)
    gz = np.ones_like(gx)
    norm = np.sqrt(gx * gx + gy * gy + gz * gz)
    gaze_vector = (gx / norm, gy / norm, gz / norm)

    return theta_x_deg, theta_y_deg, gaze_vector

def resample_uniform(timestamps, signal, fs):
    """Resample to a uniform grid at *fs* Hz via linear interpolation."""
    t_uniform = np.arange(timestamps[0], timestamps[-1], 1.0 / fs)
    interp_fn = interp1d(timestamps, signal, kind='linear',
                         bounds_error=False, fill_value=np.nan)
    return t_uniform, interp_fn(t_uniform)


def preprocess_gaze(timestamps, raw_gaze, fs=SAMPLING_RATE, cutoff=LP_CUTOFF,
                    order=LP_ORDER):
    """Resample, replace zeros with NaN, interpolate, lowpass filter."""
    t_u, gaze_u = resample_uniform(timestamps, raw_gaze, fs)

    # Zeros → NaN (missing data marker from Neon)
    gaze_u[gaze_u == 0] = np.nan

    # Linear interpolation to fill NaN
    nans = np.isnan(gaze_u)
    if nans.all():
        return t_u, gaze_u  # can't interpolate all-NaN
    if nans.any():
        good = ~nans
        gaze_u[nans] = np.interp(t_u[nans], t_u[good], gaze_u[good])
        # Extrapolate edges with nearest
        first_good = np.argmax(good)
        last_good  = len(good) - 1 - np.argmax(good[::-1])
        gaze_u[:first_good] = gaze_u[first_good]
        gaze_u[last_good+1:] = gaze_u[last_good]

    # Butterworth lowpass
    b, a = butter(order, cutoff / (0.5 * fs), btype='low')
    gaze_filt = filtfilt(b, a, gaze_u)
    return t_u, gaze_filt


def compute_baseline(t, gaze_deg, onsets, pre_s=PRE_CUE, fs=SAMPLING_RATE):
    """Global baseline: mean of the pre-cue segment across ALL trials.

    During pre-cue the subject should be fixating at centre regardless of
    the upcoming trial direction, so using all trials gives a more robust
    estimate than left-only (which can be contaminated by the previous
    trial's gaze direction).
    """
    baselines = []
    n_pre = int(pre_s * fs)
    for onset in onsets:
        idx = np.argmin(np.abs(t - onset))
        if idx >= n_pre:
            seg = gaze_deg[idx - n_pre : idx]
            baselines.append(np.nanmean(seg))
    if len(baselines) == 0:
        return 0.0
    return float(np.nanmean(baselines))


def epoch_trial(t, gaze, onset, pre_s, post_s, fs):
    """Extract epoch [onset-pre_s, onset+post_s].  Returns array of length
    fs*(pre_s+post_s), NaN-padded if out of range."""
    n_samples = int(fs * (pre_s + post_s))
    epoch = np.full(n_samples, np.nan)
    onset_idx = np.argmin(np.abs(t - onset))
    start_idx = onset_idx - int(pre_s * fs)
    for i in range(n_samples):
        src = start_idx + i
        if 0 <= src < len(gaze):
            epoch[i] = gaze[src]
    return epoch


def detect_outlier(epoch, side, params, fs=SAMPLING_RATE, pre_s=PRE_CUE):
    """Check whether a single trial epoch is an outlier.

    The analysis window should span the **steady fixation** period (after
    the saccade has landed), NOT the saccade itself.  Each sub-window's
    **mean** is tested — a single noisy sample no longer kills the trial.

    Parameters
    ----------
    epoch : 1-D array (length = fs * (pre_s + post_s))
    side  : 'left' or 'right'
    params: dict with expected_left/right, tol, analysis_start_s/end_s, check_sign

    Returns True if the trial is an outlier.
    """
    expected = params["expected_left"] if side == "left" else params["expected_right"]

    # Convert analysis window from seconds-post-onset to sample indices in epoch
    # epoch sample 0 = onset − pre_s
    a_start = int((pre_s + params["analysis_start_s"]) * fs)
    a_end   = int((pre_s + params["analysis_end_s"]) * fs)
    a_start = max(a_start, 0)
    a_end   = min(a_end, len(epoch))

    analysis = epoch[a_start:a_end]
    if len(analysis) == 0 or np.all(np.isnan(analysis)):
        return True  # no data → outlier

    win_samples = int(WINDOW_SEC * fs)
    n_windows = max(1, len(analysis) // win_samples)

    for w in range(n_windows):
        ws = w * win_samples
        we = min(ws + win_samples, len(analysis))
        window = analysis[ws:we]
        window = window[~np.isnan(window)]
        if len(window) == 0:
            return True

        win_mean = np.nanmean(window)

        # Sign check on window mean (overt only)
        if params["check_sign"]:
            if side == "left" and win_mean >= 0:
                return True
            if side == "right" and win_mean <= 0:
                return True

        # Deviation of window mean from expected target
        if np.abs(win_mean - expected) > params["tol"]:
            return True

    return False


# MAIN

def process_run(subj, task, run):
    """Process a single run: detect outlier trials, save results + figure."""
    nirs_dir = os.path.join(SNIRF_BASE, f"sub-{subj}", "nirs")
    base     = f"sub-{subj}_task-{task}_run-{run}"

    physio_tsv = os.path.join(nirs_dir, f"{base}_recording-eyetracking_physio.tsv")
    events_tsv = os.path.join(nirs_dir, f"{base}_events.tsv")

    if not os.path.isfile(physio_tsv):
        print(f"  [SKIP] Missing physio: {physio_tsv}")
        return None
    if not os.path.isfile(events_tsv):
        print(f"  [SKIP] Missing events: {events_tsv}")
        return None

    # Load
    df_physio = pd.read_csv(physio_tsv, sep='\t')
    df_events = pd.read_csv(events_tsv, sep='\t')

    timestamps = df_physio["timestamps"].values
    gaze_x_raw = df_physio["gaze2dX"].values

    # Parse Left / Right from trial_type column
    # Handle formats like "Overt Left", "Overt Right", "Covert Left", etc.
    df_events["side"] = df_events["trial_type"].str.lower().apply(
        lambda x: "left" if "left" in x else ("right" if "right" in x else "unknown")
    )

    onsets = df_events["onset"].values
    sides  = df_events["side"].values

    left_mask  = sides == "left"
    right_mask = sides == "right"
    left_onsets  = onsets[left_mask]
    right_onsets = onsets[right_mask]

    # Original 1-based trial index (position in the sorted onset sequence)
    # This matches the MATLAB convention
    trial_indices = np.arange(1, len(onsets) + 1)

    print(f"  {len(onsets)} trials ({left_mask.sum()} left, {right_mask.sum()} right)")

    # Preprocess
    t_u, gaze_filt = preprocess_gaze(timestamps, gaze_x_raw)

    # Convert pixels → degrees (FOV-based pinhole model)
    gaze_deg_raw, _, _ = pixel_to_gaze_angles(gaze_filt)

    # Baseline-correct in degree space
    baseline = compute_baseline(t_u, gaze_deg_raw, onsets)
    gaze_deg = gaze_deg_raw - baseline
    print(f"  Baseline (degrees): {baseline:.2f}°")

    # Get task-specific outlier params
    task_key = task.lower()
    if task_key not in OUTLIER_PARAMS:
        print(f"  [WARN] No outlier params for task '{task}', using overt defaults")
        task_key = "overt"
    params = OUTLIER_PARAMS[task_key]

    # Epoch & detect outliers
    n_epoch = int(SAMPLING_RATE * (PRE_CUE + POST_CUE))
    outlier_indices = []     # 1-based trial index in original order
    epochs_left  = []
    epochs_right = []
    outlier_flags = []       # parallel to onsets: True/False

    for trial_i in range(len(onsets)):
        onset = onsets[trial_i]
        side  = sides[trial_i]
        ep = epoch_trial(t_u, gaze_deg, onset, PRE_CUE, POST_CUE, SAMPLING_RATE)

        is_outlier = detect_outlier(ep, side, params)
        outlier_flags.append(is_outlier)

        if is_outlier:
            outlier_indices.append(trial_indices[trial_i])

        if side == "left":
            epochs_left.append((trial_indices[trial_i], ep, is_outlier))
        else:
            epochs_right.append((trial_indices[trial_i], ep, is_outlier))

    outlier_indices = sorted(outlier_indices)
    n_outliers = len(outlier_indices)
    print(f"  Outliers: {n_outliers}/{len(onsets)}  indices={outlier_indices}")

    # Figure: epoch plot with outliers
    # Plot from -1 s to +6 s post-onset
    plot_pre  = 1.0        # 1 s before onset
    plot_post = 6.0        # 6 s after onset
    pre_skip  = int((PRE_CUE - plot_pre) * SAMPLING_RATE)  # samples to skip
    n_plot = int((plot_pre + plot_post) * SAMPLING_RATE)
    time_vec = np.linspace(-plot_pre, plot_post, n_plot)

    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot good trials first (left=blue, right=red), then outliers on top
    for idx, ep, is_out in epochs_left:
        if not is_out:
            ax1.plot(time_vec, ep[pre_skip:pre_skip + n_plot],
                     'b', alpha=0.5, linewidth=0.8)
    for idx, ep, is_out in epochs_right:
        if not is_out:
            ax1.plot(time_vec, ep[pre_skip:pre_skip + n_plot],
                     'r', alpha=0.5, linewidth=0.8)

    # Outliers on top as dashed black
    for idx, ep, is_out in epochs_left + epochs_right:
        if is_out:
            ax1.plot(time_vec, ep[pre_skip:pre_skip + n_plot],
                     '--k', linewidth=2, alpha=0.8)

    # Mark onset with vertical line
    ax1.axvline(0, color='gray', linestyle=':', linewidth=1, label='Onset')

    ax1.set_xlabel("Time from onset (s)")
    ax1.set_ylabel("Gaze X (degrees)")
    ax1.set_title(f"{base}  |  outliers: {n_outliers}/{len(onsets)}  "
                  f"(expected L={params['expected_left']}° R={params['expected_right']}° "
                  f"tol={params['tol']}°)")

    # Legend with dummy handles
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], color='b', lw=1.5, label=f'Left ({left_mask.sum()})'),
        Line2D([0], [0], color='r', lw=1.5, label=f'Right ({right_mask.sum()})'),
        Line2D([0], [0], color='k', lw=2, ls='--', label=f'Outlier ({n_outliers})'),
    ]
    ax1.legend(handles=legend_handles, loc='upper right', fontsize=9)
    plt.tight_layout()

    # Save
    subj_out = os.path.join(OUT_DIR, f"sub-{subj}")
    os.makedirs(subj_out, exist_ok=True)
    fig_path = os.path.join(subj_out, f"{base}_gaze_outliers.png")
    plt.savefig(fig_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved figure: {fig_path}")

    return {
        "Subject": subj,
        "Task": task,
        "Run": f"Run{int(run)}",
        "OutlierIndices": "  ".join(str(i) for i in outlier_indices) if outlier_indices else "",
    }


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(MASTER_DIR, exist_ok=True)

    # Collect results keyed by (subject, task) for per-subject CSV output
    per_subj_task = {}   # (subj, task) → [row_dict, ...]
    all_results = []

    for subj in SUBJECTS:
        for task, run, _ in RUNS:
            label = f"sub-{subj}_task-{task}_run-{run}"
            print(f"\n{'='*60}")
            print(f"Processing {label}")
            print(f"{'='*60}")
            try:
                result = process_run(subj, task, run)
                if result is not None:
                    all_results.append(result)
                    key = (subj, task)
                    per_subj_task.setdefault(key, []).append(result)
            except Exception as e:
                print(f"  [ERROR] {label}: {e}")
                import traceback
                traceback.print_exc()

    # Per-subject / per-task CSVs (matches Tobii format)
    for (subj, task), rows in per_subj_task.items():
        subj_out = os.path.join(OUT_DIR, f"sub-{subj}")
        os.makedirs(subj_out, exist_ok=True)
        csv_name = f"sub-{subj}_task-{task}_gaze_outlier_results.csv"
        df_subj = pd.DataFrame(rows, columns=["Subject", "Task", "Run",
                                               "OutlierIndices"])
        # Save in subject subfolder
        csv_path = os.path.join(subj_out, csv_name)
        df_subj.to_csv(csv_path, index=False)
        print(f"  Saved: {csv_path}")

        # Also save flat copy in master folder
        master_path = os.path.join(MASTER_DIR, csv_name)
        df_subj.to_csv(master_path, index=False)
        print(f"  Saved: {master_path}")

    # Combined summary CSV (all subjects)
    if all_results:
        df_out = pd.DataFrame(all_results)
        csv_path = os.path.join(OUT_DIR, "gaze_outlier_results.csv")
        df_out.to_csv(csv_path, index=False)
        print(f"\n{'='*60}")
        print(f"Saved summary: {csv_path}")
        print(df_out.to_string(index=False))
    else:
        print("\nNo results to save.")
