"""
Gaze outlier detection — Tobii eye tracker (publication-quality figures).

Reads physio TSV + events TSV directly (no SNIRF/Homer3 dependency) for
subjects acquired with the Tobii system. Tobii gaze is normalised [0,1],
so the pixel→degree step uses (gaze − baseline) × FOV instead of the
arctan model used for Neon. Outputs per-run CSV of outlier trial indices
and PNG/PDF/SVG diagnostic figures matching the manuscript figure style.

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
from matplotlib.lines import Line2D
from scipy.signal import butter, filtfilt

# CONFIGURATION

SUBJECTS = ["01"]

TASKS = ["overt"]

RUNS = [1,2]  # run numbers to process

SNIRF_BASE = (
    r"U:\eng_research_hrc_binauralhearinglab\Sudan\Labs\Sen Lab"
    r"\Research_projects\Whole_Head_Cocktail_party"
    r"\Cocktail_party_whole_head_master_data"
)
OUT_BASE = os.path.join(SNIRF_BASE, "derivatives", "matlab_gaze_subjects")

# Signal processing
SAMPLING_RATE = 100   # Hz (Tobii Pro native)
LP_CUTOFF     = 2.0   # Hz lowpass
LP_ORDER      = 3

# Field-of-View — Tobii gaze2dX is normalised 0–1
FOV_X = 95   # degrees horizontal

# Epoch timing (seconds relative to onset)
PRE_CUE  = 2.0
POST_CUE = 15.0

# Outlier detection — from gaze_outlier_detection.py
OUTLIER_PARAMS = {
    "overt": {
        "expected_left":   -30,
        "expected_right":   30,
        "tol":              20,
        "analysis_start_s": 2.0,
        "analysis_end_s":   4.0,
        "check_sign":       True,
    },
    "covert": {
        "expected_left":    0,
        "expected_right":   0,
        "tol":              20,
        "analysis_start_s": -2.0,
        "analysis_end_s":    3.0,
        "check_sign":       False,
    },
    "control": {
        "expected_left":   -30,
        "expected_right":   30,
        "tol":              20,
        "analysis_start_s": 2.0,
        "analysis_end_s":   4.0,
        "check_sign":       True,
    },
}
WINDOW_SEC = 1.0  # sub-window size for outlier check

# Publication style (from table_maker_scatter_overt_only_pub_latency_CI.py)
AXIS_LABEL_FONT  = 18
AXIS_TICK_FONT   = 15
LEGEND_FONT      = 13
TITLE_FONT       = 16
BG_COLOR         = "#FAFAFA"

plt.rcParams.update({
    "font.size":         16,
    "font.family":       "sans-serif",
    "font.sans-serif":   ["Arial"],
    "axes.linewidth":    1.5,
    "xtick.major.width": 1.5,
    "ytick.major.width": 1.5,
    "xtick.major.size":  6,
    "ytick.major.size":  6,
})

# HELPERS

def preprocess_gaze(timestamps, raw_gaze, fs=SAMPLING_RATE, cutoff=LP_CUTOFF,
                    order=LP_ORDER):
    """Interpolate missing, lowpass filter on a uniform grid at *fs* Hz."""
    # Build uniform time grid
    t_uniform = np.arange(timestamps[0], timestamps[-1], 1.0 / fs)
    gaze_u = np.interp(t_uniform, timestamps, raw_gaze)

    # NaN / zero → interpolate
    bad = np.isnan(gaze_u) | (gaze_u == 0)
    if bad.all():
        return t_uniform, gaze_u
    if bad.any():
        good = ~bad
        gaze_u[bad] = np.interp(t_uniform[bad], t_uniform[good], gaze_u[good])
        first_good = np.argmax(good)
        last_good  = len(good) - 1 - np.argmax(good[::-1])
        gaze_u[:first_good] = gaze_u[first_good]
        gaze_u[last_good + 1:] = gaze_u[last_good]

    # Butterworth lowpass
    b, a = butter(order, cutoff / (0.5 * fs), btype="low")
    gaze_filt = filtfilt(b, a, gaze_u)
    return t_uniform, gaze_filt


def compute_baseline(t, gaze_signal, onsets, pre_s=PRE_CUE, fs=SAMPLING_RATE):
    """Global baseline: mean of pre-cue segment across all trials."""
    baselines = []
    n_pre = int(pre_s * fs)
    for onset in onsets:
        idx = np.argmin(np.abs(t - onset))
        if idx >= n_pre:
            seg = gaze_signal[idx - n_pre : idx]
            baselines.append(np.nanmean(seg))
    return float(np.nanmean(baselines)) if baselines else 0.0


def epoch_trial(t, gaze, onset, post_s, fs):
    """Extract onset-anchored segment [onset ... onset+post_s], NaN-padded."""
    n_samples = int(fs * post_s)
    epoch = np.full(n_samples, np.nan)
    onset_idx = np.argmin(np.abs(t - onset))
    start_idx = onset_idx
    for i in range(n_samples):
        src = start_idx + i
        if 0 <= src < len(gaze):
            epoch[i] = gaze[src]
    return epoch


def detect_outlier(epoch, side, params, fs=SAMPLING_RATE):
    """Check whether a single trial epoch is an outlier.

    Matches the MATLAB discard script: each sample in each 1 s window
    must satisfy sign and tolerance constraints.
    """
    expected = params["expected_left"] if side == "left" else params["expected_right"]

    a_start = int(params["analysis_start_s"] * fs)
    a_end   = int(params["analysis_end_s"] * fs)
    a_start = max(a_start, 0)
    a_end   = min(a_end, len(epoch))

    analysis = epoch[a_start:a_end]
    if len(analysis) == 0 or np.all(np.isnan(analysis)):
        return True

    win_samples = int(WINDOW_SEC * fs)
    n_windows = max(1, len(analysis) // win_samples)

    for w in range(n_windows):
        ws = w * win_samples
        we = min(ws + win_samples, len(analysis))
        window = analysis[ws:we]
        window = window[~np.isnan(window)]
        if len(window) == 0:
            return True

        if params["check_sign"]:
            if side == "left" and np.any(window >= 0):
                return True
            if side == "right" and np.any(window <= 0):
                return True

        if np.max(np.abs(window - expected)) > params["tol"]:
            return True

    return False




# MAIN PROCESSING

def process_run(subj, task, run_num):
    """Process one run: detect outliers, save results + pub-quality figures."""
    nirs_dir = os.path.join(SNIRF_BASE, f"sub-{subj}", "nirs")
    base = f"sub-{subj}_task-{task}_run-{run_num:02d}"

    physio_tsv = os.path.join(nirs_dir, f"{base}_recording-eyetracking_physio.tsv")
    events_tsv = os.path.join(nirs_dir, f"{base}_events.tsv")

    if not os.path.isfile(physio_tsv):
        print(f"  [SKIP] Missing physio: {physio_tsv}")
        return None
    if not os.path.isfile(events_tsv):
        print(f"  [SKIP] Missing events: {events_tsv}")
        return None

    # Load
    df_physio = pd.read_csv(physio_tsv, sep="\t")
    df_events = pd.read_csv(events_tsv, sep="\t")

    timestamps = df_physio["timestamps"].values
    gaze_x_raw = df_physio["gaze2dX"].values

    # Parse side from trial_type ("Overt Left" → "left")
    df_events["side"] = df_events["trial_type"].str.lower().apply(
        lambda x: "left" if "left" in x else ("right" if "right" in x else "unknown")
    )
    onsets = df_events["onset"].values
    sides  = df_events["side"].values
    left_mask  = sides == "left"
    right_mask = sides == "right"

    print(f"  {len(onsets)} trials ({left_mask.sum()} left, {right_mask.sum()} right)")

    # Preprocess
    t_u, gaze_filt = preprocess_gaze(timestamps, gaze_x_raw)

    # Baseline-correct using the MATLAB ordering
    left_onsets = onsets[left_mask]
    baseline = compute_baseline(t_u, gaze_filt, left_onsets)
    gaze_deg = (gaze_filt - baseline) * FOV_X
    print(f"  Baseline: {baseline:.6f} (normalized units)")


    # Get task-specific outlier params
    task_key = task.lower().replace("control", "control")
    # Handle combined tasks like "overtcontrol"
    for key in OUTLIER_PARAMS:
        if key in task_key:
            task_key = key
            break
    if task_key not in OUTLIER_PARAMS:
        print(f"  [WARN] No outlier params for '{task}', using overt defaults")
        task_key = "overt"
    params = OUTLIER_PARAMS[task_key]

    # Epoch & detect outliers
    trial_indices = np.arange(1, len(onsets) + 1)
    outlier_indices = []
    epochs_left  = []
    epochs_right = []

    for trial_i in range(len(onsets)):
        onset = onsets[trial_i]
        side  = sides[trial_i]
        ep = epoch_trial(t_u, gaze_deg, onset, PRE_CUE + POST_CUE, SAMPLING_RATE)
        is_outlier = detect_outlier(ep, side, params)

        if is_outlier:
            outlier_indices.append(trial_indices[trial_i])

        if side == "left":
            epochs_left.append((trial_indices[trial_i], ep, is_outlier))
        else:
            epochs_right.append((trial_indices[trial_i], ep, is_outlier))

    outlier_indices = sorted(outlier_indices)
    n_outliers = len(outlier_indices)
    print(f"  Outliers: {n_outliers}/{len(onsets)}  indices={outlier_indices}")

    # Publication figure
    plot_pre  = 0.0
    plot_post = 6.0
    pre_skip  = 0
    n_plot    = int((plot_pre + plot_post) * SAMPLING_RATE)
    time_vec  = np.linspace(plot_pre, plot_post, n_plot)

    fig, ax = plt.subplots(figsize=(7, 4.2))
    fig.patch.set_facecolor("white")

    # Good trials: semi-transparent
    for _, ep, is_out in epochs_left:
        if not is_out:
            ax.plot(time_vec, ep[pre_skip:pre_skip + n_plot],
                    color=(0, 0, 1, 0.35), linewidth=1)
    for _, ep, is_out in epochs_right:
        if not is_out:
            ax.plot(time_vec, ep[pre_skip:pre_skip + n_plot],
                    color=(1, 0, 0, 0.35), linewidth=1)

    # Outliers on top: dashed, semi-transparent
    for _, ep, is_out in epochs_left + epochs_right:
        if is_out:
            ax.plot(time_vec, ep[pre_skip:pre_skip + n_plot],
                    "--", color=(0, 0, 0, 0.6), linewidth=2)

    # Onset marker
    ax.axvline(0, color="gray", linestyle=":", linewidth=1.5, alpha=0.7)

    # Axes styling
    ax.set_facecolor(BG_COLOR)
    ax.set_xlabel("Time (s)", fontsize=AXIS_LABEL_FONT,
                  fontweight="bold", fontname="Arial")
    ax.set_ylabel("Gaze Position (degrees)", fontsize=AXIS_LABEL_FONT,
                  fontweight="bold", fontname="Arial")
    ax.set_title(f"Sub-{subj} {task} Run {run_num:02d}",
                 fontsize=TITLE_FONT, fontweight="bold", fontname="Arial")
    ax.tick_params(axis="both", labelsize=AXIS_TICK_FONT)
    ax.grid(True, alpha=0.25, linestyle=":", linewidth=0.8)
    ax.set_xlim(0, 6)

    # Legend
    legend_handles = [
        Line2D([0], [0], color="b", lw=1.5,
               label=f"Left ({left_mask.sum()})"),
        Line2D([0], [0], color="r", lw=1.5,
               label=f"Right ({right_mask.sum()})"),
        Line2D([0], [0], color="k", lw=2, ls="--",
               label=f"Outlier ({n_outliers})"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", frameon=True,
              edgecolor="black", fontsize=LEGEND_FONT, framealpha=0.95)

    fig.tight_layout()

    # Save
    subj_out = os.path.join(OUT_BASE, f"sub-{subj}")
    os.makedirs(subj_out, exist_ok=True)

    stem = f"{base}_gaze_outliers"
    fig_png = os.path.join(subj_out, f"{stem}.png")
    fig_pdf = os.path.join(subj_out, f"{stem}.pdf")
    fig_svg = os.path.join(subj_out, f"{stem}.svg")

    fig.savefig(fig_png, format="png", dpi=300, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    fig.savefig(fig_pdf, format="pdf", bbox_inches="tight",
                facecolor="white", edgecolor="none")
    fig.savefig(fig_svg, format="svg", bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)

    print(f"  Saved: {fig_png}")
    print(f"  Saved: {fig_pdf}")
    print(f"  Saved: {fig_svg}")

    # CSV results
    result = {
        "Subject": subj,
        "Task": task,
        "Run": f"Run{run_num}",
        "OutlierIndices": "  ".join(str(i) for i in outlier_indices)
                          if outlier_indices else "",
    }

    csv_path = os.path.join(subj_out,
                            f"sub-{subj}_task-{task}_gaze_outlier_results.csv")
    df_result = pd.DataFrame([result])
    # Append if file exists (multiple runs per task)
    if os.path.isfile(csv_path):
        df_existing = pd.read_csv(csv_path)
        df_result = pd.concat([df_existing, df_result], ignore_index=True)
    df_result.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")

    return result


if __name__ == "__main__":
    all_results = []

    for subj in SUBJECTS:
        for task in TASKS:
            for run_num in RUNS:
                label = f"sub-{subj}_task-{task}_run-{run_num:02d}"
                print(f"\n{'=' * 60}")
                print(f"Processing {label}")
                print(f"{'=' * 60}")
                try:
                    result = process_run(subj, task, run_num)
                    if result is not None:
                        all_results.append(result)
                except Exception as e:
                    print(f"  [ERROR] {label}: {e}")
                    import traceback
                    traceback.print_exc()

    if all_results:
        df_out = pd.DataFrame(all_results)
        summary_path = os.path.join(OUT_BASE, "gaze_outlier_summary.csv")
        os.makedirs(OUT_BASE, exist_ok=True)
        df_out.to_csv(summary_path, index=False)
        print(f"\nSaved summary: {summary_path}")
        print(df_out.to_string(index=False))
    else:
        print("\nNo results to save.")



