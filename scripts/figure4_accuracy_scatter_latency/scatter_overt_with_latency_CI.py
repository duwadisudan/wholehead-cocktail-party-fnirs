"""
Per-subject accuracy scatter with latency confidence intervals — overt (Figure 4).

Reads the accuracy summary table and renders the per-subject overt scatter
plot annotated with peak-accuracy latency and bootstrap confidence intervals.
Top panel of the vertical Figure 4 layout.

Author: Sudan Duwadi <sudan@bu.edu>
Notes: Code refactoring was AI-assisted; all scientific decisions and
       accountability remain with the author.
"""
import sys
from whichscript import configure, enable_auto_logging

configure(
    archive=True,
    archive_only=False,
    archive_dir=r"U:\eng_research_hrc_binauralhearinglab\Sudan\Labs\Sen Lab\Research_projects\Whole_Head_Cocktail_party\whichscript_archive",
    hide_sidecars=True,
    metadata=False,
    snapshot_script=False,
    snapshot_py=True,
    local_imports_snapshot=False,
)

enable_auto_logging()

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os


csv_file = r"U:\eng_research_hrc_binauralhearinglab\Sudan\Labs\Sen Lab\Research_projects\Whole_Head_Cocktail_party\Classifier_script_results\nested\rf_snr_0_20feat_balanced_depth5_oob\final_table.csv"

# Threshold for highlighting cells (values >= this threshold will be highlighted)
highlight_threshold = 62.3

output_dir = os.path.dirname(csv_file)

#########################################
# Load & prepare data
#########################################
df = pd.read_csv(csv_file)
print(f"Data loaded from: {csv_file}")
print(f"Original data shape: {df.shape}")
print(f"Original columns: {list(df.columns)}")

# Convert "N/A" strings back to NaN for proper numeric handling
for col in df.columns:
    if col != 'Subject':
        df[col] = df[col].replace(['N/A', 'nan', 'NaN'], float('nan'))
        df[col] = pd.to_numeric(df[col], errors='ignore')

# Rename columns to multi-line headers for better display
rename_map = {
    "Overt_perc": "Overt\nAccuracy",
    "Overt_peak_latency": "Overt\nPeak Latency",
    "Covert_perc": "Covert\nAccuracy",
    "Covert_peak_latency": "Covert\nPeak Latency"
}
present_renames = {k: v for k, v in rename_map.items() if k in df.columns}
if present_renames:
    df = df.rename(columns=present_renames)
    print(f"Renamed columns: {present_renames}")

# Ensure Subject is string
if 'Subject' in df.columns:
    df['Subject'] = df['Subject'].astype(str)
else:
    df.insert(0, 'Subject', df.index.astype(str))
    print("'Subject' column was missing; created from index.")

# Remove any pre-existing summary rows
summary_labels_to_remove = {"AVG", "MEAN", "MEDIAN", "ACCURACY"}
df = df[~df['Subject'].str.strip().str.upper().isin(summary_labels_to_remove)].reset_index(drop=True)

# Identify numeric columns (excluding Subject)
numeric_cols = [c for c in df.columns if c != 'Subject' and pd.api.types.is_numeric_dtype(df[c])]
print(f"Numeric columns detected: {numeric_cols}")

#########################################
# Prepare data for plotting
#########################################
plot_df = df.copy()

# Map column names
overt_acc_col = 'Overt\nAccuracy' if 'Overt\nAccuracy' in plot_df.columns else 'Overt_perc'
covert_acc_col = 'Covert\nAccuracy' if 'Covert\nAccuracy' in plot_df.columns else 'Covert_perc'
overt_lat_col = 'Overt\nPeak Latency' if 'Overt\nPeak Latency' in plot_df.columns else 'Overt_peak_latency'
covert_lat_col = 'Covert\nPeak Latency' if 'Covert\nPeak Latency' in plot_df.columns else 'Covert_peak_latency'

# Convert to numeric
for col in [overt_acc_col, covert_acc_col, overt_lat_col, covert_lat_col]:
    if col in plot_df.columns:
        plot_df[col] = pd.to_numeric(plot_df[col], errors='coerce')

# Sort by overt accuracy (descending) and reset index
plot_df = plot_df.sort_values(by=overt_acc_col, ascending=False, na_position='last').reset_index(drop=True)

# Create ranked x-axis positions
x_positions = np.arange(1, len(plot_df) + 1)

# Publication settings
BASE_FONT_SIZE = 16
AXIS_LABEL_FONT = 18
AXIS_TICK_FONT = 15
SUBJECT_TICK_FONT = 12
LEGEND_FONT = 13
PANEL_LETTER_FONT = 22
MEAN_TICK_FONT = 14
MEAN_TEXT_FONT = 13

plt.rcParams.update({
    'font.size': BASE_FONT_SIZE,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial'],
    'axes.linewidth': 1.5,
    'xtick.major.width': 1.5,
    'ytick.major.width': 1.5,
    'xtick.major.size': 6,
    'ytick.major.size': 6,
})

# Colors
overt_color = 'black'
event1_color = '#4DAF4A'  # Vibrant green - Cue Onset
event2_color = '#984EA3'  # Vibrant purple - Movie Onset (at 2s)
event3_color = '#FF7F00'  # Vibrant orange - Movie Offset (at 5s)

#########################################
# Compute per-subject latency to 95% of peak
# Method: average balanced_accuracy across all folds -> one mean curve per subject
#         then find peak and first crossing of 0.95 * peak on that mean curve
#########################################
import json

lat95_values  = []   # one value per subject (latency to 95% of peak on mean curve)
# Store curve data for diagnostic figure
diag_data = []       # list of dicts: {subj, t, mean_ba, peak_t, peak_ba, lat95_t, lat95_ba}

for subj_str in plot_df['Subject']:
    subj_id   = str(int(subj_str)).zfill(2)
    fold_file = os.path.join(output_dir, f'sub_{subj_id}_overt', 'per_fold_accuracy_curves.json')
    if not os.path.isfile(fold_file):
        print(f"  WARNING: {fold_file} not found – filling NaN")
        lat95_values.append(float('nan'))
        diag_data.append(None)
        continue
    with open(fold_file) as fh:
        folds = json.load(fh)

    # Align all folds to a common time grid (use fold 0's times as reference)
    t_ref = np.array(folds[0]['time'])
    ba_stack = []
    for fold in folds:
        t_fold  = np.array(fold['time'])
        ba_fold = np.array(fold['balanced_accuracy'])
        if len(t_fold) == len(t_ref) and np.allclose(t_fold, t_ref):
            ba_stack.append(ba_fold)
        else:
            # interpolate to common grid if needed
            ba_interp = np.interp(t_ref, t_fold, ba_fold)
            ba_stack.append(ba_interp)
    mean_ba = np.nanmean(np.array(ba_stack), axis=0)
    std_ba   = np.nanstd(np.array(ba_stack), axis=0, ddof=1)
    n_folds  = len(ba_stack)

    # Find peak on mean curve restricted to 0–5 s window
    search_mask  = (t_ref >= 0) & (t_ref <= 5)
    search_ba    = np.where(search_mask, mean_ba, -np.inf)
    peak_idx     = int(np.argmax(search_ba))
    peak_t       = float(t_ref[peak_idx])
    peak_ba      = float(mean_ba[peak_idx])

    # First crossing of lower bound of 95% CI at peak (rising edge – search from t=0 up to and including peak)
    sem_at_peak  = std_ba[peak_idx] / np.sqrt(n_folds)
    threshold    = peak_ba - 1.96 * sem_at_peak
    rising_mask  = (t_ref >= 0) & (np.arange(len(t_ref)) <= peak_idx)
    crossing_idx = np.where(rising_mask & (mean_ba >= threshold))[0]
    if len(crossing_idx) > 0:
        lat95_t  = float(t_ref[crossing_idx[0]])
        lat95_ba = float(mean_ba[crossing_idx[0]])
    else:
        lat95_t  = float('nan')
        lat95_ba = float('nan')

    lat95_values.append(lat95_t)
    diag_data.append({'subj': subj_str, 't': t_ref, 'mean_ba': mean_ba,
                      'peak_t': peak_t, 'peak_ba': peak_ba,
                      'threshold': threshold,
                      'lat95_t': lat95_t, 'lat95_ba': lat95_ba})

lat95_values = np.array(lat95_values, dtype=float)

# Filter to above-chance subjects only (for latency histogram)
above_chance_mask  = np.array(plot_df[overt_acc_col].values > highlight_threshold, dtype=bool)
lat95_above_chance = lat95_values.copy()
lat95_above_chance[np.logical_not(above_chance_mask)] = float('nan')  # blank out below-chance subjects

n_above = int(np.sum(above_chance_mask))
overall_lat95_mean = float(np.nanmean(lat95_above_chance))
print(f"Latency-to-lower-bound-of-95%-CI (on mean curve) computed for "
      f"{int(np.sum(~np.isnan(lat95_values)))}/{len(lat95_values)} subjects")
print(f"  Above-chance subjects (>{highlight_threshold}%): {n_above}/{len(lat95_values)}")
print(f"  Grand mean latency (above-chance only): {overall_lat95_mean:.3f} s")

# ==============================================================================
# DIAGNOSTIC FIGURE: mean accuracy curve per subject with
#   black * = actual peak
#   red   * = first crossing of 95% of peak (rising edge)
# ==============================================================================
n_subj    = len(diag_data)
n_cols    = 5
n_rows    = int(np.ceil(n_subj / n_cols))
fig_diag, axes_diag = plt.subplots(n_rows, n_cols,
                                    figsize=(4 * n_cols, 3.2 * n_rows),
                                    sharey=False)
axes_diag = axes_diag.flatten()

for i, dd in enumerate(diag_data):
    ax = axes_diag[i]
    if dd is None:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                transform=ax.transAxes, fontsize=10)
        ax.set_title(f"sub {plot_df['Subject'].iloc[i]}", fontsize=10)
        continue
    ax.plot(dd['t'], dd['mean_ba'] * 100, color='black', linewidth=1.5)
    ax.axhline(y=dd['threshold'] * 100, color='gray',
               linestyle=':', linewidth=1, alpha=0.7)
    # Black star at peak
    ax.plot(dd['peak_t'], dd['peak_ba'] * 100, marker='*', ms=14,
            color='black', zorder=5, label='Peak')
    # Red star at 95%-crossing
    if not np.isnan(dd['lat95_t']):
        ax.plot(dd['lat95_t'], dd['lat95_ba'] * 100, marker='*', ms=14,
                color='red', zorder=5, label='95% rise')
    ax.axvline(x=0, color=event1_color, linestyle='--', linewidth=1, alpha=0.6)
    ax.axvline(x=2, color=event2_color, linestyle='--', linewidth=1, alpha=0.6)
    ax.axvline(x=5, color=event3_color, linestyle='--', linewidth=1, alpha=0.6)
    ax.set_title(f"sub {dd['subj']}  latCI={dd['lat95_t']:.2f}s", fontsize=9)
    ax.set_xlabel('Time (s)', fontsize=8)
    ax.set_ylabel('Bal. Acc. (%)', fontsize=8)
    ax.tick_params(labelsize=8)
    ax.set_xlim(-2, 10)
    ax.set_facecolor('#FAFAFA')
    if i == 0:
        ax.legend(fontsize=7, loc='upper left', frameon=True)

# Hide any unused subplot axes
for j in range(n_subj, len(axes_diag)):
    axes_diag[j].set_visible(False)

fig_diag.suptitle('Mean accuracy curve per subject\n'
                  'Black star = peak  |  Red star = first crossing of lower bound of 95% CI',
                  fontsize=13, fontweight='bold')
fig_diag.tight_layout()

diag_png = os.path.join(output_dir, 'lat95_verification_curves.png')
fig_diag.savefig(diag_png, dpi=150, bbox_inches='tight',
                 facecolor='white', edgecolor='none')
print(f"Diagnostic verification figure saved as: {diag_png}")

# ==============================================================================
# TWO-PANEL LAYOUT:
#   A  – Accuracy scatter  (top)
#   B  – Latency-to-95%-peak histogram  (bottom)
# ==============================================================================

fig_v, (ax1_v, ax2_v) = plt.subplots(2, 1, figsize=(7, 10))

# --- PANEL A: Accuracy scatter (Overt only) ---
upper_chance = highlight_threshold  # 61.67%
lower_chance = 41.46

ax1_v.scatter(x_positions, plot_df[overt_acc_col],
              marker='o', s=120,
              color=overt_color, label='Accuracy', edgecolors='black', linewidths=1.5, alpha=0.85, zorder=3)

ax1_v.axhline(y=upper_chance, color='gray', linestyle='--', linewidth=2,
              alpha=0.7, label='Chance Upper')
ax1_v.axhline(y=lower_chance, color='gray', linestyle='-.', linewidth=2,
              alpha=0.7, label='Chance Lower')

overt_mean = plot_df[overt_acc_col].mean(skipna=True)

ax1_v.set_xlabel('Subject ID number', fontsize=AXIS_LABEL_FONT, fontweight='bold')
ax1_v.set_ylabel('Classification Performance (%)', fontsize=AXIS_LABEL_FONT, fontweight='bold')
ax1_v.set_xlim(0.5, len(plot_df) + 0.5)
ax1_v.set_ylim(0, 105)
ax1_v.set_xticks(x_positions)
ax1_v.set_xticklabels(plot_df['Subject'], rotation=70, ha='right', fontsize=SUBJECT_TICK_FONT)
ax1_v.tick_params(axis='y', labelsize=AXIS_TICK_FONT)
ax1_v.legend(loc='lower left', frameon=True, edgecolor='black', fontsize=LEGEND_FONT, framealpha=0.95)
ax1_v.grid(True, alpha=0.25, linestyle=':', linewidth=0.8)
ax1_v.text(-0.1, 1.05, 'A', transform=ax1_v.transAxes,
           fontsize=PANEL_LETTER_FONT, fontweight='bold', va='top')
ax1_v.set_facecolor('#FAFAFA')

# Right y-axis with mean tick
ax1_v_right = ax1_v.twinx()
ax1_v_right.set_ylim(0, 105)
ax1_v_right.set_yticks([overt_mean])
ax1_v_right.set_yticklabels([f'{overt_mean:.1f}%'], fontsize=MEAN_TICK_FONT, fontweight='bold')
for tick_label in ax1_v_right.get_yticklabels():
    tick_label.set_color(overt_color)
ax1_v_right.tick_params(axis='y', length=0)

# Left-pointing arrow at mean on right axis border
ax1_v.annotate('', xy=(len(plot_df) + 0.5 - 1.5, overt_mean), xytext=(len(plot_df) + 0.5, overt_mean),
               arrowprops=dict(arrowstyle='->', lw=2.5, color=overt_color),
               annotation_clip=False)

ax1_v.text(len(plot_df) + 0.55, overt_mean + 3, 'Mean',
           ha='left', va='bottom', fontsize=MEAN_TEXT_FONT, color=overt_color, fontweight='bold')
ax1_v_right.set_ylabel('')

# --- PANEL B: Latency-to-95%-peak histogram (Overt only) ---
# Peak search is restricted to 0-5 s so lat95 values are within that range
bins = np.arange(-0.25, 5.25, 0.5)

lat95_valid = lat95_above_chance[~np.isnan(lat95_above_chance)]
print(f"  Histogram includes {len(lat95_valid)} above-chance subjects")
ax2_v.hist(lat95_valid, bins=bins,
           facecolor='none', edgecolor='black', linewidth=1.5,
            rwidth=0.85, align='mid')

ax2_v.axvline(x=0, color=event1_color, linestyle='--', linewidth=2.5, alpha=0.8, label='Cue Onset')
ax2_v.axvline(x=2, color=event2_color, linestyle='--', linewidth=2.5, alpha=0.9, label='Movie Onset')
ax2_v.axvline(x=5, color=event3_color, linestyle='--', linewidth=2.5, alpha=0.8, label='Movie Offset')

overt_lat_mean = overall_lat95_mean

ax2_v.set_xlabel('Latency (s)', fontsize=AXIS_LABEL_FONT, fontweight='bold')
ax2_v.set_ylabel('Number of Subjects', fontsize=AXIS_LABEL_FONT, fontweight='bold')
ax2_v.tick_params(axis='both', labelsize=AXIS_TICK_FONT)
ax2_v.legend(loc='upper left', frameon=True, edgecolor='black', fontsize=LEGEND_FONT, framealpha=0.95)
ax2_v.grid(True, alpha=0.25, linestyle=':', linewidth=0.8, axis='y')
ax2_v.text(-0.1, 1.05, 'B', transform=ax2_v.transAxes,
           fontsize=PANEL_LETTER_FONT, fontweight='bold', va='top')
ax2_v.set_facecolor('#FAFAFA')

# Top x-axis with mean latency tick
ax2_v_top = ax2_v.twiny()
ax2_v_top.set_xlim(ax2_v.get_xlim())
ax2_v_top.set_xticks([overt_lat_mean])
ax2_v_top.set_xticklabels([f'{overt_lat_mean:.2f}s'], fontsize=MEAN_TICK_FONT, fontweight='bold')
ax2_v_top.tick_params(axis='x', pad=8)
for tick_label in ax2_v_top.get_xticklabels():
    tick_label.set_color(overt_color)
ax2_v_top.tick_params(axis='x', length=0)

# Down-pointing arrow at mean on top axis border
y_top_v = ax2_v.get_ylim()[1]
ax2_v.annotate('', xy=(overt_lat_mean, y_top_v - 0.4), xytext=(overt_lat_mean, y_top_v),
               arrowprops=dict(arrowstyle='->', lw=2.5, color=overt_color),
               annotation_clip=False)

y_max = ax2_v.get_ylim()[1]
ax2_v.text(overt_lat_mean, y_max * 1.08, 'Mean',
           ha='center', va='bottom', fontsize=MEAN_TEXT_FONT, color=overt_color, fontweight='bold', clip_on=False)
ax2_v_top.set_xlabel('')

fig_v.tight_layout()

fig_output_v = os.path.join(output_dir, 'accuracy_latency_panels_overt_only_vertical_CI.png')
fig_v.savefig(fig_output_v, format='png', dpi=300, bbox_inches='tight',
              facecolor='white', edgecolor='none')
print(f"Publication figure (2-panel, overt only) saved as: {fig_output_v}")

fig_output_v_pdf = os.path.join(output_dir, 'accuracy_latency_panels_overt_only_vertical_CI.pdf')
fig_v.savefig(fig_output_v_pdf, format='pdf', bbox_inches='tight',
              facecolor='white', edgecolor='none')
print(f"Publication figure (2-panel, overt only) saved as: {fig_output_v_pdf}")

fig_output_v_svg = os.path.join(output_dir, 'accuracy_latency_panels_overt_only_vertical_CI.svg')
fig_v.savefig(fig_output_v_svg, format='svg', bbox_inches='tight',
              facecolor='white', edgecolor='none')
print(f"Publication figure (2-panel, overt only) saved as: {fig_output_v_svg}")

plt.show()
