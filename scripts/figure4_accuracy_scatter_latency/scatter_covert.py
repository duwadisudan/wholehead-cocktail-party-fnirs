"""
Per-subject accuracy scatter — covert condition (Figure 4).

Reads the consolidated accuracy summary table and renders the per-subject
covert scatter plot with chance-threshold band and identity reference line.

Author: Sudan Duwadi <sudan@bu.edu>
Notes: Code refactoring, documentation, and commenting were AI-assisted;
       all scientific decisions and accountability remain with the author.
"""
import sys
from whichscript import configure, enable_auto_logging
from wholehead_cocktail_party.paths import load_paths, require, whichscript_archive_dir

_PATHS = load_paths()
require(_PATHS, "classifier_results_root")

configure(
    archive=True,
    archive_only=False,
    archive_dir=str(whichscript_archive_dir(_PATHS)),
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


csv_file = str(_PATHS.classifier_results_root / "nested" / "rf_snr_0_20feat_balanced_depth5_oob" / "final_table.csv")

# Threshold for highlighting cells (values >= this threshold will be highlighted)
highlight_threshold = 62.3

output_dir = os.path.dirname(csv_file)

# Load & prepare data
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

# Prepare data for plotting
plot_df = df.copy()

# Map column names
covert_acc_col = 'Covert\nAccuracy' if 'Covert\nAccuracy' in plot_df.columns else 'Covert_perc'

# Convert to numeric
if covert_acc_col in plot_df.columns:
    plot_df[covert_acc_col] = pd.to_numeric(plot_df[covert_acc_col], errors='coerce')

# Sort by covert accuracy (descending) and reset index
plot_df = plot_df.sort_values(by=covert_acc_col, ascending=False, na_position='last').reset_index(drop=True)

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
covert_color = 'black'

# SINGLE-PANEL LAYOUT:
#   A  – Accuracy scatter (Covert only)

fig_v, ax1_v = plt.subplots(1, 1, figsize=(7, 5))

# PANEL A: Accuracy scatter (Covert only)
upper_chance = highlight_threshold  # 62.3%
lower_chance = 41.46

ax1_v.scatter(x_positions, plot_df[covert_acc_col],
              marker='o', s=120,
              color=covert_color, label='Accuracy', edgecolors='black', linewidths=1.5, alpha=0.85, zorder=3)

ax1_v.axhline(y=upper_chance, color='gray', linestyle='--', linewidth=2,
              alpha=0.7, label='Chance Upper')
ax1_v.axhline(y=lower_chance, color='gray', linestyle='-.', linewidth=2,
              alpha=0.7, label='Chance Lower')

covert_mean = plot_df[covert_acc_col].mean(skipna=True)

ax1_v.set_xlabel('Subject ID number', fontsize=AXIS_LABEL_FONT, fontweight='bold')
ax1_v.set_ylabel('Classification Performance (%)', fontsize=AXIS_LABEL_FONT, fontweight='bold')
ax1_v.set_xlim(0.5, len(plot_df) + 0.5)
ax1_v.set_ylim(0, 105)
ax1_v.set_xticks(x_positions)
ax1_v.set_xticklabels(plot_df['Subject'], rotation=70, ha='right', fontsize=SUBJECT_TICK_FONT)
ax1_v.tick_params(axis='y', labelsize=AXIS_TICK_FONT)
ax1_v.legend(loc='lower left', frameon=True, edgecolor='black', fontsize=LEGEND_FONT, framealpha=0.95)
ax1_v.grid(True, alpha=0.25, linestyle=':', linewidth=0.8)

ax1_v.set_facecolor('#FAFAFA')


fig_v.tight_layout()

# Mean marker: left-pointing triangle on the right spine + "Mean" label in axes fraction coords
ax1_v.plot(1.0, covert_mean, marker='<', ms=10, color=covert_color, clip_on=False,
           transform=ax1_v.get_yaxis_transform(), zorder=5)
ax1_v.text(1.02, covert_mean, f'{covert_mean:.1f}%\nMean',
           ha='left', va='center', fontsize=MEAN_TEXT_FONT, color=covert_color, fontweight='bold',
           transform=ax1_v.get_yaxis_transform(), clip_on=False)

fig_output_v = os.path.join(output_dir, 'accuracy_panel_covert_only.png')
fig_v.savefig(fig_output_v, format='png', dpi=300, bbox_inches='tight',
              facecolor='white', edgecolor='none')
print(f"Publication figure (covert only) saved as: {fig_output_v}")

fig_output_v_pdf = os.path.join(output_dir, 'accuracy_panel_covert_only.pdf')
fig_v.savefig(fig_output_v_pdf, format='pdf', bbox_inches='tight',
              facecolor='white', edgecolor='none')
print(f"Publication figure (covert only) saved as: {fig_output_v_pdf}")

fig_output_v_svg = os.path.join(output_dir, 'accuracy_panel_covert_only.svg')
fig_v.savefig(fig_output_v_svg, format='svg', bbox_inches='tight',
              facecolor='white', edgecolor='none')
print(f"Publication figure (covert only) saved as: {fig_output_v_svg}")

plt.show()
