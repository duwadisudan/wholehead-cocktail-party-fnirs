"""
Behavioral accuracy plotter — overt vs covert (supplementary).

Python port of the legacy MATLAB behav_plotter. Reads per-subject behavioral
response CSVs, computes percent correct for the overt and covert conditions,
renders a per-subject scatter sorted by accuracy descending (overt = filled
blue, covert = open red), and exports a CSV of incorrect-trial counts.

Author: Sudan Duwadi <sudan@bu.edu>
Notes: Code refactoring, documentation, and commenting were AI-assisted;
       all scientific decisions and accountability remain with the author.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# ——— setup ———
subjects = [
    '01', '02', '03', '04', '05', '10', '11', '12', '13', '14',
    '15', '18', '20', '22', '25', '28', '30', '31', '32', '33',
    '34', '35', '39', '41', '44', '47',
]
tasks = ['overt', 'covert']

base_dir = (
    r'U:\eng_research_hrc_binauralhearinglab\Sudan\Labs\Sen Lab'
    r'\Research_projects\Whole_Head_Cocktail_party\Response_data'
)

percent_correct = np.full((len(subjects), len(tasks)), np.nan)
n_incorrect = np.full((len(subjects), len(tasks)), np.nan)

for i, sub_id in enumerate(subjects):
    sub_folder = os.path.join(base_dir, f'sub-{sub_id}')
    for j, task in enumerate(tasks):
        file_name = os.path.join(
            sub_folder, f'sub-{sub_id}_task-{task}_response_correct_trials.csv'
        )
        if os.path.isfile(file_name):
            df = pd.read_csv(file_name)

            # Trial columns are CorrectTrials_1 .. CorrectTrials_N (0/1 each)
            trial_cols = [c for c in df.columns if c.startswith('CorrectTrials_')]

            # Combine both runs (rows) into one array
            combined = df[trial_cols].values.flatten()

            n_total = combined.size
            n_correct = int(combined.sum())
            n_incorrect[i, j] = n_total - n_correct
            percent_correct[i, j] = 100.0 * n_correct / n_total
        else:
            print(f'WARNING — Missing: {file_name}')

# ——— Publication plotting parameters ———
BASE_FONT_SIZE = 16
AXIS_LABEL_FONT = 18
AXIS_TICK_FONT = 15
SUBJECT_TICK_FONT = 12
LEGEND_FONT = 13
MEAN_TEXT_FONT = 13

# Colors (matched to pairwise_accuracy_analysis.py)
OVERT_COLOR = '#4878CF'   # muted blue
COVERT_COLOR = '#D65F5F'  # muted red

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

# ——— build sorted dataframe ———
plot_df = pd.DataFrame({
    'Subject': subjects,
    'Overt': percent_correct[:, 0],
    'Covert': percent_correct[:, 1],
})
# Sort by overt accuracy descending (primary axis)
plot_df = plot_df.sort_values(by='Overt', ascending=False, na_position='last').reset_index(drop=True)
x_positions = np.arange(1, len(plot_df) + 1)

# ——— plot (scatter) ———
fig, ax = plt.subplots(figsize=(7, 5))

dot_offset = 0.15  # half-spacing between overt and covert dots

# Overt markers (filled blue)
ax.scatter(x_positions - dot_offset, plot_df['Overt'],
           marker='o', s=80, color=OVERT_COLOR, edgecolors='black',
           linewidths=0.8, zorder=3, alpha=0.85, label='Overt')

# Covert markers (open red)
ax.scatter(x_positions + dot_offset, plot_df['Covert'],
           marker='o', s=80, facecolors='white', edgecolors=COVERT_COLOR,
           linewidths=1.5, zorder=3, label='Covert')

ax.axhline(70, color='black', linestyle=':', linewidth=1.5, zorder=1)

ax.set_xlabel('Subject ID number', fontsize=AXIS_LABEL_FONT, fontweight='bold')
ax.set_ylabel('Percent Correct (%)', fontsize=AXIS_LABEL_FONT, fontweight='bold')
ax.set_xlim(0.5, len(plot_df) + 0.5)
ax.set_ylim(0, 105)
ax.set_xticks(x_positions)
ax.set_xticklabels(plot_df['Subject'], rotation=70, ha='right', fontsize=SUBJECT_TICK_FONT)
ax.tick_params(axis='y', labelsize=AXIS_TICK_FONT)
ax.legend(loc='lower right', frameon=True, edgecolor='black', fontsize=LEGEND_FONT, framealpha=0.95)
ax.grid(True, axis='y', alpha=0.25, linestyle=':', linewidth=0.8)
ax.set_facecolor('#FAFAFA')


fig.tight_layout()

# ——— save figure (PNG, PDF, SVG) ———
output_dir = base_dir
for fmt in ['png', 'pdf', 'svg']:
    out_path = os.path.join(output_dir, f'percent_correct_by_subject_task.{fmt}')
    save_kw = dict(
        format=fmt, bbox_inches='tight', facecolor='white', edgecolor='none',
    )
    if fmt == 'png':
        save_kw['dpi'] = 300
    fig.savefig(out_path, **save_kw)
    print(f'Figure saved: {out_path}')

# ——— write CSV of incorrect counts ———
csv_df = pd.DataFrame({
    'Subject': subjects,
    'overt_incorrect': n_incorrect[:, 0],
    'covert_incorrect': n_incorrect[:, 1],
})
csv_path = os.path.join(output_dir, 'incorrect_trials_by_subject.csv')
csv_df.to_csv(csv_path, index=False)
print(f'CSV saved: {csv_path}')

plt.show()
