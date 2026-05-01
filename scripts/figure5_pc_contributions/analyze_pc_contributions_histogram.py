#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Group-level ROI histogram of mean PC contributions (Figure 5 panel C).

Aggregates per-subject channel-level PCA contributions from the classifier
outputs (per-fold `dprime_pca_summary.json` files), maps channels to Brodmann
ROIs, and computes group-level ROI mean and subject-frequency statistics.
Produces the ROI-contribution histogram and the top-N ROI bar chart.

Author: Sudan Duwadi <sudan@bu.edu>
Notes: Code refactoring, documentation, and commenting were AI-assisted;
       all scientific decisions and accountability remain with the author.
"""

#%%
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
#%%

import os
import json
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('default')
sns.set_palette('husl')


# ROI helpers (mirrors analyze_channel_stability_group_roi.py)
def load_roi_mapping(roi_csv_path: str) -> dict:
    """Load ROI mapping CSV into dict channel_label -> brodmann."""
    try:
        roi_df = pd.read_csv(roi_csv_path)
        roi_dict = dict(zip(roi_df['channel_label'], roi_df['brodmann']))
        print(f" Loaded ROI mapping for {len(roi_dict)} channels")
        return roi_dict
    except Exception as e:
        print(f" Warning: Could not load ROI mapping from {roi_csv_path}: {e}")
        print("   Proceeding with original channel names…")
        return {}


def get_display_name(channel_name: str, roi_mapping: dict) -> str:
    """Return ROI display name if available; otherwise the channel name.

    Applies the same cleaning as in the stability analysis (R-/L-, remove "(n)").
    """
    if channel_name in roi_mapping:
        brodmann = roi_mapping[channel_name]
        if isinstance(brodmann, str):
            if brodmann.startswith('Right-'):
                clean_name = brodmann.replace('Right-', 'R-')
            elif brodmann.startswith('Left-'):
                clean_name = brodmann.replace('Left-', 'L-')
            else:
                clean_name = brodmann
            import re
            clean_name = re.sub(r'\s*\(\d+\)', '', clean_name)
            return clean_name
    return channel_name


# Above-chance subject filtering
def load_above_chance_subjects(
    base_dir: str,
    conditions: list,
    threshold: float,
) -> dict:
    """Return a dict {condition: set_of_subject_id_strings} for subjects whose
    mean balanced accuracy exceeds `threshold` (in %) for that condition.

    Reads `final_table.csv` from `base_dir`.  Expected columns:
      Subject, Overt_perc, Covert_perc  (or similar naming).
    Returns an empty set for a condition if the column is not found.
    """
    csv_path = os.path.join(base_dir, 'final_table.csv')
    condition_col_map = {
        'overt':  ['Overt_perc',  'Overt\nAccuracy'],
        'covert': ['Covert_perc', 'Covert\nAccuracy'],
    }
    result = {cond: set() for cond in conditions}
    if not os.path.exists(csv_path):
        print(f" final_table.csv not found at {csv_path} — above-chance filter disabled")
        return result
    try:
        df = pd.read_csv(csv_path)
        df['Subject'] = df['Subject'].astype(str).str.strip()
        for cond in conditions:
            col = next(
                (c for c in condition_col_map.get(cond.lower(), []) if c in df.columns),
                None,
            )
            if col is None:
                print(f" No accuracy column found for '{cond}' in final_table.csv — skipping filter")
                continue
            df[col] = pd.to_numeric(df[col], errors='coerce')
            above = df.loc[df[col] > threshold, 'Subject']
            # zero-pad to match subject ID format used elsewhere
            result[cond] = {str(int(s)).zfill(2) for s in above if s.isdigit()}
            print(f" Above-chance subjects for {cond} (>{threshold}%): "
                  f"{sorted(result[cond])} ({len(result[cond])}/{len(df)})")
    except Exception as e:
        print(f" Error loading final_table.csv: {e} — above-chance filter disabled")
    return result


# Load and aggregate per-subject/condition contributions
def load_subject_condition_contrib(base_dir: str, subject_id: str, condition: str):
    """Load overall channel importance per fold for a subject-condition.

    Returns
    -------
    folds : list[dict[str, float]]
        Each item is a dict: channel_name -> importance_pct (sums to 100 across selected channels)
    meta : dict
        Metadata to report.
    """
    sub_folder = f"sub_{subject_id}_{condition}"
    json_path = os.path.join(base_dir, sub_folder, 'dprime_pca_summary.json')
    if not os.path.exists(json_path):
        print(f" Missing: {json_path}")
        return [], {}
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        folds_raw = data.get('overall_channel_importance_per_fold', [])
        folds = []
        for fold in folds_raw:
            fold_map = {}
            for entry in fold:
                ch_name = str(entry.get('channel_name', entry.get('channel_idx', 'NA')))
                imp = float(entry.get('importance_pct', 0.0))
                fold_map[ch_name] = fold_map.get(ch_name, 0.0) + imp
            folds.append(fold_map)
        meta = {
            'dprime_top_n': data.get('dprime_top_n'),
            'pca_variance_threshold': data.get('pca_variance_threshold'),
            'n_outer_folds': len(folds),
        }
        print(f" Loaded {len(folds)} folds for sub-{subject_id} {condition}")
        return folds, meta
    except Exception as e:
        print(f" Error reading {json_path}: {e}")
        return [], {}


def aggregate_roi_contrib_per_subject(fold_channel_maps, roi_mapping: dict) -> pd.DataFrame:
    """Map channels to ROI per fold and average within subject.

    Parameters
    ----------
    fold_channel_maps : list[dict]
        Per-fold dict of channel_name -> importance_pct (percent, typically sums to 100 per fold)
    roi_mapping : dict
        channel_name -> brodmann mapping

    Returns
    -------
    df : DataFrame
        Columns: roi_name, mean_importance, std_importance, n_folds
    """
    roi_folds = []
    for ch_map in fold_channel_maps:
        roi_map = defaultdict(float)
        for ch_name, imp in ch_map.items():
            roi = get_display_name(ch_name, roi_mapping)
            roi_map[roi] += float(imp)
        roi_folds.append(dict(roi_map))

    all_rois = sorted({roi for m in roi_folds for roi in m.keys()})
    rows = []
    for roi in all_rois:
        vals = [m.get(roi, 0.0) for m in roi_folds]
        vals_arr = np.array(vals, dtype=float)
        rows.append({
            'roi_name': roi,
            'mean_importance': float(vals_arr.mean()) if vals_arr.size else 0.0,
            'std_importance': float(vals_arr.std(ddof=1)) if vals_arr.size > 1 else 0.0,
            'n_folds': int(len(vals_arr)),
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values('mean_importance', ascending=False).reset_index(drop=True)
    return df


def aggregate_group_roi_contrib(subject_roi_tables):
    """Aggregate per-ROI contributions across subjects.

    Parameters
    ----------
    subject_roi_tables : list[tuple[str, pd.DataFrame]]
        Sequence of (subject_id, df_subject) where df_subject has roi_name and mean_importance

    Returns
    -------
    group_df : DataFrame with columns
        - roi_name
        - n_subjects
        - subject_frequency (fraction with mean_importance > 0)
        - mean_importance (across subjects)
        - sem_importance
        - min_importance, max_importance
    """
    long_rows = []
    for subject_id, df_subj in subject_roi_tables:
        if df_subj is None or df_subj.empty:
            continue
        for _, r in df_subj.iterrows():
            long_rows.append({
                'subject_id': subject_id,
                'roi_name': r['roi_name'],
                'mean_importance': float(r['mean_importance']),
            })
    long_df = pd.DataFrame(long_rows)
    if long_df.empty:
        return pd.DataFrame()

    group_rows = []
    total_subjects = len({sid for sid, _ in subject_roi_tables})
    for roi, df_roi in long_df.groupby('roi_name'):
        subj_vals = df_roi.groupby('subject_id')['mean_importance'].sum()
        vals = subj_vals.values.astype(float)
        n_subj_present = len(subj_vals)
        freq = n_subj_present / total_subjects if total_subjects else 0.0
        mean_imp = float(vals.mean()) if vals.size else 0.0
        sem_imp = float(vals.std(ddof=1) / np.sqrt(vals.size)) if vals.size > 1 else 0.0
        group_rows.append({
            'roi_name': roi,
            'n_subjects': int(n_subj_present),
            'subject_frequency': float(freq),
            'mean_importance': mean_imp,
            'sem_importance': sem_imp,
            'min_importance': float(vals.min()) if vals.size else 0.0,
            'max_importance': float(vals.max()) if vals.size else 0.0,
        })

    group_df = pd.DataFrame(group_rows)
    if not group_df.empty:
        group_df = group_df.sort_values('mean_importance', ascending=False).reset_index(drop=True)
    return group_df


# Plotting
def create_group_roi_plots(df: pd.DataFrame, output_dir: str, top_n: int = 12):
    os.makedirs(output_dir, exist_ok=True)
    if df is None or df.empty:
        print(" No data to plot.")
        return

    # Publication font/style constants (mirrors table_maker_scatter_overt_only_pub_latency_CI.py)
    AXIS_LABEL_FONT = 20
    AXIS_TICK_FONT  = 18
    LEGEND_FONT     = 17

    plt.rcParams.update({
        'font.size':           16,
        'font.family':         'sans-serif',
        'font.sans-serif':     ['Arial'],
        'axes.linewidth':       1.5,
        'xtick.major.width':    1.5,
        'ytick.major.width':    1.5,
        'xtick.major.size':     6,
        'ytick.major.size':     6,
    })

    # Landscape figure: wider than tall so bars have breathing room
    FIG_W_IN = 10
    FIG_H_IN = 6

    # 1) Top N ROIs by mean_importance (subject_frequency on right axis)
    fig, ax = plt.subplots(figsize=(FIG_W_IN, FIG_H_IN))
    top = df.head(top_n).copy()
    x = range(len(top))
    bar_handle = ax.bar(x, top['mean_importance'], facecolor='none', edgecolor='black', linewidth=1.5, label='Mean Contribution (%)')
    ax.set_xlabel('Brain Region (ROI)', fontsize=AXIS_LABEL_FONT, fontweight='bold')
    ax.set_ylabel('Mean Contribution (%)', fontsize=AXIS_LABEL_FONT, fontweight='bold')
    ax.set_xticks(range(len(top)))
    ax.set_xticklabels(top['roi_name'], rotation=45, ha='right', fontsize=AXIS_TICK_FONT)
    ax.tick_params(axis='y', labelsize=AXIS_TICK_FONT, width=1.5, length=6)
    ax.tick_params(axis='x', width=1.5, length=6)
    ax.spines['top'].set_visible(False)
    max_val = top['mean_importance'].max()
    ax.set_ylim(0, max(13, max_val * 1.15))

    # Right axis: subject frequency as scatter
    ax_right = ax.twinx()
    scatter_handle = ax_right.scatter(x, top['subject_frequency'] * 100, color='black', s=60, zorder=3,
                                      label='Subject Frequency (%)')
    ax_right.set_ylabel('Subject Frequency (%)', fontsize=AXIS_LABEL_FONT, fontweight='bold',
                        color='black')
    ax_right.tick_params(axis='y', labelsize=AXIS_TICK_FONT, colors='black', width=1.5, length=6)
    ax_right.set_ylim(0, 115)
    ax_right.spines['top'].set_visible(False)

    fig.tight_layout()
    base_name = f'top_{top_n}_rois_by_contribution'
    for ext in ('png', 'svg', 'pdf'):
        fig.savefig(os.path.join(output_dir, f'{base_name}.{ext}'), dpi=300, bbox_inches='tight')
    plt.close(fig)

    # Standalone legend figure
    handles = [bar_handle[0], scatter_handle]
    labels  = ['Mean Contribution', 'Subject Frequency']
    fig_leg, ax_leg = plt.subplots(figsize=(2.5, 0.6))
    ax_leg.axis('off')
    ax_leg.legend(handles, labels, fontsize=LEGEND_FONT, loc='center',
                  frameon=True, edgecolor='black', framealpha=0.95,
                  ncol=2, handlelength=1.5, handletextpad=0.5, columnspacing=1.0)
    fig_leg.tight_layout(pad=0.1)
    for ext in ('png', 'svg', 'pdf'):
        fig_leg.savefig(os.path.join(output_dir, f'legend_standalone.{ext}'),
                        dpi=300, bbox_inches='tight')
    plt.close(fig_leg)


def save_group_outputs(group_df: pd.DataFrame, subject_roi_tables, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    # Group summary
    group_df.to_csv(os.path.join(output_dir, 'roi_contrib_group_summary.csv'), index=False)

    # Subject-level long table
    long_rows = []
    for subject_id, df_subj in subject_roi_tables:
        if df_subj is None or df_subj.empty:
            continue
        for _, r in df_subj.iterrows():
            long_rows.append({
                'subject_id': subject_id,
                'roi_name': r['roi_name'],
                'mean_importance': float(r['mean_importance']),
                'std_importance': float(r['std_importance']),
                'n_folds': int(r['n_folds']),
            })
    pd.DataFrame(long_rows).to_csv(os.path.join(output_dir, 'roi_contrib_subject_summary.csv'), index=False)

    # JSON quick stats
    summary = {
        'total_unique_rois': int(len(group_df)),
        'rois_in_50_percent_subjects': int((group_df['subject_frequency'] >= 0.5).sum()) if not group_df.empty else 0,
        'mean_roi_contribution': float(group_df['mean_importance'].mean()) if not group_df.empty else 0.0,
        'median_roi_contribution': float(group_df['mean_importance'].median()) if not group_df.empty else 0.0,
    }
    with open(os.path.join(output_dir, 'roi_contrib_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)


def main():
    # Config
    base_dir = r"U:\\eng_research_hrc_binauralhearinglab\\Sudan\\Labs\\Sen Lab\\Research_projects\\Whole_Head_Cocktail_party\\Classifier_script_results\\nested\\rf_snr_0_20feat_balanced_depth5_oob"
    subjects = ['01','02','03','04','05','10','11','12','13','14','15','18','20','22','25','28','30','31','32','33','34','35','39','41','44','47']
    conditions = ['overt', 'covert']
    output_base_dir = r"U:\\eng_research_hrc_binauralhearinglab\\Sudan\\Labs\\Sen Lab\\Research_projects\\Whole_Head_Cocktail_party\\Classifier_script_results\\RF_above_chance_group_roi_contributions"
    roi_csv_path = r"U:\\eng_research_hrc_binauralhearinglab\\Sudan\\Labs\\Sen Lab\\Research_projects\\Whole_Head_Cocktail_party\\ROIs\\roi_master.csv"

    # Above-chance filtering
    flag_above_chance_only = True   # set False to include all subjects
    above_chance_threshold = 61.67  # same threshold used in table_maker_scatter

    roi_mapping = load_roi_mapping(roi_csv_path)

    above_chance_subjects = (
        load_above_chance_subjects(base_dir, conditions, above_chance_threshold)
        if flag_above_chance_only
        else {cond: set(subjects) for cond in conditions}
    )

    for condition in conditions:
        print(f"\n Group ROI contributions for {condition}")
        allowed_subjects = above_chance_subjects.get(condition, set(subjects))
        if flag_above_chance_only:
            print(f"   (above-chance filter ON — {len(allowed_subjects)} subjects)")
        subject_tables = []
        for sid in subjects:
            if sid not in allowed_subjects:
                print(f"    Skipping sub-{sid} (below chance for {condition})")
                continue
            folds, _ = load_subject_condition_contrib(base_dir, sid, condition)
            if not folds:
                continue
            df_subj = aggregate_roi_contrib_per_subject(folds, roi_mapping)
            subject_tables.append((sid, df_subj))

        if not subject_tables:
            print(f" No data found for condition {condition}")
            continue

        group_df = aggregate_group_roi_contrib(subject_tables)
        out_dir = os.path.join(output_base_dir, f"group_{condition}_roi_contrib")
        print(f" Creating plots → {out_dir}")
        create_group_roi_plots(group_df, out_dir, top_n=10)
        print(f" Saving CSV/JSON summaries → {out_dir}")
        save_group_outputs(group_df, subject_tables, out_dir)

        if group_df is not None and not group_df.empty:
            print(f"\n TOP 10 ROIs for {condition}:")
            for i, r in group_df.head(10).iterrows():
                print(f" {i+1:2d}. {r['roi_name']:<25s} | mean %: {r['mean_importance']:.2f} | subj freq: {r['subject_frequency']:.0%}")

    print("\n Done.")


if __name__ == '__main__':
    main()
