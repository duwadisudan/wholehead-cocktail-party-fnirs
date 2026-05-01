#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Angular Gyrus activation reliability vs overt accuracy (supplementary).

Loads per-subject trial-level activation reliability outputs from the
trial-SNR computation, restricts to Angular Gyrus channels (left + right
pooled), and correlates the subject mean AG activation reliability against
overt RF decoding accuracy. Activation Reliability is defined as |μ|/σ where
μ is the mean HbO response across trials and σ is the across-trial SD.
The all-channels variant is plotted alongside for reference.

Author: Sudan Duwadi <sudan@bu.edu>
Notes: Code refactoring, documentation, and commenting were AI-assisted;
       all scientific decisions and accountability remain with the author.
"""

#%%

import sys
from whichscript import enable_auto_logging

enable_auto_logging()
#%%
import os
import json
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

warnings.filterwarnings("ignore")

# ------------------------------------------------------------------ CONFIG
SUBJECTS = ['01','02','03','04','05','10','11','12','13','14','15','18','20','22','25','28','30','31','32','33','34','35','39','41','44','47']
RUN_TYPE = 'overt'

PROJECT_ROOT = r"U:\eng_research_hrc_binauralhearinglab\Sudan\Labs\Sen Lab\Research_projects\Whole_Head_Cocktail_party"

# SNR results directory
SNR_DIR = os.path.join(PROJECT_ROOT,
                       "Cocktail_party_whole_head_master_data",
                       "derivatives", "processed_data",
                       f"trial_snr_{RUN_TYPE}")

# Classification results directory (for accuracy data)
CLASSIFIER_DIR = os.path.join(PROJECT_ROOT,
                              "Classifier_script_results", "nested",
                              "rf_snr_0_20feat_balanced_depth5_oob")

# Accuracy CSV
ACC_CSV = os.path.join(CLASSIFIER_DIR, "final_table.csv")
ACC_COL = "Overt_perc"

# ROI mapping
ROI_MAP_CSV = os.path.join(PROJECT_ROOT, "ROIs", "roi_master.csv")

# Output directory
OUT_DIR = CLASSIFIER_DIR
os.makedirs(OUT_DIR, exist_ok=True)

print(f"SNR directory: {SNR_DIR}")
print(f"Accuracy CSV: {ACC_CSV}")
print(f"Output directory: {OUT_DIR}")

#%%
# ------------------------------------------------------------------ LOAD ROI MAPPING
roi_df = pd.read_csv(ROI_MAP_CSV)
print(f"\nLoaded ROI mapping from {ROI_MAP_CSV}, columns: {list(roi_df.columns)}")

def get_channel_labels_for_roi(roi_name: str, mapping_df: pd.DataFrame) -> list:
    """Get channel labels (e.g., S10D87) for a given ROI from roi_master.csv."""
    mask = mapping_df['brodmann'] == roi_name
    labels = mapping_df.loc[mask, 'channel_label'].to_list()
    if not labels:
        raise RuntimeError(f'ROI string "{roi_name}" not found in brodmann column.')
    return labels

# Get Angular Gyrus channel labels (Left + Right pooled)
left_labels = get_channel_labels_for_roi("Left-AngGyrus (39)", roi_df)
right_labels = get_channel_labels_for_roi("Right-AngGyrus (39)", roi_df)
ag_labels = sorted(set(left_labels + right_labels))
ag_labels_set = set(ag_labels)
print(f"Angular Gyrus channels: {len(ag_labels)} total (Left: {len(left_labels)}, Right: {len(right_labels)})")
print(f"AG channels: {ag_labels[:5]}... (showing first 5)")

#%%
# ------------------------------------------------------------------ LOAD ACCURACY DATA
acc_df = pd.read_csv(ACC_CSV, sep=None, engine='python')
if 'Subject' not in acc_df.columns:
    raise RuntimeError("Expected a 'Subject' column in accuracy CSV.")
if ACC_COL not in acc_df.columns:
    raise RuntimeError(f"Accuracy column '{ACC_COL}' not found; available: {list(acc_df.columns)}")

# Convert subject IDs to integers
acc_df['Subject'] = acc_df['Subject'].astype(int)
acc_series_full = acc_df.set_index('Subject')[ACC_COL]
print(f"\nLoaded accuracy data for {len(acc_series_full)} subjects")
print(f"Accuracy range: {acc_series_full.min():.1f}% to {acc_series_full.max():.1f}%")

#%%
# ------------------------------------------------------------------ LOAD SNR RESULTS
ag_snr_mean = {}
ag_snr_std = {}
ag_abs_mean = {}
ag_abs_mean_std = {}
ag_std_mean = {}
ag_std_std = {}
all_ch_snr_mean = {}
all_ch_snr_std = {}
all_ch_abs_mean = {}
all_ch_abs_mean_std = {}
all_ch_std_mean = {}
all_ch_std_std = {}
n_ag_channels = {}
n_all_channels = {}

for subj_id in SUBJECTS:
    subj_int = int(subj_id)
    
    # Path to subject's SNR CSV
    subj_folder = f"sub_{subj_id}"
    snr_csv = os.path.join(SNR_DIR, subj_folder, f"snr_results_{RUN_TYPE}.csv")
    
    if not os.path.exists(snr_csv):
        print(f"Warning: SNR file not found for sub-{subj_id}: {snr_csv}")
        continue
    
    # Load SNR data
    snr_df = pd.read_csv(snr_csv)
    
    # Check if required columns exist
    required_cols = ['channel', 'snr_avg', 'abs_mean_avg', 'std_avg']
    if not all(col in snr_df.columns for col in required_cols):
        print(f"Warning: Missing required columns in SNR CSV for sub-{subj_id}")
        print(f"  Available columns: {list(snr_df.columns)}")
        continue
    
    # Filter to Angular Gyrus channels
    ag_mask = snr_df['channel'].isin(ag_labels_set)
    ag_snr_values = snr_df.loc[ag_mask, 'snr_avg'].values
    ag_abs_mean_values = snr_df.loc[ag_mask, 'abs_mean_avg'].values
    ag_std_values = snr_df.loc[ag_mask, 'std_avg'].values
    
    # All channels
    all_snr_values = snr_df['snr_avg'].values
    all_abs_mean_values = snr_df['abs_mean_avg'].values
    all_std_values = snr_df['std_avg'].values
    
    # Process Angular Gyrus channels
    if len(ag_snr_values) > 0:
        # Remove any inf/nan values before computing mean
        ag_snr_clean = ag_snr_values[np.isfinite(ag_snr_values)]
        ag_abs_mean_clean = ag_abs_mean_values[np.isfinite(ag_abs_mean_values)]
        ag_std_clean = ag_std_values[np.isfinite(ag_std_values)]
        
        if len(ag_snr_clean) > 0:
            ag_snr_mean[subj_int] = float(np.mean(ag_snr_clean))
            ag_snr_std[subj_int] = float(np.std(ag_snr_clean, ddof=1)) if len(ag_snr_clean) > 1 else 0.0
            n_ag_channels[subj_int] = len(ag_snr_clean)
            print(f"sub-{subj_id}: AG SNR = {ag_snr_mean[subj_int]:.3f} ± {ag_snr_std[subj_int]:.3f} ({n_ag_channels[subj_int]} AG channels)")
        
        if len(ag_abs_mean_clean) > 0:
            ag_abs_mean[subj_int] = float(np.mean(ag_abs_mean_clean))
            ag_abs_mean_std[subj_int] = float(np.std(ag_abs_mean_clean, ddof=1)) if len(ag_abs_mean_clean) > 1 else 0.0
        
        if len(ag_std_clean) > 0:
            ag_std_mean[subj_int] = float(np.mean(ag_std_clean))
            ag_std_std[subj_int] = float(np.std(ag_std_clean, ddof=1)) if len(ag_std_clean) > 1 else 0.0
        
        if len(ag_snr_clean) == 0:
            print(f"Warning: All AG SNR values are inf/nan for sub-{subj_id}")
    else:
        print(f"Warning: No Angular Gyrus channels found in SNR data for sub-{subj_id}")
    
    # Process all channels
    if len(all_snr_values) > 0:
        all_snr_clean = all_snr_values[np.isfinite(all_snr_values)]
        all_abs_mean_clean = all_abs_mean_values[np.isfinite(all_abs_mean_values)]
        all_std_clean = all_std_values[np.isfinite(all_std_values)]
        
        if len(all_snr_clean) > 0:
            all_ch_snr_mean[subj_int] = float(np.mean(all_snr_clean))
            all_ch_snr_std[subj_int] = float(np.std(all_snr_clean, ddof=1)) if len(all_snr_clean) > 1 else 0.0
            n_all_channels[subj_int] = len(all_snr_clean)
        
        if len(all_abs_mean_clean) > 0:
            all_ch_abs_mean[subj_int] = float(np.mean(all_abs_mean_clean))
            all_ch_abs_mean_std[subj_int] = float(np.std(all_abs_mean_clean, ddof=1)) if len(all_abs_mean_clean) > 1 else 0.0
        
        if len(all_std_clean) > 0:
            all_ch_std_mean[subj_int] = float(np.mean(all_std_clean))
            all_ch_std_std[subj_int] = float(np.std(all_std_clean, ddof=1)) if len(all_std_clean) > 1 else 0.0

print(f"\nSuccessfully loaded SNR data for {len(ag_snr_mean)} subjects")

#%%
# ------------------------------------------------------------------ BUILD ANALYSIS DATAFRAME
ag_snr_series = pd.Series(ag_snr_mean, name='ag_snr_mean')
ag_snr_std_series = pd.Series(ag_snr_std, name='ag_snr_std')
ag_abs_mean_series = pd.Series(ag_abs_mean, name='ag_abs_mean')
ag_abs_mean_std_series = pd.Series(ag_abs_mean_std, name='ag_abs_mean_std')
ag_std_series = pd.Series(ag_std_mean, name='ag_std_mean')
ag_std_std_series = pd.Series(ag_std_std, name='ag_std_std')

all_ch_snr_series = pd.Series(all_ch_snr_mean, name='all_ch_snr_mean')
all_ch_snr_std_series = pd.Series(all_ch_snr_std, name='all_ch_snr_std')
all_ch_abs_mean_series = pd.Series(all_ch_abs_mean, name='all_ch_abs_mean')
all_ch_abs_mean_std_series = pd.Series(all_ch_abs_mean_std, name='all_ch_abs_mean_std')
all_ch_std_series = pd.Series(all_ch_std_mean, name='all_ch_std_mean')
all_ch_std_std_series = pd.Series(all_ch_std_std, name='all_ch_std_std')

n_ag_channels_series = pd.Series(n_ag_channels, name='n_ag_channels')
n_all_channels_series = pd.Series(n_all_channels, name='n_all_channels')

# Combined dataframe
analysis_df_full = pd.concat([
    ag_snr_series,
    ag_snr_std_series,
    ag_abs_mean_series,
    ag_abs_mean_std_series,
    ag_std_series,
    ag_std_std_series,
    all_ch_snr_series,
    all_ch_snr_std_series,
    all_ch_abs_mean_series,
    all_ch_abs_mean_std_series,
    all_ch_std_series,
    all_ch_std_std_series,
    n_ag_channels_series,
    n_all_channels_series,
    acc_series_full.rename('accuracy')
], axis=1)

# Separate dataframes for correlation (drop NaN)
analysis_df_ag_snr = analysis_df_full.dropna(subset=['ag_snr_mean', 'accuracy'])
analysis_df_ag_abs = analysis_df_full.dropna(subset=['ag_abs_mean', 'accuracy'])
analysis_df_ag_std = analysis_df_full.dropna(subset=['ag_std_mean', 'accuracy'])

analysis_df_all_snr = analysis_df_full.dropna(subset=['all_ch_snr_mean', 'accuracy'])
analysis_df_all_abs = analysis_df_full.dropna(subset=['all_ch_abs_mean', 'accuracy'])
analysis_df_all_std = analysis_df_full.dropna(subset=['all_ch_std_mean', 'accuracy'])

print(f"\nFinal analysis (AG SNR): {len(analysis_df_ag_snr)} subjects with both AG SNR and accuracy data")
print(f"Final analysis (AG Abs Mean): {len(analysis_df_ag_abs)} subjects with both AG abs_mean and accuracy data")
print(f"Final analysis (AG Std): {len(analysis_df_ag_std)} subjects with both AG std and accuracy data")
print(f"Final analysis (All channels SNR): {len(analysis_df_all_snr)} subjects with both SNR and accuracy data")
print(f"Final analysis (All channels Abs Mean): {len(analysis_df_all_abs)} subjects with both abs_mean and accuracy data")
print(f"Final analysis (All channels Std): {len(analysis_df_all_std)} subjects with both std and accuracy data")

#%%
# ------------------------------------------------------------------ CORRELATION & PLOTS
# Create plots for Angular Gyrus: SNR, Abs Mean, and Std vs Accuracy

sns.set(style='whitegrid')

# ANGULAR GYRUS: 3 METRICS
fig_ag, axes_ag = plt.subplots(1, 3, figsize=(18, 5))

# Plot 1: AG SNR vs Accuracy
ax1 = axes_ag[0]
if len(analysis_df_ag_snr) > 1:
    r_ag_snr, p_ag_snr = pearsonr(analysis_df_ag_snr['ag_snr_mean'], analysis_df_ag_snr['accuracy'])
    print(f"\nPearson correlation (AG SNR vs Accuracy): r = {r_ag_snr:.3f}, p = {p_ag_snr:.4f}")
else:
    r_ag_snr = p_ag_snr = np.nan
    print("\nInsufficient data for correlation (AG SNR)")

ax1.scatter(analysis_df_ag_snr['ag_snr_mean'], analysis_df_ag_snr['accuracy'], 
           c='black', s=100, edgecolors='k', linewidth=0.7, alpha=0.7)
for sid, row in analysis_df_ag_snr.iterrows():
    ax1.annotate(f"{int(sid):02d}", (row['ag_snr_mean'], row['accuracy']), 
                xytext=(4, 4), textcoords='offset points', fontsize=8)
ax1.set_xlabel('Mean Angular Gyrus Activation Reliability (|μ|/σ)', fontsize=10)
ax1.set_ylabel('Overt Accuracy (%)', fontsize=10)
title1 = 'AG: Activation Reliability vs Accuracy'
if not np.isnan(r_ag_snr):
    title1 += f'\n(r = {r_ag_snr:.3f}, p = {p_ag_snr:.4f})'
ax1.set_title(title1, fontsize=11, fontweight='bold')
ax1.grid(True, alpha=0.3)

# Plot 2: AG Absolute Mean Amplitude vs Accuracy
ax2 = axes_ag[1]
if len(analysis_df_ag_abs) > 1:
    r_ag_abs, p_ag_abs = pearsonr(analysis_df_ag_abs['ag_abs_mean'], analysis_df_ag_abs['accuracy'])
    print(f"Pearson correlation (AG Abs Mean vs Accuracy): r = {r_ag_abs:.3f}, p = {p_ag_abs:.4f}")
else:
    r_ag_abs = p_ag_abs = np.nan
    print("Insufficient data for correlation (AG Abs Mean)")

ax2.scatter(analysis_df_ag_abs['ag_abs_mean'], analysis_df_ag_abs['accuracy'], 
           c='black', s=100, edgecolors='k', linewidth=0.7, alpha=0.7)
for sid, row in analysis_df_ag_abs.iterrows():
    ax2.annotate(f"{int(sid):02d}", (row['ag_abs_mean'], row['accuracy']), 
                xytext=(4, 4), textcoords='offset points', fontsize=8)
ax2.set_xlabel('Mean Angular Gyrus |Amplitude| (μM)', fontsize=10)
ax2.set_ylabel('Overt Accuracy (%)', fontsize=10)
title2 = 'AG: Amplitude vs Accuracy'
if not np.isnan(r_ag_abs):
    title2 += f'\n(r = {r_ag_abs:.3f}, p = {p_ag_abs:.4f})'
ax2.set_title(title2, fontsize=11, fontweight='bold')
ax2.grid(True, alpha=0.3)

# Plot 3: AG Std (Variability) vs Accuracy
ax3 = axes_ag[2]
if len(analysis_df_ag_std) > 1:
    r_ag_std, p_ag_std = pearsonr(analysis_df_ag_std['ag_std_mean'], analysis_df_ag_std['accuracy'])
    print(f"Pearson correlation (AG Std vs Accuracy): r = {r_ag_std:.3f}, p = {p_ag_std:.4f}")
else:
    r_ag_std = p_ag_std = np.nan
    print("Insufficient data for correlation (AG Std)")

ax3.scatter(analysis_df_ag_std['ag_std_mean'], analysis_df_ag_std['accuracy'], 
           c='black', s=100, edgecolors='k', linewidth=0.7, alpha=0.7)
for sid, row in analysis_df_ag_std.iterrows():
    ax3.annotate(f"{int(sid):02d}", (row['ag_std_mean'], row['accuracy']), 
                xytext=(4, 4), textcoords='offset points', fontsize=8)
ax3.set_xlabel('Mean Angular Gyrus Std (σ)', fontsize=10)
ax3.set_ylabel('Overt Accuracy (%)', fontsize=10)
title3 = 'AG: Variability vs Accuracy'
if not np.isnan(r_ag_std):
    title3 += f'\n(r = {r_ag_std:.3f}, p = {p_ag_std:.4f})'
ax3.set_title(title3, fontsize=11, fontweight='bold')
ax3.grid(True, alpha=0.3)

plt.suptitle(f'Angular Gyrus Metrics vs Overt Accuracy ({RUN_TYPE.capitalize()})', 
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()

plot_ag_path = os.path.join(OUT_DIR, f'angular_gyrus_activation_reliability_vs_accuracy_{RUN_TYPE}.png')
plt.savefig(plot_ag_path, dpi=300, bbox_inches='tight')
plt.close(fig_ag)
print(f'\nSaved Angular Gyrus plot: {plot_ag_path}')

# ALL CHANNELS: 3 METRICS
fig_all, axes_all = plt.subplots(1, 3, figsize=(18, 5))

# Plot 1: All Channels SNR vs Accuracy
ax1 = axes_all[0]
if len(analysis_df_all_snr) > 1:
    r_all_snr, p_all_snr = pearsonr(analysis_df_all_snr['all_ch_snr_mean'], analysis_df_all_snr['accuracy'])
    print(f"\nPearson correlation (All-Ch SNR vs Accuracy): r = {r_all_snr:.3f}, p = {p_all_snr:.4f}")
else:
    r_all_snr = p_all_snr = np.nan
    print("\nInsufficient data for correlation (All-Ch SNR)")

ax1.scatter(analysis_df_all_snr['all_ch_snr_mean'], analysis_df_all_snr['accuracy'], 
           c='black', s=100, edgecolors='k', linewidth=0.7, alpha=0.7)
for sid, row in analysis_df_all_snr.iterrows():
    ax1.annotate(f"{int(sid):02d}", (row['all_ch_snr_mean'], row['accuracy']), 
                xytext=(4, 4), textcoords='offset points', fontsize=8)
ax1.set_xlabel('Mean All-Channels Activation Reliability (|μ|/σ)', fontsize=10)
ax1.set_ylabel('Overt Accuracy (%)', fontsize=10)
title1 = 'All Channels: Activation Reliability vs Accuracy'
if not np.isnan(r_all_snr):
    title1 += f'\n(r = {r_all_snr:.3f}, p = {p_all_snr:.4f})'
ax1.set_title(title1, fontsize=11, fontweight='bold')
ax1.grid(True, alpha=0.3)

# Plot 2: All Channels Absolute Mean Amplitude vs Accuracy
ax2 = axes_all[1]
if len(analysis_df_all_abs) > 1:
    r_all_abs, p_all_abs = pearsonr(analysis_df_all_abs['all_ch_abs_mean'], analysis_df_all_abs['accuracy'])
    print(f"Pearson correlation (All-Ch Abs Mean vs Accuracy): r = {r_all_abs:.3f}, p = {p_all_abs:.4f}")
else:
    r_all_abs = p_all_abs = np.nan
    print("Insufficient data for correlation (All-Ch Abs Mean)")

ax2.scatter(analysis_df_all_abs['all_ch_abs_mean'], analysis_df_all_abs['accuracy'], 
           c='black', s=100, edgecolors='k', linewidth=0.7, alpha=0.7)
for sid, row in analysis_df_all_abs.iterrows():
    ax2.annotate(f"{int(sid):02d}", (row['all_ch_abs_mean'], row['accuracy']), 
                xytext=(4, 4), textcoords='offset points', fontsize=8)
ax2.set_xlabel('Mean All-Channels |Amplitude| (μM)', fontsize=10)
ax2.set_ylabel('Overt Accuracy (%)', fontsize=10)
title2 = 'All Channels: Amplitude vs Accuracy'
if not np.isnan(r_all_abs):
    title2 += f'\n(r = {r_all_abs:.3f}, p = {p_all_abs:.4f})'
ax2.set_title(title2, fontsize=11, fontweight='bold')
ax2.grid(True, alpha=0.3)

# Plot 3: All Channels Std (Variability) vs Accuracy
ax3 = axes_all[2]
if len(analysis_df_all_std) > 1:
    r_all_std, p_all_std = pearsonr(analysis_df_all_std['all_ch_std_mean'], analysis_df_all_std['accuracy'])
    print(f"Pearson correlation (All-Ch Std vs Accuracy): r = {r_all_std:.3f}, p = {p_all_std:.4f}")
else:
    r_all_std = p_all_std = np.nan
    print("Insufficient data for correlation (All-Ch Std)")

ax3.scatter(analysis_df_all_std['all_ch_std_mean'], analysis_df_all_std['accuracy'], 
           c='black', s=100, edgecolors='k', linewidth=0.7, alpha=0.7)
for sid, row in analysis_df_all_std.iterrows():
    ax3.annotate(f"{int(sid):02d}", (row['all_ch_std_mean'], row['accuracy']), 
                xytext=(4, 4), textcoords='offset points', fontsize=8)
ax3.set_xlabel('Mean All-Channels Std (σ)', fontsize=10)
ax3.set_ylabel('Overt Accuracy (%)', fontsize=10)
title3 = 'All Channels: Variability vs Accuracy'
if not np.isnan(r_all_std):
    title3 += f'\n(r = {r_all_std:.3f}, p = {p_all_std:.4f})'
ax3.set_title(title3, fontsize=11, fontweight='bold')
ax3.grid(True, alpha=0.3)

plt.suptitle(f'All Channels Metrics vs Overt Accuracy ({RUN_TYPE.capitalize()})', 
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()

plot_all_path = os.path.join(OUT_DIR, f'all_channels_activation_reliability_vs_accuracy_{RUN_TYPE}.png')
plt.savefig(plot_all_path, dpi=300, bbox_inches='tight')
plt.close(fig_all)
print(f'Saved All Channels plot: {plot_all_path}')

#%%
# ------------------------------------------------------------------ SAVE CSV
csv_path = os.path.join(OUT_DIR, f'angular_gyrus_activation_reliability_metrics_{RUN_TYPE}.csv')
analysis_df_full.to_csv(csv_path, index_label='Subject')
print(f'\nSaved CSV: {csv_path}')

# Summary statistics
print("\n=== Summary Statistics (Angular Gyrus) ===")
if len(analysis_df_ag_snr) > 0:
    print(f"N subjects (SNR): {len(analysis_df_ag_snr)}")
    print(f"AG SNR range: [{analysis_df_ag_snr['ag_snr_mean'].min():.3f}, {analysis_df_ag_snr['ag_snr_mean'].max():.3f}]")
    print(f"AG SNR mean ± std: {analysis_df_ag_snr['ag_snr_mean'].mean():.3f} ± {analysis_df_ag_snr['ag_snr_mean'].std():.3f}")
if len(analysis_df_ag_abs) > 0:
    print(f"AG Abs Mean range: [{analysis_df_ag_abs['ag_abs_mean'].min():.3e}, {analysis_df_ag_abs['ag_abs_mean'].max():.3e}]")
    print(f"AG Abs Mean mean ± std: {analysis_df_ag_abs['ag_abs_mean'].mean():.3e} ± {analysis_df_ag_abs['ag_abs_mean'].std():.3e}")
if len(analysis_df_ag_std) > 0:
    print(f"AG Std range: [{analysis_df_ag_std['ag_std_mean'].min():.3e}, {analysis_df_ag_std['ag_std_mean'].max():.3e}]")
    print(f"AG Std mean ± std: {analysis_df_ag_std['ag_std_mean'].mean():.3e} ± {analysis_df_ag_std['ag_std_mean'].std():.3e}")
if len(analysis_df_ag_snr) > 0:
    print(f"Accuracy range: [{analysis_df_ag_snr['accuracy'].min():.1f}%, {analysis_df_ag_snr['accuracy'].max():.1f}%]")
    print(f"Accuracy mean ± std: {analysis_df_ag_snr['accuracy'].mean():.1f}% ± {analysis_df_ag_snr['accuracy'].std():.1f}%")
    if 'n_ag_channels' in analysis_df_ag_snr.columns:
        print(f"AG channels per subject: {analysis_df_ag_snr['n_ag_channels'].min():.0f} to {analysis_df_ag_snr['n_ag_channels'].max():.0f} (mean: {analysis_df_ag_snr['n_ag_channels'].mean():.1f})")

print("\n=== Summary Statistics (All Channels) ===")
if len(analysis_df_all_snr) > 0:
    print(f"N subjects (SNR): {len(analysis_df_all_snr)}")
    print(f"All-channels SNR range: [{analysis_df_all_snr['all_ch_snr_mean'].min():.3f}, {analysis_df_all_snr['all_ch_snr_mean'].max():.3f}]")
    print(f"All-channels SNR mean ± std: {analysis_df_all_snr['all_ch_snr_mean'].mean():.3f} ± {analysis_df_all_snr['all_ch_snr_mean'].std():.3f}")
if len(analysis_df_all_abs) > 0:
    print(f"All-channels Abs Mean range: [{analysis_df_all_abs['all_ch_abs_mean'].min():.3e}, {analysis_df_all_abs['all_ch_abs_mean'].max():.3e}]")
    print(f"All-channels Abs Mean mean ± std: {analysis_df_all_abs['all_ch_abs_mean'].mean():.3e} ± {analysis_df_all_abs['all_ch_abs_mean'].std():.3e}")
if len(analysis_df_all_std) > 0:
    print(f"All-channels Std range: [{analysis_df_all_std['all_ch_std_mean'].min():.3e}, {analysis_df_all_std['all_ch_std_mean'].max():.3e}]")
    print(f"All-channels Std mean ± std: {analysis_df_all_std['all_ch_std_mean'].mean():.3e} ± {analysis_df_all_std['all_ch_std_mean'].std():.3e}")
if len(analysis_df_all_snr) > 0:
    print(f"Accuracy range: [{analysis_df_all_snr['accuracy'].min():.1f}%, {analysis_df_all_snr['accuracy'].max():.1f}%]")
    print(f"Accuracy mean ± std: {analysis_df_all_snr['accuracy'].mean():.1f}% ± {analysis_df_all_snr['accuracy'].std():.1f}%")
    if 'n_all_channels' in analysis_df_all_snr.columns:
        print(f"Total channels per subject: {analysis_df_all_snr['n_all_channels'].min():.0f} to {analysis_df_all_snr['n_all_channels'].max():.0f} (mean: {analysis_df_all_snr['n_all_channels'].mean():.1f})")

#%%
# ------------------------------------------------------------------ CORRELATION COMPARISON TABLE
print("\n=== Correlation Comparison ===")
comparison_data = {
    'Metric': ['AG SNR', 'AG Abs Mean', 'AG Std', 'All-Ch SNR', 'All-Ch Abs Mean', 'All-Ch Std'],
    'Pearson r': [
        r_ag_snr if not np.isnan(r_ag_snr) else np.nan,
        r_ag_abs if not np.isnan(r_ag_abs) else np.nan,
        r_ag_std if not np.isnan(r_ag_std) else np.nan,
        r_all_snr if not np.isnan(r_all_snr) else np.nan,
        r_all_abs if not np.isnan(r_all_abs) else np.nan,
        r_all_std if not np.isnan(r_all_std) else np.nan,
    ],
    'p-value': [
        p_ag_snr if not np.isnan(p_ag_snr) else np.nan,
        p_ag_abs if not np.isnan(p_ag_abs) else np.nan,
        p_ag_std if not np.isnan(p_ag_std) else np.nan,
        p_all_snr if not np.isnan(p_all_snr) else np.nan,
        p_all_abs if not np.isnan(p_all_abs) else np.nan,
        p_all_std if not np.isnan(p_all_std) else np.nan,
    ],
    'N subjects': [
        len(analysis_df_ag_snr), 
        len(analysis_df_ag_abs),
        len(analysis_df_ag_std),
        len(analysis_df_all_snr), 
        len(analysis_df_all_abs),
        len(analysis_df_all_std)
    ]
}
comparison_df = pd.DataFrame(comparison_data)
print(comparison_df.to_string(index=False))

# Save comparison table
comparison_path = os.path.join(OUT_DIR, f'activation_reliability_correlation_comparison_{RUN_TYPE}.csv')
comparison_df.to_csv(comparison_path, index=False)
print(f"\nSaved correlation comparison: {comparison_path}")

print("\n=== Analysis Complete ===")

# %% ------------------------------------------------------------------ PUBLICATION FIGURE
# Two-panel scatter: Left = All-Channels Activation Reliability vs Accuracy
#                   Right = AG Activation Reliability vs Accuracy
# Follows publication style from table_maker_scatter_overt_only_pub_latency_CI.py

_PUB_BASE_FONT   = 16
_PUB_AXIS_LABEL  = 18
_PUB_AXIS_TICK   = 15
_PUB_SUBJ_FONT   = 9
_PUB_PANEL_FONT  = 22
_PUB_ANNOT_FONT  = 14

plt.rcParams.update({
    'font.size': _PUB_BASE_FONT,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial'],
    'axes.linewidth': 1.5,
    'xtick.major.width': 1.5,
    'ytick.major.width': 1.5,
    'xtick.major.size': 6,
    'ytick.major.size': 6,
})

_COLOR_ALL = 'black'
_COLOR_AG  = 'black'

fig_pub, (ax_all, ax_ag) = plt.subplots(2, 1, figsize=(7, 11))

def _draw_reliability_panel(ax, x_vals, y_vals, subj_ids, color, panel_letter,
                             xlabel, r_val, p_val):
    """Draw a single scatter + regression panel in publication style."""
    # Scatter
    ax.scatter(x_vals, y_vals,
               s=120, color=color, edgecolors='black',
               linewidths=1.5, alpha=0.85, zorder=3)

    # Subject ID labels
    for xv, yv, sid in zip(x_vals, y_vals, subj_ids):
        ax.annotate(f"{int(sid):02d}", (xv, yv),
                    xytext=(4, 4), textcoords='offset points',
                    fontsize=_PUB_SUBJ_FONT, color='black')

    # Regression line
    if len(x_vals) > 2:
        m, b = np.polyfit(x_vals, y_vals, 1)
        x_line = np.linspace(x_vals.min(), x_vals.max(), 200)
        ax.plot(x_line, m * x_line + b,
                color=color, linewidth=2, linestyle='--', alpha=0.8, zorder=2)

    # r and p annotation (bottom-right inside axes)
    p_str = f'p = {p_val:.4f}' if p_val >= 0.0001 else 'p < 0.0001'
    ax.text(0.95, 0.07, f'r = {r_val:.3f},  {p_str}',
            transform=ax.transAxes,
            fontsize=_PUB_ANNOT_FONT, fontweight='bold',
            va='bottom', ha='right',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor='gray', alpha=0.85))

    # Axes formatting
    ax.set_xlabel(xlabel, fontsize=_PUB_AXIS_LABEL, fontweight='bold')
    ax.set_ylabel('Overt Accuracy (%)', fontsize=_PUB_AXIS_LABEL, fontweight='bold')
    ax.tick_params(axis='both', labelsize=_PUB_AXIS_TICK)
    ax.set_ylim(45, 105)
    ax.grid(True, alpha=0.25, linestyle=':', linewidth=0.8)
    ax.set_facecolor('#FAFAFA')
    ax.text(-0.12, 1.05, panel_letter, transform=ax.transAxes,
            fontsize=_PUB_PANEL_FONT, fontweight='bold', va='top')

# Panel A: All Channels
_df_a = analysis_df_all_snr.dropna(subset=['all_ch_snr_mean', 'accuracy'])
_draw_reliability_panel(
    ax_all,
    x_vals=_df_a['all_ch_snr_mean'].values,
    y_vals=_df_a['accuracy'].values,
    subj_ids=_df_a.index.values,
    color=_COLOR_ALL,
    panel_letter='A',
    xlabel='All-Channels Activation Reliability (|μ|/σ)',
    r_val=r_all_snr,
    p_val=p_all_snr,
)

# Panel B: Angular Gyrus
_df_b = analysis_df_ag_snr.dropna(subset=['ag_snr_mean', 'accuracy'])
_draw_reliability_panel(
    ax_ag,
    x_vals=_df_b['ag_snr_mean'].values,
    y_vals=_df_b['accuracy'].values,
    subj_ids=_df_b.index.values,
    color=_COLOR_AG,
    panel_letter='B',
    xlabel='Angular Gyrus Activation Reliability (|μ|/σ)',
    r_val=r_ag_snr,
    p_val=p_ag_snr,
)

fig_pub.tight_layout()

for _ext in ('png', 'svg', 'pdf'):
    _out = os.path.join(OUT_DIR, f'snr_vs_reliability_pub.{_ext}')
    _fmt = 'png' if _ext == 'png' else _ext
    _dpi = 300 if _ext == 'png' else None
    fig_pub.savefig(_out, format=_fmt, dpi=_dpi, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
    print(f"Saved: {_out}")

plt.show()

# %% End of file
