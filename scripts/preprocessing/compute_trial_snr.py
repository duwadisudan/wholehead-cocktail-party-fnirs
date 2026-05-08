#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute per-channel trial-to-trial SNR over the 3-8 s post-stimulus window.

For each subject, loads preprocessed HbO data and trial-level HRFs, computes
the trial-mean within the 3-8 s window per channel, and reports SNR as
mean / SD across trials. Left and Right trial conditions are computed
separately and then averaged. Per-subject results are saved as pickle and
CSV for downstream variability and correlation analyses.

Author: Sudan Duwadi <sudan@bu.edu>
Notes: Code refactoring, documentation, and commenting were AI-assisted;
       all scientific decisions and accountability remain with the author.
"""
#%%

import sys
from whichscript import enable_auto_logging

enable_auto_logging()
#%% Imports
import os
import sys
import pickle
import gzip
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from cedalion import units
import warnings

from wholehead_cocktail_party import processing_func as pf
from wholehead_cocktail_party.paths import load_paths, require
from wholehead_cocktail_party.run_config import load_run_config, require_run, resolve_subjects

_PATHS = load_paths()
require(_PATHS, "raw_root", "derivatives_root")

_RUN = load_run_config()
require_run(_RUN, supported_conditions={"overt", "covert"}, supported_modes={"full", "from-derivatives"})

# Cohort of subjects with both overt and covert runs. Override via run.yml.
_DEFAULT_COHORT = ['01','02','03','04','05','10','11','12','13','14','15','18','20','22','25','28','30','31','32','33','34','35','39','41','44','47']

warnings.filterwarnings('ignore')

#%% Configuration

# Run type from config/run.yml. Edit the file, do not edit here.
flag_run_type = _RUN.condition

print(f" Run type: '{flag_run_type}'")

# Determine file IDs based on run type
if flag_run_type.lower() == 'overt':
    selected_file_ids = ['overt_run-01', 'overt_run-02']
    stim_labels = ['Overt Left', 'Overt Right']
elif flag_run_type.lower() == 'covert':
    selected_file_ids = ['covert_run-01', 'covert_run-02']
    stim_labels = ['Covert Left', 'Covert Right']
else:
    raise ValueError(f"flag_run_type must be 'overt' or 'covert', got {flag_run_type!r}")

# Dataset configuration
cfg_dataset = {
    'root_dir': str(_PATHS.raw_root),
    'subj_ids' : resolve_subjects(_RUN, _DEFAULT_COHORT),
    'file_ids': selected_file_ids,
    'subj_id_exclude': [],
}

# Directories
rootDir_saveData = str(_PATHS.derivatives_root) + os.sep
preprocessed_dir = os.path.join(rootDir_saveData, f"preprocessed_{flag_run_type}_snr_0")

# SNR calculation parameters
SNR_CONFIG = {
    'time_window': [3, 8],  # seconds - window for SNR calculation
    'chromophore': 'HbO',   # only HbO
    't_pre': 2,             # seconds before stimulus onset (for block averaging)
    't_post': 15,           # seconds after stimulus onset
}

# Output directory
output_dir = os.path.join(rootDir_saveData, f"trial_snr_{flag_run_type}")
os.makedirs(output_dir, exist_ok=True)
print(f" Output directory: {output_dir}")

#%% Helper Functions

def load_subject_data(subj_id, preprocessed_dir):
    """Load preprocessed data for a single subject (handles both gzipped and non-gzipped files)."""
    rec_file = os.path.join(preprocessed_dir, f"rec_subj_{subj_id}.pkl")
    prune_file = os.path.join(preprocessed_dir, f"chs_pruned_subj_{subj_id}.pkl")
    
    if not os.path.exists(rec_file):
        print(f"   Missing rec file: {rec_file}")
        return None, None
    if not os.path.exists(prune_file):
        print(f"   Missing prune file: {prune_file}")
        return None, None
    
    # Try to load as gzipped first, fall back to regular pickle if not gzipped
    try:
        with gzip.open(rec_file, 'rb') as f:
            rec_runs = pickle.load(f)
    except (gzip.BadGzipFile, OSError):
        # Not gzipped, load as regular pickle
        with open(rec_file, 'rb') as f:
            rec_runs = pickle.load(f)
    
    try:
        with gzip.open(prune_file, 'rb') as f:
            chs_pruned = pickle.load(f)
    except (gzip.BadGzipFile, OSError):
        # Not gzipped, load as regular pickle
        with open(prune_file, 'rb') as f:
            chs_pruned = pickle.load(f)
    
    return rec_runs, chs_pruned


def extract_trial_level_hrfs(rec_runs, stim_labels, t_pre, t_post, ts_name='conc_p_tddr_filt'):
    """
    Extract trial-level HRFs for each run - EXACT copy of processing_func.py block_average logic
    
    Returns:
    --------
    trial_hrfs : list of xr.DataArray
        Each element is (epoch, channel, chromo, reltime) for one run
    """
    trial_hrfs_all_runs = []
    
    for run_idx, rec in enumerate(rec_runs):
        if rec is None:
            print(f"    Run {run_idx+1}: No data available")
            continue
        
        # EXACT code from block_average function
        try:
            ts = rec[ts_name].copy()
            stim = rec.stim.copy()
        except (KeyError, AttributeError):
            print(f"    Run {run_idx+1}: Missing '{ts_name}' timeseries")
            continue
        
        # Use cedalion's to_epochs method - EXACT same as block_average
        try:
            epochs = ts.cd.to_epochs(
                stim,  # stimulus dataframe
                stim_labels,  # select events
                before=t_pre,  # seconds before stimulus
                after=t_post,  # seconds after stimulus
            )
        except Exception as e:
            print(f"    Run {run_idx+1}: Failed to extract epochs: {e}")
            continue
        
        if len(epochs.epoch) == 0:
            print(f"    Run {run_idx+1}: No matching trials found")
            continue
        
        print(f"    Run {run_idx+1}: Extracted {len(epochs.epoch)} trials")
        trial_hrfs_all_runs.append(epochs)
    
    return trial_hrfs_all_runs


def calculate_snr_per_channel(trial_hrfs_all_runs, time_window, chromophore='HbO'):
    """
    Calculate SNR per channel based on trial-to-trial variability.
    
    Parameters:
    -----------
    trial_hrfs_all_runs : list of xr.DataArray
        Each element is (epoch, channel, chromo, reltime) for one run
    time_window : list [start, end]
        Time window in seconds for SNR calculation
    chromophore : str
        'HbO' or 'HbR'
    
    Returns:
    --------
    snr_results : dict
        Keys: 'left', 'right', 'avg', 'channels'
        Values: arrays of SNR per channel
    trial_avgs : dict
        Intermediate results for debugging
    """
    # Combine all runs - epochs dimension from cedalion's to_epochs
    all_trials = xr.concat(trial_hrfs_all_runs, dim='epoch')
    
    # Select chromophore
    all_trials = all_trials.sel(chromo=chromophore)
    
    # Get channels
    channels = all_trials.channel.values
    
    # Separate by trial type
    left_trials = all_trials[all_trials.trial_type.str.contains('Left')]
    right_trials = all_trials[all_trials.trial_type.str.contains('Right')]
    
    print(f"    Found {len(left_trials.epoch)} Left trials and {len(right_trials.epoch)} Right trials")
    
    # Function to calculate SNR for one condition
    def calc_snr(trials_da, condition_name):
        """Calculate SNR: abs(mean)/std across trials after averaging each trial in time window."""
        # Select time window
        trials_windowed = trials_da.sel(reltime=slice(time_window[0], time_window[1]))
        
        # Average each trial within the time window -> (epoch, channel)
        trial_avgs = trials_windowed.mean(dim='reltime')
        
        # Calculate mean and std across trials -> (channel,)
        mean_per_channel = trial_avgs.mean(dim='epoch')
        std_per_channel = trial_avgs.std(dim='epoch')
        
        # Absolute mean (amplitude metric)
        abs_mean_per_channel = np.abs(mean_per_channel)
        
        # SNR = abs(mean) / std  (captures consistency regardless of activation/deactivation)
        snr_per_channel = abs_mean_per_channel / std_per_channel
        
        # Replace inf/nan with 0
        snr_per_channel = xr.where(np.isfinite(snr_per_channel), snr_per_channel, 0)
        abs_mean_per_channel = xr.where(np.isfinite(abs_mean_per_channel), abs_mean_per_channel, 0)
        std_per_channel = xr.where(np.isfinite(std_per_channel), std_per_channel, 0)
        
        print(f"      {condition_name}: SNR range = {snr_per_channel.min().values:.3f} to {snr_per_channel.max().values:.3f}")
        
        return snr_per_channel.values, abs_mean_per_channel.values, std_per_channel.values, trial_avgs.values
    
    # Calculate SNR for left and right
    snr_left, abs_mean_left, std_left, trial_avgs_left = calc_snr(left_trials, 'Left')
    snr_right, abs_mean_right, std_right, trial_avgs_right = calc_snr(right_trials, 'Right')
    
    # Average left and right SNR
    snr_avg = (snr_left + snr_right) / 2
    abs_mean_avg = (abs_mean_left + abs_mean_right) / 2
    std_avg = (std_left + std_right) / 2
    
    snr_results = {
        'left': snr_left,
        'right': snr_right,
        'avg': snr_avg,
        'abs_mean_left': abs_mean_left,
        'abs_mean_right': abs_mean_right,
        'abs_mean_avg': abs_mean_avg,
        'std_left': std_left,
        'std_right': std_right,
        'std_avg': std_avg,
        'channels': channels,
    }
    
    trial_avgs = {
        'left': trial_avgs_left,
        'right': trial_avgs_right,
    }
    
    return snr_results, trial_avgs


def save_subject_snr(subj_id, snr_results, trial_avgs, output_dir, flag_run_type):
    """Save SNR results for one subject as pickle and CSV."""
    
    # Create subject-specific output directory
    subj_dir = os.path.join(output_dir, f"sub_{subj_id}")
    os.makedirs(subj_dir, exist_ok=True)
    
    # Save as pickle (complete data)
    pkl_file = os.path.join(subj_dir, f"snr_results_{flag_run_type}.pkl.gz")
    save_dict = {
        'snr_results': snr_results,
        'trial_avgs': trial_avgs,
        'config': SNR_CONFIG,
        'subject_id': subj_id,
        'run_type': flag_run_type,
    }
    with gzip.open(pkl_file, 'wb') as f:
        pickle.dump(save_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"   Saved pickle: {pkl_file}")
    
    # Save as CSV (just SNR values)
    csv_file = os.path.join(subj_dir, f"snr_results_{flag_run_type}.csv")
    df = pd.DataFrame({
        'channel': snr_results['channels'],
        'snr_left': snr_results['left'],
        'snr_right': snr_results['right'],
        'snr_avg': snr_results['avg'],
        'abs_mean_left': snr_results['abs_mean_left'],
        'abs_mean_right': snr_results['abs_mean_right'],
        'abs_mean_avg': snr_results['abs_mean_avg'],
        'std_left': snr_results['std_left'],
        'std_right': snr_results['std_right'],
        'std_avg': snr_results['std_avg'],
    })
    df.to_csv(csv_file, index=False)
    print(f"   Saved CSV: {csv_file}")
    
    # Create summary plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Plot Left SNR
    axes[0].bar(range(len(snr_results['left'])), snr_results['left'])
    axes[0].set_title(f"SNR - Left ({flag_run_type})")
    axes[0].set_xlabel("Channel")
    axes[0].set_ylabel("SNR")
    axes[0].grid(True, alpha=0.3)
    
    # Plot Right SNR
    axes[1].bar(range(len(snr_results['right'])), snr_results['right'])
    axes[1].set_title(f"SNR - Right ({flag_run_type})")
    axes[1].set_xlabel("Channel")
    axes[1].set_ylabel("SNR")
    axes[1].grid(True, alpha=0.3)
    
    # Plot Average SNR
    axes[2].bar(range(len(snr_results['avg'])), snr_results['avg'])
    axes[2].set_title(f"SNR - Average ({flag_run_type})")
    axes[2].set_xlabel("Channel")
    axes[2].set_ylabel("SNR")
    axes[2].grid(True, alpha=0.3)
    
    plt.suptitle(f"Subject {subj_id} - Trial-to-Trial SNR (HbO, 3-8s window)")
    plt.tight_layout()
    
    fig_file = os.path.join(subj_dir, f"snr_plot_{flag_run_type}.png")
    plt.savefig(fig_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   Saved plot: {fig_file}")


#%% Main Processing Loop

print(f"\n{'='*70}")
print(f"Starting SNR calculation for {len(cfg_dataset['subj_ids'])} subjects")
print(f"Run type: {flag_run_type}")
print(f"Time window: {SNR_CONFIG['time_window']} seconds")
print(f"Chromophore: {SNR_CONFIG['chromophore']}")
print(f"{'='*70}\n")

# Storage for all subjects
all_subjects_snr = []

for subj_idx, subj_id in enumerate(cfg_dataset['subj_ids']):
    print(f"\n[{subj_idx+1}/{len(cfg_dataset['subj_ids'])}] Processing subject {subj_id}...")
    
    # 1. Load subject data
    rec_runs, chs_pruned = load_subject_data(subj_id, preprocessed_dir)
    
    if rec_runs is None:
        print(f"    Skipping subject {subj_id} - no data found")
        continue
    
    print(f"   Loaded {len(rec_runs)} runs")
    
    # 2. Extract trial-level HRFs
    print(f"  Extracting trial-level HRFs...")
    trial_hrfs = extract_trial_level_hrfs(
        rec_runs, 
        stim_labels,
        t_pre=SNR_CONFIG['t_pre'] * units.s,  # MUST have units!
        t_post=SNR_CONFIG['t_post'] * units.s,  # MUST have units!
        ts_name='conc_p_tddr_filt'
    )
    
    if len(trial_hrfs) == 0:
        print(f"    Skipping subject {subj_id} - no trials extracted")
        continue
    
    # 3. Calculate SNR per channel
    print(f"  Calculating SNR per channel...")
    snr_results, trial_avgs = calculate_snr_per_channel(
        trial_hrfs,
        time_window=SNR_CONFIG['time_window'],
        chromophore=SNR_CONFIG['chromophore']
    )
    
    # 4. Save results
    print(f"  Saving results...")
    save_subject_snr(subj_id, snr_results, trial_avgs, output_dir, flag_run_type)
    
    # Store for summary
    all_subjects_snr.append({
        'subject_id': subj_id,
        'snr_left_mean': np.mean(snr_results['left']),
        'snr_left_std': np.std(snr_results['left']),
        'snr_right_mean': np.mean(snr_results['right']),
        'snr_right_std': np.std(snr_results['right']),
        'snr_avg_mean': np.mean(snr_results['avg']),
        'snr_avg_std': np.std(snr_results['avg']),
        'n_channels': len(snr_results['channels']),
    })
    
    print(f"   Subject {subj_id} complete!")

#%% Save Summary Across All Subjects

print(f"\n{'='*70}")
print("Creating summary across all subjects...")
print(f"{'='*70}\n")

if len(all_subjects_snr) > 0:
    # Save summary CSV
    summary_df = pd.DataFrame(all_subjects_snr)
    summary_file = os.path.join(output_dir, f"snr_summary_all_subjects_{flag_run_type}.csv")
    summary_df.to_csv(summary_file, index=False)
    print(f" Saved summary: {summary_file}")
    
    # Print summary statistics
    print("\n Summary Statistics:")
    print(f"  Total subjects processed: {len(all_subjects_snr)}")
    print(f"  Average SNR (Left):   {summary_df['snr_left_mean'].mean():.3f} ± {summary_df['snr_left_mean'].std():.3f}")
    print(f"  Average SNR (Right):  {summary_df['snr_right_mean'].mean():.3f} ± {summary_df['snr_right_mean'].std():.3f}")
    print(f"  Average SNR (Avg):    {summary_df['snr_avg_mean'].mean():.3f} ± {summary_df['snr_avg_mean'].std():.3f}")
    
    # Create summary plot
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(all_subjects_snr))
    width = 0.25
    
    ax.bar(x - width, summary_df['snr_left_mean'], width, label='Left', alpha=0.8)
    ax.bar(x, summary_df['snr_right_mean'], width, label='Right', alpha=0.8)
    ax.bar(x + width, summary_df['snr_avg_mean'], width, label='Average', alpha=0.8)
    
    ax.set_xlabel('Subject')
    ax.set_ylabel('Mean SNR')
    ax.set_title(f'Mean SNR Across Subjects ({flag_run_type}, HbO, 3-8s window)')
    ax.set_xticks(x)
    ax.set_xticklabels(summary_df['subject_id'], rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    summary_plot_file = os.path.join(output_dir, f"snr_summary_plot_{flag_run_type}.png")
    plt.savefig(summary_plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" Saved summary plot: {summary_plot_file}")
else:
    print("  No subjects were successfully processed!")

print(f"\n{'='*70}")
print(" SNR calculation complete!")
print(f"Results saved to: {output_dir}")
print(f"{'='*70}\n")

#%%
