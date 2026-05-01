"""
Group-level Brodmann ROI hemodynamic response — generic (supplementary).

Loads per-subject preprocessed HbO/HbR data and renders the per-ROI time
courses (group mean ± SEM) without condition restriction. Reference
implementation that the figure-specific variants are derived from.

Author: Sudan Duwadi <sudan@bu.edu>
Notes: Code refactoring was AI-assisted; all scientific decisions and
       accountability remain with the author.
"""
#%%
import os
import cedalion
import cedalion.nirs
import xarray as xr
from cedalion import units
import gzip
import pickle
import pdb 
import numpy as np 
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from scipy.stats import ttest_rel
from statsmodels.stats.multitest import multipletests

import sys
from wholehead_cocktail_party import processing_func as pf

import warnings
warnings.filterwarnings('ignore')

#%%

import importlib
importlib.reload(pf)


# %% Initial root directory and analysis parameters
##############################################################################

flag_load_preprocessed_data = True  # if 1, will skip load_and_preprocess function and use saved data
flag_load_preprocessed_control_data = True
flag_save_control_preprocessed_data = False # if 1, will skip load_and_preprocess function and use saved data
rootDir_saveData = "U:\\eng_research_hrc_binauralhearinglab\\Sudan\\Labs\\Sen Lab\\Research_projects\\Whole_Head_Cocktail_party\\Cocktail_party_whole_head_master_data\\derivatives\\processed_data\\"
flag_save_preprocessed_data = False
flag_run_type = 'overt' # 'overt' or 'covert'

# DEBUG: Print to confirm what flag_run_type is being used
print(f"🔍 DEBUG: flag_run_type is set to: '{flag_run_type}'")
print(f"🔍 DEBUG: This should create folders like: sub_XX_{flag_run_type}")

if flag_run_type.lower() == 'overt':
    selected_file_ids = ['overt_run-01','overt_run-02']
elif flag_run_type.lower() == 'covert':
    selected_file_ids = ['covert_run-01','covert_run-02']
else:
    raise ValueError(f"flag_run_type must be 'overt' or 'covert', got {flag_run_type!r}")

cfg_dataset = {
    'root_dir' : 'U:\eng_research_hrc_binauralhearinglab\Sudan\Labs\Sen Lab\Research_projects\Whole_Head_Cocktail_party\Cocktail_party_whole_head_master_data',
    'subj_ids' : ['01','02','03','04','05','10','11','12','13','14','15','18','20','22','25','28','30','31','32','33','34','35','39','41','44','47'],
    # 'subj_ids' : ['15','18'],
    'file_ids' : selected_file_ids,
    'subj_id_exclude' : [],
}

# Add 'filenm_lst' separately after cfg_dataset is initialized
cfg_dataset['filenm_lst'] = [
    [f"sub-{subj_id}_task-{file_id}_nirs"] 
    for subj_id in cfg_dataset['subj_ids'] 
    for file_id in cfg_dataset['file_ids']
    ]


cfg_prune = {
    'snr_thresh' : 0, # the SNR (std/mean) of a channel. 
    'sd_thresh' : [1, 80]*units.mm, # defines the lower and upper bounds for the source-detector separation that we would like to keep
    'amp_thresh' : [1e-3, 0.84]*units.V, # define whether a channel's amplitude is within a certain range
    'perc_time_clean_thresh' : 0.6,
    'sci_threshold' : 0.6,
    'psp_threshold' : 0.1,
    'window_length' : 5 * units.s,
    'flag_use_sci' : False,
    'flag_use_psp' : False
}


cfg_motion_correct = {
    'flag_do_splineSG' : False, # if True, will do splineSG motion correction
    'splineSG_p' : 0.99, 
    'splineSG_frame_size' : 10 * units.s,
    'flag_do_tddr' : True,
    'flag_do_imu_glm' : False,
    'cfg_imu_glm' : False,
}

cfg_bandpass = { 
    'fmin' : 0.01 * units.Hz,
    'fmax' : 0.5 * units.Hz
}

cfg_GLM = {
    'drift_order' : 1,
    'distance_threshold' : 20*units.mm, # for ssr
    'short_channel_method' : 'mean',
    'noise_model' : "ols",
    't_delta' : 1*units.s ,   # for seq of Gauss basis func - the temporal spacing between consecutive gaussians
    't_std' : 1*units.s ,  
    't_pre' : 2*units.s,
    't_post' : 15*units.s
   #  the temporal spacing between consecutive gaussians
    }

cfg_preprocess = {
    'median_filt' : 1, # set to 1 if you don't want to do median filtering
    'cfg_prune' : cfg_prune,
    'cfg_motion_correct' : cfg_motion_correct,
    'cfg_bandpass' : cfg_bandpass,
    'cfg_GLM': cfg_GLM
}


cfg_mse_conc = {
    'mse_val_for_bad_data' : 1e7 * units.micromolar**2, 
    'mse_amp_thresh' : 1.1e-6*units.V,
    'mse_min_thresh' : 1e0 * units.micromolar**2,
    'blockaverage_val' : 0 * units.micromolar
    }

# if block averaging on OD:
cfg_mse_od = {
    'mse_val_for_bad_data' : 1e1, 
    'mse_amp_thresh' : 1e-3*units.V,
    'mse_min_thresh' : 0.5e-3,
    'blockaverage_val' : 0 
    }


cfg_blockavg = {
    'trange_hrf' : [2, 15] * units.s,
    'trange_hrf_stat' : [4, 8],
    'stim_lst_hrf' : ['Overt Left', 'Overt Right', 'Covert Left', 'Covert Right'], 
    'flag_run_type'      : flag_run_type,   # <<-- set to 'overt' or 'covert'
    'flag_save_group_avg_hrf': True,
    'flag_save_each_subj' : False,  # if True, will save the block average data for each subject
    'cfg_mse_conc' : cfg_mse_conc,
    'cfg_mse_od' : cfg_mse_od
    }               # !!! provide list of rec str and whether or not to save weighted for each one


cfg_erbmICA = {}

cfg_ctrl = dict(cfg_dataset)
cfg_ctrl['file_ids']    = ['control_run-01']
cfg_ctrl['filenm_lst']  = [
    [f"sub-{sid}_task-control_run-01_nirs"]
    for sid in cfg_ctrl['subj_ids']
]
#%%
import gzip
from pathlib import Path

def _load_flavor(flavor, root=rootDir_saveData, snr_thresh=0):
    """
    Return (rec, chs_pruned) for a given data flavor by loading individual subject files.
    Handles both gzip-compressed and uncompressed pickle files.
    
    Parameters
    ----------
    flavor : str
        'overt', 'covert', or 'control'
    root : str or Path
        Root directory containing preprocessed data
    snr_thresh : int
        SNR threshold used during preprocessing (default=5, use 0 for no SNR filtering)
    
    Returns
    -------
    rec : list of lists
        Nested list [subj][run] of recording data
    chs_pruned : list of lists
        Nested list [subj][run] of pruned channel information
    """
    print(f"\n🔄 Loading {flavor.upper()} condition (SNR={snr_thresh})...")
    
    rec = []
    chs_pruned = []
    subj_dir = Path(root) / f"preprocessed_{flavor}_snr_{snr_thresh}"
    
    if not subj_dir.exists():
        raise FileNotFoundError(f"Directory not found: {subj_dir}")
    
    subj_ids = cfg_dataset['subj_ids']
    print(f"📁 Loading from: {subj_dir}")
    print(f"👥 Subjects: {len(subj_ids)}")
    
    for idx, subj_id in enumerate(subj_ids, 1):
        rec_file = subj_dir / f"rec_subj_{subj_id}.pkl"
        prune_file = subj_dir / f"chs_pruned_subj_{subj_id}.pkl"
        
        if rec_file.exists() and prune_file.exists():
            print(f"  [{idx:2d}/{len(subj_ids)}] Loading subject {subj_id}...", end=" ")
            
            # Try gzip first, fall back to regular pickle if not compressed
            try:
                with gzip.open(rec_file, 'rb') as f:
                    rec.append(pickle.load(f))
            except gzip.BadGzipFile:
                with open(rec_file, 'rb') as f:
                    rec.append(pickle.load(f))
            
            try:
                with gzip.open(prune_file, 'rb') as f:
                    chs_pruned.append(pickle.load(f))
            except gzip.BadGzipFile:
                with open(prune_file, 'rb') as f:
                    chs_pruned.append(pickle.load(f))
            
            print("✅")
        else:
            print(f"  [{idx:2d}/{len(subj_ids)}] Subject {subj_id}... ❌ MISSING")
            print(f"       Expected: {rec_file}")
            print(f"       Expected: {prune_file}")
    
    if not rec:
        raise FileNotFoundError(f"No subject data loaded for {flavor} from {subj_dir}")
    
    print(f"✅ Successfully loaded {len(rec)}/{len(subj_ids)} subjects for {flavor.upper()} condition\n")
    return rec, chs_pruned


def _blockavg_all_runs(rec, stim_list,
                       ts_name='conc_p_tddr_filt_postglm',
                       t_pre=cfg_blockavg['trange_hrf'][0],
                       t_post=cfg_blockavg['trange_hrf'][1]):
    """Return nested list [subj][run] of block‑average DataArrays."""
    out = [[None]*len(rec[0]) for _ in range(len(rec))]
    for s_idx in range(len(rec)):
        for r_idx in range(len(rec[s_idx])):
            _, ba = pf.block_average(rec, ts_name, stim_list,
                                     t_pre, t_post,
                                     subj_idx=s_idx, file_idx=r_idx)
            out[s_idx][r_idx] = ba
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Load each flavor and compute block averages
# ─────────────────────────────────────────────────────────────────────────────
stim_labels = {
    'overt'   : ['Overt Left', 'Overt Right'],
    'covert'  : ['Covert Left', 'Covert Right'],
    'control' : ['Control Left', 'Control Right'],          # adjust if your TSV uses a different tag
}
#%%
# Overt
rec_overt,   chs_overt   = _load_flavor('overt')

#%%

ba_overt                  = _blockavg_all_runs(rec_overt, stim_labels['overt'])
#%%
# -------------------------------------------------
# 1.  Decide where and what to name the file
# -------------------------------------------------
outdir   = Path(rootDir_saveData)          # same folder you’ve been using
outdir.mkdir(parents=True, exist_ok=True)  # create it if it doesn’t exist
fname    = 'all_sub_conc_tddr_overt.pkl.gz'         # .gz extension is optional

# -------------------------------------------------
# 2.  Dump with highest pickle protocol + gzip
# -------------------------------------------------
with gzip.open(outdir / fname, 'wb') as f:
    pickle.dump(ba_overt, f, protocol=pickle.HIGHEST_PROTOCOL)

print(f'✔️  Saved block averages → {outdir/fname}')


#%%
# Covert
rec_covert,  chs_covert  = _load_flavor('covert')
ba_covert                 = _blockavg_all_runs(rec_covert, stim_labels['covert'])

# -------------------------------------------------
# 1.  Decide where and what to name the file
# -------------------------------------------------
outdir   = Path(rootDir_saveData)          # same folder you’ve been using
outdir.mkdir(parents=True, exist_ok=True)  # create it if it doesn’t exist
fname    = 'all_sub_conc_tddr_covert.pkl.gz'         # .gz extension is optional

# -------------------------------------------------
# 2.  Dump with highest pickle protocol + gzip
# -------------------------------------------------
with gzip.open(outdir / fname, 'wb') as f:
    pickle.dump(ba_covert, f, protocol=pickle.HIGHEST_PROTOCOL)

print(f'✔️  Saved block averages → {outdir/fname}')

#%%
# Control
rec_ctrl,    chs_ctrl    = _load_flavor('control')
ba_ctrl                   = _blockavg_all_runs(rec_ctrl, stim_labels['control'])

# -------------------------------------------------
# 1.  Decide where and what to name the file
# -------------------------------------------------
outdir   = Path(rootDir_saveData)          # same folder you’ve been using
outdir.mkdir(parents=True, exist_ok=True)  # create it if it doesn’t exist
fname    = 'all_sub_conc_tddr_control.pkl.gz'         # .gz extension is optional

# -------------------------------------------------
# 2.  Dump with highest pickle protocol + gzip
# -------------------------------------------------
with gzip.open(outdir / fname, 'wb') as f:
    pickle.dump(ba_ctrl, f, protocol=pickle.HIGHEST_PROTOCOL)

print(f'✔️  Saved block averages → {outdir/fname}')



# %%
roi_df = pd.read_csv(r"U:\eng_research_hrc_binauralhearinglab\Sudan\Labs\Sen Lab\Research_projects\Whole_Head_Cocktail_party\ROIs\roi_master.csv")
roi_dict = {
    roi: roi_df.loc[roi_df.brodmann == roi, "channel_label"].to_list()
    for roi in roi_df.brodmann.unique()
}
all_rois = sorted(roi_dict)

#%%

def collapse_runs(ba_list):
    out = []
    for subj_runs in ba_list:
        da = xr.concat(subj_runs, dim="run", join="inner").mean("run")
        out.append(da)
    return out

# ── 2. ROI mean per subject (skip missing channels) ────────────────────────
def roi_mean_per_subject(subj_avg_list):
    roi_ds = []
    for da in subj_avg_list:
        roi_slices = []
        for roi, chs in roi_dict.items():
            avail = [c for c in chs if c in da.channel.values]
            if not avail:                      # no surviving channels → skip
                continue
            roi_slice = da.sel(channel=avail).mean("channel")
            roi_slice = roi_slice.expand_dims(ROI=[roi])
            roi_slices.append(roi_slice)
        roi_ds.append(xr.concat(roi_slices, dim="ROI"))
    return roi_ds

def group_mean_sem_robust(subj_roi_list):
    """Robust group mean & SEM with proper NaN handling."""
    stacked = xr.concat(subj_roi_list, dim="subj")
    mean = stacked.mean("subj", skipna=True)
    n = stacked.notnull().sum("subj")
    sem = stacked.std("subj", skipna=True) / np.sqrt(n)
    return mean, sem, n

def bonferroni_pvals_robust(subj_roi_list, win_sec=2.0, alpha=0.05, min_subjects=2):
    """
    Robust Bonferroni correction with validation.
    Only corrects for 7 windows (α/7), not across ROIs.
    """
    # 1. Filter out ROIs with insufficient subjects
    stacked = xr.concat(subj_roi_list, dim="subj")
    # Check if each ROI has at least min_subjects valid data points across any time point
    valid_roi = (stacked.notnull().sum("subj") >= min_subjects).any("reltime")
    
    # Fix: Handle the boolean array properly
    roi_keep = []
    for r in valid_roi.ROI.values:
        roi_valid = valid_roi.sel(ROI=r)
        # Convert to boolean scalar safely
        if roi_valid.values.size == 1:
            is_valid = bool(roi_valid.values.item())
        else:
            # If multiple values, check if any are True
            is_valid = bool(roi_valid.values.any())
        if is_valid:
            roi_keep.append(r)
    
    print(f"Keeping {len(roi_keep)}/{len(valid_roi.ROI)} ROIs with >= {min_subjects} valid subjects")
    
    # 2. Filter data to valid ROIs only
    subj_roi_filtered = []
    for da in subj_roi_list:
        da_filtered = da.sel(ROI=[r for r in roi_keep if r in da.ROI.values])
        subj_roi_filtered.append(da_filtered)
    
    if not subj_roi_filtered or len(roi_keep) == 0:
        print("❌ No valid ROIs found!")
        return None, None, None
    
    da = xr.concat(subj_roi_filtered, dim="subj")
    time = da.reltime.values
    dt = float(np.diff(time).mean())
    w_len = int(round(win_sec / dt))
    win_starts = np.arange(w_len, len(time) - w_len + 1, w_len)
    n_win = len(win_starts)
    
    # Corrected alpha for number of windows only
    alpha_corrected = alpha / n_win
    print(f"Simple Bonferroni: α = {alpha} / {n_win} windows = {alpha_corrected:.6f}")

    # 3. Compute p-values
    pvals = xr.full_like(
        da.isel(reltime=0).expand_dims(window=n_win),
        fill_value=np.nan,
        dtype="float64",
    )
    
    n_valid = xr.full_like(pvals, fill_value=0, dtype="int")

    for w_i, ws in enumerate(win_starts):
        we = ws + w_len
        win_mean = da.isel(reltime=slice(ws, we)).mean("reltime")
        base_mean = da.isel(reltime=slice(0, w_len)).mean("reltime")
        
        # Count valid subjects for each comparison
        win_valid = win_mean.notnull().sum("subj")
        base_valid = base_mean.notnull().sum("subj")
        n_valid.loc[dict(window=w_i)] = np.minimum(win_valid, base_valid)
        
        # Only compute t-test where we have >= min_subjects
        sufficient_data = n_valid.isel(window=w_i) >= min_subjects
        
        if sufficient_data.any():
            _, p_raw = ttest_rel(win_mean, base_mean, axis=0, nan_policy="omit")
            # Set p-values to NaN where insufficient data
            p_raw = xr.where(sufficient_data, p_raw, np.nan)
            pvals.loc[dict(window=w_i)] = p_raw

    print(f"Raw p-values range: {np.nanmin(pvals.values):.6f} to {np.nanmax(pvals.values):.6f}")
    
    return pvals, alpha_corrected, n_valid

def get_significance_mask(pvals, alpha_corrected, n_valid, min_subjects=2):
    """Create significance mask: p < α_corrected AND n_valid >= min_subjects."""
    sig_mask = (pvals < alpha_corrected) & (n_valid >= min_subjects)
    return sig_mask

#%%
# ── apply to each flavour ──────────────────────────────────────────────────
subj_avg_overt   = collapse_runs(ba_overt)
subj_avg_covert  = collapse_runs(ba_covert)
subj_avg_control = collapse_runs(ba_ctrl)

#%%

subj_roi_overt   = roi_mean_per_subject(subj_avg_overt)
subj_roi_covert  = roi_mean_per_subject(subj_avg_covert)
subj_roi_control = roi_mean_per_subject(subj_avg_control)

#%%
# APPLY ROBUST STATISTICS TO ALL CONDITIONS
# ========================================================================

print("\n🔄 Applying robust statistics to all conditions...")

# Apply to all three conditions
conditions = {
    'Overt': subj_roi_overt,
    'Covert': subj_roi_covert, 
    'Control': subj_roi_control
}

robust_results = {}

for condition_name, subj_data in conditions.items():
    print(f"\n--- {condition_name} Condition ---")
    
    # Robust statistics
    group_mean, group_sem, n_subj = group_mean_sem_robust(subj_data)
    pvals, alpha_corr, n_valid = bonferroni_pvals_robust(subj_data, min_subjects=2)
    
    if pvals is not None:
        sig_mask = get_significance_mask(pvals, alpha_corr, n_valid, min_subjects=2)
        
        total_tests = (~np.isnan(pvals)).sum().item()
        significant = sig_mask.sum().item()
        
        robust_results[condition_name] = {
            'group_mean': group_mean,
            'group_sem': group_sem,
            'n_subjects': n_subj,
            'pvals': pvals,
            'alpha_corrected': alpha_corr,
            'n_valid': n_valid,
            'sig_mask': sig_mask,
            'n_significant': significant,
            'n_total': total_tests,
            'sig_rate': 100 * significant / total_tests
        }
        
        print(f"   ✅ Significant: {significant}/{total_tests} ({100*significant/total_tests:.1f}%)")
        
        if sig_mask.any():
            sig_n_valid = n_valid.where(sig_mask)
            print(f"   📊 Subject counts: {sig_n_valid.min().item():.0f}-{sig_n_valid.max().item():.0f}")
    else:
        print(f"   ❌ No valid results for {condition_name}")

print(f"\n✅ Robust analysis complete for all conditions!")



#%%

def plot_roi_group_robust(roi, robust_results, save_path=None):
    """
    Plot group average HRFs for a given ROI across Overt, Covert, Control.
    Shows significant timeseries with thick lines based on robust method results.
    Plots both trial types (left/right) on the same figure.
    """
    # Define more distinct colors for trial types
    colors = {
        'HbO_left': [0.8, 0, 0],            # Crimson Red
        'HbO_right': [1, 0.27, 0],          # Orange Red  
        'HbR_left': [0, 0, 0.8],            # Navy Blue
        'HbR_right': [0, 0.39, 0.8],        # Royal Blue
    }
    
    # Plot settings
    sem_transparency = 0.2
    font_size = 16
    title_font_size = 20  # Adjusted for subplot
    axis_font_size = 14
    legend_font_size = 12
    line_width = 2
    significant_line_width = 8
    
    # x positions for event markers (s)
    events = [0, 2, 5]
    event_colors = ["black", "green", "orange"]
    
    # Check if ROI exists in robust results
    roi_found = False
    for condition_name in ['Overt', 'Covert', 'Control']:
        if condition_name in robust_results:
            if roi in robust_results[condition_name]['group_mean'].ROI.values:
                roi_found = True
                break
    
    if not roi_found:
        print(f"{roi} not present in robust results.")
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    condition_names = ['Overt', 'Covert', 'Control']
    
    for i, condition_name in enumerate(condition_names):
        ax = axes[i]
        
        if condition_name not in robust_results:
            ax.text(0.5, 0.5, f'No data for {condition_name}', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(condition_name, fontsize=title_font_size, fontweight="bold")
            continue
            
        results = robust_results[condition_name]
        
        # Extract data for this ROI
        if roi not in results['group_mean'].ROI.values:
            ax.text(0.5, 0.5, f'{roi} not in {condition_name}', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(condition_name, fontsize=title_font_size, fontweight="bold")
            continue
            
        group_mean = results['group_mean'].sel(ROI=roi)
        group_sem = results['group_sem'].sel(ROI=roi)
        sig_mask = results['sig_mask'].sel(ROI=roi) if 'sig_mask' in results else None
        
        # Get time vector
        t = group_mean.reltime.values
        
        # Plot each chromophore and trial type combination
        for chromo in group_mean.chromo.values:
            for trial_type in group_mean.trial_type.values:
                
                # Determine color and label
                color_key = f"{chromo}_{trial_type.lower()}"
                if color_key in colors:
                    color = colors[color_key]
                    label = f"{chromo} {trial_type}"
                else:
                    # Fallback colors (more distinct)
                    if chromo == 'HbO':
                        color = [0.8, 0, 0] if 'left' in trial_type.lower() else [1, 0.27, 0]
                    else:  # HbR
                        color = [0, 0, 0.8] if 'left' in trial_type.lower() else [0, 0.39, 0.8]
                    label = f"{chromo} {trial_type}"
                
                # Extract mean and SEM for this combination
                m = group_mean.sel(chromo=chromo, trial_type=trial_type)
                se = group_sem.sel(chromo=chromo, trial_type=trial_type)
                
                # Determine line width based on significance
                current_line_width = line_width
                if sig_mask is not None:
                    # Check if any time point is significant for this chromo/trial_type
                    chromo_trial_sig = sig_mask.sel(chromo=chromo, trial_type=trial_type)
                    if chromo_trial_sig.any():
                        current_line_width = significant_line_width
                
                # Plot the line
                ax.plot(t, m, color=color, label=label, 
                       linewidth=current_line_width, alpha=0.9)
                
                # Plot SEM as shaded area
                ax.fill_between(t, m-se, m+se, color=color, 
                               alpha=sem_transparency)
        
        # Add vertical event markers (thicker lines)
        for x, c in zip(events, event_colors):
            ax.axvline(x, linestyle="--", color=c, linewidth=3, alpha=0.8)
        
        # Formatting
        ax.set_title(condition_name, fontsize=title_font_size, fontweight="bold")
        ax.set_xlim(t.min(), t.max())
        ax.set_xlabel("Time (s)", fontsize=axis_font_size)
        ax.grid(True, linestyle=":", alpha=0.4)
        ax.tick_params(labelsize=axis_font_size-2)
    
    # Set common y-label
    axes[0].set_ylabel("Concentration Change\n(μM)", fontsize=axis_font_size)
    
    # Add legend outside the plot area (to the right)
    handles, labels = axes[0].get_legend_handles_labels()
    # if handles:  # Only add legend if there are handles
    #     fig.legend(handles, labels, fontsize=legend_font_size, 
    #               loc='center left', bbox_to_anchor=(1.02, 0.5))
    
    # Main title (without "robust analysis")
    fig.suptitle(f"ROI: {roi}", fontsize=title_font_size+4, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 0.85, 0.94])  # Leave space for legend on right
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    # plt.show()

# %%
# ROBUST PLOTTING WITH SIGNIFICANCE HIGHLIGHTING
# ========================================================================

print("\n📊 Creating publication-quality plots with significance highlighting...")

# Set up save directory
save_dir = Path("U:\\eng_research_hrc_binauralhearinglab\\Sudan\\Labs\\Sen Lab\\Research_projects\\Whole_Head_Cocktail_party\\Group_avg_results\\figures_BA_snr_0")
save_dir.mkdir(parents=True, exist_ok=True)
print(f"📁 Saving figures to: {save_dir}")

# Plot first 5 ROIs using the robust method
for roi in all_rois:
    print(f"Plotting ROI: {roi}")
    save_path = save_dir / f"HRF_robust_{roi}.png"
    plot_roi_group_robust(roi, robust_results, save_path=str(save_path))

print("\n✅ Robust plots created!")
print("   • Thick lines (width=8) indicate significant time series")
print("   • Thin lines (width=1) for non-significant time series") 
print("   • Both trial types (Left/Right) shown on same plot")
print("   • Color scheme: Crimson/Orange Red for HbO Left/Right, Navy/Royal Blue for HbR Left/Right")
print("   • Thicker event lines for better visibility")

# %%