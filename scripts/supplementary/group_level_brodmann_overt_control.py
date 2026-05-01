"""
Group-level Brodmann ROI hemodynamic response — overt-control condition (supplementary).

Variant of the group-level ROI script applied to the overt-control
(distractor) task.

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
import json
from datetime import datetime

import sys
from wholehead_cocktail_party import processing_func as pf

import warnings
warnings.filterwarnings('ignore')

#%%

import importlib
importlib.reload(pf)


# %% Initial root directory and analysis parameters
##############################################################################

flag_load_preprocessed_data = False   # load saved rec/chs if available
rootDir_saveData = "U:\\eng_research_hrc_binauralhearinglab\\Sudan\\Labs\\Sen Lab\\Research_projects\\Whole_Head_Cocktail_party\\Cocktail_party_whole_head_master_data\\derivatives\\processed_data\\overtcontrol_brodmann_snr0\\"
flag_save_preprocessed_data = True  # save rec/chs if we preprocess now
flag_run_type = 'overtcontrol'  # 'overtcontrol' only for this script

# DEBUG: Print to confirm what flag_run_type is being used
print(f"🔍 DEBUG: flag_run_type is set to: '{flag_run_type}'")
print(f"🔍 DEBUG: This should create folders like: sub_XX_{flag_run_type}")

if flag_run_type.lower() == 'overtcontrol':
    selected_file_ids = ['overtcontrol_run-01','overtcontrol_run-02','overtcontrol_run-03']
else:
    raise ValueError(f"flag_run_type must be 'overtcontrol', got {flag_run_type!r}")

cfg_dataset = {
    'root_dir' : 'U:\eng_research_hrc_binauralhearinglab\Sudan\Labs\Sen Lab\Research_projects\Whole_Head_Cocktail_party\Cocktail_party_whole_head_master_data',
    'subj_ids' : ['37','38','42','45','48','49','50','51'],
    'file_ids' : selected_file_ids,
    'subj_id_exclude' : [],
}

# Add 'filenm_lst' as a list of per-subject run lists: shape [n_subjects][n_runs]
cfg_dataset['filenm_lst'] = [
    [f"sub-{subj_id}_task-{file_id}_nirs" for file_id in cfg_dataset['file_ids']]
    for subj_id in cfg_dataset['subj_ids']
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
    'stim_lst_hrf' : ['Overtcontrol Left', 'Overtcontrol Right'],
    'flag_run_type'      : flag_run_type,   # 'overtcontrol'
    'flag_save_group_avg_hrf': True,
    'flag_save_each_subj' : False,  # if True, will save the block average data for each subject
    'cfg_mse_conc' : cfg_mse_conc,
    'cfg_mse_od' : cfg_mse_od
    }


cfg_erbmICA = {}
#%%
import gzip
from pathlib import Path

def _load_flavor(flavor, root=rootDir_saveData):
    """Return (rec, chs_pruned) for a given data flavor.

    Expects 'rec_list_{flavor}.pkl' and 'chs_pruned_subjs_{flavor}.pkl' under root.
    """
    rec_f  = f'rec_list_{flavor}.pkl'
    ch_f   = f'chs_pruned_subjs_{flavor}.pkl'
    with gzip.open(Path(root, rec_f), 'rb') as f:
        rec = pickle.load(f)
    with gzip.open(Path(root, ch_f),  'rb') as f:
        chs = pickle.load(f)
    return rec, chs


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


"""
Overtcontrol-only analysis: load or preprocess overtcontrol runs (3 runs),
compute block averages, ROI means, robust stats, and plots.
"""

# ─────────────────────────────────────────────────────────────────────────────
# 2.  Load overtcontrol and compute block averages
# ─────────────────────────────────────────────────────────────────────────────
stim_labels = {
    'overtcontrol' : ['Overtcontrol Left', 'Overtcontrol Right'],
}

# Load or preprocess overtcontrol data
if flag_load_preprocessed_data:
    try:
        rec_overtcontrol, chs_overtcontrol = _load_flavor('overtcontrol')
        print("Loaded saved overtcontrol rec/chs")
    except FileNotFoundError:
        print("Saved overtcontrol rec/chs not found. Preprocessing now...")
        rec_overtcontrol, chs_overtcontrol = pf.preprocess_batch(cfg_dataset, cfg_preprocess)
        if flag_save_preprocessed_data:
            outdir = Path(rootDir_saveData)
            outdir.mkdir(parents=True, exist_ok=True)
            with gzip.open(outdir / 'rec_list_overtcontrol.pkl', 'wb') as f:
                pickle.dump(rec_overtcontrol, f, protocol=pickle.HIGHEST_PROTOCOL)
            with gzip.open(outdir / 'chs_pruned_subjs_overtcontrol.pkl', 'wb') as f:
                pickle.dump(chs_overtcontrol, f, protocol=pickle.HIGHEST_PROTOCOL)
else:
    rec_overtcontrol, chs_overtcontrol = pf.preprocess_batch(cfg_dataset, cfg_preprocess)
    if flag_save_preprocessed_data:
        outdir = Path(rootDir_saveData)
        outdir.mkdir(parents=True, exist_ok=True)
        with gzip.open(outdir / 'rec_list_overtcontrol.pkl', 'wb') as f:
            pickle.dump(rec_overtcontrol, f, protocol=pickle.HIGHEST_PROTOCOL)
        with gzip.open(outdir / 'chs_pruned_subjs_overtcontrol.pkl', 'wb') as f:
            pickle.dump(chs_overtcontrol, f, protocol=pickle.HIGHEST_PROTOCOL)

# Block averages for all subjects/runs
ba_overtcontrol = _blockavg_all_runs(rec_overtcontrol, stim_labels['overtcontrol'])

# Save block averages snapshot
outdir   = Path(rootDir_saveData)
outdir.mkdir(parents=True, exist_ok=True)
fname    = 'all_sub_conc_tddr_overtcontrol.pkl.gz'
with gzip.open(outdir / fname, 'wb') as f:
    pickle.dump(ba_overtcontrol, f, protocol=pickle.HIGHEST_PROTOCOL)
print(f'✔️  Saved overtcontrol block averages → {outdir/fname}')



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
subj_avg_overtcontrol = collapse_runs(ba_overtcontrol)

#%%

subj_roi_overtcontrol = roi_mean_per_subject(subj_avg_overtcontrol)

#%%
# APPLY ROBUST STATISTICS TO ALL CONDITIONS
# ========================================================================

print("\n🔄 Applying robust statistics to overtcontrol...")

robust_results = {}

condition_name = 'Overtcontrol'
subj_data = subj_roi_overtcontrol
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

print(f"\n✅ Robust analysis complete for overtcontrol!")



#%%

def _save_timeseries_bundle(
    subj_roi_list,
    robust_results,
    condition_key,
    condition_attr,
    subj_ids,
    roi_order,
    root_dir,
    win_sec=2.0,
    events=((0.0, "mov onset"), (3.0, "mov offset")),
):
    """Save group/subject ROI time series and stats to NetCDF for later plotting.

    Parameters
    - subj_roi_list: list of per-subject DataArrays (ROI, chromo, trial_type, reltime)
    - robust_results: dict with keys for the condition
    - condition_key: key in robust_results (e.g., 'Overtcontrol')
    - condition_attr: string attr to store (e.g., 'overtcontrol')
    - subj_ids: list[str] subject identifiers, same order as subj_roi_list
    - roi_order: list[str] all ROI names to standardize across subjects
    - root_dir: directory to save the .nc file
    - win_sec: float, window length used in Bonferroni (seconds)
    - events: iterable of (time_s, name)
    """
    if condition_key not in robust_results:
        print(f"No robust results found for {condition_key}; skipping save.")
        return None

    rr = robust_results[condition_key]

    def _drop_scalar_reltime(da: xr.DataArray) -> xr.DataArray:
        if isinstance(da, xr.DataArray) and ("reltime" in da.coords) and ("reltime" not in da.dims):
            da = da.reset_coords("reltime", drop=True)
        return da

    # Group-level
    group_mean = _drop_scalar_reltime(rr["group_mean"].reindex(ROI=roi_order))
    group_sem = _drop_scalar_reltime(rr["group_sem"].reindex(ROI=roi_order))
    n_subjects = _drop_scalar_reltime(rr["n_subjects"].reindex(ROI=roi_order))

    # Stats aligned to full ROI set
    pvals = _drop_scalar_reltime(rr["pvals"].reindex(ROI=roi_order))
    n_valid = _drop_scalar_reltime(rr["n_valid"].reindex(ROI=roi_order))
    sig_mask = _drop_scalar_reltime(rr["sig_mask"].reindex(ROI=roi_order))

    # Subject-level concatenation; align missing ROIs to NaN
    subj_aligned = [da.reindex(ROI=roi_order) for da in subj_roi_list]
    subj_roi = xr.concat(subj_aligned, dim="subj")
    subj_roi = _drop_scalar_reltime(subj_roi)
    subj_roi = subj_roi.assign_coords(subj=("subj", subj_ids))

    # Derive timing info for windows
    t = group_mean["reltime"].values
    dt = float(np.diff(t).mean()) if len(t) > 1 else np.nan
    w_len = int(round(win_sec / dt)) if np.isfinite(dt) and dt > 0 else None
    if w_len and w_len > 0 and len(t) >= w_len:
        win_starts_idx = np.arange(w_len, len(t) - w_len + 1, w_len)
        win_starts_s = t[win_starts_idx]
        win_ends_s = win_starts_s + win_sec
    else:
        win_starts_s = np.array([])
        win_ends_s = np.array([])

    # Build dataset
    ds = xr.Dataset(
        data_vars=dict(
            group_mean=group_mean,
            group_sem=group_sem,
            n_subjects=n_subjects,
            pvals=pvals,
            sig_mask=sig_mask,
            n_valid=n_valid,
            subj_roi=subj_roi,
        ),
        attrs=dict(
            condition=condition_attr,
            created_at=datetime.utcnow().isoformat() + "Z",
            alpha=float(0.05),
            alpha_corrected=float(rr.get("alpha_corrected", np.nan)),
            win_sec=float(win_sec),
            dt=float(dt) if np.isfinite(dt) else np.nan,
            t_min=float(t.min()) if t.size else np.nan,
            t_max=float(t.max()) if t.size else np.nan,
            events_json=json.dumps([
                {"time": float(et), "name": ename} for (et, ename) in events
            ]),
            units_concentration="micromolar",
            dataset_version="1.0",
            source_script="group_level_brodmann_overtcontrol.py",
        ),
    )

    # Helpful coords for windows
    if "window" in pvals.dims:
        ds = ds.assign_coords(
            window=pvals.coords.get("window", xr.DataArray(np.arange(pvals.sizes.get("window", 0)), dims=["window"]))
        )
        if win_starts_s.size:
            ds = ds.assign_coords(window_start_s=("window", win_starts_s))
            ds = ds.assign_coords(window_end_s=("window", win_ends_s))

    # Add cleaned trial labels aligned to trial_type
    if "trial_type" in group_mean.dims:
        trial_types = group_mean["trial_type"].values
        trial_clean = [str(tt).split()[-1] if isinstance(tt, str) else str(tt) for tt in trial_types]
        ds = ds.assign_coords(trial_clean=("trial_type", np.array(trial_clean)))

    # Save
    outdir = Path(root_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / f"group_roi_timeseries_{condition_attr}.nc"
    try:
        ds.to_netcdf(outpath)
    except Exception as e:
        # Fallback to scipy engine if netCDF4 isn't available
        print(f"NetCDF write default engine failed ({e}); retrying with engine='scipy'.")
        ds.to_netcdf(outpath, engine="scipy")
    print(f"💾 Saved ROI time series bundle → {outpath}")
    return outpath

def plot_roi_group_robust(roi, robust_results, save_path=None):
    """
    Plot group average HRFs for a given ROI for Overtcontrol.
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
    
    # Event markers (s) and colors/labels
    events = [0, 3]
    event_colors = ["green", "orange"]
    event_labels = ["mov onset", "mov offset"]
    
    # Check if ROI exists in robust results
    roi_found = False
    for condition_name in ['Overtcontrol']:
        if condition_name in robust_results:
            if roi in robust_results[condition_name]['group_mean'].ROI.values:
                roi_found = True
                break

    if not roi_found:
        print(f"{roi} not present in robust results.")
        return
    
    fig, axes = plt.subplots(1, 1, figsize=(8, 6), sharey=True)
    axes = np.atleast_1d(axes)
    condition_names = ['Overtcontrol']
    
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
        # sig_mask may only include ROIs that passed min-subjects filtering
        if 'sig_mask' in results and (roi in results['sig_mask'].ROI.values):
            sig_mask = results['sig_mask'].sel(ROI=roi)
        else:
            sig_mask = None
        
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
        
        # Add vertical event markers (thicker lines) with labels
        # Place labels near the top of the axis range
        y_min, y_max = ax.get_ylim()
        y_text = y_max - 0.05 * (y_max - y_min)
        for x, c, lbl in zip(events, event_colors, event_labels):
            ax.axvline(x, linestyle="--", color=c, linewidth=3, alpha=0.9)
            ax.text(x, y_text, lbl, color=c, fontsize=axis_font_size-2,
                    ha='center', va='top', bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=1.5))
        
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
    fig.tight_layout(rect=[0, 0, 0.95, 0.94])
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    # plt.show()

# %%
# ROBUST PLOTTING WITH SIGNIFICANCE HIGHLIGHTING
# ========================================================================

print("\n📊 Creating plots with significance highlighting (overtcontrol)...")

# Set up save directory
save_dir = Path("U:\\eng_research_hrc_binauralhearinglab\\Sudan\\Labs\\Sen Lab\\Research_projects\\Whole_Head_Cocktail_party\\Group_avg_results\\figures_brodmann_overtcontrol")
save_dir.mkdir(parents=True, exist_ok=True)
print(f"📁 Saving figures to: {save_dir}")

# Plot first 5 ROIs using the robust method
for roi in all_rois:
    print(f"Plotting ROI: {roi}")
    save_path = save_dir / f"HRF_robust_{roi}.png"
    plot_roi_group_robust(roi, robust_results, save_path=str(save_path))

print("\n✅ Overtcontrol plots created!")
print("   • Thick lines (width=8) indicate significant time series")
print("   • Thin lines (width=1) for non-significant time series") 
print("   • Both trial types (Left/Right) shown on same plot")
print("   • Color scheme: Crimson/Orange Red for HbO Left/Right, Navy/Royal Blue for HbR Left/Right")
print("   • Thicker event lines for better visibility")

# %%

# Save a complete bundle for later re-plotting/combination
try:
    _save_timeseries_bundle(
        subj_roi_list=subj_roi_overtcontrol,
        robust_results=robust_results,
        condition_key="Overtcontrol",
        condition_attr=flag_run_type.lower(),
        subj_ids=cfg_dataset["subj_ids"],
        roi_order=all_rois,
        root_dir=rootDir_saveData,
        win_sec=2.0,
        events=((0.0, "mov onset"), (3.0, "mov offset")),
    )
except Exception as e:
    print(f"⚠️ Failed to save NetCDF bundle: {e}")

# %%