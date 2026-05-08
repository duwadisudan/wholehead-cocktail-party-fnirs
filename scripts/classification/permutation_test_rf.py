#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Permutation test for the Random Forest classifier (chance-level distribution).

Re-runs the per-subject Random Forest training pipeline with shuffled trial
labels for many iterations, building an empirical null distribution of
per-fold accuracies. Used to derive the chance threshold that figure scripts
overlay on accuracy time-series and scatter plots.

Author: Sudan Duwadi <sudan@bu.edu>
Notes: Code refactoring, documentation, and commenting were AI-assisted;
       all scientific decisions and accountability remain with the author.
"""

#%% Imports

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

import sys
from wholehead_cocktail_party import processing_func as pf
from wholehead_cocktail_party.paths import load_paths, require
from wholehead_cocktail_party.run_config import load_run_config, require_run, resolve_subjects

_PATHS = load_paths()
require(_PATHS, "raw_root", "derivatives_root", "classifier_results_root")

_RUN = load_run_config()
require_run(_RUN, supported_conditions={"overt", "covert"}, supported_modes={"full", "from-derivatives"})

# Default cohort for the permutation test. Existing setup runs on a single
# subject (sub-10) since 1000 permutations per subject is expensive; widen
# via config/run.yml's `subjects` field if you want a larger cohort.
_DEFAULT_COHORT = ['10']

import warnings
warnings.filterwarnings('ignore')

#%%

import importlib
importlib.reload(pf)

import os
import sys
import numpy as npo
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import loguniform
from sklearn.decomposition import PCA
from scipy.ndimage import uniform_filter1d

#%% Initial root directory and analysis parameters

# Pipeline mode and condition come from config/run.yml. require_run() above
# guards against unsupported values.
flag_run_type = _RUN.condition
flag_load_preprocessed_data = (_RUN.mode != 'full')
rootDir_saveData = str(_PATHS.derivatives_root) + os.sep
flag_save_preprocessed_data = False

# DEBUG: Print to confirm what flag_run_type is being used
print(f" DEBUG: flag_run_type is set to: '{flag_run_type}'")
print(f" DEBUG: This should create folders like: sub_XX_{flag_run_type}")

if flag_run_type.lower() == 'overt': 
    selected_file_ids = ['overt_run-01','overt_run-02']
elif flag_run_type.lower() == 'covert':
    selected_file_ids = ['covert_run-01','covert_run-02']
else:
    raise ValueError(f"flag_run_type must be 'overt' or 'covert', got {flag_run_type!r}")

cfg_dataset = {
    'root_dir' : str(_PATHS.raw_root),
    'subj_ids' : resolve_subjects(_RUN, _DEFAULT_COHORT),
    'file_ids' : selected_file_ids,
    'subj_id_exclude' : [],
}

# Add 'filenm_lst' separately after cfg_dataset is initialized
cfg_dataset['filenm_lst'] = [
    [f"sub-{subj_id}_task-{file_id}_nirs"] 
    for subj_id in cfg_dataset['subj_ids'] 
    for file_id in cfg_dataset['file_ids']
    ]


# Trial-quality filtering via augmented events
# Augmented events TSVs live in each subject's nirs/ folder.
# The 'include' column (1 = good trial) is used to drop bad trials.
flag_filter_trials = True   # set False to skip trial filtering entirely

def load_include_mask(subj_id, file_id, root_dir):
    """Load the augmented events TSV and return a boolean include mask.

    Parameters
    ----------
    subj_id  : str, e.g. '01'
    file_id  : str, e.g. 'covert_run-01'
    root_dir : str, BIDS root

    Returns
    -------
    np.ndarray of bool (length = n_trials) or None if file not found.
    """
    import pandas as pd
    # file_id looks like 'covert_run-01' -> task='covert', run='01'
    parts = file_id.rsplit('_run-', 1)
    task, run = parts[0], parts[1]
    fn = f"sub-{subj_id}_task-{task}_run-{run}_events.tsv"
    events_path = os.path.join(root_dir, f"sub-{subj_id}", "nirs", fn)
    if not os.path.isfile(events_path):
        return None
    df = pd.read_csv(events_path, sep='\t')
    if 'include' not in df.columns:
        return None
    inc = df['include']
    # treat 'n/a' or NaN as included (no filtering info available)
    mask = inc.apply(lambda v: bool(int(v)) if str(v) not in ('n/a', 'nan') else True)
    return mask.values

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

#%% Load and preprocess the data

# Load and preprocess the data
#
# This function will load all the data for the specified subject and file IDs, and preprocess the data.
# This function will also create several data quality report (DQR) figures that are saved in /derivatives/plots.
# The function will return the preprocessed data and a list of the filenames that were loaded, both as 
# two dimensional lists [subj_idx][file_idx].
# The data is returned as a recording container with the following fields:
#   timeseries - the data matrices with dimensions of ('channel', 'wavelength', 'time') 
#      or ('channel', 'HbO/HbR', 'time') depending on the data type. 
#      The following sub-fields are included:
#         'amp' - the original amplitude data slightly processed to remove negative and NaN values and to 
#            apply a 3 point median filter to remove outliers.
#         'amp_pruned' - the 'amp' data pruned according to the SNR, SD, and amplitude thresholds.
#         'od' - the optical density data
#         'od_tddr' - the optical density data after TDDR motion correction is applied
#         'conc_tddr' - the concentration data obtained from 'od_tddr'
#         'od_splineSG' and 'conc_splineSG' - returned if splineSG motion correction is applied (i.e. flag_do_splineSG=True)
#   stim - the stimulus data with 'onset', 'duration', and 'trial_type' fields and more from the events.tsv files.
#   aux_ts - the auxiliary time series data from the SNIRF files.
#      In addition, the following aux sub-fields are added during pre-processing:
#         'gvtd' - the global variance of the time derivative of the 'od' data.
#         'gvtd_tddr' - the global variance of the time derivative of the 'od_tddr' data.

"""
rec is list with dimensions nsubj x nruns
    - each index contains the rec with the following timeseries:
        amp = pruned amplitude
        amp_o = unpruned amplitde
        od = pruned optical density 
        od_o = unpruned optical density 
        od_tddr = pruned optical density with TDDR
        od_tddr_o = unpruned optical density with TDDR
        conc_tddr = pruned concentration with TDDR
        conc_tddr_o = unpruned concentration with TDDR

chs_pruned_subjs is is list with dimensions nsubj x nruns
    - contains a value per channel depending on why it was pruned 
    - not pruned = 0.4
    - SNR > 5 = 0.19
    - saturated = 0.0
    - low amplitude = 0.8
    - outside SD range = 0.65
    - SCI or PSP = 0.95
"""

# determine the number of subjects and files. Often used in loops.
n_subjects = len(cfg_dataset['subj_ids'])
n_files_per_subject = len(cfg_dataset['file_ids'])
# pdb.set_trace()
# files to load
for subj_id in cfg_dataset['subj_ids']:
    subj_idx = cfg_dataset['subj_ids'].index(subj_id)
    for file_id in cfg_dataset['file_ids']:
        file_idx = cfg_dataset['file_ids'].index(file_id)
        filenm = f'sub-{subj_id}_task-{file_id}_nirs'
        if subj_idx == 0 and file_idx == 0:
            cfg_dataset['filenm_lst'] = []
            cfg_dataset['filenm_lst'].append( [filenm] )
        elif file_idx == 0:
            cfg_dataset['filenm_lst'].append( [filenm] )
        else:
            cfg_dataset['filenm_lst'][subj_idx].append( filenm )


#%%
def load_pickle_auto(path):
    """Load either a gzip-compressed pickle or a plain pickle."""
    with open(path, 'rb') as f:
        header = f.read(2)

    if header == b'\x1f\x8b':
        with gzip.open(path, 'rb') as f:
            return pickle.load(f)

    with open(path, 'rb') as f:
        return pickle.load(f)


#%%
# Helper function to load all subjects
def load_all_subjects(subj_ids, run_type, data_dir):
    """Load all preprocessed subjects from individual files."""
    rec = []
    chs_pruned_subjs = []
    subj_dir = os.path.join(data_dir, f"preprocessed_{run_type}_snr_0")
    
    for subj_id in subj_ids:
        rec_file = os.path.join(subj_dir, f"rec_subj_{subj_id}.pkl")
        prune_file = os.path.join(subj_dir, f"chs_pruned_subj_{subj_id}.pkl")
        
        if os.path.exists(rec_file) and os.path.exists(prune_file):
            rec.append(load_pickle_auto(rec_file))
            chs_pruned_subjs.append(load_pickle_auto(prune_file))
            print(f"Loaded subject {subj_id}")
        else:
            print(f"Warning: Subject {subj_id} not found, skipping")
            rec.append(None)
            chs_pruned_subjs.append(None)
    
    return rec, chs_pruned_subjs
#%%
if not flag_load_preprocessed_data:
    print("Running load and process function - saving each subject individually")
    rec, chs_pruned_subjs = pf.preprocess_batch(cfg_dataset, cfg_preprocess)
    
    # SAVE preprocessed data - ONE SUBJECT AT A TIME
    if flag_save_preprocessed_data:
        outdir = os.path.join(cfg_dataset['root_dir'], 'derivatives', 'processed_data', f"preprocessed_{flag_run_type}_snr_0")
        os.makedirs(outdir, exist_ok=True)
        
        for subj_idx, subj_id in enumerate(cfg_dataset['subj_ids']):
            # Save this subject's data
            rec_file = os.path.join(outdir, f"rec_subj_{subj_id}.pkl")
            prune_file = os.path.join(outdir, f"chs_pruned_subj_{subj_id}.pkl")
            
            # Use temporary files for atomic writes
            rec_tmp = rec_file + '.tmp'
            prune_tmp = prune_file + '.tmp'
            
            with open(rec_tmp, 'wb') as f:
                pickle.dump(rec[subj_idx], f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(rec_tmp, rec_file)
            
            with open(prune_tmp, 'wb') as f:
                pickle.dump(chs_pruned_subjs[subj_idx], f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(prune_tmp, prune_file)
            
            print(f" Saved subject {subj_id}")
        
        print(f"All subjects saved to: {outdir}")
        
# LOAD in saved data
else:
    print("Loading saved data from individual subject files")
    data_dir = os.path.join(cfg_dataset['root_dir'], 'derivatives', 'processed_data')
    rec, chs_pruned_subjs = load_all_subjects(cfg_dataset['subj_ids'], flag_run_type, data_dir)
#%%

import copy, gzip, pickle, os, sys, numpy as np, matplotlib.pyplot as plt
import xarray as xr

from sklearn.model_selection           import StratifiedKFold
from sklearn.discriminant_analysis     import LinearDiscriminantAnalysis

import cedalion
import cedalion.nirs                    as nirs
from cedalion.nirs import split_long_short_channels
import cedalion.models.glm as glm
from cedalion.models.glm               import GaussianKernels
from cedalion import units
import pandas as pd

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib import cm

# %%

def fit_glm_excluding_test(
        ts       : xr.DataArray,            # (channel, chromo, time)
        dm_full  : xr.DataArray,            # (time, regressor, chromo)
        *,
        test_stim: pd.DataFrame,
        before_s : float,
        after_s  : float,
        method   : str  = "individual",     # "individual" | "block"
        noise_model: str = "ols",
        aux      : xr.DataArray | None = None,   # <- NEW, may be None
):
    """
    Zero-mask rows that belong to test trials and fit the GLM.

    Returns
    -------
    betas      : xarray.DataArray
    dm_masked  : xarray.DataArray  (time, regressor, chromo)  – the DM used for fit
    """
    import pandas as pd

    dm_masked = dm_full.copy(deep=True)

    # helper to get plain seconds
    def _to_s(coord):
        try:
            return coord.pint.to("s").values.astype(float)
        except Exception:
            return coord.values.astype(float)

    t_sec = _to_s(dm_masked.time)
    o_sec = _to_s(test_stim.onset)

    # build Boolean mask
    rows = np.zeros_like(t_sec, bool)
    if method.lower() == "block":
        rows |= (t_sec >= o_sec.min() - before_s) & (t_sec <= o_sec.max() + after_s)
    else:  # individual
        for o in o_sec:
            rows |= (t_sec >= o - before_s) & (t_sec <= o + after_s)

    dm_masked.values[rows, :, :] = 0      # works for (time, regressor, chromo)

    if aux is not None:
        aux_masked = aux.copy(deep=True)
        aux_masked.values[rows, ...] = 0          # <-- add this line
    else:
        aux_masked = None
    # fit GLM
    betas = glm.fit(
        ts,               # (channel, chromo, time)
        dm_masked,        # (time,    regressor, chromo)
        aux_masked,              # channel-wise SSR regressor(s)
        noise_model=noise_model,
    )

    return betas, dm_masked

#%%

def reconstruct_hrf(
        betas_cond: xr.DataArray,    # beta’s for one condition  (component × channel)
        basis_fun : GaussianKernels,
        ts_full   : xr.DataArray,    # just for timing / sampling info
        chromo    : str = "HbO"
) -> xr.DataArray:
    """
    Parameters
    ----------
    betas_cond : output of betas.sel(regressor=idx)  – dims ('regressor','channel')
    basis_fun  : the *same* GaussianKernels object you passed into make_design_matrix
    ts_full    : original timeseries – only used for dt / number of samples
    chromo     : 'HbO' or 'HbR'

    Returns
    -------
    HRF : DataArray (channel × time)  – reconstructed haemodynamic kernel
    """

    # 1) rebuild the (time × component × chromo) basis for THIS run
    basis = basis_fun(ts_full)                     # same sampling as the data
    if "chromo" in basis.dims:
        basis = basis.sel(chromo=chromo)
    basis = basis.transpose("component", "time")   # component first for mat-mul

    if "chromo" in betas_cond.dims:                #  <- NEW
        betas_cond = betas_cond.sel(chromo=chromo)

    # 2) ensure components in betas are in the same order as basis.component
    #    Cedalion’s make_design_matrix already appends components in order 00,…,
        betas_c = (betas_cond
                  .sortby('regressor')
                  .rename({'regressor':'component'})
                  .assign_coords(component=basis.component.values)
              )    
 
    # 3) matrix multiply  (component × channel)  @  (component × time)
    hrf = xr.dot(betas_c, basis, dims="component")    # channel × time
    hrf = hrf.transpose("channel", "time")

    return hrf


#%%
run_flag = cfg_blockavg.get('flag_run_type','overt').lower()
if run_flag == 'overt':
    cfg_GLM['stim_lst_hrf'] = ['Overt Left', 'Overt Right']
elif run_flag == 'covert':
    cfg_GLM['stim_lst_hrf'] = ['Covert Left', 'Covert Right']
else:
    raise ValueError(f"flag_run_type must be 'overt' or 'covert', got '{run_flag}'")

#%%

def learn_channel_hrf(rec_full, test_stim, cfg):
    """
    Learn per-channel HRF shape (Left / Right) from one run,
    zero-masking test trials, and also return the run-level SS beta.
    Robust to aux==None by falling back on any 'SS' regressors in betas.
    """
    ts_full = rec_full["conc_p_tddr_filt"]  # (chromo, channel, time)

    # 1) split long/short channels
    ts_long, ts_short = split_long_short_channels(
        ts_full, rec_full.geo3d,
        distance_threshold=cfg["distance_threshold"]
    )

    # 2) build full DM + aux
    basis = GaussianKernels(
        cfg["t_pre"], cfg["t_post"],
        cfg["t_delta"], cfg["t_std"]
    )
    dm_full, aux = glm.make_design_matrix(
        ts_long, ts_short, rec_full.stim, rec_full.geo3d,
        basis_function      = basis,
        drift_order         = cfg["drift_order"],
        short_channel_method= cfg["short_channel_method"],
    )

    # 3) fit masked GLM, passing aux so SS regressors end up in betas
    betas, dm_train = fit_glm_excluding_test(
        ts         = ts_full,
        dm_full    = dm_full,
        test_stim  = test_stim,
        before_s   = cfg["t_pre"].to("s").magnitude,
        after_s    = cfg["t_post"].to("s").magnitude,
        noise_model= cfg.get("noise_model","ols"),
        aux        = aux,  
    )

    # 4) extract SS regressor names
    if aux is not None and hasattr(aux, "regressor"):
        ss_names = list(aux.regressor.values)
    else:
        # fallback: any betas.regressor containing 'SS' or 'short'
        ss_names = [r for r in betas.regressor.values
                    if ("ss" in str(r).lower()) or ("short" in str(r).lower())]
    if not ss_names:
        raise RuntimeError("No SS regressors found in betas or aux!")

    # 5) pull out & average SS beta’s (HbO only) -> shape (n_channels,)
    ss_beta_da = betas.sel(regressor=ss_names, chromo="HbO")
    Beta_ss_global = ss_beta_da.mean(dim="regressor").values

    # 6) reconstruct Left/Right HRFs
    cond_L, cond_R = cfg['stim_lst_hrf']
    idx_L = betas.regressor.str.contains(fr"HRF.*{cond_L}")
    idx_R = betas.regressor.str.contains(fr"HRF.*{cond_R}")
    hrf_L = reconstruct_hrf(betas.sel(regressor=idx_L), basis, ts_full, chromo="HbO")
    hrf_R = reconstruct_hrf(betas.sel(regressor=idx_R), basis, ts_full, chromo="HbO")

    # 7) short-sep time-series
    ss_reg = ts_short.mean("channel")  # (chromo, time)

    return hrf_L, hrf_R, ss_reg, dm_full, dm_train, Beta_ss_global




#%%

import numpy as np
import xarray as xr
from cedalion.models.glm import fit
import re

def extract_single_trial_ts_two_regressors(
    run_full,          # one run dict
    hrfL,              # DataArray: learned Left HRF (channel × time)
    hrfR,              # DataArray: learned Right HRF (channel × time)
    ss_reg,            # DataArray: short-sep regressor (chromo × time)
    dmF,               # DataArray: full DM (time × regressor × chromo)
    idx_tr,            # train trial indices
    idx_te,            # test trial indices
    t_pre,             # pint Quantity
    t_post,            # pint Quantity
    Beta_ss_global     # np.ndarray, shape (n_channels,)
):
    """
    – Training: return raw HbO segments
    – Testing:       subtract Beta_ss_global * ss_segment
    """
    ts_full = run_full["conc_p_tddr_filt"]
    times   = ts_full.time.values
    chromos = ts_full.chromo.values
    channels= ts_full.channel.values

    seg_len = hrfL.sizes["time"]

    # compute dt in seconds correctly for both timedelta and float64 times
    if np.issubdtype(times.dtype, np.timedelta64):
        dt = (times[1] - times[0]) / np.timedelta64(1, "s")
        dt = float(dt)
    else:
        # times are plain floats (seconds)
        dt = float(times[1] - times[0])

    t_pre_s = t_pre.to("s").magnitude
    n_pre   = int(round(t_pre_s / dt))
    n_post  = seg_len - n_pre - 1
    seg_t   = np.linspace(-n_pre*dt, n_post*dt, seg_len)

    def make_trial(onset):
        # find nearest sample
        if np.issubdtype(times.dtype, np.timedelta64):
            tvals = np.array([(t - times[0]) / np.timedelta64(1,"s") for t in times])
        else:
            tvals = np.array([float(t) for t in times])
        idx0 = int(np.argmin(np.abs(tvals - float(onset))))

        inds  = np.arange(idx0-n_pre, idx0+n_post+1)
        valid = (inds>=0)&(inds<len(tvals))
        ts_seg= np.zeros((len(chromos), len(channels), seg_len))
        for ii, ii_glob in enumerate(inds):
            if valid[ii]:
                ts_seg[:,:,ii] = ts_full.isel(time=ii_glob).values

        # extract SS trace for that segment
        ss_full = ss_reg.sel(chromo="HbO").values  # (time,)
        ss_seg  = ss_full[inds.clip(0,len(ss_full)-1)]

        return ts_seg, ss_seg

    def proc(indices, is_train):
        out, labs = [], []
        for i in indices:
            onset = run_full.stim.onset.values[i]
            label = run_full.stim.trial_type.values[i]
            ts_seg, ss_seg = make_trial(onset)
            # pick HbO only
            ch_hbo = np.where(chromos=="HbO")[0][0]
            y_hbo = ts_seg[ch_hbo]  # (channel, seg_len)

            if is_train:
                y_est = y_hbo
            else:
                y_est = y_hbo - Beta_ss_global[:,None] * ss_seg[None,:]

            out.append(y_est)
            labs.append(label)

        return np.stack(out, axis=0), np.array(labs)

    ts_tr, lab_tr = proc(idx_tr, True)
    ts_te, lab_te = proc(idx_te, False)
    return ts_tr, ts_te, lab_tr, lab_te


#%%
def sliding_window_classify(
    ts_train, ts_test,
    labels_train, labels_test,
    t_rel, window_size, step_size,
    classifiers,
    scaler=None,
    pca=None,
    feature_set='max',
):
    """Sliding-window classification with optional PCA (already fit on training).

    Parameters
    ----------
    ts_train, ts_test : ndarray (n_trials, n_channels, T)
    labels_train, labels_test : 1D arrays
    t_rel : 1D array of relative time (seconds)
    window_size, step_size : ints (samples)
    classifiers : dict[str, estimator]
    scaler, pca : fitted objects or None
    """
    T = ts_train.shape[2]
    n_windows = (T - window_size) // step_size + 1
    acc = {name: np.zeros(n_windows) for name in classifiers}
    bal_acc = {name: np.zeros(n_windows) for name in classifiers}
    train_acc = {name: np.zeros(n_windows) for name in classifiers}
    train_bal_acc = {name: np.zeros(n_windows) for name in classifiers}
    preds_all = {name: [] for name in classifiers}
    times = np.array([
        t_rel[w * step_size:(w * step_size + window_size)].mean()
        for w in range(n_windows)
    ])

    for w in range(n_windows):
        start, end = w * step_size, w * step_size + window_size
        # Build features from PC timecourses within the window
        seg_tr = ts_train[:, :, start:end]
        seg_te = ts_test[:,  :, start:end]

        # Window maxima per PC
        if feature_set in ('max', 'both'):
            Xmax_tr = seg_tr.max(axis=2)
            Xmax_te = seg_te.max(axis=2)

        # OLS slope per PC within window
        if feature_set in ('slope', 'both'):
            t_win = t_rel[start:end]
            if t_win.size == 0:
                Xslope_tr = np.zeros(seg_tr.shape[:2])
                Xslope_te = np.zeros(seg_te.shape[:2])
            else:
                t0 = t_win - t_win.mean()
                denom = float((t0**2).sum())
                if denom <= 0:
                    Xslope_tr = np.zeros(seg_tr.shape[:2])
                    Xslope_te = np.zeros(seg_te.shape[:2])
                else:
                    Xslope_tr = (seg_tr * t0[None, None, :]).sum(axis=2) / denom
                    Xslope_te = (seg_te * t0[None, None, :]).sum(axis=2) / denom

        # Select/concatenate
        if feature_set == 'max':
            X_tr, X_te = Xmax_tr, Xmax_te
        elif feature_set == 'slope':
            X_tr, X_te = Xslope_tr, Xslope_te
        elif feature_set == 'both':
            X_tr = np.concatenate([Xmax_tr, Xslope_tr], axis=1)
            X_te = np.concatenate([Xmax_te, Xslope_te], axis=1)
        else:
            raise ValueError(f"Unknown feature_set: {feature_set}")
        if scaler is not None:
            X_tr_t = scaler.transform(X_tr)
            X_te_t = scaler.transform(X_te)
        else:
            X_tr_t, X_te_t = X_tr, X_te
        if pca is not None:
            X_tr_t = pca.transform(X_tr_t)
            X_te_t = pca.transform(X_te_t)
        for name, clf in classifiers.items():
            clf.fit(X_tr_t, labels_train)
            # Test accuracy
            y_pred = clf.predict(X_te_t)
            acc[name][w] = accuracy_score(labels_test, y_pred)
            bal_acc[name][w] = balanced_accuracy_score(labels_test, y_pred)
            preds_all[name].append(y_pred.copy())
            # Training accuracy (OOB if available, else re-substitution)
            if hasattr(clf, 'oob_score_'):
                train_acc[name][w] = clf.oob_score_
                # Compute OOB balanced accuracy from OOB decision function
                if hasattr(clf, 'oob_decision_function_'):
                    oob_preds = np.argmax(clf.oob_decision_function_, axis=1)
                    train_bal_acc[name][w] = balanced_accuracy_score(labels_train, oob_preds)
                else:
                    train_bal_acc[name][w] = clf.oob_score_
            else:
                y_pred_tr = clf.predict(X_tr_t)
                train_acc[name][w] = accuracy_score(labels_train, y_pred_tr)
                train_bal_acc[name][w] = balanced_accuracy_score(labels_train, y_pred_tr)

    return times, acc, {
        'balanced_acc': bal_acc,
        'predictions': preds_all,
        'train_acc': train_acc,
        'train_bal_acc': train_bal_acc,
    }


def fit_pca_from_moving_windows(ts_train, window_size, step_size, *, mode='variance', var_ratio=0.95, n_components=None):
    """(Legacy) Fit (StandardScaler, PCA) on concatenated moving-window max features.
    Kept for backward compatibility when PCA_TIMECOURSE_FIRST=False.
    """
    T = ts_train.shape[2]
    n_windows = (T - window_size) // step_size + 1
    feats = []
    for w in range(n_windows):
        s, e = w * step_size, w * step_size + window_size
        feats.append(ts_train[:, :, s:e].max(axis=2))
    X_all = np.concatenate(feats, axis=0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_all)
    if mode == 'fixed':
        if n_components is None:
            raise ValueError("n_components must be provided when mode='fixed'")
        pca = PCA(n_components=n_components, svd_solver='full', random_state=0)
    else:
        pca = PCA(n_components=var_ratio, svd_solver='full', random_state=0)
    pca.fit(X_scaled)
    return scaler, pca

# New PCA helpers (timecourse-first)

def fit_pca_on_full_timecourse(ts_train, *, n_components=20, scale=True):
    """Fit PCA on stacked timecourses (n_trials, n_channels, T) -> (n_trials*T, n_channels)."""
    n_trials, n_channels, T = ts_train.shape
    X = ts_train.transpose(0,2,1).reshape(n_trials*T, n_channels)
    scaler = StandardScaler() if scale else None
    X_proc = scaler.fit_transform(X) if scaler is not None else X
    pca = PCA(n_components=n_components, svd_solver='full', random_state=0)
    pca.fit(X_proc)
    return scaler, pca

def transform_trials_to_pc_timecourses(ts, scaler, pca):
    """Project (n_trials, n_channels, T) to (n_trials, n_components, T)."""
    n_trials, n_channels, T = ts.shape
    pc_trials = []
    for tr in range(n_trials):
        X = ts[tr].T  # (T, n_channels)
        if scaler is not None:
            Xs = (X - scaler.mean_) / scaler.scale_
        else:
            Xs = X
        scores = Xs @ pca.components_.T  # (T, n_components)
        pc_trials.append(scores.T)
    return np.stack(pc_trials, axis=0)

def good_channels(trials, labels, N, windowStartT, windowEndT, fs):
    """
    Rank channels by a composite of smoothed d' (peak + AUC) with a hold-test.
    trials: (n_trials, n_chan, T)
    labels: (n_trials,) 0/1
    N: number to select
    windowStartT/windowEndT: sample indices (int)
    fs: sampling rate (Hz)
    """
    ws = int(np.ceil(windowStartT))
    we = int(np.ceil(windowEndT))
    n_trials, n_chan, Tfull = trials.shape
    # compute d'
    dprime = np.zeros((n_chan, we), float)
    for ch in range(n_chan):
        pos = trials[labels==1, ch, :we]
        neg = trials[labels==0, ch, :we]
        mu_p = pos.mean(axis=0); mu_n = neg.mean(axis=0)
        sd_p = pos.std(axis=0, ddof=1); sd_n = neg.std(axis=0, ddof=1)
        dprime[ch] = np.abs(mu_p - mu_n) / np.sqrt(0.5*(sd_p**2 + sd_n**2) + 1e-8)
    # smooth
    smD = uniform_filter1d(dprime, size=9, axis=1, mode='nearest')
    # baseline
    base_end = max(ws, 1)
    base_mean = smD[:, :base_end].mean(axis=1)
    base_std  = smD[:, :base_end].std(axis=1, ddof=0)
    threshold = base_mean + 0.5*base_std
    # epoch
    smD2 = smD[:, ws:we+1]
    T2 = smD2.shape[1]
    # composite
    peakVal = smD2.max(axis=1)
    aucVal  = np.trapz(smD2, dx=1, axis=1) / T2
    composite_score = 0.7*peakVal + 0.3*aucVal
    # hold-test
    hold_samps = min(int(np.floor(2*fs)), T2-1)
    dip_tol    = 3
    keep       = np.zeros(n_chan, bool)
    for ch in range(n_chan):
        vec = smD2[ch]; thr = threshold[ch]
        above = np.where(vec >= thr)[0]
        if above.size == 0 or (above[0] + hold_samps) > T2:
            continue
        segment = vec[above[0]:above[0]+hold_samps]
        if np.sum(np.diff(segment) < 0) <= dip_tol:
            keep[ch] = True
    composite_score[~keep] = -np.inf
    # top N
    idx_sorted = np.argsort(composite_score)[::-1]
    return idx_sorted[:min(N, n_chan)], composite_score

#%%
# REMOVE MLflow configuration - using traditional file outputs only


# global hyper-parameters
EPS_STOP          = 0.005   # 0.5-pp absolute accuracy threshold
TOP_N_POOL        = 20
INNER_SPLITS      = 3
RANDSEARCH_TRIES  = 50
RANDSEARCH_JOBS   = 20

# PCA configuration
# PCA_MODE: 'fixed' keeps exactly PCA_N_COMPONENTS PCs; 'variance' keeps enough to reach PCA_VAR_RATIO
# Original workflow: window-max features -> PCA. New: PCA on full timecourse first then window features in PC space.
PCA_MODE                = 'fixed'
PCA_N_COMPONENTS        = 20
PCA_VAR_RATIO           = 0.95
PCA_TIMECOURSE_FIRST    = True   # NEW
SAVE_TIMECOURSE_PC_PLOT = True   # NEW

# D-prime + PCA configuration
DPRIME_TOP_N = 20
PCA_VAR_THRESHOLD = 0.95  # keep PCs until 95% variance (sklearn PCA with n_components=float)

# Feature configuration for per-window PC features
# Options:
#   'max'   -> use only window maxima per PC (existing behavior)
#   'slope' -> use only OLS slope within the window per PC
#   'both'  -> concatenate [max, slope] per PC
FEATURE_SET = 'both'

# Diagnostic / elaborate figure toggle
# When True, produces per-fold train+test accuracy curves and overlap reports
FLAG_DIAGNOSTIC_PLOTS = True

# %% ------------------------------------------------------------------ #
# 2)  HELPER: greedy forward selection (inner-fold) ------------------- #
from sklearn.base import clone
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix

def greedy_select(X_tr, y_tr, X_va, y_va,
                  base_clf, epsilon=EPS_STOP, max_iter=None):
    """
    Greedy forward selection on *channels*.
    Returns `best_subset` (list of channel indices) and `history`
    (= list of (subset, val_acc) tuples).
    """
    n_chan = X_tr.shape[1]
    pool   = list(range(n_chan))
    chosen = []
    best_acc = 0.0
    history  = []

    while True:
        gains = []
        for ch in pool:
            cand   = chosen + [ch]
            clf    = clone(base_clf)
            clf.fit(X_tr[:, cand], y_tr)
            acc    = accuracy_score(y_va, clf.predict(X_va[:, cand]))
            gains.append((acc, ch))

        acc_best, ch_best = max(gains, key=lambda t: t[0])
        history.append((chosen + [ch_best], acc_best))

        if acc_best - best_acc < epsilon:
            break
        best_acc = acc_best
        chosen.append(ch_best)
        pool.remove(ch_best)

        if max_iter and len(chosen) >= max_iter:
            break

    return chosen, history

# Epoch-overlap diagnostic
def check_epoch_overlap(stim1, stim2, tr1, te1, tr2, te2, t_pre, t_post,
                        verbose=True):
    """Detect temporal overlap between train and test epochs.

    Each trial occupies [onset - t_pre, onset + t_post].  If *any* test-epoch
    sample window overlaps with *any* train-epoch sample window (even from the
    other run), we count it.

    Parameters
    ----------
    stim1, stim2 : pandas DataFrame (must have 'onset' column, in seconds)
    tr1, te1     : index arrays for run-1 train / test splits
    tr2, te2     : index arrays for run-2 train / test splits
    t_pre, t_post: float, seconds (positive)

    Returns
    -------
    dict with keys:
        n_overlapping_pairs : int
        overlap_pairs       : list of (run_src, trial_i, run_tgt, trial_j, gap_s)
        n_train, n_test     : int
    """
    epoch_len = t_pre + t_post

    # Build list of (onset_start, onset_end, run_tag, trial_idx) for train & test
    def _epochs(stim, idxs, run_tag):
        out = []
        for i in idxs:
            o = float(stim.onset.values[i])
            out.append((o - t_pre, o + t_post, run_tag, int(i)))
        return out

    train_epochs = _epochs(stim1, tr1, 1) + _epochs(stim2, tr2, 2)
    test_epochs  = _epochs(stim1, te1, 1) + _epochs(stim2, te2, 2)

    overlaps = []
    for te_start, te_end, te_run, te_idx in test_epochs:
        for tr_start, tr_end, tr_run, tr_idx in train_epochs:
            # two intervals overlap iff start_A < end_B and start_B < end_A
            if te_start < tr_end and tr_start < te_end:
                gap = max(0, min(te_end, tr_end) - max(te_start, tr_start))
                overlaps.append((tr_run, tr_idx, te_run, te_idx, round(gap, 2)))

    result = {
        "n_overlapping_pairs": len(overlaps),
        "overlap_pairs": overlaps,
        "n_train": len(train_epochs),
        "n_test": len(test_epochs),
    }

    if verbose and len(overlaps) > 0:
        print(f"     OVERLAP: {len(overlaps)} train-test epoch pairs share samples "
              f"(out of {len(train_epochs)}×{len(test_epochs)} possible)")
        # summarize per-test-trial
        test_with_overlap = set((r, i) for _, _, r, i, _ in overlaps)
        print(f"      {len(test_with_overlap)}/{len(test_epochs)} test trials "
              f"overlap >=1 train trial (epoch={epoch_len:.1f}s)")

    return result


#%%




from sklearn.model_selection import RepeatedStratifiedKFold

# define your classifiers once
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from collections import Counter

from sklearn.model_selection import StratifiedKFold

# define RandomForest with fixed parameters - no hyperparameter tuning
# class_weight='balanced' compensates for 2:1 Left/Right imbalance
# max_depth=5, min_samples_leaf=3 regularize to reduce overfitting
# oob_score=True gives honest training-set generalization estimate
rf_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('rf', RandomForestClassifier(
        n_estimators=300, max_depth=5, min_samples_leaf=3,
        class_weight='balanced', oob_score=True,
        random_state=42, n_jobs=-1))
])

#%%
import json

N_PERMUTATIONS = 1000
N_CV_FOLDS = 5
PERMUTATION_PROGRESS_EVERY = 100
BASELINE_ONLY = True

for subj_idx, subj_id in enumerate(cfg_dataset['subj_ids']):
    if rec[subj_idx] is None:
        print(f"\n\n=== Skipping subject {subj_id}: preprocessed data not available ===")
        continue

    print(f"\n\n=== Processing subject {subj_id} ===")
    cfg = cfg_GLM

    run1_full = rec[subj_idx][0]
    run2_full = rec[subj_idx][1]

    if flag_filter_trials:
        for run_idx, (run_obj, fid) in enumerate(
                zip([run1_full, run2_full], cfg_dataset['file_ids'])):
            inc_mask = load_include_mask(subj_id, fid, cfg_dataset['root_dir'])
            if inc_mask is not None:
                n_before = len(run_obj.stim)
                run_obj.stim = run_obj.stim[inc_mask].reset_index(drop=True)
                n_after = len(run_obj.stim)
                print(f"  Run {run_idx+1} ({fid}): kept {n_after}/{n_before} trials (include==1)")
                if n_after < 6:
                    print(f"  WARNING: only {n_after} trials left; results may be unreliable")
            else:
                print(f"  Run {run_idx+1} ({fid}): no augmented events found; keeping all trials")
    else:
        print('  Trial filtering disabled (flag_filter_trials=False)')

    print('  No control-based channel exclusion applied')
    common = np.intersect1d(
        run1_full['conc_p_tddr_filt'].channel.values,
        run2_full['conc_p_tddr_filt'].channel.values
    )
    print(f"  Found {len(common)} common channels between runs")

    for run_full in (run1_full, run2_full):
        run_full['conc_p_tddr_filt'] = run_full['conc_p_tddr_filt'].sel(channel=common)

    run1_labels = np.array([0 if 'Left' in s else 1 for s in run1_full.stim.trial_type.values])
    run2_labels = np.array([0 if 'Left' in s else 1 for s in run2_full.stim.trial_type.values])

    outer_cv1 = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)
    outer_cv2 = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)
    splits1 = list(outer_cv1.split(np.zeros_like(run1_labels), run1_labels))
    splits2 = list(outer_cv2.split(np.zeros_like(run2_labels), run2_labels))

    dt = run1_full['conc_p_tddr_filt'].time.diff('time')[0].values
    fs = 1.0 / dt

    permutation_accuracies = []
    permutation_details = []

    print(f"  Starting RF permutation test with {N_PERMUTATIONS} permutations")
    print(f"  Using the first {N_CV_FOLDS} outer CV folds with shuffled labels")
    print('  Classification is restricted to the baseline window only')

    for perm_idx in range(N_PERMUTATIONS):
        if (perm_idx + 1) % PERMUTATION_PROGRESS_EVERY == 0:
            print(f"  Progress: {perm_idx + 1}/{N_PERMUTATIONS} permutations completed")

        rng = np.random.default_rng(perm_idx)
        shuffled_labels1 = rng.permutation(run1_labels)
        shuffled_labels2 = rng.permutation(run2_labels)

        perm_cv_accuracies = []
        perm_fold_details = []
        perm_pca_components = []
        perm_pca_cumvar = []
        perm_overlap_counts = []

        for fold_idx in range(N_CV_FOLDS):
            tr1, te1 = splits1[fold_idx]
            tr2, te2 = splits2[fold_idx]

            t_pre_s = cfg['t_pre'].to('s').magnitude
            t_post_s = cfg['t_post'].to('s').magnitude
            overlap_info = check_epoch_overlap(
                run1_full.stim, run2_full.stim,
                tr1, te1, tr2, te2,
                t_pre_s, t_post_s,
                verbose=(perm_idx == 0 and fold_idx == 0)
            )
            perm_overlap_counts.append(int(overlap_info['n_overlapping_pairs']))

            test_stim1 = run1_full.stim.iloc[te1]
            test_stim2 = run2_full.stim.iloc[te2]
            hrfL1, hrfR1, ss1, dmF1, dmZ1, Beta_ss_global1 = learn_channel_hrf(run1_full, test_stim1, cfg)
            hrfL2, hrfR2, ss2, dmF2, dmZ2, Beta_ss_global2 = learn_channel_hrf(run2_full, test_stim2, cfg)

            ts_tr1, ts_te1, lab_tr1, lab_te1 = extract_single_trial_ts_two_regressors(
                run_full=run1_full,
                hrfL=hrfL1,
                hrfR=hrfR1,
                ss_reg=ss1,
                dmF=dmF1,
                idx_tr=tr1,
                idx_te=te1,
                t_pre=cfg['t_pre'],
                t_post=cfg['t_post'],
                Beta_ss_global=Beta_ss_global1
            )
            ts_tr2, ts_te2, lab_tr2, lab_te2 = extract_single_trial_ts_two_regressors(
                run_full=run2_full,
                hrfL=hrfL2,
                hrfR=hrfR2,
                ss_reg=ss2,
                dmF=dmF2,
                idx_tr=tr2,
                idx_te=te2,
                t_pre=cfg['t_pre'],
                t_post=cfg['t_post'],
                Beta_ss_global=Beta_ss_global2
            )

            ts_train = np.concatenate([ts_tr1, ts_tr2], axis=0)
            ts_test = np.concatenate([ts_te1, ts_te2], axis=0)

            t_rel = np.linspace(
                -cfg['t_pre'].magnitude,
                cfg['t_post'].magnitude,
                hrfL1.sizes['time']
            )
            baseline_mask = t_rel < 0
            if not np.any(baseline_mask):
                raise RuntimeError('Baseline mask is empty; cannot run permutation test.')

            baseline_tr = ts_train[:, :, baseline_mask].mean(axis=2, keepdims=True)
            baseline_te = ts_test[:, :, baseline_mask].mean(axis=2, keepdims=True)
            ts_train = ts_train - baseline_tr
            ts_test = ts_test - baseline_te

            labels_train_bin = np.concatenate([shuffled_labels1[tr1], shuffled_labels2[tr2]])
            labels_test_bin = np.concatenate([shuffled_labels1[te1], shuffled_labels2[te2]])

            if perm_idx == 0 and fold_idx == 0:
                n_tr_left = int(np.sum(labels_train_bin == 0))
                n_tr_right = int(np.sum(labels_train_bin == 1))
                n_te_left = int(np.sum(labels_test_bin == 0))
                n_te_right = int(np.sum(labels_test_bin == 1))
                print(f"  Class balance (permuted fold 0): Train L={n_tr_left} R={n_tr_right} | Test L={n_te_left} R={n_te_right}")

            top_ch, comp_scores = good_channels(
                ts_train, labels_train_bin,
                N=DPRIME_TOP_N,
                windowStartT=2 * fs,
                windowEndT=10 * fs,
                fs=fs
            )

            ts_train_top = ts_train[:, top_ch, :]
            ts_test_top = ts_test[:, top_ch, :]

            n_trials_tr, n_top, tseg = ts_train_top.shape
            x_stack = ts_train_top.transpose(0, 2, 1).reshape(n_trials_tr * tseg, n_top)
            scaler_ch = StandardScaler()
            x_stack_z = scaler_ch.fit_transform(x_stack)
            pca = PCA(n_components=PCA_VAR_THRESHOLD, svd_solver='full', random_state=0)
            pca.fit(x_stack_z)
            kept_components = int(pca.n_components_)
            cum_var = float(pca.explained_variance_ratio_.sum())

            def project_trials(ts_arr):
                n_trials, _, tloc = ts_arr.shape
                out = []
                for tr in range(n_trials):
                    x = ts_arr[tr].T
                    xz = (x - scaler_ch.mean_) / scaler_ch.scale_
                    scores = xz @ pca.components_.T
                    out.append(scores.T)
                return np.stack(out, axis=0)

            ts_train_pc = project_trials(ts_train_top)
            ts_test_pc = project_trials(ts_test_top)

            k_used = int(min(kept_components, 10))
            used_pc_idx = np.arange(k_used, dtype=int)
            ts_train_used = ts_train_pc[:, used_pc_idx, :]
            ts_test_used = ts_test_pc[:, used_pc_idx, :]

            x_tr_baseline = ts_train_used[:, :, baseline_mask].mean(axis=2)
            x_te_baseline = ts_test_used[:, :, baseline_mask].mean(axis=2)

            rf_classifier = Pipeline([
                ('scaler', StandardScaler()),
                ('rf', RandomForestClassifier(
                    n_estimators=300,
                    max_depth=5,
                    min_samples_leaf=3,
                    class_weight='balanced',
                    oob_score=True,
                    random_state=42,
                    n_jobs=-1,
                ))
            ])
            rf_classifier.fit(x_tr_baseline, labels_train_bin)
            fold_acc = float(rf_classifier.score(x_te_baseline, labels_test_bin))
            perm_cv_accuracies.append(fold_acc)
            perm_pca_components.append(kept_components)
            perm_pca_cumvar.append(cum_var)

            valid_scores = comp_scores[comp_scores != -np.inf]
            perm_fold_details.append({
                'fold_idx': fold_idx,
                'train_size': int(len(labels_train_bin)),
                'test_size': int(len(labels_test_bin)),
                'accuracy': fold_acc,
                'n_overlapping_pairs': int(overlap_info['n_overlapping_pairs']),
                'selected_channel_indices': top_ch.tolist(),
                'selected_channel_labels': [str(common[i]) for i in top_ch],
                'mean_composite_score': float(np.mean(valid_scores)) if valid_scores.size else None,
                'mean_top_n_score': float(np.mean([comp_scores[i] for i in top_ch])) if len(top_ch) else None,
                'pca_components': kept_components,
                'pca_cumulative_variance': cum_var,
                'baseline_window_s': [float(t_rel[baseline_mask][0]), float(t_rel[baseline_mask][-1])],
            })

        perm_mean_accuracy = float(np.mean(perm_cv_accuracies))
        permutation_accuracies.append(perm_mean_accuracy)
        permutation_details.append({
            'permutation': perm_idx,
            'cv_accuracies': perm_cv_accuracies,
            'mean_accuracy': perm_mean_accuracy,
            'std_accuracy': float(np.std(perm_cv_accuracies)),
            'mean_pca_components': float(np.mean(perm_pca_components)),
            'mean_pca_cumulative_variance': float(np.mean(perm_pca_cumvar)),
            'mean_overlap_pairs': float(np.mean(perm_overlap_counts)),
            'fold_details': perm_fold_details,
        })

    permutation_accuracies = np.array(permutation_accuracies)
    summary_stats = {
        'mean_accuracy': float(permutation_accuracies.mean()),
        'std_accuracy': float(permutation_accuracies.std()),
        'percentile_95': float(np.percentile(permutation_accuracies, 95)),
        'percentile_99': float(np.percentile(permutation_accuracies, 99)),
        'min_accuracy': float(permutation_accuracies.min()),
        'max_accuracy': float(permutation_accuracies.max()),
    }

    print('\n  RF permutation test results:')
    print(f"  Mean accuracy across {N_PERMUTATIONS} permutations: {summary_stats['mean_accuracy']:.4f}")
    print(f"  Standard deviation: {summary_stats['std_accuracy']:.4f}")
    print(f"  95th percentile (chance level): {summary_stats['percentile_95']:.4f}")
    print(f"  99th percentile (chance level): {summary_stats['percentile_99']:.4f}")

    base = str(_PATHS.classifier_results_root / "permutation" / "rf_baseline_only")
    sub_folder = f'sub_{subj_id}_{flag_run_type}'
    out_dir = os.path.join(base, sub_folder)
    os.makedirs(out_dir, exist_ok=True)

    results_dict = {
        'permutation_accuracies': permutation_accuracies,
        'permutation_details': permutation_details,
        'config': {
            'subject_id': subj_id,
            'condition': flag_run_type,
            'n_permutations': N_PERMUTATIONS,
            'n_cv_folds': N_CV_FOLDS,
            'baseline_only': BASELINE_ONLY,
            'baseline_period': [float(-cfg['t_pre'].magnitude), 0.0],
            'dprime_top_n': int(DPRIME_TOP_N),
            'pca_variance_threshold': float(PCA_VAR_THRESHOLD),
            'rf_params': {
                'n_estimators': 300,
                'max_depth': 5,
                'min_samples_leaf': 3,
                'class_weight': 'balanced',
                'oob_score': True,
                'random_state': 42,
            },
        },
        'summary_stats': summary_stats,
    }

    with open(os.path.join(out_dir, 'permutation_test_rf_baseline.pkl'), 'wb') as f:
        pickle.dump(results_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    json_ready = {
        'config': results_dict['config'],
        'summary_stats': results_dict['summary_stats'],
        'permutation_accuracies': permutation_accuracies.tolist(),
    }
    with open(os.path.join(out_dir, 'permutation_summary.json'), 'w') as f:
        json.dump(json_ready, f, indent=2)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(permutation_accuracies, bins=30, color='steelblue', edgecolor='black', alpha=0.8)
    ax.axvline(summary_stats['mean_accuracy'], color='black', linestyle='--', linewidth=1.0, label='Mean')
    ax.axvline(summary_stats['percentile_95'], color='crimson', linestyle='--', linewidth=1.0, label='95th percentile')
    ax.axvline(summary_stats['percentile_99'], color='darkorange', linestyle='--', linewidth=1.0, label='99th percentile')
    ax.set_xlabel('Permutation accuracy')
    ax.set_ylabel('Count')
    ax.set_title(f'Sub {subj_id}: RF baseline-only permutation distribution')
    ax.legend(loc='upper left', fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'permutation_distribution.png'), dpi=300)
    plt.close(fig)

    print(f"  Saved permutation outputs to {out_dir}")
#%%



