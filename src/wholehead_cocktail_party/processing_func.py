#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 12:03:03 2025

@author: lcarlton
"""


import cedalion
import cedalion.datasets as datasets
import cedalion.imagereco.forward_model as fw
import cedalion.io as io
import cedalion.models.glm as glm
import cedalion.nirs as nirs
import cedalion.sigproc.frequency
from cedalion.sigproc import quality 
import pandas as pd
import xarray as xr
from cedalion import units
import cedalion.sigproc.motion_correct as motion
import numpy as np
import os.path
import pickle
import cedalion.xrutils as xrutils 


def prune_channels(rec, amp_thresh=[1e-3, 0.84]*units.V, sd_thresh=[0, 45]*units.mm, snr_thresh=5):
    
    amp_thresh_sat = [0.*units.V, amp_thresh[1]]
    amp_thresh_low = [amp_thresh[0], 1*units.V]

    _, sd_mask      = quality.sd_dist(rec['amp'], rec.geo3d, sd_thresh)
    _, amp_mask     = quality.mean_amp(rec['amp'], amp_thresh)
    _, amp_mask_sat = quality.mean_amp(rec['amp'], amp_thresh_sat)
    _, amp_mask_low = quality.mean_amp(rec['amp'], amp_thresh_low)
    snr_vals, snr_mask = quality.snr(rec['amp'], snr_thresh)

    rec['snr']      = snr_vals
    rec['sd_mask']  = sd_mask
    rec['amp_mask'] = amp_mask
    rec['snr_mask'] = snr_mask

    # ───> Include amp_mask_sat and amp_mask_low here <───
    masks = [sd_mask, amp_mask, amp_mask_sat, amp_mask_low, snr_mask]

    rec['amp_pruned'], drop_list = quality.prune_ch(rec['amp'], masks, "all")

    # ───> Add amp_mask_sat & amp_mask_low into good_ch_mask <───
    good_ch_mask = sd_mask & amp_mask & amp_mask_sat & amp_mask_low & snr_mask
    rec['good_ch_mask'] = good_ch_mask

    chs_pruned = xr.DataArray(
        np.zeros(rec['amp'].shape[0]),
        dims=["channel"],
        coords={"channel": rec['amp'].channel}
    )

    # i initialize chs_pruned to 0.4
    chs_pruned[:] = 0.4
    chs_pruned[~snr_mask[:, 0]]    = 0.19
    chs_pruned[~amp_mask_sat[:, 0]] = 0.00
    chs_pruned[~amp_mask_low[:, 0]] = 0.80

    return rec, chs_pruned


def median_filter(rec, median_filt = 3):
    pad_width = 1  # Adjust based on the kernel size
    padded_amp = rec['amp'].pad(time=(pad_width, pad_width), mode='edge')
    # Apply the median filter to the padded data
    filtered_padded_amp = padded_amp.rolling(time=median_filt, center=True).reduce(np.median)
    # Trim the padding after applying the filter
    rec['amp'] = filtered_padded_amp.isel(time=slice(pad_width, -pad_width))
    return rec    
    

def preprocess(rec, cfg_preprocess):
    """
    Clean preprocessing with consistent field naming convention:
    - Format: {type}_{pruning}_{processing}
    - type: od/conc
    - pruning: o (original) / p (pruned)  
    - processing: filt / tddr / tddr_filt / postglm
    
    Only creates TDDR fields when TDDR is actually enabled.
    """
    # Clean amplitude data
    rec['amp'] = rec['amp'].where( rec['amp']>0, 1e-18 )
    rec['amp'] = rec['amp'].where( ~rec['amp'].isnull(), 1e-18 )

    # if first value is 1e-18 then replace with second value
    indices = np.where(rec['amp'][:,0,0] == 1e-18)
    rec['amp'][indices[0],0,0] = rec['amp'][indices[0],0,1]
    indices = np.where(rec['amp'][:,1,0] == 1e-18)
    rec['amp'][indices[0],1,0] = rec['amp'][indices[0],1,1]
    
    rec['amp'] = rec['amp'].pint.dequantify().pint.quantify('V')

    # Apply median filter
    rec = median_filter(rec)
    
    # Prune channels
    rec, chs_pruned = prune_channels(rec, cfg_preprocess['cfg_prune']['amp_thresh'],
                                          cfg_preprocess['cfg_prune']['sd_thresh'],
                                          cfg_preprocess['cfg_prune']['snr_thresh'])
    
    # DPF for concentration conversion (when needed)
    dpf = xr.DataArray(
                        [1, 1],
                        dims="wavelength",
                        coords={"wavelength": rec["amp"].wavelength},
                        )
    
    # Check if TDDR is enabled
    do_tddr = cfg_preprocess["cfg_motion_correct"]["flag_do_tddr"]
    
    # ===== STEP 1: Convert to OD =====
    rec["od_o"] = cedalion.nirs.int2od(rec['amp'])
    rec["od_p"] = cedalion.nirs.int2od(rec['amp_pruned'])
    
    # Set time units
    rec['od_o'].time.attrs['units'] = units.s
    rec['od_p'].time.attrs['units'] = units.s
    
    # ===== STEP 2: Bandpass filter =====
    rec["od_o_filt"] = cedalion.sigproc.frequency.freq_filter(
        rec["od_o"],
        cfg_preprocess['cfg_bandpass']['fmin'],
        cfg_preprocess['cfg_bandpass']['fmax'])
    
    rec["od_p_filt"] = cedalion.sigproc.frequency.freq_filter(
        rec["od_p"],
        cfg_preprocess['cfg_bandpass']['fmin'],
        cfg_preprocess['cfg_bandpass']['fmax'])
    
    # ===== STEP 3: Apply TDDR (only if enabled) =====
    if do_tddr:
        # Apply TDDR to raw OD
        rec["od_o_tddr"] = motion.tddr(rec["od_o"])
        rec["od_p_tddr"] = motion.tddr(rec["od_p"])
        
        # Clean any remaining nulls
        rec['od_o_tddr'] = rec['od_o_tddr'].where( ~rec['od_o_tddr'].isnull(), 1e-18 )
        rec['od_p_tddr'] = rec['od_p_tddr'].where( ~rec['od_p_tddr'].isnull(), 1e-18 )
        
        # Bandpass filter after TDDR
        rec["od_o_tddr_filt"] = cedalion.sigproc.frequency.freq_filter(
            rec["od_o_tddr"],
            cfg_preprocess['cfg_bandpass']['fmin'],
            cfg_preprocess['cfg_bandpass']['fmax'])
            
        rec["od_p_tddr_filt"] = cedalion.sigproc.frequency.freq_filter(
            rec["od_p_tddr"],
            cfg_preprocess['cfg_bandpass']['fmin'],
            cfg_preprocess['cfg_bandpass']['fmax'])
        
        # Convert TDDR data to concentration
        rec['conc_o_tddr_filt'] = cedalion.nirs.od2conc(rec['od_o_tddr_filt'], rec.geo3d, dpf)
        rec['conc_p_tddr_filt'] = cedalion.nirs.od2conc(rec['od_p_tddr_filt'], rec.geo3d, dpf)
    
    # ===== STEP 4: Convert filtered data to concentration =====
    rec['conc_o_filt'] = cedalion.nirs.od2conc(rec['od_o_filt'], rec.geo3d, dpf)
    rec['conc_p_filt'] = cedalion.nirs.od2conc(rec['od_p_filt'], rec.geo3d, dpf)
    
    # ===== STEP 5: Apply GLM (if enabled) =====
    if cfg_preprocess['cfg_GLM'] is not None:
        if do_tddr:
            # Apply GLM to TDDR-processed concentration data
            rec = GLM(rec, 'conc_o_tddr_filt', rec.stim, cfg_preprocess['cfg_GLM'])
            rec = GLM(rec, 'conc_p_tddr_filt', rec.stim, cfg_preprocess['cfg_GLM'])
            
            # Convert GLM-processed concentration back to OD
            rec['od_o_tddr_filt_postglm'] = cedalion.nirs.conc2od(rec['conc_o_tddr_filt_postglm'], rec.geo3d, dpf)
            rec['od_p_tddr_filt_postglm'] = cedalion.nirs.conc2od(rec['conc_p_tddr_filt_postglm'], rec.geo3d, dpf)
        else:
            # Apply GLM to regular filtered concentration data (when TDDR is disabled)
            rec = GLM(rec, 'conc_o_filt', rec.stim, cfg_preprocess['cfg_GLM'])
            rec = GLM(rec, 'conc_p_filt', rec.stim, cfg_preprocess['cfg_GLM'])
            
            # Convert GLM-processed concentration back to OD
            rec['od_o_filt_postglm'] = cedalion.nirs.conc2od(rec['conc_o_filt_postglm'], rec.geo3d, dpf)
            rec['od_p_filt_postglm'] = cedalion.nirs.conc2od(rec['conc_p_filt_postglm'], rec.geo3d, dpf)
    
    # ===== STEP 6: Store concentration data in timeseries =====
    # Always store the non-TDDR filtered concentration data
    rec.timeseries['conc_o_filt'] = rec['conc_o_filt']
    rec.timeseries['conc_p_filt'] = rec['conc_p_filt']
    
    # Store GLM-processed non-TDDR data if GLM was applied and TDDR is disabled
    if cfg_preprocess['cfg_GLM'] is not None and not do_tddr:
        rec.timeseries['conc_o_filt_postglm'] = rec['conc_o_filt_postglm']
        rec.timeseries['conc_p_filt_postglm'] = rec['conc_p_filt_postglm']

    # Only store TDDR data if it was actually processed
    if do_tddr:
        rec.timeseries['conc_o_tddr_filt'] = rec['conc_o_tddr_filt']
        rec.timeseries['conc_p_tddr_filt'] = rec['conc_p_tddr_filt']
        
        # Only store GLM-processed TDDR data if GLM was actually applied to TDDR data
        if cfg_preprocess['cfg_GLM'] is not None:
            rec.timeseries['conc_o_tddr_filt_postglm'] = rec['conc_o_tddr_filt_postglm']
            rec.timeseries['conc_p_tddr_filt_postglm'] = rec['conc_p_tddr_filt_postglm']
    
    return rec, chs_pruned



def preprocess_batch(cfg_dataset, cfg_preprocess):
    
    # make sure derivatives folders exist
    der_dir = os.path.join(cfg_dataset['root_dir'], 'derivatives')
    if not os.path.exists(der_dir):
        os.makedirs(der_dir)
    der_dir = os.path.join(cfg_dataset['root_dir'], 'derivatives', 'plots')
    if not os.path.exists(der_dir):
        os.makedirs(der_dir)
    der_dir = os.path.join(cfg_dataset['root_dir'], 'derivatives', 'ica')
    if not os.path.exists(der_dir):
        os.makedirs(der_dir)
    der_dir = os.path.join(cfg_dataset['root_dir'], 'derivatives', 'processed_data')
    if not os.path.exists(der_dir):
        os.makedirs(der_dir)
    
    
    n_subjects = len(cfg_dataset['subj_ids'])
    n_files_per_subject = len(cfg_dataset['file_ids'])
    
    # loop over subjects and files
    for subj_idx in range(n_subjects):
        
        for file_idx in range(n_files_per_subject):
            
            filenm = cfg_dataset['filenm_lst'][subj_idx][file_idx]
            
            print( f"Loading {subj_idx+1} of {n_subjects} subjects, {file_idx+1} of {n_files_per_subject} files : {filenm}" )
           
            subStr = filenm.split('_')[0]
            subDir = os.path.join(cfg_dataset['root_dir'], subStr, 'nirs')
           
            file_path = os.path.join(subDir, filenm )
            records = cedalion.io.read_snirf( file_path ) 
           
            recTmp = records[0]
            stim_df = pd.read_csv( file_path[:-5] + '_events.tsv', sep='\t' )
            recTmp.stim = stim_df
            
            recTmp, chs_pruned = preprocess(recTmp, cfg_preprocess)
            

            if subj_idx == 0 and file_idx == 0:
                rec = []
                chs_pruned_subjs = []
  
            
                rec.append( [recTmp] )
                chs_pruned_subjs.append( [chs_pruned] )

            elif file_idx == 0:
                rec.append( [recTmp] )
                chs_pruned_subjs.append( [chs_pruned] )

            else:
                rec[subj_idx].append( recTmp )
                chs_pruned_subjs[subj_idx].append( chs_pruned )

    return rec, chs_pruned_subjs



def GLM(rec, timeseries, stim_list, cfg_GLM):
    """
    Apply GLM to remove artifacts and extract HRF responses.
    Uses pruned concentration data for short channel detection.
    """
    #### build design matrix
    # Use the appropriate pruned concentration data for short channel detection
    # This should match the same processing level as the timeseries being processed
    if 'tddr' in timeseries:
        conc_data = rec['conc_p_tddr_filt']
    else:
        conc_data = rec['conc_p_filt']
        
    ts_long, ts_short = cedalion.nirs.split_long_short_channels(
                            conc_data, rec.geo3d,
                            distance_threshold=cfg_GLM['distance_threshold']
                        )
                        
    # build regressors
    dm, channel_wise_regressors = glm.make_design_matrix(
        rec[timeseries],
        ts_short,
        rec.stim,
        rec.geo3d,
        basis_function=glm.GaussianKernels(cfg_GLM['t_pre'], cfg_GLM['t_post'], cfg_GLM['t_delta'], cfg_GLM['t_std']),
        drift_order=cfg_GLM['drift_order'],
        short_channel_method=cfg_GLM['short_channel_method']
    )
    
    #### fit the model
    betas = glm.fit(rec[timeseries], dm, channel_wise_regressors, noise_model=cfg_GLM['noise_model'])
    
    pred_all = glm.predict(rec[timeseries], betas, dm, channel_wise_regressors)
    pred_all = pred_all.pint.quantify('micromolar')
    
    residual = rec[timeseries] - pred_all
    
    # prediction of all HRF regressors, i.e. all regressors that start with 'HRF '
    pred_hrf = glm.predict(
                            rec[timeseries],
                            betas.sel(regressor=betas.regressor.str.startswith("HRF ")),
                            dm,
                            channel_wise_regressors
                        )
    
    pred_hrf = pred_hrf.pint.quantify('micromolar')

    rec[timeseries + '_postglm'] = pred_hrf + residual
    
    #### get average HRF prediction
    rec[timeseries + '_postglm'] = rec[timeseries + '_postglm'].transpose('chromo', 'channel', 'time')
    rec[timeseries + '_postglm'] = rec[timeseries + '_postglm'].assign_coords(samples=("time", np.arange(len(rec[timeseries + '_postglm'].time))))
    rec[timeseries + '_postglm']['time'] = rec[timeseries + '_postglm'].time.pint.quantify(units.s)
             
    return rec
    
    
    
#%% block avg funcs 
def block_average(rec, timeseries, stim_list, t_pre, t_post,subj_idx, file_idx):
    
    ts = rec[subj_idx][file_idx][timeseries].copy()

    stim = rec[subj_idx][file_idx].stim.copy()

    
    epochs = ts.cd.to_epochs(
                                stim,  # stimulus dataframe
                                stim_list,  # select events
                                before=t_pre,  # seconds before stimulus
                                after=t_post,  # seconds after stimulus
                            )
    
    baseline = epochs.sel(reltime=(epochs.reltime < 0)).mean("reltime")
    
    epochs_blcorrected = epochs - baseline    
    
    blockaverage = epochs_blcorrected.groupby('trial_type').mean('epoch')
                                            
    sources = [ch.split("D")[0] for ch in blockaverage.channel.values]  # Extract 'S#'
    detectors = ["D" + ch.split("D")[1] for ch in blockaverage.channel.values]  # Extract 'D#'

    # Assign new coordinates
    blockaverage = blockaverage.assign_coords(source=("channel", sources), detector=("channel", detectors))
  
    return epochs_blcorrected, blockaverage
    


def get_group_average(rec, timeseries , chs_pruned_subjs, cfg_dataset, cfg_blockavg):

    # choose correct mse values 
    if 'chromo' in rec[0][0][timeseries].dims:
        cfg_mse = cfg_blockavg['cfg_mse_conc']
    else:
        cfg_mse = cfg_blockavg['cfg_mse_od']
        
    n_subjects  = len(rec)
    n_files_per_subject = len(rec[0])
    all_trial_blockaverage = None
    
    for trial_type in cfg_blockavg['stim_lst_hrf']:
        
        all_subj_blockaverage = None

        for subj_idx in range( n_subjects ):
            
            for file_idx in range( n_files_per_subject ):
    
                filenm = cfg_dataset['filenm_lst'][subj_idx][file_idx]
                print( f"   Running {subj_idx+1} of {n_subjects} subjects : {filenm}" )
    
                # do the block average on the data in rec[subj_idx][file_idx][rec_str]            
                ts = rec[subj_idx][file_idx][timeseries].copy()
                
                # select the stim for the given file
                stim = rec[subj_idx][file_idx].stim.copy()
                
                epochs_tmp = ts.cd.to_epochs(
                                            stim,  # stimulus dataframe
                                            [trial_type], # select events
                                            before=cfg_blockavg['trange_hrf'][0],  # seconds before stimulus
                                            after=cfg_blockavg['trange_hrf'][1],  # seconds after stimulus
                                        )
                
            
                # epochs_tmp = epochs_tmp.assign_coords(trial_type=('epoch', [x + '-' + cfg_dataset['subj_ids'][subj_idx] for x in epochs_tmp.trial_type.values]))
    
                if file_idx == 0:
                    epochs_all = epochs_tmp
                else:
                    epochs_all = xr.concat([epochs_all, epochs_tmp], dim='epoch')
    
            # END OF LOOP OVER FILES
            
            # baseline correct and then get the block average across all epochs and runs for that subject
            baseline = epochs_all.sel(reltime=(epochs_all.reltime < 0)).mean('reltime')
            epochs = epochs_all - baseline
            subj_blockaverage = epochs.groupby('trial_type').mean('epoch')
            
            subj_blockaverage_weighted = subj_blockaverage.copy()
            n_epochs = len(epochs.epoch)
            n_chs = len(epochs.channel)
            
                
            # de-mean the epochs
            epochs_zeromean = epochs - subj_blockaverage
        
            if 'chromo' in ts.dims:
                epochs_zeromean = epochs_zeromean.stack(measurement=['channel','chromo']).sortby('chromo')
            else:
                epochs_zeromean = epochs_zeromean.stack(measurement=['channel','wavelength']).sortby('wavelength')
            epochs_zeromean = epochs_zeromean.transpose('trial_type', 'measurement', 'reltime', 'epoch')
            mse_t = (epochs_zeromean**2).sum('epoch') / (n_epochs - 1)**2 # this is squared to get variance of the mean, aka MSE of the mean

            # set bad values in mse_t to the bad value threshold
            amp = rec[subj_idx][file_idx]['amp'].mean('time').min('wavelength') # take the minimum across wavelengths
            idx_amp = np.where(amp < cfg_mse['mse_amp_thresh'])[0]
            idx_sat = np.where(chs_pruned_subjs[subj_idx][file_idx] == 0.0)[0]
            idx_bad = np.where(mse_t == 0)[0]
            idx_bad1 = idx_bad[idx_bad<n_chs]
            idx_bad2 = idx_bad[idx_bad>=n_chs] - n_chs
            
            mse_t[:,idx_amp,:] = cfg_mse['mse_val_for_bad_data']
            mse_t[:,idx_amp+n_chs,:] = cfg_mse['mse_val_for_bad_data']
            mse_t[:,idx_sat,:] = cfg_mse['mse_val_for_bad_data']
            mse_t[:,idx_sat+n_chs,:] = cfg_mse['mse_val_for_bad_data']
            mse_t[:,idx_bad,:] = cfg_mse['mse_val_for_bad_data']
            
            channels = subj_blockaverage_weighted.channel
            subj_blockaverage_weighted.loc[trial_type, :, channels.isel(channel=idx_amp),:] = cfg_mse['blockaverage_val']
            subj_blockaverage_weighted.loc[trial_type, :, channels.isel(channel=idx_sat),:] = cfg_mse['blockaverage_val']
            subj_blockaverage_weighted.loc[trial_type, :, channels.isel(channel=idx_bad1), :] = cfg_mse['blockaverage_val']
            subj_blockaverage_weighted.loc[trial_type, :, channels.isel(channel=idx_bad2),:] = cfg_mse['blockaverage_val']
            
            # set the minimum value of mse_t
            if 'chromo' in epochs.dims:
                mse_t = mse_t.unstack('measurement').transpose('trial_type', 'chromo','channel','reltime')
            else:
                mse_t = mse_t.unstack('measurement').transpose('trial_type','wavelength','channel','reltime')

            source_coord = subj_blockaverage_weighted['source']
            mse_t = mse_t.assign_coords(source=('channel',source_coord.data))
            detector_coord = subj_blockaverage_weighted['detector']
            mse_t = mse_t.assign_coords(detector=('channel',detector_coord.data))
            
            mse_t_o = mse_t.copy()
            mse_t = xr.where(mse_t < cfg_mse['mse_min_thresh'], cfg_mse['mse_min_thresh'], mse_t)

            # gather the blockaverage across subjects
            if all_subj_blockaverage is None and cfg_dataset['subj_ids'][subj_idx] not in cfg_dataset['subj_id_exclude']:
                                
                all_subj_blockaverage_weighted = subj_blockaverage_weighted / mse_t

                subjavg_blockaverage_weighted = subj_blockaverage_weighted / mse_t
                sum_mse_inv = 1/mse_t
                
                # add a subject dimension and coordinate
                all_subj_blockaverage = subj_blockaverage.expand_dims('subj')
                all_subj_blockaverage = all_subj_blockaverage.assign_coords(subj=[cfg_dataset['subj_ids'][subj_idx]])
     
                all_subj_mse = mse_t_o.expand_dims('subj') 
                all_subj_mse = all_subj_mse.assign_coords(subj=[cfg_dataset['subj_ids'][subj_idx]])
                
            elif cfg_dataset['subj_ids'][subj_idx] not in cfg_dataset['subj_id_exclude']:
                                
                all_subj_blockaverage_weighted = xr.concat([all_subj_blockaverage_weighted, subj_blockaverage_weighted / mse_t], dim='subj')
                
                subjavg_blockaverage_weighted = subjavg_blockaverage_weighted + subj_blockaverage_weighted  / mse_t
                sum_mse_inv = sum_mse_inv + 1 / mse_t
                
                subj_blockaverage = subj_blockaverage.expand_dims('subj')
                subj_blockaverage = subj_blockaverage.assign_coords(subj=[cfg_dataset['subj_ids'][subj_idx]])

                all_subj_blockaverage = xr.concat([all_subj_blockaverage, subj_blockaverage], dim='subj')
    
                mse_subj_tmp = mse_t_o.expand_dims('subj')
                mse_subj_tmp = mse_subj_tmp.assign_coords(subj=[cfg_dataset['subj_ids'][subj_idx]])

                all_subj_mse = xr.concat([all_subj_mse, mse_subj_tmp], dim='subj')
    
            else:
                print(f"   Subject {cfg_dataset['subj_ids'][subj_idx]} excluded from group average")
            
        # DONE LOOP OVER SUBJECTS
            
        # get the unweighted average
        subjavg_blockaverage = all_subj_blockaverage.mean('subj')
        
        # get the weighted average
        subjavg_blockaverage_weighted = (subjavg_blockaverage_weighted / sum_mse_inv).drop_vars('subj', errors='ignore')
    
        # get the mean mse within subjects
        mse_mean_within_subject = 1 / sum_mse_inv
        
        blockaverage_mse_subj_tmp = all_subj_mse.copy()
        blockaverage_mse_subj_tmp = xr.where(blockaverage_mse_subj_tmp < cfg_mse['mse_min_thresh'], cfg_mse['mse_min_thresh'], blockaverage_mse_subj_tmp)
    
        mse_weighted_between_subjects_tmp = (all_subj_blockaverage - subjavg_blockaverage_weighted)**2 / blockaverage_mse_subj_tmp
        mse_weighted_between_subjects = mse_weighted_between_subjects_tmp.mean('subj')
        mse_weighted_between_subjects = mse_weighted_between_subjects * mse_mean_within_subject
        # FIXME: is it an issue that mse_mean_within_subject comes from mse_t and blockaverage_mse_subj_tmp comes from mse_t_o?
     
        # blockaverage_stderr_weighted = np.sqrt(1 / blockaverage_mse_inv_mean_weighted)
        total_stderr_blockaverage = np.sqrt( mse_mean_within_subject + mse_weighted_between_subjects )
        total_stderr_blockaverage = total_stderr_blockaverage.assign_coords(trial_type=subjavg_blockaverage_weighted.trial_type)

        if all_trial_blockaverage is None:
            
            all_trial_blockaverage = subjavg_blockaverage
            all_trial_blockaverage_weighted = subjavg_blockaverage_weighted
            all_trial_total_stderr = total_stderr_blockaverage
            
            all_trial_all_subj_blockaverage = all_subj_blockaverage
            all_trial_all_subj_blockaverage_weighted = all_subj_blockaverage_weighted
            all_trial_all_subj_mse = all_subj_mse 
            
        else:

            all_trial_blockaverage = xr.concat([all_trial_blockaverage, subjavg_blockaverage], dim='trial_type')
            all_trial_blockaverage_weighted = xr.concat([all_trial_blockaverage_weighted, subjavg_blockaverage_weighted], dim='trial_type')
            all_trial_total_stderr = xr.concat([all_trial_total_stderr, total_stderr_blockaverage], dim='trial_type')
            
            all_trial_all_subj_blockaverage = xr.concat([all_trial_all_subj_blockaverage, all_subj_blockaverage], dim='trial_type')
            all_trial_all_subj_blockaverage_weighted = xr.concat([all_trial_all_subj_blockaverage_weighted, all_subj_blockaverage_weighted], dim='trial_type')
            all_trial_all_subj_mse = xr.concat([all_trial_all_subj_mse, all_subj_mse], dim='trial_type')
    # DONE LOOP OVER TRIAL_TYPES

    
    return all_trial_blockaverage, all_trial_all_subj_blockaverage, all_trial_blockaverage_weighted, all_trial_all_subj_blockaverage_weighted, all_trial_total_stderr, all_trial_all_subj_mse


