#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Per-subject PC contributions on 2D scalp plots (Figure 5 panels A and B).

Visualizes per-channel PCA contributions from the classifier analysis on 2D
scalp plots using cedalion's plotting functions. For each subject and
condition: loads contributions from the classifier summary JSON, averages
across outer folds, attaches channel geometry from the subject SNIRF file,
and renders the scalp plot.

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
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr

# Cedalion imports (same as motion_inspector_slope.py)
from cedalion.io import snirf as snirf_io
from cedalion import plots
from cedalion import units

warnings.filterwarnings('ignore')


# Load contribution data
def load_subject_condition_contrib(base_dir: str, subject_id: str, condition: str):
    """
    Load overall channel importance per fold for a subject-condition.
    
    Returns
    -------
    folds : list[dict[str, float]]
        Each item is a dict: channel_name -> importance_pct
    meta : dict
        Metadata
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
                ch_name = str(entry.get('channel_name', 'NA'))
                imp = float(entry.get('importance_pct', 0.0))
                fold_map[ch_name] = fold_map.get(ch_name, 0.0) + imp
            folds.append(fold_map)
        
        meta = {
            'top_n_pcs': data.get('top_n_pcs_selected'),
            'pca_max_components': data.get('pca_max_components'),
            'n_outer_folds': len(folds),
        }
        
        print(f" Loaded {len(folds)} folds for sub-{subject_id} {condition}")
        return folds, meta
        
    except Exception as e:
        print(f" Error reading {json_path}: {e}")
        return [], {}


def get_channel_contrib_per_subject(fold_channel_maps):
    """
    Average channel contributions across folds (no ROI aggregation).
    
    Parameters
    ----------
    fold_channel_maps : list[dict]
        Per-fold dict of channel_name -> importance_pct
    
    Returns
    -------
    df : DataFrame with columns: channel_name, mean_importance, std_importance
    """
    if not fold_channel_maps:
        return pd.DataFrame()
    
    # Get all unique channel names
    all_channels = sorted({ch for fold in fold_channel_maps for ch in fold.keys()})
    
    rows = []
    for ch_name in all_channels:
        vals = [fold.get(ch_name, 0.0) for fold in fold_channel_maps]
        vals_arr = np.array(vals, dtype=float)
        
        rows.append({
            'channel_name': ch_name,
            'mean_importance': float(vals_arr.mean()) if vals_arr.size else 0.0,
            'std_importance': float(vals_arr.std(ddof=1)) if vals_arr.size > 1 else 0.0,
            'n_folds': int(len(vals_arr)),
        })
    
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values('mean_importance', ascending=False).reset_index(drop=True)
    
    return df


# Load SNIRF for geometry (EXACT COPY from motion_inspector_slope.py)
def load_snirf_for_geometry(subject_id: str, condition: str, master_data_dir: str):
    """
    Load SNIRF file to get rec and geo3d for plotting.
    Uses run-01 for geometry (same for all runs).
    
    Parameters
    ----------
    subject_id : str
        Subject ID (e.g., '01')
    condition : str
        'overt' or 'covert'
    master_data_dir : str
        Root directory containing subject folders
    
    Returns
    -------
    rec : cedalion Recording
        Recording with geometry information
    """
    # Construct SNIRF path
    snirf_filename = f"sub-{subject_id}_task-{condition}_run-01_nirs.snirf"
    snirf_path = os.path.join(master_data_dir, f"sub-{subject_id}", "nirs", snirf_filename)
    
    if not os.path.exists(snirf_path):
        print(f" SNIRF not found: {snirf_path}")
        return None
    
    print(f" Loading SNIRF for geometry: {snirf_filename}")
    
    try:
        # Load data using cedalion (same as motion_inspector)
        records = snirf_io.read_snirf(snirf_path)
        rec = records[0] if isinstance(records, list) else records
        
        # Check if geo3d exists
        if not hasattr(rec, 'geo3d') or rec.geo3d is None:
            print(f" Warning: No geo3d found in SNIRF file")
            return None
        
        return rec
        
    except Exception as e:
        print(f" Error loading SNIRF: {e}")
        return None


# Map contributions to channel structure
def map_contributions_to_channels(channel_contrib_df, rec):
    """
    Map channel contributions from JSON to SNIRF channel structure.
    
    Parameters
    ----------
    channel_contrib_df : DataFrame
        Columns: channel_name, mean_importance
    rec : cedalion Recording
        Recording with channel structure
    
    Returns
    -------
    contrib_da : xarray.DataArray
        Contribution values aligned with rec.channel structure
        Shape: (n_channels,) with channel coordinate
    """
    if channel_contrib_df.empty or rec is None:
        return None
    
    # Get channel structure from rec
    # Following the same pattern as motion_inspector_slope.py
    try:
        # Try to get channel structure from rec.timeseries (same as motion_inspector)
        if hasattr(rec, 'timeseries') and 'amp' in rec.timeseries:
            amp_data = rec.timeseries['amp']
            rec_channels = amp_data.channel
        elif hasattr(rec, 'amp_pruned') and rec.amp_pruned is not None:
            rec_channels = rec.amp_pruned.channel
        elif hasattr(rec, 'amp') and rec.amp is not None:
            rec_channels = rec.amp.channel
        else:
            print(" Could not find channel structure in rec")
            print(f"  Available attributes: {[attr for attr in dir(rec) if not attr.startswith('_')]}")
            return None
        
        n_channels = len(rec_channels)
        
        # Try to get channel labels
        channel_labels = []
        if hasattr(rec_channels, 'label'):
            # If channel has a label attribute
            channel_labels = rec_channels.label.values.tolist()
        elif hasattr(rec_channels, 'values'):
            # Try direct values
            channel_labels = [str(ch) for ch in rec_channels.values]
        else:
            # Fallback: iterate and extract
            for ch_idx in range(n_channels):
                ch_coord = rec_channels[ch_idx]
                if hasattr(ch_coord, 'label'):
                    ch_label = str(ch_coord.label.values)
                elif hasattr(ch_coord, 'values'):
                    ch_label = str(ch_coord.values)
                else:
                    ch_label = str(ch_coord)
                channel_labels.append(ch_label)
        
        # Map contributions to channel indices
        contrib_values = np.zeros(n_channels)
        
        for i, ch_label in enumerate(channel_labels):
            # Try exact match first
            match = channel_contrib_df[channel_contrib_df['channel_name'] == ch_label]
            if not match.empty:
                contrib_values[i] = match.iloc[0]['mean_importance']
            else:
                # Try partial match or alternative formats
                # Sometimes labels might have slight differences
                contrib_values[i] = 0.0  # Default to 0 if no match
        
        # Create xarray DataArray with ALL required coordinates
        # Need to include source and detector coordinates for scalp_plot to work
        if hasattr(rec, 'timeseries') and 'amp' in rec.timeseries:
            amp_data = rec.timeseries['amp']
            # Extract all relevant coordinates from the amp data
            coords_dict = {
                'channel': amp_data.channel,
                'source': amp_data.source,
                'detector': amp_data.detector,
            }
        else:
            # Fallback: just use channel
            coords_dict = {'channel': rec_channels}
        
        contrib_da = xr.DataArray(
            contrib_values,
            dims=["channel"],
            coords=coords_dict
        )
        
        n_matched = np.sum(contrib_values > 0)
        print(f" Matched {n_matched}/{len(channel_contrib_df)} channels to SNIRF structure")
        
        return contrib_da
        
    except Exception as e:
        print(f" Error mapping contributions: {e}")
        return None


# Plotting function (ADAPTED from motion_inspector_slope.py)
def plot_channel_contributions_scalp(contrib_da, rec, subject_id, condition, output_base_dir):
    """
    Plot channel contributions on 2D scalp plot.
    
    ADAPTED from plot_slope_diff_od_corrected() in motion_inspector_slope.py
    Key changes:
    - Use contribution data instead of slope
    - Single plot (no wavelength dimension)
    - Individual color scale per subject
    
    Parameters
    ----------
    contrib_da : xarray.DataArray
        Channel contribution values (shape: n_channels)
    rec : cedalion Recording
        Recording with geo3d for plotting
    subject_id : str
        Subject ID
    condition : str
        Condition ('overt' or 'covert')
    output_base_dir : str
        Base output directory for saving plots
    """
    import os
    import matplotlib.pyplot as plt
    import cedalion.plots as plots
    
    # Create condition-specific subdirectory
    output_dir = os.path.join(output_base_dir, condition)
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup figure - SINGLE plot (portrait layout works better for posters/papers)
    fig, ax = plt.subplots(1, 1, figsize=(9, 11))
    
    # Get contribution values
    contrib_vals = contrib_da.values
    
    # Get max and round up to a clean step so the top colorbar tick reaches the top color
    import math
    raw_max = np.nanmax(np.abs(contrib_vals))
    _step = 1 if raw_max <= 5 else (2 if raw_max <= 10 else (5 if raw_max <= 20 else 10))
    vmax_contrib = math.ceil(raw_max / _step) * _step
    
    # EXACT COPY of scalp_plot call from motion_inspector_slope.py
    # Only changes: data source and title
    plots.scalp_plot(
        contrib_da,
        rec.geo3d,  # Use geo3d from recording object
        contrib_da,
        ax,
        cmap='jet',
        vmin=0,  # Contributions are positive percentages
        vmax=vmax_contrib,
        optode_labels=False,
        title=f"Channel Contributions (PCA) - sub-{subject_id} {condition}",
        optode_size=2
    )
    
    # Set larger title font
    ax.set_title(f"Channel Contributions (PCA) - sub-{subject_id} {condition}", fontsize=22, fontweight='bold')
    
    # Increase colorbar font size and ensure ticks span full range including vmax
    contrib_ticks = np.linspace(0, vmax_contrib, num=min(6, vmax_contrib + 1))
    for extra_ax in [a for a in fig.axes if a is not ax]:
        extra_ax.tick_params(labelsize=22)
        extra_ax.set_ylabel('Contribution (%)', fontsize=24, fontweight='bold')
        extra_ax.set_yticks(contrib_ticks)
        # Remove the box outline around the colorbar
        for spine in extra_ax.spines.values():
            spine.set_visible(False)

    # Set overall title with larger font
    plt.suptitle(f'PCA Channel Contributions: sub-{subject_id} {condition}', fontsize=24, fontweight='bold', y=0.98)

    # Save figure
    base_name = f"sub-{subject_id}_{condition}_channel_contributions_scalp"
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    for ext in ('png', 'svg', 'pdf'):
        output_file = os.path.join(output_dir, f"{base_name}.{ext}")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f" Saved: {output_file}")
    plt.close()
    print(f"  Max contribution (raw): {raw_max:.2f}%  →  vmax set to {vmax_contrib}")


# ROI helper functions (from analyze_pc_contributions_group_roi.py)
def load_roi_mapping(roi_csv_path: str) -> dict:
    """Load ROI mapping CSV into dict channel_label -> brodmann."""
    try:
        roi_df = pd.read_csv(roi_csv_path)
        roi_dict = dict(zip(roi_df['channel_label'], roi_df['brodmann']))
        print(f" Loaded ROI mapping for {len(roi_dict)} channels")
        return roi_dict
    except Exception as e:
        print(f" Warning: Could not load ROI mapping from {roi_csv_path}: {e}")
        return {}


def get_display_name(channel_name: str, roi_mapping: dict) -> str:
    """Return ROI display name if available; otherwise the channel name."""
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


# Group-level ROI plotting
def map_roi_values_to_channels(rec, roi_mapping: dict, roi_values_df: pd.DataFrame, value_column: str):
    """
    Map ROI-level values to individual channels.
    All channels within the same ROI get the same value.
    
    Parameters
    ----------
    rec : cedalion Recording
        Recording with channel structure
    roi_mapping : dict
        channel_name -> brodmann mapping
    roi_values_df : DataFrame
        ROI-level data with columns: roi_name, <value_column>
    value_column : str
        Column name to extract values from (e.g., 'mean_importance', 'subject_frequency')
    
    Returns
    -------
    values_da : xarray.DataArray
        ROI values mapped to channel structure
    """
    try:
        # Get channel structure from rec
        if hasattr(rec, 'timeseries') and 'amp' in rec.timeseries:
            amp_data = rec.timeseries['amp']
            rec_channels = amp_data.channel
        else:
            print(" Could not find channel structure in rec")
            return None
        
        n_channels = len(rec_channels)
        
        # Extract channel labels
        if hasattr(rec_channels, 'values'):
            channel_labels = [str(ch) for ch in rec_channels.values]
        else:
            channel_labels = [str(rec_channels[i]) for i in range(n_channels)]
        
        # Create ROI lookup from dataframe
        roi_value_dict = dict(zip(roi_values_df['roi_name'], roi_values_df[value_column]))
        
        # Map ROI values to channels
        channel_values = np.zeros(n_channels)
        n_mapped = 0
        
        for i, ch_label in enumerate(channel_labels):
            # Get ROI for this channel
            roi_name = get_display_name(ch_label, roi_mapping)
            
            # Look up value for this ROI
            if roi_name in roi_value_dict:
                channel_values[i] = roi_value_dict[roi_name]
                n_mapped += 1
            else:
                channel_values[i] = 0.0  # Default for unmapped channels
        
        print(f"  Mapped {n_mapped}/{n_channels} channels to ROI values")
        
        # Create xarray with required coordinates
        if hasattr(rec, 'timeseries') and 'amp' in rec.timeseries:
            amp_data = rec.timeseries['amp']
            coords_dict = {
                'channel': amp_data.channel,
                'source': amp_data.source,
                'detector': amp_data.detector,
            }
        else:
            coords_dict = {'channel': rec_channels}
        
        values_da = xr.DataArray(
            channel_values,
            dims=["channel"],
            coords=coords_dict
        )
        
        return values_da
        
    except Exception as e:
        print(f" Error mapping ROI values to channels: {e}")
        import traceback
        traceback.print_exc()
        return None


def plot_group_roi_contributions_scalp(roi_group_df: pd.DataFrame, rec, condition: str, 
                                      roi_mapping: dict, output_base_dir: str):
    """
    Plot group-level ROI contributions on scalp plot.
    Two side-by-side plots: mean contribution and subject frequency.
    
    Parameters
    ----------
    roi_group_df : DataFrame
        Group-level ROI data with columns: roi_name, mean_importance, subject_frequency
    rec : cedalion Recording
        Recording with geo3d for plotting
    condition : str
        Condition ('overt' or 'covert')
    roi_mapping : dict
        channel_name -> brodmann mapping
    output_base_dir : str
        Base output directory for saving plots
    """
    import os
    import matplotlib.pyplot as plt
    import cedalion.plots as plots
    
    print(f"\n{'='*60}")
    print(f"Creating GROUP-LEVEL scalp plot for {condition}")
    print(f"{'='*60}")
    
    # Create condition-specific subdirectory
    output_dir = os.path.join(output_base_dir, condition)
    os.makedirs(output_dir, exist_ok=True)
    
    # Map ROI values to channels for both metrics
    print("  Mapping mean contributions to channels...")
    contrib_da = map_roi_values_to_channels(rec, roi_mapping, roi_group_df, 'mean_importance')
    
    print("  Mapping subject frequencies to channels...")
    freq_da = map_roi_values_to_channels(rec, roi_mapping, roi_group_df, 'subject_frequency')
    
    if contrib_da is None or freq_da is None:
        print(" Could not create group plot - mapping failed")
        return
    
    # Setup figure - horizontal layout (1 row, 2 cols)
    # Designed at 2× the 88mm target (176 × 96 mm) — scale to 88mm wide in Illustrator.
    # All element sizes (optodes, linewidths, fonts) are 2× final desired values so they
    # scale correctly when the SVG/PDF is halved in Illustrator.
    _w_in = 176 / 25.4   # 176 mm -> inches  (2× 88 mm)
    _h_in = 96 / 25.4    # 96 mm  -> inches
    fig, axes = plt.subplots(1, 2, figsize=(_w_in, _h_in))

    # Get color scale limits
    import math
    raw_max_contrib = np.nanmax(np.abs(contrib_da.values))
    # Round vmax up to a "nice" step so the top colorbar tick reaches the top color.
    # AutoLocator stops below vmax when vmax is not a clean multiple, leaving the
    # hottest color unlabeled. Rounding up ensures the top tick == vmax.
    _step = 1 if raw_max_contrib <= 5 else (2 if raw_max_contrib <= 10 else (5 if raw_max_contrib <= 20 else 10))
    vmax_contrib = math.ceil(raw_max_contrib / _step) * _step

    # Plot 1: Mean ROI Contribution — 'jet' (blue=low, red=high)
    plots.scalp_plot(
        contrib_da,
        rec.geo3d,
        contrib_da,
        axes[0],
        cmap='jet',
        vmin=0,
        vmax=vmax_contrib,
        optode_labels=False,
        title='',
        optode_size=2,
        channel_lw=1.0
    )
    axes[0].set_title('')

    # Plot 2: Subject Frequency — keep 'viridis' (perceptually uniform, visually
    #   distinct from 'hot', and familiar for frequency/proportion data).
    # Convert frequency from 0-1 to 0-100% for better visualization
    freq_da_pct = freq_da * 100.0
    plots.scalp_plot(
        freq_da_pct,
        rec.geo3d,
        freq_da_pct,
        axes[1],
        cmap='viridis',
        vmin=0,
        vmax=100,
        optode_labels=False,
        title='',
        optode_size=2,
        channel_lw=1.0
    )
    axes[1].set_title('')

    # Colorbar fonts at 2× final size (14pt label → 7pt at 88mm, 12pt ticks → 6pt at 88mm)
    # Also explicitly set ticks so top tick == vmax (guaranteed, even if step rounding
    # left a gap due to floating-point or unusual vmax values).
    plot_axes = set(np.ravel(axes))
    extra_axes = [a for a in fig.axes if a not in plot_axes]
    contrib_ticks = np.linspace(0, vmax_contrib, num=min(6, vmax_contrib + 1))
    freq_ticks = [0, 25, 50, 75, 100]
    tick_sets = [contrib_ticks, freq_ticks]
    for i, extra_ax in enumerate(extra_axes):
        extra_ax.tick_params(labelsize=12)
        extra_ax.set_ylabel(extra_ax.get_ylabel(), fontsize=14, fontweight='bold')
        for spine in extra_ax.spines.values():
            spine.set_visible(False)
        if i < len(tick_sets):
            extra_ax.set_yticks(tick_sets[i])

    # Save figure (no suptitle)
    base_name = f"group_{condition}_roi_contributions_scalp"
    plt.tight_layout()
    for ext in ('png', 'svg', 'pdf'):
        output_file = os.path.join(output_dir, f"{base_name}.{ext}")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f" Saved group plot: {output_file}")
    plt.close()
    print(f"  Max ROI contribution (raw): {raw_max_contrib:.2f}%  →  vmax set to {vmax_contrib}")
    print(f"  ROIs plotted: {len(roi_group_df)}")


# Main execution
def main():
    """Main function to generate scalp plots for all subjects and conditions."""
    
    # Configuration
    base_dir = r"U:\eng_research_hrc_binauralhearinglab\Sudan\Labs\Sen Lab\Research_projects\Whole_Head_Cocktail_party\Classifier_script_results\nested\rf_snr_0_20feat_balanced_depth5_oob"
    master_data_dir = r"U:\eng_research_hrc_binauralhearinglab\Sudan\Labs\Sen Lab\Research_projects\Whole_Head_Cocktail_party\Cocktail_party_whole_head_master_data"
    output_base_dir = r"U:\eng_research_hrc_binauralhearinglab\Sudan\Labs\Sen Lab\Research_projects\Whole_Head_Cocktail_party\Classifier_script_results\RF_above_chance_scalp_plots"
    roi_csv_path = r"U:\eng_research_hrc_binauralhearinglab\Sudan\Labs\Sen Lab\Research_projects\Whole_Head_Cocktail_party\ROIs\roi_master.csv"
    
    # Path to group-level ROI analysis results (from analyze_pc_contributions_group_roi.py)
    group_roi_base_dir = r"U:\eng_research_hrc_binauralhearinglab\Sudan\Labs\Sen Lab\Research_projects\Whole_Head_Cocktail_party\Classifier_script_results\RF_above_chance_group_roi_contributions"
    
    subjects = ['01','02','03','04','05','10','11','12','13','14','15','18','20','22',
                '25','28','30','31','32','33','34','35','39','41','44','47']
    conditions = ['overt', 'covert']
    
    print("="*80)
    print("PLOTTING PCA CHANNEL CONTRIBUTIONS ON SCALP PLOTS")
    print("="*80)

    # Load SNIRF geometry once — same cap layout for all subjects and conditions
    print(f"\nLoading shared geometry from sub-{subjects[0]} {conditions[0]}...")
    rec = load_snirf_for_geometry(subjects[0], conditions[0], master_data_dir)
    if rec is None:
        print(" Could not load reference SNIRF. Aborting.")
        return

    total_plots = 0
    failed_plots = 0

    # PART 1: Individual subject plots
    for condition in conditions:
        print(f"\n{'='*80}")
        print(f"CONDITION: {condition.upper()}")
        print(f"{'='*80}")

        for subject_id in subjects:
            print(f"\n--- Subject {subject_id} ---")

            try:
                # 1. Load contribution data
                folds, meta = load_subject_condition_contrib(base_dir, subject_id, condition)
                if not folds:
                    print(f" Skipping sub-{subject_id} {condition}: No contribution data")
                    failed_plots += 1
                    continue

                # 2. Average across folds
                channel_contrib_df = get_channel_contrib_per_subject(folds)
                if channel_contrib_df.empty:
                    print(f" Skipping sub-{subject_id} {condition}: No channel data")
                    failed_plots += 1
                    continue

                print(f"  Channels with contributions: {len(channel_contrib_df)}")
                print(f"  Top channel: {channel_contrib_df.iloc[0]['channel_name']} "
                      f"({channel_contrib_df.iloc[0]['mean_importance']:.2f}%)")

                # 3. Map contributions to channel structure (reuse shared rec)
                contrib_da = map_contributions_to_channels(channel_contrib_df, rec)
                if contrib_da is None:
                    print(f" Skipping sub-{subject_id} {condition}: Could not map channels")
                    failed_plots += 1
                    continue

                # 4. Plot and save (individual subject plots disabled)
                # plot_channel_contributions_scalp(contrib_da, rec, subject_id, condition, output_base_dir)
                total_plots += 1

            except Exception as e:
                print(f" Error processing sub-{subject_id} {condition}: {e}")
                import traceback
                traceback.print_exc()
                failed_plots += 1
                continue

    # PART 2: Group-level ROI plots
    print("\n" + "="*80)
    print("GENERATING GROUP-LEVEL ROI PLOTS")
    print("="*80)

    # Load ROI mapping
    roi_mapping = load_roi_mapping(roi_csv_path)
    if not roi_mapping:
        print(" Warning: No ROI mapping loaded, skipping group plots")
    else:
        for condition in conditions:
            try:
                print(f"\n--- Group plot for {condition} ---")

                # Load group-level ROI summary from analyze_pc_contributions_group_roi.py output
                group_roi_csv = os.path.join(group_roi_base_dir, f"group_{condition}_roi_contrib",
                                            'roi_contrib_group_summary.csv')

                if not os.path.exists(group_roi_csv):
                    print(f" Group ROI CSV not found: {group_roi_csv}")
                    print(f"   Please run analyze_pc_contributions_group_roi.py first")
                    continue

                # Load group ROI data
                roi_group_df = pd.read_csv(group_roi_csv)
                print(f" Loaded group ROI data: {len(roi_group_df)} ROIs")

                # Generate group plot (reuse shared rec for geometry)
                plot_group_roi_contributions_scalp(roi_group_df, rec, condition,
                                                  roi_mapping, output_base_dir)
                total_plots += 1

            except Exception as e:
                print(f" Error creating group plot for {condition}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total plots generated: {total_plots}")
    print(f"Failed plots: {failed_plots}")
    print(f"Output directory: {output_base_dir}")
    print("\n Done!")


if __name__ == '__main__':
    import sys
    
    # Check if running in test mode
    if len(sys.argv) > 1 and sys.argv[1] == '--test':
        # Test with just one subject
        print("="*80)
        print("TEST MODE: Running single subject")
        print("="*80)
        
        base_dir = r"U:\eng_research_hrc_binauralhearinglab\Sudan\Labs\Sen Lab\Research_projects\Whole_Head_Cocktail_party\Classifier_script_results\nested\rf_snr_0_20feat_balanced_depth5_oob"
        master_data_dir = r"U:\eng_research_hrc_binauralhearinglab\Sudan\Labs\Sen Lab\Research_projects\Whole_Head_Cocktail_party\Cocktail_party_whole_head_master_data"
        output_base_dir = r"U:\eng_research_hrc_binauralhearinglab\Sudan\Labs\Sen Lab\Research_projects\Whole_Head_Cocktail_party\Classifier_script_results\RF_above_chance_scalp_plots"
        
        subject_id = '01'
        condition = 'overt'
        
        print(f"\nTesting with sub-{subject_id} {condition}")
        
        # 1. Load contribution data
        folds, meta = load_subject_condition_contrib(base_dir, subject_id, condition)
        if not folds:
            print("ERROR: Could not load contribution data")
            sys.exit(1)
        
        # 2. Average across folds
        channel_contrib_df = get_channel_contrib_per_subject(folds)
        if channel_contrib_df.empty:
            print("ERROR: No channel data")
            sys.exit(1)
        
        print(f"Channels with contributions: {len(channel_contrib_df)}")
        print(f"Top channel: {channel_contrib_df.iloc[0]['channel_name']} "
              f"({channel_contrib_df.iloc[0]['mean_importance']:.2f}%)")
        
        # 3. Load SNIRF for geometry
        rec = load_snirf_for_geometry(subject_id, condition, master_data_dir)
        if rec is None:
            print("ERROR: Could not load SNIRF")
            sys.exit(1)
        
        # 4. Map contributions to channel structure (with debugging)
        contrib_da = map_contributions_to_channels(channel_contrib_df, rec)
        if contrib_da is None:
            print("ERROR: Could not map channels")
            sys.exit(1)
        
        # 5. Plot and save
        plot_channel_contributions_scalp(contrib_da, rec, subject_id, condition, output_base_dir)
        
        print("\n Test completed successfully!")
        
    elif len(sys.argv) > 1 and sys.argv[1] == '--test-group':
        # Test group-level plot only
        print("="*80)
        print("TEST MODE: Group-level plot only")
        print("="*80)
        
        master_data_dir = r"U:\eng_research_hrc_binauralhearinglab\Sudan\Labs\Sen Lab\Research_projects\Whole_Head_Cocktail_party\Cocktail_party_whole_head_master_data"
        output_base_dir = r"U:\eng_research_hrc_binauralhearinglab\Sudan\Labs\Sen Lab\Research_projects\Whole_Head_Cocktail_party\Classifier_script_results\RF_above_chance_scalp_plots"
        roi_csv_path = r"U:\eng_research_hrc_binauralhearinglab\Sudan\Labs\Sen Lab\Research_projects\Whole_Head_Cocktail_party\ROIs\roi_master.csv"
        group_roi_base_dir = r"U:\eng_research_hrc_binauralhearinglab\Sudan\Labs\Sen Lab\Research_projects\Whole_Head_Cocktail_party\Classifier_script_results\RF_above_chance_group_roi_contributions"

        ref_subject = '01'
        condition = 'overt'
        
        print(f"\nTesting group plot for {condition}")
        
        # Load ROI mapping
        roi_mapping = load_roi_mapping(roi_csv_path)
        if not roi_mapping:
            print("ERROR: Could not load ROI mapping")
            sys.exit(1)
        
        # Load group ROI data
        group_roi_csv = os.path.join(group_roi_base_dir, f"group_{condition}_roi_contrib", 
                                    'roi_contrib_group_summary.csv')
        if not os.path.exists(group_roi_csv):
            print(f"ERROR: Group ROI CSV not found: {group_roi_csv}")
            sys.exit(1)
        
        roi_group_df = pd.read_csv(group_roi_csv)
        print(f" Loaded group ROI data: {len(roi_group_df)} ROIs")
        
        # Load reference SNIRF
        ref_rec = load_snirf_for_geometry(ref_subject, condition, master_data_dir)
        if ref_rec is None:
            print("ERROR: Could not load reference SNIRF")
            sys.exit(1)
        
        # Generate group plot
        plot_group_roi_contributions_scalp(roi_group_df, ref_rec, condition, 
                                          roi_mapping, output_base_dir)
        
        print("\n Group test completed successfully!")
        
    else:
        # Run full analysis
        main()
