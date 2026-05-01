#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2D scalp plot with Angular Gyrus ROI highlight (Figure 2).

Renders a 2D scalp projection of the cedalion probe geometry with the
Angular Gyrus channels highlighted in green. Used as the inset / overlay
companion to the hemodynamic response grid.

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
# import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr

from cedalion.io import snirf as snirf_io
from cedalion import plots
from cedalion import units

warnings.filterwarnings('ignore')


# # === Load contribution data ===
# def load_subject_condition_contrib(base_dir, subject_id, condition):
#     ...
#
# def get_channel_contrib_per_subject(fold_channel_maps):
#     ...
#
# def load_snirf_for_geometry — kept below
#
# def map_contributions_to_channels(channel_contrib_df, rec):
#     ...
#
# def plot_channel_contributions_scalp(contrib_da, rec, subject_id, condition, output_base_dir):
#     ...
#
# def map_roi_values_to_channels(rec, roi_mapping, roi_values_df, value_column):
#     ...
#
# def plot_group_roi_contributions_scalp(roi_group_df, rec, condition, roi_mapping, output_base_dir):
#     ...


# Load SNIRF for geometry
def load_snirf_for_geometry(subject_id: str, condition: str, master_data_dir: str):
    """
    Load SNIRF file to get rec and geo3d for plotting.
    Uses run-01 for geometry (same for all runs).
    """
    snirf_filename = f"sub-{subject_id}_task-{condition}_run-01_nirs.snirf"
    snirf_path = os.path.join(master_data_dir, f"sub-{subject_id}", "nirs", snirf_filename)

    if not os.path.exists(snirf_path):
        print(f" SNIRF not found: {snirf_path}")
        return None

    print(f" Loading SNIRF for geometry: {snirf_filename}")

    try:
        records = snirf_io.read_snirf(snirf_path)
        rec = records[0] if isinstance(records, list) else records

        if not hasattr(rec, 'geo3d') or rec.geo3d is None:
            print(f" Warning: No geo3d found in SNIRF file")
            return None

        return rec

    except Exception as e:
        print(f" Error loading SNIRF: {e}")
        return None


# ROI helper functions
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


# Single ROI highlight plot
def plot_single_roi_highlight(rec, roi_mapping: dict, roi_name: str, condition: str,
                              output_base_dir: str, highlight_color='black',
                              output_filename: str | None = None):
    """
    Plot a scalp map with one ROI highlighted in a chosen color.
    All other channels are rendered in grey.
    """
    import matplotlib.colors as mcolors
    import matplotlib.patches as mpatches
    import matplotlib

    if not (hasattr(rec, 'timeseries') and 'amp' in rec.timeseries):
        print(" Could not find amp timeseries in rec")
        return
    amp_data = rec.timeseries['amp']
    rec_channels = amp_data.channel
    n_channels = len(rec_channels)
    channel_labels = [str(ch) for ch in rec_channels.values]

    metric_values = np.full(n_channels, np.nan)
    n_matched = 0
    for i, ch_label in enumerate(channel_labels):
        if get_display_name(ch_label, roi_mapping) == roi_name:
            metric_values[i] = 1.0
            n_matched += 1

    print(f" Highlighted {n_matched} channels for ROI: {roi_name}")
    if n_matched == 0:
        print(f" No channels matched ROI '{roi_name}' — check spelling against roi_mapping values")
        return

    coords_dict = {
        'channel': amp_data.channel,
        'source': amp_data.source,
        'detector': amp_data.detector,
    }
    metric_da = xr.DataArray(metric_values, dims=["channel"], coords=coords_dict)

    highlight_cmap = mcolors.ListedColormap([highlight_color])
    highlight_norm = matplotlib.colors.Normalize(vmin=0, vmax=1)

    fig, ax = plt.subplots(1, 1, figsize=(9, 11))

    plots.scalp_plot(
        metric_da,
        rec.geo3d,
        metric_da,
        ax,
        cmap=highlight_cmap,
        norm=highlight_norm,
        vmin=0,
        vmax=1,
        bad_color=[0.75, 0.75, 0.75],
        optode_labels=False,
        optode_size=6,
        add_colorbar=False,
    )

    # legend_patch = mpatches.Patch(color=highlight_color, label=roi_name)
    # ax.legend(handles=[legend_patch], loc='upper right', fontsize=14)

    if output_filename is None:
        safe_name = roi_name.replace(' ', '_').replace('/', '-')
        output_filename = f"roi_highlight_{safe_name}_{condition}"

    output_dir = os.path.join(output_base_dir, condition)
    os.makedirs(output_dir, exist_ok=True)

    plt.tight_layout()
    for ext in ('png', 'svg', 'pdf'):
        output_file = os.path.join(output_dir, f"{output_filename}.{ext}")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f" Saved: {output_file}")
    plt.close()


# Execution
#%%
master_data_dir = r"U:\eng_research_hrc_binauralhearinglab\Sudan\Labs\Sen Lab\Research_projects\Whole_Head_Cocktail_party\Cocktail_party_whole_head_master_data"
output_base_dir = r"U:\eng_research_hrc_binauralhearinglab\Sudan\Labs\Sen Lab\Research_projects\Whole_Head_Cocktail_party\Classifier_script_results\RF_SNR_0_20featall_ch_PC_contribution_scalp_plots"
roi_csv_path    = r"U:\eng_research_hrc_binauralhearinglab\Sudan\Labs\Sen Lab\Research_projects\Whole_Head_Cocktail_party\ROIs\roi_master.csv"

roi_names = [
    'L-AngGyrus',
    'R-AngGyrus',
    'R-SupramargGyr',
]
highlight_color = '#2CA02C'
conditions      = ['overt']
ref_subject     = '01'

roi_mapping = load_roi_mapping(roi_csv_path)
#%%
for condition in conditions:
    ref_rec = load_snirf_for_geometry(ref_subject, condition, master_data_dir)
    if ref_rec is None:
        print(f" Could not load reference SNIRF for {condition}")
        continue
    for roi_name in roi_names:
        plot_single_roi_highlight(
            rec=ref_rec,
            roi_mapping=roi_mapping,
            roi_name=roi_name,
            condition=condition,
            output_base_dir=output_base_dir,
            highlight_color=highlight_color,
        )

# %%
