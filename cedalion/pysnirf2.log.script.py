#%%
from whichscript import configure, enable_auto_logging

configure(
    archive=True,
    archive_only=False,
    archive_dir=r"U:\eng_research_hrc_binauralhearinglab\Sudan\Labs\Sen Lab\Research_projects\Whole_Head_Cocktail_party\whichscript_archive",
    hide_sidecars=True,
    metadata=False,
    snapshot_script=True,
    snapshot_py=True,
    local_imports_snapshot=False,
)

enable_auto_logging()
#%%
"""
Scatter plot analysis: Overt vs Control
Extracts max HbO values from 3-10 second window for each ROI and creates scatter plots.
Uses exact same functions from group_level_brodmann.py for code integrity.

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

# For smart annotation positioning
try:
    from adjustText import adjust_text
    ADJUST_TEXT_AVAILABLE = True
except ImportError:
    print("⚠️  adjustText not available. Installing...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'adjustText'])
    from adjustText import adjust_text
    ADJUST_TEXT_AVAILABLE = True

# import my own functions from a different directory
import sys
sys.path.append('U:\eng_research_hrc_binauralhearinglab\Sudan\Labs\Cocktail Party\SudanCocktailParty_codes\laura_codes')
sys.path.append('U:\eng_research_hrc_binauralhearinglab\Sudan\Labs\Cocktail Party\SudanCocktailParty_codes\laura_codes\laura_img_recon')
import processing_func as pf

# Turn off all warnings
import warnings
warnings.filterwarnings('ignore')

# Custom legend handler for circles with colored edges
from matplotlib.legend_handler import HandlerBase
class HandlerCircle(HandlerBase):
    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        from matplotlib.patches import Circle
        center = 0.5 * width - 0.5 * xdescent, 0.5 * height - 0.5 * ydescent
        r = min(width, height) / 2.5
        p = Circle(center, r, facecolor=orig_handle.get_facecolor(),
                   edgecolor=orig_handle.get_edgecolor(), 
                   linewidth=orig_handle.get_linewidth(),
                   transform=trans)
        return [p]

#%%
import importlib
importlib.reload(pf)

#%% Parameters (copied from group_level_brodmann.py and RF_snr_0.py)
rootDir_saveData = "U:\\eng_research_hrc_binauralhearinglab\\Sudan\\Labs\\Sen Lab\\Research_projects\\Whole_Head_Cocktail_party\\Cocktail_party_whole_head_master_data\\derivatives\\processed_data\\"

# Subject IDs (same as group_level_brodmann.py - note: use 21 subjects that have control data)
subj_ids_overt_control = ['01','02','03','04','05','10','11','12','13','14','15','18','20','22','25','28','30','31','32','33','34','35','39','41','44','47']

# Analysis window for max extraction
TIME_WINDOW_START = 3.0  # seconds
TIME_WINDOW_END = 12.0   # seconds (changed from 10 to 12)

# Output/cache settings
output_dir = Path("U:\\eng_research_hrc_binauralhearinglab\\Sudan\\Labs\\Sen Lab\\Research_projects\\Whole_Head_Cocktail_party\\Group_avg_results\\scatter_above_chance_top5")
output_dir.mkdir(parents=True, exist_ok=True)

# Cache only what is needed for plotting the condensed top-5 figures
USE_SAVED_PLOTTING_DATA = True
PLOTTING_CACHE_PATH = output_dir / "top5_plotting_data.pkl"

# Only generate the 4 condensed top-5 plots by default
RUN_ONLY_TOP5 = True

#%% Load ROI definitions (copied from group_level_brodmann.py)
roi_df = pd.read_csv(r"U:\eng_research_hrc_binauralhearinglab\Sudan\Labs\Sen Lab\Research_projects\Whole_Head_Cocktail_party\ROIs\roi_master.csv")
roi_dict = {
    roi: roi_df.loc[roi_df.brodmann == roi, "channel_label"].to_list()
    for roi in roi_df.brodmann.unique()
}
all_rois = sorted(roi_dict)

#%% Load top ROIs from PC contribution analysis for selective annotation

def normalize_roi_name(roi_name):
    """Normalize ROI names for matching between different naming conventions."""
    import re
    # Handle hemisphere prefix variations
    roi = str(roi_name)
    roi = roi.replace('Right-', 'R-').replace('Left-', 'L-')
    # Remove (n) Brodmann numbers
    roi = re.sub(r'\s*\(\d+\)', '', roi)
    return roi.strip()


def extract_brodmann_number(roi_name):
    """
    Extract Brodmann area number from ROI name for cleaner annotation.
    
    Examples:
        'Left-Angular Gyrus (39)' -> '39'
        'Right-Primary Motor Cortex (4)' -> '4'
        'Left-Some Region (word)' -> 'Some Region (word)'  # fallback to full name
    
    Returns:
        str: Brodmann number if found, otherwise cleaned ROI name
    """
    import re
    # Try to find number in parentheses
    match = re.search(r'\((\d+)\)', roi_name)
    if match:
        return match.group(1)  # Just the number
    else:
        # Fallback: remove hemisphere prefix
        cleaned = roi_name.replace('Left-', '').replace('Right-', '')
        return cleaned


def load_top_n_rois_from_pc_contrib(csv_path, top_n=5):
    """Load top N most important ROIs from PC contribution analysis.
    
    Parameters
    ----------
    csv_path : str
        Path to roi_contrib_group_summary.csv from analyze_pc_contributions_group_roi.py
    top_n : int
        Number of top ROIs to return (default: 5)
    
    Returns
    -------
    tuple of (set of str, dict)
        Normalized ROI names of top contributors, and dict mapping normalized name to original name
    """
    try:
        df = pd.read_csv(csv_path)
        # Already sorted by mean_importance in the PC contrib script
        top_rois = df.head(top_n)['roi_name'].tolist()
        # Normalize for matching
        top_rois_normalized = {normalize_roi_name(r) for r in top_rois}
        # Also create mapping from normalized to original
        norm_to_original = {normalize_roi_name(r): r for r in top_rois}
        print(f"✓ Loaded top {top_n} ROIs from PC contribution analysis:")
        for i, roi in enumerate(top_rois, 1):
            importance = df.iloc[i-1]['mean_importance']
            print(f"   {i}. {roi} ({importance:.2f}% contribution)")
        return top_rois_normalized, norm_to_original
    except FileNotFoundError:
        print(f"⚠️  PC contribution file not found: {csv_path}")
        print("   Falling back to distance-based annotation threshold")
        return set(), {}
    except Exception as e:
        print(f"⚠️  Error loading PC contributions: {e}")
        print("   Falling back to distance-based annotation threshold")
        return set(), {}

# Load top 10 ROIs from overt PC contribution analysis
pc_contrib_overt_csv = r"U:\eng_research_hrc_binauralhearinglab\Sudan\Labs\Sen Lab\Research_projects\Whole_Head_Cocktail_party\Classifier_script_results\RF_above_chance_group_roi_contributions\group_overt_roi_contrib\roi_contrib_group_summary.csv"

print("\n🎯 Loading top ROIs from PC contribution analysis for annotation...")
TOP_ROIS_FOR_ANNOTATION, TOP_ROIS_NORM_TO_ORIG = load_top_n_rois_from_pc_contrib(pc_contrib_overt_csv, top_n=5)

# GLOBAL color mapping - ensures same ROI always gets same color across ALL plots
RING_COLORS = [
    '#E41A1C',  # Red
    '#377EB8',  # Blue
    '#4DAF4A',  # Green
    '#984EA3',  # Purple
    '#FF7F00',  # Orange
    '#FFFF33',  # Yellow
    '#A65628',  # Brown
    '#F781BF',  # Pink
    '#00CED1',  # Dark Cyan
    '#8B0000',  # Dark Red
]
# Create global color mapping based on base ROI names (strip hemisphere prefixes)
# This ensures Left/Right versions of the same ROI share the same color.
def _base_roi_name(name):
    """Return ROI name without hemisphere prefixes like 'Left-', 'Right-', 'L-', 'R-'."""
    if name is None:
        return name
    base = str(name).replace('Left-', '').replace('Right-', '')
    base = base.replace('L-', '').replace('R-', '')
    return base.strip()

# Map base ROI name -> color (sorted for consistency)
TOP_ROI_COLOR_MAP = {
    _base_roi_name(orig_name): RING_COLORS[i % len(RING_COLORS)]
    for i, (norm, orig_name) in enumerate(sorted(TOP_ROIS_NORM_TO_ORIG.items()))
}
print(f"\n🎨 Global ROI color mapping (base names):")
for roi_base, color in TOP_ROI_COLOR_MAP.items():
    print(f"   {roi_base} → {color}")

#%% Helper functions (copied exactly from RF_snr_0.py for data loading)

def load_all_subjects(subj_ids, run_type, data_dir):
    """Load all preprocessed subjects from individual files. (from RF_snr_0.py)"""
    rec = []
    chs_pruned_subjs = []
    subj_dir = os.path.join(data_dir, f"preprocessed_{run_type}_snr_0")
    
    for subj_id in subj_ids:
        rec_file = os.path.join(subj_dir, f"rec_subj_{subj_id}.pkl")
        prune_file = os.path.join(subj_dir, f"chs_pruned_subj_{subj_id}.pkl")
        
        if os.path.exists(rec_file) and os.path.exists(prune_file):
            try:
                with gzip.open(rec_file, 'rb') as f:
                    rec.append(pickle.load(f))
            except Exception:
                with open(rec_file, 'rb') as f:
                    rec.append(pickle.load(f))
            try:
                with gzip.open(prune_file, 'rb') as f:
                    chs_pruned_subjs.append(pickle.load(f))
            except Exception:
                with open(prune_file, 'rb') as f:
                    chs_pruned_subjs.append(pickle.load(f))
            print(f"✓ Loaded subject {subj_id}")
        else:
            print(f"⚠ Warning: Subject {subj_id} not found, skipping")
            rec.append(None)
            chs_pruned_subjs.append(None)
    
    return rec, chs_pruned_subjs


def _blockavg_all_runs(rec, stim_list,
                       ts_name='conc_p_tddr_filt_postglm',
                       t_pre=2*units.s,
                       t_post=15*units.s):
    """Return nested list [subj][run] of block‑average DataArrays."""
    out = [[None]*len(rec[0]) for _ in range(len(rec))]
    for s_idx in range(len(rec)):
        for r_idx in range(len(rec[s_idx])):
            _, ba = pf.block_average(rec, ts_name, stim_list,
                                     t_pre, t_post,
                                     subj_idx=s_idx, file_idx=r_idx)
            out[s_idx][r_idx] = ba
    return out


def collapse_runs(ba_list):
    """Average across runs for each subject."""
    out = []
    for subj_runs in ba_list:
        da = xr.concat(subj_runs, dim="run", join="inner").mean("run")
        out.append(da)
    return out


def roi_mean_per_subject(subj_avg_list):
    """Compute ROI-averaged data per subject."""
    roi_ds = []
    for da in subj_avg_list:
        roi_slices = []
        for roi, chs in roi_dict.items():
            avail = [c for c in chs if c in da.channel.values]
            if not avail:
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


#%% MEMORY-EFFICIENT LOADING: Process each condition separately

import gc  # For garbage collection

def process_condition(subj_ids, condition_name, trial_names, data_dir):
    """
    Load, process, and extract max values for one condition, then clear memory.
    
    Returns:
        tuple: (group_mean, group_sem, n_subj, max_vals)
    """
    print(f"\n{'='*60}")
    print(f"Processing {condition_name.upper()} condition...")
    print(f"{'='*60}")
    
    # Load data
    print(f"  1/5 Loading subjects...")
    rec, chs = load_all_subjects(subj_ids, condition_name.lower(), data_dir)
    
    # Block average
    print(f"  2/5 Block averaging...")
    ba = _blockavg_all_runs(rec, trial_names)
    
    # Clear rec to save memory
    del rec, chs
    gc.collect()
    
    # Collapse runs
    print(f"  3/5 Collapsing runs...")
    subj_avg = collapse_runs(ba)
    
    # Clear ba to save memory
    del ba
    gc.collect()
    
    # ROI averaging
    print(f"  4/5 Computing ROI averages...")
    subj_roi = roi_mean_per_subject(subj_avg)
    
    # Clear subj_avg to save memory
    del subj_avg
    gc.collect()
    
    # Group statistics
    print(f"  5/5 Computing group statistics...")
    group_mean, group_sem, n_subj = group_mean_sem_robust(subj_roi)
    
    # Clear subj_roi to save memory
    del subj_roi
    gc.collect()
    
    print(f"✅ {condition_name.upper()} processing complete!")
    
    return group_mean, group_sem, n_subj

#%%
# Process each condition sequentially unless cached plotting data is available
USING_CACHED_PLOTTING_DATA = USE_SAVED_PLOTTING_DATA and PLOTTING_CACHE_PATH.exists()

if USING_CACHED_PLOTTING_DATA:
    print(f"Using cached plotting data: {PLOTTING_CACHE_PATH}")
    with open(str(PLOTTING_CACHE_PATH), "rb") as f:
        cached = pickle.load(f)
    max_vals_overt = cached["max_vals_overt"]
    max_vals_covert = cached["max_vals_covert"]
    max_vals_control = cached["max_vals_control"]
else:
    print(" Starting memory-efficient sequential processing...")

    # 1. OVERT
    group_mean_overt, group_sem_overt, n_subj_overt = process_condition(
        subj_ids_overt_control, 'overt', ['Overt Left', 'Overt Right'], rootDir_saveData
    )
    # 2. COVERT
    group_mean_covert, group_sem_covert, n_subj_covert = process_condition(
        subj_ids_overt_control, 'covert', ['Covert Left', 'Covert Right'], rootDir_saveData
    )
    # 3. CONTROL
    group_mean_control, group_sem_control, n_subj_control = process_condition(
        subj_ids_overt_control, 'control', ['Control Left', 'Control Right'], rootDir_saveData
    )

#%% Extract max values in 3-10 second window

def extract_max_hbo_in_window(group_mean, t_start=TIME_WINDOW_START, t_end=TIME_WINDOW_END):
    """
    Extract max ABSOLUTE HbO value in specified time window for each ROI and trial type.
    
    KEY STRATEGY: To ensure left and right trials use the SAME temporal point:
    1. Find max |HbO| for left trial and right trial independently
    2. Pick whichever has HIGHER absolute magnitude
    3. Use that trial's peak TIME for BOTH left and right
    
    This avoids:
    - Comparing peak to trough (different temporal features)
    - Averaging left/right which could cancel out if bifurcated
    
    Returns:
        dict: {roi: {'left': {'max_val': float, 'max_time': float}, 
                     'right': {...}, 
                     'dominant_trial': str}}
    """
    results = {}
    
    # Filter to HbO only
    hbo_data = group_mean.sel(chromo='HbO')
    
    # Get time values
    time = hbo_data.reltime.values
    
    # Find indices within window
    window_mask = (time >= t_start) & (time <= t_end)
    windowed_time = time[window_mask]
    
    for roi in hbo_data.ROI.values:
        results[roi] = {}
        
        # Step 1: Find max |HbO| for each trial type
        trial_peaks = {}  # {trial_type: (abs_magnitude, time_idx, signed_val)}
        
        for trial_type in hbo_data.trial_type.values:
            # Extract data for this ROI and trial type
            data = hbo_data.sel(ROI=roi, trial_type=trial_type)
            windowed_data = data.isel(reltime=window_mask)
            
            # Find max of ABSOLUTE VALUE
            abs_data = np.abs(windowed_data)
            max_idx = abs_data.argmax(dim='reltime').item()
            abs_magnitude = abs_data.isel(reltime=max_idx).item()
            signed_val = windowed_data.isel(reltime=max_idx).item()
            
            trial_peaks[trial_type] = (abs_magnitude, max_idx, signed_val)
        
        # Step 2: Determine which trial has higher absolute magnitude
        dominant_trial = max(trial_peaks.keys(), 
                            key=lambda t: trial_peaks[t][0])  # Compare abs magnitudes
        dominant_time_idx = trial_peaks[dominant_trial][1]
        dominant_time = windowed_time[dominant_time_idx]
        
        # Convert to float
        if hasattr(dominant_time, 'magnitude'):
            common_time_float = float(dominant_time.magnitude)
        else:
            common_time_float = float(dominant_time)
        
        results[roi]['dominant_trial'] = str(dominant_trial)
        results[roi]['common_time'] = common_time_float
        
        # Step 3: Extract values at the COMMON time for all trials
        for trial_type in hbo_data.trial_type.values:
            data = hbo_data.sel(ROI=roi, trial_type=trial_type)
            windowed_data = data.isel(reltime=window_mask)
            
            # Get value at the COMMON (dominant) time point
            val_at_common_time = windowed_data.isel(reltime=dominant_time_idx).item()
            
            # Extract magnitude (strip units) for JSON serialization
            if hasattr(val_at_common_time, 'magnitude'):
                val_float = float(val_at_common_time.magnitude)
            else:
                val_float = float(val_at_common_time)
            
            # Store results
            trial_key = trial_type.lower()
            results[roi][trial_key] = {
                'max_val': val_float,  # Value at common dominant time
                'max_time': common_time_float  # Same time for all trials
            }
    
    return results


if not USING_CACHED_PLOTTING_DATA:
    # print(f"\nExtracting max HbO values in {TIME_WINDOW_START}-{TIME_WINDOW_END}s window...")
    max_vals_overt = extract_max_hbo_in_window(group_mean_overt)
    max_vals_covert = extract_max_hbo_in_window(group_mean_covert)
    max_vals_control = extract_max_hbo_in_window(group_mean_control)

    with open(str(PLOTTING_CACHE_PATH), "wb") as f:
        pickle.dump({
            "max_vals_overt": max_vals_overt,
            "max_vals_covert": max_vals_covert,
            "max_vals_control": max_vals_control,
        }, f)
    print(f"\n?o. Cached plotting data to: {PLOTTING_CACHE_PATH}")

if not RUN_ONLY_TOP5:
    #  Create scatter plots: Overt vs Control

    # Define colors matching the original plots
    colors = {
        'left': [0.8, 0, 0],      # Crimson Red
        'right': [1, 0.27, 0],    # Orange Red
    }

    def create_scatter_plot(roi, max_vals_overt, max_vals_control, save_path=None):
        """Create scatter plot comparing Overt vs Control for one ROI."""
    
        if roi not in max_vals_overt or roi not in max_vals_control:
            print(f"Skipping {roi} - not present in both conditions")
            return
    
        fig, ax = plt.subplots(figsize=(8, 8))
    
        # Extract values for left and right - match by 'left'/'right' in trial name
        all_vals = []
        for side, color in colors.items():
            # Find matching trials
            overt_trial = None
            control_trial = None
        
            for trial in max_vals_overt[roi].keys():
                if side in trial.lower():
                    overt_trial = trial
                    break
        
            for trial in max_vals_control[roi].keys():
                if side in trial.lower():
                    control_trial = trial
                    break
        
            if overt_trial and control_trial:
                overt_val = max_vals_overt[roi][overt_trial]['max_val']
                control_val = max_vals_control[roi][control_trial]['max_val']
            
                # Plot scatter point
                ax.scatter(overt_val, control_val, 
                          color=color, s=150, alpha=0.7,
                          label=f'{side.capitalize()}', 
                          edgecolors='black', linewidths=1.5)
            
                all_vals.extend([overt_val, control_val])
    
        if all_vals:
            min_val = min(all_vals)
            max_val = max(all_vals)
            margin = (max_val - min_val) * 0.1
            ax.plot([min_val - margin, max_val + margin], 
                   [min_val - margin, max_val + margin],
                   'k--', alpha=0.5, linewidth=2, label='Identity')
    
        # Formatting
        ax.set_xlabel('Overt', fontsize=22, fontweight='bold')
        ax.set_ylabel('Control', fontsize=22, fontweight='bold')
        ax.legend(fontsize=22)
        ax.grid(True, linestyle=':', alpha=0.4)
        ax.tick_params(labelsize=20)
    
        # Equal aspect ratio
        ax.set_aspect('equal', adjustable='box')
    
        plt.tight_layout()
    
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            fig.savefig(save_path.replace('.png', '.svg'), format='svg', bbox_inches='tight')
            fig.savefig(save_path.replace('.png', '.pdf'), format='pdf', bbox_inches='tight')
            plt.close(fig)

        return fig, ax


    print("\n📊 Creating Overt vs Control scatter plots for all ROIs...")
    scatter_dir = output_dir / "scatter_plots"
    scatter_dir.mkdir(exist_ok=True)

    for roi in all_rois:
        print(f"  Creating Overt vs Control scatter plot for {roi}")
        save_path = scatter_dir / f"scatter_overt_vs_control_{roi}.png"
        create_scatter_plot(roi, max_vals_overt, max_vals_control, save_path=str(save_path))

    print(f"✅ Saved Overt vs Control scatter plots to: {scatter_dir}")

    #%% Create COMBINED scatter plot with ALL ROIs

    def create_combined_scatter_plot(all_rois, max_vals_x, max_vals_y, 
                                     x_label='Overt', y_label='Control', save_path=None):
        """Create single scatter plot with all ROIs, separate markers for left/right.
    
        Parameters
        ----------
        max_vals_x : dict
            Max values for x-axis (e.g., Overt or Covert)
        max_vals_y : dict
            Max values for y-axis (e.g., Control or Covert)
        x_label : str
            Label for x-axis condition
        y_label : str
            Label for y-axis condition
        """
    
        fig, ax = plt.subplots(figsize=(10, 10))
    
        all_vals = []
    
        # Collect data for all ROIs
        for side, color in colors.items():
            x_vals = []
            y_vals = []
            roi_labels = []
        
            for roi in all_rois:
                if roi not in max_vals_x or roi not in max_vals_y:
                    continue
            
                # Find matching trials
                x_trial = None
                y_trial = None
            
                for trial in max_vals_x[roi].keys():
                    if side in trial.lower():
                        x_trial = trial
                        break
            
                for trial in max_vals_y[roi].keys():
                    if side in trial.lower():
                        y_trial = trial
                        break
            
                if x_trial and y_trial:
                    x_val = max_vals_x[roi][x_trial]['max_val']
                    y_val = max_vals_y[roi][y_trial]['max_val']
                
                    x_vals.append(x_val)
                    y_vals.append(y_val)
                    roi_labels.append(roi)
                    all_vals.extend([x_val, y_val])
        
            # Plot all points for this side
            if x_vals:
                ax.scatter(x_vals, y_vals, 
                          color=color, s=100, alpha=0.6,
                          label=f'{side.capitalize()} (n={len(x_vals)} ROIs)', 
                          edgecolors='black', linewidths=1)
    
        # Add identity line
        if all_vals:
            min_val = min(all_vals)
            max_val = max(all_vals)
            margin = (max_val - min_val) * 0.1
            ax.plot([min_val - margin, max_val + margin], 
                   [min_val - margin, max_val + margin],
                   'k--', alpha=0.5, linewidth=2, label='Identity', zorder=1)
    
        # Formatting
        ax.set_xlabel(f'{x_label}', fontsize=22, fontweight='bold')
        ax.set_ylabel(f'{y_label}', fontsize=22, fontweight='bold')
        ax.legend(fontsize=22, loc='upper left')
        ax.grid(True, linestyle=':', alpha=0.4)
        ax.tick_params(labelsize=20)
    
        # Equal aspect ratio
        ax.set_aspect('equal', adjustable='box')
    
        plt.tight_layout()
    
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            fig.savefig(save_path.replace('.png', '.svg'), format='svg', bbox_inches='tight')
            fig.savefig(save_path.replace('.png', '.pdf'), format='pdf', bbox_inches='tight')

        return fig, ax


    print("\n📊 Creating COMBINED scatter plots with all ROIs...")

    # Overt vs Control
    combined_save_path_overt_control = output_dir / "scatter_plots" / "scatter_ALL_ROIS_overt_vs_control.png"
    create_combined_scatter_plot(all_rois, max_vals_overt, max_vals_control, 
                                x_label='Overt', y_label='Control',
                                save_path=str(combined_save_path_overt_control))
    print(f"✅ Saved Overt vs Control combined scatter plot to: {combined_save_path_overt_control}")


    #%% Create hemisphere × trial-type specific scatter plots with annotations

    def create_hemi_trial_scatter_plot(all_rois, max_vals_x, max_vals_y, 
                                        x_label='Overt', y_label='Control',
                                        hemisphere='left', trial_side='left', save_path=None):
        """
        Create scatter plot for one hemisphere and one trial type with ROI labels.
    
        Parameters
        ----------
        max_vals_x : dict
            Max values for x-axis (e.g., Overt)
        max_vals_y : dict
            Max values for y-axis (e.g., Control or Covert)
        x_label : str
            Label for x-axis condition
        y_label : str
            Label for y-axis condition
        hemisphere : str
            'left' or 'right' - which hemisphere ROIs to plot
        trial_side : str
            'left' or 'right' - which trial type to plot
        """
    
        fig, ax = plt.subplots(figsize=(10, 10))
    
        # Filter ROIs by hemisphere
        if hemisphere.lower() == 'left':
            hemi_rois = [roi for roi in all_rois if roi.startswith('Left-')]
            title_hemi = 'Left Hemisphere'
        else:
            hemi_rois = [roi for roi in all_rois if roi.startswith('Right-')]
            title_hemi = 'Right Hemisphere'
    
        # Get color for this trial side
        color = colors[trial_side]
    
        x_vals = []
        y_vals = []
        roi_labels = []
        full_roi_names = []  # Track full ROI names for PC contrib matching
    
        for roi in hemi_rois:
            if roi not in max_vals_x or roi not in max_vals_y:
                continue
        
            # Find matching trials
            x_trial = None
            y_trial = None
        
            for trial in max_vals_x[roi].keys():
                if trial_side in trial.lower():
                    x_trial = trial
                    break
        
            for trial in max_vals_y[roi].keys():
                if trial_side in trial.lower():
                    y_trial = trial
                    break
        
            if x_trial and y_trial:
                x_val = max_vals_x[roi][x_trial]['max_val']
                y_val = max_vals_y[roi][y_trial]['max_val']
            
                x_vals.append(x_val)
                y_vals.append(y_val)
                full_roi_names.append(roi)  # Keep full name for matching
                # Clean up ROI label - remove hemisphere prefix and keep Brodmann area
                clean_label = roi.replace('Left-', '').replace('Right-', '')
                roi_labels.append(clean_label)
    
        # Plot all points
        if x_vals:
            scatter = ax.scatter(x_vals, y_vals, 
                      color='black', s=200, alpha=0.7,
                      edgecolors='black', linewidths=1.5, zorder=3)
        
            # COLOR-CODED ANNOTATION: Colored rings + matching legend
            # Uses GLOBAL color mapping (TOP_ROI_COLOR_MAP) for consistent colors across all plots
        
            # Collect top ROIs to annotate
            top_roi_data = []  # [(x, y, label, ring_color), ...]
            for i, (x, y, full_roi) in enumerate(zip(x_vals, y_vals, full_roi_names)):
                normalized_roi = normalize_roi_name(full_roi)
                if TOP_ROIS_FOR_ANNOTATION and normalized_roi in TOP_ROIS_FOR_ANNOTATION:
                    # Get the original ROI name from the global mapping
                    original_roi_name = TOP_ROIS_NORM_TO_ORIG.get(normalized_roi, full_roi.replace('Left-', '').replace('Right-', ''))
                    # Get color from global mapping using base ROI name (strip hemisphere)
                    ring_color = TOP_ROI_COLOR_MAP.get(_base_roi_name(original_roi_name), '#000000')
                    label = full_roi.replace('Left-', '').replace('Right-', '')
                    top_roi_data.append((x, y, label, ring_color, original_roi_name))
                    print(f"      Top ROI: {label} → {ring_color}")
        
            print(f"   Found {len(top_roi_data)} top PC contributor ROIs for {hemisphere} hemisphere, {trial_side} trial")
        
            # Draw colored rings around top ROI points
            for px, py, lbl, ring_col, orig_name in top_roi_data:
                ax.scatter([px], [py], s=450, facecolors='none', 
                          edgecolors=ring_col, linewidths=4, zorder=4)
        
            # Create custom legend using global color map (ensures consistency)
            if top_roi_data:
                from matplotlib.patches import Patch
                from matplotlib.lines import Line2D
            
                # Use global map to create legend - shows all top ROIs with their assigned colors
                legend_handles = []
                legend_labels = []
                # Only show ROIs that appear in this plot, but use global colors
                seen_rois = set()
                for px, py, lbl, ring_col, orig_name in top_roi_data:
                    if orig_name not in seen_rois:
                        seen_rois.add(orig_name)
                        # Create a circle patch for legend
                        handle = plt.Circle((0, 0), 1, facecolor='black', edgecolor=ring_col, linewidth=3)
                        legend_handles.append(handle)
                        # Use base ROI name (no hemisphere) for legend clarity
                        legend_labels.append(_base_roi_name(orig_name))
            
                # Add the legend
                roi_legend = ax.legend(legend_handles, legend_labels,
                                      title='Top PC Contributors',
                                      loc='upper right', fontsize=22, title_fontsize=24,
                                      framealpha=0.9, edgecolor='gray',
                                      handler_map={plt.Circle: HandlerCircle()})
                ax.add_artist(roi_legend)
        
            # Add identity line
            all_vals = x_vals + y_vals
            min_val = min(all_vals)
            max_val = max(all_vals)
            margin = (max_val - min_val) * 0.1
            ax.plot([min_val - margin, max_val + margin], 
                   [min_val - margin, max_val + margin],
                   'k--', alpha=0.5, linewidth=2, zorder=1)
        
            # Add origin lines to show quadrants
            ax.axhline(0, color='gray', linestyle='-', linewidth=1.5, alpha=0.4, zorder=0)
            ax.axvline(0, color='gray', linestyle='-', linewidth=1.5, alpha=0.4, zorder=0)
    
        # Formatting
        ax.set_xlabel(f'{x_label}', fontsize=28, fontweight='bold')
        ax.set_ylabel(f'{y_label}', fontsize=28, fontweight='bold')
        ax.grid(True, linestyle=':', alpha=0.3)
        ax.tick_params(labelsize=24)
    
        # Equal aspect ratio
        ax.set_aspect('equal', adjustable='box')
    
        plt.tight_layout()
    
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            # Also save as SVG for Adobe Illustrator editing
            svg_path = save_path.replace('.png', '.svg')
            fig.savefig(svg_path, format='svg', bbox_inches='tight')
            pdf_path = save_path.replace('.png', '.pdf')
            fig.savefig(pdf_path, format='pdf', bbox_inches='tight')
            plt.close(fig)
    
        return fig, ax


    print("\n📊 Creating hemisphere × trial-type scatter plots with ROI labels...")

    print("\n🎯 ANNOTATION STRATEGY:")
    print("  1. Top-5 PC contributor ROIs get colored rings (Red, Blue, Green, Purple, Orange)")
    print("  2. Legend shows ROI names with matching colored ring markers")
    print("  3. All other points remain as solid circles with black edges")
    print("  4. No text clutter on the plot - just visual color matching")
    print()
    if TOP_ROIS_FOR_ANNOTATION:
        print(f"  📌 Top-5 ROIs to annotate: {TOP_ROIS_FOR_ANNOTATION}")
    else:
        print("  ⚠️  No top ROIs loaded - no annotations will be added")
    print()

    # Create all 4 combinations for Overt vs Control in a SEPARATE SUBFOLDER
    combos = [
        ('left', 'left', 'LEFT_HEMI_LEFT_TRIAL'),
        ('left', 'right', 'LEFT_HEMI_RIGHT_TRIAL'),
        ('right', 'left', 'RIGHT_HEMI_LEFT_TRIAL'),
        ('right', 'right', 'RIGHT_HEMI_RIGHT_TRIAL'),
    ]

    print("\n📊 Creating Overt vs Control scatter plots (in separate subfolder)...")
    overt_control_dir = output_dir / "scatter_plots" / "overt_vs_control_main"
    overt_control_dir.mkdir(parents=True, exist_ok=True)

    for hemi, trial, filename in combos:
        save_path = overt_control_dir / f"scatter_{filename}.png"
        create_hemi_trial_scatter_plot(all_rois, max_vals_overt, max_vals_control,
                                       x_label='Overt', y_label='Control',
                                       hemisphere=hemi, trial_side=trial,
                                       save_path=str(save_path))
        print(f"✅ Saved {hemi} hemisphere, {trial} trial → {save_path.name}")

    print(f"\n✅ Overt vs Control plots saved to: {overt_control_dir}")

    print("\n📊 Creating Overt vs Covert scatter plots...")
    overt_covert_dir = output_dir / "scatter_plots" / "overt_vs_covert_main"
    overt_covert_dir.mkdir(parents=True, exist_ok=True)

    for hemi, trial, filename in combos:
        save_path = overt_covert_dir / f"scatter_{filename}.png"
        create_hemi_trial_scatter_plot(all_rois, max_vals_overt, max_vals_covert,
                                       x_label='Overt', y_label='Covert',
                                       hemisphere=hemi, trial_side=trial,
                                       save_path=str(save_path))
        print(f"✅ Saved {hemi} hemisphere, {trial} trial → {save_path.name}")

    print(f"\n✅ Overt vs Covert plots saved to: {overt_covert_dir}")

    #%% Create HRF plots with max markers for verification

    def plot_hrf_with_max_marker(roi, group_mean, group_sem, max_vals, 
                                 condition_name, save_path=None):
        """
        Plot HRF with star markers at max values for verification.
        Only plots HbO (not HbR).
        """
    
        if roi not in group_mean.ROI.values:
            print(f"Skipping {roi} - not present in {condition_name}")
            return
    
        # Extract data for this ROI
        roi_data = group_mean.sel(ROI=roi, chromo='HbO')
        roi_sem = group_sem.sel(ROI=roi, chromo='HbO')
    
        # Get time vector
        t = roi_data.reltime.values
    
        fig, ax = plt.subplots(figsize=(10, 6))
    
        # Plot each trial type
        # DEBUG: Check what trial types are available
        print(f"    DEBUG {condition_name} - Available trial types: {roi_data.trial_type.values}")
    
        for trial_type, color in colors.items():
            # trial_type from colors dict is lowercase ('left', 'right')
            # Need to match against actual trial_type values in data
            matching_trial = None
            for tt in roi_data.trial_type.values:
                if trial_type.lower() in str(tt).lower():
                    matching_trial = tt
                    break
        
            if matching_trial is None:
                print(f"    WARNING: No matching trial type found for '{trial_type}'")
                continue
        
            # Extract mean and SEM
            m = roi_data.sel(trial_type=matching_trial).values
            se = roi_sem.sel(trial_type=matching_trial).values
        
            # Plot line and shaded error
            ax.plot(t, m, color=color, label=f'HbO {matching_trial}', 
                   linewidth=2, alpha=0.9)
            ax.fill_between(t, m-se, m+se, color=color, alpha=0.2)
        
            # Add star at max value - need to find the matching trial in max_vals
            star_trial = None
            if roi in max_vals:
                for trial in max_vals[roi].keys():
                    if trial_type in trial.lower():
                        star_trial = trial
                        break
        
            if star_trial:
                max_time = max_vals[roi][star_trial]['max_time']
                max_val = max_vals[roi][star_trial]['max_val']
                ax.plot(max_time, max_val, marker='*', markersize=20, 
                       color=color, markeredgecolor='black', markeredgewidth=1.5,
                       linestyle='None', zorder=10)
    
        # Add vertical lines for events
        events = [0, 2, 5]
        event_colors = ["black", "green", "orange"]
        event_labels = ["Cue onset", "Stim onset", "Stim offset"]
    
        for x, c, label in zip(events, event_colors, event_labels):
            ax.axvline(x, linestyle="--", color=c, linewidth=2, alpha=0.7, label=label)
    
        # Shade the analysis window
        ax.axvspan(TIME_WINDOW_START, TIME_WINDOW_END, 
                  alpha=0.1, color='gray', label='Analysis Window (3-12s)')
    
        # Formatting
        ax.set_xlabel("Time (s)", fontsize=22)
        ax.set_ylabel("HbO Concentration Change (μM·mm)", fontsize=22)
        ax.legend(fontsize=22, loc='best')
        ax.grid(True, linestyle=':', alpha=0.4)
        ax.tick_params(labelsize=20)
        ax.set_xlim(t.min(), t.max())
    
        plt.tight_layout()
    
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            fig.savefig(save_path.replace('.png', '.svg'), format='svg', bbox_inches='tight')
            fig.savefig(save_path.replace('.png', '.pdf'), format='pdf', bbox_inches='tight')
            plt.close(fig)
    
        return fig, ax


    print("\n📊 Creating HRF plots with max markers for verification...")
    hrf_overt_dir = output_dir / "hrf_plots_overt"
    hrf_control_dir = output_dir / "hrf_plots_control"
    hrf_overt_dir.mkdir(exist_ok=True)
    hrf_control_dir.mkdir(exist_ok=True)

    for roi in all_rois:
        print(f"  Creating HRF plots for {roi}")
    
        # Overt
        save_path = hrf_overt_dir / f"hrf_overt_{roi}.png"
        plot_hrf_with_max_marker(roi, group_mean_overt, group_sem_overt, 
                                 max_vals_overt, "Overt", save_path=str(save_path))
    
        # Control
        save_path = hrf_control_dir / f"hrf_control_{roi}.png"
        plot_hrf_with_max_marker(roi, group_mean_control, group_sem_control,
                                 max_vals_control, "Control", save_path=str(save_path))

    print(f"✅ Saved HRF plots to: {hrf_overt_dir} and {hrf_control_dir}")

    #%% Create summary table

    summary_data = []

    # DEBUG: Check what's actually in max_vals
    print("\n🔍 DEBUG: Checking max_vals structure...")
    if max_vals_overt:
        first_roi = list(max_vals_overt.keys())[0]
        print(f"   First Overt ROI: {first_roi}")
        print(f"   Its trial types: {list(max_vals_overt[first_roi].keys())}")
    if max_vals_control:
        first_roi = list(max_vals_control.keys())[0]
        print(f"   First Control ROI: {first_roi}")
        print(f"   Its trial types: {list(max_vals_control[first_roi].keys())}")

    for roi in all_rois:
        if roi in max_vals_overt and roi in max_vals_control:
            # Match trials by whether they contain 'left' or 'right', ignoring overt/control prefix
            for side in ['left', 'right']:
                # Find the trial in overt that contains this side
                overt_trial = None
                for trial in max_vals_overt[roi].keys():
                    if side in trial.lower():
                        overt_trial = trial
                        break
            
                # Find the trial in control that contains this side
                control_trial = None
                for trial in max_vals_control[roi].keys():
                    if side in trial.lower():
                        control_trial = trial
                        break
            
                if overt_trial and control_trial:
                    summary_data.append({
                        'ROI': roi,
                        'Trial_Type': side.capitalize(),
                        'Overt_Max_HbO': max_vals_overt[roi][overt_trial]['max_val'],
                        'Overt_Max_Time': max_vals_overt[roi][overt_trial]['max_time'],
                        'Control_Max_HbO': max_vals_control[roi][control_trial]['max_val'],
                        'Control_Max_Time': max_vals_control[roi][control_trial]['max_time'],
                        'Difference': max_vals_overt[roi][overt_trial]['max_val'] - max_vals_control[roi][control_trial]['max_val']
                    })
        else:
            if roi not in max_vals_overt:
                print(f"   ✗ ROI '{roi}' NOT in max_vals_overt")
            if roi not in max_vals_control:
                print(f"   ✗ ROI '{roi}' NOT in max_vals_control")

    summary_df = pd.DataFrame(summary_data)

    if len(summary_df) == 0:
        print("\n⚠️  WARNING: No matching ROIs found in both Overt and Control conditions!")
        print(f"   Overt ROIs: {list(max_vals_overt.keys())}")
        print(f"   Control ROIs: {list(max_vals_control.keys())}")
        print("\n   This might mean:")
        print("   1. Control data preprocessing hasn't been run yet (run process_control_data.py)")
        print("   2. Different ROIs survived preprocessing in each condition")
        print("   3. No overlapping ROIs between conditions")
    else:
        summary_df.to_csv(output_dir / "overt_vs_control_summary.csv", index=False)
        print(f"\n✅ Saved summary table to: {output_dir / 'overt_vs_control_summary.csv'}")
        print(f"\nSummary statistics:")
        print(summary_df.describe())
        print(f"\n📊 Total comparisons: {len(summary_df)} (ROIs × trial types)")

    print("\n🎉 Analysis complete!")
    print(f"\nAll outputs saved to: {output_dir}")
    print(f"  - max_values_overt.json: Max values for Overt condition")
    print(f"  - max_values_covert.json: Max values for Covert condition")
    print(f"  - max_values_control.json: Max values for Control condition")
    print(f"  - scatter_plots/")
    print(f"      • scatter_ALL_ROIS_overt_vs_control.png: Combined plot (all ROIs, Overt vs Control)")
    print(f"      • overt_vs_control_main/  ← 🎯 OVERT vs CONTROL PLOTS")
    print(f"          - scatter_LEFT_HEMI_LEFT_TRIAL.png")
    print(f"          - scatter_LEFT_HEMI_RIGHT_TRIAL.png")
    print(f"          - scatter_RIGHT_HEMI_LEFT_TRIAL.png")
    print(f"          - scatter_RIGHT_HEMI_RIGHT_TRIAL.png")
    print(f"      • overt_vs_covert_main/  ← 🎯 OVERT vs COVERT PLOTS")
    print(f"          - scatter_LEFT_HEMI_LEFT_TRIAL.png")
    print(f"          - scatter_LEFT_HEMI_RIGHT_TRIAL.png")
    print(f"          - scatter_RIGHT_HEMI_LEFT_TRIAL.png")
    print(f"          - scatter_RIGHT_HEMI_RIGHT_TRIAL.png")
    print(f"  - hrf_plots_overt/: HRF plots with max markers for Overt")
    print(f"  - hrf_plots_control/: HRF plots with max markers for Control")
    print(f"  - overt_vs_control_summary.csv: Summary table with all max values")

    print(f"\n📍 Annotation Strategy:")
    print(f"  ✓ Color-coded rings: Red, Blue, Green, Purple, Orange")
    print(f"  ✓ Legend with matching ring colors shows ROI names")
    print(f"  ✓ Clean visualization - no text on data points")
    print(f"  ✓ Easy visual matching between rings and legend")


#%% Create CONDENSED Top 5 ROI scatter plots (Option 2: Four plots - 2 comparisons × 2 hemispheres)

def _compute_top5_axis(max_vals_x, max_vals_y):
    """Pick a 'nice' tick step AND snapped (lo, hi) limits from top-5 ROI values
    across BOTH hemispheres, so paired left/right panels are directly comparable.

    Returns (step, lo_snap, hi_snap) or (None, None, None) if no values found.
    """
    vals = []
    for normalized_roi, _ in TOP_ROIS_NORM_TO_ORIG.items():
        for prefix in ('Left-', 'Right-'):
            matching = next(
                (roi for roi in max_vals_x
                 if normalize_roi_name(roi) == normalized_roi and roi.startswith(prefix)),
                None,
            )
            if matching is None or matching not in max_vals_y:
                continue
            for side in ('left', 'right'):
                x_t = next((t for t in max_vals_x[matching] if side in t.lower()), None)
                y_t = next((t for t in max_vals_y[matching] if side in t.lower()), None)
                if x_t and y_t:
                    vals.extend([
                        max_vals_x[matching][x_t]['max_val'],
                        max_vals_y[matching][y_t]['max_val'],
                    ])
    if not vals:
        return None, None, None
    lo, hi = min(vals), max(vals)
    span = max(hi - lo, 1e-9)
    margin = span * 0.15
    raw_step = (span + 2 * margin) / 5
    mag = 10 ** np.floor(np.log10(abs(raw_step)))
    norm = raw_step / mag
    if norm < 1.5:
        step = 1 * mag
    elif norm < 3:
        step = 2 * mag
    elif norm < 7:
        step = 5 * mag
    else:
        step = 10 * mag
    lo_snap = np.floor((lo - margin) / step) * step
    hi_snap = np.ceil((hi + margin) / step) * step
    return step, lo_snap, hi_snap


def create_top5_condensed_scatter(max_vals_x, max_vals_y,
                                   x_label='Overt', y_label='Control',
                                   hemisphere='left',
                                   save_path=None,
                                   add_dotted_trial_connector=False,
                                   tick_step=None,
                                   xlim=None,
                                   ylim=None,
                                   label_fontsize=32,
                                   tick_fontsize=26):
    """
    Create a single condensed scatter plot showing ONLY the top 5 PC contributor ROIs
    for ONE hemisphere.
    
    Parameters
    ----------
    max_vals_x : dict
        Max values for x-axis (e.g., Overt)
    max_vals_y : dict
        Max values for y-axis (e.g., Control or Covert)
    x_label : str
        Label for x-axis condition
    y_label : str
        Label for y-axis condition
    hemisphere : str
        'left' or 'right' - which hemisphere to plot
    
    Visual encoding:
    - Color: Each ROI gets a unique color (from TOP_ROI_COLOR_MAP) - SAME across hemispheres
    - Filled circles: Left trial
    - Open circles: Right trial
    """
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Determine hemisphere prefix
    if hemisphere.lower() == 'left':
        hemi_prefix = 'Left-'
        title_hemi = 'Left Hemisphere'
    else:
        hemi_prefix = 'Right-'
        title_hemi = 'Right Hemisphere'
    
    all_vals = []

    # Iterate through top 5 ROIs only
    for normalized_roi, original_roi_name in sorted(TOP_ROIS_NORM_TO_ORIG.items()):
        # Use base ROI name (hemisphere removed) to fetch color so L/R share same color
        roi_color = TOP_ROI_COLOR_MAP.get(_base_roi_name(original_roi_name), '#000000')
        
        # Find the matching ROI in max_vals for THIS hemisphere only
        matching_roi = None
        for roi in max_vals_x.keys():
            # Check if ROI matches AND is in the correct hemisphere
            if normalize_roi_name(roi) == normalized_roi and roi.startswith(hemi_prefix):
                matching_roi = roi
                break
        
        if matching_roi is None:
            # ROI might not exist in this hemisphere (e.g., L-AngGyrus won't be in Right hemisphere data)
            continue
        
        if matching_roi not in max_vals_x or matching_roi not in max_vals_y:
            continue
        
        # Plot left and right trials
        trial_points = {}
        for trial_side in ['left', 'right']:
            # Find matching trial
            x_trial = None
            y_trial = None
            
            for trial in max_vals_x[matching_roi].keys():
                if trial_side in trial.lower():
                    x_trial = trial
                    break
            
            for trial in max_vals_y[matching_roi].keys():
                if trial_side in trial.lower():
                    y_trial = trial
                    break
            
            if x_trial and y_trial:
                x_val = max_vals_x[matching_roi][x_trial]['max_val']
                y_val = max_vals_y[matching_roi][y_trial]['max_val']
                all_vals.extend([x_val, y_val])
                trial_points[trial_side] = (x_val, y_val)
                
                # Filled for left trial, open for right trial
                if trial_side == 'left':
                    ax.scatter(x_val, y_val,
                              color=roi_color, s=1050, alpha=0.8,
                              edgecolors='black', linewidths=5.25,
                              marker='o', zorder=3)
                else:  # right trial
                    ax.scatter(x_val, y_val,
                              facecolors='white', s=1050, alpha=0.9,
                              edgecolors=roi_color, linewidths=8.25,
                              marker='o', zorder=3)

        if add_dotted_trial_connector and 'left' in trial_points and 'right' in trial_points:
            left_pt = trial_points['left']
            right_pt = trial_points['right']
            ax.plot([left_pt[0], right_pt[0]], [left_pt[1], right_pt[1]],
                    linestyle=':', color=roi_color, linewidth=5.25, alpha=0.9, zorder=2)

    # Matched fixed-step ticks. xlim/ylim and/or tick_step may be provided
    # to force comparable axes across panels (e.g. the overt-vs-covert L/R pair).
    if all_vals:
        from matplotlib.ticker import MultipleLocator, FormatStrFormatter
        lo, hi = min(all_vals), max(all_vals)
        span = max(hi - lo, 1e-9)
        margin = span * 0.15
        if tick_step is not None:
            step = tick_step
        else:
            raw_step = (span + 2 * margin) / 5
            mag = 10 ** np.floor(np.log10(abs(raw_step)))
            norm = raw_step / mag
            if norm < 1.5:
                step = 1 * mag
            elif norm < 3:
                step = 2 * mag
            elif norm < 7:
                step = 5 * mag
            else:
                step = 10 * mag
        if xlim is not None:
            x_lo, x_hi = xlim
        else:
            x_lo = np.floor((lo - margin) / step) * step
            x_hi = np.ceil((hi + margin) / step) * step
        if ylim is not None:
            y_lo, y_hi = ylim
        else:
            y_lo = np.floor((lo - margin) / step) * step
            y_hi = np.ceil((hi + margin) / step) * step
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(y_lo, y_hi)
        ax.xaxis.set_major_locator(MultipleLocator(step))
        ax.yaxis.set_major_locator(MultipleLocator(step))
        decimals = max(0, -int(np.floor(np.log10(step))))
        fmt = f'%.{decimals}f'
        ax.xaxis.set_major_formatter(FormatStrFormatter(fmt))
        ax.yaxis.set_major_formatter(FormatStrFormatter(fmt))

        # Identity line spans the overlapping x/y range
        id_lo = max(x_lo, y_lo)
        id_hi = min(x_hi, y_hi)
        ax.plot([id_lo, id_hi], [id_lo, id_hi],
                'k--', alpha=0.5, linewidth=4.5, zorder=1, label='Identity')

    # Add origin lines
    ax.axhline(0, color='gray', linestyle='-', linewidth=3.75, alpha=0.4, zorder=0)
    ax.axvline(0, color='gray', linestyle='-', linewidth=3.75, alpha=0.4, zorder=0)

    # Formatting
    ax.set_xlabel(f'{x_label}', fontsize=label_fontsize, fontweight='bold')
    ax.set_ylabel(f'{y_label}', fontsize=label_fontsize, fontweight='bold')
    ax.grid(True, linestyle=':', alpha=0.3)
    ax.tick_params(axis='both', which='major',
                   labelsize=tick_fontsize, width=2.5, length=10)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontweight('bold')
    
    # Equal aspect ratio
    ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        svg_path = save_path.replace('.png', '.svg')
        fig.savefig(svg_path, format='svg', bbox_inches='tight')
        pdf_path = save_path.replace('.png', '.pdf')
        fig.savefig(pdf_path, format='pdf', bbox_inches='tight')
        print(f"   ✅ Saved: {Path(save_path).name}")
        plt.close(fig)

    return fig, ax


# Create the condensed top 5 plots (4 total: 2 comparisons × 2 hemispheres)
print("\n" + "="*70)
print("📊 Creating CONDENSED Top 5 ROI scatter plots (4 plots)")
print("   Layout: 2 comparisons × 2 hemispheres")
print("   Same ROI = Same color across hemispheres")
print("   Filled = Left trial, Open = Right trial")
print("="*70)

top5_dir = output_dir / "scatter_plots" / "top5_condensed"
top5_dir.mkdir(parents=True, exist_ok=True)

# Shared tick step within each comparison so left/right hemispheres match.
# For overt-vs-covert we ALSO hardcode shared xlim/ylim so the two panels are
# directly comparable side-by-side in a combined figure.
step_overt_control, _, _ = _compute_top5_axis(max_vals_overt, max_vals_control)
step_overt_covert, _, _ = _compute_top5_axis(max_vals_overt, max_vals_covert)
OC_XLIM = (-5, 11)
OC_YLIM = (-6,8)

# Plot 1: Overt vs Control - Left Hemisphere
print("\n1️⃣ Overt vs Control - Left Hemisphere")
save_path = top5_dir / "top5_overt_vs_control_LEFT_HEMI.png"
create_top5_condensed_scatter(max_vals_overt, max_vals_control,
                              x_label='Overt Δ[HbO] (μM·mm)',
                              y_label='Visual Orientation Δ[HbO] (μM·mm)',
                              hemisphere='left',
                              save_path=str(save_path),
                              add_dotted_trial_connector=True,
                              tick_step=step_overt_control)

# Plot 2: Overt vs Control - Right Hemisphere
print("\n2️⃣ Overt vs Control - Right Hemisphere")
save_path = top5_dir / "top5_overt_vs_control_RIGHT_HEMI.png"
create_top5_condensed_scatter(max_vals_overt, max_vals_control,
                              x_label='Overt Δ[HbO] (μM·mm)',
                              y_label='Visual Orientation Δ[HbO] (μM·mm)',
                              hemisphere='right',
                              save_path=str(save_path),
                              add_dotted_trial_connector=True,
                              tick_step=step_overt_control)

# Plot 3: Overt vs Covert - Left Hemisphere
print("\n3️⃣ Overt vs Covert - Left Hemisphere")
save_path = top5_dir / "top5_overt_vs_covert_LEFT_HEMI.png"
create_top5_condensed_scatter(max_vals_overt, max_vals_covert,
                              x_label='Overt\nΔ[HbO] (μM·mm)',
                              y_label='Covert\nΔ[HbO] (μM·mm)',
                              hemisphere='left',
                              save_path=str(save_path),
                              add_dotted_trial_connector=True,
                              tick_step=step_overt_covert,
                              xlim=OC_XLIM,
                              ylim=OC_YLIM,
                              label_fontsize=40,
                              tick_fontsize=42)

# Plot 4: Overt vs Covert - Right Hemisphere
print("\n4️⃣ Overt vs Covert - Right Hemisphere")
save_path = top5_dir / "top5_overt_vs_covert_RIGHT_HEMI.png"
create_top5_condensed_scatter(max_vals_overt, max_vals_covert,
                              x_label='Overt\nΔ[HbO] (μM·mm)',
                              y_label='Covert\nΔ[HbO] (μM·mm)',
                              hemisphere='right',
                              save_path=str(save_path),
                              add_dotted_trial_connector=True,
                              tick_step=step_overt_covert,
                              xlim=OC_XLIM,
                              ylim=OC_YLIM,
                              label_fontsize=40,
                              tick_fontsize=42)

print(f"\n✅ Condensed Top 5 plots saved to: {top5_dir}")
print(f"   • top5_overt_vs_control_LEFT_HEMI.png/.svg/.pdf")
print(f"   • top5_overt_vs_control_RIGHT_HEMI.png/.svg/.pdf")
print(f"   • top5_overt_vs_covert_LEFT_HEMI.png/.svg/.pdf")
print(f"   • top5_overt_vs_covert_RIGHT_HEMI.png/.svg/.pdf")

#%% Standalone legend

def create_standalone_legend(save_dir):
    """Create a standalone legend figure and save as PNG, SVG, and PDF."""
    legend_handles = []
    legend_labels = []

    # One entry per top ROI (base name, no hemisphere prefix) - deduplicated
    seen_base_names = set()
    for normalized_roi, original_roi_name in sorted(TOP_ROIS_NORM_TO_ORIG.items()):
        base_name = _base_roi_name(original_roi_name)
        if base_name in seen_base_names:
            continue
        seen_base_names.add(base_name)
        roi_color = TOP_ROI_COLOR_MAP.get(base_name, '#000000')
        handle = plt.scatter([], [], color=roi_color, s=400,
                             edgecolors='black', linewidths=2.0, marker='o')
        legend_handles.append(handle)
        legend_labels.append(base_name)

    # Trial type indicators
    left_handle = plt.scatter([], [], color='gray', s=400,
                              edgecolors='black', linewidths=2.0, marker='o')
    legend_handles.append(left_handle)
    legend_labels.append('Left Trial (filled)')

    right_handle = plt.scatter([], [], facecolors='white', s=400,
                               edgecolors='gray', linewidths=3.0, marker='o')
    legend_handles.append(right_handle)
    legend_labels.append('Right Trial (open)')

    # Build a tight figure containing only the legend
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.set_axis_off()
    fig.patch.set_facecolor('none')  # transparent figure background

    legend = ax.legend(legend_handles, legend_labels,
                       loc='center', fontsize=28, title_fontsize=30,
                       framealpha=0, edgecolor='gray',
                       markerscale=1.8)

    fig.tight_layout()

    save_dir = Path(save_dir)
    for ext in ['png', 'svg', 'pdf']:
        out_path = save_dir / f"standalone_legend.{ext}"
        fig.savefig(out_path, format=ext, bbox_inches='tight', dpi=300,
                    transparent=True) 
        print(f"   ✅ Saved standalone legend: {out_path.name}")

    plt.close(fig)


print("\n📄 Creating standalone legend...")
create_standalone_legend(top5_dir)

#%%
