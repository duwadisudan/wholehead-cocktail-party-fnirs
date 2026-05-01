"""
BIDS fNIRS Data Processing Script for Spatial Attention Dataset
================================================================

This script processes fNIRS data in BIDS format from the Spatial Attention dataset.
It performs block averaging for different trial types and creates separate plots
for each subject with different trial types shown in different colors.

Dataset path: U:\eng_research_hrc_binauralhearinglab\Sudan\Labs\Sen Lab\Research_projects\Dataset_paper\Spatial_Attention_fNIRS_Dataset_BIDS
"""
#%%
import os
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import xarray as xr

import cedalion
import cedalion.io
import cedalion.nirs
import cedalion.sigproc.quality as quality
import cedalion.xrutils as xrutils
from cedalion import units

#%%
# ============================================================================
# CONFIGURATION
# ============================================================================

# Path to the BIDS dataset
BIDS_ROOT = Path(r"U:\eng_research_hrc_binauralhearinglab\Sudan\Labs\Sen Lab\Research_projects\Dataset_paper\Spatial_Attention_fNIRS_Dataset_BIDS")

# Output directory for plots
OUTPUT_DIR = Path(r"U:\eng_research_hrc_binauralhearinglab\Sudan\Labs\Sen Lab\Research_projects\Dataset_paper\Spatial_Attention_fNIRS_Dataset_BIDS\derivatives\plots_block_averages")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Processing parameters
EPOCH_BEFORE = 2  # seconds before stimulus
EPOCH_AFTER = 15  # seconds after stimulus
FREQ_FILT_LOW = 0.01  # Hz
FREQ_FILT_HIGH = 0.5  # Hz
BUTTER_ORDER = 4

# Source-detector distance thresholds (cm)
SD_DIST_MIN = 2.0
SD_DIST_MAX = 4.5

# Differential pathlength factors for each wavelength
DPF_VALUES = [1, 1]  # typical values for adult head at 690nm and 830nm

#%%
# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def find_subjects(bids_root):
    """Find all subject directories in BIDS dataset."""
    subjects = []
    for item in bids_root.iterdir():
        if item.is_dir() and item.name.startswith('sub-'):
            subjects.append(item.name)
    return sorted(subjects)


def find_snirf_files(subject_dir):
    """Find all SNIRF files for a subject."""
    nirs_dir = subject_dir / "nirs"
    if not nirs_dir.exists():
        return []
    
    snirf_files = list(nirs_dir.glob("*.snirf"))
    return sorted(snirf_files)


def find_events_file(snirf_file):
    """Find the corresponding events.tsv file for a SNIRF file."""
    # Events file should have the same base name but with _events.tsv instead of _nirs.snirf
    # e.g., sub-08_task-overt_run-03_nirs.snirf -> sub-08_task-overt_run-03_events.tsv
    events_file = snirf_file.parent / snirf_file.name.replace('_nirs.snirf', '_events.tsv')
    if events_file.exists():
        return events_file
    return None


def load_and_process_subject(snirf_path, events_path=None):
    """
    Load and process a single subject's fNIRS data.
    
    Parameters
    ----------
    snirf_path : Path
        Path to the SNIRF file
    events_path : Path, optional
        Path to the events TSV file (if not embedded in SNIRF)
    
    Returns
    -------
    rec : Recording
        Processed recording with block averages
    """
    print(f"  Loading SNIRF file: {snirf_path.name}")
    
    # Load SNIRF file
    try:
        recordings = cedalion.io.read_snirf(snirf_path)
        print(f"  DEBUG: Successfully loaded SNIRF, got {len(recordings)} recording(s)")
    except Exception as e:
        import traceback
        print(f"  ERROR during SNIRF loading!")
        print(f"  Error message: {str(e)}")
        print(f"  Full traceback:")
        traceback.print_exc()
        raise
    
    rec = recordings[0]  # Assume first recording
    
    # DEBUG: Check if stim is embedded in SNIRF
    print(f"  DEBUG: rec.stim from SNIRF: {rec.stim if rec.stim is not None else 'None'}")
    if rec.stim is not None:
        print(f"  DEBUG: Number of events in SNIRF: {len(rec.stim)}")
    
    # Load external events file if provided and stim is empty
    if events_path is not None and events_path.exists():
        print(f"  DEBUG: Events file exists: {events_path.name}")
        if rec.stim is None or len(rec.stim) == 0:
            print(f"  Loading events from: {events_path.name}")
            rec.stim = cedalion.io.bids.read_events_from_tsv(events_path)
            print(f"  DEBUG: Loaded {len(rec.stim)} events from TSV")
    else:
        if events_path is None:
            print(f"  DEBUG: No events file path provided")
        else:
            print(f"  DEBUG: Events file does not exist: {events_path}")
    
    # Check if we have stimulus information
    if rec.stim is None or len(rec.stim) == 0:
        print("  WARNING: No stimulus information found. Skipping subject.")
        return None
    
    # Get unique trial types
    trial_types = rec.stim['trial_type'].unique().tolist()
    print(f"  Found trial types: {trial_types}")
    print(f"  DEBUG: Number of events: {len(rec.stim)}")
    print(f"  DEBUG: First few event onsets: {rec.stim['onset'].head().tolist()}")
    print(f"  DEBUG: Recording duration: {rec['amp'].time.values[-1] - rec['amp'].time.values[0]} seconds")
    
    # Convert amplitude to optical density (for quality checks and later filtering)
    print("  Converting to optical density...")
    rec["od"] = cedalion.nirs.int2od(rec["amp"])
    print(f"  DEBUG: rec['od'] shape: {rec['od'].shape}")
    
    # Apply Beer-Lambert Law to get concentrations
    # Note: beer_lambert takes amplitude data, not OD (it does the conversion internally)
    print("  Applying Beer-Lambert Law...")
    dpf = xr.DataArray(
        DPF_VALUES,
        dims="wavelength",
        coords={"wavelength": rec["amp"].wavelength}
    )
    rec["conc"] = cedalion.nirs.beer_lambert(rec["amp"], rec.geo3d, dpf)
    print(f"  DEBUG: rec['conc'] shape: {rec['conc'].shape}")
    print(f"  DEBUG: rec['conc'] dims: {rec['conc'].dims}")
    if 'chromo' in rec['conc'].coords:
        print(f"  DEBUG: chromophores: {rec['conc'].chromo.values}")
    
    # Frequency filtering
    print("  Applying frequency filter...")
    rec["conc_freqfilt"] = rec["conc"].cd.freq_filter(
        fmin=FREQ_FILT_LOW,
        fmax=FREQ_FILT_HIGH,
        butter_order=BUTTER_ORDER
    )
    
    # Filter channels by source-detector distance
    print("  Filtering channels by source-detector distance...")
    sd_threshs = [SD_DIST_MIN, SD_DIST_MAX] * units.cm
    ch_dist, rec.masks["sd_mask"] = quality.sd_dist(
        rec["conc_freqfilt"], rec.geo3d, sd_threshs
    )
    rec["conc_freqfilt_LD"], _ = xrutils.apply_mask(
        rec["conc_freqfilt"], rec.masks["sd_mask"], "drop", "channel"
    )
    
    # Check if we have channels left after filtering
    print(f"  DEBUG: Channels after SD filtering: {len(rec['conc_freqfilt_LD'].channel)}")
    if len(rec["conc_freqfilt_LD"].channel) == 0:
        print("  WARNING: No channels passed distance filtering. Skipping subject.")
        return None
    
    print(f"  DEBUG: rec['conc_freqfilt_LD'] shape before epoching: {rec['conc_freqfilt_LD'].shape}")
    print(f"  DEBUG: rec['conc_freqfilt_LD'] dims: {rec['conc_freqfilt_LD'].dims}")
    
    # Segment data into epochs
    print("  Creating epochs...")
    print(f"  DEBUG: Epoching with before={EPOCH_BEFORE}s, after={EPOCH_AFTER}s")
    
    try:
        rec["epochs"] = rec["conc_freqfilt_LD"].cd.to_epochs(
            rec.stim,
            trial_types,  # Use all available trial types
            before=EPOCH_BEFORE * units.s,
            after=EPOCH_AFTER * units.s,
        )
    except Exception as e:
        print(f"  DEBUG: Error during epoching!")
        print(f"  DEBUG: Exception type: {type(e).__name__}")
        print(f"  DEBUG: Exception message: {str(e)}")
        
        # Try to manually inspect what would happen
        print(f"  DEBUG: Attempting manual epoch extraction to diagnose...")
        for i, (idx, row) in enumerate(rec.stim.iterrows()):
            onset = row['onset']
            trial_type = row['trial_type']
            print(f"    Event {i}: onset={onset:.2f}s, type={trial_type}")
            
            # Check if epoch would be valid
            start_time = onset - EPOCH_BEFORE
            end_time = onset + EPOCH_AFTER
            data_start = rec['conc_freqfilt_LD'].time.values[0]
            data_end = rec['conc_freqfilt_LD'].time.values[-1]
            
            if start_time < data_start:
                print(f"      WARNING: Start time {start_time:.2f}s < data start {data_start:.2f}s")
            if end_time > data_end:
                print(f"      WARNING: End time {end_time:.2f}s > data end {data_end:.2f}s")
        
        raise
    
    # Check if we have epochs
    if len(rec["epochs"].epoch) == 0:
        print("  WARNING: No epochs found. Skipping subject.")
        return None
    
    # Baseline correction
    print("  Applying baseline correction...")
    baseline_conc = rec["epochs"].sel(
        reltime=(rec["epochs"].reltime < 0)
    ).mean("reltime")
    rec["epochs_blcorrected"] = rec["epochs"] - baseline_conc
    
    # Block average by trial type
    print("  Computing block averages...")
    rec["blockaverage"] = rec["epochs_blcorrected"].groupby("trial_type").mean("epoch")
    
    print("  Processing complete!")
    return rec


def plot_block_averages(rec, subject_id, output_dir):
    """
    Create a plot of block averages for a subject.
    
    Parameters
    ----------
    rec : Recording
        Processed recording with block averages
    subject_id : str
        Subject identifier
    output_dir : Path
        Directory to save the plot
    """
    print(f"  Creating plot for {subject_id}...")
    
    # Get trial types and assign colors
    trial_types = rec["blockaverage"].trial_type.values
    colors = plt.cm.tab10(np.linspace(0, 1, len(trial_types)))
    color_map = {tt: colors[i] for i, tt in enumerate(trial_types)}
    
    # Determine subplot layout
    n_channels = len(rec["blockaverage"].channel)
    n_cols = int(np.ceil(np.sqrt(n_channels)))
    n_rows = int(np.ceil(n_channels / n_cols))
    
    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    fig.suptitle(f'{subject_id} - Block Averages by Trial Type', fontsize=16)
    
    # Flatten axes for easier iteration
    if n_channels == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    # Plot each channel
    for i_ch, channel in enumerate(rec["blockaverage"].channel):
        ax = axes[i_ch]
        
        for trial_type in trial_types:
            color = color_map[trial_type]
            
            # Plot HbO
            ax.plot(
                rec["blockaverage"].reltime,
                rec["blockaverage"].sel(
                    chromo="HbO", trial_type=trial_type, channel=channel
                ),
                color=color,
                linestyle='-',
                linewidth=2,
                alpha=0.8,
                label=f"{trial_type} (HbO)"
            )
            
            # Plot HbR
            ax.plot(
                rec["blockaverage"].reltime,
                rec["blockaverage"].sel(
                    chromo="HbR", trial_type=trial_type, channel=channel
                ),
                color=color,
                linestyle='--',
                linewidth=2,
                alpha=0.6,
                label=f"{trial_type} (HbR)"
            )
        
        # Formatting
        ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
        ax.axvline(x=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
        ax.grid(True, alpha=0.3)
        ax.set_title(f'Channel: {channel.values}')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Δ Concentration (µM)')
    
    # Remove extra subplots
    for i in range(n_channels, len(axes)):
        fig.delaxes(axes[i])
    
    # Create custom legend (show only on first subplot)
    legend_handles = []
    for trial_type in trial_types:
        color = color_map[trial_type]
        legend_handles.append(
            Line2D([0], [0], color=color, lw=2, ls='-', label=f"{trial_type} (HbO)")
        )
        legend_handles.append(
            Line2D([0], [0], color=color, lw=2, ls='--', label=f"{trial_type} (HbR)")
        )
    
    axes[0].legend(handles=legend_handles, loc='best', fontsize='small')
    
    plt.tight_layout()
    
    # Save figure
    output_path = output_dir / f"{subject_id}_block_averages.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Plot saved to: {output_path}")
    
    plt.close(fig)

#%%
# ============================================================================
# MAIN PROCESSING LOOP
# ============================================================================

def main():
    """Main processing function."""
    print("="*70)
    print("BIDS fNIRS Data Processing - Spatial Attention Dataset")
    print("="*70)
    print(f"\nDataset root: {BIDS_ROOT}")
    print(f"Output directory: {OUTPUT_DIR}")
    print()
    
    # Find all subjects
    subjects = find_subjects(BIDS_ROOT)
    print(f"Found {len(subjects)} subjects: {', '.join(subjects)}\n")
    
    if len(subjects) == 0:
        print("ERROR: No subjects found in the BIDS directory.")
        return
    
    # Process each subject
    success_count = 0
    failed_subjects = []
    
    for subject_id in subjects:
        print(f"\n{'='*70}")
        print(f"Processing {subject_id}")
        print(f"{'='*70}")
        
        subject_dir = BIDS_ROOT / subject_id
        
        # Find SNIRF files
        snirf_files = find_snirf_files(subject_dir)
        
        if len(snirf_files) == 0:
            print(f"  WARNING: No SNIRF files found for {subject_id}")
            failed_subjects.append(subject_id)
            continue
        
        print(f"  Found {len(snirf_files)} SNIRF file(s)")
        
        # Process first SNIRF file (can be extended to handle multiple runs)
        snirf_path = snirf_files[0]
        events_path = find_events_file(snirf_path)
        
        try:
            # Load and process
            rec = load_and_process_subject(snirf_path, events_path)
            
            if rec is not None:
                # Create plot
                plot_block_averages(rec, subject_id, OUTPUT_DIR)
                success_count += 1
            else:
                failed_subjects.append(subject_id)
                
        except Exception as e:
            import traceback
            print(f"  ERROR: Failed to process {subject_id}")
            print(f"  Error message: {str(e)}")
            print(f"  Full traceback:")
            traceback.print_exc()
            failed_subjects.append(subject_id)
    
    # Summary
    print(f"\n{'='*70}")
    print("PROCESSING COMPLETE")
    print(f"{'='*70}")
    print(f"Successfully processed: {success_count}/{len(subjects)} subjects")
    
    if failed_subjects:
        print(f"\nFailed subjects: {', '.join(failed_subjects)}")
    
    print(f"\nPlots saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

# %%
