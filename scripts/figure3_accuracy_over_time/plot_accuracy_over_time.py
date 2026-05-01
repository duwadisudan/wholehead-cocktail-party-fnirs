#!/usr/bin/env python3
"""
Accuracy-over-time curves with confidence intervals (Figure 3).

For an example subject (or set of subjects), loads the classifier per-fold
accuracy summary JSONs, computes the mean accuracy time course with
bootstrap CI across folds, and renders the time-locked accuracy plot with
stimulus event markers.

Author: Sudan Duwadi <sudan@bu.edu>
Notes: Code refactoring was AI-assisted; all scientific decisions and
       accountability remain with the author.
"""

#%%
import sys
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
#%%
import json
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import argparse
from pathlib import Path


def load_accuracy_data(json_path):
    """Load accuracy data from JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


def calculate_confidence_intervals(mean_acc, ci_acc, confidence_level=0.95):
    """Calculate confidence interval bounds."""
    mean_acc = np.array(mean_acc)
    ci_acc = np.array(ci_acc)
    
    # Calculate upper and lower bounds
    upper_bound = mean_acc + ci_acc
    lower_bound = mean_acc - ci_acc
    
    return upper_bound, lower_bound


def create_time_axis(n_points, sampling_rate=8.98, start_time=-2.0):
    """
    Create time axis for the plot matching the sliding window analysis.
    
    Based on your analysis code:
    - t_rel goes from -2s to +10s 
    - window_size = int(round(1.0 * fs)) ≈ 9 samples
    - step_size = int(round(0.5 * fs)) ≈ 4.5 samples  
    - n_windows = (T - window_size) // step_size + 1
    
    The sliding window centers are calculated as:
    times = [t_rel[w*step_size:(w*step_size+window_size)].mean() for w in range(n_windows)]
    """
    # Match your analysis exactly
    fs = sampling_rate
    window_size_samples = int(round(1.0 * fs))  # ≈ 9 samples
    step_size_samples = int(round(0.5 * fs))    # ≈ 4 samples (rounded down)
    
    # Your t_rel spans from -2s to +15s
    total_duration = 12.0  # seconds  (-2 to +15)
    total_samples = int(total_duration * fs)  # total samples in your -2 to +10 s window
    
    # Calculate time points for each sample
    t_rel = np.linspace(start_time, start_time + total_duration, total_samples, endpoint=False)
    
    # Calculate window centers exactly like your sliding_window_classify function
    time_centers = []
    for w in range(n_points):
        start_idx = w * step_size_samples
        end_idx = start_idx + window_size_samples
        if end_idx <= len(t_rel):
            window_center = t_rel[start_idx:end_idx].mean()
            time_centers.append(window_center)
        else:
            # Fallback if we run out of samples
            window_center = start_time + (w * step_size_samples + window_size_samples/2) / fs
            time_centers.append(window_center)
    
    return np.array(time_centers[:n_points])


def plot_accuracy_over_time(data, output_path, title="Accuracy Over Time for 1 Channel", 
                          sampling_rate=8.98, start_time=-2.0, event_times=None):
    """
    Create accuracy over time plot.
    
    Parameters:
    -----------
    data : dict
        Dictionary containing mean_acc, ci_acc, and max_acc
    output_path : str
        Path to save the plot
    title : str
        Plot title
    sampling_rate : float
        Sampling rate in Hz
    start_time : float
        Start time in seconds
    event_times : dict
        Dictionary with event times (e.g., {'cue_onset': -1, 'stimulus_onset': 2, 'stimulus_offset': 5})
    """
    
    # Extract data for the first classifier (assuming KNN or first available)
    classifier_keys = list(data['mean_acc'].keys())
    if not classifier_keys:
        raise ValueError("No classifier data found in JSON file")
    
    classifier = classifier_keys[0]
    mean_acc = data['mean_acc'][classifier]
    ci_acc = data['ci_acc'][classifier]
    
    # Create time axis
    time = create_time_axis(len(mean_acc), sampling_rate, start_time)
    
    # Calculate confidence intervals
    upper_bound, lower_bound = calculate_confidence_intervals(mean_acc, ci_acc)
    
    # Publication style constants — half-column (44 mm) version
    # table_maker uses 16/18/15/13 pt at figsize=(7,5); scaled to 3.5 in wide (~0.5x)
    BASE_FONT_SIZE   = 8
    AXIS_LABEL_FONT  = 9
    AXIS_TICK_FONT   = 7
    LEGEND_FONT      = 6

    # Create the plot
    plt.rcParams.update({
        'font.size':            BASE_FONT_SIZE,
        'font.family':          'sans-serif',
        'font.sans-serif':      ['Arial'],
        'axes.linewidth':       0.75,
        'xtick.major.width':    0.75,
        'ytick.major.width':    0.75,
        'xtick.major.size':     3,
        'ytick.major.size':     3,
    })
    # Half-column width: table_maker figsize=(7,10) → 88 mm → 3.5 in = 44 mm (half column)
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    
    # Convert to percentage
    mean_acc_pct = np.array(mean_acc) * 100
    upper_bound_pct = np.array(upper_bound) * 100
    lower_bound_pct = np.array(lower_bound) * 100
    
    # Plot confidence interval as shaded area (light gray)
    ax.fill_between(time, lower_bound_pct, upper_bound_pct, alpha=0.25, color='gray')

    # Plot mean accuracy
    ax.plot(time, mean_acc_pct, 'k-', linewidth=1.25)
    
    # Add horizontal reference lines
    lower_chance_level = 0.4047 * 100  # Convert to percentage
    upper_chance_level = 0.6233 * 100  # Convert to percentage
    ax.axhline(y=upper_chance_level, color='gray', linestyle='--', linewidth=1.0,
               alpha=0.7, label='Chance Upper')
    ax.axhline(y=lower_chance_level, color='gray', linestyle='-.', linewidth=1.0,
               alpha=0.7, label='Chance Lower')

    # Add event markers if provided
    if event_times is None:
        # Default event times based on the example image
        event_times = {
            'cue_onset': 0,
            'stimulus_onset': 2, 
            'stimulus_offset': 5
        }
    
    # Dictionary for event labels to display below x-axis
    event_labels = {
        'cue_onset': 'Cue onset',
        'stimulus_onset': 'Movie onset',
        'stimulus_offset': 'Movie offset'
    }
    
    # Colors matching table_maker style
    event1_color = '#4DAF4A'  # Vibrant green - Cue Onset
    event2_color = '#984EA3'  # Vibrant purple - Movie Onset
    event3_color = '#FF7F00'  # Vibrant orange - Movie Offset
    
    # Add vertical dashed lines for events (with labels for legend)
    ax.axvline(x=event_times['cue_onset'],      color=event1_color, linestyle='--', linewidth=1.25, alpha=0.8, label='Cue Onset')
    ax.axvline(x=event_times['stimulus_onset'],  color=event2_color, linestyle='--', linewidth=1.25, alpha=0.9, label='Movie Onset')
    ax.axvline(x=event_times['stimulus_offset'], color=event3_color, linestyle='--', linewidth=1.25, alpha=0.8, label='Movie Offset')
    
    # Formatting
    ax.set_xlabel('Time (s)', fontsize=AXIS_LABEL_FONT, fontweight='bold')
    ax.set_ylabel('Classification Accuracy (%)', fontsize=AXIS_LABEL_FONT, fontweight='bold')
    ax.grid(True, alpha=0.25, linestyle=':', linewidth=0.4)

    # Add legend for event markers
    ax.legend(loc='upper right', frameon=True, edgecolor='black', fontsize=LEGEND_FONT, framealpha=0.95)

    # Set axis limits
    ax.set_xlim(time[0], time[-1])
    # Set y-axis to focus on relevant data range while keeping chance levels visible
    y_min = min(lower_chance_level - 2, min(lower_bound_pct) - 2)  # Include lower chance level (in percentage)
    ax.set_ylim(y_min, 100)

    ax.tick_params(axis='both', labelsize=AXIS_TICK_FONT)
    ax.set_facecolor('#FAFAFA')
    
    # Tight layout
    fig.tight_layout()

    # Save PNG, PDF, SVG
    base_path = os.path.splitext(output_path)[0]
    fig.savefig(output_path, format='png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"Plot saved to: {output_path}")
    fig.savefig(base_path + '.pdf', format='pdf', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"Plot saved to: {base_path + '.pdf'}")
    fig.savefig(base_path + '.svg', format='svg', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"Plot saved to: {base_path + '.svg'}")
    
    # Show max accuracy info
    if 'max_acc' in data and classifier in data['max_acc']:
        max_acc = data['max_acc'][classifier]
        max_time_idx = np.argmax(mean_acc)
        max_time = time[max_time_idx]
        print(f"Maximum accuracy: {max_acc:.3f} at time {max_time:.1f}s")
    
    return plt.gcf()


def main():
    """Main function to run the plotting script."""
    parser = argparse.ArgumentParser(description='Create accuracy over time plots from JSON files')
    parser.add_argument('json_path', help='Path to the JSON file containing accuracy data')
    parser.add_argument('--output', '-o', help='Output path for the plot (optional)')
    parser.add_argument('--title', '-t', default='Accuracy Over Time for 1 Channel', 
                       help='Plot title')
    parser.add_argument('--sampling-rate', '-sr', type=float, default=8.98,
                       help='Sampling rate in Hz (default: 8.98)')
    parser.add_argument('--start-time', '-st', type=float, default=-2.0,
                       help='Start time in seconds (default: -2.0)')
    
    args = parser.parse_args()
    
    # Load data
    try:
        data = load_accuracy_data(args.json_path)
    except FileNotFoundError:
        print(f"Error: JSON file not found at {args.json_path}")
        return
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON file at {args.json_path}")
        return
    
    # Determine output path
    if args.output:
        output_path = args.output
    else:
        # Save in the same directory as the JSON file
        json_dir = os.path.dirname(args.json_path)
        json_name = os.path.splitext(os.path.basename(args.json_path))[0]
        output_path = os.path.join(json_dir, f"{json_name}_accuracy_plot.png")
    
    # Create the plot
    try:
        fig = plot_accuracy_over_time(
            data, 
            output_path, 
            title=args.title,
            sampling_rate=args.sampling_rate,
            start_time=args.start_time
        )
        plt.show()
    except Exception as e:
        print(f"Error creating plot: {e}")


def plot_from_json_file():
    """Load and plot from the specific JSON file provided by user."""
    
    json_path = r"U:\eng_research_hrc_binauralhearinglab\Sudan\Labs\Sen Lab\Research_projects\Whole_Head_Cocktail_party\Classifier_script_results\nested\rf_snr_0_20feat_balanced_depth5_oob\sub_10_overt\summary_accuracy.json"
    
    try:
        # Load data from JSON file
        data = load_accuracy_data(json_path)
        
        # Test time axis calculation
        n_points = len(data['mean_acc']['RF'])
        print(f"Number of data points: {n_points}")
        
        time_axis = create_time_axis(n_points, sampling_rate=8.98, start_time=-2.0)
        print(f"Time axis range: {time_axis[0]:.2f}s to {time_axis[-1]:.2f}s")
        print(f"Time step: {time_axis[1] - time_axis[0]:.2f}s")
        print(f"First few time points: {time_axis[:5]}")
        print(f"Last few time points: {time_axis[-5:]}")
        
        # Create output path in the same directory as the JSON file
        json_dir = os.path.dirname(json_path)
        output_path = os.path.join(json_dir, "summary_accuracy_plot.png")
        
        # Create plot with custom title
        title = "Accuracy Over Time - Subject 10 Overt"
        fig = plot_accuracy_over_time(data, output_path, title=title)
        
        print(f"Successfully created plot from: {json_path}")
        return fig
        
    except FileNotFoundError:
        print(f"Error: Could not find JSON file at {json_path}")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None





if __name__ == "__main__":
    # Check if running with command line arguments
    if len(os.sys.argv) > 1:
        main()  # Use command line interface
    else:
        # Run the direct JSON file plotting function
        print("Running plot_from_json_file()...")
        plot_from_json_file()
