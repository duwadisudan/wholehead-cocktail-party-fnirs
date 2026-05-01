"""
Build the per-subject accuracy summary table from classifier outputs (Figure 4).

Walks the per-subject classifier-result folders, reads each subject's
JSON summary, and assembles a tidy table of accuracies, latencies, and
above-chance flags used by the Figure 4 scatter and latency-CI scripts.

Author: Sudan Duwadi <sudan@bu.edu>
Notes: Code refactoring, documentation, and commenting were AI-assisted;
       all scientific decisions and accountability remain with the author.
"""
#%%
import os
import json
import pandas as pd
import numpy as np

# base directory where per‐subject folders live
BASE = r"U:\eng_research_hrc_binauralhearinglab\Sudan\Labs\Sen Lab\Research_projects\Whole_Head_Cocktail_party\Classifier_script_results\nested\rf_snr_0_20feat_balanced_depth5_oob"

# Threshold for above-chance performance
CHANCE_THRESHOLD = 62.3

def create_time_axis(n_points, sampling_rate=8.98, start_time=-2.0):
    """
    Create time axis matching the sliding window analysis.
    Based on plot_accuracy_over_time.py logic.
    """
    fs = sampling_rate
    window_size_samples = int(round(1.0 * fs))  # ≈ 9 samples
    step_size_samples = int(round(0.5 * fs))    # ≈ 4 samples
    
    total_duration = 17.0  # seconds
    total_samples = int(total_duration * fs)
    
    # Calculate time points for each sample
    t_rel = np.linspace(start_time, start_time + total_duration, total_samples, endpoint=False)
    
    # Calculate window centers
    time_centers = []
    for w in range(n_points):
        start_idx = w * step_size_samples
        end_idx = start_idx + window_size_samples
        if end_idx <= len(t_rel):
            window_center = t_rel[start_idx:end_idx].mean()
            time_centers.append(window_center)
        else:
            window_center = start_time + (w * step_size_samples + window_size_samples/2) / fs
            time_centers.append(window_center)
    
    return np.array(time_centers[:n_points])


def extract_latencies(json_path):
    """
    Extract peak accuracy latency and latency to above-chance from JSON file.
    Only considers time points between 0s and 5s (cue onset to stimulus offset).
    Returns: (peak_acc_percentage, peak_latency, chance_latency)
    """
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Get the first classifier's data (usually 'RF' or 'KNN')
        classifier_keys = list(data.get('mean_acc', {}).keys())
        if not classifier_keys:
            return np.nan, np.nan, np.nan
        
        classifier = classifier_keys[0]
        mean_acc = np.array(data['mean_acc'][classifier])
        
        # Convert to percentage
        mean_acc_pct = mean_acc * 100
        
        # Create time axis
        time_axis = create_time_axis(len(mean_acc))
        
        # Only consider time points between 0s and 5s (cue onset to stimulus offset)
        window_mask = (time_axis >= 0.0) & (time_axis <= 5.0)
        window_indices = np.where(window_mask)[0]
        
        if len(window_indices) == 0:
            return np.nan, np.nan, np.nan
        
        # Filter data to only the 0-5s window
        window_acc = mean_acc_pct[window_mask]
        window_time = time_axis[window_mask]
        
        # Find peak accuracy (only within 0-5s window)
        max_idx_relative = np.argmax(window_acc)
        peak_acc = window_acc[max_idx_relative]
        peak_latency = window_time[max_idx_relative]
        
        # If peak accuracy doesn't exceed chance threshold, set peak_latency to NaN
        if peak_acc < CHANCE_THRESHOLD:
            peak_latency = np.nan
        
        # Find latency to above-chance (first time point exceeding threshold, within 0-5s)
        above_chance_indices = np.where(window_acc >= CHANCE_THRESHOLD)[0]
        if len(above_chance_indices) > 0:
            chance_latency = window_time[above_chance_indices[0]]
        else:
            chance_latency = np.nan  # Will be converted to "N/A" later
        
        return peak_acc, peak_latency, chance_latency
        
    except Exception as e:
        print(f"Error processing {json_path}: {e}")
        return np.nan, np.nan, np.nan


results = {}

# scan each subfolder
for fname in os.listdir(BASE):
    path = os.path.join(BASE, fname)
    if not os.path.isdir(path):
        continue
    parts = fname.split("_")
    if len(parts)!=3 or parts[0]!="sub":
        continue
    subj_id  = int(parts[1])
    run_type = parts[2].lower()   # "overt" or "covert"
    json_path = os.path.join(path, "summary_accuracy.json")
    if not os.path.isfile(json_path):
        continue

    # Extract accuracy and latencies
    peak_acc, peak_latency, chance_latency = extract_latencies(json_path)
    
    # Store all metrics for this subject and run type
    results.setdefault(subj_id, {})[run_type] = {
        'accuracy': peak_acc,
        'peak_latency': peak_latency,
        'chance_latency': chance_latency
    }

# build rows, filling missing with NaN
rows = []
for subj_id, d in sorted(results.items()):
    overt_data = d.get("overt", {})
    covert_data = d.get("covert", {})
    
    rows.append({
        "Subject": subj_id,
        "Overt_perc": overt_data.get('accuracy', np.nan) if isinstance(overt_data, dict) else np.nan,
        "Overt_peak_latency": overt_data.get('peak_latency', np.nan) if isinstance(overt_data, dict) else np.nan,
        "Covert_perc": covert_data.get('accuracy', np.nan) if isinstance(covert_data, dict) else np.nan,
        "Covert_peak_latency": covert_data.get('peak_latency', np.nan) if isinstance(covert_data, dict) else np.nan,
    })
df = pd.DataFrame(rows)

# option 1: only drop subjects with *both* runs missing
df = df.dropna(how="all", subset=["Overt_perc","Covert_perc"])

# now sort by overt accuracy
df = df.sort_values("Overt_perc", ascending=False).reset_index(drop=True)

# Replace NaN with "N/A" for latency columns (but keep NaN for accuracy for proper sorting)
latency_cols = ["Overt_peak_latency", "Covert_peak_latency"]
for col in latency_cols:
    df[col] = df[col].apply(lambda x: "N/A" if pd.isna(x) else f"{x:.2f}")

out_csv = os.path.join(BASE, "final_table.csv")
df.to_csv(out_csv, index=False, float_format="%.4f")
print(f"Wrote summary to {out_csv}")
print(f"\nTable includes:")
print(f"  - Peak accuracy percentage (Overt_perc, Covert_perc)")
print(f"  - Peak accuracy latency in seconds (Overt_peak_latency, Covert_peak_latency)")
print(f"    → Set to 'N/A' if peak accuracy < {CHANCE_THRESHOLD}% (below chance)")
print(f"  - All latencies restricted to 0-5s window (cue onset to stimulus offset)")
print(f"  - 'N/A' indicates peak below threshold or data unavailable")
# %%
