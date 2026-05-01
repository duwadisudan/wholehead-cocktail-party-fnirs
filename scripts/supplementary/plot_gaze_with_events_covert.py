"""
Batch plot gaze (gaze2dX) with event markers — covert task (supplementary).

For each subject and run in the configured list, loads the aligned physio
TSV and events TSV for the covert task and produces a per-run figure of
the gaze time course with stimulus event markers overlaid. Missing files
are skipped.

Author: Sudan Duwadi <sudan@bu.edu>
Notes: Code refactoring, documentation, and commenting were AI-assisted;
       all scientific decisions and accountability remain with the author.
"""
#%%
import os
import traceback
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for batch plotting
import matplotlib.pyplot as plt

# CONFIGURATION

#%% SUBJECT LIST -- edit this list to process different subjects
SUBJECTS = [
# '30','32','31','32','33','34','35','39','41','44','47'
'31', '47'
    # add more subject IDs here, e.g. "31", "32", ...
]

# RUNS TO PROCESS
# Each entry: (task_name_in_snirf, run_number)
# If the physio or events file doesn't exist for a subject, that run is skipped.
RUNS = [
    ("overt",   "01"),
    ("overt",   "02"),
    ("covert",  "01"),
    ("covert",  "02"),
    ("control", "01"),
    ("resting", "01"),
]

# BASE DIRECTORY (where sub-XX/nirs/ folders live)
DATA_BASE = r"U:\eng_research_hrc_binauralhearinglab\Sudan\Labs\Sen Lab\Research_projects\Whole_Head_Cocktail_party\Cocktail_party_whole_head_master_data"

# OUTPUT DIRECTORY FOR PLOTS
OUT_DIR = r"U:\eng_research_hrc_binauralhearinglab\Sudan\Labs\Sen Lab\Research_projects\Whole_Head_Cocktail_party\Cocktail_party_whole_head_master_data\derivatives\eye_tracking"

# PLOTTING FUNCTION

#%%
def plot_one(subj, task, run):
    """Plot gaze + events for one subject/task/run. Returns True on success."""
    id_str = f"sub-{subj}_task-{task}_run-{run}"
    nirs_dir = os.path.join(DATA_BASE, f"sub-{subj}", "nirs")

    physio_path = os.path.join(nirs_dir, f"{id_str}_recording-eyetracking_physio.tsv")
    events_path = os.path.join(nirs_dir, f"{id_str}_events.tsv")

    # check existence
    if not os.path.isfile(physio_path):
        print(f"  [SKIP] Physio not found: {physio_path}")
        return False
    if not os.path.isfile(events_path):
        print(f"  [SKIP] Events not found: {events_path}")
        return False

    # load data
    df_physio = pd.read_csv(physio_path, sep='\t')
    df_events = pd.read_csv(events_path, sep='\t')
    print(f"  Loaded {len(df_physio)} physio samples, {len(df_events)} events")

    # plot
    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(df_physio['timestamps'], df_physio['gaze2dX'],
            linewidth=0.8, alpha=0.7, label='Gaze X position')

    # Track which labels have been added to avoid legend duplicates
    labels_added = set()

    for idx, row in df_events.iterrows():
        onset = row['onset']
        trial_type = row['trial_type']

        if 'Left' in str(trial_type):
            color = 'blue'
            legend_key = 'Left'
        elif 'Right' in str(trial_type):
            color = 'red'
            legend_key = 'Right'
        else:
            color = 'gray'
            legend_key = str(trial_type)

        label = legend_key if legend_key not in labels_added else None
        labels_added.add(legend_key)

        ax.axvline(x=onset, color=color, linestyle='--', alpha=0.7, linewidth=1.5, label=label)

    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Gaze X Position (pixels)', fontsize=12)
    ax.set_title(f'Gaze Position (X) with Events: {id_str}', fontsize=14)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # save to subject-specific figures folder
    subj_fig_dir = os.path.join(OUT_DIR, f"sub-{subj}", "figures")
    os.makedirs(subj_fig_dir, exist_ok=True)
    out_file = os.path.join(subj_fig_dir, f"{id_str}_gaze_with_events.png")
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved plot: {out_file}")
    return True


# BATCH LOOP

#%%
if __name__ == "__main__":
    summary = {"success": [], "skipped": [], "failed": []}

    for subj in SUBJECTS:
        for task, run in RUNS:
            label = f"sub-{subj}_task-{task}_run-{run}"
            print(f"\n{'='*60}")
            print(f"Plotting {label}")
            print(f"{'='*60}")
            try:
                ok = plot_one(subj, task, run)
                if ok:
                    summary["success"].append(label)
                else:
                    summary["skipped"].append(label)
            except Exception:
                print(f"  [ERROR] {label} failed:")
                traceback.print_exc()
                summary["failed"].append(label)

    # SUMMARY
    print(f"\n{'='*60}")
    print("BATCH COMPLETE")
    print(f"{'='*60}")
    print(f"  Success : {len(summary['success'])}")
    for s in summary["success"]:
        print(f"    {s}")
    print(f"  Skipped : {len(summary['skipped'])}")
    for s in summary["skipped"]:
        print(f"    {s}")
    print(f"  Failed  : {len(summary['failed'])}")
    for s in summary["failed"]:
        print(f"    {s}")

# %%
