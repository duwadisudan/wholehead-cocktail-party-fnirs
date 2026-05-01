"""
Extract Angular Gyrus ROI rows from roi_master.csv (one-time setup utility).

Filters the per-subject roi_master.csv to the Left/Right Angular Gyrus
(BA39) entries and joins against scanner_mni.csv to recover the
subject-specific scanner BA column. Writes per-subject anggyr_sub-XXX.csv.

Author: Sudan Duwadi <sudan@bu.edu>
Notes: Code refactoring was AI-assisted; all scientific decisions and
       accountability remain with the author.
"""

import pandas as pd
from pathlib import Path

# ============================================
# CHANGE THIS FOR EACH SUBJECT
# ============================================
SUBJECT_ID = "sub-663"
# ============================================

# Define base paths
base_dir = Path(r"U:\eng_research_hrc_binauralhearinglab\Sudan\Labs\Sen Lab\Research_projects\Whole_Head_Cocktail_party")
subject_dir = base_dir / "Cocktail_party_whole_head_master_data" / SUBJECT_ID / "nirs" / "atlasviewer_mni"

# Define input and output paths
input_csv = base_dir / "ROIs" / "roi_master.csv"
scanner_csv = subject_dir / "scanner_mni.csv"
output_dir = subject_dir
output_file = output_dir / f"anggyr_{SUBJECT_ID}.csv"

print(f"\n{'='*50}")
print(f"Processing subject: {SUBJECT_ID}")
print(f"{'='*50}")

# Create output directory if it doesn't exist
output_dir.mkdir(parents=True, exist_ok=True)

# Read the CSV files
print(f"Reading from: {input_csv}")
df = pd.read_csv(input_csv)

print(f"Reading scanner data from: {scanner_csv}")
scanner_df = pd.read_csv(scanner_csv)

# Rename column if it starts with '#' (e.g., '#MNIX' -> 'MNIX')
scanner_df.columns = [col.lstrip('#') for col in scanner_df.columns]

# Display the column names for verification
print(f"\nColumns in roi_master.csv: {df.columns.tolist()}")
print(f"Columns in scanner_mni.csv: {scanner_df.columns.tolist()}")
print(f"Total rows in original file: {len(df)}")
print(f"Total rows in scanner file: {len(scanner_df)}")

# First, filter roi_master by brodmann column for Angular Gyrus
second_column = df.columns[1]  # brodmann column
print(f"\nFiltering by column: '{second_column}' for Angular Gyrus")

angular_gyrus_filter = df[second_column].isin(['Left-AngGyrus (39)', 'Right-AngGyrus (39)'])
filtered_df = df[angular_gyrus_filter].copy()

print(f"Rows matching 'Left-AngGyrus (39)' or 'Right-AngGyrus (39)': {len(filtered_df)}")

# Now replace the scanner column with BA from scanner_mni.csv at the SAME indices
if len(df) == len(scanner_df):
    # Get the indices of the filtered rows and use them to get corresponding BA values
    filtered_indices = filtered_df.index
    filtered_df['scanner'] = scanner_df.loc[filtered_indices, 'BA'].values
    print(f"\nReplaced 'scanner' column with subject-specific 'BA' values at matching indices")
else:
    print(f"\n{'='*50}")
    print(f"ERROR: Row count mismatch!")
    print(f"  roi_master.csv has {len(df)} rows")
    print(f"  scanner_mni.csv has {len(scanner_df)} rows")
    print(f"{'='*50}")
    print(f"\nCannot proceed - please regenerate scanner_mni.csv with correct row count.")
    print(f"Exiting without saving output file.")
    import sys
    sys.exit(1)

# Display a preview of the filtered data
if len(filtered_df) > 0:
    print("\nPreview of filtered data:")
    print(filtered_df.head())
else:
    print("\nWarning: No rows matched the filter criteria!")

# Save the filtered data
filtered_df.to_csv(output_file, index=False)
print(f"\nFiltered data saved to: {output_file}")
