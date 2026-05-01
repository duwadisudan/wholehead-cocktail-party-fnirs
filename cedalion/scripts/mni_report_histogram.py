"""
MNI Label Switch Report: Analyze how optode labels change from template (brodmann) 
to subject-specific (scanner) MNI projections.

Reports:


1. Total label switches per subject (brodmann != scanner)
2. Switches that stay within IPL (AngGyrus -> SupramargGyr or vice versa)

Uses anggyr_sub-XXX.csv files which contain:
- brodmann: expected ROI based on template
- scanner: actual ROI based on subject-specific MNI projection

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

import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import defaultdict, Counter

import numpy as np
import matplotlib.pyplot as plt

# Subject MNI filename (used for centroid calculations)
SUBJECT_MNI_FILENAME = "just_mni.csv"

# ============================================
# CONFIGURATION - MODIFY THESE
# ============================================

# List of subject IDs to include in analysis
SUBJECT_IDS = [
    "sub-630",
    "sub-633",
    "sub-635",
    "sub-640",
    "sub-649",
    "sub-650",
    "sub-651",
    "sub-653",
    "sub-660",
    "sub-663",
    # Add more subjects here...
]

# Base directory for data
BASE_DIR = Path(
    r"U:\eng_research_hrc_binauralhearinglab\Sudan\Labs\Sen Lab\Research_projects"
    r"\Whole_Head_Cocktail_party\Cocktail_party_whole_head_master_data"
)

# Output directory
OUTPUT_DIR = BASE_DIR / "derivatives" / "mni_label_switch_report"

# ============================================
# END CONFIGURATION
# ============================================

# IPL regions (Inferior Parietal Lobule)
IPL_KEYWORDS = ["AngGyrus", "SupramargGyr"]


def extract_ba_name(label: str) -> str:
    """Extract the Brodmann area name from a label like 'Left-AngGyrus (39)'."""
    if pd.isna(label) or label == "Outside defined BAs":
        return label if pd.notna(label) else "Unknown"
    # Remove hemisphere prefix and BA number
    # e.g., "Left-AngGyrus (39)" -> "AngGyrus"
    parts = str(label).split("-")
    if len(parts) > 1:
        name_part = parts[1].split("(")[0].strip()
        return name_part
    return str(label).split("(")[0].strip()


def is_ipl_region(label: str) -> bool:
    """Check if a label is within the Inferior Parietal Lobule."""
    if pd.isna(label):
        return False
    label_str = str(label)
    return any(kw in label_str for kw in IPL_KEYWORDS)


def is_same_region(brodmann: str, scanner: str) -> bool:
    """Check if brodmann and scanner labels refer to the same region."""
    if pd.isna(brodmann) or pd.isna(scanner):
        return False
    
    # Extract core region names
    brod_name = extract_ba_name(brodmann)
    scan_name = extract_ba_name(scanner)
    
    # Check if they match (ignoring hemisphere)
    return brod_name == scan_name


def analyze_subject(subject_id: str, base_dir: Path) -> Dict:
    """Analyze label switches for a single subject.
    
    Returns dict with:
    - total_channels: total number of channels in file
    - total_switches: channels where brodmann != scanner
    - switches_still_ipl: switches where scanner is still in IPL
    - switches_outside_ipl: switches where scanner is outside IPL
    - switch_details: list of (channel, brodmann, scanner) tuples for switches
    """
    csv_path = base_dir / subject_id / "nirs" / "atlasviewer_mni" / f"anggyr_{subject_id}.csv"
    
    result = {
        'subject_id': subject_id,
        'file_found': False,
        'total_channels': 0,
        'total_switches': 0,
        'switches_still_ipl': 0,
        'switches_outside_ipl': 0,
        'switches_to_outside_ba': 0,
        'switch_details': [],
        'switch_indices': [],
        'ipl_switch_details': [],
        'outside_ipl_details': [],
        # Gross counts (all channels, not just switches)
        'gross_scanner_ipl': 0,        # Total channels where scanner is in IPL
        'gross_scanner_anggyr': 0,     # Total channels where scanner is AngGyrus
        'gross_scanner_supramarg': 0,  # Total channels where scanner is SupramargGyr
        'stayed_same': 0,              # Channels where brodmann == scanner
        'stayed_anggyr': 0,            # Stayed in AngGyrus (no switch)
    }
    
    if not csv_path.exists():
        print(f"  WARNING: File not found for {subject_id}: {csv_path}")
        return result
    
    try:
        df = pd.read_csv(csv_path)
        result['file_found'] = True
        result['total_channels'] = len(df)
        
        # Check for required columns
        if 'brodmann' not in df.columns or 'scanner' not in df.columns:
            print(f"  ERROR: Missing 'brodmann' or 'scanner' columns in {csv_path}")
            print(f"  Columns found: {df.columns.tolist()}")
            return result
        
        # Get channel label column
        channel_col = 'channel_label' if 'channel_label' in df.columns else df.columns[0]
        
        # Analyze each row
        for idx, row in df.iterrows():
            channel = row[channel_col]
            brodmann = row['brodmann']
            scanner = row['scanner']

            # Determine numeric channel index for centroid alignment
            try:
                channel_idx = int(channel)
            except Exception:
                try:
                    channel_idx = int(idx)
                except Exception:
                    # skip if cannot determine index
                    continue
            
            # Gross counts for scanner column (regardless of switch)
            if is_ipl_region(scanner):
                result['gross_scanner_ipl'] += 1
                if 'AngGyrus' in str(scanner):
                    result['gross_scanner_anggyr'] += 1
                if 'SupramargGyr' in str(scanner):
                    result['gross_scanner_supramarg'] += 1
            
            # Check if it's a switch (brodmann != scanner)
            if not is_same_region(brodmann, scanner):
                result['total_switches'] += 1
                result['switch_details'].append((channel, brodmann, scanner))
                result['switch_indices'].append(channel_idx)
                
                # Check if scanner is still in IPL
                if is_ipl_region(scanner):
                    result['switches_still_ipl'] += 1
                    result['ipl_switch_details'].append((channel, brodmann, scanner))
                else:
                    result['switches_outside_ipl'] += 1
                    result['outside_ipl_details'].append((channel, brodmann, scanner))
                    
                    # Check if it went to "Outside defined BAs"
                    if "Outside defined BAs" in str(scanner):
                        result['switches_to_outside_ba'] += 1
            else:
                # No switch - stayed the same
                result['stayed_same'] += 1
                if 'AngGyrus' in str(scanner):
                    result['stayed_anggyr'] += 1
        
        return result
        
    except Exception as e:
        print(f"  ERROR loading {csv_path}: {e}")
        return result


def generate_report(subject_results: List[Dict], output_dir: Path) -> None:
    """Generate simplified report of label switches and IPL retention."""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("MNI LABEL SWITCH REPORT - IPL RETENTION ANALYSIS")
    report_lines.append("=" * 80)
    report_lines.append("")
    report_lines.append("All channels started as AngGyrus (brodmann/template)")
    report_lines.append("Scanner = subject-specific MNI projection result")
    report_lines.append("")
    
    # Simple summary table
    report_lines.append("-" * 80)
    report_lines.append(f"{'Subject':<12} {'Total':<8} {'Switched':<10} {'Sw->IPL':<10} {'Final IPL':<12} {'IPL %':<10}")
    report_lines.append("-" * 80)
    
    total_channels = 0
    total_switches = 0
    total_switches_to_ipl = 0
    total_gross_ipl = 0
    
    for r in subject_results:
        if not r['file_found']:
            report_lines.append(f"{r['subject_id']:<12} {'FILE NOT FOUND'}")
            continue
        
        ipl_pct = 100 * r['gross_scanner_ipl'] / r['total_channels'] if r['total_channels'] > 0 else 0
        
        report_lines.append(
            f"{r['subject_id']:<12} "
            f"{r['total_channels']:<8} "
            f"{r['total_switches']:<10} "
            f"{r['switches_still_ipl']:<10} "
            f"{r['gross_scanner_ipl']:<12} "
            f"{ipl_pct:<.1f}%"
        )
        
        total_channels += r['total_channels']
        total_switches += r['total_switches']
        total_switches_to_ipl += r['switches_still_ipl']
        total_gross_ipl += r['gross_scanner_ipl']
    
    report_lines.append("-" * 80)
    total_ipl_pct = 100 * total_gross_ipl / total_channels if total_channels > 0 else 0
    report_lines.append(
        f"{'TOTAL':<12} "
        f"{total_channels:<8} "
        f"{total_switches:<10} "
        f"{total_switches_to_ipl:<10} "
        f"{total_gross_ipl:<12} "
        f"{total_ipl_pct:<.1f}%"
    )
    report_lines.append("")
    
    # Column explanations
    report_lines.append("COLUMN DEFINITIONS:")
    report_lines.append("  Total     = Total channels (all expected to be AngGyrus)")
    report_lines.append("  Switched  = Channels where scanner != brodmann (label changed)")
    report_lines.append("  Sw->IPL   = Of those switched, how many landed in IPL (SupramargGyr)")
    report_lines.append("  Final IPL = Total channels in IPL after projection (AngGyrus + SupramargGyr)")
    report_lines.append("  IPL %     = Final IPL retention rate")
    report_lines.append("")
    
    # Summary stats
    report_lines.append("=" * 80)
    report_lines.append("SUMMARY")
    report_lines.append("=" * 80)
    switch_pct = 100 * total_switches / total_channels if total_channels > 0 else 0
    stayed_same = total_channels - total_switches
    report_lines.append(f"  Channels that stayed in AngGyrus: {stayed_same}/{total_channels} ({100-switch_pct:.1f}%)")
    report_lines.append(f"  Channels that switched: {total_switches}/{total_channels} ({switch_pct:.1f}%)")
    if total_switches > 0:
        sw_to_ipl_pct = 100 * total_switches_to_ipl / total_switches
        report_lines.append(f"    - Switched but still in IPL: {total_switches_to_ipl}/{total_switches} ({sw_to_ipl_pct:.1f}%)")
    report_lines.append(f"  FINAL IPL RETENTION: {total_gross_ipl}/{total_channels} ({total_ipl_pct:.1f}%)")
    report_lines.append("")
    
    # Write report to file
    report_text = "\n".join(report_lines)
    report_path = output_dir / "label_switch_report.txt"
    report_path.write_text(report_text)
    print(f"\nReport saved to: {report_path}")
    
    # Also print to console
    print("\n" + report_text)
    
    # Save simplified CSV
    csv_rows = []
    for r in subject_results:
        if not r['file_found']:
            continue
        csv_rows.append({
            'subject_id': r['subject_id'],
            'total_channels': r['total_channels'],
            'switched': r['total_switches'],
            'switched_to_ipl': r['switches_still_ipl'],
            'final_ipl': r['gross_scanner_ipl'],
            'ipl_retention_pct': 100 * r['gross_scanner_ipl'] / r['total_channels'] if r['total_channels'] > 0 else 0,
        })
    
    if csv_rows:
        summary_df = pd.DataFrame(csv_rows)
        summary_csv_path = output_dir / "label_switch_summary.csv"
        summary_df.to_csv(summary_csv_path, index=False, float_format='%.3f')
        print(f"Summary CSV saved to: {summary_csv_path}")
    
    # Detailed switch CSV
    detail_rows = []
    for r in subject_results:
        for channel, brod, scan in r['switch_details']:
            detail_rows.append({
                'subject_id': r['subject_id'],
                'channel': channel,
                'brodmann_expected': brod,
                'scanner_actual': scan,
                'brodmann_short': extract_ba_name(brod),
                'scanner_short': extract_ba_name(scan),
                'still_in_ipl': is_ipl_region(scan),
            })
    
    if detail_rows:
        detail_df = pd.DataFrame(detail_rows)
        detail_csv_path = output_dir / "label_switch_details.csv"
        detail_df.to_csv(detail_csv_path, index=False)
        print(f"Detailed switch CSV saved to: {detail_csv_path}")


def _coerce_mni_coords(df: pd.DataFrame, csv_path: Path) -> np.ndarray:
    """Coerce first 3 columns to float, dropping a header row if present."""
    if df.shape[1] < 3:
        raise ValueError(f"Subject MNI file has unexpected shape: {csv_path} -> {df.shape}")

    # Detect a header row if any of the first 3 cells are non-numeric strings.
    first_row = df.iloc[0, 0:3].tolist()
    has_header = False
    for value in first_row:
        try:
            float(str(value).strip())
        except Exception:
            has_header = True
            break

    if has_header:
        df = df.iloc[1:, :].reset_index(drop=True)

    try:
        return df.iloc[:, 0:3].astype(float).values
    except Exception as exc:
        raise ValueError(f"Failed to coerce MNI coords to float: {csv_path}") from exc


def load_subject_mni(subject_id: str, base_dir: Path) -> Optional[np.ndarray]:
    """Load subject MNI coordinates from `just_mni.csv`.

    Returns Nx3 numpy array or None if missing/error.
    """
    csv_path = base_dir / subject_id / "nirs" / "atlasviewer_mni" / SUBJECT_MNI_FILENAME
    if not csv_path.exists():
        print(f"  WARNING: Subject MNI file not found for {subject_id}: {csv_path}")
        return None

    try:
        df = pd.read_csv(csv_path, header=None)
        return _coerce_mni_coords(df, csv_path)
    except Exception as e:
        print(f"  ERROR loading subject MNI {csv_path}: {e}")
        return None


def load_subject_mni_strict(subject_id: str, base_dir: Path) -> np.ndarray:
    """Strict loader for subject MNI coordinates from `just_mni.csv`.

    Raises on missing files or malformed data.
    """
    csv_path = base_dir / subject_id / "nirs" / "atlasviewer_mni" / SUBJECT_MNI_FILENAME
    if not csv_path.exists():
        raise FileNotFoundError(f"Subject MNI file not found for {subject_id}: {csv_path}")

    df = pd.read_csv(csv_path, header=None)
    return _coerce_mni_coords(df, csv_path)


def compute_roi_switch_histograms(subject_results: List[Dict], output_dir: Path) -> None:
    """Compute per-subject and aggregated ROI counts for switched channels.

    Saves per-subject CSVs, aggregated CSV, and aggregated bar-plot PNG.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    aggregate_counter = Counter()

    for r in subject_results:
        sid = r['subject_id']
        if not r.get('file_found'):
            continue

        roi_counts = Counter()
        for _, _, scanner in r['switch_details']:
            short = extract_ba_name(scanner)
            roi_counts[short] += 1
            aggregate_counter[short] += 1

        per_csv_path = output_dir / f"{sid}_roi_switch_counts.csv"
        if roi_counts:
            pd.DataFrame({'roi': list(roi_counts.keys()), 'count': list(roi_counts.values())})\
                .sort_values('count', ascending=False)\
                .to_csv(per_csv_path, index=False)
            print(f"Per-subject ROI counts saved: {per_csv_path}")
        else:
            pd.DataFrame(columns=['roi', 'count']).to_csv(per_csv_path, index=False)
            print(f"Per-subject ROI counts (empty) saved: {per_csv_path}")

    agg_csv_path = output_dir / "aggregated_roi_switch_counts.csv"
    if aggregate_counter:
        agg_df = pd.DataFrame({'roi': list(aggregate_counter.keys()), 'count': list(aggregate_counter.values())})\
            .sort_values('count', ascending=False)
        agg_df.to_csv(agg_csv_path, index=False)
        print(f"Aggregated ROI counts saved: {agg_csv_path}")

        plt.figure(figsize=(10, 6))
        plt.bar(agg_df['roi'].astype(str), agg_df['count'], color='tab:blue', edgecolor='black')
        plt.xticks(rotation=45, ha='right')
        plt.ylabel('Switched-to Count')
        plt.title('Aggregated ROI counts for Switched Channels')
        plt.tight_layout()
        plot_path = output_dir / "aggregated_roi_switch_counts.png"
        plt.savefig(plot_path, dpi=200)
        plt.close()
        print(f"Aggregated ROI bar plot saved: {plot_path}")
    else:
        pd.DataFrame(columns=['roi', 'count']).to_csv(agg_csv_path, index=False)
        print(f"No switches found. Empty aggregated ROI CSV saved: {agg_csv_path}")


def compute_centroids(subject_results: List[Dict], base_dir: Path, output_dir: Path) -> None:
    """Compute per-subject and global centroids for switched channels and save CSV.

    Also computes std_x/y/z and radial distance summary stats per subject.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    pooled = []

    for r in subject_results:
        sid = r['subject_id']
        row = {
            'subject_id': sid,
            'n_channels_used': 0,
            'centroid_x': np.nan, 'centroid_y': np.nan, 'centroid_z': np.nan,
            'std_x': np.nan, 'std_y': np.nan, 'std_z': np.nan,
            'mean_radial_distance': np.nan,
            'min_radial_distance': np.nan,
            'max_radial_distance': np.nan,
        }

        if not r.get('file_found'):
            rows.append(row)
            continue

        indices = r.get('switch_indices', [])
        if not indices:
            rows.append(row)
            continue

        coords = load_subject_mni(sid, base_dir)
        if coords is None:
            rows.append(row)
            continue

        valid_idx = [i for i in indices if 0 <= i < coords.shape[0]]
        if not valid_idx:
            rows.append(row)
            continue

        pts = coords[valid_idx, :]
        centroid = pts.mean(axis=0)
        stds = pts.std(axis=0)
        radial = np.linalg.norm(pts - centroid[None, :], axis=1)
        mean_radial = radial.mean() if len(radial) > 0 else np.nan
        min_radial = radial.min() if len(radial) > 0 else np.nan
        max_radial = radial.max() if len(radial) > 0 else np.nan

        row.update({
            'n_channels_used': int(pts.shape[0]),
            'centroid_x': float(centroid[0]), 'centroid_y': float(centroid[1]), 'centroid_z': float(centroid[2]),
            'std_x': float(stds[0]), 'std_y': float(stds[1]), 'std_z': float(stds[2]),
            'mean_radial_distance': float(mean_radial),
            'min_radial_distance': float(min_radial),
            'max_radial_distance': float(max_radial),
        })

        rows.append(row)
        pooled.append(pts)

    # Global centroid
    if pooled:
        all_pts = np.vstack(pooled)
        gcent = all_pts.mean(axis=0)
        gstd = all_pts.std(axis=0)
        gradial = np.linalg.norm(all_pts - gcent[None, :], axis=1)
        grand_mean = gradial.mean() if len(gradial) > 0 else np.nan
        grand_min = gradial.min() if len(gradial) > 0 else np.nan
        grand_max = gradial.max() if len(gradial) > 0 else np.nan
        rows.append({
            'subject_id': 'GLOBAL',
            'n_channels_used': int(all_pts.shape[0]),
            'centroid_x': float(gcent[0]), 'centroid_y': float(gcent[1]), 'centroid_z': float(gcent[2]),
            'std_x': float(gstd[0]), 'std_y': float(gstd[1]), 'std_z': float(gstd[2]),
            'mean_radial_distance': float(grand_mean),
            'min_radial_distance': float(grand_min),
            'max_radial_distance': float(grand_max),
        })
    else:
        rows.append({
            'subject_id': 'GLOBAL',
            'n_channels_used': 0,
            'centroid_x': np.nan, 'centroid_y': np.nan, 'centroid_z': np.nan,
            'std_x': np.nan, 'std_y': np.nan, 'std_z': np.nan,
            'mean_radial_distance': np.nan,
            'min_radial_distance': np.nan,
            'max_radial_distance': np.nan,
        })

    cent_df = pd.DataFrame(rows)
    cent_csv = output_dir / "switched_channel_centroids.csv"
    cent_df.to_csv(cent_csv, index=False, float_format='%.6f')
    print(f"Centroid CSV saved: {cent_csv}")


def compute_all_channel_centroids(subject_ids: List[str], base_dir: Path, output_dir: Path) -> None:
    """Compute centroids using all post-registered channels (from just_mni.csv).

    Saves per-subject centroid (mean of all channels) and a GLOBAL pooled centroid.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    pooled = []

    for sid in subject_ids:
        row = {
            'subject_id': sid,
            'n_channels': 0,
            'centroid_x': np.nan, 'centroid_y': np.nan, 'centroid_z': np.nan,
            'std_x': np.nan, 'std_y': np.nan, 'std_z': np.nan,
        }

        coords = load_subject_mni(sid, base_dir)
        if coords is None:
            rows.append(row)
            continue

        if coords.size == 0:
            rows.append(row)
            continue

        centroid = coords.mean(axis=0)
        stds = coords.std(axis=0)

        row.update({
            'n_channels': int(coords.shape[0]),
            'centroid_x': float(centroid[0]), 'centroid_y': float(centroid[1]), 'centroid_z': float(centroid[2]),
            'std_x': float(stds[0]), 'std_y': float(stds[1]), 'std_z': float(stds[2]),
        })

        rows.append(row)
        pooled.append(coords)

    # Global centroid across all subject channels
    if pooled:
        all_pts = np.vstack(pooled)
        gcent = all_pts.mean(axis=0)
        gstd = all_pts.std(axis=0)
        rows.append({
            'subject_id': 'GLOBAL',
            'n_channels': int(all_pts.shape[0]),
            'centroid_x': float(gcent[0]), 'centroid_y': float(gcent[1]), 'centroid_z': float(gcent[2]),
            'std_x': float(gstd[0]), 'std_y': float(gstd[1]), 'std_z': float(gstd[2]),
        })
    else:
        rows.append({
            'subject_id': 'GLOBAL',
            'n_channels': 0,
            'centroid_x': np.nan, 'centroid_y': np.nan, 'centroid_z': np.nan,
            'std_x': np.nan, 'std_y': np.nan, 'std_z': np.nan,
        })

    all_df = pd.DataFrame(rows)
    all_csv = output_dir / "all_channel_centroids_post_registration.csv"
    all_df.to_csv(all_csv, index=False, float_format='%.6f')
    print(f"All-channel centroid CSV saved: {all_csv}")


def compute_ang_gyrus_centroids(subject_ids: List[str], base_dir: Path, output_dir: Path) -> None:
    """Compute centroids for channels that started as AngGyrus (template), using post-registered positions.

    Uses `anggyr_{subject}.csv` to identify which channels are AngGyrus, then loads `just_mni.csv`
    to take their post-registration MNI coordinates. Saves per-subject LEFT/RIGHT rows to CSV.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    left_rows = []
    right_rows = []

    # Load roi_master (uses the exact extraction logic from `extract_angular_gyrus.py`)
    # Try the ROIs directory relative to the provided BASE_DIR. The extract_angular_gyrus
    # script expects roi_master at the parent folder of the subject master data, so
    # check both locations.
    roi_master_path = base_dir / "ROIs" / "roi_master.csv"
    if not roi_master_path.exists():
        alt = base_dir.parent / "ROIs" / "roi_master.csv"
        if alt.exists():
            roi_master_path = alt
        else:
            print(f"  ERROR: roi_master.csv not found at expected locations: {base_dir / 'ROIs' / 'roi_master.csv'} or {base_dir.parent / 'ROIs' / 'roi_master.csv'}")
            print("  Cannot compute AngGyrus centroids without roi_master.csv. Exiting this step.")
            return

    try:
        roi_master = pd.read_csv(roi_master_path)
    except Exception as e:
        raise RuntimeError(f"Failed to read roi_master.csv: {roi_master_path}") from e

    # The original extract_angular_gyrus script treats the 2nd column as the brodmann column
    brod_col = roi_master.columns[1]
    left_mask = roi_master[brod_col].isin(['Left-AngGyrus (39)'])
    right_mask = roi_master[brod_col].isin(['Right-AngGyrus (39)'])
    ang_mask_master = left_mask | right_mask
    ang_master_indices = roi_master.index[ang_mask_master].tolist()
    left_master_indices = roi_master.index[left_mask].tolist()
    right_master_indices = roi_master.index[right_mask].tolist()

    if not ang_master_indices or not left_master_indices or not right_master_indices:
        raise ValueError(
            "Angular Gyrus indices not found for one or more sides in roi_master.csv. "
            "Expected Left-AngGyrus (39) and Right-AngGyrus (39) rows."
        )

    for sid in subject_ids:
        csv_path = base_dir / sid / "nirs" / "atlasviewer_mni" / f"anggyr_{sid}.csv"
        # Instead of relying on the 58-row `anggyr_{sid}.csv` numeric indices, use the
        # indices selected from roi_master (these are the original row positions used
        # by scanner_mni/atlasviewer and therefore index into just_mni.csv).
        indices_all = ang_master_indices
        indices_left = left_master_indices
        indices_right = right_master_indices

        # Load subject MNI coordinates (strict)
        coords = load_subject_mni_strict(sid, base_dir)

        if coords.shape[0] != len(roi_master):
            raise ValueError(
                f"Row count mismatch for {sid}: just_mni.csv has {coords.shape[0]} rows, "
                f"roi_master.csv has {len(roi_master)} rows. "
                "Order and row count must match exactly."
            )

        # Diagnostics: ensure scanner_mni length matches roi_master (this is what
        # `extract_angular_gyrus.py` requires); report mismatch but continue if possible.
        scanner_csv = base_dir / sid / "nirs" / "atlasviewer_mni" / "scanner_mni.csv"
        if scanner_csv.exists():
            try:
                sdf = pd.read_csv(scanner_csv)
            except Exception as e:
                raise RuntimeError(f"Failed to read scanner_mni.csv for {sid}: {scanner_csv}") from e

            if len(sdf) != len(roi_master):
                raise ValueError(
                    f"Row count mismatch for {sid}: scanner_mni.csv has {len(sdf)} rows, "
                    f"roi_master.csv has {len(roi_master)} rows. "
                    "Order and row count must match exactly."
                )

        def compute_stats(label: str, indices: List[int]) -> Tuple[np.ndarray, np.ndarray, float, List[int]]:
            valid = [i for i in indices if 0 <= i < coords.shape[0]]
            if not valid:
                raise ValueError(
                    f"No valid {label} AngGyrus indices within just_mni range for {sid} "
                    f"(max index {coords.shape[0]-1})"
                )
            pts_local = coords[valid, :]
            centroid_local = pts_local.mean(axis=0)
            stds_local = pts_local.std(axis=0)
            radial_local = np.linalg.norm(pts_local - centroid_local[None, :], axis=1)
            mean_radial_local = radial_local.mean() if len(radial_local) > 0 else np.nan
            min_radial_local = radial_local.min() if len(radial_local) > 0 else np.nan
            max_radial_local = radial_local.max() if len(radial_local) > 0 else np.nan
            return centroid_local, stds_local, mean_radial_local, min_radial_local, max_radial_local, valid

        # Left AngGyrus
        centroid_l, stds_l, mean_radial_l, min_radial_l, max_radial_l, valid_left = compute_stats("LEFT", indices_left)
        left_rows.append({
            'subject_id': sid,
            'side': 'LEFT',
            'n_anggyr_channels': int(len(valid_left)),
            'centroid_x': float(centroid_l[0]), 'centroid_y': float(centroid_l[1]), 'centroid_z': float(centroid_l[2]),
            'std_x': float(stds_l[0]), 'std_y': float(stds_l[1]), 'std_z': float(stds_l[2]),
            'mean_radial_distance': float(mean_radial_l),
            'min_radial_distance': float(min_radial_l),
            'max_radial_distance': float(max_radial_l),
        })

        # Right AngGyrus
        centroid_r, stds_r, mean_radial_r, min_radial_r, max_radial_r, valid_right = compute_stats("RIGHT", indices_right)
        right_rows.append({
            'subject_id': sid,
            'side': 'RIGHT',
            'n_anggyr_channels': int(len(valid_right)),
            'centroid_x': float(centroid_r[0]), 'centroid_y': float(centroid_r[1]), 'centroid_z': float(centroid_r[2]),
            'std_x': float(stds_r[0]), 'std_y': float(stds_r[1]), 'std_z': float(stds_r[2]),
            'mean_radial_distance': float(mean_radial_r),
            'min_radial_distance': float(min_radial_r),
            'max_radial_distance': float(max_radial_r),
        })

        # Print per-subject centroids for quick validation
        try:
            print(
                f"  {sid} AngGyrus centroids LEFT/RIGHT (x,y,z): "
                f"{centroid_l[0]:.3f},{centroid_l[1]:.3f},{centroid_l[2]:.3f} | "
                f"{centroid_r[0]:.3f},{centroid_r[1]:.3f},{centroid_r[2]:.3f}"
            )
        except Exception:
            pass

        # Save per-subject mapping diagnostics: master index -> MNI coord
        try:
            mapping_rows = []
            # If roi_master has a channel identifier in column 0, include it
            channel_col_master = roi_master.columns[0] if len(roi_master.columns) > 0 else None
            for mi in indices_all:
                entry = {'master_index': int(mi)}
                if channel_col_master is not None:
                    entry['channel'] = roi_master.iloc[mi, 0]
                # include brodmann and (if possible) scanner BA from scanner_mni.csv
                entry['brodmann_master'] = roi_master.iloc[mi, 1]
                if 'sdf' in locals() and mi < len(sdf) and 'BA' in sdf.columns:
                    entry['scanner_BA'] = sdf.loc[mi, 'BA']
                x, y, z = coords[mi, :]
                entry.update({'mni_x': float(x), 'mni_y': float(y), 'mni_z': float(z)})
                mapping_rows.append(entry)

            map_df = pd.DataFrame(mapping_rows)
            map_csv = output_dir / f"{sid}_ang_gyrus_master_index_map.csv"
            map_df.to_csv(map_csv, index=False, float_format='%.6f')
            print(f"Per-subject AngGyrus mapping saved: {map_csv}")
        except Exception as e:
            print(f"  WARNING: Failed to write mapping for {sid}: {e}")

    # Global centroid for Angular Gyrus channels across subjects
    left_df = pd.DataFrame(left_rows)
    right_df = pd.DataFrame(right_rows)

    left_csv = output_dir / "ang_gyrus_left_centroids_post_registration.csv"
    right_csv = output_dir / "ang_gyrus_right_centroids_post_registration.csv"
    left_df.to_csv(left_csv, index=False, float_format='%.6f')
    right_df.to_csv(right_csv, index=False, float_format='%.6f')
    print(f"AngGyrus LEFT centroid CSV saved: {left_csv}")
    print(f"AngGyrus RIGHT centroid CSV saved: {right_csv}")

    # Headerless XYZ files for BioImage Suite (one row per subject, 3 columns)
    left_xyz = left_df[['centroid_x', 'centroid_y', 'centroid_z']]
    right_xyz = right_df[['centroid_x', 'centroid_y', 'centroid_z']]

    left_xyz_csv = output_dir / "ang_gyrus_left_centroids_xyz_noheader.csv"
    right_xyz_csv = output_dir / "ang_gyrus_right_centroids_xyz_noheader.csv"
    left_xyz.to_csv(left_xyz_csv, index=False, header=False, float_format='%.6f')
    right_xyz.to_csv(right_xyz_csv, index=False, header=False, float_format='%.6f')
    print(f"AngGyrus LEFT XYZ (no header) saved: {left_xyz_csv}")
    print(f"AngGyrus RIGHT XYZ (no header) saved: {right_xyz_csv}")


def main():
    """Main function to run label switch analysis."""
    print("=" * 60)
    print("MNI LABEL SWITCH ANALYSIS")
    print("=" * 60)
    print(f"\nSubjects: {SUBJECT_IDS}")
    print(f"Base directory: {BASE_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    
    # Analyze each subject
    print("\nAnalyzing subjects...")
    results = []
    for subject_id in SUBJECT_IDS:
        print(f"  Processing {subject_id}...")
        result = analyze_subject(subject_id, BASE_DIR)
        results.append(result)
    
    # Generate report
    generate_report(results, OUTPUT_DIR)
    
    # Additional outputs: ROI histograms and centroids (no 3D)
    print("\nComputing ROI histograms and MNI centroids for switched channels...")
    compute_roi_switch_histograms(results, OUTPUT_DIR)
    compute_centroids(results, BASE_DIR, OUTPUT_DIR)
    # Also compute centroids for all registered channels (per-subject and global)
    # Compute centroids for template AngGyrus channels after registration
    compute_ang_gyrus_centroids(SUBJECT_IDS, BASE_DIR, OUTPUT_DIR)
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
