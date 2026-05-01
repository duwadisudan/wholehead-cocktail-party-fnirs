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
from __future__ import annotations

import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict

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
        for _, row in df.iterrows():
            channel = row[channel_col]
            brodmann = row['brodmann']
            scanner = row['scanner']
            
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
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
