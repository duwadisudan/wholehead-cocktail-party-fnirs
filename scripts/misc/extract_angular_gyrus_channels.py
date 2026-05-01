#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract per-fold Angular Gyrus channel selections from classifier outputs.

For each subject and condition, walks the classifier `dprime_pca_summary.json`
fold records, joins against the channel-to-Brodmann ROI mapping in
roi_master.csv, and emits per-subject CSVs of channels that appeared in any
outer fold and map to the Angular Gyrus (left or right). Outputs include
per-channel fold-presence and importance summaries plus a per-subject
hemisphere-level aggregate table.

Author: Sudan Duwadi <sudan@bu.edu>
Notes: Code refactoring was AI-assisted; all scientific decisions and
       accountability remain with the author.

  angular_gyrus_presence_<condition>.json
    Quick JSON stats: number of subjects with any Angular Gyrus channels, total unique channels, etc.

Usage
-----
Default paths:
    python extract_angular_gyrus_channels.py

Custom:
    python extract_angular_gyrus_channels.py \
        --base-dir "U:/.../Classifier_script_results/nested/rf_control_all_ch_PCA_dprime_contribution_slope" \
        --roi-csv  "U:/.../ROIs/roi_master.csv" \
        --subjects 01 02 03 \
        --conditions overt covert \
        --out-dir  "U:/.../Classifier_script_results/angular_gyrus_channels"

Notes
-----
- channel_idx is as stored in the JSON (original index after intersecting common channels in the main script).
- If channel_name is absent from the ROI CSV it cannot be mapped and is excluded.
- Hemisphere is inferred from the brodmann / region string prefix (Left-/Right-).

Author: Post-processing assistant (2025-10-06)
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Tuple
import pandas as pd

# Keywords appearing in the ROI CSV for Angular Gyrus.
# User clarified they appear as "L-AngGyrus" / "R-AngGyrus".
# We also keep a more verbose fallback in case of earlier naming.
ANGULAR_KEYWORDS = ("AngGyrus", "Angular Gyrus")  # order: specific -> generic

# ---------------------------------------------------------------------------
# ROI mapping
# ---------------------------------------------------------------------------

def load_roi_mapping(roi_csv: str) -> Dict[str, str]:
    try:
        df = pd.read_csv(roi_csv)
        if not {'channel_label', 'brodmann'}.issubset(df.columns):
            raise ValueError("ROI CSV must contain channel_label and brodmann columns")
        mapping = dict(zip(df['channel_label'].astype(str), df['brodmann'].astype(str)))
        print(f"✓ Loaded ROI map ({len(mapping)} entries)")
        return mapping
    except Exception as e:
        print(f"✗ Failed to load ROI CSV {roi_csv}: {e}")
        return {}


def is_angular(region_name: str) -> bool:
    """Return True if ROI label corresponds to Angular Gyrus.

    Matches compact form (e.g., L-AngGyrus / R-AngGyrus) or verbose spelling.
    Case-insensitive substring check using the ANGULAR_KEYWORDS list.
    """
    if not isinstance(region_name, str):
        return False
    rlow = region_name.lower()
    return any(kw.lower() in rlow for kw in ANGULAR_KEYWORDS)


def hemisphere_from_region(region_name: str) -> str:
    """Infer hemisphere from ROI label.

    Supports prefixes: Left-, Right-, L-, R- (case-insensitive).
    Returns 'L', 'R', or 'U' (unknown).
    """
    if not isinstance(region_name, str):
        return 'U'
    rl = region_name.lower()
    if rl.startswith('left-') or rl.startswith('l-'):
        return 'L'
    if rl.startswith('right-') or rl.startswith('r-'):
        return 'R'
    return 'U'

# ---------------------------------------------------------------------------
# JSON reading per subject-condition
# ---------------------------------------------------------------------------

def load_fold_importance(base_dir: str, subject_id: str, condition: str):
    path = os.path.join(base_dir, f"sub_{subject_id}_{condition}", 'dprime_pca_summary.json')
    if not os.path.exists(path):
        print(f"  ✗ Missing {path}")
        return None
    try:
        with open(path, 'r') as f:
            data = json.load(f)
        return data.get('overall_channel_importance_per_fold', [])
    except Exception as e:
        print(f"  ✗ Error reading {path}: {e}")
        return None

# ---------------------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------------------

def extract_angular_channels(folds: List[List[dict]], roi_map: Dict[str, str]):
    """Return list of per-channel stats restricted to angular gyrus."""
    if not folds:
        return []
    total_folds = len(folds)
    # Accumulate per channel name
    agg = {}
    for fi, fold in enumerate(folds):
        for entry in fold:
            ch_name = str(entry.get('channel_name', entry.get('channel_idx', 'NA')))
            region = roi_map.get(ch_name)
            if not is_angular(region):
                continue
            imp = float(entry.get('importance_pct', 0.0))
            rec = agg.setdefault(ch_name, {'folds': {}, 'region': region, 'channel_idx': entry.get('channel_idx', None)})
            rec['folds'][fi] = rec['folds'].get(fi, 0.0) + imp
    rows = []
    for ch_name, rec in agg.items():
        folds_present = rec['folds']
        n_present = len(folds_present)
        total_imp = sum(folds_present.values())
        mean_imp = total_imp / n_present if n_present else 0.0
        hemis = hemisphere_from_region(rec['region'])
        rows.append({
            'channel_name': ch_name,
            'channel_idx': rec['channel_idx'],
            'region': rec['region'],
            'hemisphere': hemis,
            'n_folds_present': n_present,
            'total_folds': total_folds,
            'fold_presence_pct': n_present / total_folds * 100.0,
            'mean_importance_pct': mean_imp,
            'total_importance_pct': total_imp,
            'per_fold_importance_json': json.dumps({int(k): float(v) for k, v in sorted(folds_present.items())}),
        })
    rows.sort(key=lambda r: (r['hemisphere'], -r['mean_importance_pct']))
    return rows

# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Extract Angular Gyrus channel indices across folds.")
    p.add_argument('--base-dir', type=str, required=False,
                   default=r"U:\\eng_research_hrc_binauralhearinglab\\Sudan\\Labs\\Sen Lab\\Research_projects\\Whole_Head_Cocktail_party\\Classifier_script_results\\nested\\rf_snr_0_20feat_balanced_depth5_oob",
                   help='Base directory with sub_<ID>_<condition> result folders.')
    p.add_argument('--roi-csv', type=str, required=False,
                   default=r"U:\\eng_research_hrc_binauralhearinglab\\Sudan\\Labs\\Sen Lab\\Research_projects\\Whole_Head_Cocktail_party\\ROIs\\roi_master.csv",
                   help='ROI mapping CSV (channel_label, brodmann).')
    p.add_argument('--subjects', nargs='*', default=['01','02','03','04','05','10','11','12','13','14','15','18','20','22','25','28','30','31','32','33','34','35','39','41','44','47'],
                   help='Subject IDs to process.')
    p.add_argument('--conditions', nargs='*', default=['overt','covert'],
                   help='Conditions (folder suffix).')
    p.add_argument('--out-dir', type=str, required=False,
                   default=r"U:\\eng_research_hrc_binauralhearinglab\\Sudan\\Labs\\Sen Lab\\Research_projects\\Whole_Head_Cocktail_party\\Classifier_script_results\\angular_gyrus_channels_snr_0_20feat_balanced_depth5_oob",
                   help='Output base directory.')
    return p.parse_args()


def main():
    args = parse_args()
    roi_map = load_roi_mapping(args.roi_csv)
    os.makedirs(args.out_dir, exist_ok=True)

    for condition in args.conditions:
        cond_out = os.path.join(args.out_dir, condition)
        os.makedirs(cond_out, exist_ok=True)
        print(f"\n=== Condition: {condition} ===")
        per_subject_rows = []
        subject_summary = []
        for sid in args.subjects:
            folds = load_fold_importance(args.base_dir, sid, condition)
            if folds is None:
                continue
            rows = extract_angular_channels(folds, roi_map)
            if not rows:
                print(f"  sub-{sid}: (no Angular Gyrus channels)")
                continue
            for r in rows:
                r['subject_id'] = sid
                r['condition'] = condition
            df_sub = pd.DataFrame(rows)
            df_sub.to_csv(os.path.join(cond_out, f'angular_gyrus_sub-{sid}_{condition}.csv'), index=False)
            print(f"  sub-{sid}: {len(df_sub)} Angular Gyrus channels saved")
            per_subject_rows.append(df_sub)
            # summary per subject
            subject_summary.append({
                'subject_id': sid,
                'condition': condition,
                'n_channels': len(df_sub),
                'n_left': int((df_sub.hemisphere=='L').sum()),
                'n_right': int((df_sub.hemisphere=='R').sum()),
                'mean_presence_pct': float(df_sub.fold_presence_pct.mean()),
                'mean_mean_importance_pct': float(df_sub.mean_importance_pct.mean()),
            })
        # condition-level aggregation
        if per_subject_rows:
            big = pd.concat(per_subject_rows, ignore_index=True)
            big.to_csv(os.path.join(cond_out, f'angular_gyrus_channels_{condition}.csv'), index=False)
            summary_df = pd.DataFrame(subject_summary)
            summary_df.to_csv(os.path.join(cond_out, f'angular_gyrus_summary_{condition}.csv'), index=False)
            # quick JSON stats
            stats = {
                'condition': condition,
                'n_subjects_with_any': int(summary_df['subject_id'].nunique()),
                'total_unique_channels': int(big['channel_name'].nunique()),
                'total_entries': int(len(big)),
            }
            with open(os.path.join(cond_out, f'angular_gyrus_presence_{condition}.json'), 'w') as f:
                json.dump(stats, f, indent=2)
        else:
            print(f"  No Angular Gyrus channels found in any subject for {condition}.")

    print("\n✅ Done (Angular Gyrus extraction).")


if __name__ == '__main__':
    main()
