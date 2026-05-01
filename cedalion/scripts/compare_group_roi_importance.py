#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare group-level ROI importance between:
  1) PC-contribution-based ROI analysis (mean_importance % per ROI)
  2) Channel-stability-based ROI analysis (e.g., median_dprime per ROI)

This script reads the group CSV outputs from both analyses, aligns ROIs,
selects top ROIs (by contribution or stability), and plots scatter comparisons
to assess similarity. Also saves a merged CSV for transparency.

Created: Aug 29, 2025
Authors: lcarlton & sudan duwadi
"""

import os
import sys
import json
import re
from types import SimpleNamespace
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('default')
sns.set_palette('tab10')


def load_pc_contrib_group_csv(base_dir: str, condition: str) -> Optional[pd.DataFrame]:
    """Load group ROI contributions CSV produced by analyze_pc_contributions_group_roi.py.

    Expected path: {base_dir}/group_{condition}_roi_contrib/roi_contrib_group_summary.csv
    Returns a DataFrame with columns including: roi_name, mean_importance, sem_importance, subject_frequency.
    """
    path = os.path.join(base_dir, f"group_{condition}_roi_contrib", "roi_contrib_group_summary.csv")
    if not os.path.exists(path):
        print(f"✗ PC contrib CSV not found: {path}")
        return None
    try:
        df = pd.read_csv(path)
        # normalize column names
        df = df.rename(columns={
            'mean_importance': 'contrib_mean_importance',
            'sem_importance': 'contrib_sem_importance',
            'subject_frequency': 'contrib_subject_frequency'
        })
        return df
    except Exception as e:
        print(f"✗ Error reading PC contrib CSV at {path}: {e}")
        return None


def load_stability_group_csv(base_dir: Optional[str], condition: str, direct_csv: Optional[str] = None) -> Optional[pd.DataFrame]:
    """Load group ROI stability CSV produced by analyze_channel_stability_group_roi*.py.

    Two modes:
      - direct_csv: provide the exact CSV path. If provided, used directly.
      - base_dir: expected path {base_dir}/group_{condition}/roi_stability_summary.csv

    Returns a DataFrame with columns including: roi_name, median_dprime, mean_dprime, subject_frequency.
    """
    path = None
    if direct_csv:
        path = direct_csv
    elif base_dir:
        # Common convention; adjust to your run layout if needed
        path = os.path.join(base_dir, f"group_{condition}", "roi_stability_summary.csv")
    else:
        print("✗ No stability CSV path provided.")
        return None

    if not os.path.exists(path):
        print(f"✗ Stability CSV not found: {path}")
        return None
    try:
        df = pd.read_csv(path)
        # normalize column names
        df = df.rename(columns={
            'subject_frequency': 'stability_subject_frequency'
        })
        # compute stability_score used in top_stable_rois figure if possible
        if 'median_dprime' in df.columns and 'stability_subject_frequency' in df.columns:
            df['stability_score'] = 0.6 * df['median_dprime'].astype(float) + 0.4 * df['stability_subject_frequency'].astype(float)
        return df
    except Exception as e:
        print(f"✗ Error reading stability CSV at {path}: {e}")
        return None


def _clean_roi_name(name: str) -> str:
    """Normalize ROI names to improve joining across sources.

    - Right-*/Left-* to R-*/L-*
    - Remove trailing " (n)" sample counts
    - Strip whitespace
    """
    if not isinstance(name, str):
        return str(name)
    s = name.strip()
    if s.startswith('Right-'):
        s = s.replace('Right-', 'R-')
    elif s.startswith('Left-'):
        s = s.replace('Left-', 'L-')
    s = re.sub(r"\s*\(\d+\)$", "", s)
    return s


def merge_roi_tables(df_contrib: pd.DataFrame, df_stability: pd.DataFrame) -> pd.DataFrame:
    """Inner-join on normalized ROI name and keep relevant columns from each source."""
    keep_cols_contrib = ['roi_name', 'contrib_mean_importance', 'contrib_sem_importance', 'contrib_subject_frequency']
    keep_cols_stab = ['roi_name', 'median_dprime', 'mean_dprime', 'stability_subject_frequency']

    # Some stability tables may not have 'mean_dprime'; keep what's available
    for col in list(keep_cols_stab):
        if col != 'roi_name' and col not in df_stability.columns:
            keep_cols_stab.remove(col)

    df1 = df_contrib[keep_cols_contrib].copy()
    df2 = df_stability[keep_cols_stab].copy()
    # add normalized join key
    df1['roi_key'] = df1['roi_name'].map(_clean_roi_name)
    df2['roi_key'] = df2['roi_name'].map(_clean_roi_name)
    merged = pd.merge(df1.drop(columns=['roi_name']), df2.drop(columns=['roi_name']), on='roi_key', how='inner', suffixes=('_contrib', '_stab'))
    # prefer contrib roi_name for labeling
    merged = merged.rename(columns={'roi_key': 'roi_name'})
    return merged


def pick_top_rois(df_ref: pd.DataFrame, by: str, top_k: int) -> List[str]:
    """Pick top_k ROI names from df_ref by a given metric column."""
    if by not in df_ref.columns:
        print(f"⚠️ Column '{by}' not found for top selection; falling back to 'contrib_mean_importance'.")
        by = 'contrib_mean_importance'
    return df_ref.sort_values(by, ascending=False).head(top_k)['roi_name'].tolist()


def plot_scatter_compare(df: pd.DataFrame, condition: str, out_dir: str, metric_x: str, metric_y: str, title_suffix: str, filename_suffix: Optional[str] = None):
    os.makedirs(out_dir, exist_ok=True)
    plt.figure(figsize=(8, 6))
    x = df[metric_x].values
    y = df[metric_y].values
    sizes = 50 + 200*np.minimum(1.0, 0.5*(df.get('contrib_subject_frequency', 0.0).values + df.get('stability_subject_frequency', 0.0).values))
    colors = df.get('contrib_subject_frequency', pd.Series(np.ones(len(df))*0.5)).values

    sc = plt.scatter(x, y, c=colors, s=sizes, cmap='viridis', edgecolor='k', alpha=0.85)
    for _, r in df.iterrows():
        plt.text(r[metric_x], r[metric_y], r['roi_name'], fontsize=8, ha='left', va='bottom', alpha=0.8)

    # correlation stats
    try:
        from scipy.stats import pearsonr, spearmanr
        pear = pearsonr(x, y)
        spear_r, spear_p = spearmanr(x, y)
        spear = SimpleNamespace(correlation=spear_r, pvalue=spear_p)
    except Exception:
        # Fallback: numpy/pandas correlations (no p-values)
        pear_r = float(np.corrcoef(x, y)[0, 1]) if len(x) > 1 else np.nan
        spear_r = float(pd.Series(x).corr(pd.Series(y), method='spearman')) if len(x) > 1 else np.nan
        pear = SimpleNamespace(statistic=pear_r, pvalue=np.nan)
        spear = SimpleNamespace(correlation=spear_r, pvalue=np.nan)

    plt.xlabel(metric_x.replace('_', ' '))
    plt.ylabel(metric_y.replace('_', ' '))
    plt.title(f"{condition}: {title_suffix}\nPearson r={pear.statistic:.2f} (p={pear.pvalue:.3g}), Spearman rs={spear.correlation:.2f} (p={spear.pvalue:.3g})")
    cbar = plt.colorbar(sc)
    cbar.set_label('Contrib subject frequency')
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    suffix = f"_{filename_suffix}" if filename_suffix else ""
    fn = os.path.join(out_dir, f"roi_compare_scatter_{condition}{suffix}.png")
    plt.savefig(fn, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📈 Saved scatter: {fn}")


def main():
    # Config
    conditions = ['overt']  # user requested overt only

    # PC-contrib analysis base (produced by analyze_pc_contributions_group_roi.py)
    pc_contrib_base = r"U:\\eng_research_hrc_binauralhearinglab\\Sudan\\Labs\\Sen Lab\\Research_projects\\Whole_Head_Cocktail_party\\Classifier_script_results\\pc_contrib_roi_analysis"

    # Stability analysis outputs (produced by analyze_channel_stability_group_roi*.py)
    # Option A: one base dir with group_{condition}/roi_stability_summary.csv inside
    stability_base = None  # not used when direct CSV paths are provided
    # Option B: direct CSV paths per condition (preferred if layout differs)
    stability_csv_paths: Dict[str, Optional[str]] = {
        # Provided by user
        'overt': r"U:\\eng_research_hrc_binauralhearinglab\\Sudan\\Labs\\Sen Lab\\Research_projects\\Whole_Head_Cocktail_party\\Classifier_script_results\\channel_stability_analysis_brod_w_control\\group_overt_roi\\roi_stability_summary.csv",
        'covert': None,
    }

    # Output compare directory
    out_base = r"U:\\eng_research_hrc_binauralhearinglab\\Sudan\\Labs\\Sen Lab\\Research_projects\\Whole_Head_Cocktail_party\\Classifier_script_results\\roi_compare_stability"

    # Top ROI selection
    # We'll produce two views: PC-top30 and Stable-top20
    views = [
        {
            'name': 'pc_top30',
            'top_by': 'contrib_mean_importance',
            'top_k': 30,
            'metric_x': 'median_dprime',
            'metric_y': 'contrib_mean_importance',
            'title': 'Top 30 ROIs by contrib_mean_importance'
        },
        {
            'name': 'stable_top20',
            'top_by': 'stability_score',
            'top_k': 20,
            'metric_x': 'stability_score',
            'metric_y': 'contrib_mean_importance',
            'title': 'Top 20 ROIs by stability_score'
        },
    ]

    for cond in conditions:
        print(f"\n🔍 Comparing ROI group scores for condition: {cond}")
        df_contrib = load_pc_contrib_group_csv(pc_contrib_base, cond)
        if df_contrib is None or df_contrib.empty:
            print(f"❌ Skipping {cond}: no contrib table")
            continue

        stab_csv = stability_csv_paths.get(cond)
        df_stab = load_stability_group_csv(stability_base, cond, direct_csv=stab_csv)
        if df_stab is None or df_stab.empty:
            print(f"❌ Skipping {cond}: no stability table")
            continue

        merged = merge_roi_tables(df_contrib, df_stab)
        if merged.empty:
            print(f"❌ No overlapping ROIs found for {cond}")
            continue

        cond_out = os.path.join(out_base, f"compare_{cond}")
        os.makedirs(cond_out, exist_ok=True)
        merged.to_csv(os.path.join(cond_out, 'roi_compare_merged_all.csv'), index=False)

        # Generate both requested views
        for view in views:
            top_by = view['top_by']
            top_k = view['top_k']
            # reference table for top selection
            ref_df = df_contrib if top_by == 'contrib_mean_importance' else df_stab
            if top_by not in ref_df.columns:
                print(f"⚠️ '{top_by}' not found in reference table; skipping view {view['name']}")
                continue
            top_roi_names = pick_top_rois(ref_df, by=top_by, top_k=top_k)
            merged_top = merged[merged['roi_name'].isin(top_roi_names)].copy()
            if merged_top.empty:
                print(f"⚠️ No overlap after top filtering for view {view['name']}")
                continue
            merged_top.to_csv(os.path.join(cond_out, f"roi_compare_merged_{view['name']}.csv"), index=False)
            title = f"{view['title']} (n={len(merged_top)})"
            mx = view['metric_x'] if view['metric_x'] in merged_top.columns else ('median_dprime' if 'median_dprime' in merged_top.columns else 'mean_dprime')
            my = view['metric_y']
            plot_scatter_compare(
                merged_top,
                condition=cond,
                out_dir=cond_out,
                metric_x=mx,
                metric_y=my,
                title_suffix=title,
                filename_suffix=view['name'],
            )

    print("\n✅ Done.")


if __name__ == '__main__':
    main()
