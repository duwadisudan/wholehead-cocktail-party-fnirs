#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Within-subject variability vs overt accuracy (supplementary).

Compares two per-channel trial-level reliability metrics against decoding
accuracy across subjects:

    CNR = |μ| / σ            (response reliability; higher = more reliable)
    CV  = σ  / |μ|           (within-subject variability; higher = more variable)

where μ and σ are the mean and SD of the HbO response averaged over the
3-8 s post-stimulus window across trials. Two CV summaries are computed
per subject: a per-channel mean and a Jensen-safe subject-level form
(differs because of Jensen's inequality). Pearson correlations against
overt accuracy are reported for All-Channels and Angular Gyrus separately,
producing a 2x2 publication figure.

Author: Sudan Duwadi <sudan@bu.edu>
Notes: Code refactoring was AI-assisted; all scientific decisions and
       accountability remain with the author.
"""

#%%

import sys
from whichscript import enable_auto_logging

enable_auto_logging()
#%%
import os
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

try:
    from adjustText import adjust_text
    _HAS_ADJUST_TEXT = True
except ImportError:
    _HAS_ADJUST_TEXT = False
    print("Note: adjustText not installed; subject-ID labels will not be auto-spaced. "
          "Install with: pip install adjustText")

warnings.filterwarnings("ignore")

# ------------------------------------------------------------------ CONFIG
SUBJECTS = ['01','02','03','04','05','10','11','12','13','14','15','18','20','22','25','28','30','31','32','33','34','35','39','41','44','47']
RUN_TYPE = 'overt'

PROJECT_ROOT = r"U:\eng_research_hrc_binauralhearinglab\Sudan\Labs\Sen Lab\Research_projects\Whole_Head_Cocktail_party"

SNR_DIR = os.path.join(PROJECT_ROOT,
                       "Cocktail_party_whole_head_master_data",
                       "derivatives", "processed_data",
                       f"trial_snr_{RUN_TYPE}")

CLASSIFIER_DIR = os.path.join(PROJECT_ROOT,
                              "Classifier_script_results", "nested",
                              "rf_snr_0_20feat_balanced_depth5_oob")

ACC_CSV = os.path.join(CLASSIFIER_DIR, "final_table.csv")
ACC_COL = "Overt_perc"

ROI_MAP_CSV = os.path.join(PROJECT_ROOT, "ROIs", "roi_master.csv")

OUT_DIR = CLASSIFIER_DIR
os.makedirs(OUT_DIR, exist_ok=True)

print(f"SNR directory: {SNR_DIR}")
print(f"Accuracy CSV: {ACC_CSV}")
print(f"Output directory: {OUT_DIR}")

#%%
# ------------------------------------------------------------------ LOAD ROI MAPPING
roi_df = pd.read_csv(ROI_MAP_CSV)
print(f"\nLoaded ROI mapping from {ROI_MAP_CSV}, columns: {list(roi_df.columns)}")

def get_channel_labels_for_roi(roi_name: str, mapping_df: pd.DataFrame) -> list:
    mask = mapping_df['brodmann'] == roi_name
    labels = mapping_df.loc[mask, 'channel_label'].to_list()
    if not labels:
        raise RuntimeError(f'ROI string "{roi_name}" not found in brodmann column.')
    return labels

left_labels = get_channel_labels_for_roi("Left-AngGyrus (39)", roi_df)
right_labels = get_channel_labels_for_roi("Right-AngGyrus (39)", roi_df)
ag_labels = sorted(set(left_labels + right_labels))
ag_labels_set = set(ag_labels)
print(f"Angular Gyrus channels: {len(ag_labels)} total (Left: {len(left_labels)}, Right: {len(right_labels)})")

#%%
# ------------------------------------------------------------------ LOAD ACCURACY DATA
acc_df = pd.read_csv(ACC_CSV, sep=None, engine='python')
if 'Subject' not in acc_df.columns:
    raise RuntimeError("Expected a 'Subject' column in accuracy CSV.")
if ACC_COL not in acc_df.columns:
    raise RuntimeError(f"Accuracy column '{ACC_COL}' not found; available: {list(acc_df.columns)}")

acc_df['Subject'] = acc_df['Subject'].astype(int)
acc_series_full = acc_df.set_index('Subject')[ACC_COL]
print(f"\nLoaded accuracy data for {len(acc_series_full)} subjects")
print(f"Accuracy range: {acc_series_full.min():.1f}% to {acc_series_full.max():.1f}%")

#%%
# ------------------------------------------------------------------ LOAD PER-CHANNEL DATA AND COMPUTE METRICS
# For each subject, build:
#   AG and All-Channels:
#     CNR_mean   = mean( |μ_ch| / σ_ch )                [existing snr_avg, kept for comparison]
#     CV_mean    = mean( σ_ch / |μ_ch| )                [per-channel CV, then averaged]
#     CV_subject = mean(σ_ch) / mean(|μ_ch|)            [Jensen-safe subject-level CV]

ag_cnr_mean   = {}
ag_cv_mean    = {}   # mean over channels of (σ/|μ|)
ag_cv_subject = {}   # mean(σ) / mean(|μ|)
ag_abs_mean   = {}
ag_std_mean   = {}
n_ag_channels = {}

all_cnr_mean   = {}
all_cv_mean    = {}
all_cv_subject = {}
all_abs_mean   = {}
all_std_mean   = {}
n_all_channels = {}

EPS = 1e-12  # floor to avoid divide-by-zero in per-channel CV


def _per_channel_cv(abs_mu, sigma):
    """σ/|μ| with a tiny floor on |μ| to avoid divide-by-zero. Inf/NaN are dropped by caller."""
    denom = np.where(np.abs(abs_mu) < EPS, np.nan, abs_mu)
    return sigma / denom


for subj_id in SUBJECTS:
    subj_int = int(subj_id)
    subj_folder = f"sub_{subj_id}"
    snr_csv = os.path.join(SNR_DIR, subj_folder, f"snr_results_{RUN_TYPE}.csv")

    if not os.path.exists(snr_csv):
        print(f"Warning: SNR file not found for sub-{subj_id}: {snr_csv}")
        continue

    snr_df = pd.read_csv(snr_csv)
    required_cols = ['channel', 'snr_avg', 'abs_mean_avg', 'std_avg']
    if not all(col in snr_df.columns for col in required_cols):
        print(f"Warning: Missing required columns for sub-{subj_id}: {list(snr_df.columns)}")
        continue

    # AG subset
    ag_mask = snr_df['channel'].isin(ag_labels_set)
    ag_cnr  = snr_df.loc[ag_mask, 'snr_avg'].values
    ag_mu   = snr_df.loc[ag_mask, 'abs_mean_avg'].values
    ag_sig  = snr_df.loc[ag_mask, 'std_avg'].values

    all_cnr = snr_df['snr_avg'].values
    all_mu  = snr_df['abs_mean_avg'].values
    all_sig = snr_df['std_avg'].values

    # ------- Angular Gyrus
    if len(ag_cnr) > 0:
        ag_cnr_clean = ag_cnr[np.isfinite(ag_cnr)]
        ag_mu_clean  = ag_mu[np.isfinite(ag_mu)]
        ag_sig_clean = ag_sig[np.isfinite(ag_sig)]

        ag_cv_per_ch = _per_channel_cv(ag_mu, ag_sig)
        ag_cv_per_ch = ag_cv_per_ch[np.isfinite(ag_cv_per_ch)]

        if len(ag_cnr_clean) > 0:
            ag_cnr_mean[subj_int] = float(np.mean(ag_cnr_clean))
            n_ag_channels[subj_int] = len(ag_cnr_clean)
        if len(ag_cv_per_ch) > 0:
            ag_cv_mean[subj_int] = float(np.mean(ag_cv_per_ch))
        if len(ag_mu_clean) > 0 and len(ag_sig_clean) > 0:
            mean_mu  = float(np.mean(ag_mu_clean))
            mean_sig = float(np.mean(ag_sig_clean))
            ag_abs_mean[subj_int] = mean_mu
            ag_std_mean[subj_int] = mean_sig
            if mean_mu > EPS:
                ag_cv_subject[subj_int] = mean_sig / mean_mu

        if subj_int in ag_cnr_mean and subj_int in ag_cv_mean:
            print(f"sub-{subj_id}: AG  CNR={ag_cnr_mean[subj_int]:.3f}  "
                  f"CV_mean={ag_cv_mean[subj_int]:.3f}  "
                  f"CV_subj={ag_cv_subject.get(subj_int, np.nan):.3f}  "
                  f"({n_ag_channels[subj_int]} ch)")

    # ------- All channels
    if len(all_cnr) > 0:
        all_cnr_clean = all_cnr[np.isfinite(all_cnr)]
        all_mu_clean  = all_mu[np.isfinite(all_mu)]
        all_sig_clean = all_sig[np.isfinite(all_sig)]

        all_cv_per_ch = _per_channel_cv(all_mu, all_sig)
        all_cv_per_ch = all_cv_per_ch[np.isfinite(all_cv_per_ch)]

        if len(all_cnr_clean) > 0:
            all_cnr_mean[subj_int] = float(np.mean(all_cnr_clean))
            n_all_channels[subj_int] = len(all_cnr_clean)
        if len(all_cv_per_ch) > 0:
            all_cv_mean[subj_int] = float(np.mean(all_cv_per_ch))
        if len(all_mu_clean) > 0 and len(all_sig_clean) > 0:
            mean_mu  = float(np.mean(all_mu_clean))
            mean_sig = float(np.mean(all_sig_clean))
            all_abs_mean[subj_int] = mean_mu
            all_std_mean[subj_int] = mean_sig
            if mean_mu > EPS:
                all_cv_subject[subj_int] = mean_sig / mean_mu

print(f"\nLoaded data for {len(ag_cnr_mean)} subjects (AG)")

#%%
# ------------------------------------------------------------------ BUILD DATAFRAME
analysis_df_full = pd.concat([
    pd.Series(ag_cnr_mean,    name='ag_cnr_mean'),
    pd.Series(ag_cv_mean,     name='ag_cv_mean'),
    pd.Series(ag_cv_subject,  name='ag_cv_subject'),
    pd.Series(ag_abs_mean,    name='ag_abs_mean'),
    pd.Series(ag_std_mean,    name='ag_std_mean'),
    pd.Series(all_cnr_mean,   name='all_cnr_mean'),
    pd.Series(all_cv_mean,    name='all_cv_mean'),
    pd.Series(all_cv_subject, name='all_cv_subject'),
    pd.Series(all_abs_mean,   name='all_abs_mean'),
    pd.Series(all_std_mean,   name='all_std_mean'),
    pd.Series(n_ag_channels,  name='n_ag_channels'),
    pd.Series(n_all_channels, name='n_all_channels'),
    acc_series_full.rename('accuracy'),
], axis=1)


def _corr(df, x_col):
    """Pearson correlation against accuracy. Returns (r, p, sub)."""
    sub = df.dropna(subset=[x_col, 'accuracy'])
    if len(sub) < 2:
        return np.nan, np.nan, sub
    r, p = pearsonr(sub[x_col], sub['accuracy'])
    return r, p, sub


def _spearman(df, x_col):
    """Spearman rank correlation against accuracy. Returns (rho, p, sub)."""
    sub = df.dropna(subset=[x_col, 'accuracy'])
    if len(sub) < 2:
        return np.nan, np.nan, sub
    rho, p = spearmanr(sub[x_col], sub['accuracy'])
    return rho, p, sub


metrics = [
    ('ag_cnr_mean',    'AG CNR (|mu|/sigma)'),
    ('ag_cv_mean',     'AG CV mean-of-channels (sigma/|mu|)'),
    ('ag_cv_subject',  'AG CV subject-level (mean sigma / mean |mu|)'),
    ('all_cnr_mean',   'All-Ch CNR (|mu|/sigma)'),
    ('all_cv_mean',    'All-Ch CV mean-of-channels (sigma/|mu|)'),
    ('all_cv_subject', 'All-Ch CV subject-level (mean sigma / mean |mu|)'),
]

print("\n=== Pearson and Spearman correlations vs Overt Accuracy ===")
results = {}
for col, label in metrics:
    r,    p_p, sub = _corr(analysis_df_full, col)
    rho,  p_s, _   = _spearman(analysis_df_full, col)
    results[col] = (r, p_p, rho, p_s, len(sub))
    print(f"{label:50s}  Pearson r = {r:+.3f} (p = {p_p:.4f})   "
          f"Spearman ρ = {rho:+.3f} (p = {p_s:.4f})   N = {len(sub)}")

#%%
# ------------------------------------------------------------------ SAVE COMPARISON TABLE
comparison_df = pd.DataFrame({
    'Metric':       [label for _, label in metrics],
    'Pearson r':    [results[col][0] for col, _ in metrics],
    'Pearson p':    [results[col][1] for col, _ in metrics],
    'Spearman rho': [results[col][2] for col, _ in metrics],
    'Spearman p':   [results[col][3] for col, _ in metrics],
    'N subjects':   [results[col][4] for col, _ in metrics],
})
print("\n" + comparison_df.to_string(index=False))

comparison_path = os.path.join(OUT_DIR, f'cnr_vs_cv_correlation_comparison_{RUN_TYPE}.csv')
comparison_df.to_csv(comparison_path, index=False)
print(f"\nSaved correlation comparison: {comparison_path}")

csv_path = os.path.join(OUT_DIR, f'within_subject_variability_metrics_{RUN_TYPE}.csv')
analysis_df_full.to_csv(csv_path, index_label='Subject')
print(f"Saved subject-level metrics CSV: {csv_path}")

#%%
# ------------------------------------------------------------------ PUBLICATION FIGURE (2x2 side-by-side)
# Rows: All Channels (top), Angular Gyrus (bottom)
# Cols: CNR (left, |μ|/σ), CV (right, σ/|μ|)
# CV column uses the per-channel-then-averaged CV (ag_cv_mean / all_cv_mean) by default —
# switch to *_cv_subject below if you prefer the Jensen-safe variant.

USE_SUBJECT_LEVEL_CV = False  # set True to plot mean(σ)/mean(|μ|) instead of mean(σ/|μ|)

cv_ag_col  = 'ag_cv_subject'  if USE_SUBJECT_LEVEL_CV else 'ag_cv_mean'
cv_all_col = 'all_cv_subject' if USE_SUBJECT_LEVEL_CV else 'all_cv_mean'
cv_label   = 'CV (subject-level)' if USE_SUBJECT_LEVEL_CV else 'CV'

_PUB_BASE_FONT  = 16
_PUB_AXIS_LABEL = 18
_PUB_AXIS_TICK  = 15
_PUB_SUBJ_FONT  = 9
_PUB_PANEL_FONT = 22
_PUB_ANNOT_FONT = 14

plt.rcParams.update({
    'font.size': _PUB_BASE_FONT,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial'],
    'axes.linewidth': 1.5,
    'xtick.major.width': 1.5,
    'ytick.major.width': 1.5,
    'xtick.major.size': 6,
    'ytick.major.size': 6,
})


def _fmt_p(p):
    return f'p = {p:.4f}' if p >= 0.0001 else 'p < 0.0001'


def _draw_panel(ax, x, y, subj_ids, panel_letter, xlabel, r_val, p_val):
    ax.scatter(x, y, s=120, color='black', edgecolors='black',
               linewidths=1.5, alpha=0.85, zorder=3)

    # Subject ID labels — placed at the points, then de-collided by adjustText
    label_texts = []
    for xv, yv, sid in zip(x, y, subj_ids):
        t = ax.text(xv, yv, f"{int(sid):02d}",
                    fontsize=_PUB_SUBJ_FONT, color='black',
                    ha='left', va='bottom', zorder=4)
        label_texts.append(t)

    if len(x) > 2:
        m, b = np.polyfit(x, y, 1)
        x_line = np.linspace(np.min(x), np.max(x), 200)
        ax.plot(x_line, m * x_line + b, color='black',
                linewidth=2, linestyle='--', alpha=0.8, zorder=2)

    ax.text(0.95, 0.07, f'r = {r_val:.3f},  {_fmt_p(p_val)}',
            transform=ax.transAxes,
            fontsize=_PUB_ANNOT_FONT, fontweight='bold',
            va='bottom', ha='right',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor='gray', alpha=0.85))
    ax.set_xlabel(xlabel, fontsize=_PUB_AXIS_LABEL, fontweight='bold')
    ax.set_ylabel('Overt Accuracy (%)', fontsize=_PUB_AXIS_LABEL, fontweight='bold')
    ax.tick_params(axis='both', labelsize=_PUB_AXIS_TICK)
    ax.set_ylim(45, 105)
    ax.grid(True, alpha=0.25, linestyle=':', linewidth=0.8)
    ax.set_facecolor('#FAFAFA')
    ax.text(-0.12, 1.05, panel_letter, transform=ax.transAxes,
            fontsize=_PUB_PANEL_FONT, fontweight='bold', va='top')

    if _HAS_ADJUST_TEXT and label_texts:
        adjust_text(
            label_texts,
            ax=ax,
            expand_points=(1.4, 1.6),
            expand_text=(1.2, 1.4),
            force_points=(0.3, 0.4),
            force_text=(0.4, 0.6),
        )


fig, axes = plt.subplots(2, 2, figsize=(14, 11))

# Panel A: All-Ch CNR
_df = analysis_df_full.dropna(subset=['all_cnr_mean', 'accuracy'])
r, p, _ = _corr(analysis_df_full, 'all_cnr_mean')
_draw_panel(axes[0, 0], _df['all_cnr_mean'].values, _df['accuracy'].values,
            _df.index.values, 'A',
            'All-Channels CNR (|μ|/σ)', r, p)

# Panel B: All-Ch CV
_df = analysis_df_full.dropna(subset=[cv_all_col, 'accuracy'])
r, p, _ = _corr(analysis_df_full, cv_all_col)
_draw_panel(axes[0, 1], _df[cv_all_col].values, _df['accuracy'].values,
            _df.index.values, 'B',
            f'All-Channels {cv_label}', r, p)

# Panel C: AG CNR
_df = analysis_df_full.dropna(subset=['ag_cnr_mean', 'accuracy'])
r, p, _ = _corr(analysis_df_full, 'ag_cnr_mean')
_draw_panel(axes[1, 0], _df['ag_cnr_mean'].values, _df['accuracy'].values,
            _df.index.values, 'C',
            'Angular Gyrus CNR (|μ|/σ)', r, p)

# Panel D: AG CV
_df = analysis_df_full.dropna(subset=[cv_ag_col, 'accuracy'])
r, p, _ = _corr(analysis_df_full, cv_ag_col)
_draw_panel(axes[1, 1], _df[cv_ag_col].values, _df['accuracy'].values,
            _df.index.values, 'D',
            f'Angular Gyrus {cv_label}', r, p)

fig.suptitle(f'CNR vs Within-Subject Variability — Overt Accuracy ({RUN_TYPE.capitalize()})',
             fontsize=15, fontweight='bold', y=1.00)
fig.tight_layout()

cv_tag = 'subjlevel' if USE_SUBJECT_LEVEL_CV else 'meanofch'
for _ext in ('png', 'svg', 'pdf'):
    _out = os.path.join(OUT_DIR, f'cnr_vs_cv_pub_{cv_tag}_{RUN_TYPE}.{_ext}')
    _fmt = 'png' if _ext == 'png' else _ext
    _dpi = 300 if _ext == 'png' else None
    fig.savefig(_out, format=_fmt, dpi=_dpi, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"Saved: {_out}")

plt.show()

#%%
# ------------------------------------------------------------------ RIGHT-HAND-SIDE ONLY FIGURE (CV panels)
# Within-subject variability (σ/|μ|) for All-Channels (top) and Angular Gyrus (bottom),
# vertical 2x1 publication layout matching snr_vs_accuracy_analysis.py.

fig_cv, (ax_all_cv, ax_ag_cv) = plt.subplots(2, 1, figsize=(7, 11))

# Panel A: All-Ch CV
_df = analysis_df_full.dropna(subset=[cv_all_col, 'accuracy'])
r, p, _ = _corr(analysis_df_full, cv_all_col)
_draw_panel(ax_all_cv, _df[cv_all_col].values, _df['accuracy'].values,
            _df.index.values, 'A',
            f'All-Channels {cv_label}', r, p)

# Panel B: AG CV
_df = analysis_df_full.dropna(subset=[cv_ag_col, 'accuracy'])
r, p, _ = _corr(analysis_df_full, cv_ag_col)
_draw_panel(ax_ag_cv, _df[cv_ag_col].values, _df['accuracy'].values,
            _df.index.values, 'B',
            f'Angular Gyrus {cv_label}', r, p)

fig_cv.tight_layout()

for _ext in ('png', 'svg', 'pdf'):
    _out = os.path.join(OUT_DIR, f'within_subject_variability_only_{cv_tag}_{RUN_TYPE}.{_ext}')
    _fmt = 'png' if _ext == 'png' else _ext
    _dpi = 300 if _ext == 'png' else None
    fig_cv.savefig(_out, format=_fmt, dpi=_dpi, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
    print(f"Saved: {_out}")

plt.show()

# %% End of file
