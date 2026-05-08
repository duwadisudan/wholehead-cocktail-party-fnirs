"""
Pairwise statistical comparison of overt vs covert decoding accuracy
(all-channels configuration) — supporting analysis for Figure 6.

Loads per-subject RF accuracy summaries for the overt and covert conditions,
performs paired statistical comparison across subjects, and produces the
pairwise scatter / line figure.

Author: Sudan Duwadi <sudan@bu.edu>
Notes: Code refactoring, documentation, and commenting were AI-assisted;
       all scientific decisions and accountability remain with the author.
"""
#%%
from whichscript import configure, enable_auto_logging
from wholehead_cocktail_party.paths import load_paths, require, whichscript_archive_dir

_PATHS = load_paths()
require(_PATHS, "classifier_results_root")

configure(
    archive=True,
    archive_only=False,
    archive_dir=str(whichscript_archive_dir(_PATHS)),
    hide_sidecars=True,
    metadata=False,
    snapshot_script=False,
    snapshot_py=True,
    local_imports_snapshot=False,
)

enable_auto_logging()

#%%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# File path (All Channels — has both Overt_perc and Covert_perc)
file_path = str(_PATHS.classifier_results_root / "nested" / "rf_snr_0_20feat_balanced_depth5_oob" / "final_table.csv")

def load_and_clean_data(fp):
    """Load and clean the CSV data."""
    df = pd.read_csv(fp)
    df = df.dropna(subset=['Subject'])
    df = df[df['Subject'] != '']
    df = df[df['Subject'] != 'AVG']
    df['Subject'] = df['Subject'].astype(str)
    for col in ['Overt_perc', 'Covert_perc']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    print(f"Loaded: {len(df)} subjects, columns: {list(df.columns)}")
    return df

def calculate_descriptive_stats(data, label):
    """Calculate descriptive statistics for an array."""
    return {
        'label': label,
        'mean': data.mean(),
        'median': np.median(data),
        'std': data.std(ddof=1),
        'min': data.min(),
        'max': data.max(),
        'n': len(data),
    }

def perform_pairwise_tests(overt, covert):
    """Paired t-test, Wilcoxon signed-rank, and Cohen's d."""
    t_stat, p_value = stats.ttest_rel(overt, covert)
    w_stat, w_p_value = stats.wilcoxon(overt, covert, alternative='two-sided')
    differences = overt - covert
    cohens_d = np.mean(differences) / np.std(differences, ddof=1)
    return {
        'n_pairs': len(overt),
        'condition1_mean': np.mean(overt),
        'condition2_mean': np.mean(covert),
        'mean_difference': np.mean(differences),
        'paired_ttest': {'t_stat': t_stat, 'p_value': p_value},
        'wilcoxon': {'w_stat': w_stat, 'p_value': w_p_value},
        'cohens_d': cohens_d,
    }

def create_visualization(overt, covert, pairwise_result):
    """Single-panel supplementary figure for Nature Communications.

    Paired slopegraph (each line = one participant) with paired t-test.
    Style matched to table_maker_scatter_overt_only_pub_latency_CI.py.
    """
    import matplotlib as mpl

    # Font size constants (matched to reference figure)
    BASE_FONT_SIZE = 16
    AXIS_LABEL_FONT = 18
    AXIS_TICK_FONT = 15
    LEGEND_FONT = 13
    MEAN_TEXT_FONT = 13

    mpl.rcParams.update({
        'font.size': BASE_FONT_SIZE,
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial'],
        'axes.linewidth': 1.5,
        'xtick.major.width': 1.5,
        'ytick.major.width': 1.5,
        'xtick.major.size': 6,
        'ytick.major.size': 6,
        'pdf.fonttype': 42,     # editable text in PDF
        'ps.fonttype': 42,
        'svg.fonttype': 'none', # editable text in SVG
    })

    n = len(overt)

    # Palette
    c_overt = '#4878CF'   # muted blue
    c_covert = '#D65F5F'  # muted red
    c_mean = '#222222'

    # Figure
    fig, ax = plt.subplots(figsize=(7, 6))

    # Paired slopegraph
    jitter = 0.03
    for i in range(n):
        ax.plot(
            [0, 1], [overt[i], covert[i]],
            color='#999999', linewidth=1.0, alpha=0.6, zorder=1,
        )
        ax.scatter(0 + np.random.uniform(-jitter, jitter), overt[i],
                   s=80, color=c_overt, edgecolors='black', linewidths=0.8,
                   zorder=2)
        ax.scatter(1 + np.random.uniform(-jitter, jitter), covert[i],
                   s=80, color=c_covert, edgecolors='black', linewidths=0.8,
                   zorder=2)

    # Mean line
    mean1, mean2 = np.mean(overt), np.mean(covert)
    ax.plot([0, 1], [mean1, mean2], color=c_mean, linewidth=3, zorder=3)
    ax.scatter([0, 1], [mean1, mean2], s=140, color=c_mean, edgecolors='white',
               linewidths=1.5, zorder=4, label='Mean')

    # Mean value annotations
    ax.text(-0.08, mean1, f'{mean1:.1f}%', ha='right', va='center',
            fontsize=MEAN_TEXT_FONT, fontweight='bold', color=c_overt)
    ax.text(1.08, mean2, f'{mean2:.1f}%', ha='left', va='center',
            fontsize=MEAN_TEXT_FONT, fontweight='bold', color=c_covert)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Overt', 'Covert'], fontsize=AXIS_TICK_FONT)
    ax.set_xlim(-0.25, 1.25)
    ax.set_ylabel('Classification Accuracy (%)',
                   fontsize=AXIS_LABEL_FONT, fontweight='bold')
    ax.set_ylim(
        max(0, min(overt.min(), covert.min()) - 5),
        min(105, max(overt.max(), covert.max()) + 5),
    )
    ax.tick_params(axis='y', labelsize=AXIS_TICK_FONT)
    ax.legend(loc='lower left', frameon=True, edgecolor='black',
              fontsize=LEGEND_FONT, framealpha=0.95)
    ax.grid(True, alpha=0.25, linestyle=':', linewidth=0.8)
    ax.set_facecolor('#FAFAFA')

    # Significance bracket with paired t-test p-value
    p_t = pairwise_result['paired_ttest']['p_value']
    if p_t < 0.001:
        sig_label = '*** \u2014 p < 0.001'
    elif p_t >= 0.05:
        sig_label = f'n.s. \u2014 p = {p_t:.3f}'
    elif p_t < 0.01:
        sig_label = f'** \u2014 p = {p_t:.3f}'
    else:
        sig_label = f'* \u2014 p = {p_t:.3f}'

    # Push bracket down so label doesn't overlap the top spine
    y_top = ax.get_ylim()[1]
    y_bracket = y_top - 4
    ax.plot([0, 0, 1, 1], [y_bracket - 1, y_bracket, y_bracket, y_bracket - 1],
            lw=1.5, color='k')
    ax.text(0.5, y_bracket + 0.3, sig_label, ha='center', va='bottom',
            fontsize=AXIS_TICK_FONT, fontweight='bold')

    return fig

def main():
    """Main analysis function"""
    print("Loading and analyzing RF classifier accuracy data...")
    print("Comparison: Overt vs Covert (All Channels)")
    print("=" * 60)

    # Load data
    df = load_and_clean_data(file_path)

    # Keep only subjects with both Overt and Covert
    df_paired = df.dropna(subset=['Overt_perc', 'Covert_perc']).copy()
    df_paired = df_paired.sort_values('Subject').reset_index(drop=True)

    overt = df_paired['Overt_perc'].values
    covert = df_paired['Covert_perc'].values
    subjects = df_paired['Subject'].values
    n = len(overt)

    print(f"\nSubjects with both Overt and Covert data: {n}")

    # Descriptive statistics
    print("\n" + "=" * 60)
    print("DESCRIPTIVE STATISTICS")
    print("=" * 60)

    stats_overt = calculate_descriptive_stats(overt, 'Overt')
    stats_covert = calculate_descriptive_stats(covert, 'Covert')

    for s in [stats_overt, stats_covert]:
        print(f"\n{s['label']}:")
        print(f"  Mean: {s['mean']:.2f}%")
        print(f"  Median: {s['median']:.2f}%")
        print(f"  Std: {s['std']:.2f}%")
        print(f"  Range: {s['min']:.2f}% - {s['max']:.2f}%")
        print(f"  N: {s['n']}")

    # Pairwise comparison
    print("\n" + "=" * 60)
    print("PAIRWISE COMPARISON ANALYSIS")
    print("=" * 60)

    pairwise_result = perform_pairwise_tests(overt, covert)

    print(f"\nOvert vs Covert Pairwise Analysis:")
    print(f"  Number of paired observations: {pairwise_result['n_pairs']}")
    print(f"  Mean accuracy - Overt: {pairwise_result['condition1_mean']:.2f}%")
    print(f"  Mean accuracy - Covert: {pairwise_result['condition2_mean']:.2f}%")
    print(f"  Mean difference: {pairwise_result['mean_difference']:.2f}%")
    print(f"  Paired t-test: t = {pairwise_result['paired_ttest']['t_stat']:.3f}, p = {pairwise_result['paired_ttest']['p_value']:.3f}")
    print(f"  Wilcoxon signed-rank test: W = {pairwise_result['wilcoxon']['w_stat']:.3f}, p = {pairwise_result['wilcoxon']['p_value']:.3f}")
    print(f"  Effect size (Cohen's d): {pairwise_result['cohens_d']:.3f}")

    # Interpret effect size
    abs_d = abs(pairwise_result['cohens_d'])
    if abs_d < 0.2:
        effect_interpretation = "negligible"
    elif abs_d < 0.5:
        effect_interpretation = "small"
    elif abs_d < 0.8:
        effect_interpretation = "medium"
    else:
        effect_interpretation = "large"

    print(f"  Effect size interpretation: {effect_interpretation}")

    # Interpret p-value
    p_val = pairwise_result['paired_ttest']['p_value']
    if p_val < 0.001:
        significance = "highly significant (p < 0.001)"
    elif p_val < 0.01:
        significance = "very significant (p < 0.01)"
    elif p_val < 0.05:
        significance = "significant (p < 0.05)"
    else:
        significance = "not significant (p >= 0.05)"

    print(f"  Statistical significance: {significance}")

    # Create visualization
    print("\n" + "=" * 60)
    print("CREATING VISUALIZATION")
    print("=" * 60)

    fig = create_visualization(overt, covert, pairwise_result)

    # Create output directory if it doesn't exist
    output_dir = str(_PATHS.classifier_results_root / "nested" / "stats_overt_vs_covert_snr_0_20feat")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Save the plot in three formats (PDF, SVG, PNG)
    stem = "rf_overt_vs_covert_accuracy_pairwise_analysis"
    for ext in ['pdf', 'svg', 'png']:
        out_path = Path(output_dir) / f"{stem}.{ext}"
        fig.savefig(str(out_path), dpi=600, bbox_inches='tight', transparent=False)
        print(f"  Saved: {out_path}")
    print("Figures saved (PDF, SVG, PNG).")

    # Save detailed results to text file
    output_text_path = Path(output_dir) / "rf_overt_vs_covert_results.txt"
    with open(str(output_text_path), 'w') as f:
        f.write("Overt vs Covert Classification Accuracy — Pairwise Analysis\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"Data File: {file_path}\n")
        f.write(f"Subjects: {n}\n\n")

        f.write("Descriptive Statistics:\n")
        f.write("-" * 30 + "\n")
        for s in [stats_overt, stats_covert]:
            f.write(f"\n{s['label']}:\n")
            f.write(f"  Mean: {s['mean']:.2f}%\n")
            f.write(f"  Median: {s['median']:.2f}%\n")
            f.write(f"  Std: {s['std']:.2f}%\n")
            f.write(f"  Range: {s['min']:.2f}% - {s['max']:.2f}%\n")
            f.write(f"  N: {s['n']}\n")

        f.write("\nPairwise Comparison Results:\n")
        f.write("-" * 30 + "\n")
        f.write(f"  Paired observations: {pairwise_result['n_pairs']}\n")
        f.write(f"  Mean difference (Overt - Covert): {pairwise_result['mean_difference']:.2f}%\n")
        f.write(f"  Paired t-test: t({n-1}) = {pairwise_result['paired_ttest']['t_stat']:.3f}, p = {pairwise_result['paired_ttest']['p_value']:.4f}\n")
        f.write(f"  Wilcoxon test: W = {pairwise_result['wilcoxon']['w_stat']:.3f}, p = {pairwise_result['wilcoxon']['p_value']:.4f}\n")
        f.write(f"  Cohen's d: {pairwise_result['cohens_d']:.3f}\n")

    print(f"Detailed results saved as: {output_text_path}")

    # Save summary statistics to CSV
    csv_output_path = Path(output_dir) / "summary_statistics.csv"
    summary_data = []
    for s in [stats_overt, stats_covert]:
        summary_data.append({
            'Condition': s['label'],
            'Mean': s['mean'],
            'Median': s['median'],
            'Std': s['std'],
            'Min': s['min'],
            'Max': s['max'],
            'N': s['n']
        })
    pd.DataFrame(summary_data).to_csv(str(csv_output_path), index=False)
    print(f"Summary statistics saved as: {csv_output_path}")

    # Save pairwise comparison results to CSV
    pairwise_csv_path = Path(output_dir) / "pairwise_comparison_results.csv"
    pd.DataFrame([{
        'Comparison': 'Overt vs Covert',
        'N_Pairs': pairwise_result['n_pairs'],
        'Mean_Overt': pairwise_result['condition1_mean'],
        'Mean_Covert': pairwise_result['condition2_mean'],
        'Mean_Difference': pairwise_result['mean_difference'],
        'T_Statistic': pairwise_result['paired_ttest']['t_stat'],
        'T_Test_P_Value': pairwise_result['paired_ttest']['p_value'],
        'Wilcoxon_Statistic': pairwise_result['wilcoxon']['w_stat'],
        'Wilcoxon_P_Value': pairwise_result['wilcoxon']['p_value'],
        'Cohens_D': pairwise_result['cohens_d']
    }]).to_csv(str(pairwise_csv_path), index=False)
    print(f"Pairwise comparison results saved as: {pairwise_csv_path}")

    # Save individual subject data
    subjects_csv_path = Path(output_dir) / "subject_overt_vs_covert_data.csv"
    pd.DataFrame({
        'Subject': subjects,
        'Overt_perc': overt,
        'Covert_perc': covert,
        'Difference': overt - covert,
    }).to_csv(str(subjects_csv_path), index=False)
    print(f"Subject-level data saved as: {subjects_csv_path}")

    print(f"\nAll output files saved to: {output_dir}")

    plt.show()

if __name__ == "__main__":
    main()
