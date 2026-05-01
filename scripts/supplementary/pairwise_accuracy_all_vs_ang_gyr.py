"""
Pairwise statistical comparison of overt decoding accuracy:
all-channels vs left/right Angular Gyrus configuration (supplementary).

Loads per-subject classifier accuracy summaries for the two channel
configurations (all-channels feature set vs Angular-Gyrus-only), performs
paired statistical comparison across subjects, and reports the per-subject
deltas with the figure used in the supplementary materials.

Author: Sudan Duwadi <sudan@bu.edu>
Notes: Code refactoring, documentation, and commenting were AI-assisted;
       all scientific decisions and accountability remain with the author.
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
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# File paths
file1_path = r"U:\eng_research_hrc_binauralhearinglab\Sudan\Labs\Sen Lab\Research_projects\Whole_Head_Cocktail_party\Classifier_script_results\nested\rf_snr_0_20feat_balanced_depth5_oob\final_table.csv"
file2_path = r"U:\eng_research_hrc_binauralhearinglab\Sudan\Labs\Sen Lab\Research_projects\Whole_Head_Cocktail_party\Classifier_script_results\nested\rf_LR_ang_gyr_snr_0_20feat_balanced_depth5_oob\final_table.csv"

def load_and_clean_data(file_path, condition_name):
    """Load and clean the CSV data"""
    try:
        df = pd.read_csv(file_path)
        
        # Remove rows with empty subject IDs or all NaN values
        df = df.dropna(subset=['Subject'])
        df = df[df['Subject'] != '']
        
        # Remove the AVG row if it exists
        df = df[df['Subject'] != 'AVG']
        
        # Convert subject to string and numeric columns to float
        df['Subject'] = df['Subject'].astype(str)
        
        # Clean numeric columns
        for col in ['Overt_perc', 'Covert_perc']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Add condition identifier
        df['Condition'] = condition_name
        
        print(f"Loaded {condition_name}: {len(df)} subjects")
        print(f"Columns: {list(df.columns)}")
        
        return df
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def calculate_descriptive_stats(df, condition_name):
    """Calculate descriptive statistics"""
    stats_dict = {}
    
    # Only analyze Overt accuracy data
    column = 'Overt_perc'
    if column in df.columns:
        data = df[column].dropna()
        if len(data) > 0:
            stats_dict[column] = {
                'mean': data.mean(),
                'median': data.median(),
                'std': data.std(),
                'min': data.min(),
                'max': data.max(),
                'n': len(data)
            }
    
    return stats_dict

def perform_pairwise_tests(df1, df2):
    """Perform statistical tests between conditions"""
    results = {}
    
    # Find common subjects
    common_subjects = set(df1['Subject']) & set(df2['Subject'])
    print(f"Common subjects between conditions: {len(common_subjects)}")
    
    if len(common_subjects) < 3:
        print("Warning: Too few common subjects for meaningful pairwise analysis")
        return results
    
    # Only analyze Overt accuracy
    column = 'Overt_perc'
    if column in df1.columns and column in df2.columns:
        # Get data for common subjects
        data1 = []
        data2 = []
        subjects_with_data = []
        
        for subject in common_subjects:
            val1 = df1[df1['Subject'] == subject][column].iloc[0] if len(df1[df1['Subject'] == subject]) > 0 else np.nan
            val2 = df2[df2['Subject'] == subject][column].iloc[0] if len(df2[df2['Subject'] == subject]) > 0 else np.nan
            
            if pd.notna(val1) and pd.notna(val2):
                data1.append(val1)
                data2.append(val2)
                subjects_with_data.append(subject)
        
        if len(data1) >= 3:  # Need at least 3 pairs for meaningful statistics
            data1 = np.array(data1)
            data2 = np.array(data2)
            
            # Paired t-test
            t_stat, p_value = stats.ttest_rel(data1, data2)
            
            # Wilcoxon signed-rank test (non-parametric alternative)
            w_stat, w_p_value = stats.wilcoxon(data1, data2, alternative='two-sided')
            
            # Effect size (Cohen's d for paired samples)
            differences = data1 - data2
            cohens_d = np.mean(differences) / np.std(differences, ddof=1)
            
            results[column] = {
                'n_pairs': len(data1),
                'subjects': subjects_with_data,
                'condition1_mean': np.mean(data1),
                'condition2_mean': np.mean(data2),
                'mean_difference': np.mean(differences),
                'paired_ttest': {'t_stat': t_stat, 'p_value': p_value},
                'wilcoxon': {'w_stat': w_stat, 'p_value': w_p_value},
                'cohens_d': cohens_d,
                'data1': data1,
                'data2': data2
            }
    
    return results

def create_visualization(df1, df2, pairwise_results, condition1_name, condition2_name):
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
    PANEL_LETTER_FONT = 22
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

    # Data
    result = pairwise_results['Overt_perc']
    subjects = result['subjects']
    data1 = result['data1']
    data2 = result['data2']
    n = len(subjects)

    # Short condition labels for axes
    label1 = 'All channels'
    label2 = 'L/R angular\ngyrus'

    # Palette
    c_cond1 = '#4878CF'   # muted blue
    c_cond2 = '#D65F5F'   # muted red
    c_mean = '#222222'

    # Figure
    fig, ax = plt.subplots(figsize=(7, 6))

    # Paired slopegraph
    jitter = 0.03
    for i in range(n):
        ax.plot(
            [0, 1], [data1[i], data2[i]],
            color='#999999', linewidth=1.0, alpha=0.6, zorder=1,
        )
        ax.scatter(0 + np.random.uniform(-jitter, jitter), data1[i],
                   s=80, color=c_cond1, edgecolors='black', linewidths=0.8,
                   zorder=2)
        ax.scatter(1 + np.random.uniform(-jitter, jitter), data2[i],
                   s=80, color=c_cond2, edgecolors='black', linewidths=0.8,
                   zorder=2)

    # Mean line
    mean1, mean2 = np.mean(data1), np.mean(data2)
    ax.plot([0, 1], [mean1, mean2], color=c_mean, linewidth=3, zorder=3)
    ax.scatter([0, 1], [mean1, mean2], s=140, color=c_mean, edgecolors='white',
               linewidths=1.5, zorder=4, label='Mean')

    # Mean value annotations
    ax.text(-0.08, mean1, f'{mean1:.1f}%', ha='right', va='center',
            fontsize=MEAN_TEXT_FONT, fontweight='bold', color=c_cond1)
    ax.text(1.08, mean2, f'{mean2:.1f}%', ha='left', va='center',
            fontsize=MEAN_TEXT_FONT, fontweight='bold', color=c_cond2)

    ax.set_xticks([0, 1])
    ax.set_xticklabels([label1, label2], fontsize=AXIS_TICK_FONT)
    ax.set_xlim(-0.25, 1.25)
    ax.set_ylabel('Classification Accuracy (%)',
                   fontsize=AXIS_LABEL_FONT, fontweight='bold')
    ax.set_ylim(
        max(0, min(min(data1), min(data2)) - 5),
        min(105, max(max(data1), max(data2)) + 5),
    )
    ax.tick_params(axis='y', labelsize=AXIS_TICK_FONT)
    ax.legend(loc='lower left', frameon=True, edgecolor='black',
              fontsize=LEGEND_FONT, framealpha=0.95)
    ax.grid(True, alpha=0.25, linestyle=':', linewidth=0.8)
    ax.set_facecolor('#FAFAFA')

    # Significance bracket with paired t-test p-value
    p_t = result['paired_ttest']['p_value']
    if p_t < 0.001:
        sig_label = 'n.s. \u2014 p < 0.001' if p_t >= 0.05 else '*** \u2014 p < 0.001'
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
    print("Loading and analyzing KNN classifier accuracy data...")
    print("=" * 60)
    
    # Load data
    # Updated condition names for clarity
    condition1_name = "All Channels"
    condition2_name = "Left/Right Angular Gyrus"
    
    df1 = load_and_clean_data(file1_path, condition1_name)
    df2 = load_and_clean_data(file2_path, condition2_name)
    
    if df1 is None or df2 is None:
        print("Error: Could not load one or both files")
        return
    
    print("\n" + "=" * 60)
    print("DESCRIPTIVE STATISTICS")
    print("=" * 60)
    
    # Calculate descriptive statistics
    stats1 = calculate_descriptive_stats(df1, condition1_name)
    stats2 = calculate_descriptive_stats(df2, condition2_name)
    
    # Print descriptive statistics
    for stats_dict, name in [(stats1, condition1_name), (stats2, condition2_name)]:
        print(f"\n{name} Condition:")
        for column, stats in stats_dict.items():
            print(f"  {column}:")
            print(f"    Mean: {stats['mean']:.2f}%")
            print(f"    Median: {stats['median']:.2f}%")
            print(f"    Std: {stats['std']:.2f}%")
            print(f"    Range: {stats['min']:.2f}% - {stats['max']:.2f}%")
            print(f"    N: {stats['n']}")
    
    print("\n" + "=" * 60)
    print("PAIRWISE COMPARISON ANALYSIS")
    print("=" * 60)
    
    # Perform pairwise tests
    pairwise_results = perform_pairwise_tests(df1, df2)
    
    # Print pairwise results
    for condition, results in pairwise_results.items():
        print(f"\nOvert Accuracy Pairwise Analysis:")
        print(f"  Number of paired observations: {results['n_pairs']}")
        print(f"  Mean accuracy - {condition1_name}: {results['condition1_mean']:.2f}%")
        print(f"  Mean accuracy - {condition2_name}: {results['condition2_mean']:.2f}%")
        print(f"  Mean difference: {results['mean_difference']:.2f}%")
        print(f"  Paired t-test: t = {results['paired_ttest']['t_stat']:.3f}, p = {results['paired_ttest']['p_value']:.3f}")
        print(f"  Wilcoxon signed-rank test: W = {results['wilcoxon']['w_stat']:.3f}, p = {results['wilcoxon']['p_value']:.3f}")
        print(f"  Effect size (Cohen's d): {results['cohens_d']:.3f}")
        
        # Interpret effect size
        abs_d = abs(results['cohens_d'])
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
        if results['paired_ttest']['p_value'] < 0.001:
            significance = "highly significant (p < 0.001)"
        elif results['paired_ttest']['p_value'] < 0.01:
            significance = "very significant (p < 0.01)"
        elif results['paired_ttest']['p_value'] < 0.05:
            significance = "significant (p < 0.05)"
        else:
            significance = "not significant (p >= 0.05)"
        
        print(f"  Statistical significance: {significance}")
    
    # Create visualization
    print("\n" + "=" * 60)
    print("CREATING VISUALIZATION")
    print("=" * 60)
    
    fig = create_visualization(df1, df2, pairwise_results, condition1_name, condition2_name)
    
    # Create output directory if it doesn't exist
    output_dir = r"U:\eng_research_hrc_binauralhearinglab\Sudan\Labs\Sen Lab\Research_projects\Whole_Head_Cocktail_party\Classifier_script_results\nested\stats_allch_vs_LR_ang_gyr_snr_0_20feat"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Save the plot in three formats (PDF, SVG, PNG)
    stem = "rf_allchannels_vs_lrangyr_accuracy_pairwise_analysis"
    for ext in ['pdf', 'svg', 'png']:
        out_path = Path(output_dir) / f"{stem}.{ext}"
        fig.savefig(str(out_path), dpi=600, bbox_inches='tight', transparent=False)
        print(f"  Saved: {out_path}")
    print("Figures saved (PDF, SVG, PNG).")
    
    # Save detailed results to text file
    output_text_path = Path(output_dir) / "rf_accuracy_analysis_results.txt"
    with open(str(output_text_path), 'w') as f:
        f.write("All Channels vs Left/Right Angular Gyrus Pairwise Analysis Results\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("Data Files Analyzed:\n")
        f.write(f"1. {condition1_name}: {file1_path}\n")
        f.write(f"2. {condition2_name}: {file2_path}\n\n")
        
        f.write("Descriptive Statistics:\n")
        f.write("-" * 30 + "\n")
        for stats_dict, name in [(stats1, condition1_name), (stats2, condition2_name)]:
            f.write(f"\n{name} Condition:\n")
            for column, stats in stats_dict.items():
                f.write(f"  {column}:\n")
                f.write(f"    Mean: {stats['mean']:.2f}%\n")
                f.write(f"    Median: {stats['median']:.2f}%\n")
                f.write(f"    Std: {stats['std']:.2f}%\n")
                f.write(f"    Range: {stats['min']:.2f}% - {stats['max']:.2f}%\n")
                f.write(f"    N: {stats['n']}\n")
        
        f.write("\nPairwise Comparison Results:\n")
        f.write("-" * 30 + "\n")
        for condition, results in pairwise_results.items():
            f.write(f"\nOvert Accuracy:\n")
            f.write(f"  Paired observations: {results['n_pairs']}\n")
            f.write(f"  Mean difference: {results['mean_difference']:.2f}%\n")
            f.write(f"  Paired t-test: t = {results['paired_ttest']['t_stat']:.3f}, p = {results['paired_ttest']['p_value']:.3f}\n")
            f.write(f"  Wilcoxon test: W = {results['wilcoxon']['w_stat']:.3f}, p = {results['wilcoxon']['p_value']:.3f}\n")
            f.write(f"  Cohen's d: {results['cohens_d']:.3f}\n")
    
    print(f"Detailed results saved as: {output_text_path}")
    
    # Save summary statistics to CSV
    csv_output_path = Path(output_dir) / "summary_statistics.csv"
    
    # Create summary dataframe
    summary_data = []
    
    for stats_dict, name in [(stats1, condition1_name), (stats2, condition2_name)]:
        for column, stats in stats_dict.items():
            summary_data.append({
                'Condition': name,
                'Measurement': column,
                'Mean': stats['mean'],
                'Median': stats['median'],
                'Std': stats['std'],
                'Min': stats['min'],
                'Max': stats['max'],
                'N': stats['n']
            })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(str(csv_output_path), index=False)
    print(f"Summary statistics saved as: {csv_output_path}")
    
    # Save pairwise comparison results to CSV
    pairwise_csv_path = Path(output_dir) / "pairwise_comparison_results.csv"
    
    pairwise_data = []
    for condition, results in pairwise_results.items():
        pairwise_data.append({
            'Measurement': 'Overt',
            'N_Pairs': results['n_pairs'],
            'Mean_Condition1': results['condition1_mean'],
            'Mean_Condition2': results['condition2_mean'],
            'Mean_Difference': results['mean_difference'],
            'T_Statistic': results['paired_ttest']['t_stat'],
            'T_Test_P_Value': results['paired_ttest']['p_value'],
            'Wilcoxon_Statistic': results['wilcoxon']['w_stat'],
            'Wilcoxon_P_Value': results['wilcoxon']['p_value'],
            'Cohens_D': results['cohens_d']
        })
    
    if pairwise_data:
        pairwise_df = pd.DataFrame(pairwise_data)
        pairwise_df.to_csv(str(pairwise_csv_path), index=False)
        print(f"Pairwise comparison results saved as: {pairwise_csv_path}")
    
    # Save individual subject data for pairwise comparisons
    subjects_csv_path = Path(output_dir) / "subject_pairwise_data.csv"
    
    subject_data = []
    for condition, results in pairwise_results.items():
        subjects = results['subjects']
        data1 = results['data1']
        data2 = results['data2']
        
        for i, subject in enumerate(subjects):
            subject_data.append({
                'Subject': subject,
                'Measurement': 'Overt',
                'All_Channels': data1[i],
                'LR_Angular_Gyrus': data2[i],
                'Difference': data1[i] - data2[i]
            })
    
    if subject_data:
        subjects_df = pd.DataFrame(subject_data)
        subjects_df.to_csv(str(subjects_csv_path), index=False)
        print(f"Subject-level pairwise data saved as: {subjects_csv_path}")
    
    print(f"\nAll output files saved to: {output_dir}")
    
    plt.show()

if __name__ == "__main__":
    main()
