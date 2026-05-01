#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Permutation-test analysis for the Random Forest classifier.

Loads per-subject permutation-test outputs from the RF classifier pipeline
and computes empirical chance-threshold bounds at the requested alpha level.
The chance threshold is the value figure scripts overlay on accuracy plots.

Author: Sudan Duwadi <sudan@bu.edu>
Notes: Code refactoring, documentation, and commenting were AI-assisted;
       all scientific decisions and accountability remain with the author.
"""

import pickle
import json
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import stats

def load_permutation_results(filepath):
    """Load permutation test results from a .pkl or .json file."""
    ext = os.path.splitext(filepath)[1].lower()
    if ext == '.json':
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def calculate_significance_bounds(permutation_accuracies, alpha=0.05):
    """
    Calculate statistical significance bounds for classification accuracy.
    
    Parameters:
    -----------
    permutation_accuracies : array
        Array of accuracy values from permutation test
    alpha : float
        Significance level (default 0.05 for p < 0.05)
        
    Returns:
    --------
    bounds : dict
        Dictionary containing various statistical bounds
    """
    
    # Sort accuracies for percentile calculations
    sorted_acc = np.sort(permutation_accuracies)
    n_perms = len(sorted_acc)
    
    # Calculate confidence intervals
    lower_bound = np.percentile(sorted_acc, (alpha/2) * 100)     # 2.5th percentile for α=0.05
    upper_bound = np.percentile(sorted_acc, (1 - alpha/2) * 100) # 97.5th percentile for α=0.05
    
    # One-tailed bounds (more common for classification)
    upper_bound_one_tail = np.percentile(sorted_acc, (1 - alpha) * 100)  # 95th percentile for α=0.05
    lower_bound_one_tail = np.percentile(sorted_acc, alpha * 100)         # 5th percentile for α=0.05
    
    # Calculate mean and standard deviation
    mean_acc = np.mean(sorted_acc)
    std_acc = np.std(sorted_acc)
    
    # Z-score based bounds (assuming normal distribution)
    z_critical = stats.norm.ppf(1 - alpha/2)  # Two-tailed
    z_bound_upper = mean_acc + z_critical * std_acc
    z_bound_lower = mean_acc - z_critical * std_acc
    
    # One-tailed Z-score bounds
    z_critical_one = stats.norm.ppf(1 - alpha)
    z_bound_upper_one = mean_acc + z_critical_one * std_acc
    z_bound_lower_one = mean_acc - z_critical_one * std_acc
    
    bounds = {
        'alpha': alpha,
        'n_permutations': n_perms,
        'mean': mean_acc,
        'std': std_acc,
        'median': np.median(sorted_acc),
        
        # Two-tailed bounds (α/2 in each tail)
        'two_tailed': {
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'lower_percentile': (alpha/2) * 100,
            'upper_percentile': (1 - alpha/2) * 100
        },
        
        # One-tailed bounds (α in one tail)
        'one_tailed': {
            'lower_bound': lower_bound_one_tail,
            'upper_bound': upper_bound_one_tail,
            'lower_percentile': alpha * 100,
            'upper_percentile': (1 - alpha) * 100
        },
        
        # Z-score based bounds (parametric)
        'z_score_two_tailed': {
            'lower_bound': z_bound_lower,
            'upper_bound': z_bound_upper,
            'z_critical': z_critical
        },
        
        'z_score_one_tailed': {
            'lower_bound': z_bound_lower_one,
            'upper_bound': z_bound_upper_one,
            'z_critical': z_critical_one
        }
    }
    
    return bounds

def test_observed_accuracy_significance(observed_accuracy, permutation_accuracies, alpha=0.05):
    """
    Test if observed accuracy is significantly different from chance.
    
    Parameters:
    -----------
    observed_accuracy : float
        The observed classification accuracy to test
    permutation_accuracies : array
        Array of accuracy values from permutation test
    alpha : float
        Significance level
        
    Returns:
    --------
    test_results : dict
        Statistical test results
    """
    
    # Two-tailed test: is observed accuracy significantly different from chance?
    p_value_two_tailed = 2 * min(
        np.mean(permutation_accuracies >= observed_accuracy),
        np.mean(permutation_accuracies <= observed_accuracy)
    )
    
    # One-tailed test: is observed accuracy significantly better than chance?
    p_value_one_tailed_upper = np.mean(permutation_accuracies >= observed_accuracy)
    
    # One-tailed test: is observed accuracy significantly worse than chance?
    p_value_one_tailed_lower = np.mean(permutation_accuracies <= observed_accuracy)
    
    # Effect size (Cohen's d)
    mean_chance = np.mean(permutation_accuracies)
    std_chance = np.std(permutation_accuracies)
    cohens_d = (observed_accuracy - mean_chance) / std_chance
    
    test_results = {
        'observed_accuracy': observed_accuracy,
        'mean_chance_level': mean_chance,
        'std_chance_level': std_chance,
        'cohens_d': cohens_d,
        
        'two_tailed_test': {
            'p_value': p_value_two_tailed,
            'significant': p_value_two_tailed < alpha,
            'interpretation': 'significantly different from chance' if p_value_two_tailed < alpha else 'not significantly different from chance'
        },
        
        'one_tailed_test_upper': {
            'p_value': p_value_one_tailed_upper,
            'significant': p_value_one_tailed_upper < alpha,
            'interpretation': 'significantly better than chance' if p_value_one_tailed_upper < alpha else 'not significantly better than chance'
        },
        
        'one_tailed_test_lower': {
            'p_value': p_value_one_tailed_lower,
            'significant': p_value_one_tailed_lower < alpha,
            'interpretation': 'significantly worse than chance' if p_value_one_tailed_lower < alpha else 'not significantly worse than chance'
        }
    }
    
    return test_results

def create_visualization(permutation_accuracies, bounds, observed_accuracy=None, save_path=None):
    """
    Create comprehensive visualization of permutation test results.
    """
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Histogram with bounds
    ax1.hist(permutation_accuracies, bins=50, alpha=0.7, density=True, 
             color='lightblue', edgecolor='black', label='Permutation Results')
    
    # Add significance bounds
    ax1.axvline(bounds['two_tailed']['lower_bound'], color='red', linestyle='--', 
                label=f"95% CI: {bounds['two_tailed']['lower_bound']:.3f}")
    ax1.axvline(bounds['two_tailed']['upper_bound'], color='red', linestyle='--',
                label=f"95% CI: {bounds['two_tailed']['upper_bound']:.3f}")
    ax1.axvline(bounds['mean'], color='green', linestyle='-', linewidth=2,
                label=f"Mean: {bounds['mean']:.3f}")
    
    # Add observed accuracy if provided
    if observed_accuracy is not None:
        ax1.axvline(observed_accuracy, color='orange', linestyle='-', linewidth=3,
                    label=f"Observed: {observed_accuracy:.3f}")
    
    ax1.axvline(0.5, color='black', linestyle=':', alpha=0.7, label='Theoretical Chance (0.5)')
    
    ax1.set_xlabel('Classification Accuracy')
    ax1.set_ylabel('Density')
    ax1.set_title('Permutation Test Results Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Q-Q plot to check normality
    stats.probplot(permutation_accuracies, dist="norm", plot=ax2)
    ax2.set_title('Q-Q Plot: Normal Distribution Check')
    ax2.grid(True, alpha=0.3)
    
    # 3. Cumulative distribution
    sorted_acc = np.sort(permutation_accuracies)
    cumulative_prob = np.arange(1, len(sorted_acc) + 1) / len(sorted_acc)
    
    ax3.plot(sorted_acc, cumulative_prob, 'b-', linewidth=2, label='Empirical CDF')
    ax3.axhline(0.05, color='red', linestyle='--', alpha=0.7, label='5th percentile')
    ax3.axhline(0.95, color='red', linestyle='--', alpha=0.7, label='95th percentile')
    ax3.axhline(0.5, color='green', linestyle='-', alpha=0.7, label='Median')
    
    if observed_accuracy is not None:
        obs_percentile = np.mean(permutation_accuracies <= observed_accuracy)
        ax3.axvline(observed_accuracy, color='orange', linestyle='-', linewidth=2)
        ax3.axhline(obs_percentile, color='orange', linestyle=':', alpha=0.7,
                    label=f'Observed percentile: {obs_percentile:.3f}')
    
    ax3.set_xlabel('Classification Accuracy')
    ax3.set_ylabel('Cumulative Probability')
    ax3.set_title('Cumulative Distribution Function')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Summary statistics table
    ax4.axis('tight')
    ax4.axis('off')
    
    # Create summary table
    summary_data = [
        ['Statistic', 'Value'],
        ['N Permutations', f"{bounds['n_permutations']:,}"],
        ['Mean Accuracy', f"{bounds['mean']:.4f}"],
        ['Std Deviation', f"{bounds['std']:.4f}"],
        ['Median', f"{bounds['median']:.4f}"],
        ['', ''],
        ['Two-tailed 95% CI', ''],
        [f"  Lower ({bounds['two_tailed']['lower_percentile']:.1f}%)", f"{bounds['two_tailed']['lower_bound']:.4f}"],
        [f"  Upper ({bounds['two_tailed']['upper_percentile']:.1f}%)", f"{bounds['two_tailed']['upper_bound']:.4f}"],
        ['', ''],
        ['One-tailed bounds', ''],
        [f"  Lower ({bounds['one_tailed']['lower_percentile']:.1f}%)", f"{bounds['one_tailed']['lower_bound']:.4f}"],
        [f"  Upper ({bounds['one_tailed']['upper_percentile']:.1f}%)", f"{bounds['one_tailed']['upper_bound']:.4f}"],
    ]
    
    if observed_accuracy is not None:
        test_results = test_observed_accuracy_significance(observed_accuracy, permutation_accuracies)
        summary_data.extend([
            ['', ''],
            ['Observed Accuracy', f"{observed_accuracy:.4f}"],
            ['Cohen\'s d', f"{test_results['cohens_d']:.3f}"],
            ['p-value (two-tailed)', f"{test_results['two_tailed_test']['p_value']:.4f}"],
            ['p-value (one-tailed)', f"{test_results['one_tailed_test_upper']['p_value']:.4f}"],
        ])
    
    table = ax4.table(cellText=summary_data, cellLoc='left', loc='center',
                      colWidths=[0.6, 0.4])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    
    # Style the header row
    for i in range(2):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax4.set_title('Summary Statistics', pad=20)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")
    
    plt.show()
    
    return fig

def main():
    """Main analysis function."""
    
    # File path to RF permutation summary
    results_file = (
        "U:\\eng_research_hrc_binauralhearinglab\\Sudan\\Labs\\Sen Lab\\"
        "Research_projects\\Whole_Head_Cocktail_party\\Classifier_script_results\\"
        "permutation\\rf_baseline_only\\sub_10_overt\\permutation_summary.json"
    )
    
    print(" RF PERMUTATION TEST RESULTS ANALYSIS")
    print("="*50)
    
    # Load results
    try:
        results = load_permutation_results(results_file)
        permutation_accuracies = np.array(results['permutation_accuracies'])
        print(f" Loaded {len(permutation_accuracies)} permutation results")
    except FileNotFoundError:
        print(f" File not found: {results_file}")
        return
    except Exception as e:
        print(f" Error loading file: {e}")
        return
    
    # Calculate significance bounds for different alpha levels
    alpha_levels = [0.05, 0.01, 0.001]
    
    print(f"\n STATISTICAL SIGNIFICANCE BOUNDS")
    print("-" * 50)
    
    for alpha in alpha_levels:
        bounds = calculate_significance_bounds(permutation_accuracies, alpha=alpha)
        
        print(f"\n Significance level α = {alpha}")
        print(f"   Mean chance level: {bounds['mean']:.4f} ± {bounds['std']:.4f}")
        print(f"   Two-tailed {(1-alpha)*100:.0f}% CI: [{bounds['two_tailed']['lower_bound']:.4f}, {bounds['two_tailed']['upper_bound']:.4f}]")
        print(f"   One-tailed bounds: Lower {alpha*100:.0f}% = {bounds['one_tailed']['lower_bound']:.4f}, Upper {(1-alpha)*100:.0f}% = {bounds['one_tailed']['upper_bound']:.4f}")
        
        # For α=0.05, create detailed analysis
        if alpha == 0.05:
            main_bounds = bounds
    
    print(f"\n INTERPRETATION FOR α = 0.05:")
    print("-" * 50)
    print(f"• Any observed accuracy > {main_bounds['one_tailed']['upper_bound']:.4f} is significantly BETTER than chance (p < 0.05)")
    print(f"• Any observed accuracy < {main_bounds['one_tailed']['lower_bound']:.4f} is significantly WORSE than chance (p < 0.05)")
    print(f"• Accuracies between {main_bounds['one_tailed']['lower_bound']:.4f} and {main_bounds['one_tailed']['upper_bound']:.4f} are NOT significantly different from chance")
    
    # Example: Test a hypothetical observed accuracy
    print(f"\n🧪 EXAMPLE: Testing hypothetical observed accuracies")
    print("-" * 50)
    
    test_accuracies = [0.55, 0.60, 0.65, 0.70, 0.75]
    
    for test_acc in test_accuracies:
        test_results = test_observed_accuracy_significance(test_acc, permutation_accuracies)
        significance = " SIGNIFICANT" if test_results['one_tailed_test_upper']['significant'] else " NOT SIGNIFICANT"
        print(f"Accuracy {test_acc:.2f}: p = {test_results['one_tailed_test_upper']['p_value']:.4f} {significance}")
    
    # Create visualization
    print(f"\n Creating RF visualization...")
    rf_output_dir = os.path.join(os.path.dirname(results_file), "rf_analysis_outputs")
    os.makedirs(rf_output_dir, exist_ok=True)
    viz_path = os.path.join(rf_output_dir, "permutation_analysis_visualization_rf.png")
    
    # You can test with an example observed accuracy
    example_observed = 0.65  # Replace with your actual observed accuracy
    
    fig = create_visualization(
        permutation_accuracies, 
        main_bounds, 
        observed_accuracy=example_observed,
        save_path=viz_path
    )
    
    # Save detailed results
    analysis_results = {
        'permutation_data': {
            'accuracies': permutation_accuracies.tolist(),
            'n_permutations': len(permutation_accuracies)
        },
        'significance_bounds': {f'alpha_{alpha}': calculate_significance_bounds(permutation_accuracies, alpha) 
                               for alpha in alpha_levels},
        'example_tests': {f'accuracy_{acc}': test_observed_accuracy_significance(acc, permutation_accuracies) 
                         for acc in test_accuracies}
    }
    
    analysis_file = os.path.join(rf_output_dir, "permutation_significance_analysis_rf.pkl")
    with open(analysis_file, 'wb') as f:
        pickle.dump(analysis_results, f)

    # Also save a JSON copy for quick inspection
    analysis_json_file = os.path.join(rf_output_dir, "permutation_significance_analysis_rf.json")
    with open(analysis_json_file, 'w', encoding='utf-8') as f:
        json.dump(analysis_results, f, indent=2)
    
    print(f"\n Analysis results saved to: {analysis_file}")
    print(f" JSON results saved to: {analysis_json_file}")
    print(f" Analysis complete!")

if __name__ == "__main__":
    main()
