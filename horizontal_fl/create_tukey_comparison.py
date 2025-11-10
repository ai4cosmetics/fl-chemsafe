"""
Statistical comparison of model performance using Tukey's HSD test
Creates forest plot with confidence intervals for pairwise comparisons
Based on statsmodels: https://www.statsmodels.org/dev/index.html
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
import xgboost as xgb
import os
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

# Set Arial as the default font
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.sans-serif'] = ['Arial']

def calculate_metrics_from_predictions(y_true, y_pred_proba, threshold=0.5):
    """
    Calculate all metrics from predictions
    """
    y_pred = (y_pred_proba > threshold).astype(int)
    
    # Calculate metrics
    auc = roc_auc_score(y_true, y_pred_proba)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # False Negative Rate = 1 - Recall (Sensitivity)
    # FNR = FN / (FN + TP)
    fnr = 1 - recall
    
    return {
        'AUC': auc,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1': f1,
        'FNR': fnr
    }

def bootstrap_metrics_from_predictions(y_true, y_pred_proba, threshold=0.5, n_bootstrap=100, random_state=42):
    """
    Bootstrap sampling on predictions to estimate metric distributions
    """
    np.random.seed(random_state)
    
    metrics_dict = {
        'AUC': [],
        'Accuracy': [],
        'Precision': [],
        'Recall': [],
        'F1': [],
        'FNR': []
    }
    
    n_samples = len(y_true)
    
    for i in range(n_bootstrap):
        # Bootstrap sample indices
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        y_true_boot = y_true[indices]
        y_pred_proba_boot = y_pred_proba[indices]
        
        # Calculate metrics on bootstrap sample
        metrics = calculate_metrics_from_predictions(y_true_boot, y_pred_proba_boot, threshold)
        
        for metric_name, value in metrics.items():
            metrics_dict[metric_name].append(value)
    
    # Convert to numpy arrays
    for metric_name in metrics_dict:
        metrics_dict[metric_name] = np.array(metrics_dict[metric_name])
    
    return metrics_dict

def simulate_metrics_from_results(results_row, n_bootstrap=100, random_state=42):
    """
    Simulate bootstrap distributions for metrics using binomial approximation
    This is used for local models where we don't have global test predictions
    """
    np.random.seed(random_state)
    
    n_test = results_row['Test_Samples']
    
    # Get existing metrics
    auc = results_row['AUC']
    accuracy = results_row['Accuracy']
    precision = results_row['Precision']
    recall = results_row['Recall']
    f1 = results_row['F1']
    fnr = 1 - recall  # FNR = 1 - Recall
    
    # Calculate standard errors based on sample size
    # Using binomial approximation for proportions
    metrics_dict = {}
    
    for metric_name, metric_value in [('AUC', auc), ('Accuracy', accuracy), 
                                       ('Precision', precision), ('Recall', recall), 
                                       ('F1', f1), ('FNR', fnr)]:
        # Standard error depends on the metric
        if metric_name == 'AUC':
            # AUC standard error is more complex, use simplified version
            std_err = np.sqrt(metric_value * (1 - metric_value) / n_test)
        else:
            # For other metrics, use binomial approximation
            std_err = np.sqrt(metric_value * (1 - metric_value) / n_test)
        
        # Generate bootstrap distribution
        samples = np.random.normal(metric_value, std_err, n_bootstrap)
        
        # Clip to valid range
        if metric_name == 'AUC':
            samples = np.clip(samples, 0, 1)
        else:
            samples = np.clip(samples, 0, 1)
        
        metrics_dict[metric_name] = samples
    
    return metrics_dict

def get_all_model_metrics(n_bootstrap=100):
    """
    Load existing model results and predictions to generate bootstrap metric distributions
    """
    # Check if required files exist
    results_file = Path("results/complete_comparison.csv")
    predictions_file = Path("results/global_test_predictions.csv")
    
    if not results_file.exists():
        raise FileNotFoundError(f"Results file not found: {results_file}")
    if not predictions_file.exists():
        raise FileNotFoundError(f"Predictions file not found: {predictions_file}")
    
    # Load results
    results_df = pd.read_csv(results_file)
    predictions_df = pd.read_csv(predictions_file)
    
    print("Loaded existing results and predictions")
    print(f"Models found: {results_df['Model'].tolist()}")
    
    # Get true labels
    y_true = predictions_df['true_labels'].values
    
    all_models_metrics = {}
    
    # Federated Learning (FL)
    print("\nGenerating bootstrap samples for Federated Learning...")
    if 'federated_predictions' in predictions_df.columns:
        y_pred_proba_fl = predictions_df['federated_predictions'].values
        all_models_metrics['FL'] = bootstrap_metrics_from_predictions(
            y_true, y_pred_proba_fl, n_bootstrap=n_bootstrap
        )
        print(f"  Accuracy: {np.mean(all_models_metrics['FL']['Accuracy']):.4f}")
    
    # Centralised Learning (CL)
    print("Generating bootstrap samples for Centralised Learning...")
    if 'centralised_predictions' in predictions_df.columns:
        y_pred_proba_cl = predictions_df['centralised_predictions'].values
        all_models_metrics['CL'] = bootstrap_metrics_from_predictions(
            y_true, y_pred_proba_cl, n_bootstrap=n_bootstrap
        )
        print(f"  Accuracy: {np.mean(all_models_metrics['CL']['Accuracy']):.4f}")
    
    # Local Learning - AI4Cosmetics (LL_AI4C)
    print("Generating bootstrap samples for Local AI4Cosmetics...")
    ai4c_row = results_df[results_df['Model'] == 'AI4Cosmetics Local']
    if not ai4c_row.empty:
        all_models_metrics['LL_AI4C'] = simulate_metrics_from_results(
            ai4c_row.iloc[0], n_bootstrap=n_bootstrap
        )
        print(f"  Accuracy: {np.mean(all_models_metrics['LL_AI4C']['Accuracy']):.4f}")
    
    # Local Learning - SkinDoctorCP (LL_Wilm)
    print("Generating bootstrap samples for Local SkinDoctorCP...")
    wilm_row = results_df[results_df['Model'] == 'SkinDoctorCP Local']
    if not wilm_row.empty:
        all_models_metrics['LL_Wilm'] = simulate_metrics_from_results(
            wilm_row.iloc[0], n_bootstrap=n_bootstrap
        )
        print(f"  Accuracy: {np.mean(all_models_metrics['LL_Wilm']['Accuracy']):.4f}")
    
    return all_models_metrics, results_df

def perform_tukey_hsd_for_metric(all_models_metrics, metric_name):
    """
    Perform Tukey HSD test for a specific metric
    """
    # Prepare data for Tukey HSD
    all_values = []
    all_groups = []
    
    for model_name, metrics_dict in all_models_metrics.items():
        metric_values = metrics_dict[metric_name]
        all_values.extend(metric_values)
        all_groups.extend([model_name] * len(metric_values))
    
    # Perform Tukey HSD test
    tukey_result = pairwise_tukeyhsd(
        endog=all_values,
        groups=all_groups,
        alpha=0.05
    )
    
    return tukey_result

def create_subplot_for_metric(ax, all_models_metrics, metric_name, tukey_result, xlim=None, display_title=None):
    """
    Create a single subplot for one metric
    
    Parameters:
    -----------
    ax : matplotlib axis
        The axis to plot on
    all_models_metrics : dict
        Dictionary containing metrics for all models
    metric_name : str
        Name of the metric key to access data
    tukey_result : TukeyHSDResults
        Results from Tukey HSD test
    xlim : tuple, optional
        X-axis limits (min, max). If None, auto-scale
    display_title : str, optional
        Display title for the metric. If None, uses metric_name
    """
    # Use display title if provided, otherwise use metric_name
    title = display_title if display_title is not None else metric_name
    # Define pairwise comparisons in desired order
    comparisons = [
        ('FL', 'CL', 'FL vs CL'),
        ('FL', 'LL_AI4C', 'FL vs LL (AI4Cosmetics)'),
        ('FL', 'LL_Wilm', 'FL vs LL (Wilm et al., 2021)'),
        ('CL', 'LL_AI4C', 'CL vs LL (AI4Cosmetics)'),
        ('CL', 'LL_Wilm', 'CL vs LL (Wilm et al., 2021)'),
        ('LL_AI4C', 'LL_Wilm', 'LL (AI4Cosmetics) vs LL (Wilm et al., 2021)')
    ]
    
    # Calculate means and confidence intervals for differences
    results = []
    
    # Convert Tukey result to DataFrame for easier lookup
    tukey_df = pd.DataFrame(data=tukey_result.summary().data[1:], 
                            columns=tukey_result.summary().data[0])
    
    for group1, group2, label in comparisons:
        if group1 not in all_models_metrics or group2 not in all_models_metrics:
            continue
        
        # Get metric values for both groups
        values1 = all_models_metrics[group1][metric_name]
        values2 = all_models_metrics[group2][metric_name]
        
        # Calculate difference (group1 - group2)
        diff = values1 - values2
        mean_diff = np.mean(diff)
        
        # 95% confidence interval
        ci_lower, ci_upper = np.percentile(diff, [2.5, 97.5])
        
        # Check if significantly different from Tukey result
        mask1 = (tukey_df['group1'] == group1) & (tukey_df['group2'] == group2)
        mask2 = (tukey_df['group1'] == group2) & (tukey_df['group2'] == group1)
        row = tukey_df[mask1 | mask2]
        
        if not row.empty:
            reject = row['reject'].values[0]
            # If comparison is reversed in Tukey results, flip the sign
            if mask2.any() and not mask1.any():
                mean_diff = -mean_diff
                ci_lower, ci_upper = -ci_upper, -ci_lower
        else:
            reject = False
        
        results.append({
            'label': label,
            'mean_diff': mean_diff,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'significant': reject
        })
    
    # Determine colors
    colors = []
    for r in results:
        if not r['significant']:
            colors.append('#808080')  # Grey - Not significantly different
        elif r['mean_diff'] < 0:
            colors.append('#D32F2F')  # Red - First model significantly worse
        else:
            colors.append('#1976D2')  # Blue - First model significantly better
    
    # Plot error bars and points with increased vertical spacing
    y_positions = np.arange(len(results)) * 1.5  # Increased vertical spacing for readability
    
    for i, (r, color) in enumerate(zip(results, colors)):
        ax.errorbar(
            r['mean_diff'], y_positions[i],
            xerr=[[r['mean_diff'] - r['ci_lower']], [r['ci_upper'] - r['mean_diff']]],
            fmt='o',
            color=color,
            markersize=20,
            linewidth=5,
            capsize=12,
            capthick=5,
            elinewidth=5
        )
    
    # Add vertical line at 0
    ax.axvline(x=0, color='black', linestyle='--', linewidth=3, alpha=0.7)
    
    # Set labels - OPTIMIZED FONT SIZES
    ax.set_yticks(y_positions)
    ax.set_yticklabels([r['label'] for r in results], fontsize=30)
    ax.set_xlabel(f'{title} Difference (95% CI)', fontsize=30, fontweight='bold', labelpad=12)
    ax.set_title(f'{title}', fontsize=50, fontweight='bold', pad=20)
    
    # Set tick label size for x-axis
    ax.tick_params(axis='x', labelsize=30)
    
    # Set x-axis limits if provided
    if xlim is not None:
        ax.set_xlim(xlim)
    
    # Add grid
    ax.grid(axis='x', alpha=0.3, linestyle=':', linewidth=2)
    ax.set_axisbelow(True)
    
    # Ensure consistent subplot border styling
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1.5)
    
    return results

def create_comprehensive_forest_plot(all_models_metrics, n_bootstrap=100):
    """
    Create a single figure with subplots for all metrics
    """
    # Define metrics and their display titles
    metrics = ['AUC', 'Accuracy', 'Precision', 'Recall', 'F1', 'FNR']
    metric_titles = {
        'AUC': 'Area Under the ROC Curve (AUC)',
        'Accuracy': 'Accuracy',
        'Precision': 'Precision',
        'Recall': 'Recall',
        'F1': 'F1',
        'FNR': 'False Negative Rate'
    }
    
    # Sort metrics alphabetically by their display titles
    sorted_metrics = sorted(metrics, key=lambda m: metric_titles[m])
    
    print("\n" + "=" * 70)
    print("Performing Tukey HSD tests for all metrics...")
    print("=" * 70)
    
    # First pass: Calculate all results to find global max
    all_results = {}
    all_tukey_results = {}
    global_max_upper_ci = -np.inf
    
    for metric_name in sorted_metrics:
        print(f"\nProcessing {metric_name}...")
        
        # Perform Tukey HSD for this metric
        tukey_result = perform_tukey_hsd_for_metric(all_models_metrics, metric_name)
        all_tukey_results[metric_name] = tukey_result
        
        # Calculate results to find max CI
        comparisons = [
            ('FL', 'CL', 'FL vs CL'),
            ('FL', 'LL_AI4C', 'FL vs LL (AI4Cosmetics)'),
            ('FL', 'LL_Wilm', 'FL vs LL (Wilm et al., 2021)'),
            ('CL', 'LL_AI4C', 'CL vs LL (AI4Cosmetics)'),
            ('CL', 'LL_Wilm', 'CL vs LL (Wilm et al., 2021)'),
            ('LL_AI4C', 'LL_Wilm', 'LL (AI4Cosmetics) vs LL (Wilm et al., 2021)')
        ]
        
        for group1, group2, label in comparisons:
            if group1 in all_models_metrics and group2 in all_models_metrics:
                values1 = all_models_metrics[group1][metric_name]
                values2 = all_models_metrics[group2][metric_name]
                diff = values1 - values2
                ci_upper = np.percentile(diff, 97.5)
                global_max_upper_ci = max(global_max_upper_ci, ci_upper)
        
        print(f"  Completed {metric_name}")
    
    # Set uniform x-axis limits for all subplots
    xlim = (-0.2, global_max_upper_ci * 1.05)  # Add 5% padding to max
    print(f"\nGlobal x-axis limits: [{xlim[0]:.3f}, {xlim[1]:.3f}]")
    
    # Create figure with subplots (3 rows x 2 columns)
    # Optimized size to match font sizes
    fig, axes = plt.subplots(3, 2, figsize=(32, 32))
    axes = axes.flatten()
    
    # Second pass: Create subplots with uniform scale
    print("\n" + "=" * 70)
    print("Creating subplots with uniform scale...")
    print("=" * 70)
    
    for idx, metric_name in enumerate(sorted_metrics):
        # Create subplot with uniform xlim using display title
        results = create_subplot_for_metric(
            axes[idx], 
            all_models_metrics, 
            metric_name,  # Keep original metric name as key
            all_tukey_results[metric_name],
            xlim=xlim,
            display_title=metric_titles[metric_name]  # Pass display title separately
        )
        all_results[metric_name] = results
    
    # Create legend (add to the last subplot area or figure)
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#1976D2', edgecolor='black', label='First model significantly better'),
        Patch(facecolor='#D32F2F', edgecolor='black', label='First model significantly worse'),
        Patch(facecolor='#808080', edgecolor='black', label='No significant difference')
    ]
    
    # Add legend to figure - horizontal layout with smaller font
    fig.legend(handles=legend_elements, loc='lower center', ncol=3, 
              frameon=True, fancybox=True, shadow=True, fontsize=38, 
              bbox_to_anchor=(0.5, 0.02))
    
    # Adjust layout: increased horizontal and bottom spacing to prevent overlap
    plt.subplots_adjust(hspace=0.40, wspace=0.85, left=0.08, right=0.98, top=0.96, bottom=0.12)
    
    # Save figure
    os.makedirs("plots", exist_ok=True)
    output_path = "plots/tukey_hsd_all_metrics.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n{'=' * 70}")
    print(f"Comprehensive forest plot saved to: {output_path}")
    
    # Save all results to CSV
    all_comparisons = []
    for metric_name, results in all_results.items():
        for r in results:
            all_comparisons.append({
                'Metric': metric_titles[metric_name],  # Use display title
                'Comparison': r['label'],
                'Mean_Difference': r['mean_diff'],
                'CI_Lower': r['ci_lower'],
                'CI_Upper': r['ci_upper'],
                'Significant': r['significant']
            })
    
    summary_df = pd.DataFrame(all_comparisons)
    summary_path = "results/tukey_hsd_all_metrics_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary statistics saved to: {summary_path}")
    print("=" * 70)
    
    return fig, all_results

def print_mean_metrics_summary(all_models_metrics):
    """
    Print summary of mean metrics for all models
    """
    print("\n" + "=" * 70)
    print("Mean Metrics from Bootstrap Samples:")
    print("=" * 70)
    
    model_names = {
        'FL': 'Federated Learning',
        'CL': 'Centralised Learning',
        'LL_AI4C': 'Local Learning (AI4Cosmetics)',
        'LL_Wilm': 'Local Learning (Wilm et al., 2021)'
    }
    
    metrics = ['AUC', 'Accuracy', 'Precision', 'Recall', 'F1', 'FNR']
    
    for model_code, metrics_dict in all_models_metrics.items():
        model_name = model_names.get(model_code, model_code)
        print(f"\n{model_name}:")
        for metric_name in metrics:
            mean_val = np.mean(metrics_dict[metric_name])
            std_val = np.std(metrics_dict[metric_name])
            print(f"  {metric_name:12s}: {mean_val:.4f} ± {std_val:.4f}")

def main():
    """Main function to run Tukey HSD analysis and create visualization"""
    print("\n" + "=" * 70)
    print("Tukey HSD Statistical Comparison Analysis")
    print("All Metrics: AUC, Accuracy, Precision, Recall, F1, FNR")
    print("Based on: https://www.statsmodels.org/dev/index.html")
    print("=" * 70 + "\n")
    
    # Get all metrics for all models
    all_models_metrics, results_df = get_all_model_metrics(n_bootstrap=100)
    
    # Print summary
    print_mean_metrics_summary(all_models_metrics)
    
    # Create comprehensive visualization
    print("\n" + "=" * 70)
    print("Creating comprehensive forest plot visualization...")
    print("=" * 70)
    create_comprehensive_forest_plot(all_models_metrics, n_bootstrap=100)
    
    print("\n" + "=" * 70)
    print("Analysis complete!")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    main()
