#!/usr/bin/env python3
"""
Enhanced Visualization Script for De-SaTE-Transformer Results

This script creates comprehensive visualizations for model performance evaluation,
including detailed comparison plots, error analysis, and statistical summaries.

Usage:
    python scripts/visualize_results.py --results_dir results/ --test_battery JBGSRS250006644
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from sklearn.metrics import r2_score, mean_absolute_percentage_error

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.metrics import get_rmse, get_mae, get_rul_error, calculate_all_metrics
from utils.evaluation import evaluate_model_comprehensive

# Set style for better looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


def load_results(results_dir, test_battery):
    """
    Load training results and model predictions.
    
    Args:
        results_dir (str): Directory containing results
        test_battery (str): Name of the test battery
    
    Returns:
        dict: Dictionary containing loaded results
    """
    results = {}
    
    # Look for common result file patterns
    possible_files = [
        f'training_results_{test_battery}.png',
        f'final_model_{test_battery}.pth',
        f'predictions_{test_battery}.npy',
        f'ground_truth_{test_battery}.npy'
    ]
    
    for file_name in possible_files:
        file_path = os.path.join(results_dir, file_name)
        if os.path.exists(file_path):
            results[file_name] = file_path
    
    print(f"Found {len(results)} result files for {test_battery}")
    return results


def calculate_comprehensive_metrics(y_true, y_pred):
    """
    Calculate comprehensive evaluation metrics.
    
    Args:
        y_true (np.array): Ground truth values
        y_pred (np.array): Predicted values
    
    Returns:
        dict: Dictionary of metrics
    """
    metrics = {}
    
    # Basic metrics
    metrics['RMSE'] = get_rmse(y_true, y_pred)
    metrics['MAE'] = get_mae(y_true, y_pred)
    metrics['MAPE'] = mean_absolute_percentage_error(y_true, y_pred) * 100
    metrics['R2'] = r2_score(y_true, y_pred)
    
    # Additional metrics
    metrics['MSE'] = np.mean((y_true - y_pred) ** 2)
    metrics['Max_Error'] = np.max(np.abs(y_true - y_pred))
    metrics['Mean_Error'] = np.mean(y_true - y_pred)
    metrics['Std_Error'] = np.std(y_true - y_pred)
    
    # Correlation coefficient
    correlation, p_value = stats.pearsonr(y_true, y_pred)
    metrics['Pearson_Correlation'] = correlation
    metrics['P_Value'] = p_value
    
    return metrics


def create_comprehensive_visualization(y_true, y_pred, train_losses=None, 
                                     test_battery="Unknown", save_dir="results"):
    """
    Create comprehensive visualization plots.
    
    Args:
        y_true (np.array): Ground truth values
        y_pred (np.array): Predicted values
        train_losses (list): Training loss history
        test_battery (str): Name of the test battery
        save_dir (str): Directory to save plots
    """
    
    # Calculate metrics
    metrics = calculate_comprehensive_metrics(y_true, y_pred)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Prediction vs Actual Scatter Plot
    plt.subplot(3, 3, 1)
    plt.scatter(y_true, y_pred, alpha=0.6, s=50)
    
    # Perfect prediction line
    min_val, max_val = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    # Regression line
    z = np.polyfit(y_true, y_pred, 1)
    p = np.poly1d(z)
    plt.plot(y_true, p(y_true), 'g-', lw=2, alpha=0.7, label=f'Regression Line (R²={metrics["R2"]:.4f})')
    
    plt.xlabel('Actual SOH')
    plt.ylabel('Predicted SOH')
    plt.title('Prediction vs Actual SOH')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add text box with metrics
    textstr = f'RMSE: {metrics["RMSE"]:.4f}\nMAE: {metrics["MAE"]:.4f}\nMAPE: {metrics["MAPE"]:.2f}%'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=props)
    
    # 2. Time Series Comparison
    plt.subplot(3, 3, 2)
    cycles = np.arange(len(y_true))
    plt.plot(cycles, y_true, label='True SOH', linewidth=2, color='blue')
    plt.plot(cycles, y_pred, label='Predicted SOH', linewidth=2, color='red', linestyle='--')
    plt.xlabel('Cycle Number')
    plt.ylabel('SOH')
    plt.title(f'SOH Prediction Time Series - {test_battery}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. Residual Plot
    plt.subplot(3, 3, 3)
    residuals = y_true - y_pred
    plt.scatter(y_pred, residuals, alpha=0.6, s=50)
    plt.axhline(y=0, color='r', linestyle='--', lw=2)
    plt.xlabel('Predicted SOH')
    plt.ylabel('Residuals (Actual - Predicted)')
    plt.title('Residual Plot')
    plt.grid(True, alpha=0.3)
    
    # 4. Error Distribution
    plt.subplot(3, 3, 4)
    plt.hist(residuals, bins=30, alpha=0.7, density=True, color='skyblue', edgecolor='black')
    
    # Fit normal distribution
    mu, sigma = stats.norm.fit(residuals)
    x = np.linspace(residuals.min(), residuals.max(), 100)
    plt.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', lw=2, label=f'Normal fit (μ={mu:.4f}, σ={sigma:.4f})')
    
    plt.xlabel('Residuals')
    plt.ylabel('Density')
    plt.title('Error Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 5. Absolute Error over Time
    plt.subplot(3, 3, 5)
    abs_errors = np.abs(residuals)
    plt.plot(cycles, abs_errors, 'o-', markersize=4, linewidth=1, alpha=0.7)
    plt.axhline(y=metrics['MAE'], color='r', linestyle='--', lw=2, label=f'Mean AE: {metrics["MAE"]:.4f}')
    plt.xlabel('Cycle Number')
    plt.ylabel('Absolute Error')
    plt.title('Absolute Error Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 6. Training Loss (if available)
    if train_losses is not None:
        plt.subplot(3, 3, 6)
        plt.plot(train_losses, 'b-', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Training Loss')
        plt.title('Training Loss Curve')
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
    
    # 7. Box Plot of Errors
    plt.subplot(3, 3, 7)
    plt.boxplot([residuals], labels=['Residuals'])
    plt.ylabel('Error')
    plt.title('Error Distribution Summary')
    plt.grid(True, alpha=0.3)
    
    # 8. Q-Q Plot
    plt.subplot(3, 3, 8)
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title('Q-Q Plot (Normality Check)')
    plt.grid(True, alpha=0.3)
    
    # 9. Metrics Summary Table
    plt.subplot(3, 3, 9)
    plt.axis('tight')
    plt.axis('off')
    
    # Create metrics table
    metrics_data = [
        ['Metric', 'Value'],
        ['RMSE', f'{metrics["RMSE"]:.6f}'],
        ['MAE', f'{metrics["MAE"]:.6f}'],
        ['MAPE', f'{metrics["MAPE"]:.2f}%'],
        ['R²', f'{metrics["R2"]:.6f}'],
        ['Max Error', f'{metrics["Max_Error"]:.6f}'],
        ['Mean Error', f'{metrics["Mean_Error"]:.6f}'],
        ['Std Error', f'{metrics["Std_Error"]:.6f}'],
        ['Correlation', f'{metrics["Pearson_Correlation"]:.6f}']
    ]
    
    table = plt.table(cellText=metrics_data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2)
    
    # Style the header
    for i in range(len(metrics_data[0])):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    plt.title('Performance Metrics Summary', pad=20, fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    # Save the plot
    os.makedirs(save_dir, exist_ok=True)
    plot_path = os.path.join(save_dir, f'comprehensive_analysis_{test_battery}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Comprehensive analysis saved to: {plot_path}")
    
    plt.close()
    
    return metrics


def create_comparison_plots(results_dict, save_dir="results"):
    """
    Create comparison plots for multiple batteries or models.
    
    Args:
        results_dict (dict): Dictionary with battery names as keys and (y_true, y_pred) as values
        save_dir (str): Directory to save plots
    """
    
    if len(results_dict) < 2:
        print("Need at least 2 batteries for comparison plots")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Colors for different batteries
    colors = plt.cm.Set3(np.linspace(0, 1, len(results_dict)))
    
    all_metrics = {}
    
    # 1. Prediction vs Actual for all batteries
    ax1 = axes[0, 0]
    for i, (battery, (y_true, y_pred)) in enumerate(results_dict.items()):
        ax1.scatter(y_true, y_pred, alpha=0.6, label=battery, c=[colors[i]], s=50)
        all_metrics[battery] = calculate_comprehensive_metrics(y_true, y_pred)
    
    # Perfect prediction line
    all_true = np.concatenate([y_true for y_true, _ in results_dict.values()])
    all_pred = np.concatenate([y_pred for _, y_pred in results_dict.values()])
    min_val, max_val = min(all_true.min(), all_pred.min()), max(all_true.max(), all_pred.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    ax1.set_xlabel('Actual SOH')
    ax1.set_ylabel('Predicted SOH')
    ax1.set_title('Prediction vs Actual - All Batteries')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Time series comparison
    ax2 = axes[0, 1]
    for i, (battery, (y_true, y_pred)) in enumerate(results_dict.items()):
        cycles = np.arange(len(y_true))
        ax2.plot(cycles, y_true, color=colors[i], linewidth=2, label=f'{battery} - True')
        ax2.plot(cycles, y_pred, color=colors[i], linewidth=2, linestyle='--', label=f'{battery} - Pred')
    
    ax2.set_xlabel('Cycle Number')
    ax2.set_ylabel('SOH')
    ax2.set_title('SOH Time Series - All Batteries')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Metrics comparison
    ax3 = axes[1, 0]
    metrics_names = ['RMSE', 'MAE', 'MAPE', 'R2']
    x_pos = np.arange(len(metrics_names))
    
    for i, (battery, metrics) in enumerate(all_metrics.items()):
        values = [metrics[metric] for metric in metrics_names]
        ax3.bar(x_pos + i*0.25, values, 0.25, label=battery, alpha=0.7)
    
    ax3.set_xlabel('Metrics')
    ax3.set_ylabel('Value')
    ax3.set_title('Metrics Comparison')
    ax3.set_xticks(x_pos + 0.125)
    ax3.set_xticklabels(metrics_names)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Error distribution comparison
    ax4 = axes[1, 1]
    for i, (battery, (y_true, y_pred)) in enumerate(results_dict.items()):
        residuals = y_true - y_pred
        ax4.hist(residuals, bins=20, alpha=0.6, label=battery, color=colors[i], density=True)
    
    ax4.set_xlabel('Residuals')
    ax4.set_ylabel('Density')
    ax4.set_title('Error Distribution Comparison')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    os.makedirs(save_dir, exist_ok=True)
    plot_path = os.path.join(save_dir, 'battery_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Comparison plots saved to: {plot_path}")
    
    plt.close()
    
    return all_metrics


def generate_report(metrics, test_battery, save_dir="results"):
    """
    Generate a detailed text report of the evaluation.
    
    Args:
        metrics (dict): Dictionary of calculated metrics
        test_battery (str): Name of the test battery
        save_dir (str): Directory to save the report
    """
    
    report_content = f"""
# De-SaTE-Transformer Model Evaluation Report

## Test Battery: {test_battery}

### Performance Metrics

**Primary Metrics:**
- Root Mean Square Error (RMSE): {metrics['RMSE']:.6f}
- Mean Absolute Error (MAE): {metrics['MAE']:.6f}
- Mean Absolute Percentage Error (MAPE): {metrics['MAPE']:.2f}%
- R-squared (R²): {metrics['R2']:.6f}

**Additional Metrics:**
- Mean Squared Error (MSE): {metrics['MSE']:.6f}
- Maximum Error: {metrics['Max_Error']:.6f}
- Mean Error (Bias): {metrics['Mean_Error']:.6f}
- Standard Deviation of Errors: {metrics['Std_Error']:.6f}
- Pearson Correlation: {metrics['Pearson_Correlation']:.6f}

### Interpretation

**Model Accuracy:**
- The R² value of {metrics['R2']:.4f} indicates that the model explains {metrics['R2']*100:.1f}% of the variance in SOH.
- {'Good' if metrics['R2'] > 0.8 else 'Moderate' if metrics['R2'] > 0.6 else 'Poor'} correlation between predictions and actual values.

**Error Analysis:**
- Mean Absolute Error of {metrics['MAE']:.6f} means predictions are off by an average of {metrics['MAE']:.6f} units.
- {'Low' if metrics['MAPE'] < 5 else 'Moderate' if metrics['MAPE'] < 10 else 'High'} percentage error at {metrics['MAPE']:.1f}%.
- {'Unbiased' if abs(metrics['Mean_Error']) < 0.01 else 'Biased'} predictions (Mean Error: {metrics['Mean_Error']:.6f}).

**Recommendations:**
{'- Model shows good predictive performance' if metrics['R2'] > 0.8 and metrics['MAPE'] < 10 else '- Model may need improvement'}
{'- Consider feature engineering or model architecture changes' if metrics['R2'] < 0.7 else '- Model is suitable for deployment'}
{'- Check for systematic bias in predictions' if abs(metrics['Mean_Error']) > 0.01 else '- Predictions are well-calibrated'}

### Model Evaluation Guidelines

For battery SOH prediction models, consider these benchmarks:
- Excellent: R² > 0.9, MAPE < 3%
- Good: R² > 0.8, MAPE < 5%
- Acceptable: R² > 0.7, MAPE < 10%
- Poor: R² < 0.7, MAPE > 10%

Current model performance: {'Excellent' if metrics['R2'] > 0.9 and metrics['MAPE'] < 3 else 'Good' if metrics['R2'] > 0.8 and metrics['MAPE'] < 5 else 'Acceptable' if metrics['R2'] > 0.7 and metrics['MAPE'] < 10 else 'Poor'}
"""
    
    # Save report
    os.makedirs(save_dir, exist_ok=True)
    report_path = os.path.join(save_dir, f'evaluation_report_{test_battery}.md')
    with open(report_path, 'w') as f:
        f.write(report_content)
    
    print(f"Evaluation report saved to: {report_path}")
    
    return report_path


def main():
    parser = argparse.ArgumentParser(
        description='Create comprehensive visualizations for De-SaTE-Transformer results',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--results_dir', type=str, default='results',
                        help='Directory containing model results')
    parser.add_argument('--test_battery', type=str, required=True,
                        help='Name of the test battery')
    parser.add_argument('--predictions_file', type=str, default=None,
                        help='Path to predictions file (.npy)')
    parser.add_argument('--ground_truth_file', type=str, default=None,
                        help='Path to ground truth file (.npy)')
    parser.add_argument('--comparison_mode', action='store_true',
                        help='Enable comparison mode for multiple batteries')
    
    args = parser.parse_args()
    
    # Check if results directory exists
    if not os.path.exists(args.results_dir):
        print(f"Error: Results directory not found: {args.results_dir}")
        return
    
    # For demonstration, create synthetic data if no files specified
    if args.predictions_file is None or args.ground_truth_file is None:
        print("No prediction files specified. Creating demonstration with synthetic data...")
        
        # Generate synthetic data that mimics battery degradation
        np.random.seed(42)
        n_cycles = 200
        true_soh = np.linspace(1.0, 0.7, n_cycles) + np.random.normal(0, 0.01, n_cycles)
        pred_soh = true_soh + np.random.normal(0, 0.02, n_cycles)
        
        # Add some systematic errors to make it more realistic
        pred_soh += 0.01 * np.sin(np.linspace(0, 4*np.pi, n_cycles))
        
        # Ensure values are within realistic bounds
        true_soh = np.clip(true_soh, 0.6, 1.0)
        pred_soh = np.clip(pred_soh, 0.6, 1.0)
        
    else:
        # Load actual data
        true_soh = np.load(args.ground_truth_file)
        pred_soh = np.load(args.predictions_file)
    
    # Create comprehensive visualization
    print("Creating comprehensive visualization...")
    metrics = create_comprehensive_visualization(
        true_soh, pred_soh, 
        test_battery=args.test_battery,
        save_dir=args.results_dir
    )
    
    # Generate detailed report
    print("Generating evaluation report...")
    generate_report(metrics, args.test_battery, args.results_dir)
    
    # Print summary
    print("\n" + "="*60)
    print("VISUALIZATION SUMMARY")
    print("="*60)
    print(f"Test Battery: {args.test_battery}")
    print(f"RMSE: {metrics['RMSE']:.6f}")
    print(f"MAE: {metrics['MAE']:.6f}")
    print(f"MAPE: {metrics['MAPE']:.2f}%")
    print(f"R²: {metrics['R2']:.6f}")
    print(f"Results saved to: {args.results_dir}")
    print("="*60)


if __name__ == '__main__':
    main()