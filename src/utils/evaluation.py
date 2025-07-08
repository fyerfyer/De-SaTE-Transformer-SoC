"""
Comprehensive evaluation module for De-SaTE-Transformer model.

This module provides evaluation metrics calculation and visualization
following the reference implementation style.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import os
import logging
from typing import Dict, List, Tuple, Optional
from .metrics import calculate_all_metrics, get_relative_error, get_rmse, get_mae

logger = logging.getLogger(__name__)

class ComplexRadar:
    """
    Radar chart implementation based on the reference code.
    """
    
    def __init__(self, fig, variables, ranges, n_ordinate_levels=6):
        angles = np.arange(0, 360, 360./len(variables))

        axes = [fig.add_axes([0.1,0.1,0.9,0.9], polar=True,
                label = "axes{}".format(i))
                for i in range(len(variables))]
        l, text = axes[0].set_thetagrids(angles,
                                         labels=variables, fontsize=22)

        [txt.set_rotation(angle-90) for txt, angle
             in zip(text, angles)]
        [txt.set_y(txt.get_position()[1] -0.1) for txt, angle
             in zip(text, angles)]
        for ax in axes[1:]:
            ax.patch.set_visible(False)
            ax.grid("off")
            ax.xaxis.set_visible(False)
        for i, ax in enumerate(axes):
            grid = np.linspace(*ranges[i],
                               num=n_ordinate_levels)
            gridlabel = ["{}".format(round(x,3))
                         for x in grid]
            if ranges[i][0] > ranges[i][1]:
                grid = grid[::-1]  # hack to invert grid
            gridlabel[0] = ""  # clean up origin
            self._set_rgrids(ax, grid, labels=gridlabel, angle=angles[i])
            ax.set_ylim(*ranges[i])
        
        # variables for plotting
        self.angle = np.deg2rad(np.r_[angles, angles[0]])
        self.ranges = ranges
        self.ax = axes[0]
        self.ax.xaxis.grid(True, color='k', linestyle='--', linewidth=2)
    
    def _set_rgrids(self, ax, radii, labels=None, angle=None, fmt=None, **kwargs):
        """Set the radial locations and labels of the r grids."""
        radii = np.asarray(radii)
        ax.set_yticks(radii)
        if labels is not None:
            ax.set_yticklabels(labels)
        elif fmt is not None:
            ax.yaxis.set_major_formatter(plt.FormatStrFormatter(fmt))
        if angle is None:
            angle = ax.get_rlabel_position()
        ax.set_rlabel_position(angle)
        for t in ax.yaxis.get_ticklabels():
            t.update(kwargs)
        return ax.yaxis.get_gridlines(), ax.yaxis.get_ticklabels()
    
    def _scale_data(self, data, ranges):
        """scales data to ranges, inverts if the scale is reversed"""
        x1, x2 = ranges[0]
        d = data[0]

        if x1 > x2:
            d = self._invert(d, (x1, x2))
            x1, x2 = x2, x1

        sdata = [d]

        for d, (y1, y2) in zip(data[1:], ranges[1:]):
            if y1 > y2:
                d = self._invert(d, (y1, y2))
                y1, y2 = y2, y1

            sdata.append((d-y1) / (y2-y1) * (x2 - x1) + x1)

        return sdata
    
    def _invert(self, x, limits):
        """inverts a value x on a scale from limits[0] to limits[1]"""
        return limits[1] - (x - limits[0])
    
    def plot(self, data, *args, **kw):
        sdata = self._scale_data(data, self.ranges)
        self.ax.plot(self.angle, np.r_[sdata, sdata[0]], *args, **kw)
    
    def fill(self, data, *args, **kw):
        sdata = self._scale_data(data, self.ranges)
        self.ax.fill(self.angle, np.r_[sdata, sdata[0]], *args, **kw)


class ModelEvaluator:
    """
    Comprehensive model evaluation class with visualization capabilities.
    """
    
    def __init__(self, output_dir: str = "results"):
        self.output_dir = output_dir
        self.results = {}
        os.makedirs(output_dir, exist_ok=True)
    
    def evaluate_predictions(self, y_true: np.ndarray, y_pred: np.ndarray, 
                           model_name: str = "model") -> Dict[str, float]:
        """
        Evaluate predictions and calculate standard metrics.
        
        Args:
            y_true: Ground truth values
            y_pred: Predicted values
            model_name: Name identifier for the model
        
        Returns:
            Dictionary containing evaluation metrics
        """
        # Calculate metrics
        metrics = calculate_all_metrics(y_true, y_pred)
        
        # Store results
        self.results[model_name] = {
            'y_true': y_true,
            'y_pred': y_pred,
            'metrics': metrics
        }
        
        logger.info(f"Evaluation metrics for {model_name}:")
        logger.info(f"  RE: {metrics['RE']:.6f}")
        logger.info(f"  RMSE: {metrics['RMSE']:.6f}")
        logger.info(f"  MAE: {metrics['MAE']:.6f}")
        logger.info(f"  E_value: {metrics['E_value']:.2f}%")
        logger.info(f"  E_score: {metrics['E_score']:.2f}/15")
        
        return metrics
    
    def create_radar_chart(self, metrics_dict: Dict[str, Dict[str, float]], 
                          save_path: Optional[str] = None) -> str:
        """
        Create radar chart visualization following the reference implementation.
        
        Args:
            metrics_dict: Dictionary with model_name -> metrics mapping
            save_path: Optional custom save path
        
        Returns:
            Path to saved plot
        """
        if not metrics_dict:
            raise ValueError("No metrics provided for radar chart")
        
        # Define ranges based on the reference implementation
        metrics_names = ["RE", "RMSE", "MAE"]
        ranges = [(0.025, 0.06), (0.07, 0.16), (0.007, 0.027)]
        
        # Create figure
        plt.style.use('seaborn-v0_8')
        fig = plt.figure(figsize=(10, 10))
        radar = ComplexRadar(fig, metrics_names, ranges)
        
        # Colors for different models
        colors = ['r', 'b', 'g', 'orange', 'purple']
        patches = []
        
        for i, (model_name, metrics) in enumerate(metrics_dict.items()):
            color = colors[i % len(colors)]
            data = [metrics['RE'], metrics['RMSE'], metrics['MAE']]
            
            radar.fill(data, '-', color=color, alpha=0.3)
            patches.append(mpatches.Patch(color=color, alpha=0.5, label=model_name))
        
        plt.legend(handles=patches, loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        # Save plot
        if save_path is None:
            save_path = os.path.join(self.output_dir, 'evaluation_radar_chart.png')
        
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=300)
        plt.close()
        
        logger.info(f"Radar chart saved to: {save_path}")
        return save_path
    
    def create_comparison_plots(self, save_path: Optional[str] = None) -> str:
        """
        Create comparison plots similar to the reference implementation.
        
        Args:
            save_path: Optional custom save path
        
        Returns:
            Path to saved plot
        """
        if not self.results:
            raise ValueError("No evaluation results available for plotting")
        
        metrics_names = ['RE', 'RMSE', 'MAE']
        n_models = len(self.results)
        
        plt.figure(figsize=(16, 12))
        
        # Create subplot for each metric
        for i, metric in enumerate(metrics_names):
            plt.subplot(2, 2, i + 1)
            plt.title(f'Mean {metric} Values Comparison', fontsize=14)
            
            model_names = list(self.results.keys())
            values = [self.results[model]['metrics'][metric] for model in model_names]
            
            bars = plt.bar(model_names, values, alpha=0.7, color=f'C{i}')
            plt.ylabel(f'Mean {metric}')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                        f'{value:.4f}', ha='center', va='bottom', fontsize=10)
        
        # Create prediction vs actual plot
        plt.subplot(2, 2, 4)
        plt.title('Prediction vs Actual Comparison', fontsize=14)
        
        for model_name, result in self.results.items():
            y_true = result['y_true']
            y_pred = result['y_pred']
            plt.scatter(y_true, y_pred, alpha=0.6, label=model_name, s=30)
        
        # Perfect prediction line
        all_true = np.concatenate([result['y_true'] for result in self.results.values()])
        all_pred = np.concatenate([result['y_pred'] for result in self.results.values()])
        min_val, max_val = min(all_true.min(), all_pred.min()), max(all_true.max(), all_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        
        plt.xlabel('Actual SOH')
        plt.ylabel('Predicted SOH')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        if save_path is None:
            save_path = os.path.join(self.output_dir, 'evaluation_comparison.png')
        
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        logger.info(f"Comparison plots saved to: {save_path}")
        return save_path
    
    def create_prediction_time_series(self, y_true: np.ndarray, y_pred: np.ndarray,
                                    battery_id: str = "battery", 
                                    save_path: Optional[str] = None) -> str:
        """
        Create time series plot comparing predicted vs actual SOH values.
        
        Args:
            y_true: Ground truth SOH values
            y_pred: Predicted SOH values
            battery_id: Battery identifier for the plot title
            save_path: Optional custom save path
        
        Returns:
            Path to saved plot
        """
        plt.figure(figsize=(14, 8))
        
        cycles = np.arange(len(y_true))
        
        plt.plot(cycles, y_true, label='True SOH', linewidth=2, color='blue', marker='o', markersize=4)
        plt.plot(cycles, y_pred, label='Predicted SOH', linewidth=2, color='red', 
                linestyle='--', marker='s', markersize=4)
        
        plt.xlabel('Cycle Number', fontsize=12)
        plt.ylabel('State of Health (SOH)', fontsize=12)
        plt.title(f'SOH Prediction vs Actual - {battery_id}', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Add metrics text box
        metrics = calculate_all_metrics(y_true, y_pred)
        textstr = f'RE: {metrics["RE"]:.4f}\nRMSE: {metrics["RMSE"]:.4f}\nMAE: {metrics["MAE"]:.4f}\nE Value: {metrics["E_value"]:.2f}%\nE Score: {metrics["E_score"]:.2f}/15'
        
        # Add weighted metrics if available
        if 'E_value_weighted' in metrics:
            textstr += f'\nWeighted E: {metrics["E_value_weighted"]:.2f}%\nWeighted Score: {metrics["E_score_weighted"]:.2f}/15'
        
        # Add practical range metrics if available
        if 'E_value_practical' in metrics:
            textstr += f'\nPractical E: {metrics["E_value_practical"]:.2f}%\nPractical Score: {metrics["E_score_practical"]:.2f}/15'
        
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
        
        # Save plot
        if save_path is None:
            save_path = os.path.join(self.output_dir, f'soh_prediction_{battery_id}.png')
        
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        logger.info(f"Time series plot saved to: {save_path}")
        return save_path
    
    def create_e_value_analysis(self, y_true: np.ndarray, y_pred: np.ndarray,
                              battery_id: str = "battery", 
                              save_path: Optional[str] = None) -> str:
        """
        Create comprehensive E value analysis plot.
        
        Args:
            y_true: Ground truth SOH values
            y_pred: Predicted SOH values
            battery_id: Battery identifier for the plot title
            save_path: Optional custom save path
        
        Returns:
            Path to saved plot
        """
        from .metrics import get_e_value, calculate_e_score
        
        # Calculate E value and score
        e_value = get_e_value(y_true, y_pred)
        e_score = calculate_e_score(e_value)
        
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. E Value Progress Chart
        cycles = np.arange(len(y_true))
        cumulative_e_values = []
        
        for i in range(3, len(y_true) + 1):  # Start from 3 to avoid edge cases
            partial_e = get_e_value(y_true[:i], y_pred[:i])
            cumulative_e_values.append(partial_e)
        
        ax1.plot(cycles[2:], cumulative_e_values, 'b-', linewidth=2, label='Cumulative E Value')
        ax1.axhline(y=10, color='r', linestyle='--', linewidth=2, label='E = 10% Threshold')
        ax1.fill_between(cycles[2:], 0, cumulative_e_values, alpha=0.3, color='blue')
        ax1.set_xlabel('Cycle Number')
        ax1.set_ylabel('E Value (%)')
        ax1.set_title('E Value Progress Over Cycles')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Error Distribution
        relative_errors = np.abs((y_true - y_pred) / y_true) * 100
        ax2.hist(relative_errors, bins=20, alpha=0.7, color='green', edgecolor='black')
        ax2.axvline(x=np.mean(relative_errors), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: {np.mean(relative_errors):.2f}%')
        ax2.set_xlabel('Relative Error (%)')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Distribution of Relative Errors')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. E Score Visualization
        e_values_range = np.linspace(0, 20, 100)
        e_scores_range = [calculate_e_score(e) for e in e_values_range]
        
        ax3.plot(e_values_range, e_scores_range, 'purple', linewidth=3, label='E Score Function')
        ax3.scatter([e_value], [e_score], color='red', s=100, zorder=5, 
                   label=f'Current: E={e_value:.2f}%, Score={e_score:.2f}')
        ax3.axvline(x=10, color='orange', linestyle='--', alpha=0.7, label='E = 10% Threshold')
        ax3.set_xlabel('E Value (%)')
        ax3.set_ylabel('E Score (out of 15)')
        ax3.set_title('E Score vs E Value Function')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Performance Dashboard
        ax4.axis('off')
        
        # Create performance text
        performance_text = f"""
        E Value Analysis for {battery_id}
        
        E Value: {e_value:.2f}%
        E Score: {e_score:.2f}/15
        
        Performance Level:
        {'✓ Excellent' if e_value <= 3 else '✗ Excellent'} (E ≤ 3%)
        {'✓ Good' if 3 < e_value <= 5 else '✗ Good'} (3% < E ≤ 5%)
        {'✓ Acceptable' if 5 < e_value <= 10 else '✗ Acceptable'} (5% < E ≤ 10%)
        {'✓ Poor' if e_value > 10 else '✗ Poor'} (E > 10%)
        
        Scoring Criteria:
        • E ≤ 10%: Score = (10 - E) × 1.5
        • E > 10%: Score = 0
        
        Max Possible Score: 15 points
        """
        
        ax4.text(0.1, 0.9, performance_text, transform=ax4.transAxes, fontsize=12,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        
        # Save plot
        if save_path is None:
            save_path = os.path.join(self.output_dir, f'e_value_analysis_{battery_id}.png')
        
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        logger.info(f"E value analysis plot saved to: {save_path}")
        return save_path
    
    def save_results_csv(self, save_path: Optional[str] = None) -> str:
        """
        Save evaluation results to CSV file following the reference format.
        
        Args:
            save_path: Optional custom save path
        
        Returns:
            Path to saved CSV file
        """
        if not self.results:
            raise ValueError("No evaluation results available to save")
        
        # Prepare data for CSV
        data = []
        for model_name, result in self.results.items():
            metrics = result['metrics']
            data.append({
                'Model': model_name,
                'RE': metrics['RE'],
                'RMSE': metrics['RMSE'],
                'MAE': metrics['MAE'],
                'E_value': metrics['E_value'],
                'E_score': metrics['E_score']
            })
        
        df = pd.DataFrame(data)
        
        # Save CSV
        if save_path is None:
            save_path = os.path.join(self.output_dir, 'evaluation_results.csv')
        
        df.to_csv(save_path, index=False)
        logger.info(f"Results saved to CSV: {save_path}")
        
        return save_path
    
    def sanity_check(self, y_true: np.ndarray, y_pred: np.ndarray, 
                    model_name: str = "model") -> Dict[str, bool]:
        """
        Perform sanity checks on the evaluation data.
        
        Args:
            y_true: Ground truth values
            y_pred: Predicted values
            model_name: Model name for logging
        
        Returns:
            Dictionary of check results
        """
        checks = {}
        
        # Check for empty or all-zero data
        checks['not_empty'] = len(y_true) > 0 and len(y_pred) > 0
        checks['not_all_zeros_true'] = not np.all(y_true == 0)
        checks['not_all_zeros_pred'] = not np.all(y_pred == 0)
        
        # Check for reasonable SOH values (typically between 0.5 and 1.0)
        checks['reasonable_true_range'] = np.all((y_true >= 0.3) & (y_true <= 1.2))
        checks['reasonable_pred_range'] = np.all((y_pred >= 0.3) & (y_pred <= 1.2))
        
        # Check for reasonable metrics
        if checks['not_empty'] and checks['not_all_zeros_true'] and checks['not_all_zeros_pred']:
            metrics = calculate_all_metrics(y_true, y_pred)
            checks['reasonable_re'] = metrics['RE'] < 1.0  # RE should be less than 100%
            checks['reasonable_rmse'] = metrics['RMSE'] < 0.5  # RMSE should be reasonable
            checks['reasonable_mae'] = metrics['MAE'] < 0.5  # MAE should be reasonable
        else:
            checks['reasonable_re'] = False
            checks['reasonable_rmse'] = False
            checks['reasonable_mae'] = False
        
        # Log results
        failed_checks = [check for check, passed in checks.items() if not passed]
        if failed_checks:
            logger.warning(f"Sanity check failures for {model_name}: {failed_checks}")
        else:
            logger.info(f"All sanity checks passed for {model_name}")
        
        return checks


def evaluate_model_comprehensive(y_true: np.ndarray, y_pred: np.ndarray,
                               model_name: str = "De-SaTE-Transformer",
                               battery_id: str = "test_battery",
                               output_dir: str = "results") -> Dict:
    """
    Comprehensive model evaluation with all visualizations.
    
    Args:
        y_true: Ground truth SOH values
        y_pred: Predicted SOH values
        model_name: Name of the model being evaluated
        battery_id: Battery identifier
        output_dir: Output directory for results
    
    Returns:
        Dictionary containing all evaluation results and file paths
    """
    evaluator = ModelEvaluator(output_dir)
    
    # Perform sanity checks
    sanity_results = evaluator.sanity_check(y_true, y_pred, model_name)
    
    # Calculate metrics
    metrics = evaluator.evaluate_predictions(y_true, y_pred, model_name)
    
    # Create visualizations
    results = {
        'metrics': metrics,
        'sanity_checks': sanity_results,
        'plots': {}
    }
    
    try:
        # Time series plot
        ts_path = evaluator.create_prediction_time_series(y_true, y_pred, battery_id)
        results['plots']['time_series'] = ts_path
        
        # E value analysis plot
        e_analysis_path = evaluator.create_e_value_analysis(y_true, y_pred, battery_id)
        results['plots']['e_value_analysis'] = e_analysis_path
        
        # Radar chart
        radar_path = evaluator.create_radar_chart({model_name: metrics})
        results['plots']['radar_chart'] = radar_path
        
        # Comparison plots
        comp_path = evaluator.create_comparison_plots()
        results['plots']['comparison'] = comp_path
        
        # Save CSV results
        csv_path = evaluator.save_results_csv()
        results['csv_results'] = csv_path
        
    except Exception as e:
        logger.error(f"Error creating visualizations: {e}")
        results['error'] = str(e)
    
    return results