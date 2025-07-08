#!/usr/bin/env python3
"""
Re-evaluate existing model results with E value metrics.

This script loads existing predictions and ground truth data,
calculates the new E value metrics, and generates updated
evaluation reports and visualizations.
"""

import sys
import os
import numpy as np
import argparse
import logging

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.evaluation import evaluate_model_comprehensive
from utils.metrics import calculate_all_metrics, get_e_value, calculate_e_score

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def load_results(results_dir: str, battery_id: str):
    """
    Load existing prediction results.
    
    Args:
        results_dir: Directory containing results
        battery_id: Battery identifier
    
    Returns:
        Tuple of (ground_truth, predictions) arrays
    """
    try:
        # Try to load ground truth
        gt_path = os.path.join(results_dir, f'ground_truth_test_{battery_id}.npy')
        if os.path.exists(gt_path):
            ground_truth = np.load(gt_path)
        else:
            raise FileNotFoundError(f"Ground truth file not found: {gt_path}")
        
        # Try to load predictions
        pred_path = os.path.join(results_dir, f'predictions_test_{battery_id}.npy')
        if os.path.exists(pred_path):
            predictions = np.load(pred_path)
        else:
            raise FileNotFoundError(f"Predictions file not found: {pred_path}")
        
        logging.info(f"Loaded results for {battery_id}:")
        logging.info(f"  Ground truth shape: {ground_truth.shape}")
        logging.info(f"  Predictions shape: {predictions.shape}")
        
        return ground_truth, predictions
        
    except Exception as e:
        logging.error(f"Error loading results: {e}")
        raise

def generate_enhanced_report(ground_truth: np.ndarray, predictions: np.ndarray, 
                           battery_id: str, output_dir: str):
    """
    Generate an enhanced evaluation report with E value metrics.
    
    Args:
        ground_truth: Ground truth SOH values
        predictions: Predicted SOH values
        battery_id: Battery identifier
        output_dir: Output directory for the report
    """
    # Calculate all metrics including E value
    metrics = calculate_all_metrics(ground_truth, predictions)
    
    # Generate report content
    report_content = f"""
# Enhanced De-SaTE-Transformer Model Evaluation Report

## Test Battery: {battery_id}

### Performance Metrics

**Primary Metrics:**
- Root Mean Square Error (RMSE): {metrics['RMSE']:.6f}
- Mean Absolute Error (MAE): {metrics['MAE']:.6f}
- Relative Error (RE): {metrics['RE']:.6f}

**Project-Specific Metrics:**
- E Value (Original): {metrics['E_value']:.2f}%
- E Score (Original): {metrics['E_score']:.2f}/15
- E Value (Weighted): {metrics.get('E_value_weighted', 0):.2f}%
- E Score (Weighted): {metrics.get('E_score_weighted', 0):.2f}/15
- E Value (Practical Range): {metrics.get('E_value_practical', 0):.2f}%
- E Score (Practical Range): {metrics.get('E_score_practical', 0):.2f}/15

### Detailed Analysis

#### E Value Interpretation
The E value represents the mean percentage error for capacity prediction:

**Original E Value ({metrics['E_value']:.2f}%):**
- **Formula**: E = mean(|predicted_capacity - true_capacity| / true_capacity) × 100%
- **Scoring**: {"(10 - E) × 1.5 = " + f"{metrics['E_score']:.2f}" if metrics['E_value'] <= 10 else "0 (E > 10%)"} points

**Weighted E Value ({metrics.get('E_value_weighted', 0):.2f}%):**
- **Formula**: Uses SOH range-based weighting to reduce low-SOH impact
- **Weighting**: High SOH (0.6-0.8) × 3.0, Mid SOH (0.3-0.6) × 2.0, Low SOH (<0.3) × 0.5
- **Scoring**: {"(10 - E) × 1.5 = " + f"{metrics.get('E_score_weighted', 0):.2f}" if metrics.get('E_value_weighted', 100) <= 10 else "0 (E > 10%)"} points

**Practical Range E Value ({metrics.get('E_value_practical', 0):.2f}%):**
- **Formula**: E calculated only for SOH range 0.6-0.8 (most operationally relevant)
- **Coverage**: {metrics.get('practical_range_info', {}).get('coverage', 0):.1f}% of data points
- **Scoring**: {"(10 - E) × 1.5 = " + f"{metrics.get('E_score_practical', 0):.2f}" if metrics.get('E_value_practical', 100) <= 10 else "0 (E > 10%)"} points

#### Performance Classification
Based on E value ({metrics['E_value']:.2f}%):
- **Excellent** (E ≤ 3%): {"✓" if metrics['E_value'] <= 3 else "✗"}
- **Good** (3% < E ≤ 5%): {"✓" if 3 < metrics['E_value'] <= 5 else "✗"}
- **Acceptable** (5% < E ≤ 10%): {"✓" if 5 < metrics['E_value'] <= 10 else "✗"}
- **Poor** (E > 10%): {"✓" if metrics['E_value'] > 10 else "✗"}

**Current Performance Level: {"Excellent" if metrics['E_value'] <= 3 else "Good" if metrics['E_value'] <= 5 else "Acceptable" if metrics['E_value'] <= 10 else "Poor"}**

### Statistical Summary

**Data Characteristics:**
- Number of data points: {len(ground_truth)}
- Ground truth range: {ground_truth.min():.4f} - {ground_truth.max():.4f}
- Predictions range: {predictions.min():.4f} - {predictions.max():.4f}
- Mean ground truth: {ground_truth.mean():.4f}
- Mean predictions: {predictions.mean():.4f}

**Error Statistics:**
- Maximum absolute error: {np.max(np.abs(ground_truth - predictions)):.6f}
- Minimum absolute error: {np.min(np.abs(ground_truth - predictions)):.6f}
- Standard deviation of errors: {np.std(ground_truth - predictions):.6f}
- Mean bias (predictions - actual): {np.mean(predictions - ground_truth):.6f}

### Model Assessment

#### Strengths
{"- Excellent capacity prediction accuracy (E ≤ 3%)" if metrics['E_value'] <= 3 else ""}
{"- Good capacity prediction accuracy (3% < E ≤ 5%)" if 3 < metrics['E_value'] <= 5 else ""}
{"- Acceptable capacity prediction accuracy (5% < E ≤ 10%)" if 5 < metrics['E_value'] <= 10 else ""}
- Low RMSE value: {metrics['RMSE']:.6f}
- Consistent prediction performance

#### Areas for Improvement
{"- Model needs significant improvement (E > 10%)" if metrics['E_value'] > 10 else ""}
{"- Could improve to reach excellent level (target E ≤ 3%)" if 3 < metrics['E_value'] <= 10 else ""}
- Consider ensemble methods for better accuracy
- Explore feature engineering for SOH estimation

### Recommendations

#### Immediate Actions
1. **Data Quality**: Review data processing pipeline for SOH calculation
2. **Model Architecture**: {"Consider model refinement" if metrics['E_value'] > 5 else "Current architecture performs well"}
3. **Training Strategy**: {"Implement advanced training techniques" if metrics['E_value'] > 10 else "Continue current approach"}

#### Future Improvements
1. **Cross-validation**: Test on multiple battery types
2. **Ensemble Methods**: Combine multiple models for better performance
3. **Feature Engineering**: Extract more meaningful SOH indicators
4. **Transfer Learning**: Pre-train on larger datasets

### Conclusion

The De-SaTE-Transformer model achieved an E value of {metrics['E_value']:.2f}%, earning {metrics['E_score']:.2f} out of 15 possible points in the project evaluation criteria. 

{"The model demonstrates excellent performance and is ready for deployment." if metrics['E_value'] <= 3 else 
 "The model shows good performance with room for improvement." if metrics['E_value'] <= 5 else
 "The model has acceptable performance but requires optimization." if metrics['E_value'] <= 10 else
 "The model requires significant improvement before deployment."}

---
*Report generated with enhanced E value metrics*
"""
    
    # Save the enhanced report
    report_path = os.path.join(output_dir, f'enhanced_evaluation_report_{battery_id}.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    logging.info(f"Enhanced evaluation report saved to: {report_path}")
    return report_path

def main():
    parser = argparse.ArgumentParser(description="Re-evaluate model results with E value metrics")
    parser.add_argument('--results_dir', type=str, default='results',
                       help='Directory containing existing results')
    parser.add_argument('--battery_id', type=str, default='JBGSRS250006644',
                       help='Battery ID for evaluation')
    parser.add_argument('--output_dir', type=str, default='results',
                       help='Output directory for enhanced reports')
    
    args = parser.parse_args()
    
    setup_logging()
    
    try:
        # Load existing results
        logging.info(f"Loading results for battery {args.battery_id}")
        ground_truth, predictions = load_results(args.results_dir, args.battery_id)
        
        # Calculate E value metrics
        logging.info("Calculating E value metrics...")
        e_value = get_e_value(ground_truth, predictions)
        e_score = calculate_e_score(e_value)
        
        logging.info(f"E Value: {e_value:.2f}%")
        logging.info(f"E Score: {e_score:.2f}/15")
        
        # Generate enhanced evaluation report
        logging.info("Generating enhanced evaluation report...")
        report_path = generate_enhanced_report(ground_truth, predictions, 
                                             args.battery_id, args.output_dir)
        
        # Generate comprehensive evaluation with visualizations
        logging.info("Generating comprehensive evaluation with E value analysis...")
        results = evaluate_model_comprehensive(
            ground_truth, predictions,
            model_name="De-SaTE-Transformer",
            battery_id=args.battery_id,
            output_dir=args.output_dir
        )
        
        # Summary
        logging.info("="*50)
        logging.info("EVALUATION SUMMARY")
        logging.info("="*50)
        logging.info(f"Battery: {args.battery_id}")
        logging.info(f"E Value: {e_value:.2f}%")
        logging.info(f"E Score: {e_score:.2f}/15")
        logging.info(f"Performance: {'Excellent' if e_value <= 3 else 'Good' if e_value <= 5 else 'Acceptable' if e_value <= 10 else 'Poor'}")
        logging.info(f"Enhanced Report: {report_path}")
        
        if 'plots' in results:
            logging.info("Generated Visualizations:")
            for plot_type, plot_path in results['plots'].items():
                logging.info(f"  - {plot_type}: {plot_path}")
        
        logging.info("Enhanced evaluation completed successfully!")
        
    except Exception as e:
        logging.error(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()