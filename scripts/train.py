#!/usr/bin/env python3
"""
De-SaTE-Transformer Training Script

This script trains a De-SaTE-Transformer model for battery State of Health (SOH) prediction.
It can be run from the command line with various configuration options.

Usage:
    python scripts/train.py --data_path /path/to/data --epochs 100 --batch_size 128
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import numpy as np
import time
import matplotlib.pyplot as plt
from pathlib import Path

# Add the src directory to the path so we can import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from model.transformer import TransformerModel
from utils.data_loader import get_data_loaders
from utils.metrics import get_rmse, get_mae, get_rul_error, calculate_all_metrics
from utils.evaluation import evaluate_model_comprehensive


def setup_seed(seed):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def train_one_epoch(model, optimizer, criterion, train_loader, device, window_size):
    """Train the model for one epoch."""
    model.train()
    total_loss = 0.0
    start_time = time.time()

    for batch, (data, targets) in enumerate(train_loader):
        data, targets = data.to(device), targets.to(device)
        
        # Transformer model expects shape [sequence_length, batch_size, features]
        data = data.permute(1, 0, 2)

        optimizer.zero_grad()
        output = model(data)
        
        # We only care about the last prediction of the sequence
        loss = criterion(output[-1, :, :], targets)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
    
    return total_loss / len(train_loader)


def evaluate(model, criterion, test_loader, device, window_size):
    """Evaluate the model on test data."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data, targets in test_loader:
            data, targets = data.to(device), targets.to(device)
            data = data.permute(1, 0, 2)
            
            output = model(data)
            
            total_loss += criterion(output[-1, :, :], targets).item()
            
            all_preds.append(output[-1, :, :].cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            
    predictions = np.concatenate(all_preds).flatten()
    targets = np.concatenate(all_targets).flatten()
    
    return total_loss / len(test_loader), predictions, targets


def create_enhanced_model(feature_size, num_layers, dropout, nhead):
    """Create a transformer model with proper input/output projections."""
    class EnhancedTransformerModel(nn.Module):
        def __init__(self, feature_size, num_layers, dropout, nhead):
            super().__init__()
            self.input_projection = nn.Linear(1, feature_size)
            self.transformer = TransformerModel(feature_size, num_layers, dropout, nhead)
            self.output_projection = nn.Linear(feature_size, 1)

        def forward(self, src):
            # src shape: [sequence_length, batch_size, 1]
            projected_src = self.input_projection(src)
            output = self.transformer.transformer_encoder(projected_src, self.transformer.src_mask)
            output = self.output_projection(output)
            return output
    
    return EnhancedTransformerModel(feature_size, num_layers, dropout, nhead)


def main():
    parser = argparse.ArgumentParser(
        description='Train De-SaTE-Transformer for Battery SOH Prediction',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data parameters
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to the processed dataset (.npy file)')
    parser.add_argument('--test_battery', type=str, required=True,
                        help='Name of the battery to use for testing')
    parser.add_argument('--samples', type=int, default=None,
                        help='Number of samples to use (None for all)')
    
    # Model parameters
    parser.add_argument('--feature_size', type=int, default=128,
                        help='Feature size (d_model) for the transformer')
    parser.add_argument('--nhead', type=int, default=8,
                        help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=3,
                        help='Number of transformer encoder layers')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout probability')
    parser.add_argument('--window_size', type=int, default=16,
                        help='Sequence window size')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Directory to save results')
    parser.add_argument('--save_model', action='store_true',
                        help='Save the trained model')
    
    args = parser.parse_args()
    
    # Setup
    setup_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    print("Loading data...")
    train_loader, test_loader, test_sequence = get_data_loaders(
        args.data_path,
        test_battery_name=args.test_battery,
        window_size=args.window_size,
        batch_size=args.batch_size
    )
    
    if train_loader is None:
        print("Failed to load data. Exiting.")
        return
    
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Create model
    print("Creating model...")
    model = create_enhanced_model(
        args.feature_size, args.num_layers, args.dropout, args.nhead
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Setup training
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
    
    # Training loop
    print("Starting training...")
    best_loss = float('inf')
    train_losses = []
    
    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
        train_loss = train_one_epoch(model, optimizer, criterion, train_loader, device, args.window_size)
        train_losses.append(train_loss)
        
        # Print progress
        if epoch % 10 == 0 or epoch == 1:
            elapsed = time.time() - epoch_start_time
            print(f'Epoch {epoch:3d}/{args.epochs} | Time: {elapsed:5.2f}s | Train Loss: {train_loss:.6f}')
        
        scheduler.step()
        
        # Save best model
        if args.save_model and train_loss < best_loss:
            best_loss = train_loss
            model_path = os.path.join(args.output_dir, f'best_model_{args.test_battery}.pth')
            torch.save(model.state_dict(), model_path)
    
    print("Training completed!")
    
    # Evaluation
    print("Evaluating model...")
    test_loss, predictions, targets = evaluate(model, criterion, test_loader, device, args.window_size)
    
    # Calculate comprehensive metrics including E value
    metrics = calculate_all_metrics(targets, predictions)
    
    # Calculate RUL error (assuming typical battery capacity values)
    # This is a simplified calculation - in practice, you'd need dataset-specific thresholds
    rated_capacity = np.max(test_sequence)
    failure_threshold = rated_capacity * 0.7
    
    # Reconstruct full prediction sequence
    split_point = args.window_size + 1
    full_pred_sequence = np.concatenate([
        np.array(test_sequence[:split_point]), 
        predictions
    ])
    full_true_sequence = test_sequence[:len(full_pred_sequence)]
    
    try:
        rul_err = get_rul_error(full_true_sequence, full_pred_sequence, failure_threshold)
    except:
        rul_err = "N/A"
    
    # Print results with ALL E value metrics including weighted ones
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Test Loss: {test_loss:.6f}")
    print(f"RMSE: {metrics['RMSE']:.6f}")
    print(f"MAE: {metrics['MAE']:.6f}")
    print(f"RE: {metrics['RE']:.6f}")
    print(f"E Value (Original): {metrics['E_value']:.2f}%")
    print(f"E Score (Original): {metrics['E_score']:.2f}/15")
    
    # Show weighted metrics if available
    if 'E_value_weighted' in metrics:
        print(f"E Value (Weighted): {metrics['E_value_weighted']:.2f}%")
        print(f"E Score (Weighted): {metrics['E_score_weighted']:.2f}/15")
    
    # Show practical range metrics if available
    if 'E_value_practical' in metrics:
        print(f"E Value (Practical Range): {metrics['E_value_practical']:.2f}%")
        print(f"E Score (Practical Range): {metrics['E_score_practical']:.2f}/15")
        practical_info = metrics.get('practical_range_info', {})
        if 'coverage' in practical_info:
            print(f"Practical Range Coverage: {practical_info['coverage']:.1f}%")
    
    print(f"RUL Error: {rul_err}")
    print("="*50)
    
    # Create visualization
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Training loss
    plt.subplot(2, 2, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    
    # Plot 2: Prediction vs actual
    plt.subplot(2, 2, 2)
    plt.scatter(targets, predictions, alpha=0.6)
    plt.plot([targets.min(), targets.max()], [targets.min(), targets.max()], 'r--', lw=2)
    plt.xlabel('Actual SOH')
    plt.ylabel('Predicted SOH')
    plt.title('Prediction vs Actual')
    plt.grid(True)
    
    # Plot 3: Time series comparison
    plt.subplot(2, 1, 2)
    plt.plot(full_true_sequence, label='True Capacity', linewidth=2)
    plt.plot(full_pred_sequence, label='Predicted Capacity', linestyle='--', linewidth=2)
    plt.axvline(x=split_point, color='r', linestyle=':', alpha=0.7, label='Train/Test Split')
    plt.axhline(y=failure_threshold, color='g', linestyle='-.', alpha=0.7, 
                label=f'Failure Threshold ({failure_threshold:.3f})')
    plt.title(f'SOH Prediction for {args.test_battery}')
    plt.xlabel('Cycle Number')
    plt.ylabel('Capacity')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(args.output_dir, f'training_results_{args.test_battery}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Results plot saved to: {plot_path}")
    plt.close()
    
    # Save predictions and ground truth for enhanced visualization
    pred_path = os.path.join(args.output_dir, f'predictions_{args.test_battery}.npy')
    truth_path = os.path.join(args.output_dir, f'ground_truth_{args.test_battery}.npy')
    train_loss_path = os.path.join(args.output_dir, f'train_losses_{args.test_battery}.npy')
    
    np.save(pred_path, predictions)
    np.save(truth_path, targets)
    np.save(train_loss_path, train_losses)
    print(f"Predictions saved to: {pred_path}")
    print(f"Ground truth saved to: {truth_path}")
    print(f"Training losses saved to: {train_loss_path}")
    
    # Generate comprehensive evaluation with E values
    print("\nGenerating comprehensive evaluation with E values...")
    try:
        battery_name = args.test_battery.replace('test_', '')  # Remove test_ prefix if present
        comprehensive_results = evaluate_model_comprehensive(
            targets, predictions,
            model_name="De-SaTE-Transformer",
            battery_id=battery_name,
            output_dir=args.output_dir
        )
        print(f"Comprehensive evaluation completed!")
        comp_metrics = comprehensive_results['metrics']
        print(f"E Value (Original): {comp_metrics['E_value']:.2f}%")
        print(f"E Score (Original): {comp_metrics['E_score']:.2f}/15")
        
        if 'E_value_weighted' in comp_metrics:
            print(f"E Value (Weighted): {comp_metrics['E_value_weighted']:.2f}%")
            print(f"E Score (Weighted): {comp_metrics['E_score_weighted']:.2f}/15")
        
        if 'E_value_practical' in comp_metrics:
            print(f"E Value (Practical Range): {comp_metrics['E_value_practical']:.2f}%")
            print(f"E Score (Practical Range): {comp_metrics['E_score_practical']:.2f}/15")
        
        if 'plots' in comprehensive_results:
            print("Generated E value visualizations:")
            for plot_type, plot_path in comprehensive_results['plots'].items():
                print(f"  - {plot_type}: {plot_path}")
                
    except Exception as e:
        print(f"Warning: Comprehensive evaluation failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Save final model
    if args.save_model:
        final_model_path = os.path.join(args.output_dir, f'final_model_{args.test_battery}.pth')
        torch.save(model.state_dict(), final_model_path)
        print(f"Final model saved to: {final_model_path}")
    
    # Generate enhanced visualization
    print("\nGenerating enhanced visualization...")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    viz_script = os.path.join(script_dir, 'visualize_results.py')
    
    if os.path.exists(viz_script):
        import subprocess
        viz_cmd = [
            'python', viz_script,
            '--results_dir', args.output_dir,
            '--test_battery', args.test_battery,
            '--predictions_file', pred_path,
            '--ground_truth_file', truth_path
        ]
        try:
            subprocess.run(viz_cmd, check=True)
            print("Enhanced visualization completed successfully!")
        except subprocess.CalledProcessError as e:
            print(f"Warning: Enhanced visualization failed: {e}")
    else:
        print("Enhanced visualization script not found. Skipping.")


if __name__ == '__main__':
    main()