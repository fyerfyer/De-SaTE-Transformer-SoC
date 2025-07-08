#!/usr/bin/env python3
"""
Data Preprocessing Script for De-SaTE-Transformer

This script preprocesses battery degradation datasets for training.
It supports multiple datasets and provides flexible configuration options.

Usage:
    python scripts/preprocess.py --dataset NASA --raw_data_dir /path/to/raw/data
    python scripts/preprocess.py --all --raw_data_dir /path/to/raw/data
"""

import os
import sys
import argparse
from pathlib import Path

# Add the src directory to the path so we can import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import processing modules
try:
    from data_processing.official_processor import OfficialProcessor
    DATA_PROCESSOR_AVAILABLE = True
except ImportError:
    print("Warning: Official data processor not found. Creating placeholder...")
    DATA_PROCESSOR_AVAILABLE = False
    
    class OfficialProcessor:
        def __init__(self, raw_data_path, output_path):
            self.raw_data_path = raw_data_path
            self.output_path = output_path
            
        def process(self):
            print(f"Processing data from {self.raw_data_path} to {self.output_path}")
            print("Error: OfficialProcessor class not found. Please ensure the src/data_processing/official_processor.py file exists.")
            
        def save(self):
            print("Saving processed data...")
            print("Error: OfficialProcessor class not found. Please ensure the src/data_processing/official_processor.py file exists.")


def process_official_dataset(raw_data_dir, output_dir, **kwargs):
    """
    Process the official SOH dataset.
    
    Args:
        raw_data_dir (str): Path to raw data directory
        output_dir (str): Path to output directory
        **kwargs: Additional processing parameters
    """
    print("Processing Official SOH Dataset...")
    
    # Look for the official data structure
    official_data_path = os.path.join(raw_data_dir, 'SOH-dataset')
    
    if not os.path.exists(official_data_path):
        print(f"Warning: Official data not found at {official_data_path}")
        print("Please ensure the data is structured as: raw_data_dir/SOH-dataset/")
        return False
    
    try:
        # Create processor instance with correct parameters
        processor = OfficialProcessor(raw_data_dir, output_dir)
        
        # Process the data
        processor.process()
        
        # Save the processed data
        processor.save()
        
        print("Official dataset processing completed successfully!")
        return True
    except Exception as e:
        print(f"Error processing official dataset: {e}")
        import traceback
        traceback.print_exc()
        return False


def process_nasa_dataset(raw_data_dir, output_dir, **kwargs):
    """
    Process NASA battery dataset.
    
    Args:
        raw_data_dir (str): Path to raw data directory
        output_dir (str): Path to output directory
        **kwargs: Additional processing parameters
    """
    print("Processing NASA Dataset...")
    
    nasa_data_path = os.path.join(raw_data_dir, 'NASA')
    
    if not os.path.exists(nasa_data_path):
        print(f"Warning: NASA data not found at {nasa_data_path}")
        return False
    
    # NASA processing is not implemented yet
    print(f"NASA data found at: {nasa_data_path}")
    print("Note: NASA processing logic needs to be implemented")
    print("      A reference implementation is available in previous-implementation/src/data_processing/nasa_processor.py")
    return True


def process_calce_dataset(raw_data_dir, output_dir, **kwargs):
    """
    Process CALCE battery dataset.
    
    Args:
        raw_data_dir (str): Path to raw data directory
        output_dir (str): Path to output directory
        **kwargs: Additional processing parameters
    """
    print("Processing CALCE Dataset...")
    
    calce_data_path = os.path.join(raw_data_dir, 'CALCE')
    
    if not os.path.exists(calce_data_path):
        print(f"Warning: CALCE data not found at {calce_data_path}")
        return False
    
    # CALCE processing is not implemented yet
    print(f"CALCE data found at: {calce_data_path}")
    print("Note: CALCE processing logic needs to be implemented")
    print("      A reference implementation is available in previous-implementation/src/data_processing/calce_processor.py")
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Preprocess battery degradation datasets for De-SaTE-Transformer',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data selection
    parser.add_argument('--dataset', type=str, 
                        choices=['official', 'nasa', 'calce', 'oxford'],
                        help='Specific dataset to process')
    parser.add_argument('--all', action='store_true',
                        help='Process all available datasets')
    
    # Input/Output paths
    parser.add_argument('--raw_data_dir', type=str, 
                        default='dataset',
                        help='Directory containing raw datasets')
    parser.add_argument('--output_dir', type=str, 
                        default='data/processed_data',
                        help='Directory to save processed data')
    
    # Processing parameters
    parser.add_argument('--samples', type=int, default=None,
                        help='Maximum number of samples to process (None for all)')
    parser.add_argument('--normalize', action='store_true',
                        help='Normalize the capacity values')
    parser.add_argument('--smooth', action='store_true',
                        help='Apply smoothing to capacity curves')
    parser.add_argument('--min_cycles', type=int, default=50,
                        help='Minimum number of cycles required for a battery')
    
    # Advanced options
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose output')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite existing processed files')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.dataset and not args.all:
        print("Error: Please specify --dataset [NAME] or use --all to process all datasets")
        parser.print_help()
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory: {os.path.abspath(args.output_dir)}")
    print(f"Raw data directory: {os.path.abspath(args.raw_data_dir)}")
    
    # Check if raw data directory exists
    if not os.path.exists(args.raw_data_dir):
        print(f"Error: Raw data directory not found: {args.raw_data_dir}")
        return
    
    # Processing parameters
    processing_kwargs = {
        'samples': args.samples,
        'normalize': args.normalize,
        'smooth': args.smooth,
        'min_cycles': args.min_cycles,
        'verbose': args.verbose,
        'overwrite': args.overwrite
    }
    
    # Process datasets
    success_count = 0
    total_count = 0
    
    datasets_to_process = []
    if args.all:
        datasets_to_process = ['official', 'nasa', 'calce']
    else:
        datasets_to_process = [args.dataset]
    
    print("\n" + "="*60)
    print("STARTING DATA PREPROCESSING")
    print("="*60)
    
    for dataset in datasets_to_process:
        print(f"\n{'-'*40}")
        print(f"Processing {dataset.upper()} Dataset")
        print(f"{'-'*40}")
        
        total_count += 1
        
        if dataset == 'official':
            success = process_official_dataset(args.raw_data_dir, args.output_dir, **processing_kwargs)
        elif dataset == 'nasa':
            success = process_nasa_dataset(args.raw_data_dir, args.output_dir, **processing_kwargs)
        elif dataset == 'calce':
            success = process_calce_dataset(args.raw_data_dir, args.output_dir, **processing_kwargs)
        else:
            print(f"Error: Unknown dataset '{dataset}'")
            success = False
        
        if success:
            success_count += 1
    
    # Summary
    print("\n" + "="*60)
    print("PREPROCESSING SUMMARY")
    print("="*60)
    print(f"Successfully processed: {success_count}/{total_count} datasets")
    
    if success_count > 0:
        print(f"\nProcessed data saved to: {args.output_dir}")
        print("\nTo train a model, use:")
        print(f"python scripts/train.py --data_path {args.output_dir}/[dataset]_processed.npy --test_battery [battery_name]")
    
    print("\nPreprocessing completed!")


if __name__ == '__main__':
    main()