"""
Budget Optimizer CLI Interface
Command-line interface and lightweight wrapper for budget optimization
"""

import sys
import os
import argparse
import json
import pandas as pd
from typing import Dict, Any, Optional

# Add the project root to the path to import from models
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

try:
    from models.mmm.budget_optimizer import BudgetOptimizer as CoreBudgetOptimizer
    from models.mmm.econometric_mmm import EconometricMMM
except ImportError as e:
    print(f"Warning: Could not import core budget optimizer: {e}")
    CoreBudgetOptimizer = None
    EconometricMMM = None

def create_sample_data() -> pd.DataFrame:
    """Create sample marketing data for testing"""
    import numpy as np
    
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    
    data = pd.DataFrame({
        'date': dates,
        'revenue': np.random.normal(10000, 2000, len(dates)),
        'tv_spend': np.random.normal(1000, 200, len(dates)),
        'digital_spend': np.random.normal(800, 150, len(dates)),
        'print_spend': np.random.normal(500, 100, len(dates)),
        'radio_spend': np.random.normal(300, 75, len(dates))
    })
    
    return data

def run_optimization(data_file: Optional[str] = None, 
                    config_file: Optional[str] = None,
                    output_file: Optional[str] = None) -> Dict[str, Any]:
    """
    Run budget optimization
    
    Args:
        data_file: Path to marketing data CSV file
        config_file: Path to optimization configuration JSON file
        output_file: Path to save optimization results
        
    Returns:
        Optimization results
    """
    if CoreBudgetOptimizer is None:
        return {
            'error': 'Budget optimizer not available',
            'success': False
        }
    
    try:
        # Load data
        if data_file and os.path.exists(data_file):
            print(f"[OPTIMIZER] Loading data from {data_file}")
            data = pd.read_csv(data_file)
        else:
            print(f"[OPTIMIZER] Using sample data")
            data = create_sample_data()
        
        # Load configuration
        config = {}
        if config_file and os.path.exists(config_file):
            print(f"[OPTIMIZER] Loading config from {config_file}")
            with open(config_file, 'r') as f:
                config = json.load(f)
        
        # Set up MMM model
        mmm = EconometricMMM()
        
        # Fit MMM model
        spend_columns = [col for col in data.columns if col.endswith('_spend')]
        mmm_results = mmm.fit(data, spend_columns=spend_columns)
        
        # Set up optimizer
        optimizer_config = config.get('optimizer', {})
        optimizer = CoreBudgetOptimizer(
            mmm_model=mmm,
            **optimizer_config
        )
        
        # Run optimization
        current_allocation = {}
        total_budget = config.get('total_budget', 100000)
        
        for channel in spend_columns:
            current_allocation[channel.replace('_spend', '')] = data[channel].mean() * 7  # Weekly budget
        
        optimization_config = config.get('optimization', {})
        optimization_results = optimizer.optimize_budget_allocation(
            current_allocation=current_allocation,
            total_budget=total_budget,
            **optimization_config
        )
        
        # Save results
        if output_file:
            print(f"[OPTIMIZER] Saving results to {output_file}")
            with open(output_file, 'w') as f:
                json.dump(optimization_results, f, indent=2, default=str)
        
        return optimization_results
        
    except Exception as e:
        return {
            'error': str(e),
            'success': False
        }

def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(description='Media Mix Model Budget Optimizer')
    
    parser.add_argument('--data', '-d', type=str, 
                       help='Path to marketing data CSV file')
    parser.add_argument('--config', '-c', type=str,
                       help='Path to optimization configuration JSON file')
    parser.add_argument('--output', '-o', type=str,
                       help='Path to save optimization results JSON file')
    parser.add_argument('--sample', action='store_true',
                       help='Run with sample data')
    
    args = parser.parse_args()
    
    print("="*60)
    print("Media Mix Model Budget Optimizer")
    print("="*60)
    
    if args.sample or not args.data:
        print("[INFO] Running optimization with sample data")
        results = run_optimization(
            data_file=None,
            config_file=args.config,
            output_file=args.output
        )
    else:
        results = run_optimization(
            data_file=args.data,
            config_file=args.config,
            output_file=args.output
        )
    
    if results.get('success', False):
        print("\n[SUCCESS] Optimization completed successfully!")
        
        if 'optimal_allocation' in results:
            print("\nOptimal Budget Allocation:")
            for channel, budget in results['optimal_allocation'].items():
                print(f"  {channel}: ${budget:,.0f}")
        
        if 'performance_improvement' in results:
            improvement = results['performance_improvement']
            print(f"\nProjected Performance Improvement:")
            print(f"  ROI: {improvement.get('roi_improvement', 0):.1%}")
            print(f"  Revenue: ${improvement.get('revenue_improvement', 0):,.0f}")
    else:
        print(f"\n[ERROR] Optimization failed: {results.get('error', 'Unknown error')}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())