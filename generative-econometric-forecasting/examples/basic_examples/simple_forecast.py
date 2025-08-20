#!/usr/bin/env python3
"""
Simple Economic Forecasting Example

Demonstrates basic usage of the generative econometric forecasting platform
for creating simple economic forecasts.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.forecasting import *
from src.reports.simple_reporting import SimpleReporter

def main():
    """
    Run a simple economic forecast using default models
    """
    print("🔮 Simple Economic Forecasting Example")
    print("=" * 50)
    
    try:
        # Initialize reporter
        reporter = SimpleReporter()
        
        # Create simple forecast configuration
        forecast_config = {
            "indicators": ["GDP", "CPI", "UNEMPLOYMENT"],
            "horizon": 12,  # 12 months
            "model_type": "ensemble",
            "confidence_level": 0.95
        }
        
        print("📊 Running forecast with configuration:")
        for key, value in forecast_config.items():
            print(f"  {key}: {value}")
        
        # Generate forecast (simulated for demo)
        print("\n🔄 Generating forecasts...")
        
        # Simulate forecast results
        forecast_results = {
            "GDP": {"forecast": [2.1, 2.3, 2.0], "confidence": 0.85},
            "CPI": {"forecast": [3.2, 3.1, 2.9], "confidence": 0.78},
            "UNEMPLOYMENT": {"forecast": [5.1, 4.9, 4.7], "confidence": 0.82}
        }
        
        print("\n📈 Forecast Results:")
        for indicator, results in forecast_results.items():
            forecast_vals = results["forecast"]
            confidence = results["confidence"]
            print(f"  {indicator}: {forecast_vals[0]:.1f}% → {forecast_vals[-1]:.1f}% (confidence: {confidence:.1%})")
        
        # Generate report
        print("\n📋 Generating report...")
        
        # Create outputs directory if it doesn't exist
        outputs_dir = project_root / "outputs" / "reports"
        outputs_dir.mkdir(parents=True, exist_ok=True)
        
        report_path = outputs_dir / "simple_forecast_example.txt"
        
        with open(report_path, 'w') as f:
            f.write("Simple Economic Forecast Report\n")
            f.write("=" * 40 + "\n\n")
            
            f.write("Configuration:\n")
            for key, value in forecast_config.items():
                f.write(f"  {key}: {value}\n")
            
            f.write("\nForecast Results:\n")
            for indicator, results in forecast_results.items():
                forecast_vals = results["forecast"]
                confidence = results["confidence"]
                f.write(f"  {indicator}: {forecast_vals[0]:.1f}% → {forecast_vals[-1]:.1f}% (confidence: {confidence:.1%})\n")
        
        print(f"✅ Report saved to: {report_path}")
        print("\n🎉 Simple forecast completed successfully!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during forecasting: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)