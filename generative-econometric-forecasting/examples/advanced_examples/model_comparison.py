#!/usr/bin/env python3
"""
Advanced Model Comparison Example

Demonstrates comparing different forecasting models (traditional vs AI-powered)
and evaluating their performance across multiple economic indicators.
"""

import sys
from pathlib import Path
from datetime import datetime
import json

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

def main():
    """
    Compare multiple forecasting models and evaluate performance
    """
    print("🔬 Advanced Model Comparison Example")
    print("=" * 50)
    
    try:
        # Define models to compare
        models = {
            "Traditional ARIMA": {
                "type": "statistical",
                "accuracy": 0.72,
                "speed": "fast",
                "interpretability": "high"
            },
            "Neural Network": {
                "type": "deep_learning",
                "accuracy": 0.85,
                "speed": "medium",
                "interpretability": "medium"
            },
            "Foundation Model (TimeGPT)": {
                "type": "foundation",
                "accuracy": 0.91,
                "speed": "slow",
                "interpretability": "low"
            },
            "Ensemble Hybrid": {
                "type": "ensemble",
                "accuracy": 0.88,
                "speed": "medium",
                "interpretability": "medium"
            }
        }
        
        # Economic indicators to forecast
        indicators = ["GDP", "CPI", "UNEMPLOYMENT", "INTEREST_RATES"]
        
        print("🎯 Comparing Models:")
        for model_name, specs in models.items():
            print(f"  📊 {model_name} ({specs['type']})")
            print(f"     Accuracy: {specs['accuracy']:.1%} | Speed: {specs['speed']} | Interpretability: {specs['interpretability']}")
        
        print(f"\n📈 Forecasting Indicators: {', '.join(indicators)}")
        
        # Simulate model performance comparison
        print("\n🔄 Running comparative analysis...")
        
        comparison_results = {}
        for indicator in indicators:
            comparison_results[indicator] = {}
            
            for model_name, specs in models.items():
                # Simulate performance metrics
                base_accuracy = specs["accuracy"]
                
                # Add some indicator-specific variation
                if indicator == "GDP" and specs["type"] == "foundation":
                    accuracy = min(0.95, base_accuracy + 0.03)  # Foundation models excel at GDP
                elif indicator == "UNEMPLOYMENT" and specs["type"] == "statistical":
                    accuracy = max(0.60, base_accuracy - 0.08)  # Traditional struggles with employment
                else:
                    accuracy = base_accuracy
                
                comparison_results[indicator][model_name] = {
                    "accuracy": accuracy,
                    "mae": round((1 - accuracy) * 2.5, 2),  # Mean Absolute Error
                    "confidence": round(accuracy * 0.9, 2)
                }
        
        # Display results
        print("\n📊 Comparative Performance Results:")
        print("-" * 80)
        
        for indicator in indicators:
            print(f"\n🎯 {indicator}:")
            
            # Sort models by accuracy for this indicator
            sorted_models = sorted(
                comparison_results[indicator].items(),
                key=lambda x: x[1]["accuracy"],
                reverse=True
            )
            
            for rank, (model_name, metrics) in enumerate(sorted_models, 1):
                accuracy = metrics["accuracy"]
                mae = metrics["mae"]
                confidence = metrics["confidence"]
                
                medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else "  "
                print(f"  {medal} {model_name:<25} Accuracy: {accuracy:.1%} | MAE: {mae:.2f} | Confidence: {confidence:.1%}")
        
        # Generate recommendations
        print("\n💡 Model Recommendations:")
        
        # Find best overall performer
        overall_scores = {}
        for model_name in models.keys():
            total_accuracy = sum(comparison_results[ind][model_name]["accuracy"] for ind in indicators)
            overall_scores[model_name] = total_accuracy / len(indicators)
        
        best_overall = max(overall_scores, key=overall_scores.get)
        print(f"  🏆 Best Overall: {best_overall} ({overall_scores[best_overall]:.1%} avg accuracy)")
        
        # Find best for speed
        fast_models = [name for name, specs in models.items() if specs["speed"] == "fast"]
        if fast_models:
            best_fast = max(fast_models, key=lambda m: overall_scores[m])
            print(f"  ⚡ Best for Speed: {best_fast} ({overall_scores[best_fast]:.1%} accuracy)")
        
        # Find most interpretable
        interpretable = [name for name, specs in models.items() if specs["interpretability"] == "high"]
        if interpretable:
            best_interpretable = max(interpretable, key=lambda m: overall_scores[m])
            print(f"  🔍 Most Interpretable: {best_interpretable} ({overall_scores[best_interpretable]:.1%} accuracy)")
        
        # Save detailed comparison report
        outputs_dir = project_root / "outputs" / "reports"
        outputs_dir.mkdir(parents=True, exist_ok=True)
        
        report_data = {
            "timestamp": datetime.now().isoformat(),
            "models_compared": models,
            "indicators": indicators,
            "performance_results": comparison_results,
            "overall_scores": overall_scores,
            "recommendations": {
                "best_overall": best_overall,
                "best_for_speed": best_fast if fast_models else None,
                "most_interpretable": best_interpretable if interpretable else None
            }
        }
        
        report_path = outputs_dir / f"model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        print(f"\n✅ Detailed comparison report saved to: {report_path}")
        print("\n🎉 Model comparison completed!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during model comparison: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)