#!/usr/bin/env python3
"""
Simple Attribution Analysis Example

Demonstrates basic media mix modeling attribution analysis
to understand which marketing channels drive the most conversions.
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
    Run simple attribution analysis across marketing channels
    """
    print("📊 Simple Attribution Analysis Example")
    print("=" * 45)
    
    try:
        # Simulate marketing channel data
        channel_data = {
            "TV": {
                "spend": 120000,
                "impressions": 2500000,
                "attributed_conversions": 1250,
                "cost_per_conversion": 96.0
            },
            "Digital Display": {
                "spend": 85000,
                "impressions": 5200000,
                "attributed_conversions": 1180,
                "cost_per_conversion": 72.0
            },
            "Social Media": {
                "spend": 65000,
                "impressions": 3800000,
                "attributed_conversions": 980,
                "cost_per_conversion": 66.3
            },
            "Search": {
                "spend": 95000,
                "impressions": 1800000,
                "attributed_conversions": 1520,
                "cost_per_conversion": 62.5
            },
            "Email": {
                "spend": 25000,
                "impressions": 450000,
                "attributed_conversions": 650,
                "cost_per_conversion": 38.5
            }
        }
        
        print("🎯 Marketing Channel Performance:")
        print("-" * 70)
        
        total_spend = sum(data["spend"] for data in channel_data.values())
        total_conversions = sum(data["attributed_conversions"] for data in channel_data.values())
        
        # Calculate attribution metrics
        attribution_results = {}
        
        for channel, data in channel_data.items():
            spend = data["spend"]
            conversions = data["attributed_conversions"]
            cost_per_conv = data["cost_per_conversion"]
            
            # Calculate key metrics
            spend_share = spend / total_spend
            conversion_share = conversions / total_conversions
            efficiency_score = conversion_share / spend_share  # >1 means efficient
            
            attribution_results[channel] = {
                "spend_share": spend_share,
                "conversion_share": conversion_share,
                "efficiency_score": efficiency_score,
                "cost_per_conversion": cost_per_conv,
                "attributed_conversions": conversions
            }
            
            # Display results
            efficiency_emoji = "🟢" if efficiency_score > 1.2 else "🟡" if efficiency_score > 0.8 else "🔴"
            
            print(f"{channel:<15} | Spend: ${spend:,} ({spend_share:.1%}) | "
                  f"Conversions: {conversions:,} ({conversion_share:.1%}) | "
                  f"Efficiency: {efficiency_score:.2f} {efficiency_emoji}")
        
        print("-" * 70)
        print(f"{'TOTAL':<15} | Spend: ${total_spend:,} (100.0%) | "
              f"Conversions: {total_conversions:,} (100.0%) |")
        
        # Generate insights and recommendations
        print("\n💡 Attribution Insights:")
        
        # Sort channels by efficiency
        sorted_by_efficiency = sorted(
            attribution_results.items(),
            key=lambda x: x[1]["efficiency_score"],
            reverse=True
        )
        
        # Top performer
        top_channel, top_metrics = sorted_by_efficiency[0]
        print(f"  🏆 Most Efficient Channel: {top_channel} "
              f"(efficiency score: {top_metrics['efficiency_score']:.2f})")
        
        # Underperformer
        bottom_channel, bottom_metrics = sorted_by_efficiency[-1]
        if bottom_metrics['efficiency_score'] < 0.8:
            print(f"  ⚠️  Underperforming Channel: {bottom_channel} "
                  f"(efficiency score: {bottom_metrics['efficiency_score']:.2f})")
        
        # Cost efficiency insights
        best_cost_channel = min(channel_data.items(), key=lambda x: x[1]["cost_per_conversion"])
        print(f"  💰 Lowest Cost Per Conversion: {best_cost_channel[0]} "
              f"(${best_cost_channel[1]['cost_per_conversion']:.2f})")
        
        # Volume insights
        highest_volume = max(attribution_results.items(), key=lambda x: x[1]["attributed_conversions"])
        print(f"  📈 Highest Volume Channel: {highest_volume[0]} "
              f"({highest_volume[1]['attributed_conversions']:,} conversions)")
        
        # Budget reallocation recommendations
        print("\n🔄 Budget Reallocation Recommendations:")
        
        recommendations = []
        
        for channel, metrics in attribution_results.items():
            if metrics["efficiency_score"] > 1.3:  # High efficiency
                recommendations.append(f"📈 Increase {channel} budget (high efficiency: {metrics['efficiency_score']:.2f})")
            elif metrics["efficiency_score"] < 0.7:  # Low efficiency
                recommendations.append(f"📉 Consider reducing {channel} budget (low efficiency: {metrics['efficiency_score']:.2f})")
        
        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                print(f"  {i}. {rec}")
        else:
            print("  ✅ All channels performing within acceptable efficiency ranges")
        
        # Calculate potential budget reallocation
        print("\n📊 Optimal Budget Allocation Simulation:")
        
        # Simple reallocation: move budget from least to most efficient
        if len(sorted_by_efficiency) >= 2:
            least_efficient = sorted_by_efficiency[-1]
            most_efficient = sorted_by_efficiency[0]
            
            reallocation_amount = min(25000, channel_data[least_efficient[0]]["spend"] * 0.2)  # Max 20% or $25k
            
            print(f"  💸 Move ${reallocation_amount:,} from {least_efficient[0]} to {most_efficient[0]}")
            
            # Calculate expected impact
            expected_lost_conversions = reallocation_amount / channel_data[least_efficient[0]]["cost_per_conversion"]
            expected_gained_conversions = reallocation_amount / channel_data[most_efficient[0]]["cost_per_conversion"]
            net_gain = expected_gained_conversions - expected_lost_conversions
            
            print(f"  📈 Expected net conversion gain: {net_gain:+.0f} conversions")
            print(f"  💰 Estimated additional revenue: ${net_gain * 150:+,.0f} (assuming $150 LTV)")
        
        # Save attribution report
        outputs_dir = project_root / "outputs"
        outputs_dir.mkdir(parents=True, exist_ok=True)
        
        report_data = {
            "analysis_type": "simple_attribution",
            "timestamp": datetime.now().isoformat(),
            "channel_performance": channel_data,
            "attribution_metrics": attribution_results,
            "total_spend": total_spend,
            "total_conversions": total_conversions,
            "recommendations": recommendations,
            "top_performer": {
                "channel": top_channel,
                "efficiency_score": top_metrics["efficiency_score"]
            }
        }
        
        report_path = outputs_dir / f"simple_attribution_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        print(f"\n✅ Attribution report saved to: {report_path}")
        print("\n🎉 Simple attribution analysis completed!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during attribution analysis: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)