#!/usr/bin/env python3
"""
Cross-Channel Synergy Analysis Example

Demonstrates advanced MMM analysis to identify synergistic effects
between marketing channels and optimize multi-channel campaigns.
"""

import sys
from pathlib import Path
from datetime import datetime
import json
import itertools

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

def main():
    """
    Analyze cross-channel synergies and interaction effects
    """
    print("🔗 Cross-Channel Synergy Analysis Example")
    print("=" * 50)
    
    try:
        # Base channel performance (individual effects)
        base_performance = {
            "TV": {"spend": 100000, "base_conversions": 800, "base_roi": 3.2},
            "Digital": {"spend": 80000, "base_conversions": 720, "base_roi": 3.6},
            "Social": {"spend": 60000, "base_conversions": 540, "base_roi": 3.0},
            "Search": {"spend": 70000, "base_conversions": 770, "base_roi": 4.4},
            "Email": {"spend": 30000, "base_conversions": 450, "base_roi": 6.0}
        }
        
        print("📊 Base Channel Performance (Individual Effects):")
        print("-" * 65)
        
        total_base_spend = sum(data["spend"] for data in base_performance.values())
        total_base_conversions = sum(data["base_conversions"] for data in base_performance.values())
        
        for channel, data in base_performance.items():
            spend = data["spend"]
            conversions = data["base_conversions"]
            roi = data["base_roi"]
            
            print(f"{channel:<8} | Spend: ${spend:,} | "
                  f"Conversions: {conversions:,} | ROI: {roi:.1f}x")
        
        print("-" * 65)
        print(f"{'TOTAL':<8} | Spend: ${total_base_spend:,} | "
              f"Conversions: {total_base_conversions:,} | "
              f"Avg ROI: {(total_base_conversions * 150) / total_base_spend:.1f}x")
        
        # Define synergy effects between channel pairs
        synergy_matrix = {
            ("TV", "Digital"): {"multiplier": 1.15, "strength": "Medium"},
            ("TV", "Social"): {"multiplier": 1.08, "strength": "Low"},
            ("TV", "Search"): {"multiplier": 1.25, "strength": "High"},
            ("Digital", "Social"): {"multiplier": 1.12, "strength": "Medium"},
            ("Digital", "Search"): {"multiplier": 1.18, "strength": "Medium"},
            ("Social", "Email"): {"multiplier": 1.22, "strength": "High"},
            ("Search", "Email"): {"multiplier": 1.10, "strength": "Low"}
        }
        
        print("\n🔗 Identified Channel Synergies:")
        print("-" * 50)
        
        for (channel1, channel2), synergy in synergy_matrix.items():
            multiplier = synergy["multiplier"]
            strength = synergy["strength"]
            lift = (multiplier - 1) * 100
            
            strength_emoji = "🔥" if strength == "High" else "⚡" if strength == "Medium" else "💫"
            
            print(f"{channel1} + {channel2:<8} | "
                  f"Synergy: {multiplier:.2f}x ({lift:+.0f}%) {strength_emoji} {strength}")
        
        # Calculate synergy-adjusted performance
        print("\n📈 Synergy-Adjusted Performance Analysis:")
        
        # For each channel, calculate its synergy benefit from other active channels
        synergy_adjusted_performance = {}
        
        for channel, base_data in base_performance.items():
            base_conversions = base_data["base_conversions"]
            total_synergy_lift = 0
            synergy_sources = []
            
            # Check for synergies with all other channels
            for other_channel in base_performance.keys():
                if other_channel != channel:
                    # Check if there's a synergy between these channels
                    synergy_key1 = (channel, other_channel)
                    synergy_key2 = (other_channel, channel)
                    
                    if synergy_key1 in synergy_matrix:
                        multiplier = synergy_matrix[synergy_key1]["multiplier"]
                        synergy_lift = (multiplier - 1) * base_conversions
                        total_synergy_lift += synergy_lift
                        synergy_sources.append({
                            "partner": other_channel,
                            "lift": synergy_lift,
                            "multiplier": multiplier
                        })
                    elif synergy_key2 in synergy_matrix:
                        multiplier = synergy_matrix[synergy_key2]["multiplier"]
                        synergy_lift = (multiplier - 1) * base_conversions
                        total_synergy_lift += synergy_lift
                        synergy_sources.append({
                            "partner": other_channel,
                            "lift": synergy_lift,
                            "multiplier": multiplier
                        })
            
            adjusted_conversions = base_conversions + total_synergy_lift
            adjusted_roi = (adjusted_conversions * 150) / base_data["spend"]  # Assuming $150 LTV
            
            synergy_adjusted_performance[channel] = {
                "base_conversions": base_conversions,
                "synergy_lift": total_synergy_lift,
                "adjusted_conversions": adjusted_conversions,
                "adjusted_roi": adjusted_roi,
                "synergy_sources": synergy_sources
            }
        
        # Display adjusted performance
        print("\nChannel Performance with Synergy Effects:")
        print("-" * 80)
        
        total_adjusted_conversions = 0
        for channel, data in synergy_adjusted_performance.items():
            base_conv = data["base_conversions"]
            synergy_lift = data["synergy_lift"]
            adjusted_conv = data["adjusted_conversions"]
            adjusted_roi = data["adjusted_roi"]
            
            total_adjusted_conversions += adjusted_conv
            
            lift_pct = (synergy_lift / base_conv * 100) if base_conv > 0 else 0
            
            print(f"{channel:<8} | Base: {base_conv:,} | "
                  f"Synergy: +{synergy_lift:,.0f} ({lift_pct:+.0f}%) | "
                  f"Total: {adjusted_conv:,.0f} | ROI: {adjusted_roi:.1f}x")
        
        total_synergy_lift = total_adjusted_conversions - total_base_conversions
        synergy_value = total_synergy_lift * 150  # $150 LTV
        
        print("-" * 80)
        print(f"{'TOTAL':<8} | Base: {total_base_conversions:,} | "
              f"Synergy: +{total_synergy_lift:,.0f} | "
              f"Total: {total_adjusted_conversions:,.0f}")
        print(f"\n💰 Total Synergy Value: ${synergy_value:,.0f} additional revenue")
        
        # Identify strongest synergy combinations
        print("\n🔥 Top Synergy Opportunities:")
        
        # Calculate potential lift from individual synergies
        synergy_opportunities = []
        for (ch1, ch2), synergy_data in synergy_matrix.items():
            multiplier = synergy_data["multiplier"]
            strength = synergy_data["strength"]
            
            # Calculate potential additional conversions from this synergy
            ch1_conversions = base_performance[ch1]["base_conversions"]
            ch2_conversions = base_performance[ch2]["base_conversions"]
            
            # Synergy affects both channels
            potential_lift = ((multiplier - 1) * (ch1_conversions + ch2_conversions))
            potential_value = potential_lift * 150
            
            synergy_opportunities.append({
                "channels": f"{ch1} + {ch2}",
                "multiplier": multiplier,
                "strength": strength,
                "potential_lift": potential_lift,
                "potential_value": potential_value
            })
        
        # Sort by potential value
        synergy_opportunities.sort(key=lambda x: x["potential_value"], reverse=True)
        
        for i, opportunity in enumerate(synergy_opportunities[:3], 1):  # Top 3
            channels = opportunity["channels"]
            multiplier = opportunity["multiplier"]
            strength = opportunity["strength"]
            value = opportunity["potential_value"]
            
            strength_emoji = "🔥" if strength == "High" else "⚡" if strength == "Medium" else "💫"
            
            print(f"  {i}. {channels} | "
                  f"Multiplier: {multiplier:.2f}x | "
                  f"Value: ${value:,.0f} {strength_emoji}")
        
        # Campaign optimization recommendations
        print("\n🎯 Multi-Channel Campaign Recommendations:")
        
        recommendations = []
        
        # Find the highest-value synergy pair
        top_synergy = synergy_opportunities[0]
        recommendations.append(f"Prioritize {top_synergy['channels']} combination for maximum synergy value")
        
        # Identify channels that participate in multiple strong synergies
        channel_synergy_count = {}
        for (ch1, ch2), synergy_data in synergy_matrix.items():
            if synergy_data["strength"] in ["High", "Medium"]:
                channel_synergy_count[ch1] = channel_synergy_count.get(ch1, 0) + 1
                channel_synergy_count[ch2] = channel_synergy_count.get(ch2, 0) + 1
        
        hub_channel = max(channel_synergy_count, key=channel_synergy_count.get)
        recommendations.append(f"Use {hub_channel} as campaign anchor (participates in {channel_synergy_count[hub_channel]} strong synergies)")
        
        # Timing recommendations
        recommendations.append("Launch complementary channels within 1-2 weeks for optimal synergy capture")
        recommendations.append("Monitor cross-channel attribution to validate synergy assumptions")
        
        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec}")
        
        # Save synergy analysis report
        outputs_dir = project_root / "outputs"
        outputs_dir.mkdir(parents=True, exist_ok=True)
        
        synergy_report = {
            "analysis_type": "cross_channel_synergy",
            "timestamp": datetime.now().isoformat(),
            "base_performance": base_performance,
            "synergy_matrix": {f"{k[0]}+{k[1]}": v for k, v in synergy_matrix.items()},
            "synergy_adjusted_performance": synergy_adjusted_performance,
            "total_synergy_value": synergy_value,
            "top_opportunities": synergy_opportunities[:5],
            "recommendations": recommendations
        }
        
        report_path = outputs_dir / f"synergy_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w') as f:
            json.dump(synergy_report, f, indent=2)
        
        print(f"\n✅ Synergy analysis report saved to: {report_path}")
        print("\n🎉 Cross-channel synergy analysis completed!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during synergy analysis: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)