#!/usr/bin/env python3
"""
Budget Optimization Example

Demonstrates how to use MMM insights to optimize marketing budget allocation
across different channels for maximum ROI.
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
    Run budget optimization analysis
    """
    print("💰 Budget Optimization Example")
    print("=" * 40)
    
    try:
        # Current budget and performance
        current_allocation = {
            "TV": {"budget": 120000, "roi": 2.8, "saturation": 0.75},
            "Digital Display": {"budget": 85000, "roi": 3.2, "saturation": 0.60},
            "Social Media": {"budget": 65000, "roi": 3.8, "saturation": 0.45},
            "Search": {"budget": 95000, "roi": 4.2, "saturation": 0.80},
            "Email": {"budget": 25000, "roi": 6.5, "saturation": 0.30},
            "Radio": {"budget": 40000, "roi": 2.1, "saturation": 0.85}
        }
        
        total_budget = sum(channel["budget"] for channel in current_allocation.values())
        
        print(f"📊 Current Budget: ${total_budget:,}")
        print("\n🎯 Current Channel Performance:")
        print("-" * 60)
        
        current_total_roi = 0
        for channel, data in current_allocation.items():
            budget = data["budget"]
            roi = data["roi"]
            saturation = data["saturation"]
            
            budget_pct = budget / total_budget
            revenue = budget * roi
            current_total_roi += revenue
            
            saturation_emoji = "🔴" if saturation > 0.8 else "🟡" if saturation > 0.6 else "🟢"
            
            print(f"{channel:<15} | Budget: ${budget:,} ({budget_pct:.1%}) | "
                  f"ROI: {roi:.1f}x | Saturation: {saturation:.1%} {saturation_emoji}")
        
        print("-" * 60)
        print(f"{'TOTAL':<15} | Budget: ${total_budget:,} | "
              f"Revenue: ${current_total_roi:,.0f} | Overall ROI: {current_total_roi/total_budget:.1f}x")
        
        # Optimization scenarios
        print("\n🔧 Budget Optimization Scenarios:")
        
        # Scenario 1: Redistribute based on ROI and saturation
        print("\n1️⃣ Scenario 1: ROI-Optimized Allocation")
        
        # Calculate efficiency score (ROI / saturation level)
        efficiency_scores = {}
        for channel, data in current_allocation.items():
            # Higher efficiency = high ROI + low saturation
            efficiency_scores[channel] = data["roi"] * (1 - data["saturation"])
        
        # Redistribute budget based on efficiency
        total_efficiency = sum(efficiency_scores.values())
        optimized_allocation_1 = {}
        
        for channel, efficiency in efficiency_scores.items():
            optimal_share = efficiency / total_efficiency
            optimized_allocation_1[channel] = {
                "budget": int(total_budget * optimal_share),
                "roi": current_allocation[channel]["roi"],
                "saturation": current_allocation[channel]["saturation"]
            }
        
        # Calculate projected performance
        optimized_roi_1 = 0
        print("Optimized Allocation:")
        for channel, data in optimized_allocation_1.items():
            budget = data["budget"]
            roi = data["roi"]
            
            # Adjust ROI based on new budget (diminishing returns)
            current_budget = current_allocation[channel]["budget"]
            budget_change_ratio = budget / current_budget if current_budget > 0 else 1
            
            # Simple diminishing returns model
            if budget_change_ratio > 1.5:  # Big increase
                adjusted_roi = roi * 0.9  # 10% efficiency loss
            elif budget_change_ratio > 1.2:  # Moderate increase
                adjusted_roi = roi * 0.95  # 5% efficiency loss
            elif budget_change_ratio < 0.5:  # Big decrease
                adjusted_roi = roi * 1.1  # 10% efficiency gain
            elif budget_change_ratio < 0.8:  # Moderate decrease
                adjusted_roi = roi * 1.05  # 5% efficiency gain
            else:
                adjusted_roi = roi
            
            revenue = budget * adjusted_roi
            optimized_roi_1 += revenue
            
            budget_change = budget - current_allocation[channel]["budget"]
            change_emoji = "📈" if budget_change > 0 else "📉" if budget_change < 0 else "➡️"
            
            print(f"  {channel:<15} | ${budget:,} ({budget_change:+,}) {change_emoji} | "
                  f"Adj. ROI: {adjusted_roi:.1f}x")
        
        improvement_1 = optimized_roi_1 - current_total_roi
        print(f"\nProjected Revenue: ${optimized_roi_1:,.0f} ({improvement_1:+,.0f} vs current)")
        print(f"ROI Improvement: {improvement_1/total_budget:.2f}x additional return")
        
        # Scenario 2: Focus on high-growth potential
        print("\n2️⃣ Scenario 2: Growth-Focused Allocation")
        
        # Identify channels with low saturation (high growth potential)
        growth_potential = {}
        for channel, data in current_allocation.items():
            # Growth potential = (1 - saturation) * ROI
            growth_potential[channel] = (1 - data["saturation"]) * data["roi"]
        
        total_growth_potential = sum(growth_potential.values())
        optimized_allocation_2 = {}
        
        for channel, potential in growth_potential.items():
            growth_share = potential / total_growth_potential
            optimized_allocation_2[channel] = {
                "budget": int(total_budget * growth_share),
                "roi": current_allocation[channel]["roi"],
                "saturation": current_allocation[channel]["saturation"]
            }
        
        print("Growth-Focused Allocation:")
        optimized_roi_2 = 0
        for channel, data in optimized_allocation_2.items():
            budget = data["budget"]
            roi = data["roi"]
            
            # Similar diminishing returns calculation
            current_budget = current_allocation[channel]["budget"]
            budget_change_ratio = budget / current_budget if current_budget > 0 else 1
            
            if budget_change_ratio > 1.5:
                adjusted_roi = roi * 0.9
            elif budget_change_ratio > 1.2:
                adjusted_roi = roi * 0.95
            elif budget_change_ratio < 0.5:
                adjusted_roi = roi * 1.1
            elif budget_change_ratio < 0.8:
                adjusted_roi = roi * 1.05
            else:
                adjusted_roi = roi
            
            revenue = budget * adjusted_roi
            optimized_roi_2 += revenue
            
            budget_change = budget - current_allocation[channel]["budget"]
            change_emoji = "📈" if budget_change > 0 else "📉" if budget_change < 0 else "➡️"
            
            print(f"  {channel:<15} | ${budget:,} ({budget_change:+,}) {change_emoji} | "
                  f"Growth Potential: {growth_potential[channel]:.1f}")
        
        improvement_2 = optimized_roi_2 - current_total_roi
        print(f"\nProjected Revenue: ${optimized_roi_2:,.0f} ({improvement_2:+,.0f} vs current)")
        
        # Compare scenarios
        print("\n📈 Scenario Comparison:")
        print(f"  Current Performance:    ${current_total_roi:,.0f} (baseline)")
        print(f"  ROI-Optimized Scenario: ${optimized_roi_1:,.0f} ({improvement_1:+,.0f})")
        print(f"  Growth-Focused Scenario: ${optimized_roi_2:,.0f} ({improvement_2:+,.0f})")
        
        # Recommendation
        if improvement_1 > improvement_2:
            recommended_scenario = "ROI-Optimized"
            recommended_improvement = improvement_1
            recommended_allocation = optimized_allocation_1
        else:
            recommended_scenario = "Growth-Focused"
            recommended_improvement = improvement_2
            recommended_allocation = optimized_allocation_2
        
        print(f"\n💡 Recommendation: {recommended_scenario} Allocation")
        print(f"   Expected improvement: ${recommended_improvement:+,.0f} additional revenue")
        print(f"   ROI lift: +{recommended_improvement/total_budget:.2f}x")
        
        # Implementation steps
        print("\n🚀 Implementation Steps:")
        implementation_steps = []
        
        for channel, data in recommended_allocation.items():
            current_budget = current_allocation[channel]["budget"]
            new_budget = data["budget"]
            change = new_budget - current_budget
            
            if abs(change) > total_budget * 0.05:  # Significant change (>5% of total budget)
                if change > 0:
                    implementation_steps.append(f"Increase {channel} budget by ${change:,}")
                else:
                    implementation_steps.append(f"Reduce {channel} budget by ${abs(change):,}")
        
        for i, step in enumerate(implementation_steps, 1):
            print(f"  {i}. {step}")
        
        # Save optimization report
        outputs_dir = project_root / "outputs"
        outputs_dir.mkdir(parents=True, exist_ok=True)
        
        optimization_report = {
            "analysis_type": "budget_optimization",
            "timestamp": datetime.now().isoformat(),
            "current_allocation": current_allocation,
            "current_performance": {
                "total_budget": total_budget,
                "total_revenue": current_total_roi,
                "overall_roi": current_total_roi / total_budget
            },
            "scenarios": {
                "roi_optimized": {
                    "allocation": optimized_allocation_1,
                    "projected_revenue": optimized_roi_1,
                    "improvement": improvement_1
                },
                "growth_focused": {
                    "allocation": optimized_allocation_2,
                    "projected_revenue": optimized_roi_2,
                    "improvement": improvement_2
                }
            },
            "recommendation": {
                "scenario": recommended_scenario,
                "expected_improvement": recommended_improvement,
                "implementation_steps": implementation_steps
            }
        }
        
        report_path = outputs_dir / f"budget_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w') as f:
            json.dump(optimization_report, f, indent=2)
        
        print(f"\n✅ Optimization report saved to: {report_path}")
        print("\n🎉 Budget optimization analysis completed!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during budget optimization: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)