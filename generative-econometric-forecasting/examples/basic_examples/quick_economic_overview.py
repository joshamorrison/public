#!/usr/bin/env python3
"""
Quick Economic Overview Example

Demonstrates rapid economic indicator analysis with minimal setup.
"""

import sys
import os
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

def main():
    """
    Generate a quick economic overview report
    """
    print("📊 Quick Economic Overview Example")
    print("=" * 45)
    
    try:
        # Simulate current economic indicators
        current_indicators = {
            "GDP Growth": {"value": 2.1, "trend": "stable", "vs_previous": 0.1},
            "Inflation (CPI)": {"value": 3.2, "trend": "declining", "vs_previous": -0.3},
            "Unemployment": {"value": 5.1, "trend": "improving", "vs_previous": -0.2},
            "Interest Rate": {"value": 5.25, "trend": "stable", "vs_previous": 0.0},
            "Consumer Confidence": {"value": 68.5, "trend": "improving", "vs_previous": 2.3}
        }
        
        print("🔍 Current Economic Snapshot:")
        print(f"📅 As of: {datetime.now().strftime('%Y-%m-%d')}")
        print()
        
        for indicator, data in current_indicators.items():
            value = data["value"]
            trend = data["trend"]
            change = data["vs_previous"]
            
            # Format change with appropriate symbol
            change_symbol = "📈" if change > 0 else "📉" if change < 0 else "➡️"
            change_text = f"{change:+.1f}" if change != 0 else "no change"
            
            # Format trend
            trend_emoji = {
                "improving": "✅",
                "declining": "⚠️", 
                "stable": "➡️",
                "volatile": "📊"
            }.get(trend, "❓")
            
            print(f"  {indicator:<20}: {value}% {change_symbol} ({change_text}) {trend_emoji} {trend}")
        
        # Generate quick insights
        print("\n💡 Quick Insights:")
        
        insights = []
        
        if current_indicators["GDP Growth"]["trend"] == "stable":
            insights.append("📍 GDP growth remains steady, indicating economic stability")
        
        if current_indicators["Inflation (CPI)"]["trend"] == "declining":
            insights.append("📉 Inflation is moderating, potential for policy adjustments")
        
        if current_indicators["Unemployment"]["trend"] == "improving":
            insights.append("👥 Labor market continues to strengthen")
        
        if current_indicators["Consumer Confidence"]["vs_previous"] > 0:
            insights.append("🛍️ Consumer sentiment is improving")
        
        for i, insight in enumerate(insights, 1):
            print(f"  {i}. {insight}")
        
        # Quick forecast preview
        print("\n🔮 30-Day Outlook:")
        next_month_forecast = {
            "GDP Growth": 2.0,
            "Inflation (CPI)": 3.0,
            "Unemployment": 4.9
        }
        
        for indicator, forecast in next_month_forecast.items():
            current = current_indicators[indicator]["value"]
            change = forecast - current
            direction = "↗️" if change > 0 else "↘️" if change < 0 else "➡️"
            print(f"  {indicator}: {current}% → {forecast}% {direction}")
        
        # Save quick report
        outputs_dir = project_root / "outputs" / "reports"
        outputs_dir.mkdir(parents=True, exist_ok=True)
        
        report_path = outputs_dir / f"quick_overview_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        with open(report_path, 'w') as f:
            f.write("Quick Economic Overview Report\n")
            f.write("=" * 35 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("Current Economic Indicators:\n")
            for indicator, data in current_indicators.items():
                f.write(f"  {indicator}: {data['value']}% ({data['trend']})\n")
            
            f.write("\nInsights:\n")
            for insight in insights:
                f.write(f"  • {insight.split(' ', 1)[1]}\n")  # Remove emoji for text report
        
        print(f"\n✅ Quick overview saved to: {report_path}")
        print("\n🎉 Economic overview completed!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error generating overview: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)