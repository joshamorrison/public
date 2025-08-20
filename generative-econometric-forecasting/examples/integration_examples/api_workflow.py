#!/usr/bin/env python3
"""
API Integration Workflow Example

Demonstrates how to integrate the forecasting platform with external APIs
and create automated forecasting workflows.
"""

import sys
import json
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

def simulate_external_api_call(endpoint: str, params: dict = None):
    """
    Simulate external API calls for demo purposes
    In production, this would make actual HTTP requests
    """
    if endpoint == "fred_economic_data":
        return {
            "status": "success",
            "data": {
                "GDP": [2.1, 2.3, 2.0, 1.9],
                "CPI": [3.2, 3.1, 2.9, 3.0],
                "timestamp": datetime.now().isoformat()
            }
        }
    elif endpoint == "market_sentiment":
        return {
            "status": "success", 
            "data": {
                "sentiment_score": 0.65,
                "confidence": 0.82,
                "trend": "positive"
            }
        }
    elif endpoint == "notify_stakeholders":
        return {"status": "success", "message": "Notifications sent"}
    
    return {"status": "error", "message": "Unknown endpoint"}

def main():
    """
    Demonstrate end-to-end API integration workflow
    """
    print("🔗 API Integration Workflow Example")
    print("=" * 45)
    
    try:
        # Step 1: Fetch external economic data
        print("📥 Step 1: Fetching external economic data...")
        
        fred_response = simulate_external_api_call("fred_economic_data", {
            "indicators": ["GDP", "CPI"],
            "period": "quarterly"
        })
        
        if fred_response["status"] == "success":
            economic_data = fred_response["data"]
            print(f"  ✅ Retrieved data for: {', '.join(economic_data.keys())}")
            print(f"  📅 Last updated: {economic_data['timestamp']}")
        else:
            raise Exception(f"Failed to fetch economic data: {fred_response['message']}")
        
        # Step 2: Get market sentiment
        print("\n📊 Step 2: Analyzing market sentiment...")
        
        sentiment_response = simulate_external_api_call("market_sentiment")
        
        if sentiment_response["status"] == "success":
            sentiment_data = sentiment_response["data"]
            sentiment_score = sentiment_data["sentiment_score"]
            sentiment_trend = sentiment_data["trend"]
            print(f"  📈 Market sentiment: {sentiment_score:.1%} ({sentiment_trend})")
            print(f"  🎯 Confidence level: {sentiment_data['confidence']:.1%}")
        else:
            raise Exception("Failed to fetch sentiment data")
        
        # Step 3: Run integrated forecast
        print("\n🔮 Step 3: Running integrated forecast...")
        
        # Simulate forecast incorporating external data
        forecast_config = {
            "base_data": economic_data,
            "sentiment_adjustment": sentiment_score,
            "horizon_months": 6,
            "confidence_level": 0.90
        }
        
        # Enhanced forecast results incorporating sentiment
        enhanced_forecast = {
            "GDP": {
                "baseline": [2.0, 1.9, 1.8],
                "sentiment_adjusted": [2.1, 2.0, 1.9],  # Slightly higher due to positive sentiment
                "confidence": 0.87
            },
            "CPI": {
                "baseline": [2.8, 2.7, 2.6],
                "sentiment_adjusted": [2.9, 2.8, 2.7],  # Market optimism may drive prices
                "confidence": 0.84
            }
        }
        
        print("  📈 Forecast Results (Sentiment-Adjusted):")
        for indicator, forecasts in enhanced_forecast.items():
            baseline = forecasts["baseline"]
            adjusted = forecasts["sentiment_adjusted"]
            confidence = forecasts["confidence"]
            
            print(f"    {indicator}:")
            print(f"      Baseline: {baseline[0]:.1f}% → {baseline[-1]:.1f}%")
            print(f"      Adjusted: {adjusted[0]:.1f}% → {adjusted[-1]:.1f}% (confidence: {confidence:.1%})")
        
        # Step 4: Generate alerts and notifications
        print("\n🚨 Step 4: Processing alerts...")
        
        alerts = []
        
        # Check for significant forecast changes
        for indicator, forecasts in enhanced_forecast.items():
            baseline_change = forecasts["baseline"][-1] - forecasts["baseline"][0]
            adjusted_change = forecasts["sentiment_adjusted"][-1] - forecasts["sentiment_adjusted"][0]
            sentiment_impact = abs(adjusted_change - baseline_change)
            
            if sentiment_impact > 0.15:  # Significant sentiment impact
                alerts.append({
                    "type": "sentiment_impact",
                    "indicator": indicator,
                    "impact": sentiment_impact,
                    "message": f"Market sentiment significantly affecting {indicator} forecast"
                })
            
            if forecasts["confidence"] < 0.85:  # Low confidence
                alerts.append({
                    "type": "low_confidence",
                    "indicator": indicator,
                    "confidence": forecasts["confidence"],
                    "message": f"Low confidence in {indicator} forecast ({forecasts['confidence']:.1%})"
                })
        
        if alerts:
            print(f"  ⚠️  Generated {len(alerts)} alerts:")
            for alert in alerts:
                print(f"    • {alert['message']}")
        else:
            print("  ✅ No alerts generated - forecasts within normal parameters")
        
        # Step 5: Send notifications
        print("\n📤 Step 5: Sending stakeholder notifications...")
        
        notification_payload = {
            "forecast_summary": enhanced_forecast,
            "alerts": alerts,
            "sentiment_data": sentiment_data,
            "generated_at": datetime.now().isoformat()
        }
        
        notify_response = simulate_external_api_call("notify_stakeholders", notification_payload)
        
        if notify_response["status"] == "success":
            print("  ✅ Stakeholder notifications sent successfully")
        else:
            print("  ⚠️  Warning: Failed to send notifications")
        
        # Step 6: Save workflow results
        print("\n💾 Step 6: Saving workflow results...")
        
        outputs_dir = project_root / "outputs" / "reports"
        outputs_dir.mkdir(parents=True, exist_ok=True)
        
        workflow_results = {
            "workflow_id": f"api_integration_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "execution_time": datetime.now().isoformat(),
            "external_data": {
                "economic_data": economic_data,
                "sentiment_data": sentiment_data
            },
            "forecast_results": enhanced_forecast,
            "alerts_generated": alerts,
            "notification_status": notify_response["status"]
        }
        
        results_path = outputs_dir / f"api_workflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_path, 'w') as f:
            json.dump(workflow_results, f, indent=2)
        
        print(f"  📁 Workflow results saved to: {results_path}")
        
        # Summary
        print("\n🎉 API Integration Workflow Completed!")
        print("=" * 45)
        print("✅ External data integration")
        print("✅ Sentiment-adjusted forecasting") 
        print("✅ Automated alert generation")
        print("✅ Stakeholder notifications")
        print("✅ Results persistence")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in API workflow: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)