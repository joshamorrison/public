"""
Simple Executive Reporting System
Generates JSON and text reports for economic forecasting results.
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

class SimpleEconomicReporter:
    """Generate economic forecast reports in multiple formats."""
    
    def __init__(self, output_dir: str = "outputs/reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.report_metadata = {
            'generated_at': datetime.now().isoformat(),
            'report_version': '1.0',
            'platform': 'Generative Econometric Forecasting'
        }
    
    def generate_reports(self, 
                        economic_data: Dict[str, pd.Series],
                        forecast_results: Dict[str, Dict],
                        sentiment_analysis: Optional[Dict] = None,
                        ai_analysis: Optional[str] = None) -> Dict[str, str]:
        """Generate comprehensive reports in multiple formats."""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"economic_forecast_report_{timestamp}"
        
        generated_files = {}
        
        # Generate JSON report
        json_file = self._generate_json_report(
            economic_data, forecast_results, sentiment_analysis, 
            ai_analysis, base_filename
        )
        generated_files['json'] = json_file
        
        # Generate executive summary
        summary_file = self._generate_executive_summary(
            economic_data, forecast_results, sentiment_analysis,
            ai_analysis, base_filename
        )
        generated_files['summary'] = summary_file
        
        # Generate CSV export
        csv_file = self._generate_csv_export(
            economic_data, forecast_results, base_filename
        )
        generated_files['csv'] = csv_file
        
        return generated_files
    
    def _generate_json_report(self, 
                             economic_data: Dict[str, pd.Series],
                             forecast_results: Dict[str, Dict],
                             sentiment_analysis: Optional[Dict],
                             ai_analysis: Optional[str],
                             base_filename: str) -> str:
        """Generate comprehensive JSON report."""
        
        json_data = {
            'metadata': self.report_metadata,
            'report_summary': {
                'indicators_analyzed': len(economic_data),
                'forecast_horizon_months': 6,
                'data_sources': ['FRED API', 'RSS News Feeds'],
                'models_used': ['Exponential Smoothing', 'Sentiment Analysis'],
                'confidence_level': 0.95
            },
            'economic_data': {},
            'forecasts': {},
            'sentiment_analysis': sentiment_analysis or {},
            'ai_analysis': ai_analysis or '',
            'recommendations': self._generate_recommendations(forecast_results, sentiment_analysis)
        }
        
        # Convert economic data
        for indicator, series in economic_data.items():
            json_data['economic_data'][indicator] = {
                'latest_value': float(series.iloc[-1]),
                'latest_date': str(series.index[-1]),
                'data_points': len(series),
                'historical_trend': self._calculate_trend(series)
            }
        
        # Convert forecast results
        for indicator, results in forecast_results.items():
            forecast_series = results.get('forecast', pd.Series())
            if len(forecast_series) > 0:
                json_data['forecasts'][indicator] = {
                    'forecast_values': forecast_series.tolist(),
                    'methodology': results.get('method', 'Exponential Smoothing'),
                    'trend_analysis': {
                        'direction': 'increasing' if forecast_series.iloc[-1] > forecast_series.iloc[0] else 'decreasing',
                        'monthly_change': float(np.mean(np.diff(forecast_series))),
                        'total_change_pct': float((forecast_series.iloc[-1] - forecast_series.iloc[0]) / forecast_series.iloc[0] * 100)
                    }
                }
        
        # Save JSON file
        json_file = self.output_dir / f"{base_filename}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"JSON report generated: {json_file}")
        return str(json_file)
    
    def _generate_executive_summary(self,
                                   economic_data: Dict[str, pd.Series],
                                   forecast_results: Dict[str, Dict],
                                   sentiment_analysis: Optional[Dict],
                                   ai_analysis: Optional[str],
                                   base_filename: str) -> str:
        """Generate executive summary text file."""
        
        summary_file = self.output_dir / f"{base_filename}_executive_summary.txt"
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("ECONOMIC FORECAST EXECUTIVE SUMMARY\\n")
            f.write("=" * 60 + "\\n\\n")
            
            f.write(f"Report Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}\\n")
            f.write(f"Forecast Horizon: 6 months\\n")
            f.write(f"Indicators Analyzed: {len(economic_data)}\\n\\n")
            
            # Key findings
            f.write("KEY ECONOMIC INDICATORS:\\n")
            f.write("-" * 30 + "\\n")
            
            for indicator, series in economic_data.items():
                current_value = series.iloc[-1]
                f.write(f"• {indicator.upper()}: {current_value:.2f}\\n")
                
                if indicator in forecast_results:
                    forecast_series = forecast_results[indicator].get('forecast', pd.Series())
                    if len(forecast_series) > 0:
                        forecast_value = forecast_series.iloc[-1]
                        change = ((forecast_value - current_value) / current_value * 100)
                        direction = "increase" if change > 0 else "decrease"
                        f.write(f"  → Projected to {direction} to {forecast_value:.2f} ({change:+.1f}%)\\n")
            
            f.write("\\n")
            
            # Sentiment analysis
            if sentiment_analysis:
                f.write("MARKET SENTIMENT ANALYSIS:\\n")
                f.write("-" * 30 + "\\n")
                sentiment_score = sentiment_analysis.get('overall_sentiment', 0)
                f.write(f"• Overall Sentiment Score: {sentiment_score:.2f}\\n")
                f.write(f"• Market Mood: {self._interpret_sentiment_score(sentiment_score)}\\n")
                f.write(f"• Articles Analyzed: {sentiment_analysis.get('articles_analyzed', 0)}\\n")
                f.write(f"• Data Source: {sentiment_analysis.get('data_source', 'Unknown')}\\n\\n")
            
            # AI Analysis
            if ai_analysis:
                f.write("AI-GENERATED INSIGHTS:\\n")
                f.write("-" * 30 + "\\n")
                # Clean the AI analysis text
                clean_analysis = ai_analysis.replace('\\n', ' ')[:500] + "..."
                f.write(clean_analysis + "\\n\\n")
            
            # Recommendations
            f.write("STRATEGIC RECOMMENDATIONS:\\n")
            f.write("-" * 30 + "\\n")
            recommendations = self._generate_recommendations(forecast_results, sentiment_analysis)
            for i, rec in enumerate(recommendations[:5], 1):
                f.write(f"{i}. {rec}\\n")
            
            f.write("\\n")
            f.write("=" * 60 + "\\n")
            f.write("Generated by Generative Econometric Forecasting Platform\\n")
        
        logger.info(f"Executive summary generated: {summary_file}")
        return str(summary_file)
    
    def _generate_csv_export(self, 
                            economic_data: Dict[str, pd.Series],
                            forecast_results: Dict[str, Dict],
                            base_filename: str) -> str:
        """Generate CSV data export."""
        
        # Create summary table
        summary_data = []
        
        for indicator, series in economic_data.items():
            current_value = series.iloc[-1]
            current_date = str(series.index[-1])
            
            row = {
                'indicator': indicator,
                'current_value': current_value,
                'current_date': current_date,
                'data_points': len(series)
            }
            
            if indicator in forecast_results:
                forecast_series = forecast_results[indicator].get('forecast', pd.Series())
                if len(forecast_series) > 0:
                    forecast_value = forecast_series.iloc[-1]
                    change_pct = ((forecast_value - current_value) / current_value * 100)
                    row.update({
                        'forecast_6m': forecast_value,
                        'change_percent': change_pct,
                        'trend_direction': 'increasing' if change_pct > 0 else 'decreasing'
                    })
            
            summary_data.append(row)
        
        # Convert to DataFrame and save
        summary_df = pd.DataFrame(summary_data)
        csv_file = self.output_dir / f"{base_filename}_summary.csv"
        summary_df.to_csv(csv_file, index=False)
        
        logger.info(f"CSV export generated: {csv_file}")
        return str(csv_file)
    
    def _calculate_trend(self, series: pd.Series) -> str:
        """Calculate trend direction for a time series."""
        if len(series) < 2:
            return 'insufficient_data'
        
        recent_avg = series.tail(6).mean()
        earlier_avg = series.head(6).mean()
        
        if recent_avg > earlier_avg * 1.02:
            return 'increasing'
        elif recent_avg < earlier_avg * 0.98:
            return 'decreasing'
        else:
            return 'stable'
    
    def _interpret_sentiment_score(self, score: float) -> str:
        """Interpret sentiment score for reporting."""
        if score > 0.3:
            return "Very Positive"
        elif score > 0.1:
            return "Positive"
        elif score > -0.1:
            return "Neutral"
        elif score > -0.3:
            return "Negative"
        else:
            return "Very Negative"
    
    def _generate_recommendations(self,
                                 forecast_results: Dict[str, Dict],
                                 sentiment_analysis: Optional[Dict]) -> List[str]:
        """Generate strategic recommendations."""
        
        recommendations = []
        
        # Forecast-based recommendations
        for indicator, results in forecast_results.items():
            forecast_series = results.get('forecast', pd.Series())
            if len(forecast_series) > 0:
                trend = forecast_series.iloc[-1] - forecast_series.iloc[0]
                
                if indicator.lower() == 'gdp' and trend > 0:
                    recommendations.append("Economic growth trajectory supports increased investment planning")
                elif indicator.lower() == 'unemployment' and trend < 0:
                    recommendations.append("Declining unemployment suggests favorable labor market conditions")
                elif indicator.lower() == 'inflation' and trend > 0:
                    recommendations.append("Rising inflation forecasts warrant monetary policy attention")
        
        # Sentiment-based recommendations
        if sentiment_analysis:
            sentiment_score = sentiment_analysis.get('overall_sentiment', 0)
            if sentiment_score > 0.2:
                recommendations.append("Positive market sentiment supports aggressive growth strategies")
            elif sentiment_score < -0.2:
                recommendations.append("Negative sentiment suggests defensive positioning and risk management")
        
        # General recommendations
        recommendations.extend([
            "Continue monitoring economic indicators for trend changes",
            "Diversify forecasting models to improve prediction accuracy",
            "Integrate real-time sentiment analysis for dynamic adjustments",
            "Review forecast performance against actual outcomes regularly"
        ])
        
        return recommendations

def test_simple_reporting():
    """Test simple reporting functionality."""
    print("[REPORTS] Testing Simple Executive Reporting System")
    print("-" * 50)
    
    # Create sample data
    dates = pd.date_range('2023-01-01', periods=24, freq='M')
    
    sample_economic_data = {
        'gdp': pd.Series(np.random.normal(24000, 500, 24), index=dates),
        'unemployment': pd.Series(np.random.normal(4.0, 0.5, 24), index=dates),
        'inflation': pd.Series(np.random.normal(2.5, 0.3, 24), index=dates)
    }
    
    sample_forecast_results = {
        'gdp': {
            'forecast': pd.Series([24200, 24400, 24600, 24800, 25000, 25200]),
            'method': 'Exponential Smoothing'
        },
        'unemployment': {
            'forecast': pd.Series([3.9, 3.8, 3.7, 3.6, 3.5, 3.4]),
            'method': 'Exponential Smoothing'
        },
        'inflation': {
            'forecast': pd.Series([2.4, 2.3, 2.2, 2.1, 2.0, 1.9]),
            'method': 'Exponential Smoothing'
        }
    }
    
    sample_sentiment = {
        'overall_sentiment': 0.15,
        'articles_analyzed': 8,
        'data_source': 'real_news'
    }
    
    sample_ai_analysis = "Economic outlook remains positive with GDP growth projected to continue. Unemployment trending downward while inflation shows moderating trends."
    
    # Create reporter
    reporter = SimpleEconomicReporter()
    
    # Generate reports
    print("[GENERATE] Creating comprehensive reports...")
    
    generated_files = reporter.generate_reports(
        economic_data=sample_economic_data,
        forecast_results=sample_forecast_results,
        sentiment_analysis=sample_sentiment,
        ai_analysis=sample_ai_analysis
    )
    
    # Display results
    print(f"\\n[SUCCESS] Reports generated:")
    for format_type, file_path in generated_files.items():
        file_size = os.path.getsize(file_path) / 1024  # KB
        print(f"  {format_type.upper()}: {file_path} ({file_size:.1f} KB)")
    
    return generated_files

if __name__ == "__main__":
    test_simple_reporting()