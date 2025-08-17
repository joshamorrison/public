"""
Executive Reporting System
Generates professional PDF and JSON reports for economic forecasting results.
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
import logging
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.backends.backend_pdf import PdfPages
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    from reportlab.graphics.shapes import Drawing
    from reportlab.graphics.charts.linecharts import HorizontalLineChart
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

logger = logging.getLogger(__name__)

class EconomicForecastReport:
    """Generate comprehensive economic forecast reports."""
    
    def __init__(self, output_dir: str = "outputs/reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Report metadata
        self.report_metadata = {
            'generated_at': datetime.now().isoformat(),
            'report_version': '1.0',
            'platform': 'Generative Econometric Forecasting',
            'author': 'AI-Powered Economic Analysis System'
        }
    
    def generate_comprehensive_report(self, 
                                    economic_data: Dict[str, pd.Series],
                                    forecast_results: Dict[str, Dict],
                                    sentiment_analysis: Optional[Dict] = None,
                                    ai_analysis: Optional[str] = None,
                                    model_performance: Optional[Dict] = None) -> Dict[str, str]:
        """Generate comprehensive report in multiple formats."""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"economic_forecast_report_{timestamp}"
        
        generated_files = {}
        
        # Generate JSON report
        json_file = self._generate_json_report(
            economic_data, forecast_results, sentiment_analysis, 
            ai_analysis, model_performance, base_filename
        )
        generated_files['json'] = json_file
        
        # Generate CSV data export
        csv_file = self._generate_csv_export(
            economic_data, forecast_results, base_filename
        )
        generated_files['csv'] = csv_file
        
        # Generate PDF report if available
        if REPORTLAB_AVAILABLE and MATPLOTLIB_AVAILABLE:
            pdf_file = self._generate_pdf_report(
                economic_data, forecast_results, sentiment_analysis,
                ai_analysis, model_performance, base_filename
            )
            generated_files['pdf'] = pdf_file
        else:
            logger.warning("PDF generation not available (missing reportlab/matplotlib)")
        
        # Generate executive summary
        summary_file = self._generate_executive_summary(
            economic_data, forecast_results, sentiment_analysis,
            ai_analysis, base_filename
        )
        generated_files['summary'] = summary_file
        
        return generated_files
    
    def _generate_json_report(self, 
                             economic_data: Dict[str, pd.Series],
                             forecast_results: Dict[str, Dict],
                             sentiment_analysis: Optional[Dict],
                             ai_analysis: Optional[str],
                             model_performance: Optional[Dict],
                             base_filename: str) -> str:
        """Generate comprehensive JSON report."""
        
        # Convert data to JSON-serializable format
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
            'model_performance': model_performance or {},
            'recommendations': self._generate_recommendations(forecast_results, sentiment_analysis)
        }
        
        # Convert economic data
        for indicator, series in economic_data.items():
            json_data['economic_data'][indicator] = {
                'latest_value': float(series.iloc[-1]),
                'latest_date': series.index[-1].isoformat() if hasattr(series.index[-1], 'isoformat') else str(series.index[-1]),
                'data_points': len(series),
                'historical_data': [
                    {'date': idx.isoformat() if hasattr(idx, 'isoformat') else str(idx), 'value': float(val)}
                    for idx, val in series.tail(12).items()  # Last 12 months
                ]
            }
        
        # Convert forecast results
        for indicator, results in forecast_results.items():
            forecast_series = results.get('forecast', pd.Series())
            json_data['forecasts'][indicator] = {
                'forecast_values': forecast_series.tolist() if hasattr(forecast_series, 'tolist') else list(forecast_series),
                'forecast_dates': [
                    (datetime.now() + timedelta(days=30*i)).strftime('%Y-%m-%d')
                    for i in range(len(forecast_series))
                ],
                'methodology': results.get('method', 'Exponential Smoothing'),
                'confidence_intervals': results.get('confidence_intervals', []),
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
    
    def _generate_csv_export(self, 
                            economic_data: Dict[str, pd.Series],
                            forecast_results: Dict[str, Dict],
                            base_filename: str) -> str:
        """Generate CSV data export."""
        
        # Create combined dataframe
        all_data = {}
        
        # Add historical data
        for indicator, series in economic_data.items():
            all_data[f"{indicator}_historical"] = series
        
        # Add forecast data
        for indicator, results in forecast_results.items():
            forecast_series = results.get('forecast', pd.Series())
            if len(forecast_series) > 0:
                # Create future dates
                last_date = economic_data[indicator].index[-1] if indicator in economic_data else datetime.now()
                future_dates = pd.date_range(
                    start=last_date + timedelta(days=30),
                    periods=len(forecast_series),
                    freq='M'
                )
                forecast_df = pd.Series(forecast_series.values, index=future_dates)
                all_data[f"{indicator}_forecast"] = forecast_df
        
        # Combine into single dataframe
        combined_df = pd.DataFrame(all_data)
        
        # Save CSV file
        csv_file = self.output_dir / f"{base_filename}_data.csv"
        combined_df.to_csv(csv_file)
        
        logger.info(f"CSV export generated: {csv_file}")
        return str(csv_file)
    
    def _generate_pdf_report(self,
                            economic_data: Dict[str, pd.Series],
                            forecast_results: Dict[str, Dict],
                            sentiment_analysis: Optional[Dict],
                            ai_analysis: Optional[str],
                            model_performance: Optional[Dict],
                            base_filename: str) -> str:
        """Generate professional PDF report."""
        
        pdf_file = self.output_dir / f"{base_filename}.pdf"
        
        # Create PDF document
        doc = SimpleDocTemplate(str(pdf_file), pagesize=A4)
        styles = getSampleStyleSheet()
        story = []
        
        # Title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.darkblue,
            alignment=1  # Center alignment
        )
        story.append(Paragraph("Economic Forecast Report", title_style))
        story.append(Spacer(1, 20))
        
        # Executive Summary
        story.append(Paragraph("Executive Summary", styles['Heading2']))
        
        summary_text = self._create_executive_summary_text(
            economic_data, forecast_results, sentiment_analysis
        )
        story.append(Paragraph(summary_text, styles['Normal']))
        story.append(Spacer(1, 20))
        
        # Key Metrics Table
        story.append(Paragraph("Key Economic Indicators", styles['Heading2']))
        
        # Create metrics table
        table_data = [['Indicator', 'Current Value', '6-Month Forecast', 'Change']]
        
        for indicator, series in economic_data.items():
            current_value = f"{series.iloc[-1]:.1f}"
            
            if indicator in forecast_results:
                forecast_series = forecast_results[indicator].get('forecast', pd.Series())
                if len(forecast_series) > 0:
                    forecast_value = f"{forecast_series.iloc[-1]:.1f}"
                    change_pct = ((forecast_series.iloc[-1] - series.iloc[-1]) / series.iloc[-1] * 100)
                    change_str = f"{change_pct:+.1f}%"
                else:
                    forecast_value = "N/A"
                    change_str = "N/A"
            else:
                forecast_value = "N/A"
                change_str = "N/A"
            
            table_data.append([
                indicator.upper(),
                current_value,
                forecast_value,
                change_str
            ])
        
        table = Table(table_data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(table)
        story.append(Spacer(1, 20))
        
        # Sentiment Analysis Section
        if sentiment_analysis:
            story.append(Paragraph("Market Sentiment Analysis", styles['Heading2']))
            
            sentiment_text = f\"\"\"Overall Sentiment Score: {sentiment_analysis.get('overall_sentiment', 0):.2f}<br/>Articles Analyzed: {sentiment_analysis.get('articles_analyzed', 0)}<br/>Market Mood: {self._interpret_sentiment_score(sentiment_analysis.get('overall_sentiment', 0))}<br/>Data Reliability: {sentiment_analysis.get('data_source', 'Unknown')}\"\"\"\n            \n            story.append(Paragraph(sentiment_text, styles['Normal']))\n            story.append(Spacer(1, 20))\n        \n        # AI Analysis Section\n        if ai_analysis:\n            story.append(Paragraph("AI-Generated Insights", styles['Heading2']))\n            \n            # Clean and format AI analysis\n            clean_analysis = ai_analysis.replace('\\n', '<br/>')[:1000] + \"...\"\n            story.append(Paragraph(clean_analysis, styles['Normal']))\n            story.append(Spacer(1, 20))\n        \n        # Recommendations\n        story.append(Paragraph("Strategic Recommendations", styles['Heading2']))\n        recommendations = self._generate_recommendations(forecast_results, sentiment_analysis)\n        \n        for i, rec in enumerate(recommendations[:5], 1):\n            story.append(Paragraph(f\"{i}. {rec}\", styles['Normal']))\n        \n        story.append(Spacer(1, 20))\n        \n        # Footer\n        footer_text = f\"Generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}<br/>Generative Econometric Forecasting Platform\"\n        story.append(Paragraph(footer_text, styles['Normal']))\n        \n        # Build PDF\n        doc.build(story)\n        \n        logger.info(f"PDF report generated: {pdf_file}\")\n        return str(pdf_file)\n    \n    def _generate_executive_summary(self,\n                                   economic_data: Dict[str, pd.Series],\n                                   forecast_results: Dict[str, Dict],\n                                   sentiment_analysis: Optional[Dict],\n                                   ai_analysis: Optional[str],\n                                   base_filename: str) -> str:\n        \"\"\"Generate executive summary text file.\"\"\"\n        \n        summary_file = self.output_dir / f\"{base_filename}_executive_summary.txt\"\n        \n        with open(summary_file, 'w', encoding='utf-8') as f:\n            f.write(\"EXECUTIVE SUMMARY\\n\")\n            f.write(\"=\" * 50 + \"\\n\\n\")\n            \n            f.write(f\"Report Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}\\n\")\n            f.write(f\"Forecast Horizon: 6 months\\n\")\n            f.write(f\"Indicators Analyzed: {len(economic_data)}\\n\\n\")\n            \n            # Key findings\n            f.write(\"KEY FINDINGS:\\n\")\n            f.write(\"-\" * 20 + \"\\n\")\n            \n            for indicator, series in economic_data.items():\n                current_value = series.iloc[-1]\n                f.write(f\"• {indicator.upper()}: Currently {current_value:.1f}\\n\")\n                \n                if indicator in forecast_results:\n                    forecast_series = forecast_results[indicator].get('forecast', pd.Series())\n                    if len(forecast_series) > 0:\n                        forecast_value = forecast_series.iloc[-1]\n                        change = ((forecast_value - current_value) / current_value * 100)\n                        direction = \"increase\" if change > 0 else \"decrease\"\n                        f.write(f\"  Forecasted to {direction} to {forecast_value:.1f} ({change:+.1f}%)\\n\")\n            \n            f.write(\"\\n\")\n            \n            # Sentiment analysis\n            if sentiment_analysis:\n                f.write(\"MARKET SENTIMENT:\\n\")\n                f.write(\"-\" * 20 + \"\\n\")\n                sentiment_score = sentiment_analysis.get('overall_sentiment', 0)\n                f.write(f\"• Overall sentiment: {sentiment_score:.2f}\\n\")\n                f.write(f\"• Market mood: {self._interpret_sentiment_score(sentiment_score)}\\n\")\n                f.write(f\"• Articles analyzed: {sentiment_analysis.get('articles_analyzed', 0)}\\n\\n\")\n            \n            # Recommendations\n            f.write(\"RECOMMENDATIONS:\\n\")\n            f.write(\"-\" * 20 + \"\\n\")\n            recommendations = self._generate_recommendations(forecast_results, sentiment_analysis)\n            for i, rec in enumerate(recommendations[:3], 1):\n                f.write(f\"{i}. {rec}\\n\")\n        \n        logger.info(f\"Executive summary generated: {summary_file}\")\n        return str(summary_file)\n    \n    def _create_executive_summary_text(self,\n                                      economic_data: Dict[str, pd.Series],\n                                      forecast_results: Dict[str, Dict],\n                                      sentiment_analysis: Optional[Dict]) -> str:\n        \"\"\"Create executive summary text for PDF.\"\"\"\n        \n        summary_parts = []\n        \n        # Overall assessment\n        summary_parts.append(\n            f\"This report analyzes {len(economic_data)} key economic indicators \"\n            f\"and provides 6-month forecasts based on current market conditions \"\n            f\"and sentiment analysis.\"\n        )\n        \n        # Key trends\n        increasing_indicators = []\n        decreasing_indicators = []\n        \n        for indicator, series in economic_data.items():\n            if indicator in forecast_results:\n                forecast_series = forecast_results[indicator].get('forecast', pd.Series())\n                if len(forecast_series) > 0:\n                    change = forecast_series.iloc[-1] - series.iloc[-1]\n                    if change > 0:\n                        increasing_indicators.append(indicator.upper())\n                    else:\n                        decreasing_indicators.append(indicator.upper())\n        \n        if increasing_indicators:\n            summary_parts.append(\n                f\"Forecasts indicate upward trends in {', '.join(increasing_indicators)}.\"\n            )\n        \n        if decreasing_indicators:\n            summary_parts.append(\n                f\"Projected declines are expected in {', '.join(decreasing_indicators)}.\"\n            )\n        \n        # Sentiment impact\n        if sentiment_analysis:\n            sentiment_score = sentiment_analysis.get('overall_sentiment', 0)\n            if sentiment_score > 0.1:\n                summary_parts.append(\"Market sentiment analysis indicates positive outlook.\")\n            elif sentiment_score < -0.1:\n                summary_parts.append(\"Market sentiment analysis suggests cautious approach.\")\n            else:\n                summary_parts.append(\"Market sentiment remains neutral.\")\n        \n        return \" \".join(summary_parts)\n    \n    def _interpret_sentiment_score(self, score: float) -> str:\n        \"\"\"Interpret sentiment score for reporting.\"\"\"\n        if score > 0.3:\n            return \"Very Positive\"\n        elif score > 0.1:\n            return \"Positive\"\n        elif score > -0.1:\n            return \"Neutral\"\n        elif score > -0.3:\n            return \"Negative\"\n        else:\n            return \"Very Negative\"\n    \n    def _generate_recommendations(self,\n                                 forecast_results: Dict[str, Dict],\n                                 sentiment_analysis: Optional[Dict]) -> List[str]:\n        \"\"\"Generate strategic recommendations.\"\"\"\n        \n        recommendations = []\n        \n        # Forecast-based recommendations\n        for indicator, results in forecast_results.items():\n            forecast_series = results.get('forecast', pd.Series())\n            if len(forecast_series) > 0:\n                trend = forecast_series.iloc[-1] - forecast_series.iloc[0]\n                \n                if indicator.lower() == 'gdp' and trend > 0:\n                    recommendations.append(\"Economic growth trajectory supports increased investment planning\")\n                elif indicator.lower() == 'unemployment' and trend < 0:\n                    recommendations.append(\"Declining unemployment suggests favorable labor market conditions\")\n                elif indicator.lower() == 'inflation' and trend > 0:\n                    recommendations.append(\"Rising inflation forecasts warrant monetary policy attention\")\n        \n        # Sentiment-based recommendations\n        if sentiment_analysis:\n            sentiment_score = sentiment_analysis.get('overall_sentiment', 0)\n            if sentiment_score > 0.2:\n                recommendations.append(\"Positive market sentiment supports aggressive growth strategies\")\n            elif sentiment_score < -0.2:\n                recommendations.append(\"Negative sentiment suggests defensive positioning and risk management\")\n        \n        # General recommendations\n        recommendations.extend([\n            \"Continue monitoring economic indicators for trend changes\",\n            \"Diversify forecasting models to improve prediction accuracy\",\n            \"Integrate real-time sentiment analysis for dynamic adjustments\",\n            \"Regular review of forecast performance against actual outcomes\"\n        ])\n        \n        return recommendations\n\ndef test_executive_reporting():\n    \"\"\"Test executive reporting functionality.\"\"\"\n    print(\"[REPORTS] Testing Executive Reporting System\")\n    print(\"-\" * 50)\n    \n    # Create sample data\n    dates = pd.date_range('2023-01-01', periods=24, freq='M')\n    \n    sample_economic_data = {\n        'gdp': pd.Series(np.random.normal(24000, 500, 24), index=dates),\n        'unemployment': pd.Series(np.random.normal(4.0, 0.5, 24), index=dates),\n        'inflation': pd.Series(np.random.normal(2.5, 0.3, 24), index=dates)\n    }\n    \n    sample_forecast_results = {\n        'gdp': {\n            'forecast': pd.Series([24200, 24400, 24600, 24800, 25000, 25200]),\n            'method': 'Exponential Smoothing'\n        },\n        'unemployment': {\n            'forecast': pd.Series([3.9, 3.8, 3.7, 3.6, 3.5, 3.4]),\n            'method': 'Exponential Smoothing'\n        },\n        'inflation': {\n            'forecast': pd.Series([2.4, 2.3, 2.2, 2.1, 2.0, 1.9]),\n            'method': 'Exponential Smoothing'\n        }\n    }\n    \n    sample_sentiment = {\n        'overall_sentiment': 0.15,\n        'articles_analyzed': 8,\n        'data_source': 'real_news'\n    }\n    \n    sample_ai_analysis = \"Economic outlook remains positive with GDP growth projected to continue. Unemployment trending downward while inflation shows moderating trends.\"\n    \n    # Create reporter\n    reporter = EconomicForecastReport()\n    \n    # Generate reports\n    print(\"[GENERATE] Creating comprehensive reports...\")\n    \n    generated_files = reporter.generate_comprehensive_report(\n        economic_data=sample_economic_data,\n        forecast_results=sample_forecast_results,\n        sentiment_analysis=sample_sentiment,\n        ai_analysis=sample_ai_analysis\n    )\n    \n    # Display results\n    print(f\"\\n[SUCCESS] Reports generated:\")\n    for format_type, file_path in generated_files.items():\n        file_size = os.path.getsize(file_path) / 1024  # KB\n        print(f\"  {format_type.upper()}: {file_path} ({file_size:.1f} KB)\")\n    \n    return generated_files\n\nif __name__ == \"__main__\":\n    test_executive_reporting()