"""
Sentiment-Adjusted Forecasting
Integrates news sentiment analysis with economic forecasting to improve prediction accuracy.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import logging
from datetime import datetime, timedelta
import sys
import os

# Using relative imports - no path manipulation needed

try:
    from data.unstructured.news_client import NewsClient
    from data.unstructured.sentiment_analyzer import EconomicSentimentAnalyzer
    NEWS_INTEGRATION_AVAILABLE = True
except ImportError:
    NEWS_INTEGRATION_AVAILABLE = False

logger = logging.getLogger(__name__)

class SentimentAdjustedForecaster:
    """Forecaster that adjusts predictions based on news sentiment."""
    
    def __init__(self, 
                 sentiment_weight: float = 0.1,
                 sentiment_decay: float = 0.8,
                 newsapi_key: Optional[str] = None):
        """
        Initialize sentiment-adjusted forecaster.
        
        Args:
            sentiment_weight: How much sentiment affects forecasts (0-1)
            sentiment_decay: How quickly sentiment impact decays over time
            newsapi_key: News API key for real-time sentiment
        """
        self.sentiment_weight = sentiment_weight
        self.sentiment_decay = sentiment_decay
        self.newsapi_key = newsapi_key
        
        # Initialize news and sentiment components
        if NEWS_INTEGRATION_AVAILABLE:
            self.news_client = NewsClient(newsapi_key=newsapi_key)
            self.sentiment_analyzer = EconomicSentimentAnalyzer(
                use_finbert=True, 
                use_openai=False
            )
        else:
            self.news_client = None
            self.sentiment_analyzer = None
            logger.warning("News integration not available")
        
        # Sentiment impact mapping for different economic indicators
        self.indicator_sensitivity = {
            'gdp': {'positive': 0.02, 'negative': -0.015},
            'unemployment': {'positive': -0.01, 'negative': 0.015},  # Inverse relationship
            'inflation': {'positive': 0.005, 'negative': -0.01},
            'consumer_confidence': {'positive': 0.03, 'negative': -0.025},
            'stock_market': {'positive': 0.05, 'negative': -0.04}
        }
    
    def get_current_sentiment(self, days_back: int = 7) -> Dict[str, Any]:
        """Get current news sentiment for economic indicators."""
        if not self.news_client or not self.sentiment_analyzer:
            # Return mock sentiment for demo
            return {
                'overall_sentiment': 0.1,  # Slightly positive
                'sentiment_strength': 0.6,
                'articles_analyzed': 5,
                'sentiment_breakdown': {'positive': 40, 'neutral': 40, 'negative': 20},
                'data_source': 'mock'
            }
        
        try:
            # Fetch recent economic news
            news_articles = self.news_client.fetch_rss_feeds(max_articles_per_feed=5)
            
            if not news_articles:
                logger.warning("No news articles available")
                return self._get_default_sentiment()
            
            # Filter for economic relevance
            economic_articles = self.news_client.filter_economic_articles(
                news_articles, 
                min_relevance_score=0.1
            )
            
            if not economic_articles:
                logger.warning("No economically relevant articles found")
                return self._get_default_sentiment()
            
            # Analyze sentiment
            sentiment_df = self.sentiment_analyzer.analyze_articles_sentiment(economic_articles)
            
            if len(sentiment_df) == 0:
                return self._get_default_sentiment()
            
            # Calculate sentiment metrics
            metrics = self.sentiment_analyzer.calculate_sentiment_metrics(sentiment_df)
            
            return {
                'overall_sentiment': metrics.get('overall_sentiment_score', 0),
                'sentiment_strength': abs(metrics.get('overall_sentiment_score', 0)),
                'articles_analyzed': len(sentiment_df),
                'sentiment_breakdown': metrics.get('sentiment_percentages', {}),
                'data_source': 'real_news'
            }
            
        except Exception as e:
            logger.error(f"Error getting sentiment: {e}")
            return self._get_default_sentiment()
    
    def _get_default_sentiment(self) -> Dict[str, Any]:
        """Return default neutral sentiment."""
        return {
            'overall_sentiment': 0.0,
            'sentiment_strength': 0.0,
            'articles_analyzed': 0,
            'sentiment_breakdown': {'neutral': 100},
            'data_source': 'default'
        }
    
    def calculate_sentiment_adjustment(self, 
                                    indicator: str, 
                                    sentiment_data: Dict[str, Any],
                                    forecast_horizon: int = 6) -> np.ndarray:
        """Calculate sentiment adjustments for forecasts."""
        
        overall_sentiment = sentiment_data.get('overall_sentiment', 0)
        sentiment_strength = sentiment_data.get('sentiment_strength', 0)
        
        # Get indicator-specific sensitivity
        sensitivity = self.indicator_sensitivity.get(
            indicator.lower(), 
            {'positive': 0.01, 'negative': -0.01}
        )
        
        # Calculate base adjustment
        if overall_sentiment > 0:
            base_adjustment = sensitivity['positive'] * sentiment_strength
        else:
            base_adjustment = sensitivity['negative'] * sentiment_strength
        
        # Create adjustment array with decay over time
        adjustments = []
        for month in range(forecast_horizon):
            # Sentiment impact decays over time
            decay_factor = self.sentiment_decay ** month
            adjustment = base_adjustment * decay_factor * self.sentiment_weight
            adjustments.append(adjustment)
        
        return np.array(adjustments)
    
    def adjust_forecast(self, 
                       base_forecast: np.ndarray,
                       indicator: str,
                       sentiment_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Adjust base forecast using current sentiment."""
        
        if sentiment_data is None:
            sentiment_data = self.get_current_sentiment()
        
        # Calculate sentiment adjustments
        adjustments = self.calculate_sentiment_adjustment(
            indicator, 
            sentiment_data, 
            forecast_horizon=len(base_forecast)
        )
        
        # Apply adjustments (percentage changes)
        adjusted_forecast = base_forecast * (1 + adjustments)
        
        # Calculate confidence adjustment
        sentiment_strength = sentiment_data.get('sentiment_strength', 0)
        confidence_impact = min(sentiment_strength * 0.1, 0.05)  # Max 5% confidence change
        
        return {
            'original_forecast': base_forecast,
            'adjusted_forecast': adjusted_forecast,
            'sentiment_adjustments': adjustments,
            'sentiment_data': sentiment_data,
            'confidence_adjustment': confidence_impact,
            'adjustment_summary': {
                'max_adjustment': np.max(np.abs(adjustments)),
                'avg_adjustment': np.mean(adjustments),
                'total_articles': sentiment_data.get('articles_analyzed', 0),
                'sentiment_direction': 'positive' if sentiment_data.get('overall_sentiment', 0) > 0 else 'negative'
            }
        }
    
    def batch_adjust_forecasts(self, 
                             forecasts: Dict[str, np.ndarray],
                             sentiment_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Adjust multiple indicator forecasts using sentiment."""
        
        if sentiment_data is None:
            sentiment_data = self.get_current_sentiment()
        
        adjusted_results = {}
        
        for indicator, forecast in forecasts.items():
            adjusted_results[indicator] = self.adjust_forecast(
                forecast, 
                indicator, 
                sentiment_data
            )
        
        # Summary statistics
        total_adjustments = []
        for result in adjusted_results.values():
            total_adjustments.extend(result['sentiment_adjustments'])
        
        summary = {
            'indicators_adjusted': len(adjusted_results),
            'sentiment_data': sentiment_data,
            'overall_adjustment_impact': {
                'max_adjustment': np.max(np.abs(total_adjustments)) if total_adjustments else 0,
                'avg_adjustment': np.mean(total_adjustments) if total_adjustments else 0,
                'adjustment_range': [np.min(total_adjustments), np.max(total_adjustments)] if total_adjustments else [0, 0]
            }
        }
        
        return {
            'adjusted_forecasts': adjusted_results,
            'summary': summary
        }

class SentimentForecastAnalyzer:
    """Analyzer for sentiment-adjusted forecast performance."""
    
    def __init__(self):
        self.analysis_history = []
    
    def analyze_sentiment_impact(self, 
                                adjustment_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the impact of sentiment adjustments."""
        
        summary = adjustment_results.get('summary', {})
        sentiment_data = summary.get('sentiment_data', {})
        
        analysis = {
            'sentiment_signal_strength': sentiment_data.get('sentiment_strength', 0),
            'forecast_impact_level': self._categorize_impact(
                summary.get('overall_adjustment_impact', {}).get('max_adjustment', 0)
            ),
            'market_mood': self._interpret_sentiment(sentiment_data),
            'reliability_score': self._calculate_reliability(sentiment_data),
            'recommendations': self._generate_recommendations(adjustment_results)
        }
        
        self.analysis_history.append({
            'timestamp': datetime.now(),
            'analysis': analysis,
            'raw_data': adjustment_results
        })
        
        return analysis
    
    def _categorize_impact(self, max_adjustment: float) -> str:
        """Categorize the impact level of sentiment adjustments."""
        if max_adjustment > 0.03:
            return 'High Impact'
        elif max_adjustment > 0.015:
            return 'Moderate Impact'
        elif max_adjustment > 0.005:
            return 'Low Impact'
        else:
            return 'Minimal Impact'
    
    def _interpret_sentiment(self, sentiment_data: Dict[str, Any]) -> str:
        """Interpret overall market sentiment."""
        sentiment_score = sentiment_data.get('overall_sentiment', 0)
        strength = sentiment_data.get('sentiment_strength', 0)
        
        if strength < 0.2:
            return 'Neutral/Mixed Signals'
        elif sentiment_score > 0.1:
            return 'Optimistic' if strength > 0.5 else 'Cautiously Positive'
        elif sentiment_score < -0.1:
            return 'Pessimistic' if strength > 0.5 else 'Cautiously Negative'
        else:
            return 'Balanced Sentiment'
    
    def _calculate_reliability(self, sentiment_data: Dict[str, Any]) -> float:
        """Calculate reliability score for sentiment analysis."""
        articles_count = sentiment_data.get('articles_analyzed', 0)
        data_source = sentiment_data.get('data_source', 'unknown')
        
        # Base score from article count
        if articles_count >= 10:
            base_score = 0.9
        elif articles_count >= 5:
            base_score = 0.7
        elif articles_count >= 2:
            base_score = 0.5
        else:
            base_score = 0.3
        
        # Adjust for data source
        if data_source == 'real_news':
            source_multiplier = 1.0
        elif data_source == 'mock':
            source_multiplier = 0.6
        else:
            source_multiplier = 0.4
        
        return min(base_score * source_multiplier, 1.0)
    
    def _generate_recommendations(self, adjustment_results: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []
        
        summary = adjustment_results.get('summary', {})
        sentiment_data = summary.get('sentiment_data', {})
        impact = summary.get('overall_adjustment_impact', {})
        
        # Sentiment strength recommendations
        sentiment_strength = sentiment_data.get('sentiment_strength', 0)
        if sentiment_strength > 0.7:
            recommendations.append("Strong sentiment signal detected - consider increasing forecast confidence")
        elif sentiment_strength < 0.3:
            recommendations.append("Weak sentiment signal - rely more on fundamental analysis")
        
        # Impact level recommendations
        max_adjustment = impact.get('max_adjustment', 0)
        if max_adjustment > 0.05:
            recommendations.append("High sentiment impact - review news sources for market drivers")
        elif max_adjustment < 0.01:
            recommendations.append("Low sentiment impact - current forecasts likely stable")
        
        # Data quality recommendations
        articles_count = sentiment_data.get('articles_analyzed', 0)
        if articles_count < 3:
            recommendations.append("Limited news data - consider expanding news sources")
        
        return recommendations

def test_sentiment_adjusted_forecasting():
    """Test sentiment-adjusted forecasting functionality."""
    print("[SENTIMENT] Testing Sentiment-Adjusted Forecasting")
    print("-" * 50)
    
    # Create forecaster
    forecaster = SentimentAdjustedForecaster(
        sentiment_weight=0.15,
        sentiment_decay=0.8
    )
    
    # Test sentiment retrieval
    print("[TEST] Getting current sentiment...")
    sentiment_data = forecaster.get_current_sentiment()
    print(f"  Sentiment Score: {sentiment_data['overall_sentiment']:.3f}")
    print(f"  Articles Analyzed: {sentiment_data['articles_analyzed']}")
    print(f"  Data Source: {sentiment_data['data_source']}")
    
    # Create sample forecasts
    sample_forecasts = {
        'gdp': np.array([23800, 24000, 24200, 24400, 24600, 24800]),
        'unemployment': np.array([4.2, 4.1, 4.0, 3.9, 3.8, 3.7]),
        'inflation': np.array([2.7, 2.6, 2.5, 2.4, 2.3, 2.2])
    }
    
    print(f"\n[ADJUST] Applying sentiment adjustments...")
    results = forecaster.batch_adjust_forecasts(sample_forecasts, sentiment_data)
    
    # Display results
    for indicator, result in results['adjusted_forecasts'].items():
        original = result['original_forecast']
        adjusted = result['adjusted_forecast']
        max_change = np.max(np.abs(result['sentiment_adjustments']))
        
        print(f"\n  {indicator.upper()}:")
        print(f"    Original forecast: {original[0]:.1f} -> {original[-1]:.1f}")
        print(f"    Adjusted forecast: {adjusted[0]:.1f} -> {adjusted[-1]:.1f}")
        print(f"    Max adjustment: {max_change*100:.2f}%")
    
    # Analyze impact
    analyzer = SentimentForecastAnalyzer()
    analysis = analyzer.analyze_sentiment_impact(results)
    
    print(f"\n[ANALYSIS] Sentiment Impact Analysis:")
    print(f"  Market Mood: {analysis['market_mood']}")
    print(f"  Impact Level: {analysis['forecast_impact_level']}")
    print(f"  Reliability Score: {analysis['reliability_score']:.2f}")
    
    if analysis['recommendations']:
        print(f"  Recommendations:")
        for rec in analysis['recommendations']:
            print(f"    - {rec}")
    
    return True

if __name__ == "__main__":
    test_sentiment_adjusted_forecasting()