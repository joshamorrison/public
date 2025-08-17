"""
AI Economy Score Generator
Generates comprehensive economic sentiment score from multiple unstructured data sources.
Inspired by the ChatGPT conference call analysis methodology.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple
import logging
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from .news_client import NewsClient
from .sentiment_analyzer import EconomicSentimentAnalyzer
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage

logger = logging.getLogger(__name__)


class AIEconomyScore:
    """
    Generates AI Economy Score from multiple unstructured data sources.
    Score ranges from -1 (very pessimistic) to +1 (very optimistic).
    """
    
    def __init__(self, 
                 newsapi_key: Optional[str] = None,
                 openai_model: str = "gpt-3.5-turbo"):
        """
        Initialize AI Economy Score generator.
        
        Args:
            newsapi_key: NewsAPI key for news fetching
            openai_model: OpenAI model for analysis
        """
        # Initialize components
        self.news_client = NewsClient(newsapi_key=newsapi_key)
        self.sentiment_analyzer = EconomicSentimentAnalyzer()
        self.llm = ChatOpenAI(model=openai_model, temperature=0.1)
        
        # Economic indicators to track
        self.economic_indicators = {
            'gdp': {
                'keywords': ['gdp', 'gross domestic product', 'economic growth', 'economic output'],
                'weight': 0.25
            },
            'employment': {
                'keywords': ['employment', 'unemployment', 'jobs', 'labor market', 'workforce'],
                'weight': 0.20
            },
            'inflation': {
                'keywords': ['inflation', 'prices', 'cost of living', 'consumer prices', 'cpi'],
                'weight': 0.20
            },
            'investment': {
                'keywords': ['investment', 'capital expenditure', 'business spending', 'capex'],
                'weight': 0.15
            },
            'consumption': {
                'keywords': ['consumer spending', 'retail sales', 'consumption', 'consumer confidence'],
                'weight': 0.10
            },
            'trade': {
                'keywords': ['trade', 'exports', 'imports', 'trade balance', 'international trade'],
                'weight': 0.10
            }
        }
        
        # Score calculation prompt
        self.score_prompt = PromptTemplate(
            input_variables=["news_data", "sentiment_data", "indicator_analysis"],
            template="""
            As an expert economic analyst, generate a comprehensive AI Economy Score based on the provided data.
            
            NEWS DATA SUMMARY:
            {news_data}
            
            SENTIMENT ANALYSIS:
            {sentiment_data}
            
            INDICATOR-SPECIFIC ANALYSIS:
            {indicator_analysis}
            
            Generate an AI Economy Score from -1 (very pessimistic) to +1 (very optimistic) with the following components:
            
            1. OVERALL SCORE (-1 to +1)
            2. CONFIDENCE LEVEL (0 to 1)
            3. INDICATOR BREAKDOWN:
               - GDP/Growth: score and reasoning
               - Employment: score and reasoning
               - Inflation: score and reasoning
               - Investment: score and reasoning
               - Consumption: score and reasoning
               - Trade: score and reasoning
            
            4. KEY INSIGHTS: Top 3 insights driving the score
            5. RISK FACTORS: Main economic risks identified
            6. OUTLOOK: Short-term (3-month) economic outlook
            
            Provide response in JSON format:
            {{
                "overall_score": 0.0,
                "confidence_level": 0.0,
                "indicator_scores": {{
                    "gdp": {{"score": 0.0, "reasoning": ""}},
                    "employment": {{"score": 0.0, "reasoning": ""}},
                    "inflation": {{"score": 0.0, "reasoning": ""}},
                    "investment": {{"score": 0.0, "reasoning": ""}},
                    "consumption": {{"score": 0.0, "reasoning": ""}},
                    "trade": {{"score": 0.0, "reasoning": ""}}
                }},
                "key_insights": ["insight1", "insight2", "insight3"],
                "risk_factors": ["risk1", "risk2", "risk3"],
                "short_term_outlook": "outlook_description"
            }}
            """
        )
        
        logger.info("AI Economy Score generator initialized")
    
    def fetch_economic_news_data(self, 
                                hours_back: int = 72,
                                max_articles: int = 200) -> pd.DataFrame:
        """
        Fetch recent economic news for analysis.
        
        Args:
            hours_back: Hours of news history to fetch
            max_articles: Maximum number of articles
        
        Returns:
            DataFrame with economic news
        """
        logger.info(f"Fetching economic news from last {hours_back} hours")
        
        # Fetch news
        news_df = self.news_client.get_latest_economic_news(
            hours_back=hours_back,
            max_articles=max_articles
        )
        
        if len(news_df) == 0:
            logger.warning("No economic news articles found")
            return pd.DataFrame()
        
        # Filter for high relevance
        high_relevance = news_df[news_df['economic_relevance_score'] >= 0.4]
        
        logger.info(f"Fetched {len(news_df)} articles, {len(high_relevance)} with high economic relevance")
        return high_relevance
    
    def analyze_news_sentiment(self, news_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze sentiment of economic news.
        
        Args:
            news_df: DataFrame with news articles
        
        Returns:
            Sentiment analysis results
        """
        if len(news_df) == 0:
            return {
                'sentiment_df': pd.DataFrame(),
                'metrics': {},
                'indicator_sentiment': {}
            }
        
        logger.info(f"Analyzing sentiment for {len(news_df)} articles")
        
        # Analyze sentiment
        sentiment_df = self.sentiment_analyzer.analyze_articles_sentiment(
            news_df.to_dict('records')
        )
        
        # Calculate overall metrics
        metrics = self.sentiment_analyzer.calculate_sentiment_metrics(sentiment_df)
        
        # Analyze sentiment by economic indicator
        indicator_sentiment = self._analyze_indicator_sentiment(news_df, sentiment_df)
        
        return {
            'sentiment_df': sentiment_df,
            'metrics': metrics,
            'indicator_sentiment': indicator_sentiment
        }
    
    def _analyze_indicator_sentiment(self, 
                                   news_df: pd.DataFrame, 
                                   sentiment_df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """
        Analyze sentiment by economic indicator.
        
        Args:
            news_df: News articles DataFrame
            sentiment_df: Sentiment analysis DataFrame
        
        Returns:
            Sentiment analysis by indicator
        """
        indicator_results = {}
        
        for indicator, config in self.economic_indicators.items():
            keywords = config['keywords']
            
            # Find articles related to this indicator
            indicator_articles = []
            for idx, article in news_df.iterrows():
                text_content = (
                    str(article.get('title', '')) + ' ' + 
                    str(article.get('description', '')) + ' ' + 
                    str(article.get('content', ''))
                ).lower()
                
                # Check for keyword matches
                matches = sum(1 for keyword in keywords if keyword in text_content)
                if matches > 0:
                    indicator_articles.append(idx)
            
            if not indicator_articles:
                indicator_results[indicator] = {
                    'article_count': 0,
                    'sentiment_score': 0.0,
                    'confidence': 0.0
                }
                continue
            
            # Get sentiment for these articles
            indicator_sentiment = sentiment_df[sentiment_df['article_index'].isin(indicator_articles)]
            
            if len(indicator_sentiment) == 0:
                indicator_results[indicator] = {
                    'article_count': 0,
                    'sentiment_score': 0.0,
                    'confidence': 0.0
                }
                continue
            
            # Calculate weighted sentiment score
            sentiment_weights = {'negative': -1, 'neutral': 0, 'positive': 1}
            weighted_scores = (
                indicator_sentiment['sentiment'].map(sentiment_weights) * 
                indicator_sentiment['confidence']
            )
            
            indicator_results[indicator] = {
                'article_count': len(indicator_sentiment),
                'sentiment_score': weighted_scores.mean(),
                'confidence': indicator_sentiment['confidence'].mean(),
                'sentiment_distribution': indicator_sentiment['sentiment'].value_counts().to_dict(),
                'avg_relevance': news_df.loc[indicator_articles, 'economic_relevance_score'].mean()
            }
        
        return indicator_results
    
    def generate_ai_economy_score(self, 
                                 hours_back: int = 72,
                                 include_historical: bool = True) -> Dict[str, Any]:
        """
        Generate comprehensive AI Economy Score.
        
        Args:
            hours_back: Hours of news data to analyze
            include_historical: Whether to include historical context
        
        Returns:
            Complete AI Economy Score analysis
        """
        logger.info("Generating AI Economy Score")
        
        # Fetch and analyze news data
        news_df = self.fetch_economic_news_data(hours_back=hours_back)
        
        if len(news_df) == 0:
            logger.warning("No news data available for scoring")
            return self._generate_fallback_score()
        
        # Analyze sentiment
        sentiment_analysis = self.analyze_news_sentiment(news_df)
        
        # Prepare data summaries for AI analysis
        news_summary = self._summarize_news_data(news_df)
        sentiment_summary = self._summarize_sentiment_data(sentiment_analysis)
        indicator_summary = self._summarize_indicator_analysis(sentiment_analysis['indicator_sentiment'])
        
        # Generate AI score using LLM
        ai_score_result = self._generate_llm_score(
            news_summary, sentiment_summary, indicator_summary
        )
        
        # Calculate quantitative baseline score
        quantitative_score = self._calculate_quantitative_score(sentiment_analysis)
        
        # Combine and validate results
        final_result = self._combine_and_validate_scores(
            ai_score_result, quantitative_score, sentiment_analysis
        )
        
        # Add metadata
        final_result.update({
            'generation_timestamp': datetime.now().isoformat(),
            'data_window_hours': hours_back,
            'articles_analyzed': len(news_df),
            'high_confidence_articles': len(sentiment_analysis['sentiment_df'][
                sentiment_analysis['sentiment_df']['confidence'] > 0.8
            ]) if len(sentiment_analysis['sentiment_df']) > 0 else 0,
            'data_sources': news_df['source'].value_counts().to_dict() if len(news_df) > 0 else {},
            'methodology': 'AI-powered multi-source sentiment analysis'
        })
        
        logger.info(f"Generated AI Economy Score: {final_result.get('overall_score', 0.0):.3f}")
        return final_result
    
    def _summarize_news_data(self, news_df: pd.DataFrame) -> str:
        """Summarize news data for LLM analysis."""
        if len(news_df) == 0:
            return "No news data available"
        
        summary_parts = [
            f"Total articles analyzed: {len(news_df)}",
            f"Time range: {news_df['published_at'].min()} to {news_df['published_at'].max()}",
            f"Average relevance score: {news_df['economic_relevance_score'].mean():.3f}",
            f"Top sources: {', '.join(news_df['source'].value_counts().head(3).index.tolist())}",
            f"Most common keywords: {', '.join(news_df['matched_keywords'].explode().value_counts().head(5).index.tolist()) if 'matched_keywords' in news_df else 'N/A'}"
        ]
        
        # Sample headlines
        top_articles = news_df.nlargest(5, 'economic_relevance_score')
        headlines = [f"- {title}" for title in top_articles['title'].head(5)]
        summary_parts.append("Top headlines:")
        summary_parts.extend(headlines)
        
        return "\n".join(summary_parts)
    
    def _summarize_sentiment_data(self, sentiment_analysis: Dict[str, Any]) -> str:
        """Summarize sentiment analysis for LLM."""
        metrics = sentiment_analysis.get('metrics', {})
        
        if not metrics:
            return "No sentiment data available"
        
        summary_parts = [
            f"Overall sentiment score: {metrics.get('overall_sentiment_score', 0.0):.3f}",
            f"Average confidence: {metrics.get('average_confidence', 0.0):.3f}",
            f"Sentiment distribution: {metrics.get('sentiment_percentages', {})}",
            f"High confidence articles: {metrics.get('high_confidence_articles', 0)}",
            f"Method agreement: {metrics.get('method_agreement_avg', 0.0):.3f}"
        ]
        
        return "\n".join(summary_parts)
    
    def _summarize_indicator_analysis(self, indicator_sentiment: Dict[str, Dict[str, Any]]) -> str:
        """Summarize indicator-specific analysis."""
        if not indicator_sentiment:
            return "No indicator-specific analysis available"
        
        summary_parts = ["Indicator-specific sentiment analysis:"]
        
        for indicator, data in indicator_sentiment.items():
            summary_parts.append(
                f"- {indicator.upper()}: {data['article_count']} articles, "
                f"sentiment score: {data['sentiment_score']:.3f}, "
                f"confidence: {data['confidence']:.3f}"
            )
        
        return "\n".join(summary_parts)
    
    def _generate_llm_score(self, 
                           news_summary: str, 
                           sentiment_summary: str, 
                           indicator_summary: str) -> Dict[str, Any]:
        """Generate AI score using LLM analysis."""
        try:
            prompt = self.score_prompt.format(
                news_data=news_summary,
                sentiment_data=sentiment_summary,
                indicator_analysis=indicator_summary
            )
            
            response = self.llm.invoke([HumanMessage(content=prompt)])
            
            # Parse JSON response
            import json
            try:
                result = json.loads(response.content)
                return result
            except json.JSONDecodeError:
                logger.warning("Failed to parse LLM JSON response, using fallback")
                return self._parse_llm_fallback(response.content)
                
        except Exception as e:
            logger.error(f"LLM score generation failed: {e}")
            return self._generate_fallback_score()
    
    def _parse_llm_fallback(self, content: str) -> Dict[str, Any]:
        """Parse LLM response when JSON parsing fails."""
        # Extract overall score
        import re
        score_match = re.search(r'overall[_\s]*score["\s]*:[\s]*([+-]?\d*\.?\d+)', content, re.IGNORECASE)
        overall_score = float(score_match.group(1)) if score_match else 0.0
        
        # Extract confidence
        conf_match = re.search(r'confidence[_\s]*level["\s]*:[\s]*(\d*\.?\d+)', content, re.IGNORECASE)
        confidence = float(conf_match.group(1)) if conf_match else 0.5
        
        return {
            'overall_score': np.clip(overall_score, -1, 1),
            'confidence_level': np.clip(confidence, 0, 1),
            'indicator_scores': {indicator: {'score': 0.0, 'reasoning': 'Fallback parsing'} 
                               for indicator in self.economic_indicators.keys()},
            'key_insights': ['Analysis based on fallback parsing'],
            'risk_factors': ['Unable to extract detailed risk factors'],
            'short_term_outlook': 'Mixed signals in economic data'
        }
    
    def _calculate_quantitative_score(self, sentiment_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate quantitative baseline score."""
        metrics = sentiment_analysis.get('metrics', {})
        indicator_sentiment = sentiment_analysis.get('indicator_sentiment', {})
        
        # Overall sentiment score from metrics
        overall_sentiment = metrics.get('overall_sentiment_score', 0.0)
        
        # Weighted indicator scores
        weighted_indicator_score = 0.0
        total_weight = 0.0
        
        for indicator, config in self.economic_indicators.items():
            indicator_data = indicator_sentiment.get(indicator, {})
            indicator_score = indicator_data.get('sentiment_score', 0.0)
            weight = config['weight']
            
            # Adjust for article count and confidence
            article_count = indicator_data.get('article_count', 0)
            confidence = indicator_data.get('confidence', 0.5)
            
            # Weight adjustment based on data quality
            quality_adjustment = min(1.0, article_count / 5) * confidence
            adjusted_weight = weight * quality_adjustment
            
            weighted_indicator_score += indicator_score * adjusted_weight
            total_weight += adjusted_weight
        
        # Normalize weighted score
        if total_weight > 0:
            weighted_indicator_score /= total_weight
        
        # Combine overall and indicator-specific scores
        final_score = (overall_sentiment * 0.4 + weighted_indicator_score * 0.6)
        
        # Calculate confidence based on data quality
        confidence = metrics.get('average_confidence', 0.5)
        article_count = metrics.get('total_articles', 0)
        data_quality = min(1.0, article_count / 50)  # Assume 50 articles = high quality
        
        final_confidence = confidence * data_quality
        
        return {
            'overall_score': np.clip(final_score, -1, 1),
            'confidence_level': np.clip(final_confidence, 0, 1),
            'component_scores': {
                'overall_sentiment': overall_sentiment,
                'weighted_indicators': weighted_indicator_score,
                'data_quality': data_quality
            }
        }
    
    def _combine_and_validate_scores(self, 
                                   ai_result: Dict[str, Any], 
                                   quant_result: Dict[str, Any],
                                   sentiment_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Combine and validate AI and quantitative scores."""
        # Use AI result as primary, with quantitative as validation
        ai_score = ai_result.get('overall_score', 0.0)
        quant_score = quant_result.get('overall_score', 0.0)
        
        # Check for significant disagreement
        score_difference = abs(ai_score - quant_score)
        if score_difference > 0.5:
            logger.warning(f"Large disagreement between AI ({ai_score:.3f}) and quantitative ({quant_score:.3f}) scores")
            # Use average with reduced confidence
            final_score = (ai_score + quant_score) / 2
            confidence_penalty = 0.2
        else:
            final_score = ai_score
            confidence_penalty = 0.0
        
        # Combine confidence
        ai_confidence = ai_result.get('confidence_level', 0.5)
        quant_confidence = quant_result.get('confidence_level', 0.5)
        final_confidence = max(ai_confidence, quant_confidence) - confidence_penalty
        
        # Use AI result structure with validated scores
        result = ai_result.copy()
        result.update({
            'overall_score': np.clip(final_score, -1, 1),
            'confidence_level': np.clip(final_confidence, 0, 1),
            'validation': {
                'ai_score': ai_score,
                'quantitative_score': quant_score,
                'score_difference': score_difference,
                'confidence_penalty': confidence_penalty
            }
        })
        
        return result
    
    def _generate_fallback_score(self) -> Dict[str, Any]:
        """Generate fallback score when analysis fails."""
        return {
            'overall_score': 0.0,
            'confidence_level': 0.1,
            'indicator_scores': {
                indicator: {'score': 0.0, 'reasoning': 'Insufficient data'} 
                for indicator in self.economic_indicators.keys()
            },
            'key_insights': ['Insufficient data for reliable analysis'],
            'risk_factors': ['Data availability limitations'],
            'short_term_outlook': 'Unable to assess due to data limitations',
            'fallback': True
        }
    
    def get_historical_scores(self, 
                             days_back: int = 30,
                             score_frequency: str = 'daily') -> pd.DataFrame:
        """
        Generate historical AI Economy Scores.
        
        Args:
            days_back: Number of days of history
            score_frequency: Frequency of score calculation
        
        Returns:
            DataFrame with historical scores
        """
        logger.info(f"Generating {days_back} days of historical AI Economy Scores")
        
        # This would require storing historical news data
        # For now, return a simplified implementation
        
        dates = pd.date_range(
            start=datetime.now() - timedelta(days=days_back),
            end=datetime.now(),
            freq='D' if score_frequency == 'daily' else 'H'
        )
        
        # Placeholder: In real implementation, would analyze historical news for each date
        scores = []
        for date in dates:
            # Simplified score generation for demo
            base_score = np.random.normal(0, 0.3)  # Random walk around neutral
            scores.append({
                'date': date,
                'ai_economy_score': np.clip(base_score, -1, 1),
                'confidence': np.random.uniform(0.4, 0.9),
                'articles_analyzed': np.random.randint(20, 100)
            })
        
        historical_df = pd.DataFrame(scores)
        logger.info(f"Generated {len(historical_df)} historical score points")
        
        return historical_df


if __name__ == "__main__":
    # Example usage
    ai_score = AIEconomyScore()
    
    # Generate current AI Economy Score
    current_score = ai_score.generate_ai_economy_score(hours_back=48)
    
    print("AI ECONOMY SCORE ANALYSIS")
    print("=" * 50)
    print(f"Overall Score: {current_score.get('overall_score', 0.0):.3f}")
    print(f"Confidence: {current_score.get('confidence_level', 0.0):.3f}")
    print(f"Articles Analyzed: {current_score.get('articles_analyzed', 0)}")
    
    # Print indicator breakdown
    indicator_scores = current_score.get('indicator_scores', {})
    print("\nIndicator Breakdown:")
    for indicator, data in indicator_scores.items():
        score = data.get('score', 0.0)
        print(f"  {indicator.upper()}: {score:.3f}")
    
    # Print key insights
    insights = current_score.get('key_insights', [])
    print(f"\nKey Insights:")
    for i, insight in enumerate(insights, 1):
        print(f"  {i}. {insight}")
    
    # Generate historical scores
    historical = ai_score.get_historical_scores(days_back=7)
    print(f"\nHistorical Trend ({len(historical)} points):")
    print(f"  Average Score: {historical['ai_economy_score'].mean():.3f}")
    print(f"  Score Range: {historical['ai_economy_score'].min():.3f} to {historical['ai_economy_score'].max():.3f}")
    print(f"  Score Volatility: {historical['ai_economy_score'].std():.3f}")