"""
Economic Sentiment Analyzer
Analyzes sentiment in financial news and economic texts using FinBERT and custom models.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("Transformers not available. Install with: pip install transformers")

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logging.warning("PyTorch not available. Install with: pip install torch")

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage

logger = logging.getLogger(__name__)


class EconomicSentimentAnalyzer:
    """Analyzes economic sentiment in text using multiple approaches."""
    
    def __init__(self, 
                 use_finbert: bool = True,
                 use_openai: bool = True,
                 openai_model: str = "gpt-3.5-turbo",
                 device: Optional[str] = None):
        """
        Initialize sentiment analyzer.
        
        Args:
            use_finbert: Whether to use FinBERT model
            use_openai: Whether to use OpenAI for sentiment analysis
            openai_model: OpenAI model to use
            device: Device for model inference
        """
        self.use_finbert = use_finbert and TRANSFORMERS_AVAILABLE and TORCH_AVAILABLE
        self.use_openai = use_openai
        self.device = device or ("cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu")
        
        # Initialize FinBERT
        self.finbert_pipeline = None
        if self.use_finbert:
            try:
                self.finbert_pipeline = pipeline(
                    "sentiment-analysis",
                    model="ProsusAI/finbert",
                    device=0 if self.device == "cuda" else -1
                )
                logger.info("FinBERT sentiment analyzer initialized")
            except Exception as e:
                logger.warning(f"FinBERT initialization failed: {e}")
                self.use_finbert = False
        
        # Initialize OpenAI client
        self.llm = None
        if self.use_openai:
            try:
                self.llm = ChatOpenAI(model=openai_model, temperature=0.1)
                logger.info(f"OpenAI {openai_model} initialized for sentiment analysis")
            except Exception as e:
                logger.warning(f"OpenAI initialization failed: {e}")
                self.use_openai = False
        
        # Economic sentiment keywords
        self.positive_economic_terms = [
            'growth', 'expansion', 'recovery', 'boom', 'bullish', 'optimistic',
            'upbeat', 'strong', 'robust', 'healthy', 'improving', 'rising',
            'gains', 'increase', 'surge', 'rally', 'confidence', 'stability'
        ]
        
        self.negative_economic_terms = [
            'recession', 'decline', 'slowdown', 'crisis', 'bearish', 'pessimistic',
            'weak', 'fragile', 'unstable', 'falling', 'dropping', 'plunge',
            'crash', 'downturn', 'uncertainty', 'volatility', 'risk', 'concern'
        ]
        
        # Sentiment prompt template
        self.sentiment_prompt = PromptTemplate(
            input_variables=["text"],
            template="""
            Analyze the economic sentiment of the following text. Consider the implications for:
            - Economic growth prospects
            - Market confidence
            - Investment climate
            - Economic stability
            
            Text: {text}
            
            Provide your analysis in JSON format:
            {{
                "overall_sentiment": "positive/negative/neutral",
                "confidence_score": 0.0-1.0,
                "economic_indicators": {{
                    "growth_outlook": "positive/negative/neutral",
                    "market_confidence": "positive/negative/neutral", 
                    "investment_climate": "positive/negative/neutral",
                    "stability_assessment": "positive/negative/neutral"
                }},
                "key_phrases": ["phrase1", "phrase2"],
                "reasoning": "Brief explanation of sentiment assessment"
            }}
            """
        )
        
        logger.info("Economic sentiment analyzer initialized")
    
    def analyze_finbert_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment using FinBERT.
        
        Args:
            text: Text to analyze
        
        Returns:
            FinBERT sentiment results
        """
        if not self.finbert_pipeline:
            return {'error': 'FinBERT not available'}
        
        try:
            # Truncate text if too long
            max_length = 512
            if len(text) > max_length:
                text = text[:max_length]
            
            result = self.finbert_pipeline(text)[0]
            
            return {
                'label': result['label'].lower(),
                'confidence': result['score'],
                'model': 'finbert'
            }
            
        except Exception as e:
            logger.error(f"FinBERT analysis failed: {e}")
            return {'error': str(e)}
    
    def analyze_keyword_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment using economic keyword matching.
        
        Args:
            text: Text to analyze
        
        Returns:
            Keyword-based sentiment results
        """
        text_lower = text.lower()
        
        positive_matches = sum(1 for term in self.positive_economic_terms if term in text_lower)
        negative_matches = sum(1 for term in self.negative_economic_terms if term in text_lower)
        
        total_matches = positive_matches + negative_matches
        
        if total_matches == 0:
            sentiment = 'neutral'
            confidence = 0.5
        elif positive_matches > negative_matches:
            sentiment = 'positive'
            confidence = min(0.9, 0.5 + (positive_matches / (total_matches + 1)) * 0.4)
        elif negative_matches > positive_matches:
            sentiment = 'negative'
            confidence = min(0.9, 0.5 + (negative_matches / (total_matches + 1)) * 0.4)
        else:
            sentiment = 'neutral'
            confidence = 0.6
        
        # Find matched terms
        positive_found = [term for term in self.positive_economic_terms if term in text_lower]
        negative_found = [term for term in self.negative_economic_terms if term in text_lower]
        
        return {
            'label': sentiment,
            'confidence': confidence,
            'positive_matches': positive_matches,
            'negative_matches': negative_matches,
            'positive_terms': positive_found,
            'negative_terms': negative_found,
            'model': 'keyword_matching'
        }
    
    def analyze_openai_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment using OpenAI.
        
        Args:
            text: Text to analyze
        
        Returns:
            OpenAI sentiment analysis results
        """
        if not self.llm:
            return {'error': 'OpenAI not available'}
        
        try:
            prompt = self.sentiment_prompt.format(text=text[:2000])  # Limit length
            response = self.llm.invoke([HumanMessage(content=prompt)])
            
            # Try to parse JSON response
            import json
            try:
                result = json.loads(response.content)
                result['model'] = 'openai'
                return result
            except json.JSONDecodeError:
                # Fallback if JSON parsing fails
                content = response.content.lower()
                if 'positive' in content:
                    sentiment = 'positive'
                elif 'negative' in content:
                    sentiment = 'negative'
                else:
                    sentiment = 'neutral'
                
                return {
                    'overall_sentiment': sentiment,
                    'confidence_score': 0.7,
                    'reasoning': response.content,
                    'model': 'openai_fallback'
                }
            
        except Exception as e:
            logger.error(f"OpenAI sentiment analysis failed: {e}")
            return {'error': str(e)}
    
    def analyze_text_sentiment(self, 
                              text: str,
                              methods: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Analyze sentiment using multiple methods and combine results.
        
        Args:
            text: Text to analyze
            methods: List of methods to use
        
        Returns:
            Combined sentiment analysis results
        """
        if methods is None:
            methods = []
            if self.use_finbert:
                methods.append('finbert')
            if self.use_openai:
                methods.append('openai')
            methods.append('keyword')
        
        results = {
            'text_length': len(text),
            'methods_used': methods,
            'individual_results': {},
            'combined_result': {}
        }
        
        # Run individual analyses
        if 'finbert' in methods and self.use_finbert:
            finbert_result = self.analyze_finbert_sentiment(text)
            results['individual_results']['finbert'] = finbert_result
        
        if 'keyword' in methods:
            keyword_result = self.analyze_keyword_sentiment(text)
            results['individual_results']['keyword'] = keyword_result
        
        if 'openai' in methods and self.use_openai:
            openai_result = self.analyze_openai_sentiment(text)
            results['individual_results']['openai'] = openai_result
        
        # Combine results
        results['combined_result'] = self._combine_sentiment_results(
            results['individual_results']
        )
        
        return results
    
    def _combine_sentiment_results(self, individual_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Combine multiple sentiment analysis results.
        
        Args:
            individual_results: Dictionary of individual analysis results
        
        Returns:
            Combined sentiment result
        """
        if not individual_results:
            return {'sentiment': 'neutral', 'confidence': 0.5}
        
        # Extract valid results
        valid_results = []
        for method, result in individual_results.items():
            if 'error' not in result:
                # Normalize labels
                if method == 'finbert':
                    label = result.get('label', 'neutral')
                    confidence = result.get('confidence', 0.5)
                elif method == 'keyword':
                    label = result.get('label', 'neutral')
                    confidence = result.get('confidence', 0.5)
                elif method == 'openai':
                    label = result.get('overall_sentiment', 'neutral')
                    confidence = result.get('confidence_score', 0.5)
                else:
                    continue
                
                valid_results.append({
                    'method': method,
                    'sentiment': label,
                    'confidence': confidence
                })
        
        if not valid_results:
            return {'sentiment': 'neutral', 'confidence': 0.5}
        
        # Weight by confidence and combine
        weighted_scores = {'positive': 0, 'negative': 0, 'neutral': 0}
        total_weight = 0
        
        for result in valid_results:
            sentiment = result['sentiment']
            confidence = result['confidence']
            
            # Method weights
            method_weight = {
                'finbert': 0.4,
                'openai': 0.4, 
                'keyword': 0.2
            }.get(result['method'], 0.2)
            
            weight = confidence * method_weight
            weighted_scores[sentiment] += weight
            total_weight += weight
        
        # Normalize weights
        if total_weight > 0:
            for sentiment in weighted_scores:
                weighted_scores[sentiment] /= total_weight
        
        # Determine final sentiment
        final_sentiment = max(weighted_scores, key=weighted_scores.get)
        final_confidence = weighted_scores[final_sentiment]
        
        return {
            'sentiment': final_sentiment,
            'confidence': final_confidence,
            'sentiment_scores': weighted_scores,
            'num_methods': len(valid_results),
            'method_agreement': self._calculate_agreement(valid_results)
        }
    
    def _calculate_agreement(self, results: List[Dict[str, Any]]) -> float:
        """Calculate agreement between methods."""
        if len(results) < 2:
            return 1.0
        
        sentiments = [r['sentiment'] for r in results]
        most_common = max(set(sentiments), key=sentiments.count)
        agreement = sentiments.count(most_common) / len(sentiments)
        
        return agreement
    
    def analyze_articles_sentiment(self, 
                                  articles: List[Dict[str, Any]],
                                  text_field: str = 'content',
                                  batch_size: int = 10) -> pd.DataFrame:
        """
        Analyze sentiment for a list of articles.
        
        Args:
            articles: List of article dictionaries
            text_field: Field containing text to analyze
            batch_size: Number of articles to process at once
        
        Returns:
            DataFrame with sentiment analysis results
        """
        results = []
        
        for i, article in enumerate(articles):
            try:
                # Extract text
                text = article.get(text_field, '')
                if not text:
                    # Fallback to title + description
                    text = (article.get('title', '') + ' ' + 
                           article.get('description', '')).strip()
                
                if not text:
                    logger.warning(f"No text found for article {i}")
                    continue
                
                # Analyze sentiment
                sentiment_result = self.analyze_text_sentiment(text)
                
                # Combine with article metadata
                result = {
                    'article_index': i,
                    'title': article.get('title', ''),
                    'source': article.get('source', ''),
                    'published_at': article.get('published_at', ''),
                    'text_length': len(text),
                    'sentiment': sentiment_result['combined_result'].get('sentiment', 'neutral'),
                    'confidence': sentiment_result['combined_result'].get('confidence', 0.5),
                    'sentiment_scores': sentiment_result['combined_result'].get('sentiment_scores', {}),
                    'method_agreement': sentiment_result['combined_result'].get('method_agreement', 0.0),
                    'methods_used': sentiment_result.get('methods_used', [])
                }
                
                # Add individual method results
                for method, method_result in sentiment_result.get('individual_results', {}).items():
                    if 'error' not in method_result:
                        if method == 'keyword':
                            result[f'{method}_positive_matches'] = method_result.get('positive_matches', 0)
                            result[f'{method}_negative_matches'] = method_result.get('negative_matches', 0)
                
                results.append(result)
                
                # Progress logging
                if (i + 1) % batch_size == 0:
                    logger.info(f"Processed {i + 1}/{len(articles)} articles")
                
            except Exception as e:
                logger.error(f"Error processing article {i}: {e}")
                continue
        
        logger.info(f"Completed sentiment analysis for {len(results)} articles")
        return pd.DataFrame(results)
    
    def calculate_sentiment_metrics(self, sentiment_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate aggregate sentiment metrics.
        
        Args:
            sentiment_df: DataFrame with sentiment analysis results
        
        Returns:
            Aggregate sentiment metrics
        """
        if len(sentiment_df) == 0:
            return {}
        
        # Overall sentiment distribution
        sentiment_counts = sentiment_df['sentiment'].value_counts()
        sentiment_percentages = sentiment_df['sentiment'].value_counts(normalize=True) * 100
        
        # Weighted sentiment score (-1 to 1)
        sentiment_weights = {'negative': -1, 'neutral': 0, 'positive': 1}
        weighted_scores = sentiment_df['sentiment'].map(sentiment_weights) * sentiment_df['confidence']
        overall_sentiment_score = weighted_scores.mean()
        
        # Confidence metrics
        avg_confidence = sentiment_df['confidence'].mean()
        confidence_by_sentiment = sentiment_df.groupby('sentiment')['confidence'].mean()
        
        # Time-based analysis if timestamps available
        time_analysis = {}
        if 'published_at' in sentiment_df.columns:
            sentiment_df['published_at'] = pd.to_datetime(sentiment_df['published_at'], errors='coerce')
            recent_24h = sentiment_df[sentiment_df['published_at'] >= datetime.now() - pd.Timedelta(days=1)]
            
            if len(recent_24h) > 0:
                recent_sentiment_score = (
                    recent_24h['sentiment'].map(sentiment_weights) * recent_24h['confidence']
                ).mean()
                time_analysis['recent_24h_sentiment_score'] = recent_sentiment_score
                time_analysis['recent_24h_article_count'] = len(recent_24h)
        
        return {
            'total_articles': len(sentiment_df),
            'sentiment_distribution': sentiment_counts.to_dict(),
            'sentiment_percentages': sentiment_percentages.to_dict(),
            'overall_sentiment_score': overall_sentiment_score,
            'average_confidence': avg_confidence,
            'confidence_by_sentiment': confidence_by_sentiment.to_dict(),
            'high_confidence_articles': len(sentiment_df[sentiment_df['confidence'] > 0.8]),
            'method_agreement_avg': sentiment_df['method_agreement'].mean(),
            'time_analysis': time_analysis
        }


if __name__ == "__main__":
    # Example usage
    analyzer = EconomicSentimentAnalyzer()
    
    # Test with sample economic texts
    sample_texts = [
        "The economy is showing strong growth with rising employment and robust consumer confidence.",
        "Market uncertainty and declining GDP raise concerns about potential recession risks.",
        "The Federal Reserve maintains interest rates while monitoring inflation trends."
    ]
    
    for i, text in enumerate(sample_texts):
        result = analyzer.analyze_text_sentiment(text)
        print(f"Text {i+1}: {result['combined_result']['sentiment']} "
              f"(confidence: {result['combined_result']['confidence']:.3f})")
    
    # Test with sample articles
    sample_articles = [
        {
            'title': 'GDP Growth Exceeds Expectations',
            'content': 'Economic data released today shows GDP growth of 3.2%, exceeding analyst forecasts and indicating a strong economic recovery.',
            'source': 'Financial Times',
            'published_at': '2024-01-15T10:00:00Z'
        },
        {
            'title': 'Market Volatility Continues',
            'content': 'Stock markets continue to experience significant volatility amid concerns about global economic slowdown and trade tensions.',
            'source': 'Reuters',
            'published_at': '2024-01-15T12:00:00Z'
        }
    ]
    
    # Analyze articles
    sentiment_df = analyzer.analyze_articles_sentiment(sample_articles)
    print(f"\nAnalyzed {len(sentiment_df)} articles")
    print(sentiment_df[['title', 'sentiment', 'confidence']].to_string())
    
    # Calculate metrics
    metrics = analyzer.calculate_sentiment_metrics(sentiment_df)
    print(f"\nSentiment metrics: {metrics}")