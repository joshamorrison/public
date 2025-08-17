"""
News Data Client
Fetches financial and economic news from multiple sources for sentiment analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union
import logging
from datetime import datetime, timedelta
import requests
import time
import os
from urllib.parse import urlencode

try:
    from newsapi import NewsApiClient
    NEWSAPI_AVAILABLE = True
except ImportError:
    NEWSAPI_AVAILABLE = False
    logging.warning("NewsAPI not available. Install with: pip install newsapi-python")

try:
    import feedparser
    FEEDPARSER_AVAILABLE = True
except ImportError:
    FEEDPARSER_AVAILABLE = False
    logging.warning("Feedparser not available. Install with: pip install feedparser")

try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False
    logging.warning("BeautifulSoup not available. Install with: pip install beautifulsoup4")

logger = logging.getLogger(__name__)


class NewsClient:
    """Client for fetching financial and economic news from multiple sources."""
    
    def __init__(self, 
                 newsapi_key: Optional[str] = None,
                 rate_limit_delay: float = 1.0):
        """
        Initialize news client.
        
        Args:
            newsapi_key: NewsAPI key (or use NEWSAPI_KEY env var)
            rate_limit_delay: Delay between API calls in seconds
        """
        self.rate_limit_delay = rate_limit_delay
        
        # Initialize NewsAPI client if available
        self.newsapi_key = newsapi_key or os.getenv('NEWSAPI_KEY')
        if self.newsapi_key and NEWSAPI_AVAILABLE:
            self.newsapi = NewsApiClient(api_key=self.newsapi_key)
            logger.info("NewsAPI client initialized")
        else:
            self.newsapi = None
            logger.warning("NewsAPI not available")
        
        # RSS feeds for financial news
        self.rss_feeds = {
            'reuters_business': 'https://feeds.reuters.com/reuters/businessNews',
            'reuters_economy': 'https://feeds.reuters.com/reuters/economicNews',
            'bloomberg_economics': 'https://feeds.bloomberg.com/economics/news.rss',
            'ft_economics': 'https://www.ft.com/rss/world/economies',
            'wsj_economy': 'https://feeds.wsj.com/public/resources/documents/economy.xml',
            'cnbc_economy': 'https://www.cnbc.com/id/20910258/device/rss/rss.html',
            'marketwatch_economy': 'http://feeds.marketwatch.com/marketwatch/economy/'
        }
        
        # Economic keywords for filtering
        self.economic_keywords = [
            'gdp', 'inflation', 'unemployment', 'interest rate', 'federal reserve',
            'central bank', 'monetary policy', 'fiscal policy', 'recession',
            'economic growth', 'consumer confidence', 'retail sales', 
            'housing market', 'employment', 'productivity', 'trade deficit',
            'dollar', 'currency', 'bond yield', 'stock market', 'treasury'
        ]
        
        logger.info("News client initialized with RSS feeds and economic keywords")
    
    def fetch_news_articles(self, 
                           query: str = "economy OR GDP OR inflation OR unemployment",
                           sources: Optional[List[str]] = None,
                           from_date: Optional[datetime] = None,
                           to_date: Optional[datetime] = None,
                           language: str = 'en',
                           max_articles: int = 100) -> List[Dict[str, Any]]:
        """
        Fetch news articles from NewsAPI.
        
        Args:
            query: Search query
            sources: News sources to search
            from_date: Start date for articles
            to_date: End date for articles
            language: Article language
            max_articles: Maximum number of articles
        
        Returns:
            List of article dictionaries
        """
        if not self.newsapi:
            logger.warning("NewsAPI not available")
            return []
        
        try:
            # Set default date range
            if not from_date:
                from_date = datetime.now() - timedelta(days=7)
            if not to_date:
                to_date = datetime.now()
            
            # Financial news sources
            if not sources:
                sources = [
                    'reuters', 'bloomberg', 'financial-times', 'wall-street-journal',
                    'cnbc', 'marketwatch', 'business-insider', 'the-economist'
                ]
            
            articles = []
            
            # Fetch from everything endpoint for broader coverage
            everything_response = self.newsapi.get_everything(
                q=query,
                sources=','.join(sources) if sources else None,
                from_param=from_date.strftime('%Y-%m-%d'),
                to=to_date.strftime('%Y-%m-%d'),
                language=language,
                sort_by='relevancy',
                page_size=min(max_articles, 100)
            )
            
            if everything_response['status'] == 'ok':
                for article in everything_response['articles']:
                    processed_article = {
                        'title': article.get('title', ''),
                        'description': article.get('description', ''),
                        'content': article.get('content', ''),
                        'url': article.get('url', ''),
                        'source': article.get('source', {}).get('name', ''),
                        'published_at': article.get('publishedAt', ''),
                        'author': article.get('author', ''),
                        'fetched_at': datetime.now().isoformat()
                    }
                    articles.append(processed_article)
            
            # Rate limiting
            time.sleep(self.rate_limit_delay)
            
            logger.info(f"Fetched {len(articles)} articles from NewsAPI")
            return articles
            
        except Exception as e:
            logger.error(f"Error fetching news articles: {e}")
            return []
    
    def fetch_rss_feeds(self, 
                       feeds: Optional[List[str]] = None,
                       max_articles_per_feed: int = 50) -> List[Dict[str, Any]]:
        """
        Fetch articles from RSS feeds.
        
        Args:
            feeds: List of feed names to fetch (all if None)
            max_articles_per_feed: Maximum articles per feed
        
        Returns:
            List of article dictionaries
        """
        if not FEEDPARSER_AVAILABLE:
            logger.warning("Feedparser not available")
            return []
        
        if feeds is None:
            feeds = list(self.rss_feeds.keys())
        
        all_articles = []
        
        for feed_name in feeds:
            if feed_name not in self.rss_feeds:
                logger.warning(f"Unknown feed: {feed_name}")
                continue
            
            try:
                feed_url = self.rss_feeds[feed_name]
                logger.info(f"Fetching from {feed_name}: {feed_url}")
                
                # Parse RSS feed
                feed = feedparser.parse(feed_url)
                
                for entry in feed.entries[:max_articles_per_feed]:
                    article = {
                        'title': entry.get('title', ''),
                        'description': entry.get('summary', ''),
                        'content': entry.get('content', [{}])[0].get('value', '') if entry.get('content') else '',
                        'url': entry.get('link', ''),
                        'source': feed_name,
                        'published_at': entry.get('published', ''),
                        'author': entry.get('author', ''),
                        'fetched_at': datetime.now().isoformat()
                    }
                    all_articles.append(article)
                
                # Rate limiting
                time.sleep(self.rate_limit_delay)
                
            except Exception as e:
                logger.error(f"Error fetching from {feed_name}: {e}")
                continue
        
        logger.info(f"Fetched {len(all_articles)} articles from RSS feeds")
        return all_articles
    
    def filter_economic_articles(self, 
                                articles: List[Dict[str, Any]],
                                min_relevance_score: float = 0.3) -> List[Dict[str, Any]]:
        """
        Filter articles for economic relevance.
        
        Args:
            articles: List of articles
            min_relevance_score: Minimum relevance score
        
        Returns:
            Filtered articles with relevance scores
        """
        filtered_articles = []
        
        for article in articles:
            # Calculate relevance score
            text_content = (
                article.get('title', '') + ' ' + 
                article.get('description', '') + ' ' + 
                article.get('content', '')
            ).lower()
            
            keyword_matches = 0
            for keyword in self.economic_keywords:
                if keyword.lower() in text_content:
                    keyword_matches += 1
            
            relevance_score = keyword_matches / len(self.economic_keywords)
            
            if relevance_score >= min_relevance_score:
                article['economic_relevance_score'] = relevance_score
                article['matched_keywords'] = [
                    kw for kw in self.economic_keywords 
                    if kw.lower() in text_content
                ]
                filtered_articles.append(article)
        
        # Sort by relevance score
        filtered_articles.sort(key=lambda x: x['economic_relevance_score'], reverse=True)
        
        logger.info(f"Filtered to {len(filtered_articles)} economically relevant articles")
        return filtered_articles
    
    def get_latest_economic_news(self, 
                                hours_back: int = 24,
                                max_articles: int = 100) -> pd.DataFrame:
        """
        Get latest economic news as DataFrame.
        
        Args:
            hours_back: How many hours back to search
            max_articles: Maximum number of articles
        
        Returns:
            DataFrame with economic news articles
        """
        # Set time range
        to_date = datetime.now()
        from_date = to_date - timedelta(hours=hours_back)
        
        # Fetch from multiple sources
        articles = []
        
        # Try NewsAPI first
        if self.newsapi:
            api_articles = self.fetch_news_articles(
                query="(economy OR economic OR GDP OR inflation OR unemployment OR 'federal reserve' OR 'central bank')",
                from_date=from_date,
                to_date=to_date,
                max_articles=max_articles // 2
            )
            articles.extend(api_articles)
        
        # Fetch from RSS feeds
        rss_articles = self.fetch_rss_feeds(max_articles_per_feed=10)
        
        # Filter RSS articles by date
        recent_rss = []
        for article in rss_articles:
            try:
                if article.get('published_at'):
                    pub_date = pd.to_datetime(article['published_at'])
                    if pub_date >= from_date:
                        recent_rss.append(article)
            except:
                continue
        
        articles.extend(recent_rss)
        
        # Filter for economic relevance
        economic_articles = self.filter_economic_articles(articles)
        
        # Convert to DataFrame
        if economic_articles:
            df = pd.DataFrame(economic_articles)
            
            # Clean and standardize
            df['published_at'] = pd.to_datetime(df['published_at'], errors='coerce')
            df = df.sort_values('published_at', ascending=False)
            df = df.drop_duplicates(subset=['title', 'source'])
            df = df.head(max_articles)
            
            logger.info(f"Created DataFrame with {len(df)} economic news articles")
            return df
        else:
            logger.warning("No economic articles found")
            return pd.DataFrame()
    
    def search_economic_topics(self, 
                              topics: List[str],
                              days_back: int = 30) -> Dict[str, List[Dict[str, Any]]]:
        """
        Search for specific economic topics.
        
        Args:
            topics: List of economic topics to search
            days_back: Number of days to search back
        
        Returns:
            Dictionary of articles by topic
        """
        results = {}
        
        for topic in topics:
            logger.info(f"Searching for topic: {topic}")
            
            # Search NewsAPI
            topic_articles = []
            if self.newsapi:
                api_articles = self.fetch_news_articles(
                    query=f'"{topic}" OR {topic}',
                    from_date=datetime.now() - timedelta(days=days_back),
                    max_articles=50
                )
                topic_articles.extend(api_articles)
            
            # Search RSS feeds
            rss_articles = self.fetch_rss_feeds()
            
            # Filter for topic relevance
            relevant_articles = []
            for article in rss_articles + topic_articles:
                text_content = (
                    article.get('title', '') + ' ' + 
                    article.get('description', '') + ' ' + 
                    article.get('content', '')
                ).lower()
                
                if topic.lower() in text_content:
                    article['topic_relevance'] = text_content.count(topic.lower())
                    relevant_articles.append(article)
            
            # Sort by relevance and remove duplicates
            relevant_articles.sort(key=lambda x: x.get('topic_relevance', 0), reverse=True)
            
            # Remove duplicates by title
            seen_titles = set()
            unique_articles = []
            for article in relevant_articles:
                title = article.get('title', '')
                if title not in seen_titles:
                    seen_titles.add(title)
                    unique_articles.append(article)
            
            results[topic] = unique_articles[:20]  # Top 20 per topic
            logger.info(f"Found {len(results[topic])} articles for {topic}")
        
        return results
    
    def export_articles(self, 
                       articles: List[Dict[str, Any]], 
                       filename: str = None) -> str:
        """
        Export articles to JSON file.
        
        Args:
            articles: Articles to export
            filename: Output filename
        
        Returns:
            Path to exported file
        """
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"economic_news_{timestamp}.json"
        
        import json
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(articles, f, indent=2, default=str, ensure_ascii=False)
            
            logger.info(f"Exported {len(articles)} articles to {filename}")
            return filename
            
        except Exception as e:
            logger.error(f"Error exporting articles: {e}")
            return ""
    
    def get_news_summary_stats(self, articles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get summary statistics for fetched articles."""
        if not articles:
            return {}
        
        df = pd.DataFrame(articles)
        
        stats = {
            'total_articles': len(articles),
            'unique_sources': df['source'].nunique() if 'source' in df else 0,
            'date_range': {
                'earliest': df['published_at'].min() if 'published_at' in df else None,
                'latest': df['published_at'].max() if 'published_at' in df else None
            },
            'avg_relevance_score': df['economic_relevance_score'].mean() if 'economic_relevance_score' in df else 0,
            'sources_breakdown': df['source'].value_counts().to_dict() if 'source' in df else {},
            'top_keywords': {}
        }
        
        # Count keyword frequencies
        if 'matched_keywords' in df:
            all_keywords = []
            for keywords in df['matched_keywords'].dropna():
                if isinstance(keywords, list):
                    all_keywords.extend(keywords)
            
            from collections import Counter
            keyword_counts = Counter(all_keywords)
            stats['top_keywords'] = dict(keyword_counts.most_common(10))
        
        return stats


if __name__ == "__main__":
    # Example usage
    client = NewsClient()
    
    # Get latest economic news
    news_df = client.get_latest_economic_news(hours_back=48, max_articles=50)
    print(f"Fetched {len(news_df)} recent economic articles")
    
    if len(news_df) > 0:
        print(f"Top sources: {news_df['source'].value_counts().head()}")
        print(f"Average relevance: {news_df['economic_relevance_score'].mean():.3f}")
    
    # Search for specific topics
    topics = ['inflation', 'unemployment', 'GDP']
    topic_results = client.search_economic_topics(topics, days_back=7)
    
    for topic, articles in topic_results.items():
        print(f"{topic}: {len(articles)} articles")
    
    # Get summary statistics
    all_articles = []
    for articles in topic_results.values():
        all_articles.extend(articles)
    
    if all_articles:
        stats = client.get_news_summary_stats(all_articles)
        print(f"Summary stats: {stats}")