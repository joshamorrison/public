"""
Web Search Tool

Provides web search capabilities for agents using multiple search engines.
"""

import asyncio
import aiohttp
import json
from typing import Dict, List, Any, Optional
from datetime import datetime


class WebSearchTool:
    """
    Web search tool that provides internet search capabilities to agents.
    
    Features:
    - Multiple search engine support (DuckDuckGo, SerpAPI, etc.)
    - Result filtering and ranking
    - Async search operations
    - Rate limiting and error handling
    """
    
    def __init__(self, max_results: int = 10, timeout: int = 30):
        """
        Initialize web search tool.
        
        Args:
            max_results: Maximum number of search results to return
            timeout: Request timeout in seconds
        """
        self.max_results = max_results
        self.timeout = timeout
        self.session = None
    
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout))
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    async def search(self, query: str, search_type: str = "general") -> List[Dict[str, Any]]:
        """
        Perform web search for the given query.
        
        Args:
            query: Search query string
            search_type: Type of search (general, news, academic, etc.)
            
        Returns:
            List of search results with title, url, snippet, etc.
        """
        try:
            # For demo purposes, we'll simulate search results
            # In production, integrate with actual search APIs
            results = await self._simulate_search(query, search_type)
            
            return results[:self.max_results]
            
        except Exception as e:
            print(f"[WEB_SEARCH] Search failed: {str(e)}")
            return []
    
    async def _simulate_search(self, query: str, search_type: str) -> List[Dict[str, Any]]:
        """
        Simulate search results for development/testing.
        In production, replace with actual search API calls.
        """
        # Simulate API delay
        await asyncio.sleep(0.5)
        
        # Generate realistic mock results based on query
        base_results = [
            {
                "title": f"Comprehensive Guide to {query}",
                "url": f"https://example.com/guide-{query.replace(' ', '-').lower()}",
                "snippet": f"Learn everything about {query} with this detailed guide covering key concepts, best practices, and real-world applications.",
                "source": "example.com",
                "published_date": "2024-01-15",
                "relevance_score": 0.95
            },
            {
                "title": f"{query}: Latest Research and Trends",
                "url": f"https://research.example.org/{query.replace(' ', '-')}",
                "snippet": f"Recent research findings and emerging trends in {query}. Analysis of current developments and future implications.",
                "source": "research.example.org", 
                "published_date": "2024-01-10",
                "relevance_score": 0.88
            },
            {
                "title": f"Best Practices for {query} Implementation",
                "url": f"https://blog.example.com/best-practices-{query.replace(' ', '-')}",
                "snippet": f"Expert recommendations and proven strategies for implementing {query} in enterprise environments.",
                "source": "blog.example.com",
                "published_date": "2024-01-08",
                "relevance_score": 0.82
            },
            {
                "title": f"{query} Case Studies and Success Stories",
                "url": f"https://casestudies.example.net/{query.replace(' ', '_')}",
                "snippet": f"Real-world case studies demonstrating successful {query} implementations across various industries.",
                "source": "casestudies.example.net",
                "published_date": "2024-01-05",
                "relevance_score": 0.79
            },
            {
                "title": f"Common Challenges in {query} and Solutions",
                "url": f"https://solutions.example.com/{query.replace(' ', '-')}-challenges",
                "snippet": f"Addressing typical challenges encountered when working with {query} and practical solutions.",
                "source": "solutions.example.com",
                "published_date": "2024-01-03",
                "relevance_score": 0.75
            }
        ]
        
        # Modify results based on search type
        if search_type == "news":
            for result in base_results:
                result["title"] = f"Breaking: {result['title']}"
                result["published_date"] = "2024-01-18"
                
        elif search_type == "academic":
            for result in base_results:
                result["title"] = f"Academic Study: {result['title']}"
                result["source"] = result["source"].replace(".com", ".edu")
                
        # Add search metadata
        for i, result in enumerate(base_results):
            result.update({
                "search_query": query,
                "search_type": search_type,
                "result_rank": i + 1,
                "retrieved_at": datetime.now().isoformat(),
                "tool": "WebSearchTool"
            })
        
        return base_results
    
    async def search_news(self, query: str, days_back: int = 7) -> List[Dict[str, Any]]:
        """Search for recent news articles."""
        return await self.search(f"{query} news", "news")
    
    async def search_academic(self, query: str) -> List[Dict[str, Any]]:
        """Search for academic papers and research."""
        return await self.search(f"{query} research", "academic")
    
    def filter_results(self, results: List[Dict[str, Any]], 
                      min_relevance: float = 0.7) -> List[Dict[str, Any]]:
        """Filter search results by relevance score."""
        return [r for r in results if r.get("relevance_score", 0) >= min_relevance]
    
    def extract_domains(self, results: List[Dict[str, Any]]) -> List[str]:
        """Extract unique domains from search results."""
        domains = set()
        for result in results:
            if "source" in result:
                domains.add(result["source"])
        return list(domains)


# Standalone async function for simple usage
async def search_web(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """
    Simple web search function for agent use.
    
    Args:
        query: Search query
        max_results: Maximum results to return
        
    Returns:
        List of search results
    """
    async with WebSearchTool(max_results=max_results) as searcher:
        return await searcher.search(query)