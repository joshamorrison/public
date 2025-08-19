"""
Document Processor Tool

Provides document processing capabilities for agents including text extraction,
analysis, summarization, and content transformation.
"""

import asyncio
import re
from typing import Dict, List, Any, Optional
from datetime import datetime
import json


class DocumentProcessorTool:
    """
    Document processing tool that provides text analysis and processing capabilities.
    
    Features:
    - Text extraction and cleaning
    - Document summarization
    - Keyword extraction
    - Content analysis and scoring
    - Format conversion capabilities
    """
    
    def __init__(self, max_length: int = 100000):
        """
        Initialize document processor.
        
        Args:
            max_length: Maximum document length to process
        """
        self.max_length = max_length
    
    async def process_text(self, text: str, operation: str = "analyze") -> Dict[str, Any]:
        """
        Process text content with specified operation.
        
        Args:
            text: Input text to process
            operation: Type of processing (analyze, summarize, extract_keywords, etc.)
            
        Returns:
            Processing results with metadata
        """
        if len(text) > self.max_length:
            text = text[:self.max_length]
        
        start_time = datetime.now()
        
        try:
            if operation == "analyze":
                result = await self._analyze_text(text)
            elif operation == "summarize":
                result = await self._summarize_text(text)
            elif operation == "extract_keywords":
                result = await self._extract_keywords(text)
            elif operation == "clean":
                result = await self._clean_text(text)
            elif operation == "classify":
                result = await self._classify_content(text)
            else:
                result = {"error": f"Unknown operation: {operation}"}
            
            # Add processing metadata
            result.update({
                "operation": operation,
                "input_length": len(text),
                "processing_time": (datetime.now() - start_time).total_seconds(),
                "processed_at": datetime.now().isoformat(),
                "tool": "DocumentProcessorTool"
            })
            
            return result
            
        except Exception as e:
            return {
                "error": str(e),
                "operation": operation,
                "processed_at": datetime.now().isoformat()
            }
    
    async def _analyze_text(self, text: str) -> Dict[str, Any]:
        """Analyze text content for various metrics."""
        # Simulate processing delay
        await asyncio.sleep(0.2)
        
        # Basic text analysis
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        paragraphs = text.split('\n\n')
        
        # Calculate readability metrics (simplified)
        avg_words_per_sentence = len(words) / max(len(sentences), 1)
        avg_chars_per_word = sum(len(word) for word in words) / max(len(words), 1)
        
        # Content scoring (simplified)
        complexity_score = min(avg_words_per_sentence / 20 * avg_chars_per_word / 5, 1.0)
        readability_score = max(1.0 - complexity_score, 0.0)
        
        return {
            "word_count": len(words),
            "sentence_count": len([s for s in sentences if s.strip()]),
            "paragraph_count": len([p for p in paragraphs if p.strip()]),
            "character_count": len(text),
            "avg_words_per_sentence": round(avg_words_per_sentence, 2),
            "avg_chars_per_word": round(avg_chars_per_word, 2),
            "readability_score": round(readability_score, 2),
            "complexity_score": round(complexity_score, 2),
            "estimated_reading_time_minutes": round(len(words) / 200, 1)  # Assume 200 WPM
        }
    
    async def _summarize_text(self, text: str, max_sentences: int = 3) -> Dict[str, Any]:
        """Generate a summary of the text content."""
        await asyncio.sleep(0.3)
        
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) <= max_sentences:
            summary = text
        else:
            # Simple extractive summarization (take first, middle, and last sentences)
            selected_indices = [0]
            if len(sentences) > 2:
                selected_indices.append(len(sentences) // 2)
            if len(sentences) > 1:
                selected_indices.append(len(sentences) - 1)
            
            selected_indices = list(set(selected_indices))[:max_sentences]
            summary = '. '.join([sentences[i] for i in sorted(selected_indices)]) + '.'
        
        return {
            "summary": summary,
            "original_sentence_count": len(sentences),
            "summary_sentence_count": len(re.split(r'[.!?]+', summary)),
            "compression_ratio": round(len(summary) / len(text), 2),
            "summary_type": "extractive"
        }
    
    async def _extract_keywords(self, text: str, max_keywords: int = 10) -> Dict[str, Any]:
        """Extract keywords and key phrases from text."""
        await asyncio.sleep(0.2)
        
        # Simple keyword extraction (frequency-based)
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        
        # Remove common stop words
        stop_words = {
            'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 
            'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his',
            'how', 'its', 'may', 'new', 'now', 'old', 'see', 'two', 'who', 'boy',
            'did', 'she', 'use', 'way', 'with', 'this', 'that', 'they', 'have',
            'from', 'they', 'know', 'want', 'been', 'good', 'much', 'some', 'time',
            'very', 'when', 'come', 'here', 'just', 'like', 'long', 'make', 'many',
            'over', 'such', 'take', 'than', 'them', 'well', 'were'
        }
        
        filtered_words = [word for word in words if word not in stop_words]
        
        # Count word frequencies
        word_freq = {}
        for word in filtered_words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # Sort by frequency and take top keywords
        keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:max_keywords]
        
        return {
            "keywords": [{"word": word, "frequency": freq, "score": round(freq/len(filtered_words), 3)} 
                        for word, freq in keywords],
            "total_words_analyzed": len(filtered_words),
            "unique_words": len(word_freq),
            "keyword_count": len(keywords)
        }
    
    async def _clean_text(self, text: str) -> Dict[str, Any]:
        """Clean and normalize text content."""
        await asyncio.sleep(0.1)
        
        original_length = len(text)
        
        # Basic text cleaning
        cleaned = text
        
        # Remove extra whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned)
        
        # Remove special characters (keep basic punctuation)
        cleaned = re.sub(r'[^\w\s.,!?;:\-()"\']', '', cleaned)
        
        # Remove excessive punctuation
        cleaned = re.sub(r'[.]{2,}', '...', cleaned)
        cleaned = re.sub(r'[!]{2,}', '!', cleaned)
        cleaned = re.sub(r'[?]{2,}', '?', cleaned)
        
        # Trim whitespace
        cleaned = cleaned.strip()
        
        return {
            "cleaned_text": cleaned,
            "original_length": original_length,
            "cleaned_length": len(cleaned),
            "characters_removed": original_length - len(cleaned),
            "cleaning_ratio": round((original_length - len(cleaned)) / original_length, 3)
        }
    
    async def _classify_content(self, text: str) -> Dict[str, Any]:
        """Classify content type and tone."""
        await asyncio.sleep(0.2)
        
        text_lower = text.lower()
        
        # Simple content type classification
        content_indicators = {
            "technical": ["implementation", "algorithm", "system", "method", "process", "framework"],
            "business": ["strategy", "market", "revenue", "customer", "growth", "profit"],
            "academic": ["research", "study", "analysis", "findings", "methodology", "conclusion"],
            "news": ["reported", "according", "sources", "breaking", "update", "latest"]
        }
        
        type_scores = {}
        for content_type, indicators in content_indicators.items():
            score = sum(1 for indicator in indicators if indicator in text_lower)
            type_scores[content_type] = score
        
        # Determine primary content type
        primary_type = max(type_scores, key=type_scores.get) if type_scores else "general"
        
        # Simple tone analysis
        positive_words = ["good", "great", "excellent", "success", "improve", "benefit", "advantage"]
        negative_words = ["bad", "poor", "fail", "problem", "issue", "challenge", "difficult"]
        
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count > negative_count:
            tone = "positive"
        elif negative_count > positive_count:
            tone = "negative"
        else:
            tone = "neutral"
        
        return {
            "content_type": primary_type,
            "type_confidence": round(type_scores.get(primary_type, 0) / max(sum(type_scores.values()), 1), 2),
            "tone": tone,
            "sentiment_indicators": {
                "positive_words": positive_count,
                "negative_words": negative_count
            },
            "type_scores": type_scores
        }


# Standalone async functions for simple usage
async def analyze_document(text: str) -> Dict[str, Any]:
    """Simple document analysis function for agent use."""
    processor = DocumentProcessorTool()
    return await processor.process_text(text, "analyze")

async def summarize_document(text: str) -> Dict[str, Any]:
    """Simple document summarization function for agent use."""
    processor = DocumentProcessorTool()
    return await processor.process_text(text, "summarize")

async def extract_keywords(text: str) -> Dict[str, Any]:
    """Simple keyword extraction function for agent use."""
    processor = DocumentProcessorTool()
    return await processor.process_text(text, "extract_keywords")