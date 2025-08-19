"""
Unit Tests for Tools

Tests all tool functionality including web search, document processing,
and calculation engine capabilities.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock

from src.tools.web_search import WebSearchTool, search_web
from src.tools.document_processor import DocumentProcessorTool, analyze_document, summarize_document, extract_keywords
from src.tools.calculation_engine import CalculationEngineTool, calculate_stats, evaluate_math, compound_interest


class TestWebSearchTool:
    """Test cases for WebSearchTool."""
    
    @pytest.fixture
    def search_tool(self):
        """Create a web search tool for testing."""
        return WebSearchTool(max_results=5, timeout=10)
    
    @pytest.mark.asyncio
    async def test_search_tool_context_manager(self, search_tool):
        """Test async context manager functionality."""
        async with search_tool as tool:
            assert tool.session is not None
        # Session should be closed after context exit
    
    @pytest.mark.asyncio
    async def test_basic_search(self, search_tool):
        """Test basic search functionality."""
        async with search_tool as tool:
            results = await tool.search("machine learning", "general")
            
            assert isinstance(results, list)
            assert len(results) <= 5  # max_results limit
            
            if results:
                result = results[0]
                assert "title" in result
                assert "url" in result
                assert "snippet" in result
                assert "source" in result
                assert "search_query" in result
                assert result["search_query"] == "machine learning"
    
    @pytest.mark.asyncio
    async def test_news_search(self, search_tool):
        """Test news search functionality."""
        async with search_tool as tool:
            results = await tool.search_news("AI developments")
            
            assert isinstance(results, list)
            if results:
                result = results[0]
                assert "Breaking:" in result["title"]
                assert result["search_type"] == "news"
    
    @pytest.mark.asyncio
    async def test_academic_search(self, search_tool):
        """Test academic search functionality."""
        async with search_tool as tool:
            results = await tool.search_academic("neural networks")
            
            assert isinstance(results, list)
            if results:
                result = results[0]
                assert "Academic Study:" in result["title"]
                assert ".edu" in result["source"]
                assert result["search_type"] == "academic"
    
    def test_filter_results(self, search_tool):
        """Test result filtering."""
        mock_results = [
            {"title": "High relevance", "relevance_score": 0.9},
            {"title": "Medium relevance", "relevance_score": 0.6},
            {"title": "Low relevance", "relevance_score": 0.3}
        ]
        
        filtered = search_tool.filter_results(mock_results, min_relevance=0.7)
        assert len(filtered) == 1
        assert filtered[0]["title"] == "High relevance"
    
    def test_extract_domains(self, search_tool):
        """Test domain extraction."""
        mock_results = [
            {"source": "example.com"},
            {"source": "test.org"},
            {"source": "example.com"},  # Duplicate
            {"source": "another.net"}
        ]
        
        domains = search_tool.extract_domains(mock_results)
        assert len(domains) == 3  # Unique domains
        assert "example.com" in domains
        assert "test.org" in domains
        assert "another.net" in domains
    
    @pytest.mark.asyncio
    async def test_standalone_search_function(self):
        """Test standalone search function."""
        results = await search_web("test query", max_results=3)
        
        assert isinstance(results, list)
        assert len(results) <= 3
        if results:
            assert "search_query" in results[0]
            assert results[0]["search_query"] == "test query"


class TestDocumentProcessorTool:
    """Test cases for DocumentProcessorTool."""
    
    @pytest.fixture
    def processor(self):
        """Create a document processor for testing."""
        return DocumentProcessorTool(max_length=10000)
    
    @pytest.fixture
    def sample_text(self):
        """Sample text for testing."""
        return """
        This is a sample document for testing the document processor tool.
        It contains multiple sentences and paragraphs to test various features.
        
        The document processor should be able to analyze this text and extract
        meaningful information such as word count, readability metrics, and keywords.
        
        This tool supports operations like summarization, keyword extraction,
        text cleaning, and content classification.
        """
    
    @pytest.mark.asyncio
    async def test_text_analysis(self, processor, sample_text):
        """Test text analysis functionality."""
        result = await processor.process_text(sample_text, "analyze")
        
        assert "operation" in result
        assert result["operation"] == "analyze"
        assert "word_count" in result
        assert "sentence_count" in result
        assert "paragraph_count" in result
        assert "readability_score" in result
        assert "complexity_score" in result
        assert "estimated_reading_time_minutes" in result
        
        assert result["word_count"] > 0
        assert result["sentence_count"] > 0
        assert result["paragraph_count"] > 0
        assert 0 <= result["readability_score"] <= 1
        assert 0 <= result["complexity_score"] <= 1
    
    @pytest.mark.asyncio
    async def test_text_summarization(self, processor, sample_text):
        """Test text summarization."""
        result = await processor.process_text(sample_text, "summarize")
        
        assert "operation" in result
        assert result["operation"] == "summarize"
        assert "summary" in result
        assert "original_sentence_count" in result
        assert "summary_sentence_count" in result
        assert "compression_ratio" in result
        assert "summary_type" in result
        
        assert len(result["summary"]) > 0
        assert result["compression_ratio"] > 0
        assert result["summary_type"] == "extractive"
    
    @pytest.mark.asyncio
    async def test_keyword_extraction(self, processor, sample_text):
        """Test keyword extraction."""
        result = await processor.process_text(sample_text, "extract_keywords")
        
        assert "operation" in result
        assert result["operation"] == "extract_keywords"
        assert "keywords" in result
        assert "total_words_analyzed" in result
        assert "unique_words" in result
        assert "keyword_count" in result
        
        assert isinstance(result["keywords"], list)
        if result["keywords"]:
            keyword = result["keywords"][0]
            assert "word" in keyword
            assert "frequency" in keyword
            assert "score" in keyword
    
    @pytest.mark.asyncio
    async def test_text_cleaning(self, processor):
        """Test text cleaning functionality."""
        dirty_text = "This   has    extra  spaces!!!  And weird___characters@#$"
        result = await processor.process_text(dirty_text, "clean")
        
        assert "operation" in result
        assert result["operation"] == "clean"
        assert "cleaned_text" in result
        assert "original_length" in result
        assert "cleaned_length" in result
        assert "characters_removed" in result
        assert "cleaning_ratio" in result
        
        cleaned = result["cleaned_text"]
        assert "  " not in cleaned  # No double spaces
        assert "!!!" not in cleaned  # Reduced punctuation
        assert len(cleaned) <= len(dirty_text)
    
    @pytest.mark.asyncio
    async def test_content_classification(self, processor):
        """Test content classification."""
        technical_text = "This algorithm implements a neural network framework using advanced machine learning methods."
        result = await processor.process_text(technical_text, "classify")
        
        assert "operation" in result
        assert result["operation"] == "classify"
        assert "content_type" in result
        assert "type_confidence" in result
        assert "tone" in result
        assert "sentiment_indicators" in result
        assert "type_scores" in result
        
        # Should classify as technical due to keywords
        assert result["content_type"] in ["technical", "general"]
        assert result["tone"] in ["positive", "negative", "neutral"]
    
    @pytest.mark.asyncio
    async def test_length_limiting(self):
        """Test document length limiting."""
        processor = DocumentProcessorTool(max_length=50)
        long_text = "word " * 100  # 500 characters
        
        result = await processor.process_text(long_text, "analyze")
        
        assert result["input_length"] == 50  # Should be truncated
    
    @pytest.mark.asyncio
    async def test_error_handling(self, processor):
        """Test error handling."""
        result = await processor.process_text("test", "unknown_operation")
        
        assert "error" in result
        assert "Unknown operation" in result["error"]
    
    @pytest.mark.asyncio
    async def test_standalone_functions(self, sample_text):
        """Test standalone function interfaces."""
        # Test analyze function
        analysis = await analyze_document(sample_text)
        assert "word_count" in analysis
        
        # Test summarize function  
        summary = await summarize_document(sample_text)
        assert "summary" in summary
        
        # Test keyword extraction
        keywords = await extract_keywords(sample_text)
        assert "keywords" in keywords


class TestCalculationEngineTool:
    """Test cases for CalculationEngineTool."""
    
    @pytest.fixture
    def calculator(self):
        """Create a calculation engine for testing."""
        return CalculationEngineTool()
    
    @pytest.mark.asyncio
    async def test_basic_arithmetic(self, calculator):
        """Test basic arithmetic operations."""
        data = {"type": "arithmetic", "a": 10, "b": 5, "operator": "+"}
        result = await calculator.calculate("basic_math", data)
        
        assert "result" in result
        assert result["result"] == 15
        assert result["expression"] == "10 + 5"
        assert result["operator"] == "+"
        
        # Test other operators
        operations = [
            (10, 5, "-", 5),
            (10, 5, "*", 50),
            (10, 5, "/", 2.0),
            (10, 3, "//", 3),
            (10, 3, "%", 1),
            (2, 3, "**", 8)
        ]
        
        for a, b, op, expected in operations:
            data = {"type": "arithmetic", "a": a, "b": b, "operator": op}
            result = await calculator.calculate("basic_math", data)
            assert result["result"] == expected
    
    @pytest.mark.asyncio
    async def test_advanced_math(self, calculator):
        """Test advanced mathematical functions."""
        import math
        
        functions = [
            ("sqrt", 16, 4.0),
            ("abs", -5, 5),
            ("ceil", 4.3, 5),
            ("floor", 4.7, 4)
        ]
        
        for func, value, expected in functions:
            data = {"type": "advanced", "function": func, "value": value}
            result = await calculator.calculate("basic_math", data)
            assert result["result"] == expected
    
    @pytest.mark.asyncio
    async def test_statistics(self, calculator):
        """Test statistical calculations."""
        numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        result = await calculator.calculate("statistics", numbers)
        
        assert "count" in result
        assert "sum" in result
        assert "mean" in result
        assert "median" in result
        assert "std_deviation" in result
        assert "variance" in result
        assert "min" in result
        assert "max" in result
        assert "range" in result
        
        assert result["count"] == 10
        assert result["sum"] == 55
        assert result["mean"] == 5.5
        assert result["median"] == 5.5
        assert result["min"] == 1
        assert result["max"] == 10
        assert result["range"] == 9
    
    @pytest.mark.asyncio
    async def test_financial_calculations(self, calculator):
        """Test financial calculations."""
        # Test compound interest
        data = {
            "type": "compound_interest",
            "principal": 1000,
            "rate": 0.05,
            "time": 2,
            "compound_frequency": 1
        }
        result = await calculator.calculate("financial", data)
        
        assert "final_amount" in result
        assert "interest_earned" in result
        assert "roi_percentage" in result
        assert result["principal"] == 1000
        assert result["rate"] == 0.05
        assert result["time"] == 2
        
        # Test loan payment calculation
        loan_data = {
            "type": "loan_payment",
            "principal": 100000,
            "annual_rate": 0.05,
            "years": 30
        }
        loan_result = await calculator.calculate("financial", loan_data)
        
        assert "monthly_payment" in loan_result
        assert "total_payments" in loan_result
        assert "total_interest" in loan_result
        assert "number_of_payments" in loan_result
    
    @pytest.mark.asyncio
    async def test_data_analysis(self, calculator):
        """Test data analysis capabilities."""
        data = [
            {"value": 10, "category": "A"},
            {"value": 20, "category": "B"}, 
            {"value": 30, "category": "A"},
            {"value": 40, "category": "B"}
        ]
        
        result = await calculator.calculate("data_analysis", data)
        
        assert "total_records" in result
        assert "numeric_columns" in result
        assert "column_summaries" in result
        assert result["total_records"] == 4
    
    @pytest.mark.asyncio
    async def test_expression_evaluation(self, calculator):
        """Test safe expression evaluation."""
        # Test safe expressions
        safe_expressions = [
            ("2 + 3 * 4", 14),
            ("(10 + 5) / 3", 5.0),
            ("abs(-10)", 10),
            ("max(1, 2, 3)", 3),
            ("min(1, 2, 3)", 1)
        ]
        
        for expr, expected in safe_expressions:
            result = await calculator.calculate("expression", expr)
            assert "result" in result
            assert result["result"] == expected
            assert result["expression"] == expr
    
    @pytest.mark.asyncio
    async def test_expression_security(self, calculator):
        """Test expression evaluation security."""
        # Test potentially dangerous expressions
        dangerous_expressions = [
            "import os",
            "__import__('os').system('ls')",
            "exec('print(1)')",
            "eval('1+1')"
        ]
        
        for expr in dangerous_expressions:
            result = await calculator.calculate("expression", expr)
            assert "error" in result  # Should reject dangerous expressions
    
    @pytest.mark.asyncio
    async def test_error_handling(self, calculator):
        """Test error handling in calculations."""
        # Division by zero
        data = {"type": "arithmetic", "a": 10, "b": 0, "operator": "/"}
        result = await calculator.calculate("basic_math", data)
        assert result["result"] == float('inf')  # Handled gracefully
        
        # Invalid statistics data
        result = await calculator.calculate("statistics", "not a list")
        assert "error" in result
        
        # Unknown operation
        result = await calculator.calculate("unknown_operation", {})
        assert "error" in result
    
    @pytest.mark.asyncio
    async def test_standalone_functions(self):
        """Test standalone function interfaces."""
        # Test statistics function
        stats = await calculate_stats([1, 2, 3, 4, 5])
        assert "mean" in stats
        
        # Test math evaluation
        math_result = await evaluate_math("2 + 2")
        assert math_result["result"] == 4
        
        # Test compound interest
        interest = await compound_interest(1000, 0.05, 1)
        assert "final_amount" in interest
        assert "interest_earned" in interest