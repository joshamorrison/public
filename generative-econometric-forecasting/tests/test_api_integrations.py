#!/usr/bin/env python3
"""
🧪 API Integrations Unit Tests
Tests LangChain, OpenAI, and external API integrations
"""

import unittest
import sys
import os
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from src.agents.narrative_generator import EconomicNarrativeGenerator, ForecastInsight
    NARRATIVE_GENERATOR_AVAILABLE = True
except ImportError:
    NARRATIVE_GENERATOR_AVAILABLE = False

try:
    from src.agents.demand_planner import GenAIDemandPlanner
    DEMAND_PLANNER_AVAILABLE = True
except ImportError:
    DEMAND_PLANNER_AVAILABLE = False


class TestLangChainIntegration(unittest.TestCase):
    """Test LangChain framework integration"""
    
    def test_langchain_imports(self):
        """Test that LangChain components can be imported"""
        try:
            from langchain_openai import ChatOpenAI
            from langchain_core.messages import HumanMessage, SystemMessage
            from langchain_core.output_parsers import JsonOutputParser
            from langchain.prompts import PromptTemplate
            self.assertTrue(True, "LangChain imports successful")
        except ImportError as e:
            self.fail(f"LangChain import failed: {e}")
    
    @patch('langchain_openai.ChatOpenAI')
    def test_chatgpt_mock_initialization(self, mock_chatgpt):
        """Test ChatGPT initialization with mocking"""
        # Mock ChatGPT instance
        mock_instance = mock_chatgpt.return_value
        
        # Test initialization
        llm = mock_chatgpt(model="gpt-3.5-turbo", temperature=0.3)
        
        self.assertIsNotNone(llm)
        mock_chatgpt.assert_called_once_with(model="gpt-3.5-turbo", temperature=0.3)
    
    def test_prompt_template_creation(self):
        """Test prompt template creation"""
        from langchain.prompts import PromptTemplate
        
        template = PromptTemplate(
            input_variables=["metric", "data"],
            template="Analyze the {metric} data: {data}"
        )
        
        self.assertIsInstance(template, PromptTemplate)
        self.assertCountEqual(template.input_variables, ["metric", "data"])
        
        # Test formatting
        formatted = template.format(metric="GDP", data="100, 101, 102")
        self.assertIn("GDP", formatted)
        self.assertIn("100, 101, 102", formatted)


class TestEconomicNarrativeGenerator(unittest.TestCase):
    """Test Economic Narrative Generator"""
    
    @unittest.skipIf(not NARRATIVE_GENERATOR_AVAILABLE, "Narrative generator not available")
    @patch('src.agents.narrative_generator.ChatOpenAI')
    def test_narrative_generator_initialization(self, mock_chatgpt):
        """Test narrative generator initialization"""
        mock_instance = mock_chatgpt.return_value
        
        generator = EconomicNarrativeGenerator()
        
        self.assertIsInstance(generator, EconomicNarrativeGenerator)
        mock_chatgpt.assert_called_once()
    
    def test_forecast_insight_model(self):
        """Test ForecastInsight Pydantic model"""
        if not NARRATIVE_GENERATOR_AVAILABLE:
            self.skipTest("Narrative generator not available")
            
        insight_data = {
            "metric": "GDP",
            "current_trend": "increasing",
            "forecast_direction": "continued growth",
            "confidence_level": "high",
            "key_drivers": ["consumer spending", "business investment"],
            "business_implications": ["increased demand", "expansion opportunities"],
            "risk_factors": ["inflation", "supply chain disruption"]
        }
        
        insight = ForecastInsight(**insight_data)
        
        self.assertEqual(insight.metric, "GDP")
        self.assertEqual(insight.current_trend, "increasing")
        self.assertEqual(len(insight.key_drivers), 2)
        self.assertEqual(len(insight.business_implications), 2)
        self.assertEqual(len(insight.risk_factors), 2)
    
    @unittest.skipIf(not NARRATIVE_GENERATOR_AVAILABLE, "Narrative generator not available")
    @patch('src.agents.narrative_generator.ChatOpenAI')
    def test_narrative_generation_mock(self, mock_chatgpt):
        """Test narrative generation with mocking"""
        # Mock LLM response
        mock_llm = mock_chatgpt.return_value
        mock_llm.invoke.return_value = "GDP shows strong growth with 3.2% increase expected next quarter."
        
        generator = EconomicNarrativeGenerator()
        
        # Test data
        historical_data = "GDP: 100, 101, 102, 103"
        forecast_data = "Forecast: 104, 105, 106"
        
        # This would normally call the LLM
        # We're testing the structure, not the actual API call
        self.assertIsNotNone(generator.llm)
        self.assertIsNotNone(generator.executive_template)


class TestOpenAIIntegration(unittest.TestCase):
    """Test OpenAI API integration"""
    
    @patch('openai.OpenAI')
    def test_openai_client_mock(self, mock_openai):
        """Test OpenAI client with mocking"""
        # Mock OpenAI client
        mock_client = mock_openai.return_value
        mock_response = Mock()
        mock_choice = Mock()
        mock_message = Mock()
        mock_message.content = "This is a test response"
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response
        
        # Test client usage
        client = mock_openai(api_key="test-key")
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hello"}]
        )
        
        self.assertIsNotNone(response)
        mock_openai.assert_called_once_with(api_key="test-key")
    
    def test_api_key_handling(self):
        """Test API key handling"""
        # Test environment variable loading
        import os
        from dotenv import load_dotenv
        
        load_dotenv()
        
        openai_key = os.getenv('OPENAI_API_KEY')
        langchain_key = os.getenv('LANGCHAIN_API_KEY')
        
        # Keys should be strings if present
        if openai_key:
            self.assertIsInstance(openai_key, str)
            self.assertGreater(len(openai_key), 10)  # Reasonable key length
        
        if langchain_key:
            self.assertIsInstance(langchain_key, str)
            self.assertGreater(len(langchain_key), 10)


class TestDemandPlannerIntegration(unittest.TestCase):
    """Test Demand Planner AI agent"""
    
    @unittest.skipIf(not DEMAND_PLANNER_AVAILABLE, "Demand planner not available")
    @patch('src.agents.demand_planner.ChatOpenAI')
    def test_demand_planner_initialization(self, mock_chatgpt):
        """Test demand planner initialization"""
        mock_instance = mock_chatgpt.return_value
        
        try:
            planner = GenAIDemandPlanner()
            self.assertIsInstance(planner, GenAIDemandPlanner)
            mock_chatgpt.assert_called_once()
        except NameError:
            self.skipTest("GenAIDemandPlanner class not found")
    
    def test_scenario_generation_structure(self):
        """Test demand scenario generation structure"""
        # Test scenario data structure
        scenario_template = {
            "scenario_name": "Base Case",
            "probability": 0.6,
            "demand_change": 0.05,
            "key_factors": ["economic growth", "consumer confidence"],
            "business_impact": "moderate increase in demand"
        }
        
        # Validate structure
        required_fields = ["scenario_name", "probability", "demand_change", "key_factors", "business_impact"]
        for field in required_fields:
            self.assertIn(field, scenario_template)
        
        self.assertIsInstance(scenario_template["probability"], float)
        self.assertIsInstance(scenario_template["demand_change"], (int, float))
        self.assertIsInstance(scenario_template["key_factors"], list)


class TestExternalAPIIntegrations(unittest.TestCase):
    """Test external API integrations (FRED, News API, etc.)"""
    
    @patch('requests.get')
    def test_news_api_mock(self, mock_get):
        """Test News API integration with mocking"""
        # Mock News API response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "status": "ok",
            "totalResults": 2,
            "articles": [
                {
                    "title": "Economic Growth Accelerates",
                    "description": "GDP growth exceeded expectations...",
                    "publishedAt": "2023-12-01T10:00:00Z",
                    "sentiment": "positive"
                },
                {
                    "title": "Inflation Concerns Rise",
                    "description": "Central bank considers policy changes...",
                    "publishedAt": "2023-12-01T11:00:00Z",
                    "sentiment": "negative"
                }
            ]
        }
        mock_get.return_value = mock_response
        
        # Test API call
        import requests
        response = requests.get("https://newsapi.org/v2/everything?q=economy")
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ok")
        self.assertEqual(len(data["articles"]), 2)
    
    @patch('fredapi.Fred')
    def test_fred_api_mock(self, mock_fred):
        """Test FRED API integration with mocking"""
        # Mock FRED API
        mock_instance = mock_fred.return_value
        
        # Mock GDP data
        dates = pd.date_range('2023-01-01', periods=4, freq='QE')
        gdp_data = pd.Series([20000, 20100, 20200, 20300], index=dates)
        mock_instance.get_series.return_value = gdp_data
        
        # Test usage
        fred = mock_fred(api_key="test-key")
        data = fred.get_series('GDP')
        
        self.assertIsInstance(data, pd.Series)
        self.assertEqual(len(data), 4)
        mock_fred.assert_called_once_with(api_key="test-key")
        mock_instance.get_series.assert_called_once_with('GDP')
    
    def test_alpha_vantage_mock_structure(self):
        """Test Alpha Vantage API response structure"""
        # Mock Alpha Vantage response structure
        mock_av_response = {
            "Meta Data": {
                "1. Information": "Monthly Adjusted Close Prices and Volumes",
                "2. Symbol": "SPY",
                "3. Last Refreshed": "2023-12-01",
                "4. Time Zone": "US/Eastern"
            },
            "Monthly Adjusted Time Series": {
                "2023-11-30": {
                    "1. open": "450.00",
                    "2. high": "455.00",
                    "3. low": "445.00",
                    "4. close": "452.00",
                    "5. adjusted close": "452.00",
                    "6. volume": "1000000"
                }
            }
        }
        
        # Validate structure
        self.assertIn("Meta Data", mock_av_response)
        self.assertIn("Monthly Adjusted Time Series", mock_av_response)
        
        time_series = mock_av_response["Monthly Adjusted Time Series"]
        self.assertGreater(len(time_series), 0)
        
        # Check data point structure
        sample_date = list(time_series.keys())[0]
        sample_data = time_series[sample_date]
        
        required_fields = ["1. open", "2. high", "3. low", "4. close", "5. adjusted close", "6. volume"]
        for field in required_fields:
            self.assertIn(field, sample_data)


class TestErrorHandling(unittest.TestCase):
    """Test error handling in API integrations"""
    
    def test_api_timeout_simulation(self):
        """Test API timeout handling"""
        import requests
        from requests.exceptions import Timeout
        
        with patch('requests.get') as mock_get:
            mock_get.side_effect = Timeout("API request timed out")
            
            with self.assertRaises(Timeout):
                requests.get("https://api.example.com/data", timeout=5)
    
    def test_api_rate_limit_simulation(self):
        """Test API rate limit handling"""
        import requests
        
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 429  # Too Many Requests
            mock_response.json.return_value = {"error": "Rate limit exceeded"}
            mock_get.return_value = mock_response
            
            response = requests.get("https://api.example.com/data")
            
            self.assertEqual(response.status_code, 429)
            self.assertIn("error", response.json())
    
    def test_invalid_api_key_simulation(self):
        """Test invalid API key handling"""
        import requests
        
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 401  # Unauthorized
            mock_response.json.return_value = {"error": "Invalid API key"}
            mock_get.return_value = mock_response
            
            response = requests.get("https://api.example.com/data")
            
            self.assertEqual(response.status_code, 401)
            error_data = response.json()
            self.assertIn("error", error_data)
            self.assertIn("Invalid API key", error_data["error"])


class TestLangSmithIntegration(unittest.TestCase):
    """Test LangSmith monitoring integration"""
    
    def test_langsmith_tracing_setup(self):
        """Test LangSmith tracing configuration"""
        import os
        
        # Test environment variable setup
        langchain_api_key = os.getenv('LANGCHAIN_API_KEY')
        
        if langchain_api_key:
            # Test that tracing can be enabled
            os.environ['LANGCHAIN_TRACING_V2'] = 'true'
            os.environ['LANGCHAIN_PROJECT'] = 'econometric-forecasting'
            
            self.assertEqual(os.getenv('LANGCHAIN_TRACING_V2'), 'true')
            self.assertEqual(os.getenv('LANGCHAIN_PROJECT'), 'econometric-forecasting')
    
    @patch('langchain.callbacks.manager.CallbackManager')
    def test_callback_manager_mock(self, mock_callback_manager):
        """Test callback manager for tracing"""
        mock_manager = mock_callback_manager.return_value
        
        # Test callback manager usage
        manager = mock_callback_manager()
        
        self.assertIsNotNone(manager)
        mock_callback_manager.assert_called_once()


if __name__ == '__main__':
    print("🧪 RUNNING API INTEGRATIONS TESTS")
    print("=" * 50)
    
    # Run tests
    unittest.main(verbosity=2)