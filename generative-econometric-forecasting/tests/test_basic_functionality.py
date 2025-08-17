#!/usr/bin/env python3
"""
🧪 Basic Functionality Tests
Tests core platform functionality without external dependencies
"""

import unittest
import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class TestBasicFunctionality(unittest.TestCase):
    """Test basic platform functionality"""
    
    def setUp(self):
        """Set up test data"""
        # Create sample time series data
        dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='M')
        np.random.seed(42)
        values = 100 + np.cumsum(np.random.normal(0.5, 2, len(dates)))
        self.sample_data = pd.Series(values, index=dates)
    
    def test_data_creation(self):
        """Test sample data creation"""
        self.assertIsInstance(self.sample_data, pd.Series)
        self.assertGreater(len(self.sample_data), 0)
        self.assertTrue(self.sample_data.index.is_monotonic_increasing)
    
    def test_basic_forecasting_imports(self):
        """Test that core modules can be imported"""
        try:
            from models.forecasting_models import EconometricForecaster
            from data.fred_client import FredDataClient
            self.assertTrue(True, "Core modules imported successfully")
        except ImportError as e:
            self.fail(f"Failed to import core modules: {e}")
    
    def test_foundation_models_imports(self):
        """Test foundation model imports"""
        try:
            from models.foundation_models.huggingface_forecaster import HybridFoundationEnsemble
            self.assertTrue(True, "Foundation models imported successfully")
        except ImportError as e:
            self.fail(f"Failed to import foundation models: {e}")
    
    def test_agents_imports(self):
        """Test AI agents imports"""
        try:
            from src.agents.narrative_generator import EconomicNarrativeGenerator
            from src.agents.demand_planner import GenAIDemandPlanner
            self.assertTrue(True, "AI agents imported successfully")
        except ImportError as e:
            self.fail(f"Failed to import AI agents: {e}")

class TestDataValidation(unittest.TestCase):
    """Test data validation functionality"""
    
    def test_time_series_validation(self):
        """Test time series data validation"""
        # Valid time series
        dates = pd.date_range('2020-01-01', periods=12, freq='M')
        values = np.random.randn(12)
        ts = pd.Series(values, index=dates)
        
        self.assertIsInstance(ts.index, pd.DatetimeIndex)
        self.assertEqual(len(ts), 12)
    
    def test_missing_data_handling(self):
        """Test handling of missing data"""
        dates = pd.date_range('2020-01-01', periods=10, freq='M')
        values = [1, 2, np.nan, 4, 5, np.nan, 7, 8, 9, 10]
        ts = pd.Series(values, index=dates)
        
        # Test missing data detection
        self.assertTrue(ts.isnull().any())
        self.assertEqual(ts.isnull().sum(), 2)
        
        # Test forward fill
        ts_filled = ts.fillna(method='ffill')
        self.assertFalse(ts_filled.isnull().any())

if __name__ == '__main__':
    print("🧪 RUNNING BASIC FUNCTIONALITY TESTS")
    print("=" * 50)
    
    # Run tests
    unittest.main(verbosity=2)