"""
Pytest configuration and shared fixtures for the test suite.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

@pytest.fixture
def sample_gdp_data():
    """Create sample GDP time series data for testing."""
    dates = pd.date_range('2020-01-01', '2024-01-01', freq='QS')
    # Realistic GDP values with growth trend and COVID impact
    values = []
    base_value = 21000
    
    for i, date in enumerate(dates):
        # COVID impact in 2020 Q2
        if date.year == 2020 and date.quarter == 2:
            value = base_value * 0.9  # 10% drop
        else:
            # Normal growth with some volatility
            growth_rate = 0.005 + np.random.normal(0, 0.002)
            value = base_value * (1 + growth_rate)
            base_value = value
        
        values.append(value)
    
    return pd.Series(values, index=dates, name='GDP')

@pytest.fixture
def sample_unemployment_data():
    """Create sample unemployment rate data for testing."""
    dates = pd.date_range('2020-01-01', '2024-01-01', freq='MS')
    values = []
    base_rate = 3.5
    
    for i, date in enumerate(dates):
        # COVID spike in 2020
        if date.year == 2020 and date.month >= 4 and date.month <= 8:
            multiplier = 3.0 if date.month == 4 else 2.5 if date.month == 5 else 1.8
            value = base_rate * multiplier
        else:
            # Gradual return to normal with noise
            noise = np.random.normal(0, 0.1)
            value = max(3.0, base_rate + noise)
            base_rate = value * 0.99 + 3.5 * 0.01  # Mean reversion
        
        values.append(value)
    
    return pd.Series(values, index=dates, name='unemployment')

@pytest.fixture
def sample_inflation_data():
    """Create sample inflation rate data for testing."""
    dates = pd.date_range('2020-01-01', '2024-01-01', freq='MS')
    values = []
    base_rate = 2.0
    
    for i, date in enumerate(dates):
        # Inflation surge in 2021-2022
        if date.year == 2021:
            multiplier = 2.0 + (date.month / 12) * 1.5  # Rising through 2021
        elif date.year == 2022:
            multiplier = 3.5 - (date.month / 12) * 1.5  # Falling through 2022
        else:
            multiplier = 1.0
        
        noise = np.random.normal(0, 0.2)
        value = max(0, base_rate * multiplier + noise)
        values.append(value)
    
    return pd.Series(values, index=dates, name='inflation')

@pytest.fixture
def sample_multi_indicator_data(sample_gdp_data, sample_unemployment_data, sample_inflation_data):
    """Create multi-indicator DataFrame for testing."""
    # Align data to monthly frequency
    gdp_monthly = sample_gdp_data.resample('MS').ffill()
    
    # Combine all indicators
    data = pd.DataFrame({
        'gdp': gdp_monthly,
        'unemployment': sample_unemployment_data,
        'inflation': sample_inflation_data
    })
    
    return data.fillna(method='ffill')

@pytest.fixture
def mock_fred_client():
    """Create a mock FRED client for testing without API calls."""
    from data.fred_client import FredDataClient
    
    class MockFredClient(FredDataClient):
        def __init__(self):
            self.api_key = "test_key"
            self.fred = None
            self.use_cache_fallback = True
            self.cache_data = {
                'gdp': {
                    'name': 'Real GDP',
                    'data': self._generate_mock_data('gdp')
                },
                'unemployment': {
                    'name': 'Unemployment Rate',
                    'data': self._generate_mock_data('unemployment')
                },
                'inflation': {
                    'name': 'Inflation Rate',
                    'data': self._generate_mock_data('inflation')
                }
            }
        
        def _generate_mock_data(self, indicator):
            """Generate mock time series data."""
            dates = pd.date_range('2020-01-01', '2024-01-01', freq='MS')
            
            if indicator == 'gdp':
                values = [21000 + i * 50 for i in range(len(dates))]
            elif indicator == 'unemployment':
                values = [3.5 + np.sin(i/6) for i in range(len(dates))]
            else:  # inflation
                values = [2.0 + np.sin(i/12) * 0.5 for i in range(len(dates))]
            
            return {date.strftime('%Y-%m-%d'): value for date, value in zip(dates, values)}
    
    return MockFredClient()

@pytest.fixture
def temp_output_dir():
    """Create temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)

@pytest.fixture
def api_test_client():
    """Create test client for API testing."""
    from fastapi.testclient import TestClient
    from api.main import app
    
    return TestClient(app)

@pytest.fixture
def sample_forecast_request():
    """Create sample forecast request for API testing."""
    return {
        "indicators": ["gdp", "unemployment"],
        "horizon": 6,
        "method": "statistical",
        "model_tier": "tier3",
        "confidence_interval": 0.95,
        "generate_report": False
    }

@pytest.fixture
def sample_scenario_request():
    """Create sample scenario analysis request."""
    return {
        "indicators": ["gdp", "unemployment"],
        "scenarios": ["baseline", "optimistic", "pessimistic"],
        "horizon": 12,
        "monte_carlo_runs": 100
    }

@pytest.fixture(scope="session")
def test_config():
    """Test configuration settings."""
    return {
        "api_timeout": 30,
        "max_forecast_horizon": 24,
        "test_data_start": "2020-01-01",
        "test_data_end": "2024-01-01",
        "performance_thresholds": {
            "forecast_time": 10.0,  # seconds
            "api_response_time": 5.0,  # seconds
            "memory_usage": 1024 * 1024 * 500  # 500MB
        }
    }

# Pytest configuration
def pytest_configure(config):
    """Configure pytest settings."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "api: marks tests as API tests"
    )
    config.addinivalue_line(
        "markers", "performance: marks tests as performance tests"
    )

def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test location."""
    for item in items:
        # Add markers based on test file location
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "performance" in str(item.fspath):
            item.add_marker(pytest.mark.performance)
        elif "api" in str(item.fspath) or "test_api" in item.name:
            item.add_marker(pytest.mark.api)
        
        # Mark slow tests
        if "slow" in item.name or any(marker.name == "slow" for marker in item.iter_markers()):
            item.add_marker(pytest.mark.slow)