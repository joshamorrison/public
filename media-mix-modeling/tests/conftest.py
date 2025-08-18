"""
Pytest configuration and shared fixtures
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import os

@pytest.fixture
def sample_marketing_data():
    """Generate sample marketing data for testing"""
    np.random.seed(42)  # For reproducible tests
    
    # Generate 52 weeks of data
    dates = pd.date_range('2023-01-01', periods=52, freq='W')
    
    data = pd.DataFrame({
        'date': dates,
        'tv_spend': np.random.normal(50000, 10000, 52),
        'digital_spend': np.random.normal(30000, 8000, 52),
        'radio_spend': np.random.normal(20000, 5000, 52),
        'print_spend': np.random.normal(15000, 3000, 52),
        'social_spend': np.random.normal(25000, 6000, 52),
        'tv_impressions': np.random.normal(1000000, 200000, 52),
        'digital_impressions': np.random.normal(800000, 150000, 52),
        'radio_impressions': np.random.normal(500000, 100000, 52),
        'print_impressions': np.random.normal(300000, 50000, 52),
        'social_impressions': np.random.normal(600000, 120000, 52),
    })
    
    # Create correlated revenue (realistic relationship)
    data['revenue'] = (
        data['tv_spend'] * 0.8 +
        data['digital_spend'] * 1.2 +
        data['radio_spend'] * 0.6 +
        data['print_spend'] * 0.4 +
        data['social_spend'] * 1.0 +
        np.random.normal(20000, 5000, 52)
    )
    
    # Add conversions
    data['conversions'] = (data['revenue'] / 50).astype(int)
    
    # Ensure no negative values
    for col in data.columns:
        if col != 'date':
            data[col] = np.maximum(data[col], 0)
    
    return data

@pytest.fixture
def small_marketing_data():
    """Generate small dataset for quick tests"""
    np.random.seed(42)
    
    data = pd.DataFrame({
        'date': pd.date_range('2023-01-01', periods=10, freq='W'),
        'tv_spend': np.random.normal(50000, 5000, 10),
        'digital_spend': np.random.normal(30000, 3000, 10),
        'revenue': np.random.normal(100000, 10000, 10)
    })
    
    # Ensure positive values
    for col in data.columns:
        if col != 'date':
            data[col] = np.maximum(data[col], 0)
    
    return data

@pytest.fixture
def temp_directory():
    """Create temporary directory for test files"""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)

@pytest.fixture
def mock_environment_variables():
    """Mock environment variables for testing"""
    original_env = {}
    test_env = {
        'KAGGLE_USERNAME': 'test_user',
        'KAGGLE_KEY': 'test_key',
        'HF_TOKEN': 'test_hf_token',
        'MLFLOW_TRACKING_URI': 'file:///tmp/mlflow'
    }
    
    # Store original values
    for key in test_env:
        original_env[key] = os.environ.get(key)
        os.environ[key] = test_env[key]
    
    yield test_env
    
    # Restore original values
    for key, value in original_env.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value

@pytest.fixture
def sample_budget_allocation():
    """Sample budget allocation for optimization tests"""
    return {
        'tv': 50000,
        'digital': 30000,
        'radio': 20000,
        'print': 15000,
        'social': 25000
    }

@pytest.fixture
def sample_model_parameters():
    """Sample MMM model parameters"""
    return {
        'adstock_rate': 0.5,
        'saturation_param': 0.6,
        'regularization_alpha': 0.1
    }

# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "aws: marks tests requiring AWS credentials"
    )
    config.addinivalue_line(
        "markers", "r_integration: marks tests requiring R integration"
    )

def pytest_collection_modifyitems(config, items):
    """Automatically mark tests based on name patterns"""
    for item in items:
        # Mark slow tests
        if "slow" in item.nodeid or "integration" in item.nodeid:
            item.add_marker(pytest.mark.slow)
        
        # Mark AWS tests
        if "aws" in item.nodeid or "sagemaker" in item.nodeid:
            item.add_marker(pytest.mark.aws)
        
        # Mark R integration tests
        if "r_integration" in item.nodeid or "_r_" in item.nodeid:
            item.add_marker(pytest.mark.r_integration)