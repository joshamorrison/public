"""
Tests for data client functionality
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

from data.media_data_client import MediaDataClient

class TestMediaDataClient:
    """Test cases for MediaDataClient"""
    
    def test_init_default(self):
        """Test default initialization"""
        client = MediaDataClient()
        
        assert client.cache_dir == "./cache/data"
        assert 'synthetic' in client.data_sources
        assert client.data_sources['synthetic'] == True
    
    def test_init_with_params(self):
        """Test initialization with custom parameters"""
        client = MediaDataClient(
            kaggle_username="test_user",
            kaggle_key="test_key",
            hf_token="test_token",
            cache_dir="./custom_cache"
        )
        
        assert client.kaggle_username == "test_user"
        assert client.kaggle_key == "test_key" 
        assert client.hf_token == "test_token"
        assert client.cache_dir == "./custom_cache"
    
    def test_synthetic_data_generation(self):
        """Test synthetic data generation"""
        client = MediaDataClient()
        
        # Test the core method that we know exists
        result = client.get_best_available_data()
        
        # Should return tuple of (data, info, source_type)
        assert isinstance(result, tuple)
        assert len(result) == 3
        
        data, info, source_type = result
        
        # Check data structure
        assert isinstance(data, pd.DataFrame)
        assert len(data) > 0
        assert 'revenue' in data.columns
        
        # Check info
        assert isinstance(info, dict)
        assert 'description' in info
        
        # Check source type
        assert source_type == 'SYNTHETIC'
    
    def test_synthetic_data_quality(self):
        """Test quality of synthetic data"""
        client = MediaDataClient()
        data, info, _ = client.get_best_available_data()
        
        # Check for spend columns
        spend_columns = [col for col in data.columns if col.endswith('_spend')]
        assert len(spend_columns) > 0
        
        # Check for positive values
        for col in spend_columns:
            assert (data[col] >= 0).all(), f"Negative values found in {col}"
        
        # Check revenue column
        assert 'revenue' in data.columns
        assert (data['revenue'] >= 0).all()
        
        # Check data completeness
        assert data.isnull().sum().sum() == 0, "Missing values found in synthetic data"
    
    def test_data_source_checking(self):
        """Test data source availability checking"""
        client = MediaDataClient()
        
        # Should have at least synthetic available
        assert client.data_sources['synthetic'] == True
        
        # Check the structure
        assert isinstance(client.data_sources, dict)
        expected_sources = ['kaggle', 'huggingface', 'synthetic']
        for source in expected_sources:
            assert source in client.data_sources
    
    @patch.dict('os.environ', {'KAGGLE_USERNAME': 'test', 'KAGGLE_KEY': 'test'})
    def test_kaggle_credentials_detection(self):
        """Test Kaggle credentials detection"""
        client = MediaDataClient()
        # Should detect mocked environment variables
        assert client.kaggle_username == 'test'
        assert client.kaggle_key == 'test'
    
    @patch.dict('os.environ', {'HF_TOKEN': 'test_token'})
    def test_huggingface_token_detection(self):
        """Test HuggingFace token detection"""
        client = MediaDataClient()
        assert client.hf_token == 'test_token'
    
    def test_cache_directory_creation(self, temp_directory):
        """Test cache directory creation"""
        cache_path = temp_directory / "test_cache"
        client = MediaDataClient(cache_dir=str(cache_path))
        
        # Cache directory should be created
        assert cache_path.exists()
        assert cache_path.is_dir()
    
    def test_data_consistency(self):
        """Test that multiple calls return consistent data structure"""
        client = MediaDataClient()
        
        # Get data multiple times
        result1 = client.get_best_available_data()
        result2 = client.get_best_available_data()
        
        data1, info1, source1 = result1
        data2, info2, source2 = result2
        
        # Should have same structure
        assert data1.columns.tolist() == data2.columns.tolist()
        assert len(data1) == len(data2)
        assert source1 == source2
        
        # Info should have same keys
        assert set(info1.keys()) == set(info2.keys())

class TestDataValidation:
    """Test data validation and quality checks"""
    
    def test_data_types(self):
        """Test that data has correct types"""
        client = MediaDataClient()
        data, _, _ = client.get_best_available_data()
        
        # Check numeric columns
        numeric_columns = [col for col in data.columns if col != 'date']
        for col in numeric_columns:
            assert pd.api.types.is_numeric_dtype(data[col]), f"{col} should be numeric"
        
        # Check date column if present
        if 'date' in data.columns:
            assert pd.api.types.is_datetime64_any_dtype(data['date']) or \
                   pd.api.types.is_object_dtype(data['date']), "Date column should be datetime or object"
    
    def test_data_ranges(self):
        """Test that data values are in reasonable ranges"""
        client = MediaDataClient()
        data, _, _ = client.get_best_available_data()
        
        # Check spend columns are positive
        spend_columns = [col for col in data.columns if col.endswith('_spend')]
        for col in spend_columns:
            assert (data[col] >= 0).all(), f"{col} should be non-negative"
            assert data[col].max() > 0, f"{col} should have some positive values"
        
        # Check revenue is positive
        if 'revenue' in data.columns:
            assert (data['revenue'] >= 0).all(), "Revenue should be non-negative"
            assert data['revenue'].max() > 0, "Revenue should have positive values"
    
    def test_data_correlations(self):
        """Test that synthetic data has realistic correlations"""
        client = MediaDataClient()
        data, _, _ = client.get_best_available_data()
        
        if 'revenue' in data.columns:
            spend_columns = [col for col in data.columns if col.endswith('_spend')]
            
            if spend_columns:
                # Calculate correlations with revenue
                correlations = data[spend_columns + ['revenue']].corr()['revenue']
                
                # Spend should generally correlate positively with revenue
                for col in spend_columns:
                    correlation = correlations[col]
                    # Allow for some variation but expect positive correlation
                    assert correlation > -0.5, f"{col} correlation with revenue is too negative: {correlation}"

class TestErrorHandling:
    """Test error handling and edge cases"""
    
    def test_invalid_cache_directory(self):
        """Test handling of invalid cache directory"""
        # This should handle the error gracefully
        try:
            client = MediaDataClient(cache_dir="/invalid/path/that/cannot/be/created")
            # Should still work, possibly creating alternative cache
            result = client.get_best_available_data()
            assert isinstance(result, tuple)
        except Exception as e:
            # If it fails, should be a specific expected error
            assert "permission" in str(e).lower() or "path" in str(e).lower()
    
    def test_missing_environment_variables(self):
        """Test behavior when environment variables are missing"""
        with patch.dict('os.environ', {}, clear=True):
            client = MediaDataClient()
            
            # Should still work with synthetic data
            result = client.get_best_available_data()
            assert isinstance(result, tuple)
            
            data, info, source_type = result
            assert source_type == 'SYNTHETIC'