"""
Unit tests for FRED data client functionality.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock

from data.fred_client import FredDataClient, validate_data_quality


class TestFredDataClient:
    """Test suite for FredDataClient."""
    
    def test_init_without_api_key(self):
        """Test initialization without API key."""
        client = FredDataClient(api_key=None, use_cache_fallback=True)
        assert client.api_key is None
        assert client.fred is None
        assert client.use_cache_fallback is True
    
    def test_init_with_api_key(self):
        """Test initialization with API key."""
        with patch('data.fred_client.Fred') as mock_fred:
            mock_fred.return_value = MagicMock()
            
            client = FredDataClient(api_key="test_key")
            assert client.api_key == "test_key"
            assert client.fred is not None
            mock_fred.assert_called_once_with(api_key="test_key")
    
    def test_indicator_mapping(self):
        """Test economic indicator mapping."""
        client = FredDataClient()
        
        expected_mappings = {
            'gdp': 'GDPC1',
            'unemployment': 'UNRATE',
            'inflation': 'CPIAUCSL',
            'interest_rate': 'DGS10'
        }
        
        for key, expected_value in expected_mappings.items():
            assert client.indicators[key] == expected_value
    
    def test_fetch_indicator_with_cache(self, mock_fred_client):
        """Test fetching indicator from cache."""
        data = mock_fred_client.fetch_indicator('gdp', start_date='2020-01-01')
        
        assert isinstance(data, pd.Series)
        assert len(data) > 0
        assert data.name == 'gdp'
        assert data.index[0] >= pd.Timestamp('2020-01-01')
    
    def test_fetch_multiple_indicators(self, mock_fred_client):
        """Test fetching multiple indicators."""
        indicators = ['gdp', 'unemployment', 'inflation']
        data = mock_fred_client.fetch_multiple_indicators(
            indicators, start_date='2020-01-01'
        )
        
        assert isinstance(data, pd.DataFrame)
        assert list(data.columns) == indicators
        assert len(data) > 0
    
    def test_fetch_dashboard_data(self, mock_fred_client):
        """Test fetching dashboard data."""
        data = mock_fred_client.get_economic_dashboard_data(start_date='2020-01-01')
        
        assert isinstance(data, pd.DataFrame)
        assert 'gdp' in data.columns
        assert 'unemployment' in data.columns
        assert len(data) > 0
    
    def test_invalid_indicator(self, mock_fred_client):
        """Test handling of invalid indicator."""
        with pytest.raises(ValueError, match="No data available"):
            mock_fred_client.fetch_indicator('invalid_indicator')
    
    def test_date_filtering(self, mock_fred_client):
        """Test date range filtering."""
        start_date = '2022-01-01'
        end_date = '2022-12-31'
        
        data = mock_fred_client.fetch_indicator(
            'gdp', start_date=start_date, end_date=end_date
        )
        
        assert data.index[0] >= pd.Timestamp(start_date)
        assert data.index[-1] <= pd.Timestamp(end_date)


class TestDataValidation:
    """Test suite for data validation functions."""
    
    def test_validate_data_quality_good_data(self, sample_gdp_data):
        """Test validation with good quality data."""
        quality_report = validate_data_quality(sample_gdp_data.to_frame())
        
        assert quality_report['sufficient_data'] is True
        assert quality_report['excessive_missing'] is False
        assert quality_report['total_observations'] > 0
        assert isinstance(quality_report['date_range'], tuple)
    
    def test_validate_data_quality_insufficient_data(self):
        """Test validation with insufficient data."""
        # Create small dataset
        small_data = pd.DataFrame({
            'value': [1, 2, 3]
        }, index=pd.date_range('2024-01-01', periods=3))
        
        quality_report = validate_data_quality(small_data)
        
        assert quality_report['sufficient_data'] is False
        assert quality_report['total_observations'] == 3
    
    def test_validate_data_quality_missing_values(self):
        """Test validation with missing values."""
        # Create data with missing values
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        values = np.random.randn(100)
        values[50:60] = np.nan  # 10% missing
        
        data = pd.DataFrame({'value': values}, index=dates)
        quality_report = validate_data_quality(data)
        
        assert quality_report['missing_percentage']['value'] == 10.0
        assert quality_report['excessive_missing'] is False
    
    def test_validate_data_quality_excessive_missing(self):
        """Test validation with excessive missing values."""
        # Create data with >20% missing values
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        values = np.random.randn(100)
        values[70:] = np.nan  # 30% missing
        
        data = pd.DataFrame({'value': values}, index=dates)
        quality_report = validate_data_quality(data)
        
        assert quality_report['missing_percentage']['value'] == 30.0
        assert quality_report['excessive_missing'] is True
    
    def test_validate_data_quality_frequency_detection(self, sample_gdp_data):
        """Test frequency detection in validation."""
        monthly_data = sample_gdp_data.resample('MS').ffill().to_frame()
        quality_report = validate_data_quality(monthly_data)
        
        # Should detect monthly frequency
        assert quality_report['frequency'] is not None


class TestCacheManagement:
    """Test suite for cache data management."""
    
    def test_cache_data_loading(self):
        """Test loading cached data."""
        with patch('data.fred_client.Path.exists') as mock_exists:
            with patch('builtins.open', create=True) as mock_open:
                with patch('json.load') as mock_json_load:
                    # Mock file exists
                    mock_exists.return_value = True
                    
                    # Mock JSON data
                    mock_json_load.return_value = {
                        'gdp': {
                            'name': 'Real GDP',
                            'data': {'2020-01-01': 21000, '2020-04-01': 21100}
                        }
                    }
                    
                    client = FredDataClient(use_cache_fallback=True)
                    assert client.cache_data is not None
                    assert 'gdp' in client.cache_data
    
    def test_cache_fallback_without_file(self):
        """Test cache fallback when no file exists."""
        with patch('data.fred_client.Path.exists') as mock_exists:
            mock_exists.return_value = False
            
            client = FredDataClient(use_cache_fallback=True)
            assert client.cache_data == {}
    
    def test_get_cached_indicator(self, mock_fred_client):
        """Test retrieving cached indicator data."""
        # This tests the internal _get_cached_indicator method
        cached_data = mock_fred_client._get_cached_indicator('gdp', '2020-01-01')
        
        assert cached_data is not None
        assert isinstance(cached_data, pd.Series)
        assert len(cached_data) > 0
    
    def test_get_cached_indicator_missing(self, mock_fred_client):
        """Test retrieving non-existent cached indicator."""
        cached_data = mock_fred_client._get_cached_indicator('nonexistent')
        assert cached_data is None


class TestErrorHandling:
    """Test suite for error handling scenarios."""
    
    def test_api_error_handling(self):
        """Test handling of API errors."""
        with patch('data.fred_client.Fred') as mock_fred_class:
            mock_fred = MagicMock()
            mock_fred.get_series.side_effect = Exception("API Error")
            mock_fred_class.return_value = mock_fred
            
            client = FredDataClient(api_key="test_key", use_cache_fallback=False)
            
            with pytest.raises(Exception):
                client.fetch_indicator('gdp')
    
    def test_api_error_with_fallback(self, mock_fred_client):
        """Test API error with cache fallback."""
        # Mock the fred client to raise an error
        with patch.object(mock_fred_client, 'fred') as mock_fred:
            mock_fred.get_series.side_effect = Exception("API Error")
            
            # Should fall back to cache
            data = mock_fred_client.fetch_indicator('gdp')
            assert data is not None
            assert isinstance(data, pd.Series)
    
    def test_invalid_date_format(self, mock_fred_client):
        """Test handling of invalid date formats."""
        with pytest.raises(ValueError):
            mock_fred_client.fetch_indicator('gdp', start_date='invalid-date')
    
    def test_empty_indicators_list(self, mock_fred_client):
        """Test handling of empty indicators list."""
        with pytest.raises(ValueError):
            mock_fred_client.fetch_multiple_indicators([])
    
    def test_no_data_sources_available(self):
        """Test when no data sources are available."""
        client = FredDataClient(api_key=None, use_cache_fallback=False)
        
        with pytest.raises(ValueError, match="No data available"):
            client.fetch_indicator('gdp')


@pytest.mark.integration
class TestFredIntegration:
    """Integration tests that may require actual FRED API access."""
    
    @pytest.mark.slow
    def test_real_api_connection(self):
        """Test real API connection if API key is available."""
        import os
        api_key = os.getenv('FRED_API_KEY')
        
        if not api_key or api_key == 'your_fred_api_key_here':
            pytest.skip("No valid FRED API key available")
        
        client = FredDataClient(api_key=api_key)
        
        try:
            data = client.fetch_indicator('gdp', start_date='2023-01-01')
            assert isinstance(data, pd.Series)
            assert len(data) > 0
        except Exception as e:
            pytest.skip(f"FRED API not accessible: {e}")
    
    @pytest.mark.slow
    def test_search_functionality(self):
        """Test series search functionality."""
        import os
        api_key = os.getenv('FRED_API_KEY')
        
        if not api_key or api_key == 'your_fred_api_key_here':
            pytest.skip("No valid FRED API key available")
        
        client = FredDataClient(api_key=api_key)
        
        try:
            results = client.search_series('unemployment', limit=5)
            assert isinstance(results, pd.DataFrame)
            assert len(results) <= 5
        except Exception as e:
            pytest.skip(f"FRED API search not accessible: {e}")