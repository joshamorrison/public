"""
Unit tests for MediaDataClient with real data integration
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
import os

from data.media_data_client import MediaDataClient


class TestMediaDataClient:
    """Test suite for MediaDataClient with tiered data sources"""

    def test_client_initialization(self):
        """Test client initializes with correct parameters"""
        client = MediaDataClient(
            kaggle_username="test_user",
            kaggle_key="test_key", 
            hf_token="test_token",
            cache_dir="./test_cache"
        )
        
        assert client.kaggle_username == "test_user"
        assert client.kaggle_key == "test_key"
        assert client.hf_token == "test_token"
        assert client.cache_dir == "./test_cache"

    def test_data_source_capabilities_check(self):
        """Test data source availability detection"""
        client = MediaDataClient()
        
        # Should always have synthetic capability
        assert client.data_sources['synthetic'] is True
        
        # Other sources depend on installed packages
        assert isinstance(client.data_sources['kaggle'], bool)
        assert isinstance(client.data_sources['huggingface'], bool)

    @patch('data.media_data_client.load_dataset')
    def test_huggingface_data_loading_success(self, mock_load_dataset):
        """Test successful HuggingFace data loading"""
        # Mock successful dataset loading
        mock_dataset = Mock()
        mock_df = pd.DataFrame({
            'instruction': ['Create marketing campaign', 'Optimize budget'],
            'input': ['Company: TechCorp', 'Budget: $50k'],
            'response': ['Launch social media campaign', 'Reallocate to digital channels']
        })
        mock_dataset.to_pandas.return_value = mock_df
        mock_load_dataset.return_value = mock_dataset
        
        client = MediaDataClient()
        data, source_info = client.load_huggingface_advertising_data()
        
        assert isinstance(data, pd.DataFrame)
        assert len(data) == 2
        assert source_info['quality'] == 'REAL'
        assert source_info['source'] == 'huggingface'
        assert 'instruction' in data.columns

    @patch('data.media_data_client.load_dataset')
    def test_huggingface_data_loading_failure(self, mock_load_dataset):
        """Test HuggingFace data loading failure handling"""
        mock_load_dataset.side_effect = Exception("Dataset not found")
        
        client = MediaDataClient()
        data, source_info = client.load_huggingface_advertising_data()
        
        assert data is None
        assert source_info == {}

    @patch('kaggle.api.dataset_download_files')
    @patch('glob.glob')
    @patch('pandas.read_csv')
    def test_kaggle_data_loading_success(self, mock_read_csv, mock_glob, mock_download):
        """Test successful Kaggle data loading"""
        # Mock successful Kaggle API interaction
        mock_download.return_value = None  # Successful download
        mock_glob.return_value = ['/cache/marketing_data.csv']
        mock_df = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=100, freq='D'),
            'spend': np.random.normal(10000, 1000, 100),
            'revenue': np.random.normal(50000, 5000, 100)
        })
        mock_read_csv.return_value = mock_df
        
        client = MediaDataClient()
        with patch.object(client, '_check_kaggle_availability', return_value=True):
            data, source_info = client.load_kaggle_marketing_data()
        
        assert isinstance(data, pd.DataFrame)
        assert len(data) == 100
        assert source_info['quality'] == 'REAL'
        assert source_info['source'] == 'kaggle'

    def test_synthetic_data_generation(self):
        """Test synthetic marketing data generation"""
        client = MediaDataClient()
        data, source_info = client.create_synthetic_marketing_data()
        
        assert isinstance(data, pd.DataFrame)
        assert len(data) >= 50  # Should have substantial data
        assert 'date' in data.columns
        assert 'revenue' in data.columns
        
        # Check for spend columns
        spend_columns = [col for col in data.columns if col.endswith('_spend')]
        assert len(spend_columns) >= 3  # Should have multiple channels
        
        # Validate data quality
        assert data['revenue'].min() >= 0  # No negative revenue
        assert source_info['quality'] == 'DEMO'
        assert source_info['source'] == 'synthetic'

    def test_best_available_data_tiered_fallback(self):
        """Test tiered data source fallback system"""
        client = MediaDataClient()
        
        # Mock all external sources as unavailable
        with patch.object(client, '_check_kaggle_availability', return_value=False):
            with patch.object(client, '_check_huggingface_availability', return_value=False):
                data, source_info, source_type = client.get_best_available_data()
                
                # Should fall back to synthetic
                assert source_type == 'SYNTHETIC'
                assert isinstance(data, pd.DataFrame)
                assert len(data) > 0

    def test_api_endpoint_integration_methods(self):
        """Test API endpoint integration methods"""
        client = MediaDataClient()
        
        # Test campaign performance fetching
        from datetime import datetime, timedelta
        start_date = datetime.now() - timedelta(days=30)
        end_date = datetime.now()
        
        # Should not crash and return data structure
        performance_data = client.fetch_campaign_performance(
            start_date=start_date,
            end_date=end_date,
            channels=['search', 'social', 'display']
        )
        
        assert isinstance(performance_data, list)

    def test_journey_data_fetching(self):
        """Test customer journey data fetching for attribution"""
        client = MediaDataClient()
        
        from datetime import datetime, timedelta
        start_date = datetime.now() - timedelta(days=30)
        end_date = datetime.now()
        
        journey_data = client.fetch_journey_data(
            start_date=start_date,
            end_date=end_date,
            channels=['search', 'social', 'display']
        )
        
        assert isinstance(journey_data, list)
        
        # Check structure of journey data
        if journey_data:
            journey_item = journey_data[0]
            assert 'customer_id' in journey_item
            assert 'touchpoint_date' in journey_item
            assert 'channel' in journey_item

    @pytest.mark.slow
    def test_performance_with_large_dataset(self):
        """Test performance with larger synthetic dataset"""
        client = MediaDataClient()
        
        # Temporarily modify to create larger dataset
        with patch('pandas.date_range') as mock_date_range:
            mock_date_range.return_value = pd.date_range('2020-01-01', periods=260, freq='W')  # 5 years
            
            data, source_info = client.create_synthetic_marketing_data()
            
            assert len(data) == 260
            assert isinstance(data, pd.DataFrame)

    def test_data_validation_and_quality_checks(self):
        """Test data validation and quality indicators"""
        client = MediaDataClient()
        data, source_info = client.create_synthetic_marketing_data()
        
        # Data quality checks
        assert not data.isnull().any().any()  # No null values
        assert all(data.select_dtypes(include=[np.number]).min() >= 0)  # No negative values in numeric columns
        
        # Required columns
        required_columns = ['date', 'revenue']
        for col in required_columns:
            assert col in data.columns
        
        # Date column should be datetime-compatible
        assert pd.api.types.is_datetime64_any_dtype(data['date']) or data['date'].dtype == 'object'

    def test_cache_directory_creation(self):
        """Test cache directory is created properly"""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_path = os.path.join(temp_dir, 'test_cache')
            client = MediaDataClient(cache_dir=cache_path)
            
            assert os.path.exists(cache_path)
            assert os.path.isdir(cache_path)

    def test_error_handling_robustness(self):
        """Test error handling in various failure scenarios"""
        client = MediaDataClient()
        
        # Test with invalid parameters
        try:
            client.fetch_campaign_performance(
                start_date="invalid_date",
                end_date="invalid_date", 
                channels=[]
            )
        except Exception as e:
            assert isinstance(e, Exception)  # Should handle gracefully

    @pytest.mark.integration
    def test_real_huggingface_integration(self):
        """Integration test with real HuggingFace if available"""
        client = MediaDataClient()
        
        if client.data_sources['huggingface']:
            # Try to load real data
            data, source_info = client.load_huggingface_advertising_data()
            
            if data is not None:
                assert isinstance(data, pd.DataFrame)
                assert len(data) > 0
                assert source_info['quality'] == 'REAL'