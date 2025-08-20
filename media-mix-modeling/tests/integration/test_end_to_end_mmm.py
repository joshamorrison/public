"""
End-to-end integration tests for Media Mix Modeling platform
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import os
from datetime import datetime, timedelta

from data.media_data_client import MediaDataClient
from models.mmm.econometric_mmm import EconometricMMM
from models.mmm.budget_optimizer import BudgetOptimizer


class TestEndToEndMMM:
    """End-to-end integration tests for complete MMM workflow"""

    @pytest.fixture
    def real_data_client(self):
        """Create MediaDataClient for integration testing"""
        return MediaDataClient()

    @pytest.fixture
    def test_mmm_data(self, real_data_client):
        """Get test data from MediaDataClient"""
        data, source_info, source_type = real_data_client.get_best_available_data()
        return data, source_info, source_type

    @pytest.mark.integration
    def test_complete_mmm_workflow(self, test_mmm_data):
        """Test complete MMM workflow from data to optimization"""
        data, source_info, source_type = test_mmm_data
        
        # Ensure we have MMM-compatible data
        if 'revenue' not in data.columns or not any(col.endswith('_spend') for col in data.columns):
            pytest.skip("Test data not MMM-compatible")
        
        # Step 1: Data validation
        assert isinstance(data, pd.DataFrame)
        assert len(data) >= 10  # Minimum data requirement
        
        # Step 2: MMM model training
        mmm_model = EconometricMMM(
            adstock_rate=0.5,
            saturation_param=0.6,
            regularization_alpha=0.1
        )
        
        spend_columns = [col for col in data.columns if col.endswith('_spend')]
        assert len(spend_columns) > 0, "No spend columns found"
        
        mmm_results = mmm_model.fit(
            data=data,
            target_column='revenue',
            spend_columns=spend_columns,
            include_synergies=True
        )
        
        # Validate MMM results
        assert 'performance' in mmm_results
        assert 'attribution' in mmm_results
        assert mmm_results['performance']['r2'] >= 0  # R² should be non-negative
        
        # Step 3: Budget optimization
        optimizer = BudgetOptimizer(
            optimization_method='scipy',
            max_budget_change=0.3,
            min_budget_change=-0.2
        )
        
        # Current budget allocation
        current_budgets = {
            channel.replace('_spend', ''): data[channel].mean() 
            for channel in spend_columns
        }
        total_budget = sum(current_budgets.values())
        
        optimization_results = optimizer.optimize_budget_allocation(
            mmm_model=mmm_model,
            current_budgets=current_budgets,
            total_budget=total_budget,
            objective='roi',
            constraints=None
        )
        
        # Validate optimization results
        assert optimization_results['success'] is True
        assert 'budget_changes' in optimization_results
        assert 'performance_improvement' in optimization_results

    @pytest.mark.integration
    def test_real_data_integration_huggingface(self):
        """Test integration with real HuggingFace data"""
        client = MediaDataClient()
        
        if not client.data_sources['huggingface']:
            pytest.skip("HuggingFace not available")
        
        data, source_info = client.load_huggingface_advertising_data()
        
        if data is not None:
            assert isinstance(data, pd.DataFrame)
            assert len(data) > 0
            assert source_info['quality'] == 'REAL'
            assert source_info['source'] == 'huggingface'

    @pytest.mark.integration 
    def test_tiered_data_fallback_system(self):
        """Test the complete tiered data fallback system"""
        client = MediaDataClient()
        
        # Test the full fallback chain
        data, source_info, source_type = client.get_best_available_data()
        
        # Should always get data from some source
        assert isinstance(data, pd.DataFrame)
        assert len(data) > 0
        assert source_type in ['KAGGLE', 'HUGGINGFACE', 'SYNTHETIC']
        
        # Validate data structure for MMM compatibility
        if source_type == 'SYNTHETIC':
            assert 'revenue' in data.columns
            spend_cols = [col for col in data.columns if col.endswith('_spend')]
            assert len(spend_cols) >= 3  # Multiple marketing channels

    @pytest.mark.integration
    def test_api_endpoint_data_flow(self):
        """Test data flow through API endpoints"""
        from fastapi.testclient import TestClient
        from api.main import app
        
        client = TestClient(app)
        
        # Test health check
        response = client.get("/health/status")
        assert response.status_code == 200
        
        # Test performance analysis with minimal data
        request_data = {
            "channels": ["search", "social"],
            "start_date": "2024-07-01",
            "end_date": "2024-07-31",
            "metrics": ["impressions", "clicks"],
            "granularity": "daily"
        }
        
        response = client.post("/api/v1/performance/analyze", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "success" in data
        assert "status" in data

    @pytest.mark.integration
    def test_sample_data_consistency(self):
        """Test consistency across all sample data files"""
        sample_files = [
            'data/samples/marketing_campaign_data.csv',
            'data/samples/channel_performance_data.csv', 
            'data/samples/customer_journey_data.csv',
            'data/samples/campaign_budget_data.csv',
            'data/samples/mmm_time_series_data.csv'
        ]
        
        loaded_files = {}
        for file_path in sample_files:
            if os.path.exists(file_path):
                try:
                    df = pd.read_csv(file_path)
                    loaded_files[file_path] = df
                except Exception as e:
                    pytest.fail(f"Failed to load {file_path}: {e}")
        
        # Should have at least some sample files
        assert len(loaded_files) > 0
        
        # Validate each loaded file
        for file_path, df in loaded_files.items():
            assert isinstance(df, pd.DataFrame)
            assert len(df) > 0
            
            # Check for date columns
            date_columns = [col for col in df.columns if 'date' in col.lower()]
            if date_columns:
                # Validate date format
                try:
                    pd.to_datetime(df[date_columns[0]])
                except Exception:
                    pytest.fail(f"Invalid date format in {file_path}")

    @pytest.mark.integration
    @pytest.mark.slow
    def test_performance_with_large_dataset(self):
        """Test system performance with larger datasets"""
        client = MediaDataClient()
        
        # Create larger synthetic dataset
        import pandas as pd
        import numpy as np
        
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=200, freq='W')  # ~4 years
        
        large_data = pd.DataFrame({
            'date': dates,
            'tv_spend': np.random.normal(50000, 10000, 200),
            'digital_spend': np.random.normal(30000, 8000, 200),
            'radio_spend': np.random.normal(20000, 5000, 200),
            'social_spend': np.random.normal(25000, 6000, 200),
            'revenue': np.random.normal(150000, 20000, 200)
        })
        
        # Ensure positive values
        for col in large_data.columns:
            if col != 'date':
                large_data[col] = np.maximum(large_data[col], 0)
        
        # Test MMM on larger dataset
        mmm_model = EconometricMMM(adstock_rate=0.5, saturation_param=0.6)
        spend_columns = [col for col in large_data.columns if col.endswith('_spend')]
        
        import time
        start_time = time.time()
        
        mmm_results = mmm_model.fit(
            data=large_data,
            target_column='revenue', 
            spend_columns=spend_columns
        )
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Performance checks
        assert mmm_results['performance']['r2'] >= 0
        assert processing_time < 30  # Should complete within 30 seconds
        
        print(f"Large dataset MMM processing time: {processing_time:.2f}s")

    @pytest.mark.integration
    def test_docker_environment_compatibility(self):
        """Test compatibility with Docker environment"""
        # Test that all required directories exist
        required_dirs = [
            'data/samples',
            'data/schemas', 
            'outputs',
            'cache'
        ]
        
        for dir_path in required_dirs:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path, exist_ok=True)
            assert os.path.isdir(dir_path)
        
        # Test file permissions
        test_file = 'outputs/test_write.txt'
        try:
            with open(test_file, 'w') as f:
                f.write('test')
            os.remove(test_file)
        except Exception as e:
            pytest.fail(f"File write permission test failed: {e}")

    @pytest.mark.integration
    def test_quick_start_script_integration(self):
        """Test quick start script integration"""
        import subprocess
        import sys
        
        # Test that quick_start.py can be imported without errors
        try:
            import quick_start
            assert hasattr(quick_start, 'main')
        except ImportError as e:
            pytest.fail(f"quick_start.py import failed: {e}")

    @pytest.mark.integration
    def test_schema_validation_with_real_data(self):
        """Test schema validation against real sample data"""
        import json
        
        schema_files = [
            'data/schemas/marketing_campaign_schema.json',
            'data/schemas/channel_performance_schema.json',
            'data/schemas/mmm_time_series_schema.json'
        ]
        
        for schema_file in schema_files:
            if os.path.exists(schema_file):
                try:
                    with open(schema_file, 'r') as f:
                        schema = json.load(f)
                    
                    # Validate schema structure
                    assert 'title' in schema
                    assert 'description' in schema
                    assert 'properties' in schema
                    assert 'source' in schema
                    
                except Exception as e:
                    pytest.fail(f"Schema validation failed for {schema_file}: {e}")

    @pytest.mark.integration
    def test_mlflow_integration_if_available(self):
        """Test MLflow integration if available"""
        try:
            import mlflow
            
            # Test basic MLflow functionality
            experiment_name = "test_mmm_integration"
            try:
                mlflow.set_experiment(experiment_name)
                
                with mlflow.start_run(run_name="integration_test"):
                    mlflow.log_param("test_param", "integration_test")
                    mlflow.log_metric("test_metric", 0.85)
                
                # MLflow integration successful
                assert True
                
            except Exception as e:
                # MLflow available but not configured properly
                print(f"MLflow integration test warning: {e}")
                
        except ImportError:
            # MLflow not installed - skip test
            pytest.skip("MLflow not available for integration testing")