"""
Performance tests for Media Mix Modeling platform
"""

import pytest
import time
import psutil
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from pathlib import Path

from data.media_data_client import MediaDataClient
from models.mmm.econometric_mmm import EconometricMMM
from models.mmm.budget_optimizer import BudgetOptimizer


class TestSystemPerformance:
    """Performance tests for MMM platform components"""

    @pytest.fixture
    def large_dataset(self):
        """Generate large dataset for performance testing"""
        np.random.seed(42)
        n_weeks = 260  # 5 years of weekly data
        
        dates = pd.date_range('2019-01-01', periods=n_weeks, freq='W')
        
        data = pd.DataFrame({
            'date': dates,
            'tv_spend': np.random.normal(50000, 10000, n_weeks),
            'digital_spend': np.random.normal(30000, 8000, n_weeks), 
            'radio_spend': np.random.normal(20000, 5000, n_weeks),
            'print_spend': np.random.normal(15000, 3000, n_weeks),
            'social_spend': np.random.normal(25000, 6000, n_weeks),
            'ott_spend': np.random.normal(18000, 4000, n_weeks),
            'podcast_spend': np.random.normal(12000, 2500, n_weeks),
            'influencer_spend': np.random.normal(8000, 2000, n_weeks)
        })
        
        # Generate realistic revenue with media mix effects
        spend_cols = [col for col in data.columns if col.endswith('_spend')]
        revenue_base = 100000
        
        # Media effects with diminishing returns
        revenue_effects = sum(
            np.random.uniform(0.3, 1.2) * np.sqrt(data[col] / 1000)
            for col in spend_cols
        )
        
        data['revenue'] = (
            revenue_base + 
            revenue_effects + 
            np.random.normal(0, 10000, n_weeks)
        ).clip(lower=0)
        
        data['conversions'] = (data['revenue'] / 45).astype(int)
        
        # Ensure no negative values
        for col in data.columns:
            if col != 'date':
                data[col] = np.maximum(data[col], 0)
        
        return data

    @pytest.mark.performance
    def test_data_loading_performance(self):
        """Test data loading performance across different sources"""
        client = MediaDataClient()
        
        # Test synthetic data generation performance
        start_time = time.time()
        data, source_info = client.create_synthetic_marketing_data()
        synthetic_time = time.time() - start_time
        
        assert synthetic_time < 2.0, f"Synthetic data generation too slow: {synthetic_time:.2f}s"
        assert len(data) > 0
        
        # Test best available data performance
        start_time = time.time()
        data, source_info, source_type = client.get_best_available_data()
        fallback_time = time.time() - start_time
        
        assert fallback_time < 5.0, f"Data fallback system too slow: {fallback_time:.2f}s"
        
        print(f"Performance metrics:")
        print(f"  Synthetic generation: {synthetic_time:.3f}s")
        print(f"  Fallback system: {fallback_time:.3f}s")

    @pytest.mark.performance
    def test_mmm_model_performance(self, large_dataset):
        """Test MMM model training performance with large dataset"""
        mmm_model = EconometricMMM(
            adstock_rate=0.5,
            saturation_param=0.6,
            regularization_alpha=0.1
        )
        
        spend_columns = [col for col in large_dataset.columns if col.endswith('_spend')]
        
        # Measure training time
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        mmm_results = mmm_model.fit(
            data=large_dataset,
            target_column='revenue',
            spend_columns=spend_columns,
            include_synergies=True
        )
        
        training_time = time.time() - start_time
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        memory_used = end_memory - start_memory
        
        # Performance assertions
        assert training_time < 30.0, f"MMM training too slow: {training_time:.2f}s"
        assert memory_used < 500, f"MMM training uses too much memory: {memory_used:.1f}MB"
        assert mmm_results['performance']['r2'] >= 0
        
        print(f"MMM Performance metrics:")
        print(f"  Training time: {training_time:.3f}s")
        print(f"  Memory usage: {memory_used:.1f}MB")
        print(f"  R²: {mmm_results['performance']['r2']:.3f}")
        print(f"  Data size: {len(large_dataset)} observations, {len(spend_columns)} channels")

    @pytest.mark.performance
    def test_budget_optimization_performance(self, large_dataset):
        """Test budget optimization performance"""
        # First train MMM model
        mmm_model = EconometricMMM(adstock_rate=0.5, saturation_param=0.6)
        spend_columns = [col for col in large_dataset.columns if col.endswith('_spend')]
        
        mmm_results = mmm_model.fit(
            data=large_dataset,
            target_column='revenue', 
            spend_columns=spend_columns
        )
        
        # Test optimization performance
        optimizer = BudgetOptimizer(
            optimization_method='scipy',
            max_budget_change=0.3,
            min_budget_change=-0.2
        )
        
        current_budgets = {
            channel.replace('_spend', ''): large_dataset[channel].mean()
            for channel in spend_columns
        }
        total_budget = sum(current_budgets.values())
        
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        optimization_results = optimizer.optimize_budget_allocation(
            mmm_model=mmm_model,
            current_budgets=current_budgets,
            total_budget=total_budget,
            objective='roi',
            constraints=None
        )
        
        optimization_time = time.time() - start_time
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024
        memory_used = end_memory - start_memory
        
        # Performance assertions
        assert optimization_time < 15.0, f"Budget optimization too slow: {optimization_time:.2f}s"
        assert memory_used < 200, f"Optimization uses too much memory: {memory_used:.1f}MB"
        assert optimization_results['success'] is True
        
        print(f"Optimization Performance metrics:")
        print(f"  Optimization time: {optimization_time:.3f}s")
        print(f"  Memory usage: {memory_used:.1f}MB")
        print(f"  Channels optimized: {len(current_budgets)}")

    @pytest.mark.performance
    def test_api_endpoint_performance(self):
        """Test API endpoint response time performance"""
        from fastapi.testclient import TestClient
        from api.main import app
        
        client = TestClient(app)
        
        # Test health endpoint performance
        response_times = []
        for _ in range(10):
            start_time = time.time()
            response = client.get("/health/status")
            response_time = time.time() - start_time
            response_times.append(response_time)
            
            assert response.status_code == 200
        
        avg_response_time = np.mean(response_times)
        max_response_time = np.max(response_times)
        
        assert avg_response_time < 1.0, f"Average health check too slow: {avg_response_time:.3f}s"
        assert max_response_time < 2.0, f"Max health check too slow: {max_response_time:.3f}s"
        
        print(f"API Performance metrics:")
        print(f"  Health check avg: {avg_response_time:.3f}s")
        print(f"  Health check max: {max_response_time:.3f}s")

    @pytest.mark.performance
    def test_concurrent_api_requests(self):
        """Test API performance under concurrent load"""
        from fastapi.testclient import TestClient
        from api.main import app
        
        client = TestClient(app)
        
        def make_request():
            start_time = time.time()
            response = client.get("/health/status")
            response_time = time.time() - start_time
            return response.status_code, response_time
        
        # Test with 20 concurrent requests
        num_requests = 20
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_request) for _ in range(num_requests)]
            results = [future.result() for future in as_completed(futures)]
        
        total_time = time.time() - start_time
        
        # Analyze results
        status_codes = [result[0] for result in results]
        response_times = [result[1] for result in results]
        
        success_rate = sum(1 for code in status_codes if code == 200) / len(status_codes)
        avg_response_time = np.mean(response_times)
        max_response_time = np.max(response_times)
        
        # Performance assertions
        assert success_rate >= 0.95, f"Success rate too low: {success_rate:.2%}"
        assert avg_response_time < 2.0, f"Concurrent avg response time too slow: {avg_response_time:.3f}s"
        assert total_time < 10.0, f"Total concurrent test time too slow: {total_time:.3f}s"
        
        print(f"Concurrent API Performance:")
        print(f"  Requests: {num_requests}")
        print(f"  Success rate: {success_rate:.2%}")
        print(f"  Avg response time: {avg_response_time:.3f}s")
        print(f"  Max response time: {max_response_time:.3f}s")
        print(f"  Total time: {total_time:.3f}s")

    @pytest.mark.performance
    def test_memory_usage_stability(self, large_dataset):
        """Test memory usage stability over multiple operations"""
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        # Perform multiple MMM operations
        for i in range(5):
            mmm_model = EconometricMMM(adstock_rate=0.5, saturation_param=0.6)
            spend_columns = [col for col in large_dataset.columns if col.endswith('_spend')]
            
            mmm_results = mmm_model.fit(
                data=large_dataset,
                target_column='revenue',
                spend_columns=spend_columns
            )
            
            # Check memory after each iteration
            current_memory = psutil.Process().memory_info().rss / 1024 / 1024
            memory_growth = current_memory - initial_memory
            
            # Should not grow beyond reasonable limits
            assert memory_growth < 1000, f"Memory leak detected: {memory_growth:.1f}MB growth"
        
        final_memory = psutil.Process().memory_info().rss / 1024 / 1024
        total_growth = final_memory - initial_memory
        
        print(f"Memory Stability metrics:")
        print(f"  Initial memory: {initial_memory:.1f}MB")
        print(f"  Final memory: {final_memory:.1f}MB")
        print(f"  Total growth: {total_growth:.1f}MB")

    @pytest.mark.performance
    def test_disk_io_performance(self):
        """Test disk I/O performance for data operations"""
        # Test sample data loading performance
        sample_files = [
            'data/samples/marketing_campaign_data.csv',
            'data/samples/channel_performance_data.csv',
            'data/samples/mmm_time_series_data.csv'
        ]
        
        load_times = []
        file_sizes = []
        
        for file_path in sample_files:
            if Path(file_path).exists():
                # Measure file size
                file_size = Path(file_path).stat().st_size / 1024  # KB
                file_sizes.append(file_size)
                
                # Measure load time
                start_time = time.time()
                df = pd.read_csv(file_path)
                load_time = time.time() - start_time
                load_times.append(load_time)
                
                assert isinstance(df, pd.DataFrame)
                assert len(df) > 0
        
        if load_times:
            avg_load_time = np.mean(load_times)
            total_size = sum(file_sizes)
            
            # Performance assertions
            assert avg_load_time < 1.0, f"File loading too slow: {avg_load_time:.3f}s average"
            
            print(f"Disk I/O Performance:")
            print(f"  Files loaded: {len(load_times)}")
            print(f"  Avg load time: {avg_load_time:.3f}s")
            print(f"  Total data size: {total_size:.1f}KB")

    @pytest.mark.performance
    @pytest.mark.slow
    def test_stress_test_large_workload(self):
        """Stress test with large workload simulation"""
        # Simulate realistic production workload
        client = MediaDataClient()
        
        # Generate multiple large datasets
        datasets = []
        for i in range(3):
            data, _ = client.create_synthetic_marketing_data()
            datasets.append(data)
        
        # Process each dataset
        processing_times = []
        for i, data in enumerate(datasets):
            start_time = time.time()
            
            # MMM modeling
            mmm_model = EconometricMMM(adstock_rate=0.5, saturation_param=0.6)
            spend_columns = [col for col in data.columns if col.endswith('_spend')]
            
            mmm_results = mmm_model.fit(
                data=data,
                target_column='revenue',
                spend_columns=spend_columns
            )
            
            # Budget optimization
            optimizer = BudgetOptimizer()
            current_budgets = {
                channel.replace('_spend', ''): data[channel].mean()
                for channel in spend_columns
            }
            total_budget = sum(current_budgets.values())
            
            optimization_results = optimizer.optimize_budget_allocation(
                mmm_model=mmm_model,
                current_budgets=current_budgets,
                total_budget=total_budget,
                objective='roi'
            )
            
            processing_time = time.time() - start_time
            processing_times.append(processing_time)
            
            assert mmm_results['performance']['r2'] >= 0
            assert optimization_results['success'] is True
        
        avg_processing_time = np.mean(processing_times)
        max_processing_time = np.max(processing_times)
        
        # Stress test assertions
        assert avg_processing_time < 60.0, f"Stress test avg time too slow: {avg_processing_time:.1f}s"
        assert max_processing_time < 120.0, f"Stress test max time too slow: {max_processing_time:.1f}s"
        
        print(f"Stress Test Performance:")
        print(f"  Datasets processed: {len(datasets)}")
        print(f"  Avg processing time: {avg_processing_time:.1f}s")
        print(f"  Max processing time: {max_processing_time:.1f}s")

    @pytest.mark.performance
    def test_quick_start_performance_benchmark(self):
        """Benchmark the 5-minute quick start performance"""
        total_start_time = time.time()
        step_times = {}
        
        # Step 1: Data loading
        step_start = time.time()
        client = MediaDataClient()
        data, source_info, source_type = client.get_best_available_data()
        step_times['data_loading'] = time.time() - step_start
        
        # Step 2: MMM (if data compatible)
        if 'revenue' in data.columns:
            step_start = time.time()
            mmm_model = EconometricMMM()
            spend_columns = [col for col in data.columns if col.endswith('_spend')]
            
            if spend_columns:
                mmm_results = mmm_model.fit(
                    data=data,
                    target_column='revenue',
                    spend_columns=spend_columns
                )
                step_times['mmm_modeling'] = time.time() - step_start
        
        # Step 3: API startup
        step_start = time.time()
        from fastapi.testclient import TestClient
        from api.main import app
        api_client = TestClient(app)
        response = api_client.get("/health/status")
        step_times['api_startup'] = time.time() - step_start
        
        total_time = time.time() - total_start_time
        
        # 5-minute benchmark assertions
        assert step_times['data_loading'] < 30.0, "Data loading exceeds 30s for 5-min demo"
        assert step_times.get('mmm_modeling', 0) < 120.0, "MMM modeling exceeds 2min for 5-min demo"
        assert step_times['api_startup'] < 10.0, "API startup exceeds 10s for 5-min demo"
        assert total_time < 300.0, "Total time exceeds 5-minute demo target"
        assert response.status_code == 200
        
        print(f"5-Minute Demo Performance Benchmark:")
        print(f"  Data loading: {step_times['data_loading']:.1f}s")
        if 'mmm_modeling' in step_times:
            print(f"  MMM modeling: {step_times['mmm_modeling']:.1f}s")
        print(f"  API startup: {step_times['api_startup']:.1f}s")
        print(f"  Total time: {total_time:.1f}s")
        print(f"  5-min target: {'✓ PASS' if total_time < 300 else '✗ FAIL'}")