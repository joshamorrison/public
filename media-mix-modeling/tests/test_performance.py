"""
Performance and benchmarking tests for MMM platform
Tests model training speed, memory usage, and inference latency
"""

import pytest
import time
import psutil
import os
import pandas as pd
import numpy as np
from unittest.mock import patch

from models.mmm.econometric_mmm import EconometricMMM
from models.mmm.budget_optimizer import BudgetOptimizer
from data.media_data_client import MediaDataClient


class TestPerformance:
    """Performance benchmarking tests"""
    
    def test_model_training_performance(self, sample_marketing_data):
        """Test MMM model training performance benchmarks"""
        # Performance benchmarks
        MAX_TRAINING_TIME_SECONDS = 30
        MAX_MEMORY_USAGE_MB = 500
        
        # Monitor memory before training
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / 1024 / 1024
        
        # Time model training
        start_time = time.time()
        
        model = EconometricMMM()
        spend_columns = [col for col in sample_marketing_data.columns if col.endswith('_spend')]
        
        results = model.fit(
            data=sample_marketing_data,
            target_column='revenue',
            spend_columns=spend_columns
        )
        
        end_time = time.time()
        training_time = end_time - start_time
        
        # Monitor memory after training
        memory_after = process.memory_info().rss / 1024 / 1024
        memory_usage = memory_after - memory_before
        
        # Performance assertions
        assert training_time < MAX_TRAINING_TIME_SECONDS, f"Training took {training_time:.2f}s, expected < {MAX_TRAINING_TIME_SECONDS}s"
        assert memory_usage < MAX_MEMORY_USAGE_MB, f"Memory usage {memory_usage:.1f}MB, expected < {MAX_MEMORY_USAGE_MB}MB"
        
        # Model quality assertions
        assert results['performance']['r2_score'] > 0, "Model should have positive R² score"
        assert results['performance']['mape'] < 50, "Model MAPE should be reasonable"
    
    def test_prediction_latency(self, sample_marketing_data):
        """Test model prediction latency benchmarks"""
        MAX_PREDICTION_TIME_MS = 100
        
        # Train model
        model = EconometricMMM()
        spend_columns = [col for col in sample_marketing_data.columns if col.endswith('_spend')]
        
        model.fit(
            data=sample_marketing_data,
            target_column='revenue',
            spend_columns=spend_columns
        )
        
        # Prepare test data
        test_data = sample_marketing_data[spend_columns].iloc[:1]
        
        # Benchmark prediction latency
        latencies = []
        for _ in range(100):
            start_time = time.time()
            prediction = model.predict(test_data)
            end_time = time.time()
            latencies.append((end_time - start_time) * 1000)  # Convert to ms
        
        mean_latency = np.mean(latencies)
        p95_latency = np.percentile(latencies, 95)
        
        assert mean_latency < MAX_PREDICTION_TIME_MS, f"Mean prediction latency {mean_latency:.2f}ms > {MAX_PREDICTION_TIME_MS}ms"
        assert p95_latency < MAX_PREDICTION_TIME_MS * 2, f"P95 prediction latency {p95_latency:.2f}ms too high"
    
    def test_budget_optimization_performance(self, sample_marketing_data):
        """Test budget optimization performance"""
        MAX_OPTIMIZATION_TIME_SECONDS = 10
        
        # Train model
        model = EconometricMMM()
        spend_columns = [col for col in sample_marketing_data.columns if col.endswith('_spend')]
        
        model.fit(
            data=sample_marketing_data,
            target_column='revenue', 
            spend_columns=spend_columns
        )
        
        # Time budget optimization
        optimizer = BudgetOptimizer(objective='roi', max_iterations=100)
        
        start_time = time.time()
        
        result = optimizer.optimize(
            mmm_model=model,
            total_budget=100000
        )
        
        end_time = time.time()
        optimization_time = end_time - start_time
        
        assert optimization_time < MAX_OPTIMIZATION_TIME_SECONDS, f"Optimization took {optimization_time:.2f}s"
        assert 'allocation' in result, "Optimization should return allocation"
        assert 'projected_roi' in result, "Optimization should return projected ROI"
    
    def test_data_loading_performance(self):
        """Test data loading performance across sources"""
        MAX_LOAD_TIME_SECONDS = 5
        
        client = MediaDataClient()
        
        start_time = time.time()
        data, info, source_type = client.get_best_available_data()
        end_time = time.time()
        
        load_time = end_time - start_time
        
        assert load_time < MAX_LOAD_TIME_SECONDS, f"Data loading took {load_time:.2f}s"
        assert len(data) > 0, "Should load non-empty dataset"
        assert 'revenue' in data.columns, "Should include revenue column"
    
    def test_memory_usage_scaling(self):
        """Test memory usage with different data sizes"""
        data_sizes = [100, 500, 1000]
        memory_usages = []
        
        for size in data_sizes:
            # Generate test data of specific size
            test_data = pd.DataFrame({
                'revenue': np.random.normal(50000, 10000, size),
                'tv_spend': np.random.normal(10000, 2000, size),
                'digital_spend': np.random.normal(8000, 1500, size),
                'radio_spend': np.random.normal(3000, 500, size)
            })
            
            # Monitor memory usage
            process = psutil.Process(os.getpid())
            memory_before = process.memory_info().rss / 1024 / 1024
            
            # Train model
            model = EconometricMMM()
            model.fit(
                data=test_data,
                target_column='revenue',
                spend_columns=['tv_spend', 'digital_spend', 'radio_spend']
            )
            
            memory_after = process.memory_info().rss / 1024 / 1024
            memory_usage = memory_after - memory_before
            memory_usages.append(memory_usage)
            
            # Clean up
            del model, test_data
        
        # Memory usage should scale reasonably
        # Should not grow exponentially with data size
        memory_ratio = memory_usages[-1] / memory_usages[0]
        data_ratio = data_sizes[-1] / data_sizes[0]
        
        assert memory_ratio < data_ratio * 2, "Memory usage scaling should be reasonable"


class TestConcurrency:
    """Test concurrent operations and thread safety"""
    
    def test_concurrent_predictions(self, sample_marketing_data):
        """Test thread safety of model predictions"""
        import threading
        import queue
        
        # Train model
        model = EconometricMMM()
        spend_columns = [col for col in sample_marketing_data.columns if col.endswith('_spend')]
        
        model.fit(
            data=sample_marketing_data,
            target_column='revenue',
            spend_columns=spend_columns
        )
        
        # Prepare test data
        test_data = sample_marketing_data[spend_columns].iloc[:5]
        
        # Results queue
        results = queue.Queue()
        errors = queue.Queue()
        
        def make_predictions():
            try:
                for _ in range(10):
                    prediction = model.predict(test_data)
                    results.put(prediction)
            except Exception as e:
                errors.put(e)
        
        # Start multiple threads
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=make_predictions)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # Check results
        assert errors.empty(), f"Concurrent predictions failed: {list(errors.queue)}"
        assert not results.empty(), "Should have prediction results"
        
        # All predictions should be consistent
        all_predictions = list(results.queue)
        first_prediction = all_predictions[0]
        
        for prediction in all_predictions[1:]:
            np.testing.assert_array_almost_equal(
                prediction, first_prediction, decimal=5,
                err_msg="Concurrent predictions should be identical"
            )


class TestScalability:
    """Test system scalability with larger datasets"""
    
    @pytest.mark.slow
    def test_large_dataset_handling(self):
        """Test MMM with large datasets (marked as slow test)"""
        # Generate large synthetic dataset
        large_size = 5000  # ~100 years of daily data
        
        large_data = pd.DataFrame({
            'revenue': np.random.normal(50000, 10000, large_size),
            'tv_spend': np.random.normal(10000, 2000, large_size),
            'digital_spend': np.random.normal(8000, 1500, large_size),
            'radio_spend': np.random.normal(3000, 500, large_size),
            'print_spend': np.random.normal(2000, 400, large_size),
            'social_spend': np.random.normal(5000, 1000, large_size)
        })
        
        # Should handle large dataset without errors
        model = EconometricMMM()
        spend_columns = [col for col in large_data.columns if col.endswith('_spend')]
        
        start_time = time.time()
        results = model.fit(
            data=large_data,
            target_column='revenue',
            spend_columns=spend_columns
        )
        end_time = time.time()
        
        training_time = end_time - start_time
        
        # Should complete within reasonable time (even for large dataset)
        assert training_time < 120, f"Large dataset training took {training_time:.2f}s"
        assert results['performance']['r2_score'] >= 0, "Should produce valid model"
    
    @pytest.mark.slow 
    def test_many_channels_handling(self):
        """Test MMM with many marketing channels"""
        # Generate data with many channels
        num_channels = 20
        data_size = 500
        
        # Base data
        multi_channel_data = pd.DataFrame({
            'revenue': np.random.normal(100000, 20000, data_size)
        })
        
        # Add many channels
        for i in range(num_channels):
            channel_name = f'channel_{i:02d}_spend'
            multi_channel_data[channel_name] = np.random.normal(5000, 1000, data_size)
        
        # Should handle many channels
        model = EconometricMMM(regularization_alpha=0.5)  # Higher regularization for many features
        spend_columns = [col for col in multi_channel_data.columns if col.endswith('_spend')]
        
        results = model.fit(
            data=multi_channel_data,
            target_column='revenue',
            spend_columns=spend_columns
        )
        
        assert len(spend_columns) == num_channels, f"Should handle {num_channels} channels"
        assert results['performance']['r2_score'] >= 0, "Should produce valid model with many channels"


def benchmark_mmm_training(data_size=1000, num_iterations=5):
    """
    Benchmark function for MMM training performance
    Can be called from other modules for performance testing
    """
    results = {
        'training_times': [],
        'memory_usages': [],
        'prediction_latencies': []
    }
    
    for i in range(num_iterations):
        # Generate test data
        test_data = pd.DataFrame({
            'revenue': np.random.normal(50000, 10000, data_size),
            'tv_spend': np.random.normal(10000, 2000, data_size),
            'digital_spend': np.random.normal(8000, 1500, data_size),
            'radio_spend': np.random.normal(3000, 500, data_size)
        })
        
        # Monitor memory
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / 1024 / 1024
        
        # Time training
        start_time = time.time()
        
        model = EconometricMMM()
        results_mmm = model.fit(
            data=test_data,
            target_column='revenue',
            spend_columns=['tv_spend', 'digital_spend', 'radio_spend']
        )
        
        training_time = time.time() - start_time
        
        # Memory usage
        memory_after = process.memory_info().rss / 1024 / 1024
        memory_usage = memory_after - memory_before
        
        # Prediction latency
        test_prediction_data = test_data[['tv_spend', 'digital_spend', 'radio_spend']].iloc[:1]
        
        pred_start = time.time()
        prediction = model.predict(test_prediction_data)
        pred_latency = (time.time() - pred_start) * 1000  # Convert to ms
        
        # Store results
        results['training_times'].append(training_time)
        results['memory_usages'].append(memory_usage)
        results['prediction_latencies'].append(pred_latency)
    
    # Calculate summary statistics
    return {
        'training_time': np.mean(results['training_times']),
        'training_time_std': np.std(results['training_times']),
        'memory_mb': np.mean(results['memory_usages']),
        'memory_mb_std': np.std(results['memory_usages']),
        'prediction_latency_ms': np.mean(results['prediction_latencies']),
        'prediction_latency_ms_std': np.std(results['prediction_latencies']),
        'iterations': num_iterations,
        'data_size': data_size
    }


if __name__ == "__main__":
    # Run benchmark when script is executed directly
    print("Running MMM Performance Benchmark...")
    
    benchmark_results = benchmark_mmm_training(data_size=1000, num_iterations=3)
    
    print(f"""
    Performance Benchmark Results:
    ==============================
    Training Time: {benchmark_results['training_time']:.2f} ± {benchmark_results['training_time_std']:.2f} seconds
    Memory Usage: {benchmark_results['memory_mb']:.1f} ± {benchmark_results['memory_mb_std']:.1f} MB
    Prediction Latency: {benchmark_results['prediction_latency_ms']:.2f} ± {benchmark_results['prediction_latency_ms_std']:.2f} ms
    Data Size: {benchmark_results['data_size']} records
    Iterations: {benchmark_results['iterations']}
    """)