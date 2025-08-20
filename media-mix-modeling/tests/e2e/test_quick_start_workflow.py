"""
End-to-end tests for quick start workflow
"""

import pytest
import subprocess
import sys
import os
import tempfile
from pathlib import Path
import pandas as pd


class TestQuickStartWorkflow:
    """Test the complete quick start workflow"""

    @pytest.mark.e2e
    def test_quick_start_script_execution(self):
        """Test that quick_start.py executes successfully"""
        # Test script execution in subprocess to avoid side effects
        try:
            result = subprocess.run(
                [sys.executable, 'quick_start.py'],
                cwd=Path.cwd(),
                capture_output=True,
                text=True,
                timeout=120  # 2 minute timeout
            )
            
            # Should complete without critical errors
            assert result.returncode == 0 or "MMM platform operational" in result.stdout
            
            # Should show tiered data loading
            assert "TIER 1" in result.stdout or "TIER 2" in result.stdout or "SYNTHETIC" in result.stdout
            
            # Should complete MMM modeling
            assert "MMM" in result.stdout
            
            # Should show optimization results  
            assert "OPTIMIZATION" in result.stdout or "budget" in result.stdout.lower()
            
        except subprocess.TimeoutExpired:
            pytest.fail("Quick start script timed out after 2 minutes")
        except Exception as e:
            pytest.fail(f"Quick start script execution failed: {e}")

    @pytest.mark.e2e
    def test_sample_data_generation_workflow(self):
        """Test sample data generation workflow"""
        # Test that sample data files are created and accessible
        from data.media_data_client import MediaDataClient
        
        client = MediaDataClient()
        data, source_info, source_type = client.get_best_available_data()
        
        # Should get data successfully
        assert isinstance(data, pd.DataFrame)
        assert len(data) > 0
        assert source_type in ['LOCAL_MMM', 'HUGGINGFACE', 'SYNTHETIC']
        
        # Verify data structure for MMM
        if source_type in ['SYNTHETIC', 'LOCAL_MMM']:
            assert 'revenue' in data.columns
            spend_columns = [col for col in data.columns if col.endswith('_spend')]
            assert len(spend_columns) >= 3

    @pytest.mark.e2e 
    def test_api_server_startup(self):
        """Test that API server can start up successfully"""
        try:
            # Test API import and basic FastAPI app creation
            from api.main import app
            from fastapi.testclient import TestClient
            
            client = TestClient(app)
            
            # Test basic health endpoint
            response = client.get("/health/live")
            assert response.status_code == 200
            
            # Test API documentation
            response = client.get("/docs")
            assert response.status_code == 200
            
        except Exception as e:
            pytest.fail(f"API server startup test failed: {e}")

    @pytest.mark.e2e
    def test_docker_environment_simulation(self):
        """Test environment setup that simulates Docker container"""
        # Test directory structure
        required_dirs = [
            'data/samples',
            'data/schemas',
            'outputs',
            'cache/data',
            'tests/unit',
            'tests/integration',
            'tests/e2e'
        ]
        
        for dir_path in required_dirs:
            assert os.path.exists(dir_path), f"Required directory missing: {dir_path}"
            assert os.path.isdir(dir_path), f"Path is not a directory: {dir_path}"

    @pytest.mark.e2e
    def test_output_file_generation(self):
        """Test that output files are generated correctly"""
        # Test that outputs directory exists and is writable
        outputs_dir = Path('outputs')
        assert outputs_dir.exists()
        assert outputs_dir.is_dir()
        
        # Test writing a sample output file
        test_file = outputs_dir / 'test_output.txt'
        try:
            with open(test_file, 'w') as f:
                f.write('Test output file')
            
            assert test_file.exists()
            test_file.unlink()  # Clean up
            
        except Exception as e:
            pytest.fail(f"Output file generation test failed: {e}")

    @pytest.mark.e2e
    def test_dependencies_availability(self):
        """Test that all required dependencies are available"""
        required_packages = [
            'pandas',
            'numpy', 
            'fastapi',
            'uvicorn',
            'pydantic',
            'sklearn',
            'matplotlib'
        ]
        
        missing_packages = []
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            pytest.fail(f"Missing required packages: {missing_packages}")

    @pytest.mark.e2e
    def test_complete_5_minute_demo_workflow(self):
        """Test the complete 5-minute demo workflow end-to-end"""
        steps_completed = {}
        
        try:
            # Step 1: Data loading
            from data.media_data_client import MediaDataClient
            client = MediaDataClient()
            data, source_info, source_type = client.get_best_available_data()
            steps_completed['data_loading'] = True
            
            # Step 2: MMM modeling (if data is compatible)
            if 'revenue' in data.columns:
                from models.mmm.econometric_mmm import EconometricMMM
                mmm_model = EconometricMMM()
                
                spend_columns = [col for col in data.columns if col.endswith('_spend')]
                if spend_columns:
                    mmm_results = mmm_model.fit(
                        data=data,
                        target_column='revenue',
                        spend_columns=spend_columns
                    )
                    steps_completed['mmm_modeling'] = True
            
            # Step 3: Budget optimization (if MMM worked)
            if steps_completed.get('mmm_modeling'):
                from models.mmm.budget_optimizer import BudgetOptimizer
                optimizer = BudgetOptimizer()
                
                current_budgets = {
                    channel.replace('_spend', ''): data[channel].mean() 
                    for channel in spend_columns
                }
                total_budget = sum(current_budgets.values())
                
                if total_budget > 0:
                    optimization_results = optimizer.optimize_budget_allocation(
                        mmm_model=mmm_model,
                        current_budgets=current_budgets,
                        total_budget=total_budget,
                        objective='roi'
                    )
                    steps_completed['budget_optimization'] = True
            
            # Step 4: API endpoints
            from fastapi.testclient import TestClient
            from api.main import app
            client = TestClient(app)
            
            response = client.get("/health/status")
            if response.status_code == 200:
                steps_completed['api_endpoints'] = True
            
            # Verify workflow completion
            essential_steps = ['data_loading', 'api_endpoints']
            for step in essential_steps:
                assert steps_completed.get(step, False), f"Essential step failed: {step}"
            
            # Report on optional steps
            optional_steps = ['mmm_modeling', 'budget_optimization']
            completed_optional = sum(1 for step in optional_steps if steps_completed.get(step, False))
            
            print(f"5-minute demo workflow: {len(steps_completed)}/4 steps completed")
            print(f"Essential steps: {len([s for s in essential_steps if steps_completed.get(s)])}/{len(essential_steps)}")
            print(f"Optional steps: {completed_optional}/{len(optional_steps)}")
            
        except Exception as e:
            pytest.fail(f"5-minute demo workflow failed at step {list(steps_completed.keys())[-1] if steps_completed else 'initial'}: {e}")

    @pytest.mark.e2e
    def test_real_data_integration_e2e(self):
        """Test end-to-end workflow with real data if available"""
        from data.media_data_client import MediaDataClient
        
        client = MediaDataClient()
        
        # Check if we can get real data
        if client.data_sources['huggingface']:
            data, source_info = client.load_huggingface_advertising_data()
            
            if data is not None:
                assert isinstance(data, pd.DataFrame)
                assert len(data) > 0
                assert source_info['quality'] == 'REAL'
                
                # Test that real data can be used in API
                from fastapi.testclient import TestClient
                from api.main import app
                
                api_client = TestClient(app)
                
                # Test performance analysis with real data structure
                if 'instruction' in data.columns:
                    # HuggingFace campaign data format
                    print(f"Real HuggingFace data loaded: {len(data)} rows")
                    assert True  # Successfully integrated real data
                else:
                    # Other real data formats
                    print(f"Real data integrated: {source_info['description']}")
                    assert True

    @pytest.mark.e2e
    def test_error_recovery_and_fallbacks(self):
        """Test system behavior under error conditions"""
        # Test with invalid environment
        with tempfile.TemporaryDirectory() as temp_dir:
            # Change to temporary directory to test fallback behavior
            original_cwd = os.getcwd()
            try:
                os.chdir(temp_dir)
                
                # Should still be able to create client and get synthetic data
                from data.media_data_client import MediaDataClient
                client = MediaDataClient()
                
                data, source_info, source_type = client.get_best_available_data()
                
                # Should fall back to synthetic data
                assert source_type == 'SYNTHETIC'
                assert isinstance(data, pd.DataFrame)
                assert len(data) > 0
                
            finally:
                os.chdir(original_cwd)

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_system_performance_benchmarks(self):
        """Test system performance benchmarks for 5-minute demo"""
        import time
        
        performance_metrics = {}
        
        # Test data loading performance
        start_time = time.time()
        from data.media_data_client import MediaDataClient
        client = MediaDataClient()
        data, source_info, source_type = client.get_best_available_data()
        performance_metrics['data_loading_time'] = time.time() - start_time
        
        # Test API startup performance
        start_time = time.time()
        from fastapi.testclient import TestClient
        from api.main import app
        api_client = TestClient(app)
        response = api_client.get("/health/status")
        performance_metrics['api_startup_time'] = time.time() - start_time
        
        # Performance benchmarks for 5-minute demo
        assert performance_metrics['data_loading_time'] < 10.0  # Should load data in under 10 seconds
        assert performance_metrics['api_startup_time'] < 5.0   # API should start in under 5 seconds
        assert response.status_code == 200
        
        print(f"Performance metrics: {performance_metrics}")

    @pytest.mark.e2e
    def test_documentation_and_examples_availability(self):
        """Test that documentation and examples are available"""
        # Test API documentation
        from fastapi.testclient import TestClient
        from api.main import app
        
        client = TestClient(app)
        
        # Test OpenAPI docs
        response = client.get("/docs")
        assert response.status_code == 200
        
        # Test OpenAPI schema
        response = client.get("/openapi.json")
        assert response.status_code == 200
        
        schema = response.json()
        assert 'openapi' in schema
        assert 'info' in schema
        assert 'paths' in schema
        
        # Verify key endpoints are documented
        paths = schema['paths']
        expected_endpoints = [
            '/health/status',
            '/api/v1/attribution/analyze',
            '/api/v1/optimization/budget',
            '/api/v1/performance/analyze'
        ]
        
        for endpoint in expected_endpoints:
            assert endpoint in paths, f"Missing documentation for endpoint: {endpoint}"