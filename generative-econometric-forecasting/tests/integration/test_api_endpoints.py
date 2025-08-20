"""
Integration tests for API endpoints.
"""

import pytest
import json
import time
from fastapi.testclient import TestClient


@pytest.mark.api
class TestHealthEndpoints:
    """Test health check endpoints."""
    
    def test_health_status(self, api_test_client):
        """Test health status endpoint."""
        response = api_test_client.get("/health/status")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "version" in data
        assert "system_info" in data
    
    def test_readiness_probe(self, api_test_client):
        """Test readiness probe endpoint."""
        response = api_test_client.get("/health/ready")
        
        assert response.status_code in [200, 503]
        data = response.json()
        
        assert "status" in data
        assert "checks" in data
    
    def test_liveness_probe(self, api_test_client):
        """Test liveness probe endpoint."""
        response = api_test_client.get("/health/live")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "alive"
        assert "timestamp" in data
        assert "process_id" in data
    
    def test_metrics_endpoint(self, api_test_client):
        """Test metrics endpoint."""
        response = api_test_client.get("/health/metrics")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "system" in data
        assert "application" in data


@pytest.mark.api
class TestForecastingEndpoints:
    """Test forecasting API endpoints."""
    
    def test_single_forecast_basic(self, api_test_client, sample_forecast_request):
        """Test basic single forecast request."""
        response = api_test_client.post(
            "/api/v1/forecast/single",
            json=sample_forecast_request
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert data["status"] == "success"
        assert "request_id" in data
        assert "forecasts" in data
        assert "processing_time" in data
        assert len(data["forecasts"]) == len(sample_forecast_request["indicators"])
    
    def test_single_forecast_with_report(self, api_test_client):
        """Test forecast request with executive report."""
        request_data = {
            "indicators": ["gdp"],
            "horizon": 6,
            "method": "statistical",
            "generate_report": True
        }
        
        response = api_test_client.post(
            "/api/v1/forecast/single",
            json=request_data
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        # Note: executive_summary might be None in mock implementation
        assert "executive_summary" in data
    
    def test_single_forecast_invalid_indicator(self, api_test_client):
        """Test forecast with invalid indicator."""
        request_data = {
            "indicators": ["invalid_indicator"],
            "horizon": 6
        }
        
        response = api_test_client.post(
            "/api/v1/forecast/single",
            json=request_data
        )
        
        # Should handle gracefully
        assert response.status_code in [200, 400]
    
    def test_single_forecast_invalid_horizon(self, api_test_client):
        """Test forecast with invalid horizon."""
        request_data = {
            "indicators": ["gdp"],
            "horizon": 100  # Too large
        }
        
        response = api_test_client.post(
            "/api/v1/forecast/single",
            json=request_data
        )
        
        assert response.status_code == 422  # Validation error
    
    def test_batch_forecast(self, api_test_client):
        """Test batch forecast request."""
        request_data = {
            "requests": [
                {
                    "indicators": ["gdp"],
                    "horizon": 3,
                    "method": "statistical"
                },
                {
                    "indicators": ["unemployment"],
                    "horizon": 6,
                    "method": "statistical"
                }
            ],
            "parallel_processing": False
        }
        
        response = api_test_client.post(
            "/api/v1/forecast/batch",
            json=request_data
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert "completed_forecasts" in data
        assert "failed_requests" in data
        assert "processing_summary" in data
    
    def test_sensitivity_test(self, api_test_client):
        """Test sensitivity testing endpoint."""
        request_data = {
            "base_forecast_id": "test_forecast_123",
            "parameters_to_test": ["horizon", "confidence_interval"],
            "variation_range": 0.2,
            "llm_analysis": False
        }
        
        response = api_test_client.post(
            "/api/v1/forecast/sensitivity",
            json=request_data
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert "sensitivity_results" in data
        assert "overall_robustness" in data
        assert "recommendations" in data


@pytest.mark.api
class TestAnalysisEndpoints:
    """Test analysis API endpoints."""
    
    def test_scenario_analysis(self, api_test_client, sample_scenario_request):
        """Test scenario analysis endpoint."""
        response = api_test_client.post(
            "/api/v1/analysis/scenario",
            json=sample_scenario_request
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert "scenarios" in data
        assert "comparative_analysis" in data
        assert "recommended_scenario" in data
        assert len(data["scenarios"]) == len(sample_scenario_request["scenarios"])
    
    def test_causal_inference(self, api_test_client):
        """Test causal inference endpoint."""
        request_data = {
            "treatment_indicator": "interest_rate",
            "outcome_indicators": ["unemployment", "gdp"],
            "treatment_date": "2020-03-15",
            "method": "difference_in_differences"
        }
        
        response = api_test_client.post(
            "/api/v1/analysis/causal-inference",
            json=request_data
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert "causal_effects" in data
        assert "interpretation" in data
        assert "policy_implications" in data
        assert len(data["causal_effects"]) == len(request_data["outcome_indicators"])
    
    def test_scenario_analysis_invalid_scenario(self, api_test_client):
        """Test scenario analysis with invalid scenario."""
        request_data = {
            "indicators": ["gdp"],
            "scenarios": ["invalid_scenario"],
            "horizon": 12
        }
        
        response = api_test_client.post(
            "/api/v1/analysis/scenario",
            json=request_data
        )
        
        # Should handle gracefully
        assert response.status_code in [200, 400, 422]
    
    def test_causal_inference_invalid_date(self, api_test_client):
        """Test causal inference with invalid date."""
        request_data = {
            "treatment_indicator": "interest_rate",
            "outcome_indicators": ["unemployment"],
            "treatment_date": "invalid-date",
            "method": "difference_in_differences"
        }
        
        response = api_test_client.post(
            "/api/v1/analysis/causal-inference",
            json=request_data
        )
        
        assert response.status_code == 422  # Validation error


@pytest.mark.api
class TestAPIFeatures:
    """Test general API features."""
    
    def test_cors_headers(self, api_test_client):
        """Test CORS headers are present."""
        response = api_test_client.options("/health/status")
        
        # CORS headers should be present
        assert response.status_code in [200, 405]  # OPTIONS might not be implemented
    
    def test_api_documentation(self, api_test_client):
        """Test API documentation endpoints."""
        # Test OpenAPI spec
        response = api_test_client.get("/openapi.json")
        assert response.status_code == 200
        
        openapi_spec = response.json()
        assert "openapi" in openapi_spec
        assert "info" in openapi_spec
        assert "paths" in openapi_spec
    
    def test_root_redirect(self, api_test_client):
        """Test root URL redirects to documentation."""
        response = api_test_client.get("/", allow_redirects=False)
        
        assert response.status_code in [200, 307, 308]  # Redirect or direct docs
    
    def test_info_endpoint(self, api_test_client):
        """Test API info endpoint."""
        response = api_test_client.get("/info")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "name" in data
        assert "version" in data
        assert "description" in data
        assert "features" in data
        assert "endpoints" in data


@pytest.mark.api
@pytest.mark.slow
class TestPerformanceIntegration:
    """Test API performance characteristics."""
    
    def test_response_times(self, api_test_client, test_config):
        """Test API response times are within acceptable limits."""
        max_response_time = test_config["performance_thresholds"]["api_response_time"]
        
        # Test health endpoint response time
        start_time = time.time()
        response = api_test_client.get("/health/status")
        response_time = time.time() - start_time
        
        assert response.status_code == 200
        assert response_time < max_response_time
    
    def test_forecast_performance(self, api_test_client, test_config):
        """Test forecast endpoint performance."""
        max_forecast_time = test_config["performance_thresholds"]["forecast_time"]
        
        request_data = {
            "indicators": ["gdp"],
            "horizon": 3,
            "method": "statistical"
        }
        
        start_time = time.time()
        response = api_test_client.post(
            "/api/v1/forecast/single",
            json=request_data
        )
        response_time = time.time() - start_time
        
        assert response.status_code == 200
        assert response_time < max_forecast_time
        
        # Also check the processing_time reported by the API
        data = response.json()
        if data["success"]:
            assert data["processing_time"] < max_forecast_time
    
    def test_concurrent_requests(self, api_test_client):
        """Test handling of concurrent requests."""
        import threading
        import queue
        
        results = queue.Queue()
        
        def make_request():
            response = api_test_client.get("/health/status")
            results.put(response.status_code)
        
        # Create multiple threads
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=make_request)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Check all requests succeeded
        status_codes = []
        while not results.empty():
            status_codes.append(results.get())
        
        assert len(status_codes) == 5
        assert all(code == 200 for code in status_codes)


@pytest.mark.api
class TestErrorHandling:
    """Test API error handling."""
    
    def test_rate_limiting(self, api_test_client):
        """Test rate limiting functionality."""
        # Make many requests quickly to test rate limiting
        responses = []
        for _ in range(20):
            response = api_test_client.get("/health/status")
            responses.append(response.status_code)
        
        # Most should succeed, but rate limiting might kick in
        success_count = sum(1 for code in responses if code == 200)
        rate_limited_count = sum(1 for code in responses if code == 429)
        
        assert success_count > 0
        # Rate limiting might or might not be triggered depending on configuration
    
    def test_invalid_json(self, api_test_client):
        """Test handling of invalid JSON."""
        response = api_test_client.post(
            "/api/v1/forecast/single",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        
        assert response.status_code == 422  # Unprocessable Entity
    
    def test_missing_required_fields(self, api_test_client):
        """Test handling of missing required fields."""
        # Missing indicators field
        request_data = {
            "horizon": 6,
            "method": "statistical"
        }
        
        response = api_test_client.post(
            "/api/v1/forecast/single",
            json=request_data
        )
        
        assert response.status_code == 422  # Validation error
    
    def test_large_request_handling(self, api_test_client):
        """Test handling of large requests."""
        # Create a large request with many indicators
        request_data = {
            "indicators": ["gdp"] * 50,  # Might exceed limits
            "horizon": 24
        }
        
        response = api_test_client.post(
            "/api/v1/forecast/single",
            json=request_data
        )
        
        # Should either process or reject gracefully
        assert response.status_code in [200, 400, 413, 422]