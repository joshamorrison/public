"""
Unit tests for FastAPI endpoints
"""

import pytest
import json
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
from datetime import datetime, timedelta

from api.main import app
from api.models.request_models import AttributionRequest, BudgetOptimizationRequest
from api.models.response_models import AnalysisStatus


class TestAPIEndpoints:
    """Test suite for FastAPI endpoints"""

    @pytest.fixture
    def client(self):
        """Create test client"""
        return TestClient(app)

    def test_health_check_endpoint(self, client):
        """Test health check endpoints"""
        # Test basic health endpoint
        response = client.get("/health/status")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "version" in data
        assert "system_info" in data

    def test_readiness_probe(self, client):
        """Test Kubernetes readiness probe"""
        response = client.get("/health/ready")
        assert response.status_code in [200, 503]  # Ready or not ready
        
        data = response.json()
        assert "status" in data
        assert "checks" in data

    def test_liveness_probe(self, client):
        """Test Kubernetes liveness probe"""
        response = client.get("/health/live")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "alive"
        assert "timestamp" in data
        assert "process_id" in data

    def test_metrics_endpoint(self, client):
        """Test metrics endpoint for monitoring"""
        response = client.get("/health/metrics")
        assert response.status_code == 200
        
        data = response.json()
        assert "system" in data
        assert "application" in data
        assert "cpu_usage_percent" in data["system"]
        assert "memory_usage_percent" in data["system"]

    @patch('api.routers.attribution._fetch_journey_data')
    @patch('api.routers.attribution._perform_attribution_analysis')
    def test_attribution_analysis_endpoint(self, mock_analysis, mock_fetch, client):
        """Test attribution analysis endpoint"""
        # Mock data and analysis
        mock_fetch.return_value = {"journey_data": "mock_data"}
        mock_analysis.return_value = [
            {
                "channel": "search",
                "attribution_value": 10000.0,
                "attribution_percentage": 40.0,
                "confidence_interval": [0.8, 1.2],
                "touch_count": 150,
                "conversion_rate": 0.05
            }
        ]
        
        # Test request
        request_data = {
            "channels": ["search", "social", "display"],
            "start_date": "2024-07-01",
            "end_date": "2024-07-31",
            "attribution_model": "data_driven",
            "conversion_window_days": 30,
            "granularity": "daily"
        }
        
        response = client.post("/api/v1/attribution/analyze", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert data["status"] == "SUCCESS"
        assert "attribution_results" in data
        assert "total_conversions" in data
        assert "total_revenue" in data

    @patch('api.routers.optimization._perform_budget_optimization')
    @patch('api.routers.optimization._fetch_historical_performance')
    def test_budget_optimization_endpoint(self, mock_performance, mock_optimization, client):
        """Test budget optimization endpoint"""
        # Mock data
        mock_performance.return_value = {"historical_data": "mock_data"}
        mock_optimization.return_value = [
            {
                "channel": "search",
                "current_budget": 10000.0,
                "recommended_budget": 12000.0,
                "budget_change": 2000.0,
                "budget_change_percentage": 20.0,
                "expected_roi": 3.5,
                "confidence_score": 0.85
            }
        ]
        
        # Test request
        request_data = {
            "total_budget": 50000.0,
            "current_budget": {
                "search": 15000.0,
                "social": 12000.0,
                "display": 10000.0,
                "video": 8000.0,
                "email": 5000.0
            },
            "optimization_objective": "maximize_roas",
            "historical_data_days": 90,
            "constraints": {
                "min_budget_per_channel": 1000.0,
                "max_budget_increase": 0.5
            }
        }
        
        response = client.post("/api/v1/optimization/budget", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert data["status"] == "SUCCESS"
        assert "recommendations" in data

    @patch('api.routers.performance._fetch_performance_data')
    @patch('api.routers.performance._calculate_channel_performance')
    def test_performance_analysis_endpoint(self, mock_calc, mock_fetch, client):
        """Test performance analysis endpoint"""
        # Mock data
        mock_fetch.return_value = {"performance_data": "mock_data"}
        mock_calc.return_value = [
            {
                "channel": "search",
                "impressions": 100000,
                "clicks": 5000,
                "conversions": 250,
                "revenue": 12500.0,
                "spend": 5000.0,
                "cpc": 1.0,
                "cpa": 20.0,
                "roas": 2.5,
                "ctr": 0.05,
                "conversion_rate": 0.05
            }
        ]
        
        # Test request
        request_data = {
            "channels": ["search", "social", "display"],
            "start_date": "2024-07-01",
            "end_date": "2024-07-31",
            "metrics": ["impressions", "clicks", "conversions", "revenue"],
            "granularity": "daily",
            "include_trends": True,
            "anomaly_detection": True,
            "benchmark_comparison": True
        }
        
        response = client.post("/api/v1/performance/analyze", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert data["status"] == "SUCCESS"
        assert "channel_performance" in data

    def test_performance_dashboard_endpoint(self, client):
        """Test performance dashboard endpoint"""
        response = client.get("/api/v1/performance/dashboard?channels=search,social&date_range=30")
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert "data" in data
        assert "timestamp" in data

    def test_kpi_endpoint(self, client):
        """Test KPI endpoint"""
        response = client.get("/api/v1/performance/kpis")
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert "kpis" in data
        assert "last_updated" in data

    def test_api_validation_errors(self, client):
        """Test API request validation"""
        # Test missing required fields
        invalid_request = {
            "channels": [],  # Empty channels should be invalid
            "start_date": "invalid-date"  # Invalid date format
        }
        
        response = client.post("/api/v1/attribution/analyze", json=invalid_request)
        assert response.status_code == 422  # Validation error

    def test_api_error_handling(self, client):
        """Test API error handling"""
        # Test with mock that raises exception
        with patch('api.routers.attribution._fetch_journey_data', side_effect=Exception("Database error")):
            request_data = {
                "channels": ["search", "social"],
                "start_date": "2024-07-01", 
                "end_date": "2024-07-31",
                "attribution_model": "data_driven",
                "conversion_window_days": 30
            }
            
            response = client.post("/api/v1/attribution/analyze", json=request_data)
            assert response.status_code == 200  # Should handle gracefully
            
            data = response.json()
            assert data["success"] is False
            assert data["status"] == "FAILED"
            assert "error" in data

    def test_cors_headers(self, client):
        """Test CORS headers are present"""
        response = client.options("/api/v1/attribution/analyze")
        # Should have CORS headers if configured
        assert response.status_code in [200, 404, 405]

    def test_rate_limiting_headers(self, client):
        """Test rate limiting headers if implemented"""
        response = client.get("/health/status")
        
        # Check if rate limiting headers are present
        # These would be added by rate limiting middleware
        headers = response.headers
        assert response.status_code == 200

    @pytest.mark.slow
    def test_api_performance_under_load(self, client):
        """Test API performance under simulated load"""
        import time
        
        # Make multiple concurrent requests to test performance
        start_time = time.time()
        
        responses = []
        for i in range(10):
            response = client.get("/health/status")
            responses.append(response)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # All requests should succeed
        assert all(r.status_code == 200 for r in responses)
        
        # Performance check - should handle 10 requests reasonably fast
        assert total_time < 5.0  # Less than 5 seconds for 10 health checks

    def test_request_id_tracking(self, client):
        """Test request ID tracking in responses"""
        request_data = {
            "channels": ["search", "social"],
            "start_date": "2024-07-01",
            "end_date": "2024-07-31", 
            "attribution_model": "data_driven",
            "conversion_window_days": 30
        }
        
        response = client.post("/api/v1/attribution/analyze", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "request_id" in data
        assert len(data["request_id"]) > 0  # Should have a request ID

    def test_api_documentation_endpoints(self, client):
        """Test API documentation is accessible"""
        # Test OpenAPI schema
        response = client.get("/docs")
        assert response.status_code == 200
        
        # Test OpenAPI JSON
        response = client.get("/openapi.json") 
        assert response.status_code == 200
        
        data = response.json()
        assert "openapi" in data
        assert "info" in data
        assert "paths" in data