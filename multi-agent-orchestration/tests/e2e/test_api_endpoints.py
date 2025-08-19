"""
End-to-End API Tests

Tests complete API functionality including all endpoints and real workflows.
"""

import pytest
import asyncio
import json
from httpx import AsyncClient
from fastapi.testclient import TestClient

from src.api.main import app
from src.multi_agent_platform import MultiAgentPlatform


class TestAPIEndpoints:
    """End-to-end tests for API endpoints."""
    
    @pytest.fixture
    def test_client(self):
        """Create test client for API testing."""
        return TestClient(app)
    
    @pytest.fixture
    async def async_client(self):
        """Create async test client."""
        async with AsyncClient(app=app, base_url="http://test") as client:
            yield client
    
    def test_health_endpoint(self, test_client):
        """Test health check endpoint."""
        response = test_client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "version" in data
        assert "uptime_seconds" in data
    
    def test_platform_status(self, test_client):
        """Test platform status endpoint."""
        response = test_client.get("/api/v1/platform/status")
        
        assert response.status_code == 200
        data = response.json()
        assert "platform_id" in data
        assert "agents" in data
        assert "orchestration_patterns" in data
        assert "created_at" in data
    
    def test_list_agents(self, test_client):
        """Test listing agents endpoint."""
        response = test_client.get("/api/v1/agents/")
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        
        # Should have default agents
        agent_ids = [agent["agent_id"] for agent in data]
        assert "research" in agent_ids
        assert "analysis" in agent_ids
        assert "summary" in agent_ids
    
    def test_get_agent_details(self, test_client):
        """Test getting agent details."""
        response = test_client.get("/api/v1/agents/research")
        
        assert response.status_code == 200
        data = response.json()
        assert data["agent_id"] == "research"
        assert "name" in data
        assert "description" in data
        assert "capabilities" in data
        assert "performance_metrics" in data
    
    def test_get_nonexistent_agent(self, test_client):
        """Test getting details for nonexistent agent."""
        response = test_client.get("/api/v1/agents/nonexistent")
        
        assert response.status_code == 404
        data = response.json()
        assert "error" in data
    
    @pytest.mark.asyncio
    async def test_execute_single_task(self, async_client):
        """Test executing a single task."""
        task_data = {
            "agent_id": "research",
            "task": {
                "type": "research",
                "description": "Research machine learning trends",
                "priority": "high"
            }
        }
        
        response = await async_client.post("/api/v1/tasks/execute", json=task_data)
        
        assert response.status_code == 200
        data = response.json()
        assert "task_id" in data
        assert "result" in data
        assert data["result"]["success"] is True
        assert "content" in data["result"]
        assert "confidence" in data["result"]
    
    @pytest.mark.asyncio
    async def test_execute_pipeline_pattern(self, async_client):
        """Test executing pipeline orchestration pattern."""
        pipeline_data = {
            "pattern": "pipeline",
            "agents": ["research", "analysis", "summary"],
            "task": {
                "type": "research_and_analyze",
                "description": "Complete research and analysis pipeline",
                "data": {"topic": "artificial intelligence"}
            }
        }
        
        response = await async_client.post("/api/v1/orchestration/execute", json=pipeline_data)
        
        assert response.status_code == 200
        data = response.json()
        assert "orchestration_id" in data
        assert "pattern" in data
        assert data["pattern"] == "pipeline"
        assert "results" in data
        assert len(data["results"]) == 3  # Three agents in pipeline
        
        # Check each result
        for result in data["results"]:
            assert "agent_id" in result
            assert "content" in result
            assert "confidence" in result
            assert "success" in result
    
    @pytest.mark.asyncio
    async def test_execute_parallel_pattern(self, async_client):
        """Test executing parallel orchestration pattern."""
        parallel_data = {
            "pattern": "parallel", 
            "agents": ["research", "analysis"],
            "task": {
                "type": "parallel_processing",
                "description": "Process data in parallel",
                "data": {"dataset": "sample_data"}
            }
        }
        
        response = await async_client.post("/api/v1/orchestration/execute", json=parallel_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["pattern"] == "parallel"
        assert len(data["results"]) == 2
        
        # Verify both agents processed
        agent_ids = {result["agent_id"] for result in data["results"]}
        assert "research" in agent_ids
        assert "analysis" in agent_ids
    
    @pytest.mark.asyncio
    async def test_execute_supervisor_pattern(self, async_client):
        """Test executing supervisor orchestration pattern."""
        supervisor_data = {
            "pattern": "supervisor",
            "agents": ["research", "analysis", "summary"],
            "task": {
                "type": "supervised_analysis",
                "description": "Supervised multi-agent analysis",
                "data": {"complexity": "high"}
            }
        }
        
        response = await async_client.post("/api/v1/orchestration/execute", json=supervisor_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["pattern"] == "supervisor"
        assert "supervisor_result" in data
        assert "agent_results" in data
        assert "coordination_metadata" in data
    
    @pytest.mark.asyncio
    async def test_execute_reflective_pattern(self, async_client):
        """Test executing reflective orchestration pattern."""
        reflective_data = {
            "pattern": "reflective",
            "agents": ["analysis", "summary"],
            "task": {
                "type": "reflective_analysis",
                "description": "Analysis with reflection and improvement",
                "data": {"iterations": 2}
            }
        }
        
        response = await async_client.post("/api/v1/orchestration/execute", json=reflective_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["pattern"] == "reflective"
        assert "iterations" in data
        assert len(data["iterations"]) >= 1
        
        # Check iteration structure
        iteration = data["iterations"][0]
        assert "iteration_number" in iteration
        assert "results" in iteration
        assert "reflection" in iteration
    
    def test_execute_invalid_pattern(self, test_client):
        """Test executing invalid orchestration pattern."""
        invalid_data = {
            "pattern": "nonexistent_pattern",
            "agents": ["research"],
            "task": {"type": "test"}
        }
        
        response = test_client.post("/api/v1/orchestration/execute", json=invalid_data)
        
        assert response.status_code == 400
        data = response.json()
        assert "error" in data
    
    def test_execute_with_invalid_agent(self, test_client):
        """Test executing task with invalid agent."""
        invalid_data = {
            "pattern": "pipeline",
            "agents": ["nonexistent_agent"],
            "task": {"type": "test"}
        }
        
        response = test_client.post("/api/v1/orchestration/execute", json=invalid_data)
        
        assert response.status_code == 400
        data = response.json()
        assert "error" in data
    
    def test_missing_required_fields(self, test_client):
        """Test API with missing required fields."""
        # Missing pattern
        response = test_client.post("/api/v1/orchestration/execute", json={
            "agents": ["research"],
            "task": {"type": "test"}
        })
        assert response.status_code == 422  # Validation error
        
        # Missing agents
        response = test_client.post("/api/v1/orchestration/execute", json={
            "pattern": "pipeline",
            "task": {"type": "test"}
        })
        assert response.status_code == 422
        
        # Missing task
        response = test_client.post("/api/v1/orchestration/execute", json={
            "pattern": "pipeline",
            "agents": ["research"]
        })
        assert response.status_code == 422
    
    def test_agent_capabilities(self, test_client):
        """Test agent capabilities endpoint."""
        response = test_client.get("/api/v1/agents/research/capabilities")
        
        assert response.status_code == 200
        data = response.json()
        assert "capabilities" in data
        assert isinstance(data["capabilities"], list)
        assert len(data["capabilities"]) > 0
    
    def test_agent_performance_metrics(self, test_client):
        """Test agent performance metrics endpoint."""
        response = test_client.get("/api/v1/agents/research/metrics")
        
        assert response.status_code == 200
        data = response.json()
        assert "agent_id" in data
        assert "performance_metrics" in data
        metrics = data["performance_metrics"]
        assert "tasks_completed" in metrics
        assert "average_confidence" in metrics
        assert "success_rate" in metrics
    
    @pytest.mark.asyncio
    async def test_concurrent_requests(self, async_client):
        """Test handling concurrent API requests."""
        # Create multiple concurrent tasks
        tasks = []
        for i in range(5):
            task_data = {
                "agent_id": "research",
                "task": {
                    "type": "research",
                    "description": f"Concurrent task {i}",
                    "id": f"concurrent_{i}"
                }
            }
            task = async_client.post("/api/v1/tasks/execute", json=task_data)
            tasks.append(task)
        
        # Wait for all tasks to complete
        responses = await asyncio.gather(*tasks)
        
        # Verify all succeeded
        for response in responses:
            assert response.status_code == 200
            data = response.json()
            assert data["result"]["success"] is True
    
    def test_api_error_handling(self, test_client):
        """Test API error handling."""
        # Test with malformed JSON
        response = test_client.post(
            "/api/v1/tasks/execute",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422
        
        # Test with wrong content type
        response = test_client.post(
            "/api/v1/tasks/execute",
            data="some data",
            headers={"Content-Type": "text/plain"}
        )
        assert response.status_code == 422
    
    def test_cors_headers(self, test_client):
        """Test CORS headers are present."""
        response = test_client.options("/api/v1/agents/")
        
        # Should have CORS headers
        assert "access-control-allow-origin" in response.headers
        assert "access-control-allow-methods" in response.headers
    
    def test_rate_limiting_simulation(self, test_client):
        """Test rate limiting behavior (simulated)."""
        # Note: Actual rate limiting depends on deployment configuration
        # This test verifies the endpoint can handle rapid requests
        
        responses = []
        for i in range(10):
            response = test_client.get("/health")
            responses.append(response.status_code)
        
        # All requests should succeed in test environment
        assert all(status == 200 for status in responses)
    
    @pytest.mark.asyncio
    async def test_large_task_processing(self, async_client):
        """Test processing large tasks."""
        large_task = {
            "agent_id": "analysis",
            "task": {
                "type": "analysis",
                "description": "Large data analysis task",
                "data": {
                    "dataset": list(range(1000)),  # Large dataset
                    "parameters": {"complexity": "high"},
                    "metadata": {"large_task": True}
                }
            }
        }
        
        response = await async_client.post("/api/v1/tasks/execute", json=large_task)
        
        assert response.status_code == 200
        data = response.json()
        assert data["result"]["success"] is True
        # Should handle large data without issues
    
    def test_api_documentation_endpoints(self, test_client):
        """Test API documentation endpoints."""
        # Test OpenAPI schema
        response = test_client.get("/openapi.json")
        assert response.status_code == 200
        
        # Verify it's valid JSON
        openapi_spec = response.json()
        assert "openapi" in openapi_spec
        assert "info" in openapi_spec
        assert "paths" in openapi_spec
        
        # Test Swagger UI (if enabled)
        response = test_client.get("/docs")
        assert response.status_code == 200
        
        # Test ReDoc (if enabled)  
        response = test_client.get("/redoc")
        assert response.status_code == 200
    
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_long_running_workflow(self, async_client):
        """Test long-running workflow execution."""
        workflow_data = {
            "pattern": "pipeline",
            "agents": ["research", "analysis", "summary"],
            "task": {
                "type": "comprehensive_analysis",
                "description": "Long-running comprehensive analysis",
                "data": {
                    "depth": "extensive",
                    "iterations": 3
                }
            }
        }
        
        response = await async_client.post("/api/v1/orchestration/execute", json=workflow_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["pattern"] == "pipeline"
        assert len(data["results"]) == 3
        
        # Verify all agents completed successfully
        for result in data["results"]:
            assert result["success"] is True
            assert len(result["content"]) > 0