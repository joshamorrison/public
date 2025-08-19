"""
Unit Tests for BaseAgent

Tests core agent functionality including message handling, performance metrics,
and task processing capabilities.
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import Mock, patch

from src.agents.base_agent import BaseAgent, AgentMessage, AgentResult


class TestBaseAgent:
    """Test cases for BaseAgent functionality."""
    
    @pytest.fixture
    def concrete_agent(self):
        """Create a concrete implementation of BaseAgent for testing."""
        class ConcreteAgent(BaseAgent):
            async def process_task(self, task):
                return AgentResult(
                    agent_id=self.agent_id,
                    task_id=task.get("id", "test_task"),
                    content=f"Processed: {task.get('description', 'No description')}",
                    confidence=0.9,
                    metadata={"test": True},
                    timestamp=datetime.now()
                )
            
            def get_capabilities(self):
                return ["testing", "validation"]
        
        return ConcreteAgent("test_agent", "Test Agent", "Agent for testing")
    
    def test_agent_initialization(self, concrete_agent):
        """Test agent initialization."""
        assert concrete_agent.agent_id == "test_agent"
        assert concrete_agent.name == "Test Agent"
        assert concrete_agent.description == "Agent for testing"
        assert concrete_agent.created_at is not None
        assert len(concrete_agent.message_history) == 0
        assert concrete_agent.performance_metrics["tasks_completed"] == 0
    
    def test_capabilities(self, concrete_agent):
        """Test capability management."""
        capabilities = concrete_agent.get_capabilities()
        assert "testing" in capabilities
        assert "validation" in capabilities
        
        assert concrete_agent.can_handle_task("testing")
        assert concrete_agent.can_handle_task("validation")
        assert not concrete_agent.can_handle_task("unknown")
    
    @pytest.mark.asyncio
    async def test_send_message(self, concrete_agent):
        """Test message sending functionality."""
        message = await concrete_agent.send_message(
            recipient="other_agent",
            content="Test message",
            message_type="info"
        )
        
        assert isinstance(message, AgentMessage)
        assert message.sender == "test_agent"
        assert message.recipient == "other_agent"
        assert message.content == "Test message"
        assert message.message_type == "info"
        assert len(concrete_agent.message_history) == 1
    
    @pytest.mark.asyncio
    async def test_process_task(self, concrete_agent):
        """Test task processing."""
        task = {
            "id": "test_task_1",
            "description": "Test task processing",
            "type": "testing"
        }
        
        result = await concrete_agent.process_task(task)
        
        assert isinstance(result, AgentResult)
        assert result.agent_id == "test_agent"
        assert result.task_id == "test_task_1"
        assert "Processed: Test task processing" in result.content
        assert result.confidence == 0.9
        assert result.success is True
    
    def test_performance_metrics_update(self, concrete_agent):
        """Test performance metrics updates."""
        initial_tasks = concrete_agent.performance_metrics["tasks_completed"]
        
        result = AgentResult(
            agent_id="test_agent",
            task_id="test",
            content="Test result",
            confidence=0.8,
            metadata={},
            timestamp=datetime.now(),
            success=True
        )
        
        concrete_agent.update_performance_metrics(result, 1.5)
        
        metrics = concrete_agent.performance_metrics
        assert metrics["tasks_completed"] == initial_tasks + 1
        assert metrics["total_processing_time"] == 1.5
        assert metrics["average_confidence"] == 0.8
        assert metrics["success_rate"] == 1.0
    
    def test_performance_metrics_multiple_tasks(self, concrete_agent):
        """Test performance metrics with multiple tasks."""
        # First task - success
        result1 = AgentResult(
            agent_id="test_agent", task_id="1", content="Result 1",
            confidence=0.9, metadata={}, timestamp=datetime.now(), success=True
        )
        concrete_agent.update_performance_metrics(result1, 1.0)
        
        # Second task - failure
        result2 = AgentResult(
            agent_id="test_agent", task_id="2", content="Result 2",
            confidence=0.5, metadata={}, timestamp=datetime.now(), success=False
        )
        concrete_agent.update_performance_metrics(result2, 2.0)
        
        metrics = concrete_agent.performance_metrics
        assert metrics["tasks_completed"] == 2
        assert metrics["total_processing_time"] == 3.0
        assert metrics["average_confidence"] == 0.7  # (0.9 + 0.5) / 2
        assert metrics["success_rate"] == 0.5  # 1 success out of 2 tasks
    
    @pytest.mark.asyncio
    async def test_llm_integration(self, concrete_agent):
        """Test LLM integration methods."""
        # Test basic response generation
        response = await concrete_agent.generate_response("Test prompt")
        assert isinstance(response, str)
        assert len(response) > 0
        
        # Test with context
        context = {"key": "value", "data": [1, 2, 3]}
        response_with_context = await concrete_agent.generate_response(
            "Analyze this data", context
        )
        assert isinstance(response_with_context, str)
        assert len(response_with_context) > 0
    
    @pytest.mark.asyncio
    async def test_tool_integration(self, concrete_agent):
        """Test tool integration methods."""
        # Test web search
        search_results = await concrete_agent.use_web_search("test query")
        assert isinstance(search_results, list)
        
        # Test text analysis
        analysis_result = await concrete_agent.analyze_text("Sample text for analysis")
        assert isinstance(analysis_result, dict)
        assert "word_count" in analysis_result
        
        # Test calculation
        stats_result = await concrete_agent.calculate_statistics([1, 2, 3, 4, 5])
        assert isinstance(stats_result, dict)
        assert "mean" in stats_result
    
    @pytest.mark.asyncio
    async def test_process_with_tools_research(self, concrete_agent):
        """Test process_with_tools for research tasks."""
        task = {
            "type": "research",
            "description": "machine learning trends",
            "id": "research_task"
        }
        
        result = await concrete_agent.process_with_tools(task)
        
        assert isinstance(result, AgentResult)
        assert result.success is True
        assert "Research Analysis" in result.content
        assert result.confidence > 0
        assert "search" in result.metadata.get("tools_used", [])
    
    @pytest.mark.asyncio
    async def test_process_with_tools_analysis(self, concrete_agent):
        """Test process_with_tools for analysis tasks."""
        task = {
            "type": "analysis",
            "description": "analyze performance data",
            "data": [10, 20, 30, 40, 50],
            "id": "analysis_task"
        }
        
        result = await concrete_agent.process_with_tools(task)
        
        assert isinstance(result, AgentResult)
        assert result.success is True
        assert result.confidence > 0
    
    @pytest.mark.asyncio
    async def test_process_with_tools_error_handling(self, concrete_agent):
        """Test error handling in process_with_tools."""
        # Test with invalid task that causes error
        task = {
            "type": "invalid_type",
            "description": None,  # This might cause issues
            "id": "error_task"
        }
        
        # Mock a tool method to raise an exception
        with patch.object(concrete_agent, 'use_web_search', side_effect=Exception("Network error")):
            task["type"] = "research"
            result = await concrete_agent.process_with_tools(task)
            
            assert isinstance(result, AgentResult)
            assert result.success is False
            assert "failed" in result.content.lower()
            assert result.error_message is not None
    
    def test_get_status(self, concrete_agent):
        """Test agent status retrieval."""
        status = concrete_agent.get_status()
        
        assert status["agent_id"] == "test_agent"
        assert status["name"] == "Test Agent"
        assert status["description"] == "Agent for testing"
        assert "capabilities" in status
        assert "performance_metrics" in status
        assert "created_at" in status
    
    def test_reset_metrics(self, concrete_agent):
        """Test metrics reset functionality."""
        # Add some metrics first
        result = AgentResult(
            agent_id="test_agent", task_id="test", content="Test",
            confidence=0.8, metadata={}, timestamp=datetime.now()
        )
        concrete_agent.update_performance_metrics(result, 1.0)
        
        # Verify metrics are not zero
        assert concrete_agent.performance_metrics["tasks_completed"] > 0
        
        # Reset and verify
        concrete_agent.reset_metrics()
        metrics = concrete_agent.performance_metrics
        assert metrics["tasks_completed"] == 0
        assert metrics["total_processing_time"] == 0.0
        assert metrics["average_confidence"] == 0.0
        assert metrics["success_rate"] == 0.0
    
    def test_string_representations(self, concrete_agent):
        """Test string representation methods."""
        str_repr = str(concrete_agent)
        assert "Test Agent" in str_repr
        assert "test_agent" in str_repr
        
        repr_str = repr(concrete_agent)
        assert "BaseAgent" in repr_str
        assert "test_agent" in repr_str
        assert "testing" in repr_str
        assert "validation" in repr_str


class TestAgentMessage:
    """Test cases for AgentMessage."""
    
    def test_message_creation(self):
        """Test message creation with all fields."""
        message = AgentMessage(
            id="msg_1",
            sender="agent_1",
            recipient="agent_2", 
            content="Test message",
            message_type="info",
            timestamp=datetime.now()
        )
        
        assert message.id == "msg_1"
        assert message.sender == "agent_1"
        assert message.recipient == "agent_2"
        assert message.content == "Test message"
        assert message.message_type == "info"
    
    def test_message_auto_fields(self):
        """Test automatic field generation."""
        message = AgentMessage(
            id="",  # Should auto-generate
            sender="agent_1",
            recipient="agent_2",
            content="Test message", 
            message_type="info",
            timestamp=None  # Should auto-generate
        )
        
        assert len(message.id) > 0
        assert message.timestamp is not None


class TestAgentResult:
    """Test cases for AgentResult."""
    
    def test_result_creation(self):
        """Test result creation."""
        result = AgentResult(
            agent_id="test_agent",
            task_id="test_task",
            content="Test result content",
            confidence=0.85,
            metadata={"key": "value"},
            timestamp=datetime.now(),
            success=True
        )
        
        assert result.agent_id == "test_agent"
        assert result.task_id == "test_task"
        assert result.content == "Test result content"
        assert result.confidence == 0.85
        assert result.metadata["key"] == "value"
        assert result.success is True
        assert result.error_message is None
    
    def test_result_auto_timestamp(self):
        """Test automatic timestamp generation."""
        result = AgentResult(
            agent_id="test_agent",
            task_id="test_task",
            content="Test content",
            confidence=0.9,
            metadata={},
            timestamp=None  # Should auto-generate
        )
        
        assert result.timestamp is not None
    
    def test_failed_result(self):
        """Test failed result creation."""
        result = AgentResult(
            agent_id="test_agent",
            task_id="test_task",
            content="Error occurred",
            confidence=0.0,
            metadata={"error_code": 500},
            timestamp=datetime.now(),
            success=False,
            error_message="Something went wrong"
        )
        
        assert result.success is False
        assert result.error_message == "Something went wrong"
        assert result.confidence == 0.0