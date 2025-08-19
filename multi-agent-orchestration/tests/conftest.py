"""
Pytest Configuration and Fixtures

Shared test fixtures and configuration for the entire test suite.
"""

import pytest
import asyncio
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any

from src.agents.base_agent import BaseAgent
from src.agents.research_agent import ResearchAgent
from src.agents.analysis_agent import AnalysisAgent
from src.agents.summary_agent import SummaryAgent
from src.multi_agent_platform import MultiAgentPlatform
from src.tools.web_search import WebSearchTool
from src.tools.document_processor import DocumentProcessorTool
from src.tools.calculation_engine import CalculationEngineTool


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path)


@pytest.fixture
def mock_agent():
    """Create a mock agent for testing."""
    class MockAgent(BaseAgent):
        def __init__(self, agent_id: str = "test_agent"):
            super().__init__(
                agent_id=agent_id,
                name="Test Agent",
                description="Mock agent for testing"
            )
        
        async def process_task(self, task: Dict[str, Any]):
            """Mock task processing."""
            return {
                "content": f"Processed task: {task.get('description', 'Unknown task')}",
                "confidence": 0.85,
                "metadata": {"mock": True}
            }
        
        def get_capabilities(self):
            return ["testing", "mocking", "validation"]
    
    return MockAgent()


@pytest.fixture
def research_agent():
    """Create a research agent for testing."""
    return ResearchAgent()


@pytest.fixture
def analysis_agent():
    """Create an analysis agent for testing."""
    return AnalysisAgent()


@pytest.fixture
def summary_agent():
    """Create a summary agent for testing."""
    return SummaryAgent()


@pytest.fixture
def agent_registry():
    """Create a registry of test agents."""
    return {
        "research": ResearchAgent(),
        "analysis": AnalysisAgent(),
        "summary": SummaryAgent()
    }


@pytest.fixture
def platform(agent_registry):
    """Create a multi-agent platform for testing."""
    platform = MultiAgentPlatform()
    
    # Register test agents
    for agent_id, agent in agent_registry.items():
        platform.register_agent(agent)
    
    return platform


@pytest.fixture
def web_search_tool():
    """Create a web search tool for testing."""
    return WebSearchTool(max_results=5, timeout=10)


@pytest.fixture
def document_processor():
    """Create a document processor for testing."""
    return DocumentProcessorTool(max_length=50000)


@pytest.fixture
def calculation_engine():
    """Create a calculation engine for testing."""
    return CalculationEngineTool()


@pytest.fixture
def sample_tasks():
    """Sample tasks for testing."""
    return [
        {
            "id": "task_1",
            "type": "research",
            "description": "Research machine learning trends",
            "priority": "high",
            "metadata": {"domain": "AI/ML"}
        },
        {
            "id": "task_2", 
            "type": "analysis",
            "description": "Analyze research data",
            "priority": "medium",
            "metadata": {"input_source": "task_1"}
        },
        {
            "id": "task_3",
            "type": "summary",
            "description": "Create executive summary",
            "priority": "high", 
            "metadata": {"target_audience": "executives"}
        }
    ]


@pytest.fixture
def sample_workflow_data():
    """Sample workflow data for testing."""
    return {
        "nodes": [
            {"id": "start", "type": "start", "name": "Begin"},
            {"id": "research", "type": "agent", "agent_id": "research", "name": "Research Phase"},
            {"id": "analysis", "type": "agent", "agent_id": "analysis", "name": "Analysis Phase"},
            {"id": "summary", "type": "agent", "agent_id": "summary", "name": "Summary Phase"},
            {"id": "end", "type": "end", "name": "Complete"}
        ],
        "edges": [
            {"source": "start", "target": "research"},
            {"source": "research", "target": "analysis"},
            {"source": "analysis", "target": "summary"},
            {"source": "summary", "target": "end"}
        ]
    }


# Test markers
pytest_plugins = []

# Custom markers
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "e2e: marks tests as end-to-end tests"
    )
    config.addinivalue_line(
        "markers", "slow: marks tests as slow running"
    )
    config.addinivalue_line(
        "markers", "network: marks tests that require network access"
    )


# Async test utilities
@pytest.fixture
async def async_test_context():
    """Provide async test context."""
    context = {"started_at": asyncio.get_event_loop().time()}
    yield context
    context["completed_at"] = asyncio.get_event_loop().time()
    context["duration"] = context["completed_at"] - context["started_at"]