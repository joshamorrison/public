"""
Integration Tests for Workflows

Tests workflow creation, execution, and visualization with real agent interactions.
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import Mock

from src.workflows.graph_builder import WorkflowGraphBuilder, NodeType, EdgeType
from src.workflows.workflow_executor import WorkflowExecutor, ExecutionState
from src.workflows.workflow_visualizer import WorkflowVisualizer, VisualizationConfig
from src.agents.base_agent import BaseAgent, AgentResult


class TestWorkflowIntegration:
    """Integration tests for complete workflow functionality."""
    
    @pytest.fixture
    def mock_agents(self):
        """Create mock agents for workflow testing."""
        class MockAgent(BaseAgent):
            def __init__(self, agent_id: str, delay: float = 0.1):
                super().__init__(agent_id, f"Mock {agent_id}", f"Mock agent {agent_id}")
                self.delay = delay
            
            async def process_task(self, task):
                await asyncio.sleep(self.delay)
                return AgentResult(
                    agent_id=self.agent_id,
                    task_id=task.get("id", "mock_task"),
                    content=f"Processed by {self.agent_id}: {task.get('description', 'No description')}",
                    confidence=0.9,
                    metadata={"mock": True, "agent": self.agent_id},
                    timestamp=datetime.now()
                )
            
            def get_capabilities(self):
                return [f"{self.agent_id}_capability"]
        
        return {
            "research": MockAgent("research", 0.1),
            "analysis": MockAgent("analysis", 0.2), 
            "summary": MockAgent("summary", 0.1)
        }
    
    @pytest.fixture
    def workflow_builder(self):
        """Create a workflow builder for testing."""
        return WorkflowGraphBuilder(
            workflow_id="test_workflow",
            name="Test Workflow",
            description="A test workflow for integration testing"
        )
    
    @pytest.fixture
    def workflow_executor(self, mock_agents):
        """Create a workflow executor with mock agents."""
        return WorkflowExecutor(mock_agents)
    
    def test_simple_pipeline_creation(self, workflow_builder):
        """Test creating a simple pipeline workflow."""
        builder = workflow_builder.create_pipeline(
            agent_ids=["research", "analysis", "summary"],
            start_name="Research Pipeline",
            end_name="Pipeline Complete"
        )
        
        graph = builder.build()
        
        # Verify graph structure
        assert graph.id == "test_workflow"
        assert graph.name == "Test Workflow"
        assert len(graph.nodes) == 5  # start + 3 agents + end
        assert len(graph.edges) == 4  # Linear connections
        assert graph.start_node == "pipeline_start"
        assert "pipeline_end" in graph.end_nodes
        
        # Verify node types
        assert graph.nodes["pipeline_start"].node_type == NodeType.START
        assert graph.nodes["pipeline_end"].node_type == NodeType.END
        assert graph.nodes["agent_research"].node_type == NodeType.AGENT
        assert graph.nodes["agent_research"].agent_id == "research"
    
    def test_parallel_workflow_creation(self, workflow_builder):
        """Test creating a parallel workflow."""
        builder = workflow_builder.create_parallel_workflow(
            agent_ids=["research", "analysis"],
            start_name="Parallel Start",
            end_name="Parallel End"
        )
        
        graph = builder.build()
        
        # Verify structure
        assert len(graph.nodes) == 5  # start + parallel + 2 agents + end
        assert len(graph.edges) == 5  # start->parallel, parallel->2 agents, 2 agents->end
        
        # Check parallel execution setup
        parallel_node = graph.nodes["parallel_exec"]
        assert parallel_node.node_type == NodeType.PARALLEL
        
        # Verify parallel edges
        parallel_edges = [e for e in graph.edges.values() 
                         if e.edge_type == EdgeType.PARALLEL_BRANCH]
        assert len(parallel_edges) == 2
    
    def test_custom_workflow_creation(self, workflow_builder):
        """Test creating a custom workflow with conditions."""
        # Build custom workflow
        builder = (workflow_builder
                  .add_start_node("start", "Begin Process")
                  .add_agent_node("research", "research_node", "Research Task")
                  .add_condition_node(lambda ctx: ctx.get("research_confidence", 0) > 0.8,
                                    "quality_check", "Quality Check")
                  .add_agent_node("analysis", "analysis_node", "Analysis Task") 
                  .add_agent_node("summary", "summary_node", "Summary Task")
                  .add_end_node("end", "Process Complete"))
        
        # Add edges
        builder = (builder
                  .add_edge("start", "research_node")
                  .add_edge("research_node", "quality_check")
                  .add_conditional_edge("quality_check", "analysis_node", "research_confidence > 0.8")
                  .add_edge("analysis_node", "summary_node")
                  .add_edge("summary_node", "end"))
        
        graph = builder.build()
        
        # Verify custom structure
        assert len(graph.nodes) == 6
        assert graph.nodes["quality_check"].node_type == NodeType.CONDITION
        
        # Check conditional edge
        conditional_edges = [e for e in graph.edges.values() 
                           if e.edge_type == EdgeType.CONDITIONAL]
        assert len(conditional_edges) == 1
        assert conditional_edges[0].condition == "research_confidence > 0.8"
    
    @pytest.mark.asyncio
    async def test_pipeline_execution(self, workflow_builder, workflow_executor):
        """Test executing a simple pipeline workflow."""
        # Create pipeline
        builder = workflow_builder.create_pipeline(["research", "analysis", "summary"])
        graph = builder.build()
        
        # Execute workflow
        initial_context = {"input_data": "test data for processing"}
        execution = await workflow_executor.execute_workflow(graph, initial_context)
        
        # Verify execution completed successfully
        assert execution.state == ExecutionState.COMPLETED
        assert execution.start_time is not None
        assert execution.end_time is not None
        assert execution.workflow_id == "test_workflow"
        
        # Check all nodes completed
        for node_id in graph.nodes.keys():
            node_exec = execution.node_executions[node_id]
            assert node_exec.state == ExecutionState.COMPLETED
            if node_exec.result:
                assert isinstance(node_exec.result, (dict, AgentResult))
        
        # Verify context was updated with results
        assert "result_agent_research" in execution.global_context
        assert "result_agent_analysis" in execution.global_context
        assert "result_agent_summary" in execution.global_context
    
    @pytest.mark.asyncio
    async def test_parallel_execution(self, workflow_builder, workflow_executor):
        """Test executing a parallel workflow."""
        # Create parallel workflow
        builder = workflow_builder.create_parallel_workflow(["research", "analysis"])
        graph = builder.build()
        
        # Execute workflow
        start_time = datetime.now()
        execution = await workflow_executor.execute_workflow(graph)
        end_time = datetime.now()
        
        # Verify successful completion
        assert execution.state == ExecutionState.COMPLETED
        
        # Check that parallel execution was faster than sequential
        # (Mock agents have different delays: research=0.1s, analysis=0.2s)
        # Parallel should be ~0.2s, sequential would be ~0.3s
        total_time = (end_time - start_time).total_seconds()
        assert total_time < 1.0  # Should complete quickly with mocks
        
        # Verify both parallel branches completed
        research_exec = execution.node_executions["agent_research"]
        analysis_exec = execution.node_executions["agent_analysis"]
        assert research_exec.state == ExecutionState.COMPLETED
        assert analysis_exec.state == ExecutionState.COMPLETED
    
    @pytest.mark.asyncio
    async def test_workflow_with_failure(self, workflow_builder, mock_agents):
        """Test workflow execution with agent failure."""
        # Create a failing agent
        class FailingAgent(BaseAgent):
            def __init__(self):
                super().__init__("failing_agent", "Failing Agent", "Always fails")
            
            async def process_task(self, task):
                raise Exception("Simulated failure")
            
            def get_capabilities(self):
                return ["failing"]
        
        # Add failing agent
        agents_with_failure = {**mock_agents, "failing": FailingAgent()}
        executor = WorkflowExecutor(agents_with_failure)
        
        # Create workflow with failing agent
        builder = workflow_builder.create_pipeline(["research", "failing", "summary"])
        graph = builder.build()
        
        # Execute workflow
        execution = await executor.execute_workflow(graph)
        
        # Verify workflow failed
        assert execution.state == ExecutionState.FAILED
        
        # Check that research completed but failing agent failed
        research_exec = execution.node_executions["agent_research"]
        failing_exec = execution.node_executions["agent_failing"]
        summary_exec = execution.node_executions["agent_summary"]
        
        assert research_exec.state == ExecutionState.COMPLETED
        assert failing_exec.state == ExecutionState.FAILED
        assert failing_exec.error is not None
        # Summary should not have executed due to dependency failure
        assert summary_exec.state == ExecutionState.PENDING
    
    def test_workflow_visualization_mermaid(self, workflow_builder):
        """Test Mermaid diagram generation."""
        builder = workflow_builder.create_pipeline(["research", "analysis", "summary"])
        graph = builder.build()
        
        visualizer = WorkflowVisualizer()
        mermaid_diagram = visualizer.generate_mermaid(graph)
        
        # Verify Mermaid format
        assert mermaid_diagram.startswith("graph TD")
        assert "pipeline_start" in mermaid_diagram
        assert "pipeline_end" in mermaid_diagram
        assert "agent_research" in mermaid_diagram
        assert "-->" in mermaid_diagram  # Normal edges
    
    def test_workflow_visualization_graphviz(self, workflow_builder):
        """Test Graphviz DOT generation."""
        builder = workflow_builder.create_pipeline(["research", "analysis"])
        graph = builder.build()
        
        visualizer = WorkflowVisualizer()
        dot_diagram = visualizer.generate_graphviz(graph)
        
        # Verify DOT format
        assert dot_diagram.startswith("digraph workflow")
        assert "rankdir=TD" in dot_diagram
        assert "agent_research" in dot_diagram
        assert "->" in dot_diagram  # Edge notation
        assert dot_diagram.endswith("}")
    
    def test_workflow_visualization_json(self, workflow_builder):
        """Test JSON visualization format."""
        builder = workflow_builder.create_parallel_workflow(["research", "analysis"])
        graph = builder.build()
        
        visualizer = WorkflowVisualizer()
        json_str = visualizer.generate_json_for_web(graph)
        
        # Parse and verify JSON structure
        import json
        viz_data = json.loads(json_str)
        
        assert "workflow" in viz_data
        assert "nodes" in viz_data
        assert "edges" in viz_data
        assert viz_data["workflow"]["id"] == "test_workflow"
        assert len(viz_data["nodes"]) == len(graph.nodes)
        assert len(viz_data["edges"]) == len(graph.edges)
        
        # Check node format
        node = viz_data["nodes"][0]
        assert "id" in node
        assert "label" in node
        assert "type" in node
        assert "group" in node
    
    def test_workflow_visualization_html(self, workflow_builder):
        """Test HTML interactive visualization."""
        builder = workflow_builder.create_pipeline(["research"])
        graph = builder.build()
        
        visualizer = WorkflowVisualizer()
        html_content = visualizer.generate_html_interactive(graph)
        
        # Verify HTML structure
        assert html_content.startswith("<!DOCTYPE html>")
        assert "<title>Workflow Visualization:" in html_content
        assert "vis-network" in html_content  # vis.js library
        assert "workflow-container" in html_content
        assert graph.name in html_content
    
    @pytest.mark.asyncio
    async def test_execution_monitoring(self, workflow_builder, workflow_executor):
        """Test workflow execution monitoring."""
        builder = workflow_builder.create_pipeline(["research", "analysis"])
        graph = builder.build()
        
        # Start execution
        execution_task = asyncio.create_task(
            workflow_executor.execute_workflow(graph)
        )
        
        # Give it a moment to start
        await asyncio.sleep(0.05)
        
        # Check status during execution
        execution = workflow_executor.active_executions
        if execution:
            execution_id = list(execution.keys())[0]
            status = workflow_executor.get_execution_status(execution_id)
            assert status is not None
            
            summary = workflow_executor.get_execution_summary(execution_id)
            assert "execution_id" in summary
            assert "progress" in summary
            assert "state" in summary
        
        # Wait for completion
        final_execution = await execution_task
        assert final_execution.state == ExecutionState.COMPLETED
    
    @pytest.mark.asyncio
    async def test_workflow_cancellation(self, workflow_builder, mock_agents):
        """Test workflow cancellation."""
        # Create slow agents
        class SlowAgent(BaseAgent):
            def __init__(self, agent_id: str):
                super().__init__(agent_id, f"Slow {agent_id}", "Slow processing agent")
            
            async def process_task(self, task):
                await asyncio.sleep(2.0)  # Long delay
                return AgentResult(
                    agent_id=self.agent_id, task_id="slow_task",
                    content="Slow result", confidence=0.8,
                    metadata={}, timestamp=datetime.now()
                )
            
            def get_capabilities(self):
                return ["slow_processing"]
        
        slow_agents = {"slow": SlowAgent("slow")}
        executor = WorkflowExecutor(slow_agents)
        
        builder = workflow_builder.create_pipeline(["slow"])
        graph = builder.build()
        
        # Start execution
        execution_task = asyncio.create_task(
            executor.execute_workflow(graph)
        )
        
        # Give it time to start
        await asyncio.sleep(0.1)
        
        # Cancel execution
        execution_id = list(executor.active_executions.keys())[0]
        cancelled = await executor.cancel_execution(execution_id)
        assert cancelled is True
        
        # Verify cancellation
        status = executor.get_execution_status(execution_id)
        assert status.state == ExecutionState.CANCELLED
        
        # Clean up
        execution_task.cancel()
        try:
            await execution_task
        except asyncio.CancelledError:
            pass
    
    def test_workflow_graph_serialization(self, workflow_builder):
        """Test workflow graph serialization and deserialization."""
        builder = workflow_builder.create_pipeline(["research", "analysis"])
        graph = builder.build()
        
        # Test to_dict
        graph_dict = builder.to_dict()
        assert graph_dict["id"] == "test_workflow"
        assert "nodes" in graph_dict
        assert "edges" in graph_dict
        assert len(graph_dict["nodes"]) == len(graph.nodes)
        
        # Test to_json
        json_str = builder.to_json()
        assert isinstance(json_str, str)
        
        # Verify JSON is valid
        import json
        parsed = json.loads(json_str)
        assert parsed["id"] == "test_workflow"
    
    def test_workflow_validation(self, workflow_builder):
        """Test workflow graph validation."""
        # Valid workflow
        builder = workflow_builder.create_pipeline(["research"])
        assert builder.validate_graph() is True
        
        # Invalid workflow - no start node
        empty_builder = WorkflowGraphBuilder("empty", "Empty", "No nodes")
        empty_builder.add_end_node()
        assert empty_builder.validate_graph() is False
        
        # Invalid workflow - no end node
        no_end_builder = WorkflowGraphBuilder("no_end", "No End", "Missing end")
        no_end_builder.add_start_node()
        assert no_end_builder.validate_graph() is False