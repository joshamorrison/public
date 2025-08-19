"""
Workflow Executor

Executes complex workflows defined by WorkflowGraph structures.
"""

import asyncio
import json
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from .graph_builder import WorkflowGraph, NodeType, EdgeType
from ..agents.base_agent import BaseAgent


class ExecutionState(Enum):
    """States of workflow execution."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class NodeExecution:
    """Execution state for a single node."""
    node_id: str
    state: ExecutionState = ExecutionState.PENDING
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowExecution:
    """Complete execution state for a workflow."""
    workflow_id: str
    execution_id: str
    state: ExecutionState = ExecutionState.PENDING
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    node_executions: Dict[str, NodeExecution] = field(default_factory=dict)
    global_context: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class WorkflowExecutor:
    """
    Executes workflows defined by WorkflowGraph structures.
    
    Handles complex execution patterns including:
    - Sequential execution
    - Parallel execution
    - Conditional branching
    - Error handling and recovery
    - State management
    """
    
    def __init__(self, agents: Dict[str, BaseAgent]):
        """
        Initialize workflow executor.
        
        Args:
            agents: Dictionary mapping agent IDs to agent instances
        """
        self.agents = agents
        self.active_executions: Dict[str, WorkflowExecution] = {}
        self._execution_counter = 0
    
    async def execute_workflow(self, graph: WorkflowGraph, 
                             initial_context: Dict[str, Any] = None) -> WorkflowExecution:
        """
        Execute a complete workflow.
        
        Args:
            graph: Workflow graph to execute
            initial_context: Initial context data
            
        Returns:
            WorkflowExecution: Complete execution results
        """
        execution_id = f"exec_{self._execution_counter}_{int(datetime.now().timestamp())}"
        self._execution_counter += 1
        
        execution = WorkflowExecution(
            workflow_id=graph.id,
            execution_id=execution_id,
            start_time=datetime.now(),
            global_context=initial_context or {}
        )
        
        # Initialize node executions
        for node_id in graph.nodes.keys():
            execution.node_executions[node_id] = NodeExecution(node_id=node_id)
        
        self.active_executions[execution_id] = execution
        
        try:
            execution.state = ExecutionState.RUNNING
            
            # Start execution from start node
            if graph.start_node:
                await self._execute_node_chain(graph, execution, graph.start_node)
            
            # Check if all end nodes completed successfully
            all_end_completed = all(
                execution.node_executions[node_id].state == ExecutionState.COMPLETED
                for node_id in graph.end_nodes
            )
            
            if all_end_completed:
                execution.state = ExecutionState.COMPLETED
            else:
                execution.state = ExecutionState.FAILED
                
        except Exception as e:
            execution.state = ExecutionState.FAILED
            execution.metadata["error"] = str(e)
        
        finally:
            execution.end_time = datetime.now()
            # Keep execution in memory for result retrieval
        
        return execution
    
    async def _execute_node_chain(self, graph: WorkflowGraph, 
                                 execution: WorkflowExecution, 
                                 node_id: str) -> bool:
        """
        Execute a node and its dependent chain.
        
        Args:
            graph: Workflow graph
            execution: Current execution state
            node_id: Node to execute
            
        Returns:
            True if execution successful, False otherwise
        """
        node_exec = execution.node_executions[node_id]
        
        # Skip if already completed or failed
        if node_exec.state in [ExecutionState.COMPLETED, ExecutionState.FAILED]:
            return node_exec.state == ExecutionState.COMPLETED
        
        # Check dependencies
        dependencies = self._get_node_dependencies(graph, node_id)
        for dep_id in dependencies:
            dep_exec = execution.node_executions[dep_id]
            if dep_exec.state != ExecutionState.COMPLETED:
                # Dependency not ready, execute it first
                if not await self._execute_node_chain(graph, execution, dep_id):
                    return False
        
        # Execute current node
        success = await self._execute_single_node(graph, execution, node_id)
        
        if not success:
            return False
        
        # Execute dependent nodes
        dependents = self._get_node_dependents(graph, node_id)
        
        # Check for parallel execution
        parallel_dependents = []
        sequential_dependents = []
        
        for dep_id in dependents:
            edge = self._find_edge(graph, node_id, dep_id)
            if edge and edge.edge_type == EdgeType.PARALLEL_BRANCH:
                parallel_dependents.append(dep_id)
            else:
                sequential_dependents.append(dep_id)
        
        # Execute parallel dependents concurrently
        if parallel_dependents:
            tasks = [
                self._execute_node_chain(graph, execution, dep_id)
                for dep_id in parallel_dependents
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Check if any parallel execution failed
            if not all(isinstance(r, bool) and r for r in results):
                return False
        
        # Execute sequential dependents
        for dep_id in sequential_dependents:
            # Check conditional edges
            edge = self._find_edge(graph, node_id, dep_id)
            if edge and edge.edge_type == EdgeType.CONDITIONAL:
                if not self._evaluate_condition(edge.condition, execution.global_context):
                    continue
            
            if not await self._execute_node_chain(graph, execution, dep_id):
                return False
        
        return True
    
    async def _execute_single_node(self, graph: WorkflowGraph, 
                                  execution: WorkflowExecution, 
                                  node_id: str) -> bool:
        """Execute a single node."""
        node = graph.nodes[node_id]
        node_exec = execution.node_executions[node_id]
        
        node_exec.state = ExecutionState.RUNNING
        node_exec.start_time = datetime.now()
        
        try:
            if node.node_type == NodeType.START:
                # Start nodes just pass through
                node_exec.result = {"message": "Workflow started"}
                
            elif node.node_type == NodeType.END:
                # End nodes collect final results
                node_exec.result = {"message": "Workflow completed"}
                
            elif node.node_type == NodeType.AGENT:
                # Execute agent task
                if node.agent_id not in self.agents:
                    raise ValueError(f"Agent {node.agent_id} not found")
                
                agent = self.agents[node.agent_id]
                
                # Prepare task from context
                task = {
                    "type": node.metadata.get("task_type", "general"),
                    "description": node.metadata.get("description", f"Task for {node.name}"),
                    "context": execution.global_context,
                    **node.metadata
                }
                
                # Execute agent task
                result = await agent.process_with_tools(task)
                node_exec.result = result
                
                # Update global context with result
                execution.global_context[f"result_{node_id}"] = result.content
                execution.global_context[f"metadata_{node_id}"] = result.metadata
                
            elif node.node_type == NodeType.CONDITION:
                # Evaluate condition
                if node.condition_func:
                    result = node.condition_func(execution.global_context)
                    node_exec.result = {"condition_result": result}
                    execution.global_context[f"condition_{node_id}"] = result
                else:
                    raise ValueError(f"Condition node {node_id} has no condition function")
                
            elif node.node_type == NodeType.PARALLEL:
                # Parallel nodes coordinate execution
                node_exec.result = {"message": "Parallel execution coordinated"}
                
            else:
                raise ValueError(f"Unknown node type: {node.node_type}")
            
            node_exec.state = ExecutionState.COMPLETED
            node_exec.end_time = datetime.now()
            return True
            
        except Exception as e:
            node_exec.state = ExecutionState.FAILED
            node_exec.error = str(e)
            node_exec.end_time = datetime.now()
            return False
    
    def _get_node_dependencies(self, graph: WorkflowGraph, node_id: str) -> List[str]:
        """Get dependencies for a node."""
        dependencies = []
        for edge in graph.edges.values():
            if edge.target_node == node_id:
                dependencies.append(edge.source_node)
        return dependencies
    
    def _get_node_dependents(self, graph: WorkflowGraph, node_id: str) -> List[str]:
        """Get dependents for a node."""
        dependents = []
        for edge in graph.edges.values():
            if edge.source_node == node_id:
                dependents.append(edge.target_node)
        return dependents
    
    def _find_edge(self, graph: WorkflowGraph, source: str, target: str) -> Optional[Any]:
        """Find edge between two nodes."""
        for edge in graph.edges.values():
            if edge.source_node == source and edge.target_node == target:
                return edge
        return None
    
    def _evaluate_condition(self, condition: str, context: Dict[str, Any]) -> bool:
        """Evaluate a condition string."""
        try:
            # Simple condition evaluation (extend as needed)
            # For security, only allow basic comparisons
            if condition:
                # Replace context variables
                for key, value in context.items():
                    condition = condition.replace(f"{{{key}}}", str(value))
                
                # Basic evaluation (extend with safer methods in production)
                return eval(condition, {"__builtins__": {}})
            return True
        except:
            return False
    
    def get_execution_status(self, execution_id: str) -> Optional[WorkflowExecution]:
        """Get status of a workflow execution."""
        return self.active_executions.get(execution_id)
    
    def get_execution_summary(self, execution_id: str) -> Dict[str, Any]:
        """Get summary of workflow execution."""
        execution = self.active_executions.get(execution_id)
        if not execution:
            return {"error": "Execution not found"}
        
        completed_nodes = sum(
            1 for node_exec in execution.node_executions.values()
            if node_exec.state == ExecutionState.COMPLETED
        )
        
        failed_nodes = sum(
            1 for node_exec in execution.node_executions.values()
            if node_exec.state == ExecutionState.FAILED
        )
        
        total_nodes = len(execution.node_executions)
        
        duration = None
        if execution.start_time and execution.end_time:
            duration = (execution.end_time - execution.start_time).total_seconds()
        
        return {
            "execution_id": execution_id,
            "workflow_id": execution.workflow_id,
            "state": execution.state.value,
            "progress": {
                "completed": completed_nodes,
                "failed": failed_nodes,
                "total": total_nodes,
                "percentage": round((completed_nodes / total_nodes) * 100, 1)
            },
            "duration_seconds": duration,
            "start_time": execution.start_time.isoformat() if execution.start_time else None,
            "end_time": execution.end_time.isoformat() if execution.end_time else None
        }
    
    async def cancel_execution(self, execution_id: str) -> bool:
        """Cancel a running workflow execution."""
        execution = self.active_executions.get(execution_id)
        if not execution:
            return False
        
        if execution.state == ExecutionState.RUNNING:
            execution.state = ExecutionState.CANCELLED
            execution.end_time = datetime.now()
            return True
        
        return False