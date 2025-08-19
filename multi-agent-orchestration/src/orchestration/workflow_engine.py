"""
Workflow Engine

Core pattern execution engine that orchestrates different multi-agent patterns
and manages workflow execution across the platform.
"""

import asyncio
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from enum import Enum

from ..patterns.pipeline_pattern import PipelinePattern
from ..patterns.supervisor_pattern import SupervisorPattern
from ..patterns.parallel_pattern import ParallelPattern
from ..patterns.reflective_pattern import ReflectivePattern
from .result_aggregator import ResultAggregator
from .state_manager import StateManager


class WorkflowStatus(Enum):
    """Workflow execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class WorkflowEngine:
    """
    Core workflow execution engine for multi-agent patterns.
    
    The workflow engine:
    - Orchestrates execution of different pattern types
    - Manages workflow lifecycle and state
    - Provides unified execution interface
    - Handles pattern composition and chaining
    - Tracks execution metrics and history
    """

    def __init__(self, engine_id: str = "workflow-engine-001"):
        """
        Initialize the workflow engine.
        
        Args:
            engine_id: Unique identifier for this engine instance
        """
        self.engine_id = engine_id
        self.name = "Multi-Agent Workflow Engine"
        self.version = "1.0.0"
        
        # Core components
        self.result_aggregator = ResultAggregator()
        self.state_manager = StateManager()
        
        # Execution tracking
        self.active_workflows: Dict[str, Dict[str, Any]] = {}
        self.workflow_history: List[Dict[str, Any]] = []
        self.execution_metrics = {
            "total_workflows": 0,
            "successful_workflows": 0,
            "failed_workflows": 0,
            "average_execution_time": 0.0,
            "patterns_executed": {
                "pipeline": 0,
                "supervisor": 0,
                "parallel": 0,
                "reflective": 0
            }
        }

    async def execute_pattern(self, pattern: Union[PipelinePattern, SupervisorPattern, ParallelPattern, ReflectivePattern],
                            task: Dict[str, Any], 
                            workflow_id: Optional[str] = None,
                            execution_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute a single multi-agent pattern.
        
        Args:
            pattern: Pattern instance to execute
            task: Task specification for the pattern
            workflow_id: Optional workflow identifier
            execution_config: Optional execution configuration
            
        Returns:
            Workflow execution result
        """
        if not workflow_id:
            workflow_id = f"workflow_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:17]}"
        
        start_time = datetime.now()
        
        # Initialize workflow tracking
        workflow_context = {
            "workflow_id": workflow_id,
            "pattern_type": type(pattern).__name__.replace("Pattern", "").lower(),
            "pattern_id": getattr(pattern, 'pattern_id', 'unknown'),
            "status": WorkflowStatus.RUNNING,
            "start_time": start_time,
            "task": task,
            "config": execution_config or {},
            "result": None,
            "error": None
        }
        
        self.active_workflows[workflow_id] = workflow_context
        
        print(f"[WORKFLOW_ENGINE] Starting workflow {workflow_id}")
        print(f"[WORKFLOW_ENGINE] Pattern type: {workflow_context['pattern_type']}")
        
        try:
            # Store initial state
            await self.state_manager.store_state(workflow_id, {
                "phase": "execution",
                "status": "running",
                "start_time": start_time.isoformat()
            })
            
            # Execute pattern based on type
            if isinstance(pattern, PipelinePattern):
                result = await self._execute_pipeline(pattern, task, workflow_context)
            elif isinstance(pattern, SupervisorPattern):
                result = await self._execute_supervisor(pattern, task, workflow_context)
            elif isinstance(pattern, ParallelPattern):
                result = await self._execute_parallel(pattern, task, workflow_context)
            elif isinstance(pattern, ReflectivePattern):
                result = await self._execute_reflective(pattern, task, workflow_context)
            else:
                raise ValueError(f"Unsupported pattern type: {type(pattern)}")
            
            # Process successful execution
            workflow_context["status"] = WorkflowStatus.COMPLETED
            workflow_context["result"] = result
            workflow_context["success"] = result.get("success", True)
            
            # Update metrics
            self.execution_metrics["successful_workflows"] += 1
            self.execution_metrics["patterns_executed"][workflow_context["pattern_type"]] += 1
            
            print(f"[WORKFLOW_ENGINE] Workflow {workflow_id} completed successfully")
            
        except Exception as e:
            # Handle execution failure
            workflow_context["status"] = WorkflowStatus.FAILED
            workflow_context["error"] = str(e)
            workflow_context["success"] = False
            
            self.execution_metrics["failed_workflows"] += 1
            
            print(f"[WORKFLOW_ENGINE] Workflow {workflow_id} failed: {str(e)}")
        
        finally:
            # Finalize workflow
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            workflow_context["end_time"] = end_time
            workflow_context["execution_time"] = execution_time
            
            # Update overall metrics
            self.execution_metrics["total_workflows"] += 1
            total_time = self.execution_metrics["average_execution_time"] * (self.execution_metrics["total_workflows"] - 1)
            self.execution_metrics["average_execution_time"] = (total_time + execution_time) / self.execution_metrics["total_workflows"]
            
            # Store final state
            await self.state_manager.store_state(workflow_id, {
                "phase": "completed",
                "status": workflow_context["status"].value,
                "end_time": end_time.isoformat(),
                "execution_time": execution_time,
                "success": workflow_context.get("success", False)
            })
            
            # Move to history and clean up active workflows
            self.workflow_history.append(workflow_context.copy())
            if workflow_id in self.active_workflows:
                del self.active_workflows[workflow_id]
        
        return {
            "workflow_id": workflow_id,
            "status": workflow_context["status"].value,
            "success": workflow_context.get("success", False),
            "result": workflow_context.get("result"),
            "error": workflow_context.get("error"),
            "execution_time": workflow_context.get("execution_time", 0),
            "pattern_type": workflow_context["pattern_type"]
        }

    async def _execute_pipeline(self, pattern: PipelinePattern, task: Dict[str, Any], 
                              context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute pipeline pattern."""
        print(f"[WORKFLOW_ENGINE] Executing pipeline with {len(pattern.pipeline_stages)} stages")
        
        # Add workflow context to task
        enriched_task = task.copy()
        enriched_task["workflow_context"] = {
            "workflow_id": context["workflow_id"],
            "engine_id": self.engine_id
        }
        
        result = await pattern.execute(enriched_task)
        
        # Enhance result with workflow information
        result["workflow_metadata"] = {
            "engine_id": self.engine_id,
            "workflow_id": context["workflow_id"],
            "pattern_configuration": pattern.get_pipeline_configuration()
        }
        
        return result

    async def _execute_supervisor(self, pattern: SupervisorPattern, task: Dict[str, Any], 
                                context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute supervisor pattern."""
        print(f"[WORKFLOW_ENGINE] Executing supervisor with {len(pattern.supervisor.specialist_agents)} specialists")
        
        # Add workflow context to task
        enriched_task = task.copy()
        enriched_task["workflow_context"] = {
            "workflow_id": context["workflow_id"],
            "engine_id": self.engine_id
        }
        
        result = await pattern.execute(enriched_task)
        
        # Enhance result with workflow information
        result["workflow_metadata"] = {
            "engine_id": self.engine_id,
            "workflow_id": context["workflow_id"],
            "pattern_configuration": pattern.get_pattern_configuration()
        }
        
        return result

    async def _execute_parallel(self, pattern: ParallelPattern, task: Dict[str, Any], 
                              context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute parallel pattern."""
        print(f"[WORKFLOW_ENGINE] Executing parallel with {len(pattern.parallel_agents)} agents")
        
        # Add workflow context to task
        enriched_task = task.copy()
        enriched_task["workflow_context"] = {
            "workflow_id": context["workflow_id"],
            "engine_id": self.engine_id
        }
        
        # Extract execution parameters from config
        config = context.get("config", {})
        max_concurrent = config.get("max_concurrent")
        timeout = config.get("timeout")
        
        result = await pattern.execute(enriched_task, max_concurrent, timeout)
        
        # Enhance result with workflow information
        result["workflow_metadata"] = {
            "engine_id": self.engine_id,
            "workflow_id": context["workflow_id"],
            "pattern_configuration": pattern.get_pattern_configuration()
        }
        
        return result

    async def _execute_reflective(self, pattern: ReflectivePattern, task: Dict[str, Any], 
                                context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute reflective pattern."""
        critics_count = len(pattern.critic_agents)
        primary_agent = pattern.primary_agent.name if pattern.primary_agent else "None"
        
        print(f"[WORKFLOW_ENGINE] Executing reflective with primary agent: {primary_agent}, critics: {critics_count}")
        
        # Add workflow context to task
        enriched_task = task.copy()
        enriched_task["workflow_context"] = {
            "workflow_id": context["workflow_id"],
            "engine_id": self.engine_id
        }
        
        result = await pattern.execute(enriched_task)
        
        # Enhance result with workflow information
        result["workflow_metadata"] = {
            "engine_id": self.engine_id,
            "workflow_id": context["workflow_id"],
            "pattern_configuration": pattern.get_pattern_configuration()
        }
        
        return result

    async def execute_workflow_chain(self, workflow_chain: List[Dict[str, Any]], 
                                   initial_task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a chain of patterns in sequence.
        
        Args:
            workflow_chain: List of pattern configurations to execute
            initial_task: Initial task for the workflow chain
            
        Returns:
            Chain execution result
        """
        chain_id = f"chain_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:17]}"
        start_time = datetime.now()
        
        print(f"[WORKFLOW_ENGINE] Starting workflow chain {chain_id} with {len(workflow_chain)} patterns")
        
        chain_result = {
            "chain_id": chain_id,
            "start_time": start_time,
            "patterns_executed": [],
            "final_result": None,
            "success": True,
            "error": None
        }
        
        try:
            current_task = initial_task.copy()
            
            # Execute each pattern in the chain
            for i, pattern_config in enumerate(workflow_chain):
                pattern = pattern_config["pattern"]
                config = pattern_config.get("config", {})
                
                workflow_id = f"{chain_id}_step_{i+1}"
                
                # Execute pattern
                result = await self.execute_pattern(pattern, current_task, workflow_id, config)
                
                chain_result["patterns_executed"].append({
                    "step": i + 1,
                    "pattern_type": result["pattern_type"],
                    "workflow_id": result["workflow_id"],
                    "success": result["success"],
                    "execution_time": result["execution_time"]
                })
                
                # Check for failure
                if not result["success"]:
                    chain_result["success"] = False
                    chain_result["error"] = f"Chain failed at step {i+1}: {result.get('error', 'Unknown error')}"
                    break
                
                # Prepare task for next pattern (if any)
                if i < len(workflow_chain) - 1:
                    # Use result as input for next pattern
                    current_task = await self._prepare_chain_task(result, current_task, i+1)
                else:
                    # Final result
                    chain_result["final_result"] = result["result"]
            
            chain_result["end_time"] = datetime.now()
            chain_result["total_execution_time"] = (chain_result["end_time"] - start_time).total_seconds()
            
            print(f"[WORKFLOW_ENGINE] Chain {chain_id} completed: {'SUCCESS' if chain_result['success'] else 'FAILED'}")
            
            return chain_result
            
        except Exception as e:
            chain_result["success"] = False
            chain_result["error"] = str(e)
            chain_result["end_time"] = datetime.now()
            
            print(f"[WORKFLOW_ENGINE] Chain {chain_id} failed with error: {str(e)}")
            return chain_result

    async def _prepare_chain_task(self, previous_result: Dict[str, Any], 
                                original_task: Dict[str, Any], step: int) -> Dict[str, Any]:
        """
        Prepare task for next step in workflow chain.
        
        Args:
            previous_result: Result from previous pattern
            original_task: Original task specification
            step: Current step number
            
        Returns:
            Task prepared for next pattern
        """
        next_task = original_task.copy()
        
        # Add context from previous step
        next_task["chain_context"] = {
            "step": step,
            "previous_result": previous_result.get("result"),
            "previous_pattern": previous_result.get("pattern_type"),
            "previous_success": previous_result.get("success", False)
        }
        
        # Use aggregated result as primary content if available
        if previous_result.get("result"):
            result_content = previous_result["result"]
            
            # Extract content based on result structure
            if isinstance(result_content, dict):
                if "final_result" in result_content:
                    content = str(result_content["final_result"])
                elif "fused_result" in result_content and result_content["fused_result"]:
                    content = str(result_content["fused_result"].get("content", ""))
                else:
                    content = str(result_content)
            else:
                content = str(result_content)
            
            # Update task description with previous results
            next_task["description"] = f"Process results from previous step: {content[:200]}..."
            next_task["previous_step_output"] = content
        
        return next_task

    def get_active_workflows(self) -> Dict[str, Any]:
        """Get information about currently active workflows."""
        return {
            "active_count": len(self.active_workflows),
            "workflows": {
                workflow_id: {
                    "pattern_type": context["pattern_type"],
                    "status": context["status"].value,
                    "start_time": context["start_time"].isoformat(),
                    "running_time": (datetime.now() - context["start_time"]).total_seconds()
                }
                for workflow_id, context in self.active_workflows.items()
            }
        }

    def get_execution_metrics(self) -> Dict[str, Any]:
        """Get comprehensive execution metrics."""
        success_rate = 0
        if self.execution_metrics["total_workflows"] > 0:
            success_rate = self.execution_metrics["successful_workflows"] / self.execution_metrics["total_workflows"]
        
        return {
            "engine_id": self.engine_id,
            "engine_version": self.version,
            "total_workflows_executed": self.execution_metrics["total_workflows"],
            "successful_workflows": self.execution_metrics["successful_workflows"],
            "failed_workflows": self.execution_metrics["failed_workflows"],
            "success_rate": success_rate,
            "average_execution_time": self.execution_metrics["average_execution_time"],
            "patterns_usage": self.execution_metrics["patterns_executed"],
            "active_workflows": len(self.active_workflows),
            "workflow_history_length": len(self.workflow_history)
        }

    def get_workflow_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get workflow execution history."""
        history = self.workflow_history.copy()
        history.sort(key=lambda x: x["start_time"], reverse=True)
        
        if limit:
            history = history[:limit]
        
        # Return simplified history
        return [
            {
                "workflow_id": w["workflow_id"],
                "pattern_type": w["pattern_type"],
                "status": w["status"].value,
                "success": w.get("success", False),
                "start_time": w["start_time"].isoformat(),
                "execution_time": w.get("execution_time", 0),
                "error": w.get("error")
            }
            for w in history
        ]

    async def cancel_workflow(self, workflow_id: str) -> bool:
        """
        Cancel an active workflow.
        
        Args:
            workflow_id: Workflow to cancel
            
        Returns:
            True if cancelled, False if not found or already completed
        """
        if workflow_id not in self.active_workflows:
            return False
        
        workflow_context = self.active_workflows[workflow_id]
        workflow_context["status"] = WorkflowStatus.CANCELLED
        workflow_context["end_time"] = datetime.now()
        
        # Move to history
        self.workflow_history.append(workflow_context.copy())
        del self.active_workflows[workflow_id]
        
        print(f"[WORKFLOW_ENGINE] Workflow {workflow_id} cancelled")
        return True

    def clear_history(self):
        """Clear workflow history."""
        self.workflow_history.clear()
        print("[WORKFLOW_ENGINE] Workflow history cleared")