"""
Pipeline Pattern

Sequential agent collaboration with handoffs and quality gates.
Agents work in a linear workflow: Agent A → Agent B → Agent C → Output
"""

import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime

from ..agents.base_agent import BaseAgent, AgentResult


class PipelinePattern:
    """
    Pipeline orchestration pattern for sequential agent workflows.
    
    The pipeline pattern:
    - Executes agents in a predefined sequence
    - Passes results from one agent to the next
    - Implements quality gates between stages
    - Supports error recovery and rollback
    - Maintains execution history and metrics
    """

    def __init__(self, pattern_id: str = "pipeline-001"):
        """
        Initialize the pipeline pattern.
        
        Args:
            pattern_id: Unique identifier for this pattern instance
        """
        self.pattern_id = pattern_id
        self.name = "Pipeline Pattern"
        self.description = "Sequential agent workflow with quality gates"
        self.pipeline_stages: List[Dict[str, Any]] = []
        self.execution_history: List[Dict[str, Any]] = []
        self.quality_gates: Dict[int, Dict[str, Any]] = {}

    def add_stage(self, agent: BaseAgent, stage_name: str, 
                  quality_gate: Optional[Dict[str, Any]] = None,
                  rollback_on_failure: bool = True) -> int:
        """
        Add a stage to the pipeline.
        
        Args:
            agent: Agent to execute in this stage
            stage_name: Human-readable name for the stage
            quality_gate: Optional quality requirements for this stage
            rollback_on_failure: Whether to rollback on stage failure
            
        Returns:
            Stage index in the pipeline
        """
        stage_index = len(self.pipeline_stages)
        
        stage_config = {
            "stage_index": stage_index,
            "agent": agent,
            "stage_name": stage_name,
            "rollback_on_failure": rollback_on_failure,
            "created_at": datetime.now()
        }
        
        self.pipeline_stages.append(stage_config)
        
        # Add quality gate if specified
        if quality_gate:
            self.quality_gates[stage_index] = quality_gate
        
        print(f"[PIPELINE] Added stage {stage_index}: {stage_name} ({agent.name})")
        return stage_index

    async def execute(self, initial_task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the complete pipeline.
        
        Args:
            initial_task: Initial task to feed into the pipeline
            
        Returns:
            Pipeline execution results with all stage outputs
        """
        execution_id = f"exec_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        start_time = datetime.now()
        
        print(f"[PIPELINE] Starting execution {execution_id}")
        print(f"[PIPELINE] Pipeline stages: {len(self.pipeline_stages)}")
        
        execution_result = {
            "execution_id": execution_id,
            "pattern_type": "pipeline",
            "start_time": start_time,
            "stages_executed": [],
            "final_result": None,
            "success": True,
            "error_message": None
        }
        
        try:
            current_task = initial_task.copy()
            current_task["pipeline_execution_id"] = execution_id
            
            # Execute each stage in sequence
            for stage_config in self.pipeline_stages:
                stage_result = await self._execute_stage(stage_config, current_task, execution_result)
                
                # Check if stage failed
                if not stage_result["success"]:
                    execution_result["success"] = False
                    execution_result["error_message"] = stage_result.get("error_message", "Stage execution failed")
                    
                    # Handle rollback if enabled
                    if stage_config["rollback_on_failure"]:
                        await self._handle_rollback(execution_result, stage_config["stage_index"])
                    
                    break
                
                # Pass result to next stage
                current_task = self._prepare_next_stage_input(stage_result, current_task)
            
            execution_result["end_time"] = datetime.now()
            execution_result["total_execution_time"] = (execution_result["end_time"] - start_time).total_seconds()
            
            # Set final result
            if execution_result["success"] and execution_result["stages_executed"]:
                final_stage = execution_result["stages_executed"][-1]
                execution_result["final_result"] = final_stage["agent_result"].content
            
            # Store execution history
            self.execution_history.append(execution_result)
            
            print(f"[PIPELINE] Execution {execution_id} completed: {'SUCCESS' if execution_result['success'] else 'FAILED'}")
            return execution_result
            
        except Exception as e:
            execution_result["success"] = False
            execution_result["error_message"] = str(e)
            execution_result["end_time"] = datetime.now()
            
            print(f"[PIPELINE] Execution {execution_id} failed with error: {str(e)}")
            return execution_result

    async def _execute_stage(self, stage_config: Dict[str, Any], 
                           current_task: Dict[str, Any],
                           execution_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a single pipeline stage.
        
        Args:
            stage_config: Configuration for the stage
            current_task: Current task data
            execution_result: Overall execution context
            
        Returns:
            Stage execution result
        """
        stage_index = stage_config["stage_index"]
        stage_name = stage_config["stage_name"]
        agent = stage_config["agent"]
        
        print(f"[PIPELINE] Executing stage {stage_index}: {stage_name}")
        stage_start = datetime.now()
        
        stage_result = {
            "stage_index": stage_index,
            "stage_name": stage_name,
            "agent_id": agent.agent_id,
            "agent_name": agent.name,
            "start_time": stage_start,
            "success": True,
            "agent_result": None,
            "quality_gate_passed": True,
            "error_message": None
        }
        
        try:
            # Execute agent
            agent_result = await agent.process_task(current_task)
            stage_result["agent_result"] = agent_result
            stage_result["success"] = agent_result.success
            
            if not agent_result.success:
                stage_result["error_message"] = agent_result.error_message
            else:
                # Check quality gate if exists
                if stage_index in self.quality_gates:
                    quality_passed = await self._check_quality_gate(
                        stage_index, agent_result, self.quality_gates[stage_index]
                    )
                    stage_result["quality_gate_passed"] = quality_passed
                    
                    if not quality_passed:
                        stage_result["success"] = False
                        stage_result["error_message"] = "Quality gate failed"
            
            stage_result["end_time"] = datetime.now()
            stage_result["execution_time"] = (stage_result["end_time"] - stage_start).total_seconds()
            
            execution_result["stages_executed"].append(stage_result)
            
            status = "SUCCESS" if stage_result["success"] else "FAILED"
            print(f"[PIPELINE] Stage {stage_index} completed: {status}")
            
            return stage_result
            
        except Exception as e:
            stage_result["success"] = False
            stage_result["error_message"] = str(e)
            stage_result["end_time"] = datetime.now()
            
            execution_result["stages_executed"].append(stage_result)
            print(f"[PIPELINE] Stage {stage_index} failed with error: {str(e)}")
            
            return stage_result

    async def _check_quality_gate(self, stage_index: int, agent_result: AgentResult, 
                                quality_gate: Dict[str, Any]) -> bool:
        """
        Check if agent result passes the quality gate.
        
        Args:
            stage_index: Index of the stage
            agent_result: Result from the agent
            quality_gate: Quality gate requirements
            
        Returns:
            True if quality gate passes, False otherwise
        """
        print(f"[PIPELINE] Checking quality gate for stage {stage_index}")
        
        # Check minimum confidence requirement
        min_confidence = quality_gate.get("min_confidence", 0.0)
        if agent_result.confidence < min_confidence:
            print(f"[PIPELINE] Quality gate failed: confidence {agent_result.confidence:.2f} < {min_confidence:.2f}")
            return False
        
        # Check minimum content length requirement
        min_length = quality_gate.get("min_content_length", 0)
        if len(agent_result.content) < min_length:
            print(f"[PIPELINE] Quality gate failed: content length {len(agent_result.content)} < {min_length}")
            return False
        
        # Check required keywords
        required_keywords = quality_gate.get("required_keywords", [])
        content_lower = agent_result.content.lower()
        for keyword in required_keywords:
            if keyword.lower() not in content_lower:
                print(f"[PIPELINE] Quality gate failed: missing required keyword '{keyword}'")
                return False
        
        print(f"[PIPELINE] Quality gate passed for stage {stage_index}")
        return True

    def _prepare_next_stage_input(self, stage_result: Dict[str, Any], 
                                current_task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare input for the next pipeline stage.
        
        Args:
            stage_result: Result from the current stage
            current_task: Current task context
            
        Returns:
            Task input for the next stage
        """
        agent_result = stage_result["agent_result"]
        
        # Create task for next stage
        next_task = current_task.copy()
        next_task.update({
            "description": f"Process results from {stage_result['stage_name']}: {agent_result.content[:200]}...",
            "previous_stage_output": agent_result.content,
            "previous_stage_confidence": agent_result.confidence,
            "previous_stage_metadata": agent_result.metadata,
            "pipeline_context": {
                "current_stage": stage_result["stage_index"] + 1,
                "total_stages": len(self.pipeline_stages),
                "execution_id": current_task.get("pipeline_execution_id"),
                "accumulated_confidence": agent_result.confidence  # Could be enhanced with confidence propagation
            }
        })
        
        return next_task

    async def _handle_rollback(self, execution_result: Dict[str, Any], failed_stage: int):
        """
        Handle pipeline rollback on stage failure.
        
        Args:
            execution_result: Current execution result
            failed_stage: Index of the failed stage
        """
        print(f"[PIPELINE] Handling rollback from failed stage {failed_stage}")
        
        # For now, just log the rollback
        # In a more sophisticated implementation, this could:
        # - Undo changes made by previous stages
        # - Reset state to a previous checkpoint
        # - Trigger alternative workflows
        
        rollback_info = {
            "rollback_triggered": True,
            "failed_stage": failed_stage,
            "rollback_time": datetime.now(),
            "stages_to_rollback": list(range(failed_stage))
        }
        
        execution_result["rollback_info"] = rollback_info
        print(f"[PIPELINE] Rollback completed for {len(rollback_info['stages_to_rollback'])} stages")

    def get_pipeline_configuration(self) -> Dict[str, Any]:
        """
        Get current pipeline configuration.
        
        Returns:
            Pipeline configuration details
        """
        return {
            "pattern_id": self.pattern_id,
            "pattern_name": self.name,
            "stages_count": len(self.pipeline_stages),
            "stages": [
                {
                    "stage_index": stage["stage_index"],
                    "stage_name": stage["stage_name"], 
                    "agent_name": stage["agent"].name,
                    "agent_id": stage["agent"].agent_id,
                    "has_quality_gate": stage["stage_index"] in self.quality_gates,
                    "rollback_on_failure": stage["rollback_on_failure"]
                }
                for stage in self.pipeline_stages
            ],
            "quality_gates": len(self.quality_gates),
            "executions_count": len(self.execution_history)
        }

    def get_execution_metrics(self) -> Dict[str, Any]:
        """
        Get pipeline execution metrics.
        
        Returns:
            Execution performance metrics
        """
        if not self.execution_history:
            return {"no_executions": True}
        
        successful_executions = [ex for ex in self.execution_history if ex["success"]]
        failed_executions = [ex for ex in self.execution_history if not ex["success"]]
        
        # Calculate average execution time for successful runs
        avg_execution_time = 0
        if successful_executions:
            total_time = sum(ex.get("total_execution_time", 0) for ex in successful_executions)
            avg_execution_time = total_time / len(successful_executions)
        
        return {
            "total_executions": len(self.execution_history),
            "successful_executions": len(successful_executions),
            "failed_executions": len(failed_executions),
            "success_rate": len(successful_executions) / len(self.execution_history),
            "average_execution_time": avg_execution_time,
            "last_execution_time": self.execution_history[-1].get("start_time") if self.execution_history else None
        }

    def clear_history(self):
        """Clear execution history."""
        self.execution_history = []
        print(f"[PIPELINE] Execution history cleared")