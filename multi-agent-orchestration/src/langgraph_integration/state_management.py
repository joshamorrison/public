"""
LangGraph State Management

State schemas and management for LangGraph-based multi-agent workflows.
"""

from typing import Dict, List, Any, Optional, TypedDict, Annotated
from dataclasses import dataclass, field
from datetime import datetime
import operator

# LangGraph imports with fallback
try:
    from langgraph.graph import StateGraph, END, START
    from langgraph.checkpoint.memory import MemorySaver
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    StateGraph = None
    END = "END"
    START = "START"
    MemorySaver = None


class MultiAgentState(TypedDict):
    """
    Shared state for multi-agent workflows.
    
    This state is passed between agents and maintained throughout
    the workflow execution.
    """
    # Core workflow data
    input_data: Dict[str, Any]
    output_data: Dict[str, Any]
    current_step: str
    completed_steps: List[str]
    
    # Agent results and intermediate data
    agent_results: Dict[str, Any]
    intermediate_data: Dict[str, Any]
    
    # Workflow metadata
    workflow_id: str
    workflow_type: str
    start_time: datetime
    current_agent: Optional[str]
    
    # Error handling and feedback
    errors: List[Dict[str, Any]]
    feedback: List[Dict[str, Any]]
    retry_count: int
    
    # Quality and evaluation
    quality_scores: Dict[str, float]
    confidence_scores: Dict[str, float]
    
    # Coordination and control
    next_action: Optional[str]
    routing_decisions: List[Dict[str, Any]]
    parallel_results: Dict[str, Any]


class WorkflowState(TypedDict):
    """
    Extended state for complex workflow management.
    
    Includes advanced features for workflow coordination and monitoring.
    """
    # Inherit from MultiAgentState
    input_data: Dict[str, Any]
    output_data: Dict[str, Any]
    current_step: str
    completed_steps: List[str]
    agent_results: Dict[str, Any]
    intermediate_data: Dict[str, Any]
    workflow_id: str
    workflow_type: str
    start_time: datetime
    current_agent: Optional[str]
    errors: List[Dict[str, Any]]
    feedback: List[Dict[str, Any]]
    retry_count: int
    quality_scores: Dict[str, float]
    confidence_scores: Dict[str, float]
    next_action: Optional[str]
    routing_decisions: List[Dict[str, Any]]
    parallel_results: Dict[str, Any]
    
    # Extended workflow features
    iteration_count: int
    max_iterations: int
    convergence_threshold: float
    supervisor_decisions: List[Dict[str, Any]]
    agent_assignments: Dict[str, str]
    resource_usage: Dict[str, Any]
    
    # Advanced monitoring
    performance_metrics: Dict[str, Any]
    cost_tracking: Dict[str, Any]
    trace_data: List[Dict[str, Any]]


# State reduction functions for LangGraph
def add_agent_result(state: MultiAgentState, agent_id: str, result: Any) -> MultiAgentState:
    """Add agent result to state."""
    new_state = state.copy()
    new_state["agent_results"][agent_id] = result
    return new_state


def update_current_step(state: MultiAgentState, step: str) -> MultiAgentState:
    """Update current workflow step."""
    new_state = state.copy()
    if new_state["current_step"]:
        new_state["completed_steps"].append(new_state["current_step"])
    new_state["current_step"] = step
    return new_state


def add_error(state: MultiAgentState, error: Dict[str, Any]) -> MultiAgentState:
    """Add error to state."""
    new_state = state.copy()
    new_state["errors"].append({
        **error,
        "timestamp": datetime.now(),
        "step": state["current_step"]
    })
    return new_state


def add_feedback(state: MultiAgentState, feedback: Dict[str, Any]) -> MultiAgentState:
    """Add feedback to state."""
    new_state = state.copy()
    new_state["feedback"].append({
        **feedback,
        "timestamp": datetime.now(),
        "step": state["current_step"]
    })
    return new_state


@dataclass
class StateManager:
    """
    Manages state transitions and validation for LangGraph workflows.
    """
    
    def __init__(self):
        """Initialize state manager."""
        self.checkpointer = MemorySaver() if LANGGRAPH_AVAILABLE else None
        self.state_validators: Dict[str, Any] = {}
    
    def create_initial_state(self, workflow_id: str, workflow_type: str,
                           input_data: Dict[str, Any]) -> MultiAgentState:
        """Create initial state for a new workflow."""
        return {
            "input_data": input_data,
            "output_data": {},
            "current_step": "start",
            "completed_steps": [],
            "agent_results": {},
            "intermediate_data": {},
            "workflow_id": workflow_id,
            "workflow_type": workflow_type,
            "start_time": datetime.now(),
            "current_agent": None,
            "errors": [],
            "feedback": [],
            "retry_count": 0,
            "quality_scores": {},
            "confidence_scores": {},
            "next_action": None,
            "routing_decisions": [],
            "parallel_results": {}
        }
    
    def create_workflow_state(self, workflow_id: str, workflow_type: str,
                            input_data: Dict[str, Any], max_iterations: int = 10) -> WorkflowState:
        """Create initial state for complex workflows."""
        base_state = self.create_initial_state(workflow_id, workflow_type, input_data)
        
        # Convert to WorkflowState and add extended fields
        workflow_state = WorkflowState(base_state)
        workflow_state.update({
            "iteration_count": 0,
            "max_iterations": max_iterations,
            "convergence_threshold": 0.95,
            "supervisor_decisions": [],
            "agent_assignments": {},
            "resource_usage": {},
            "performance_metrics": {},
            "cost_tracking": {},
            "trace_data": []
        })
        
        return workflow_state
    
    def validate_state(self, state: MultiAgentState) -> bool:
        """Validate state structure and content."""
        required_fields = [
            "workflow_id", "workflow_type", "input_data", "current_step"
        ]
        
        for field in required_fields:
            if field not in state:
                return False
        
        # Additional validation rules
        if not isinstance(state["completed_steps"], list):
            return False
        
        if not isinstance(state["agent_results"], dict):
            return False
        
        return True
    
    def merge_parallel_results(self, state: MultiAgentState, 
                             results: Dict[str, Any]) -> MultiAgentState:
        """Merge results from parallel agent execution."""
        new_state = state.copy()
        new_state["parallel_results"].update(results)
        
        # Merge into main agent results
        for agent_id, result in results.items():
            new_state["agent_results"][agent_id] = result
        
        return new_state
    
    def should_continue(self, state: MultiAgentState) -> bool:
        """Determine if workflow should continue."""
        # Check for terminal conditions
        if state["current_step"] == "end":
            return False
        
        # Check retry limit
        if state["retry_count"] >= 3:
            return False
        
        # Check for critical errors
        critical_errors = [
            error for error in state["errors"]
            if error.get("severity", "medium") == "critical"
        ]
        
        if critical_errors:
            return False
        
        return True
    
    def get_next_step(self, state: MultiAgentState, routing_rules: Dict[str, Any]) -> str:
        """Determine next step based on current state and routing rules."""
        current_step = state["current_step"]
        
        # Check routing rules
        if current_step in routing_rules:
            rule = routing_rules[current_step]
            
            # Simple conditional routing
            if "condition" in rule:
                condition = rule["condition"]
                if self._evaluate_condition(condition, state):
                    return rule.get("if_true", "end")
                else:
                    return rule.get("if_false", "end")
            
            # Direct routing
            return rule.get("next", "end")
        
        return "end"
    
    def _evaluate_condition(self, condition: str, state: MultiAgentState) -> bool:
        """Evaluate routing condition."""
        try:
            # Simple condition evaluation
            # In production, implement more sophisticated condition parsing
            
            if "confidence >" in condition:
                threshold = float(condition.split(">")[1].strip())
                avg_confidence = sum(state["confidence_scores"].values()) / max(len(state["confidence_scores"]), 1)
                return avg_confidence > threshold
            
            if "errors == 0" in condition:
                return len(state["errors"]) == 0
            
            if "completed_steps contains" in condition:
                step = condition.split("contains")[1].strip().strip("'\"")
                return step in state["completed_steps"]
            
            return True
            
        except Exception:
            return False
    
    def checkpoint_state(self, state: MultiAgentState, step_name: str) -> str:
        """Create a checkpoint of the current state."""
        if not self.checkpointer:
            return f"checkpoint_{step_name}_{int(datetime.now().timestamp())}"
        
        # In a real implementation, save to checkpointer
        checkpoint_id = f"checkpoint_{step_name}_{int(datetime.now().timestamp())}"
        
        # Simulate checkpointing
        return checkpoint_id
    
    def restore_state(self, checkpoint_id: str) -> Optional[MultiAgentState]:
        """Restore state from checkpoint."""
        if not self.checkpointer:
            return None
        
        # In a real implementation, restore from checkpointer
        # For now, return None to indicate checkpoint not found
        return None
    
    def get_state_summary(self, state: MultiAgentState) -> Dict[str, Any]:
        """Get summary of current state."""
        return {
            "workflow_id": state["workflow_id"],
            "workflow_type": state["workflow_type"],
            "current_step": state["current_step"],
            "completed_steps_count": len(state["completed_steps"]),
            "agent_results_count": len(state["agent_results"]),
            "error_count": len(state["errors"]),
            "feedback_count": len(state["feedback"]),
            "retry_count": state["retry_count"],
            "avg_confidence": sum(state["confidence_scores"].values()) / max(len(state["confidence_scores"]), 1),
            "runtime_seconds": (datetime.now() - state["start_time"]).total_seconds(),
            "current_agent": state["current_agent"],
            "next_action": state["next_action"]
        }