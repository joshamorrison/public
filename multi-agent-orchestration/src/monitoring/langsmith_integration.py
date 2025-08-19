"""
LangSmith Integration

Provides comprehensive monitoring and tracing for multi-agent workflows using LangSmith.
"""

import os
import json
import asyncio
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from contextlib import asynccontextmanager

# LangSmith imports with fallback
try:
    from langsmith import Client, RunTree
    from langsmith.evaluation import evaluate
    from langsmith.schemas import Run, Example
    LANGSMITH_AVAILABLE = True
except ImportError:
    LANGSMITH_AVAILABLE = False
    Client = None
    RunTree = None


class TraceLevel(Enum):
    """Trace detail levels."""
    MINIMAL = "minimal"
    STANDARD = "standard"
    DETAILED = "detailed"
    DEBUG = "debug"


@dataclass
class LangSmithConfig:
    """Configuration for LangSmith integration."""
    api_key: Optional[str] = None
    project_name: str = "multi-agent-orchestration"
    trace_level: TraceLevel = TraceLevel.STANDARD
    auto_trace: bool = True
    batch_size: int = 100
    flush_interval: int = 30
    enable_evaluations: bool = True
    custom_metadata: Dict[str, Any] = field(default_factory=dict)


class LangSmithMonitor:
    """
    LangSmith monitoring integration for multi-agent workflows.
    
    Provides comprehensive tracing, evaluation, and analytics for agent interactions
    and workflow execution.
    """
    
    def __init__(self, config: LangSmithConfig = None):
        """
        Initialize LangSmith monitor.
        
        Args:
            config: LangSmith configuration
        """
        self.config = config or LangSmithConfig()
        self.client = None
        self.active_runs: Dict[str, Any] = {}
        self.pending_traces = []
        self._session_id = f"session_{int(datetime.now().timestamp())}"
        
        # Initialize client if available
        if LANGSMITH_AVAILABLE:
            self._initialize_client()
        else:
            print("[LANGSMITH] LangSmith not available, using simulation mode")
    
    def _initialize_client(self):
        """Initialize LangSmith client."""
        try:
            api_key = self.config.api_key or os.getenv("LANGSMITH_API_KEY")
            if api_key:
                self.client = Client(api_key=api_key)
                print(f"[LANGSMITH] Connected to project: {self.config.project_name}")
            else:
                print("[LANGSMITH] No API key found, using simulation mode")
        except Exception as e:
            print(f"[LANGSMITH] Failed to initialize client: {str(e)}")
    
    @asynccontextmanager
    async def trace_workflow(self, workflow_name: str, inputs: Dict[str, Any] = None,
                           metadata: Dict[str, Any] = None):
        """
        Context manager for tracing complete workflows.
        
        Args:
            workflow_name: Name of the workflow
            inputs: Workflow inputs
            metadata: Additional metadata
        """
        run_id = f"workflow_{workflow_name}_{int(datetime.now().timestamp())}"
        
        # Start workflow trace
        await self.start_run(
            run_id=run_id,
            name=workflow_name,
            run_type="workflow",
            inputs=inputs or {},
            metadata={
                "session_id": self._session_id,
                "workflow_type": "multi_agent",
                **(metadata or {})
            }
        )
        
        try:
            yield run_id
        except Exception as e:
            await self.end_run(run_id, error=str(e))
            raise
        else:
            await self.end_run(run_id)
    
    @asynccontextmanager
    async def trace_agent(self, agent_id: str, task: Dict[str, Any],
                         parent_run_id: str = None):
        """
        Context manager for tracing individual agent execution.
        
        Args:
            agent_id: Agent identifier
            task: Task being processed
            parent_run_id: Parent workflow run ID
        """
        run_id = f"agent_{agent_id}_{int(datetime.now().timestamp())}"
        
        # Start agent trace
        await self.start_run(
            run_id=run_id,
            name=f"Agent: {agent_id}",
            run_type="agent",
            inputs=task,
            parent_run_id=parent_run_id,
            metadata={
                "agent_id": agent_id,
                "task_type": task.get("type", "unknown"),
                "session_id": self._session_id
            }
        )
        
        try:
            yield run_id
        except Exception as e:
            await self.end_run(run_id, error=str(e))
            raise
        else:
            await self.end_run(run_id)
    
    async def start_run(self, run_id: str, name: str, run_type: str,
                       inputs: Dict[str, Any], parent_run_id: str = None,
                       metadata: Dict[str, Any] = None):
        """Start a new trace run."""
        run_data = {
            "id": run_id,
            "name": name,
            "run_type": run_type,
            "inputs": inputs,
            "start_time": datetime.now(),
            "parent_run_id": parent_run_id,
            "metadata": {
                **self.config.custom_metadata,
                **(metadata or {})
            }
        }
        
        self.active_runs[run_id] = run_data
        
        if self.client and self.config.auto_trace:
            try:
                # Create LangSmith run
                if LANGSMITH_AVAILABLE:
                    run_tree = RunTree(
                        name=name,
                        run_type=run_type,
                        inputs=inputs,
                        project_name=self.config.project_name,
                        parent_run=self.active_runs.get(parent_run_id, {}).get("langsmith_run"),
                        extra=metadata or {}
                    )
                    run_data["langsmith_run"] = run_tree
            except Exception as e:
                print(f"[LANGSMITH] Error starting run: {str(e)}")
    
    async def end_run(self, run_id: str, outputs: Dict[str, Any] = None, 
                     error: str = None):
        """End a trace run."""
        if run_id not in self.active_runs:
            return
        
        run_data = self.active_runs[run_id]
        run_data["end_time"] = datetime.now()
        run_data["duration"] = (run_data["end_time"] - run_data["start_time"]).total_seconds()
        
        if outputs:
            run_data["outputs"] = outputs
        if error:
            run_data["error"] = error
        
        # Update LangSmith run
        if self.client and "langsmith_run" in run_data:
            try:
                langsmith_run = run_data["langsmith_run"]
                if outputs:
                    langsmith_run.end(outputs=outputs)
                elif error:
                    langsmith_run.end(error=error)
                else:
                    langsmith_run.end()
                
                # Post to LangSmith
                if LANGSMITH_AVAILABLE:
                    langsmith_run.post()
                    
            except Exception as e:
                print(f"[LANGSMITH] Error ending run: {str(e)}")
        
        # Store for batch processing
        self.pending_traces.append(run_data.copy())
        
        # Remove from active runs
        del self.active_runs[run_id]
    
    async def log_agent_step(self, run_id: str, step_name: str, 
                           inputs: Dict[str, Any], outputs: Dict[str, Any],
                           metadata: Dict[str, Any] = None):
        """Log individual agent step within a run."""
        step_data = {
            "step_name": step_name,
            "inputs": inputs,
            "outputs": outputs,
            "timestamp": datetime.now(),
            "metadata": metadata or {}
        }
        
        if run_id in self.active_runs:
            if "steps" not in self.active_runs[run_id]:
                self.active_runs[run_id]["steps"] = []
            self.active_runs[run_id]["steps"].append(step_data)
    
    async def log_tool_usage(self, run_id: str, tool_name: str,
                           tool_inputs: Dict[str, Any], tool_outputs: Dict[str, Any],
                           duration: float):
        """Log tool usage within an agent run."""
        tool_data = {
            "tool_name": tool_name,
            "inputs": tool_inputs,
            "outputs": tool_outputs,
            "duration": duration,
            "timestamp": datetime.now()
        }
        
        if run_id in self.active_runs:
            if "tools" not in self.active_runs[run_id]:
                self.active_runs[run_id]["tools"] = []
            self.active_runs[run_id]["tools"].append(tool_data)
    
    async def evaluate_agent_output(self, run_id: str, evaluation_criteria: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate agent output using LangSmith evaluations."""
        if not self.config.enable_evaluations or not self.client:
            return {"status": "skipped", "reason": "evaluations disabled or client unavailable"}
        
        try:
            if run_id not in self.active_runs:
                return {"status": "error", "reason": "run not found"}
            
            run_data = self.active_runs[run_id]
            
            # Simple evaluation metrics
            evaluation_results = {
                "relevance_score": self._evaluate_relevance(run_data, evaluation_criteria),
                "quality_score": self._evaluate_quality(run_data, evaluation_criteria),
                "efficiency_score": self._evaluate_efficiency(run_data, evaluation_criteria),
                "evaluated_at": datetime.now().isoformat()
            }
            
            # Store evaluation results
            run_data["evaluation"] = evaluation_results
            
            return evaluation_results
            
        except Exception as e:
            return {"status": "error", "reason": str(e)}
    
    def _evaluate_relevance(self, run_data: Dict[str, Any], criteria: Dict[str, Any]) -> float:
        """Evaluate relevance of agent output."""
        # Simple heuristic - in production, use LLM-based evaluation
        outputs = run_data.get("outputs", {})
        if not outputs:
            return 0.0
        
        # Check if output contains key terms from input
        inputs = run_data.get("inputs", {})
        input_text = str(inputs).lower()
        output_text = str(outputs).lower()
        
        # Simple keyword overlap
        input_words = set(input_text.split())
        output_words = set(output_text.split())
        
        if not input_words:
            return 0.5
        
        overlap = len(input_words.intersection(output_words))
        return min(overlap / len(input_words), 1.0)
    
    def _evaluate_quality(self, run_data: Dict[str, Any], criteria: Dict[str, Any]) -> float:
        """Evaluate quality of agent output."""
        # Simple heuristic based on output length and structure
        outputs = run_data.get("outputs", {})
        if not outputs:
            return 0.0
        
        output_text = str(outputs)
        
        # Quality indicators
        quality_score = 0.0
        
        # Length check (not too short, not too long)
        if 50 <= len(output_text) <= 2000:
            quality_score += 0.3
        
        # Structure check (has periods, proper capitalization)
        if "." in output_text and any(c.isupper() for c in output_text):
            quality_score += 0.3
        
        # Completeness check (no obvious errors)
        if "error" not in output_text.lower() and "failed" not in output_text.lower():
            quality_score += 0.4
        
        return min(quality_score, 1.0)
    
    def _evaluate_efficiency(self, run_data: Dict[str, Any], criteria: Dict[str, Any]) -> float:
        """Evaluate efficiency of agent execution."""
        duration = run_data.get("duration", float('inf'))
        
        # Simple efficiency based on duration
        if duration <= 1.0:
            return 1.0
        elif duration <= 5.0:
            return 0.8
        elif duration <= 15.0:
            return 0.6
        elif duration <= 30.0:
            return 0.4
        else:
            return 0.2
    
    async def flush_traces(self):
        """Flush pending traces to LangSmith."""
        if not self.pending_traces:
            return
        
        batch = self.pending_traces[:self.config.batch_size]
        self.pending_traces = self.pending_traces[self.config.batch_size:]
        
        if self.client:
            try:
                # In a real implementation, batch upload to LangSmith
                print(f"[LANGSMITH] Flushing {len(batch)} traces to LangSmith")
            except Exception as e:
                print(f"[LANGSMITH] Error flushing traces: {str(e)}")
    
    def get_session_analytics(self) -> Dict[str, Any]:
        """Get analytics for current session."""
        completed_traces = len(self.pending_traces)
        active_traces = len(self.active_runs)
        
        # Calculate basic metrics
        total_duration = sum(
            trace.get("duration", 0) for trace in self.pending_traces
            if "duration" in trace
        )
        
        avg_duration = total_duration / completed_traces if completed_traces > 0 else 0
        
        # Error rate
        error_count = sum(
            1 for trace in self.pending_traces
            if "error" in trace
        )
        error_rate = error_count / completed_traces if completed_traces > 0 else 0
        
        return {
            "session_id": self._session_id,
            "completed_traces": completed_traces,
            "active_traces": active_traces,
            "total_duration": round(total_duration, 2),
            "avg_duration": round(avg_duration, 2),
            "error_rate": round(error_rate, 2),
            "langsmith_enabled": self.client is not None
        }
    
    def export_traces(self, format: str = "json") -> str:
        """Export traces in specified format."""
        if format == "json":
            return json.dumps({
                "session_id": self._session_id,
                "traces": self.pending_traces,
                "exported_at": datetime.now().isoformat()
            }, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported export format: {format}")


# Decorator for automatic tracing
def trace_agent_method(monitor: LangSmithMonitor):
    """Decorator for automatic agent method tracing."""
    def decorator(func):
        async def wrapper(self, *args, **kwargs):
            agent_id = getattr(self, 'agent_id', 'unknown_agent')
            method_name = func.__name__
            
            async with monitor.trace_agent(f"{agent_id}_{method_name}", {"args": args, "kwargs": kwargs}) as run_id:
                start_time = datetime.now()
                try:
                    result = await func(self, *args, **kwargs)
                    end_time = datetime.now()
                    duration = (end_time - start_time).total_seconds()
                    
                    await monitor.log_agent_step(
                        run_id, method_name,
                        {"args": args, "kwargs": kwargs},
                        {"result": result},
                        {"duration": duration}
                    )
                    
                    return result
                except Exception as e:
                    await monitor.end_run(run_id, error=str(e))
                    raise
        
        return wrapper
    return decorator