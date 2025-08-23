"""
LangSmith Client for AutoML Agent Platform

Provides comprehensive monitoring and tracing for multi-agent workflows.
Tracks agent performance, execution traces, and collaboration patterns.
"""

import os
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
import json
import logging
from contextlib import contextmanager

try:
    from langsmith import Client
    from langsmith.schemas import Run, RunTypeEnum
    LANGSMITH_AVAILABLE = True
except ImportError:
    LANGSMITH_AVAILABLE = False
    logging.warning("LangSmith not available. Install with: pip install langsmith")

logger = logging.getLogger(__name__)

class LangSmithTracker:
    """
    LangSmith integration for AutoML Agent monitoring.
    
    Provides comprehensive tracking of:
    - Agent execution traces
    - Performance metrics
    - Collaboration patterns
    - Experiment results
    """
    
    def __init__(self, project_name: Optional[str] = None):
        """Initialize LangSmith tracker."""
        self.enabled = LANGSMITH_AVAILABLE and self._is_configured()
        self.project_name = project_name or os.getenv('LANGCHAIN_PROJECT', 'automl-agent')
        self.client = None
        self.active_runs: Dict[str, str] = {}  # task_id -> run_id mapping
        
        if self.enabled:
            self._initialize_client()
        else:
            logger.info("LangSmith tracking disabled (not configured or unavailable)")
    
    def _is_configured(self) -> bool:
        """Check if LangSmith is properly configured."""
        api_key = os.getenv('LANGCHAIN_API_KEY')
        tracing_enabled = os.getenv('LANGCHAIN_TRACING_V2', '').lower() == 'true'
        return bool(api_key) and tracing_enabled
    
    def _initialize_client(self):
        """Initialize the LangSmith client."""
        try:
            api_key = os.getenv('LANGCHAIN_API_KEY')
            endpoint = os.getenv('LANGCHAIN_ENDPOINT', 'https://api.smith.langchain.com')
            
            self.client = Client(api_key=api_key, api_url=endpoint)
            logger.info(f"LangSmith client initialized for project: {self.project_name}")
            
            # Test connection
            self.client.list_runs(project_name=self.project_name, limit=1)
            
        except Exception as e:
            logger.error(f"Failed to initialize LangSmith client: {e}")
            self.enabled = False
            self.client = None
    
    @contextmanager
    def trace_agent_execution(
        self, 
        agent_name: str, 
        task_description: str,
        task_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Context manager for tracing agent execution.
        
        Usage:
            with tracker.trace_agent_execution("EDAAgent", "Analyze customer data") as run_id:
                # Agent execution code
                pass
        """
        if not self.enabled:
            yield None
            return
        
        task_id = task_id or str(uuid.uuid4())
        run_name = f"{agent_name}_{task_id[:8]}"
        
        # Prepare inputs
        inputs = {
            "agent_name": agent_name,
            "task_description": task_description,
            "task_id": task_id,
            "timestamp": datetime.now().isoformat()
        }
        
        if metadata:
            inputs.update(metadata)
        
        try:
            # Start run
            run = self.client.create_run(
                name=run_name,
                run_type="chain",  # Agent execution is like a chain
                inputs=inputs,
                project_name=self.project_name,
                tags=[f"agent:{agent_name}", "automl", "multi-agent"]
            )
            
            run_id = str(run.id)
            self.active_runs[task_id] = run_id
            
            logger.debug(f"Started LangSmith trace for {agent_name}: {run_id}")
            
            yield run_id
            
        except Exception as e:
            logger.error(f"Failed to start LangSmith trace: {e}")
            yield None
        
        finally:
            # Clean up
            if task_id in self.active_runs:
                del self.active_runs[task_id]
    
    def log_agent_result(
        self,
        task_id: str,
        success: bool,
        outputs: Dict[str, Any],
        error: Optional[str] = None,
        metrics: Optional[Dict[str, float]] = None
    ):
        """Log agent execution results."""
        if not self.enabled or task_id not in self.active_runs:
            return
        
        run_id = self.active_runs[task_id]
        
        try:
            # Prepare outputs
            result_outputs = {
                "success": success,
                "timestamp": datetime.now().isoformat(),
                "execution_time": outputs.get("execution_time", 0),
                "results": outputs
            }
            
            if error:
                result_outputs["error"] = error
            
            if metrics:
                result_outputs["metrics"] = metrics
            
            # Update run with results
            self.client.update_run(
                run_id=run_id,
                outputs=result_outputs,
                end_time=datetime.now(),
                error=error if error else None
            )
            
            logger.debug(f"Logged results to LangSmith run: {run_id}")
            
        except Exception as e:
            logger.error(f"Failed to log results to LangSmith: {e}")
    
    def log_agent_communication(
        self,
        from_agent: str,
        to_agent: str,
        message: str,
        message_type: str = "info",
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Log inter-agent communication."""
        if not self.enabled:
            return
        
        try:
            # Create a communication run
            communication_data = {
                "from_agent": from_agent,
                "to_agent": to_agent,
                "message": message,
                "message_type": message_type,
                "timestamp": datetime.now().isoformat()
            }
            
            if metadata:
                communication_data.update(metadata)
            
            run = self.client.create_run(
                name=f"communication_{from_agent}_to_{to_agent}",
                run_type="tool",  # Communication is like a tool call
                inputs=communication_data,
                outputs={"delivered": True},
                project_name=self.project_name,
                tags=["communication", "inter-agent", f"from:{from_agent}", f"to:{to_agent}"]
            )
            
            logger.debug(f"Logged communication to LangSmith: {from_agent} -> {to_agent}")
            
        except Exception as e:
            logger.error(f"Failed to log communication to LangSmith: {e}")
    
    def log_workflow_execution(
        self,
        workflow_id: str,
        agents: List[str],
        start_time: datetime,
        end_time: Optional[datetime] = None,
        success: bool = True,
        results: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None
    ):
        """Log complete workflow execution."""
        if not self.enabled:
            return
        
        try:
            inputs = {
                "workflow_id": workflow_id,
                "agents": agents,
                "start_time": start_time.isoformat(),
                "agent_count": len(agents)
            }
            
            outputs = {
                "success": success,
                "end_time": (end_time or datetime.now()).isoformat(),
                "execution_time": ((end_time or datetime.now()) - start_time).total_seconds(),
                "results": results or {}
            }
            
            if error:
                outputs["error"] = error
            
            run = self.client.create_run(
                name=f"workflow_{workflow_id}",
                run_type="chain",
                inputs=inputs,
                outputs=outputs,
                start_time=start_time,
                end_time=end_time or datetime.now(),
                project_name=self.project_name,
                tags=["workflow", "multi-agent", f"agents:{len(agents)}"] + [f"agent:{agent}" for agent in agents],
                error=error if error else None
            )
            
            logger.info(f"Logged workflow execution to LangSmith: {workflow_id}")
            
        except Exception as e:
            logger.error(f"Failed to log workflow to LangSmith: {e}")
    
    def get_project_statistics(self, days_back: int = 7) -> Dict[str, Any]:
        """Get project statistics from LangSmith."""
        if not self.enabled:
            return {}
        
        try:
            # Calculate date range
            end_time = datetime.now()
            start_time = end_time - timedelta(days=days_back)
            
            # Get runs from the project
            runs = list(self.client.list_runs(
                project_name=self.project_name,
                start_time=start_time,
                end_time=end_time
            ))
            
            # Calculate statistics
            total_runs = len(runs)
            successful_runs = sum(1 for run in runs if not run.error)
            agent_runs = [run for run in runs if any(tag.startswith('agent:') for tag in (run.tags or []))]
            workflow_runs = [run for run in runs if 'workflow' in (run.tags or [])]
            
            # Agent performance
            agent_stats = {}
            for run in agent_runs:
                agent_tags = [tag for tag in (run.tags or []) if tag.startswith('agent:')]
                if agent_tags:
                    agent_name = agent_tags[0].split(':')[1]
                    if agent_name not in agent_stats:
                        agent_stats[agent_name] = {'total': 0, 'successful': 0, 'avg_duration': 0}
                    
                    agent_stats[agent_name]['total'] += 1
                    if not run.error:
                        agent_stats[agent_name]['successful'] += 1
                    
                    if run.start_time and run.end_time:
                        duration = (run.end_time - run.start_time).total_seconds()
                        agent_stats[agent_name]['avg_duration'] += duration
            
            # Calculate averages
            for agent_name in agent_stats:
                if agent_stats[agent_name]['total'] > 0:
                    agent_stats[agent_name]['avg_duration'] /= agent_stats[agent_name]['total']
                    agent_stats[agent_name]['success_rate'] = agent_stats[agent_name]['successful'] / agent_stats[agent_name]['total']
            
            return {
                "project_name": self.project_name,
                "date_range": f"{start_time.isoformat()} to {end_time.isoformat()}",
                "total_runs": total_runs,
                "successful_runs": successful_runs,
                "success_rate": successful_runs / total_runs if total_runs > 0 else 0,
                "agent_runs": len(agent_runs),
                "workflow_runs": len(workflow_runs),
                "agent_statistics": agent_stats
            }
            
        except Exception as e:
            logger.error(f"Failed to get project statistics: {e}")
            return {}
    
    def is_enabled(self) -> bool:
        """Check if LangSmith tracking is enabled."""
        return self.enabled

# Global instance for easy access
tracker = LangSmithTracker()

def get_tracker() -> LangSmithTracker:
    """Get the global LangSmith tracker instance."""
    return tracker