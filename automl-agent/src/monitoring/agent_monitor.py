"""
Agent Performance Monitor for AutoML Platform

Comprehensive monitoring system for tracking agent performance,
execution metrics, and collaboration patterns in multi-agent workflows.
"""

import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
import json
import logging
from contextlib import contextmanager

from .langsmith_client import get_tracker

logger = logging.getLogger(__name__)

@dataclass
class AgentMetrics:
    """Agent performance metrics."""
    agent_name: str
    total_executions: int = 0
    successful_executions: int = 0
    failed_executions: int = 0
    total_execution_time: float = 0.0
    avg_execution_time: float = 0.0
    min_execution_time: float = float('inf')
    max_execution_time: float = 0.0
    last_execution: Optional[datetime] = None
    error_rate: float = 0.0
    success_rate: float = 0.0
    current_load: int = 0  # Number of active executions
    
    def update_execution(self, execution_time: float, success: bool):
        """Update metrics with new execution data."""
        self.total_executions += 1
        self.total_execution_time += execution_time
        self.avg_execution_time = self.total_execution_time / self.total_executions
        
        if execution_time < self.min_execution_time:
            self.min_execution_time = execution_time
        if execution_time > self.max_execution_time:
            self.max_execution_time = execution_time
            
        if success:
            self.successful_executions += 1
        else:
            self.failed_executions += 1
            
        self.success_rate = self.successful_executions / self.total_executions
        self.error_rate = self.failed_executions / self.total_executions
        self.last_execution = datetime.now()

@dataclass 
class ExecutionContext:
    """Context for tracking individual agent executions."""
    agent_name: str
    task_id: str
    start_time: datetime
    task_description: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    langsmith_run_id: Optional[str] = None

class AgentPerformanceMonitor:
    """
    Comprehensive agent performance monitoring system.
    
    Features:
    - Real-time performance metrics
    - Execution tracing
    - Load balancing insights
    - Error pattern analysis
    - LangSmith integration
    """
    
    def __init__(self, buffer_size: int = 1000):
        """Initialize the performance monitor."""
        self.metrics: Dict[str, AgentMetrics] = {}
        self.active_executions: Dict[str, ExecutionContext] = {}
        self.execution_history: deque = deque(maxlen=buffer_size)
        self.communication_log: deque = deque(maxlen=buffer_size) 
        self.langsmith_tracker = get_tracker()
        self._lock = threading.Lock()
        
        # Performance callbacks
        self.performance_callbacks: List[Callable] = []
        self.alert_callbacks: List[Callable] = []
        
        logger.info("Agent Performance Monitor initialized")
    
    def get_or_create_metrics(self, agent_name: str) -> AgentMetrics:
        """Get or create metrics for an agent."""
        if agent_name not in self.metrics:
            self.metrics[agent_name] = AgentMetrics(agent_name=agent_name)
        return self.metrics[agent_name]
    
    @contextmanager
    def track_execution(
        self, 
        agent_name: str, 
        task_description: str,
        task_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Context manager for tracking agent execution.
        
        Usage:
            with monitor.track_execution("EDAAgent", "Analyze data") as ctx:
                # Agent execution code
                pass
        """
        import uuid
        task_id = task_id or str(uuid.uuid4())
        start_time = datetime.now()
        
        # Create execution context
        context = ExecutionContext(
            agent_name=agent_name,
            task_id=task_id,
            start_time=start_time,
            task_description=task_description,
            metadata=metadata or {}
        )
        
        # Update current load
        with self._lock:
            metrics = self.get_or_create_metrics(agent_name)
            metrics.current_load += 1
            self.active_executions[task_id] = context
        
        # Start LangSmith tracing
        langsmith_context = None
        if self.langsmith_tracker.is_enabled():
            langsmith_context = self.langsmith_tracker.trace_agent_execution(
                agent_name=agent_name,
                task_description=task_description,
                task_id=task_id,
                metadata=metadata
            )
            context.langsmith_run_id = langsmith_context.__enter__()
        
        success = False
        error = None
        
        try:
            yield context
            success = True
            
        except Exception as e:
            error = str(e)
            success = False
            logger.error(f"Agent {agent_name} execution failed: {e}")
            raise
            
        finally:
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            # Update metrics
            with self._lock:
                metrics = self.get_or_create_metrics(agent_name)
                metrics.update_execution(execution_time, success)
                metrics.current_load -= 1
                
                # Remove from active executions
                if task_id in self.active_executions:
                    del self.active_executions[task_id]
                
                # Add to execution history
                execution_record = {
                    "agent_name": agent_name,
                    "task_id": task_id,
                    "task_description": task_description,
                    "start_time": start_time,
                    "end_time": end_time,
                    "execution_time": execution_time,
                    "success": success,
                    "error": error,
                    "metadata": metadata or {}
                }
                self.execution_history.append(execution_record)
            
            # Log to LangSmith
            if langsmith_context and self.langsmith_tracker.is_enabled():
                self.langsmith_tracker.log_agent_result(
                    task_id=task_id,
                    success=success,
                    outputs={
                        "execution_time": execution_time,
                        "end_time": end_time.isoformat()
                    },
                    error=error
                )
                langsmith_context.__exit__(None, None, None)
            
            # Check for performance alerts
            self._check_performance_alerts(agent_name, metrics)
            
            # Trigger performance callbacks
            self._trigger_callbacks(agent_name, execution_record)
    
    def log_agent_communication(
        self,
        from_agent: str,
        to_agent: str,
        message: str,
        message_type: str = "info",
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Log communication between agents."""
        timestamp = datetime.now()
        
        communication_record = {
            "timestamp": timestamp,
            "from_agent": from_agent,
            "to_agent": to_agent,
            "message": message,
            "message_type": message_type,
            "metadata": metadata or {}
        }
        
        with self._lock:
            self.communication_log.append(communication_record)
        
        # Log to LangSmith
        if self.langsmith_tracker.is_enabled():
            self.langsmith_tracker.log_agent_communication(
                from_agent=from_agent,
                to_agent=to_agent,
                message=message,
                message_type=message_type,
                metadata=metadata
            )
        
        logger.debug(f"Agent communication: {from_agent} -> {to_agent}: {message}")
    
    def get_agent_metrics(self, agent_name: str) -> Optional[AgentMetrics]:
        """Get metrics for a specific agent."""
        return self.metrics.get(agent_name)
    
    def get_all_metrics(self) -> Dict[str, AgentMetrics]:
        """Get metrics for all agents."""
        return dict(self.metrics)
    
    def get_system_overview(self) -> Dict[str, Any]:
        """Get overall system performance overview."""
        with self._lock:
            total_agents = len(self.metrics)
            total_executions = sum(m.total_executions for m in self.metrics.values())
            total_successful = sum(m.successful_executions for m in self.metrics.values())
            total_failed = sum(m.failed_executions for m in self.metrics.values())
            active_agents = sum(1 for m in self.metrics.values() if m.current_load > 0)
            total_load = sum(m.current_load for m in self.metrics.values())
            
            avg_success_rate = (total_successful / total_executions) if total_executions > 0 else 0
            avg_execution_time = sum(m.avg_execution_time * m.total_executions for m in self.metrics.values()) / total_executions if total_executions > 0 else 0
            
            # Recent performance (last hour)
            recent_threshold = datetime.now() - timedelta(hours=1)
            recent_executions = [
                record for record in self.execution_history
                if record["start_time"] > recent_threshold
            ]
            
            return {
                "timestamp": datetime.now(),
                "total_agents": total_agents,
                "active_agents": active_agents,
                "total_load": total_load,
                "performance_summary": {
                    "total_executions": total_executions,
                    "successful_executions": total_successful,
                    "failed_executions": total_failed,
                    "success_rate": avg_success_rate,
                    "average_execution_time": avg_execution_time
                },
                "recent_performance": {
                    "executions_last_hour": len(recent_executions),
                    "success_rate_last_hour": sum(1 for r in recent_executions if r["success"]) / len(recent_executions) if recent_executions else 0
                },
                "agent_load_distribution": {
                    agent_name: metrics.current_load 
                    for agent_name, metrics in self.metrics.items()
                }
            }
    
    def get_performance_trends(self, hours_back: int = 24) -> Dict[str, Any]:
        """Get performance trends over time."""
        threshold = datetime.now() - timedelta(hours=hours_back)
        
        with self._lock:
            recent_executions = [
                record for record in self.execution_history
                if record["start_time"] > threshold
            ]
        
        # Group by hour
        hourly_stats = defaultdict(lambda: {"executions": 0, "successes": 0, "total_time": 0.0})
        
        for record in recent_executions:
            hour_key = record["start_time"].strftime("%Y-%m-%d %H:00")
            hourly_stats[hour_key]["executions"] += 1
            if record["success"]:
                hourly_stats[hour_key]["successes"] += 1
            hourly_stats[hour_key]["total_time"] += record["execution_time"]
        
        # Calculate rates
        trends = {}
        for hour, stats in hourly_stats.items():
            trends[hour] = {
                "executions": stats["executions"],
                "success_rate": stats["successes"] / stats["executions"] if stats["executions"] > 0 else 0,
                "avg_execution_time": stats["total_time"] / stats["executions"] if stats["executions"] > 0 else 0
            }
        
        return {
            "time_period": f"Last {hours_back} hours",
            "trends": dict(sorted(trends.items()))
        }
    
    def _check_performance_alerts(self, agent_name: str, metrics: AgentMetrics):
        """Check for performance issues and trigger alerts."""
        alerts = []
        
        # High error rate alert
        if metrics.error_rate > 0.1 and metrics.total_executions >= 5:  # >10% error rate with sufficient data
            alerts.append({
                "type": "high_error_rate",
                "agent": agent_name,
                "error_rate": metrics.error_rate,
                "message": f"Agent {agent_name} has high error rate: {metrics.error_rate:.1%}"
            })
        
        # Slow execution alert
        if metrics.avg_execution_time > 60:  # >60 seconds average
            alerts.append({
                "type": "slow_execution",
                "agent": agent_name,
                "avg_time": metrics.avg_execution_time,
                "message": f"Agent {agent_name} has slow average execution time: {metrics.avg_execution_time:.1f}s"
            })
        
        # High load alert
        if metrics.current_load > 5:  # More than 5 concurrent executions
            alerts.append({
                "type": "high_load",
                "agent": agent_name,
                "load": metrics.current_load,
                "message": f"Agent {agent_name} has high load: {metrics.current_load} concurrent executions"
            })
        
        # Trigger alert callbacks
        for alert in alerts:
            for callback in self.alert_callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    logger.error(f"Alert callback failed: {e}")
    
    def _trigger_callbacks(self, agent_name: str, execution_record: Dict[str, Any]):
        """Trigger performance callbacks."""
        for callback in self.performance_callbacks:
            try:
                callback(agent_name, execution_record)
            except Exception as e:
                logger.error(f"Performance callback failed: {e}")
    
    def add_performance_callback(self, callback: Callable):
        """Add a callback for performance events."""
        self.performance_callbacks.append(callback)
    
    def add_alert_callback(self, callback: Callable):
        """Add a callback for performance alerts."""
        self.alert_callbacks.append(callback)
    
    def get_communication_log(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent agent communication log."""
        with self._lock:
            return list(self.communication_log)[-limit:]
    
    def reset_metrics(self):
        """Reset all performance metrics."""
        with self._lock:
            self.metrics.clear()
            self.active_executions.clear()
            self.execution_history.clear()
            self.communication_log.clear()
        
        logger.info("Agent performance metrics reset")

# Global monitor instance
monitor = AgentPerformanceMonitor()

def get_monitor() -> AgentPerformanceMonitor:
    """Get the global agent performance monitor."""
    return monitor