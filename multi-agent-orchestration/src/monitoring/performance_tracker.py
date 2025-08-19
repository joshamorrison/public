"""
Performance Tracking

Comprehensive performance monitoring and metrics collection for multi-agent workflows.
"""

import time
import asyncio
import psutil
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
from enum import Enum
import json


class MetricType(Enum):
    """Types of metrics to track."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


@dataclass
class MetricPoint:
    """Individual metric data point."""
    name: str
    value: float
    timestamp: datetime
    labels: Dict[str, str] = field(default_factory=dict)
    metric_type: MetricType = MetricType.GAUGE


@dataclass
class PerformanceSnapshot:
    """Performance snapshot at a point in time."""
    timestamp: datetime
    cpu_percent: float
    memory_mb: float
    memory_percent: float
    active_agents: int
    completed_tasks: int
    error_rate: float
    avg_response_time: float


class MetricsCollector:
    """
    Collects and aggregates performance metrics.
    
    Provides a centralized way to track various performance indicators
    across the multi-agent platform.
    """
    
    def __init__(self, retention_hours: int = 24):
        """
        Initialize metrics collector.
        
        Args:
            retention_hours: How long to retain metrics data
        """
        self.retention_hours = retention_hours
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.counters: Dict[str, float] = defaultdict(float)
        self.gauges: Dict[str, float] = defaultdict(float)
        self.histograms: Dict[str, List[float]] = defaultdict(list)
        self.timers: Dict[str, List[float]] = defaultdict(list)
        
        # Labels for metric organization
        self.labels: Dict[str, Dict[str, str]] = {}
        
        # Start cleanup task
        self._cleanup_task = None
        self._start_cleanup_task()
    
    def _start_cleanup_task(self):
        """Start background cleanup task."""
        if self._cleanup_task is None:
            self._cleanup_task = asyncio.create_task(self._cleanup_old_metrics())
    
    async def _cleanup_old_metrics(self):
        """Clean up old metrics data."""
        while True:
            try:
                cutoff_time = datetime.now() - timedelta(hours=self.retention_hours)
                
                for metric_name, points in self.metrics.items():
                    # Remove old points
                    while points and points[0].timestamp < cutoff_time:
                        points.popleft()
                
                await asyncio.sleep(300)  # Cleanup every 5 minutes
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"[METRICS] Cleanup error: {str(e)}")
                await asyncio.sleep(60)
    
    def increment_counter(self, name: str, value: float = 1.0, labels: Dict[str, str] = None):
        """Increment a counter metric."""
        full_name = self._get_metric_name(name, labels)
        self.counters[full_name] += value
        
        self._record_metric(name, self.counters[full_name], MetricType.COUNTER, labels)
    
    def set_gauge(self, name: str, value: float, labels: Dict[str, str] = None):
        """Set a gauge metric value."""
        full_name = self._get_metric_name(name, labels)
        self.gauges[full_name] = value
        
        self._record_metric(name, value, MetricType.GAUGE, labels)
    
    def record_histogram(self, name: str, value: float, labels: Dict[str, str] = None):
        """Record a value in a histogram."""
        full_name = self._get_metric_name(name, labels)
        self.histograms[full_name].append(value)
        
        # Keep histogram size manageable
        if len(self.histograms[full_name]) > 1000:
            self.histograms[full_name] = self.histograms[full_name][-1000:]
        
        self._record_metric(name, value, MetricType.HISTOGRAM, labels)
    
    def record_timer(self, name: str, duration: float, labels: Dict[str, str] = None):
        """Record a timing measurement."""
        full_name = self._get_metric_name(name, labels)
        self.timers[full_name].append(duration)
        
        # Keep timer size manageable
        if len(self.timers[full_name]) > 1000:
            self.timers[full_name] = self.timers[full_name][-1000:]
        
        self._record_metric(name, duration, MetricType.TIMER, labels)
    
    def _get_metric_name(self, name: str, labels: Dict[str, str] = None) -> str:
        """Generate full metric name with labels."""
        if not labels:
            return name
        
        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"
    
    def _record_metric(self, name: str, value: float, metric_type: MetricType, 
                      labels: Dict[str, str] = None):
        """Record a metric point."""
        point = MetricPoint(
            name=name,
            value=value,
            timestamp=datetime.now(),
            labels=labels or {},
            metric_type=metric_type
        )
        
        self.metrics[name].append(point)
        
        if labels:
            self.labels[name] = labels
    
    def get_metric_summary(self, name: str, minutes: int = 60) -> Dict[str, Any]:
        """Get summary statistics for a metric."""
        if name not in self.metrics:
            return {"error": "Metric not found"}
        
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        recent_points = [
            p for p in self.metrics[name] 
            if p.timestamp >= cutoff_time
        ]
        
        if not recent_points:
            return {"error": "No recent data"}
        
        values = [p.value for p in recent_points]
        
        return {
            "name": name,
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "avg": sum(values) / len(values),
            "latest": values[-1] if values else None,
            "period_minutes": minutes,
            "metric_type": recent_points[0].metric_type.value
        }
    
    def get_histogram_percentiles(self, name: str, percentiles: List[float] = None) -> Dict[str, float]:
        """Get percentile values for a histogram metric."""
        percentiles = percentiles or [50, 90, 95, 99]
        full_name = self._get_metric_name(name)
        
        if full_name not in self.histograms:
            return {}
        
        values = sorted(self.histograms[full_name])
        if not values:
            return {}
        
        result = {}
        for p in percentiles:
            index = int((p / 100.0) * len(values))
            index = min(index, len(values) - 1)
            result[f"p{int(p)}"] = values[index]
        
        return result
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all current metric values."""
        return {
            "counters": dict(self.counters),
            "gauges": dict(self.gauges),
            "histograms": {
                name: {
                    "count": len(values),
                    "latest": values[-1] if values else None,
                    "percentiles": self.get_histogram_percentiles(name.split("{")[0])
                }
                for name, values in self.histograms.items()
            },
            "timers": {
                name: {
                    "count": len(values),
                    "avg": sum(values) / len(values) if values else 0,
                    "percentiles": self.get_histogram_percentiles(name.split("{")[0])
                }
                for name, values in self.timers.items()
            }
        }


class PerformanceTracker:
    """
    Tracks system and application performance metrics.
    
    Monitors resource usage, agent performance, and system health.
    """
    
    def __init__(self, sample_interval: int = 30):
        """
        Initialize performance tracker.
        
        Args:
            sample_interval: Seconds between performance samples
        """
        self.sample_interval = sample_interval
        self.metrics = MetricsCollector()
        self.snapshots: deque = deque(maxlen=2880)  # 24 hours at 30s intervals
        
        # Performance tracking state
        self.start_time = datetime.now()
        self.agent_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "tasks_completed": 0,
            "total_time": 0.0,
            "error_count": 0,
            "last_active": None
        })
        
        # Start monitoring
        self._monitoring_task = None
        self._start_monitoring()
    
    def _start_monitoring(self):
        """Start background monitoring task."""
        if self._monitoring_task is None:
            self._monitoring_task = asyncio.create_task(self._collect_system_metrics())
    
    async def _collect_system_metrics(self):
        """Collect system performance metrics."""
        while True:
            try:
                # System metrics
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                
                # Application metrics
                active_agents = len([
                    agent_id for agent_id, stats in self.agent_stats.items()
                    if stats["last_active"] and 
                    (datetime.now() - stats["last_active"]).seconds < 300
                ])
                
                completed_tasks = sum(
                    stats["tasks_completed"] for stats in self.agent_stats.values()
                )
                
                total_errors = sum(
                    stats["error_count"] for stats in self.agent_stats.values()
                )
                
                error_rate = total_errors / max(completed_tasks, 1)
                
                # Average response time
                avg_response_time = sum(
                    stats["total_time"] / max(stats["tasks_completed"], 1)
                    for stats in self.agent_stats.values()
                ) / max(len(self.agent_stats), 1)
                
                # Create snapshot
                snapshot = PerformanceSnapshot(
                    timestamp=datetime.now(),
                    cpu_percent=cpu_percent,
                    memory_mb=memory.used / 1024 / 1024,
                    memory_percent=memory.percent,
                    active_agents=active_agents,
                    completed_tasks=completed_tasks,
                    error_rate=error_rate,
                    avg_response_time=avg_response_time
                )
                
                self.snapshots.append(snapshot)
                
                # Record metrics
                self.metrics.set_gauge("system.cpu_percent", cpu_percent)
                self.metrics.set_gauge("system.memory_mb", snapshot.memory_mb)
                self.metrics.set_gauge("system.memory_percent", memory.percent)
                self.metrics.set_gauge("agents.active_count", active_agents)
                self.metrics.set_gauge("tasks.completed_total", completed_tasks)
                self.metrics.set_gauge("tasks.error_rate", error_rate)
                self.metrics.set_gauge("performance.avg_response_time", avg_response_time)
                
                await asyncio.sleep(self.sample_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"[PERFORMANCE] Monitoring error: {str(e)}")
                await asyncio.sleep(self.sample_interval)
    
    def record_agent_task(self, agent_id: str, duration: float, success: bool = True):
        """Record agent task completion."""
        stats = self.agent_stats[agent_id]
        stats["tasks_completed"] += 1
        stats["total_time"] += duration
        stats["last_active"] = datetime.now()
        
        if not success:
            stats["error_count"] += 1
        
        # Record metrics
        self.metrics.increment_counter("agent.tasks_completed", labels={"agent_id": agent_id})
        self.metrics.record_timer("agent.task_duration", duration, labels={"agent_id": agent_id})
        
        if not success:
            self.metrics.increment_counter("agent.errors", labels={"agent_id": agent_id})
    
    def record_workflow_execution(self, workflow_type: str, duration: float, 
                                agent_count: int, success: bool = True):
        """Record workflow execution metrics."""
        self.metrics.record_timer("workflow.duration", duration, 
                                labels={"workflow_type": workflow_type})
        self.metrics.set_gauge("workflow.agent_count", agent_count,
                             labels={"workflow_type": workflow_type})
        
        if success:
            self.metrics.increment_counter("workflow.completed", 
                                         labels={"workflow_type": workflow_type})
        else:
            self.metrics.increment_counter("workflow.failed",
                                         labels={"workflow_type": workflow_type})
    
    def get_performance_summary(self, hours: int = 1) -> Dict[str, Any]:
        """Get performance summary for specified time period."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_snapshots = [
            s for s in self.snapshots 
            if s.timestamp >= cutoff_time
        ]
        
        if not recent_snapshots:
            return {"error": "No recent performance data"}
        
        # Calculate averages
        avg_cpu = sum(s.cpu_percent for s in recent_snapshots) / len(recent_snapshots)
        avg_memory = sum(s.memory_mb for s in recent_snapshots) / len(recent_snapshots)
        avg_agents = sum(s.active_agents for s in recent_snapshots) / len(recent_snapshots)
        avg_response_time = sum(s.avg_response_time for s in recent_snapshots) / len(recent_snapshots)
        
        # Find peaks
        peak_cpu = max(s.cpu_percent for s in recent_snapshots)
        peak_memory = max(s.memory_mb for s in recent_snapshots)
        
        return {
            "period_hours": hours,
            "samples": len(recent_snapshots),
            "averages": {
                "cpu_percent": round(avg_cpu, 2),
                "memory_mb": round(avg_memory, 2),
                "active_agents": round(avg_agents, 1),
                "response_time": round(avg_response_time, 3)
            },
            "peaks": {
                "cpu_percent": round(peak_cpu, 2),
                "memory_mb": round(peak_memory, 2)
            },
            "current": {
                "cpu_percent": recent_snapshots[-1].cpu_percent,
                "memory_mb": round(recent_snapshots[-1].memory_mb, 2),
                "active_agents": recent_snapshots[-1].active_agents,
                "uptime_hours": (datetime.now() - self.start_time).total_seconds() / 3600
            }
        }
    
    def get_agent_performance(self, agent_id: str = None) -> Dict[str, Any]:
        """Get performance metrics for specific agent or all agents."""
        if agent_id:
            if agent_id not in self.agent_stats:
                return {"error": "Agent not found"}
            
            stats = self.agent_stats[agent_id]
            return {
                "agent_id": agent_id,
                "tasks_completed": stats["tasks_completed"],
                "total_time": round(stats["total_time"], 3),
                "avg_task_time": round(
                    stats["total_time"] / max(stats["tasks_completed"], 1), 3
                ),
                "error_count": stats["error_count"],
                "error_rate": round(
                    stats["error_count"] / max(stats["tasks_completed"], 1), 3
                ),
                "last_active": stats["last_active"].isoformat() if stats["last_active"] else None
            }
        else:
            return {
                agent_id: self.get_agent_performance(agent_id)
                for agent_id in self.agent_stats.keys()
            }
    
    def export_performance_data(self, format: str = "json") -> str:
        """Export performance data."""
        if format == "json":
            return json.dumps({
                "summary": self.get_performance_summary(24),
                "agent_performance": self.get_agent_performance(),
                "metrics": self.metrics.get_all_metrics(),
                "exported_at": datetime.now().isoformat()
            }, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def cleanup(self):
        """Clean up monitoring resources."""
        if self._monitoring_task:
            self._monitoring_task.cancel()
        if self.metrics._cleanup_task:
            self.metrics._cleanup_task.cancel()


# Context manager for timing operations
class TimingContext:
    """Context manager for timing operations."""
    
    def __init__(self, metrics: MetricsCollector, name: str, labels: Dict[str, str] = None):
        self.metrics = metrics
        self.name = name
        self.labels = labels
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration = time.time() - self.start_time
            self.metrics.record_timer(self.name, duration, self.labels)