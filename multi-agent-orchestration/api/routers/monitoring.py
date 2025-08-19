"""
Monitoring API Router

FastAPI router for monitoring, metrics, and observability endpoints.
Provides platform health, performance metrics, and analytics.
"""

import sys
import os
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'src'))

from fastapi import APIRouter, HTTPException, Depends, Query, status
from fastapi.responses import JSONResponse

from src.multi_agent_platform import MultiAgentPlatform
from ..models.request_models import MonitoringQuery
from ..models.response_models import MonitoringResponse


router = APIRouter()


# Global platform instance (will be set by main.py)
_platform_instance = None

def set_platform_instance(platform: MultiAgentPlatform):
    """Set the global platform instance."""
    global _platform_instance
    _platform_instance = platform

async def get_platform() -> MultiAgentPlatform:
    """Get platform instance."""
    if _platform_instance is None:
        raise HTTPException(status_code=500, detail="Platform not initialized")
    return _platform_instance


@router.get(
    "/metrics",
    response_model=MonitoringResponse,
    summary="Get Platform Metrics",
    description="Get comprehensive platform metrics and performance data"
)
async def get_metrics(
    time_range: str = Query("1h", description="Time range for metrics (1h, 1d, 1w)"),
    pattern_type: Optional[str] = Query(None, description="Filter by pattern type"),
    agent_id: Optional[str] = Query(None, description="Filter by agent ID"),
    include_details: bool = Query(False, description="Include detailed metrics"),
    platform: MultiAgentPlatform = Depends(get_platform)
):
    """Get platform metrics and analytics."""
    try:
        # Get platform status and performance data
        platform_status = platform.get_platform_status()
        performance_analytics = platform.get_performance_analytics()
        
        # Parse time range
        time_multipliers = {"1h": 1, "1d": 24, "1w": 168}
        hours = time_multipliers.get(time_range, 1)
        
        # Build platform metrics
        platform_metrics = {
            "total_agents": len(platform.registered_agents),
            "total_patterns": len(platform.active_patterns),
            "total_executions": platform_status["platform_metrics"]["workflows_executed"],
            "uptime_seconds": platform_status["platform_info"]["uptime_seconds"],
            "success_rate": performance_analytics["performance_summary"]["overall_success_rate"],
            "average_response_time": performance_analytics["performance_summary"]["average_execution_time"],
            "active_workflows": performance_analytics["performance_summary"]["active_workflows"]
        }
        
        # Build pattern-specific metrics
        pattern_metrics = {}
        for pattern_id, pattern in platform.active_patterns.items():
            pattern_type_name = type(pattern).__name__.replace('Pattern', '').lower()
            
            if pattern_type and pattern_type_name != pattern_type:
                continue
            
            if pattern_type_name not in pattern_metrics:
                pattern_metrics[pattern_type_name] = {
                    "count": 0,
                    "executions": 0,
                    "avg_execution_time": 0.0,
                    "success_rate": 0.0
                }
            
            pattern_metrics[pattern_type_name]["count"] += 1
            # In a real implementation, these would come from execution history
            pattern_metrics[pattern_type_name]["executions"] += 10  # Mock data
            pattern_metrics[pattern_type_name]["avg_execution_time"] = 2.5  # Mock data
            pattern_metrics[pattern_type_name]["success_rate"] = 0.94  # Mock data
        
        # Build agent-specific metrics
        agent_metrics = {}
        agents_data = platform.list_agents()
        
        for agent_id_key, agent_info in agents_data.items():
            if agent_id and agent_id_key != agent_id:
                continue
            
            agent_metrics[agent_id_key] = {
                "agent_type": agent_info["type"],
                "capabilities_count": len(agent_info["capabilities"]),
                "performance": agent_info["performance"]
            }
        
        # Build performance trends (mock data for now)
        performance_trends = {
            "execution_count_trend": {
                "timestamps": _generate_timestamps(hours),
                "values": _generate_trend_data(hours, base_value=50, variation=10)
            },
            "success_rate_trend": {
                "timestamps": _generate_timestamps(hours),
                "values": _generate_trend_data(hours, base_value=0.95, variation=0.05)
            },
            "response_time_trend": {
                "timestamps": _generate_timestamps(hours),
                "values": _generate_trend_data(hours, base_value=2.3, variation=0.8)
            }
        }
        
        # Check for alerts (basic implementation)
        alerts = []
        if platform_metrics["success_rate"] < 0.9:
            alerts.append({
                "type": "warning",
                "message": f"Success rate below threshold: {platform_metrics['success_rate']:.1%}",
                "timestamp": datetime.now().isoformat(),
                "severity": "medium"
            })
        
        if platform_metrics["average_response_time"] > 10:
            alerts.append({
                "type": "warning", 
                "message": f"High response time: {platform_metrics['average_response_time']:.1f}s",
                "timestamp": datetime.now().isoformat(),
                "severity": "medium"
            })
        
        return MonitoringResponse(
            timestamp=datetime.now(),
            time_range=time_range,
            platform_metrics=platform_metrics,
            pattern_metrics=pattern_metrics,
            agent_metrics=agent_metrics,
            performance_trends=performance_trends,
            alerts=alerts
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get metrics: {str(e)}"
        )


@router.get(
    "/health",
    summary="Detailed Health Check",
    description="Get detailed health information for all platform components"
)
async def detailed_health_check(
    platform: MultiAgentPlatform = Depends(get_platform)
):
    """Get detailed health check for all components."""
    try:
        platform_status = platform.get_platform_status()
        
        health_data = {
            "overall_status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "components": {
                "platform": {
                    "status": "healthy",
                    "uptime_seconds": platform_status["platform_info"]["uptime_seconds"],
                    "version": platform_status["platform_info"]["version"]
                },
                "workflow_engine": {
                    "status": "healthy",
                    "metrics": platform_status["component_status"]["workflow_engine"]
                },
                "result_aggregator": {
                    "status": "healthy", 
                    "metrics": platform_status["component_status"]["result_aggregator"]
                },
                "state_manager": {
                    "status": "healthy",
                    "metrics": platform_status["component_status"]["state_manager"]
                },
                "feedback_loop": {
                    "status": "healthy",
                    "metrics": platform_status["component_status"]["feedback_loop"]
                },
                "agents": {
                    "status": "healthy",
                    "registered_count": platform_status["registry_status"]["registered_agents"],
                    "active_count": platform_status["registry_status"]["registered_agents"]
                },
                "patterns": {
                    "status": "healthy",
                    "active_count": platform_status["registry_status"]["active_patterns"]
                }
            },
            "resource_usage": {
                "memory_usage": "normal",  # Would be actual memory monitoring
                "cpu_usage": "normal",     # Would be actual CPU monitoring
                "disk_usage": "normal"     # Would be actual disk monitoring
            }
        }
        
        return health_data
        
    except Exception as e:
        return {
            "overall_status": "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "error": str(e),
            "components": {
                "platform": {"status": "error", "error": str(e)}
            }
        }


@router.get(
    "/performance", 
    summary="Performance Analytics",
    description="Get detailed performance analytics and bottleneck analysis"
)
async def get_performance_analytics(
    time_range: str = Query("1h", description="Time range for analysis"),
    include_recommendations: bool = Query(True, description="Include optimization recommendations"),
    platform: MultiAgentPlatform = Depends(get_platform)
):
    """Get performance analytics and optimization recommendations."""
    try:
        performance_data = platform.get_performance_analytics()
        
        # Enhanced performance analysis
        analysis = {
            "timestamp": datetime.now().isoformat(),
            "time_range": time_range,
            "summary": performance_data["performance_summary"],
            "detailed_metrics": {
                "execution_distribution": {
                    "fast_executions": {"count": 45, "percentage": 75.0, "threshold": "< 2s"},
                    "medium_executions": {"count": 12, "percentage": 20.0, "threshold": "2-5s"}, 
                    "slow_executions": {"count": 3, "percentage": 5.0, "threshold": "> 5s"}
                },
                "pattern_performance": performance_data["pattern_performance"],
                "agent_efficiency": performance_data["agent_performance"],
                "bottlenecks": {
                    "slowest_pattern": "reflective",
                    "highest_failure_rate": "parallel",
                    "resource_intensive": "supervisor"
                }
            }
        }
        
        if include_recommendations:
            analysis["recommendations"] = _generate_performance_recommendations(performance_data)
        
        return analysis
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get performance analytics: {str(e)}"
        )


@router.get(
    "/alerts",
    summary="Active Alerts",
    description="Get list of active alerts and system warnings"
)
async def get_alerts(
    severity: Optional[str] = Query(None, description="Filter by severity level"),
    alert_type: Optional[str] = Query(None, description="Filter by alert type"),
    platform: MultiAgentPlatform = Depends(get_platform)
):
    """Get active alerts and warnings."""
    try:
        performance_data = platform.get_performance_analytics()
        
        # Generate alerts based on metrics
        alerts = []
        
        # Performance alerts
        if performance_data["performance_summary"]["overall_success_rate"] < 0.9:
            alerts.append({
                "id": "perf_001",
                "type": "performance",
                "severity": "medium",
                "title": "Low Success Rate",
                "message": f"Overall success rate is {performance_data['performance_summary']['overall_success_rate']:.1%}, below 90% threshold",
                "timestamp": datetime.now().isoformat(),
                "source": "performance_monitor"
            })
        
        if performance_data["performance_summary"]["average_execution_time"] > 5.0:
            alerts.append({
                "id": "perf_002",
                "type": "performance",
                "severity": "high",
                "title": "High Response Time",
                "message": f"Average execution time is {performance_data['performance_summary']['average_execution_time']:.1f}s, exceeds 5s threshold",
                "timestamp": datetime.now().isoformat(),
                "source": "performance_monitor"
            })
        
        # Resource alerts (mock data)
        alerts.append({
            "id": "res_001",
            "type": "resource",
            "severity": "low",
            "title": "Memory Usage",
            "message": "Memory usage is at 75% of available capacity",
            "timestamp": datetime.now().isoformat(),
            "source": "resource_monitor"
        })
        
        # Apply filters
        if severity:
            alerts = [alert for alert in alerts if alert["severity"] == severity]
        
        if alert_type:
            alerts = [alert for alert in alerts if alert["type"] == alert_type]
        
        return {
            "alerts": alerts,
            "total_count": len(alerts),
            "severity_breakdown": {
                "high": len([a for a in alerts if a["severity"] == "high"]),
                "medium": len([a for a in alerts if a["severity"] == "medium"]),
                "low": len([a for a in alerts if a["severity"] == "low"])
            }
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get alerts: {str(e)}"
        )


@router.get(
    "/logs",
    summary="System Logs",
    description="Get recent system logs and events"
)
async def get_system_logs(
    level: Optional[str] = Query(None, description="Filter by log level"),
    component: Optional[str] = Query(None, description="Filter by component"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of log entries")
):
    """Get system logs and events."""
    try:
        # Mock log data (in production, integrate with actual logging system)
        logs = [
            {
                "timestamp": (datetime.now() - timedelta(minutes=5)).isoformat(),
                "level": "INFO",
                "component": "workflow_engine",
                "message": "Pipeline pattern executed successfully",
                "execution_id": "exec_001",
                "metadata": {"duration": 2.3, "pattern": "pipeline"}
            },
            {
                "timestamp": (datetime.now() - timedelta(minutes=10)).isoformat(),
                "level": "WARNING",
                "component": "parallel_pattern",
                "message": "Agent timeout during parallel execution",
                "execution_id": "exec_002",
                "metadata": {"timeout": 30, "agent": "analyst_001"}
            },
            {
                "timestamp": (datetime.now() - timedelta(minutes=15)).isoformat(),
                "level": "ERROR",
                "component": "supervisor_pattern",
                "message": "Supervisor agent failed to delegate task",
                "execution_id": "exec_003",
                "metadata": {"error": "AgentNotAvailable", "task": "research"}
            },
            {
                "timestamp": (datetime.now() - timedelta(minutes=20)).isoformat(),
                "level": "INFO",
                "component": "feedback_loop",
                "message": "Performance feedback processed",
                "metadata": {"feedback_count": 5, "improvements": 2}
            }
        ]
        
        # Apply filters
        if level:
            logs = [log for log in logs if log["level"] == level.upper()]
        
        if component:
            logs = [log for log in logs if log["component"] == component]
        
        # Apply limit
        logs = logs[:limit]
        
        return {
            "logs": logs,
            "total_count": len(logs),
            "level_breakdown": {
                "ERROR": len([l for l in logs if l["level"] == "ERROR"]),
                "WARNING": len([l for l in logs if l["level"] == "WARNING"]),
                "INFO": len([l for l in logs if l["level"] == "INFO"]),
                "DEBUG": len([l for l in logs if l["level"] == "DEBUG"])
            }
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get logs: {str(e)}"
        )


# Helper functions

def _generate_timestamps(hours: int, interval_minutes: int = 5) -> List[str]:
    """Generate timestamps for trend data."""
    timestamps = []
    current = datetime.now() - timedelta(hours=hours)
    end_time = datetime.now()
    
    while current <= end_time:
        timestamps.append(current.isoformat())
        current += timedelta(minutes=interval_minutes)
    
    return timestamps


def _generate_trend_data(hours: int, base_value: float, variation: float) -> List[float]:
    """Generate mock trend data."""
    import random
    data_points = (hours * 60) // 5  # 5-minute intervals
    
    return [
        max(0, base_value + random.uniform(-variation, variation))
        for _ in range(data_points)
    ]


def _generate_performance_recommendations(performance_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Generate performance optimization recommendations."""
    recommendations = []
    
    success_rate = performance_data["performance_summary"]["overall_success_rate"]
    if success_rate < 0.9:
        recommendations.append({
            "category": "reliability",
            "priority": "high",
            "title": "Improve Success Rate",
            "description": f"Success rate is {success_rate:.1%}. Consider adding retry mechanisms and better error handling.",
            "actions": [
                "Add exponential backoff retry logic",
                "Implement circuit breaker pattern",
                "Review and fix common failure points"
            ]
        })
    
    avg_time = performance_data["performance_summary"]["average_execution_time"]
    if avg_time > 3.0:
        recommendations.append({
            "category": "performance",
            "priority": "medium", 
            "title": "Optimize Execution Time",
            "description": f"Average execution time is {avg_time:.1f}s. Consider optimization strategies.",
            "actions": [
                "Profile slow-running patterns",
                "Implement parallel processing where possible",
                "Add caching for repeated operations",
                "Optimize agent task delegation"
            ]
        })
    
    return recommendations