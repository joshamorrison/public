"""
API Response Models

Pydantic models for API response validation and documentation.
"""

from datetime import datetime
from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel, Field


class HealthCheckResponse(BaseModel):
    """Health check response model."""
    status: str = Field(..., description="Overall health status")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")
    platform_id: str = Field(..., description="Platform instance ID")
    version: str = Field(..., description="Platform version")
    uptime_seconds: float = Field(..., description="Platform uptime in seconds")
    components: Dict[str, Any] = Field(..., description="Component health status")
    
    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "timestamp": "2024-01-01T12:00:00Z",
                "platform_id": "api_platform_001",
                "version": "1.0.0", 
                "uptime_seconds": 3600.5,
                "components": {
                    "platform": "healthy",
                    "workflow_engine": "healthy",
                    "agents": "healthy",
                    "patterns": "healthy"
                }
            }
        }


class PlatformStatusResponse(BaseModel):
    """Comprehensive platform status response."""
    platform_info: Dict[str, Any] = Field(..., description="Platform information")
    component_status: Dict[str, Any] = Field(..., description="Component status details")
    registry_status: Dict[str, Any] = Field(..., description="Agent and pattern registry status")
    platform_metrics: Dict[str, Any] = Field(..., description="Platform metrics")
    performance_summary: Dict[str, Any] = Field(..., description="Performance summary")
    agent_count: int = Field(..., description="Total registered agents")
    pattern_count: int = Field(..., description="Total active patterns")
    
    class Config:
        schema_extra = {
            "example": {
                "platform_info": {
                    "platform_id": "api_platform_001",
                    "version": "1.0.0",
                    "uptime_seconds": 7200
                },
                "agent_count": 12,
                "pattern_count": 4,
                "performance_summary": {
                    "overall_success_rate": 0.95,
                    "average_execution_time": 2.3
                }
            }
        }


class AgentResponse(BaseModel):
    """Agent information response."""
    agent_id: str = Field(..., description="Agent unique identifier")
    name: str = Field(..., description="Agent name")
    agent_type: str = Field(..., description="Agent type")
    capabilities: List[str] = Field(..., description="Agent capabilities")
    performance_metrics: Dict[str, Any] = Field(..., description="Agent performance metrics")
    status: str = Field(..., description="Agent status")
    created_at: datetime = Field(..., description="Agent creation timestamp")
    
    class Config:
        schema_extra = {
            "example": {
                "agent_id": "researcher_001",
                "name": "Market Research Specialist",
                "agent_type": "researcher", 
                "capabilities": ["web_search", "data_analysis", "report_generation"],
                "performance_metrics": {
                    "tasks_completed": 45,
                    "average_confidence": 0.87,
                    "success_rate": 0.93
                },
                "status": "active",
                "created_at": "2024-01-01T10:00:00Z"
            }
        }


class PatternResponse(BaseModel):
    """Pattern information response."""
    pattern_id: str = Field(..., description="Pattern unique identifier") 
    name: str = Field(..., description="Pattern name")
    pattern_type: str = Field(..., description="Pattern type")
    configuration: Dict[str, Any] = Field(..., description="Pattern configuration")
    agents_involved: List[str] = Field(..., description="Agent IDs involved in this pattern")
    status: str = Field(..., description="Pattern status")
    created_at: datetime = Field(..., description="Pattern creation timestamp")
    execution_count: int = Field(0, description="Number of times pattern has been executed")
    
    class Config:
        schema_extra = {
            "example": {
                "pattern_id": "research_pipeline_001",
                "name": "Comprehensive Research Pipeline",
                "pattern_type": "pipeline",
                "configuration": {
                    "stages": 4,
                    "quality_gates": True
                },
                "agents_involved": ["researcher_001", "analyst_001", "synthesizer_001", "critic_001"],
                "status": "active",
                "created_at": "2024-01-01T09:00:00Z",
                "execution_count": 15
            }
        }


class ExecutionResultResponse(BaseModel):
    """Workflow execution result response."""
    execution_id: str = Field(..., description="Execution unique identifier")
    pattern_id: str = Field(..., description="Pattern used for execution")
    task_id: Optional[str] = Field(None, description="Task identifier")
    status: str = Field(..., description="Execution status")
    success: bool = Field(..., description="Execution success flag")
    started_at: datetime = Field(..., description="Execution start timestamp")
    completed_at: Optional[datetime] = Field(None, description="Execution completion timestamp")
    execution_time: float = Field(..., description="Total execution time in seconds")
    result: Optional[Dict[str, Any]] = Field(None, description="Execution result data")
    error: Optional[str] = Field(None, description="Error message if execution failed")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Execution metadata")
    
    class Config:
        schema_extra = {
            "example": {
                "execution_id": "exec_20240101_120000_001",
                "pattern_id": "research_pipeline_001",
                "task_id": "research_task_001",
                "status": "completed",
                "success": True,
                "started_at": "2024-01-01T12:00:00Z",
                "completed_at": "2024-01-01T12:05:30Z", 
                "execution_time": 330.5,
                "result": {
                    "final_result": "Comprehensive market analysis completed...",
                    "confidence": 0.89,
                    "stages_completed": 4
                },
                "metadata": {
                    "agents_used": ["researcher_001", "analyst_001"],
                    "pattern_type": "pipeline"
                }
            }
        }


class WorkflowResponse(BaseModel):
    """Workflow information and status response."""
    workflow_id: str = Field(..., description="Workflow unique identifier")
    pattern_id: str = Field(..., description="Associated pattern ID")
    name: str = Field(..., description="Workflow name")
    description: Optional[str] = Field(None, description="Workflow description")
    status: str = Field(..., description="Workflow status")
    executions: List[ExecutionResultResponse] = Field(default_factory=list, description="Recent executions")
    total_executions: int = Field(0, description="Total number of executions")
    success_rate: float = Field(0.0, description="Execution success rate")
    average_execution_time: float = Field(0.0, description="Average execution time")
    
    class Config:
        schema_extra = {
            "example": {
                "workflow_id": "workflow_001",
                "pattern_id": "research_pipeline_001", 
                "name": "Market Research Workflow",
                "status": "active",
                "total_executions": 25,
                "success_rate": 0.92,
                "average_execution_time": 285.7
            }
        }


class MonitoringResponse(BaseModel):
    """Monitoring data response."""
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")
    time_range: str = Field(..., description="Time range for metrics")
    platform_metrics: Dict[str, Any] = Field(..., description="Platform-wide metrics")
    pattern_metrics: Dict[str, Any] = Field(..., description="Pattern-specific metrics") 
    agent_metrics: Dict[str, Any] = Field(..., description="Agent-specific metrics")
    performance_trends: Dict[str, Any] = Field(..., description="Performance trend data")
    alerts: List[Dict[str, Any]] = Field(default_factory=list, description="Active alerts")
    
    class Config:
        schema_extra = {
            "example": {
                "timestamp": "2024-01-01T12:00:00Z",
                "time_range": "1h",
                "platform_metrics": {
                    "total_executions": 150,
                    "success_rate": 0.94,
                    "average_response_time": 2.8
                },
                "pattern_metrics": {
                    "pipeline": {"executions": 65, "success_rate": 0.96},
                    "parallel": {"executions": 45, "success_rate": 0.92}
                },
                "alerts": []
            }
        }


class ErrorResponse(BaseModel):
    """Standard error response model."""
    error: str = Field(..., description="Error type")
    detail: str = Field(..., description="Error details")
    timestamp: datetime = Field(default_factory=datetime.now, description="Error timestamp")
    request_id: Optional[str] = Field(None, description="Request identifier for tracing")
    
    class Config:
        schema_extra = {
            "example": {
                "error": "ValidationError",
                "detail": "Invalid pattern_type provided",
                "timestamp": "2024-01-01T12:00:00Z",
                "request_id": "req_20240101_120000_001"
            }
        }


class BulkOperationResponse(BaseModel):
    """Response for bulk operations."""
    total_requests: int = Field(..., description="Total number of requests processed")
    successful: int = Field(..., description="Number of successful operations")
    failed: int = Field(..., description="Number of failed operations")
    results: List[Dict[str, Any]] = Field(..., description="Individual operation results")
    execution_time: float = Field(..., description="Total execution time")
    
    class Config:
        schema_extra = {
            "example": {
                "total_requests": 3,
                "successful": 2, 
                "failed": 1,
                "results": [
                    {"id": "req_1", "status": "success", "result": "..."},
                    {"id": "req_2", "status": "success", "result": "..."},
                    {"id": "req_3", "status": "failed", "error": "..."}
                ],
                "execution_time": 15.7
            }
        }