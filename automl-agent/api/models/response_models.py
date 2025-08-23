from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List, Union
from datetime import datetime
from enum import Enum
from .request_models import JobStatusEnum

class AgentResult(BaseModel):
    agent_type: str
    task_description: str
    status: str
    performance_metrics: Optional[Dict[str, float]] = None
    artifacts: Optional[Dict[str, Any]] = None
    quality_score: Optional[float] = None
    execution_time: Optional[float] = None
    recommendations: Optional[List[str]] = None
    error_message: Optional[str] = None

class JobResult(BaseModel):
    job_id: str
    agent_results: List[AgentResult]
    overall_quality: Optional[float] = None
    workflow_summary: Optional[str] = None
    total_execution_time: Optional[float] = None
    refinement_iterations: int = 0

class JobStatus(BaseModel):
    job_id: str
    status: JobStatusEnum
    progress: float = Field(ge=0.0, le=1.0)
    result: Optional[Union[AgentResult, JobResult]] = None
    error: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    estimated_completion: Optional[datetime] = None

class DataUploadResponse(BaseModel):
    data_id: str
    filename: str
    shape: tuple
    columns: List[str]
    dtypes: Dict[str, str]
    sample: List[Dict[str, Any]]
    upload_timestamp: datetime

class AgentInfo(BaseModel):
    name: str
    description: str
    capabilities: List[str]
    supported_task_types: List[str]
    performance_metrics: List[str]
    quality_indicators: List[str]

class APIResponse(BaseModel):
    success: bool
    message: str
    data: Optional[Any] = None
    timestamp: datetime

class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
    error_code: Optional[str] = None
    timestamp: datetime

class HealthCheckResponse(BaseModel):
    status: str
    timestamp: datetime
    version: str
    active_jobs: int
    system_metrics: Optional[Dict[str, Any]] = None

class CollaborationMessage(BaseModel):
    sender_agent: str
    receiver_agent: Optional[str] = None  # None for broadcast
    message_type: str  # "quality_feedback", "data_request", "refinement_suggestion"
    content: Dict[str, Any]
    timestamp: datetime
    priority: int = Field(default=1, ge=1, le=5)

class QualityFeedback(BaseModel):
    evaluating_agent: str
    target_agent: str
    quality_score: float = Field(ge=0.0, le=1.0)
    feedback_type: str  # "improvement_needed", "quality_met", "refinement_suggestion"
    specific_feedback: List[str]
    suggested_improvements: Optional[List[str]] = None
    performance_metrics: Dict[str, float]