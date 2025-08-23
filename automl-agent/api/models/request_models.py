from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List, Union
from datetime import datetime
from enum import Enum

class JobStatusEnum(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class AgentTypeEnum(str, Enum):
    EDA = "eda"
    DATA_HYGIENE = "data_hygiene"
    FEATURE_ENGINEERING = "feature_engineering"
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    NLP = "nlp"
    COMPUTER_VISION = "computer_vision"
    TIME_SERIES = "time_series"
    HYPERPARAMETER_TUNING = "hyperparameter_tuning"
    ENSEMBLE = "ensemble"

class CollaborationModeEnum(str, Enum):
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    ADAPTIVE = "adaptive"

class TaskRequest(BaseModel):
    task_description: str = Field(..., description="Natural language description of the ML task")
    target_column: Optional[str] = Field(None, description="Target column for supervised learning")
    task_type: Optional[str] = Field(None, description="Override automatic task detection")
    quality_threshold: float = Field(0.8, ge=0.0, le=1.0, description="Minimum quality threshold for completion")
    max_iterations: int = Field(5, ge=1, le=20, description="Maximum refinement iterations")

class AgentExecutionRequest(BaseModel):
    task_description: str = Field(..., description="Task description")
    parameters: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Agent-specific parameters")
    data_source: Optional[str] = Field(None, description="Data source identifier or path")

class WorkflowRequest(BaseModel):
    task_description: str = Field(..., description="Overall workflow description")
    agents: List[AgentTypeEnum] = Field(..., description="List of agent types to include in workflow")
    quality_threshold: float = Field(0.8, ge=0.0, le=1.0, description="Quality threshold for the entire workflow")
    collaboration_mode: CollaborationModeEnum = Field(CollaborationModeEnum.SEQUENTIAL, description="How agents should collaborate")
    max_iterations: int = Field(3, ge=1, le=10, description="Maximum refinement iterations for the workflow")

class RefinementRequest(BaseModel):
    requesting_agent: str
    target_agent: str
    refinement_type: str  # "feature_engineering", "hyperparameter_tuning", "data_cleaning"
    current_quality: float
    target_quality: float
    specific_requests: List[str]
    context: Optional[Dict[str, Any]] = None