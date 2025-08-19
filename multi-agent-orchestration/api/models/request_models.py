"""
API Request Models

Pydantic models for API request validation and documentation.
"""

from datetime import datetime
from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel, Field, validator
from enum import Enum


class PatternType(str, Enum):
    """Supported orchestration patterns."""
    PIPELINE = "pipeline"
    SUPERVISOR = "supervisor" 
    PARALLEL = "parallel"
    REFLECTIVE = "reflective"


class AgentType(str, Enum):
    """Supported agent types."""
    BASE = "base"
    SUPERVISOR = "supervisor"
    RESEARCHER = "researcher"
    ANALYST = "analyst"
    CRITIC = "critic"
    SYNTHESIZER = "synthesizer"


class Priority(str, Enum):
    """Task priority levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


class CreateAgentRequest(BaseModel):
    """Request to create a new agent."""
    agent_type: AgentType = Field(..., description="Type of agent to create")
    agent_id: Optional[str] = Field(None, description="Custom agent ID (auto-generated if not provided)")
    name: Optional[str] = Field(None, description="Human-readable agent name")
    description: Optional[str] = Field(None, description="Agent description and purpose")
    capabilities: Optional[List[str]] = Field(default_factory=list, description="List of agent capabilities")
    configuration: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Agent-specific configuration")
    
    class Config:
        schema_extra = {
            "example": {
                "agent_type": "researcher", 
                "name": "Market Research Specialist",
                "description": "Specialized agent for market research and competitive analysis",
                "capabilities": ["web_search", "data_analysis", "report_generation"],
                "configuration": {
                    "max_search_results": 10,
                    "research_depth": "comprehensive"
                }
            }
        }


class PatternConfigurationRequest(BaseModel):
    """Request to configure an orchestration pattern."""
    pattern_type: PatternType = Field(..., description="Type of pattern to create")
    pattern_id: Optional[str] = Field(None, description="Custom pattern ID")
    name: Optional[str] = Field(None, description="Pattern name")
    description: Optional[str] = Field(None, description="Pattern description")
    
    # Pattern-specific configurations
    agents: Optional[List[str]] = Field(default_factory=list, description="List of agent IDs to use in this pattern")
    configuration: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Pattern-specific configuration")
    
    # Pipeline-specific
    pipeline_stages: Optional[List[Dict[str, Any]]] = Field(None, description="Pipeline stage configuration")
    
    # Supervisor-specific  
    supervisor_agent_id: Optional[str] = Field(None, description="ID of supervisor agent")
    specialist_agents: Optional[Dict[str, str]] = Field(None, description="Mapping of specializations to agent IDs")
    
    # Parallel-specific
    parallel_agents: Optional[List[Dict[str, Any]]] = Field(None, description="Parallel agent configuration")
    fusion_strategy: Optional[str] = Field("comprehensive_synthesis", description="Result fusion strategy")
    max_concurrent: Optional[int] = Field(3, description="Maximum concurrent agents")
    
    # Reflective-specific
    primary_agent_id: Optional[str] = Field(None, description="Primary agent for reflective pattern")
    critic_agent_ids: Optional[List[str]] = Field(None, description="Critic agent IDs for feedback")
    max_iterations: Optional[int] = Field(3, description="Maximum reflection iterations") 
    convergence_threshold: Optional[float] = Field(0.85, description="Confidence threshold for convergence")
    
    class Config:
        schema_extra = {
            "example": {
                "pattern_type": "pipeline",
                "name": "Content Creation Pipeline", 
                "description": "Sequential workflow for creating high-quality content",
                "pipeline_stages": [
                    {"stage_name": "research", "agent_type": "researcher"},
                    {"stage_name": "analysis", "agent_type": "analyst"},
                    {"stage_name": "synthesis", "agent_type": "synthesizer"},
                    {"stage_name": "review", "agent_type": "critic"}
                ],
                "configuration": {
                    "quality_gates": True,
                    "rollback_on_failure": True
                }
            }
        }


class TaskRequirements(BaseModel):
    """Task requirements and constraints."""
    priority: Priority = Field(Priority.MEDIUM, description="Task priority level")
    deadline: Optional[datetime] = Field(None, description="Task deadline")
    quality_level: Optional[str] = Field("standard", description="Required quality level")
    resources: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Resource constraints")
    constraints: Optional[List[str]] = Field(default_factory=list, description="Task constraints")


class ExecuteWorkflowRequest(BaseModel):
    """Request to execute a workflow using a specific pattern."""
    pattern_id: str = Field(..., description="ID of pattern to execute")
    task: Dict[str, Any] = Field(..., description="Task definition and parameters")
    task_id: Optional[str] = Field(None, description="Custom task ID")
    requirements: Optional[TaskRequirements] = Field(None, description="Task requirements")
    execution_config: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Execution configuration")
    async_execution: bool = Field(False, description="Execute asynchronously")
    callback_url: Optional[str] = Field(None, description="Callback URL for async results")
    
    @validator('task')
    def validate_task(cls, v):
        """Validate task structure."""
        required_fields = ['type', 'description']
        for field in required_fields:
            if field not in v:
                raise ValueError(f"Task must include '{field}' field")
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "pattern_id": "research_pipeline_001",
                "task": {
                    "type": "research",
                    "description": "Comprehensive market analysis for AI orchestration platforms",
                    "scope": "global",
                    "depth": "detailed"
                },
                "requirements": {
                    "priority": "high", 
                    "quality_level": "publication_ready"
                },
                "execution_config": {
                    "timeout": 300,
                    "max_retries": 2
                },
                "async_execution": False
            }
        }


class BulkWorkflowRequest(BaseModel):
    """Request to execute multiple workflows."""
    workflows: List[ExecuteWorkflowRequest] = Field(..., description="List of workflows to execute")
    execution_mode: str = Field("parallel", description="Execution mode: parallel, sequential")
    max_concurrent: Optional[int] = Field(5, description="Maximum concurrent workflows")
    
    class Config:
        schema_extra = {
            "example": {
                "workflows": [
                    {
                        "pattern_id": "research_pipeline_001",
                        "task": {"type": "research", "description": "Market analysis"},
                        "async_execution": True
                    },
                    {
                        "pattern_id": "competitive_parallel_001", 
                        "task": {"type": "competitive_analysis", "description": "Competitor analysis"},
                        "async_execution": True
                    }
                ],
                "execution_mode": "parallel",
                "max_concurrent": 2
            }
        }


class MonitoringQuery(BaseModel):
    """Query parameters for monitoring endpoints."""
    time_range: Optional[str] = Field("1h", description="Time range for metrics (1h, 1d, 1w)")
    pattern_type: Optional[PatternType] = Field(None, description="Filter by pattern type")
    agent_id: Optional[str] = Field(None, description="Filter by agent ID")
    include_details: bool = Field(False, description="Include detailed metrics")
    
    class Config:
        schema_extra = {
            "example": {
                "time_range": "24h",
                "pattern_type": "pipeline",
                "include_details": True
            }
        }