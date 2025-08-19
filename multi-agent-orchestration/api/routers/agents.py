"""
Agents API Router

FastAPI router for agent management endpoints.
Provides CRUD operations for agents and pattern management.
"""

import sys
import os
from typing import List, Dict, Any, Optional
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'src'))

from fastapi import APIRouter, HTTPException, Depends, Query, Path, status
from fastapi.responses import JSONResponse

from src.multi_agent_platform import MultiAgentPlatform
from ..models.request_models import CreateAgentRequest, PatternConfigurationRequest
from ..models.response_models import AgentResponse, PatternResponse, BulkOperationResponse


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
    "/",
    response_model=List[AgentResponse],
    summary="List All Agents",
    description="Get list of all registered agents with their capabilities and status"
)
async def list_agents(
    agent_type: Optional[str] = Query(None, description="Filter by agent type"),
    status_filter: Optional[str] = Query(None, alias="status", description="Filter by agent status"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of agents to return"),
    offset: int = Query(0, ge=0, description="Number of agents to skip"),
    platform: MultiAgentPlatform = Depends(get_platform)
):
    """List all registered agents with optional filtering."""
    try:
        agents_data = platform.list_agents()
        
        # Apply filters
        filtered_agents = []
        for agent_id, agent_info in agents_data.items():
            # Type filter
            if agent_type and agent_info["type"].lower() != agent_type.lower():
                continue
                
            # Status filter (assuming active status for now)
            if status_filter and status_filter.lower() != "active":
                continue
                
            agent_response = AgentResponse(
                agent_id=agent_id,
                name=agent_info["name"],
                agent_type=agent_info["type"],
                capabilities=agent_info["capabilities"],
                performance_metrics=agent_info["performance"],
                status="active",  # Default status
                created_at=datetime.now()  # Would be tracked in real implementation
            )
            filtered_agents.append(agent_response)
        
        # Apply pagination
        total_agents = len(filtered_agents)
        paginated_agents = filtered_agents[offset:offset + limit]
        
        return paginated_agents
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list agents: {str(e)}"
        )


@router.post(
    "/",
    response_model=AgentResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create New Agent",
    description="Create and register a new agent with specified type and configuration"
)
async def create_agent(
    agent_request: CreateAgentRequest,
    platform: MultiAgentPlatform = Depends(get_platform)
):
    """Create a new agent."""
    try:
        # Create agent using platform
        agent = platform.create_agent(
            agent_type=agent_request.agent_type.value,
            agent_id=agent_request.agent_id,
            **agent_request.configuration
        )
        
        # Get agent capabilities
        capabilities = agent.get_capabilities() if hasattr(agent, 'get_capabilities') else []
        
        return AgentResponse(
            agent_id=agent.agent_id,
            name=agent_request.name or agent.name,
            agent_type=agent_request.agent_type.value,
            capabilities=capabilities,
            performance_metrics=agent.performance_metrics,
            status="active",
            created_at=datetime.now()
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid agent configuration: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create agent: {str(e)}"
        )


@router.get(
    "/{agent_id}",
    response_model=AgentResponse,
    summary="Get Agent Details",
    description="Get detailed information about a specific agent"
)
async def get_agent(
    agent_id: str = Path(..., description="Agent unique identifier"),
    platform: MultiAgentPlatform = Depends(get_platform)
):
    """Get details of a specific agent."""
    try:
        agent = platform.get_agent(agent_id)
        if not agent:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Agent '{agent_id}' not found"
            )
        
        capabilities = agent.get_capabilities() if hasattr(agent, 'get_capabilities') else []
        
        return AgentResponse(
            agent_id=agent.agent_id,
            name=agent.name,
            agent_type=type(agent).__name__.replace('Agent', '').lower(),
            capabilities=capabilities,
            performance_metrics=agent.performance_metrics,
            status="active",
            created_at=datetime.now()  # Would be tracked in real implementation
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get agent: {str(e)}"
        )


@router.delete(
    "/{agent_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete Agent",
    description="Remove an agent from the platform"
)
async def delete_agent(
    agent_id: str = Path(..., description="Agent unique identifier"),
    platform: MultiAgentPlatform = Depends(get_platform)
):
    """Delete an agent."""
    try:
        if agent_id not in platform.registered_agents:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Agent '{agent_id}' not found"
            )
        
        # Remove agent from platform
        del platform.registered_agents[agent_id]
        
        return None
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete agent: {str(e)}"
        )


# Pattern Management Endpoints

@router.get(
    "/patterns/",
    response_model=List[PatternResponse],
    summary="List All Patterns",
    description="Get list of all active orchestration patterns"
)
async def list_patterns(
    pattern_type: Optional[str] = Query(None, description="Filter by pattern type"),
    platform: MultiAgentPlatform = Depends(get_platform)
):
    """List all active patterns."""
    try:
        patterns_data = platform.list_patterns()
        
        pattern_responses = []
        for pattern_id, pattern_info in patterns_data.items():
            # Type filter
            if pattern_type and pattern_info["type"].lower() != pattern_type.lower():
                continue
            
            # Get agents involved in this pattern
            pattern_obj = platform.get_pattern(pattern_id)
            agents_involved = []
            
            if hasattr(pattern_obj, 'pipeline_stages'):
                agents_involved = [stage['agent'].agent_id for stage in pattern_obj.pipeline_stages]
            elif hasattr(pattern_obj, 'supervisor') and hasattr(pattern_obj.supervisor, 'specialist_agents'):
                agents_involved = list(pattern_obj.supervisor.specialist_agents.keys())
            elif hasattr(pattern_obj, 'parallel_agents'):
                agents_involved = [agent_config['agent'].agent_id for agent_config in pattern_obj.parallel_agents]
            elif hasattr(pattern_obj, 'primary_agent'):
                agents_involved = [pattern_obj.primary_agent.agent_id] if pattern_obj.primary_agent else []
                if hasattr(pattern_obj, 'critic_agents'):
                    agents_involved.extend([agent.agent_id for agent in pattern_obj.critic_agents])
            
            pattern_response = PatternResponse(
                pattern_id=pattern_id,
                name=pattern_info["name"],
                pattern_type=pattern_info["type"],
                configuration=pattern_info["configuration"],
                agents_involved=agents_involved,
                status="active",
                created_at=datetime.now(),
                execution_count=0  # Would be tracked in real implementation
            )
            pattern_responses.append(pattern_response)
        
        return pattern_responses
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list patterns: {str(e)}"
        )


@router.post(
    "/patterns/",
    response_model=PatternResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create New Pattern",
    description="Create a new orchestration pattern with specified configuration"
)
async def create_pattern(
    pattern_request: PatternConfigurationRequest,
    platform: MultiAgentPlatform = Depends(get_platform)
):
    """Create a new orchestration pattern."""
    try:
        # Create pattern based on type
        if pattern_request.pattern_type == "supervisor" and pattern_request.supervisor_agent_id:
            supervisor_agent = platform.get_agent(pattern_request.supervisor_agent_id)
            if not supervisor_agent:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Supervisor agent '{pattern_request.supervisor_agent_id}' not found"
                )
            pattern = platform.create_pattern(
                pattern_request.pattern_type.value,
                pattern_request.pattern_id,
                supervisor_agent=supervisor_agent
            )
        elif pattern_request.pattern_type == "reflective" and pattern_request.primary_agent_id:
            primary_agent = platform.get_agent(pattern_request.primary_agent_id)
            if not primary_agent:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Primary agent '{pattern_request.primary_agent_id}' not found"
                )
            pattern = platform.create_pattern(
                pattern_request.pattern_type.value,
                pattern_request.pattern_id,
                primary_agent=primary_agent
            )
        else:
            pattern = platform.create_pattern(
                pattern_request.pattern_type.value,
                pattern_request.pattern_id
            )
        
        # Get agents involved
        agents_involved = []
        if hasattr(pattern, 'pipeline_stages'):
            agents_involved = [stage['agent'].agent_id for stage in pattern.pipeline_stages]
        elif hasattr(pattern, 'supervisor') and hasattr(pattern.supervisor, 'specialist_agents'):
            agents_involved = list(pattern.supervisor.specialist_agents.keys())
        
        return PatternResponse(
            pattern_id=pattern.pattern_id,
            name=pattern_request.name or pattern.name,
            pattern_type=pattern_request.pattern_type.value,
            configuration=pattern_request.configuration,
            agents_involved=agents_involved,
            status="active",
            created_at=datetime.now(),
            execution_count=0
        )
        
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid pattern configuration: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create pattern: {str(e)}"
        )


@router.get(
    "/patterns/{pattern_id}",
    response_model=PatternResponse,
    summary="Get Pattern Details",
    description="Get detailed information about a specific orchestration pattern"
)
async def get_pattern(
    pattern_id: str = Path(..., description="Pattern unique identifier"),
    platform: MultiAgentPlatform = Depends(get_platform)
):
    """Get details of a specific pattern."""
    try:
        pattern = platform.get_pattern(pattern_id)
        if not pattern:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Pattern '{pattern_id}' not found"
            )
        
        # Get pattern configuration
        configuration = pattern.get_pattern_configuration() if hasattr(pattern, 'get_pattern_configuration') else {}
        
        # Get agents involved
        agents_involved = []
        if hasattr(pattern, 'pipeline_stages'):
            agents_involved = [stage['agent'].agent_id for stage in pattern.pipeline_stages]
        elif hasattr(pattern, 'supervisor') and hasattr(pattern.supervisor, 'specialist_agents'):
            agents_involved = list(pattern.supervisor.specialist_agents.keys())
        elif hasattr(pattern, 'parallel_agents'):
            agents_involved = [agent_config['agent'].agent_id for agent_config in pattern.parallel_agents]
        elif hasattr(pattern, 'primary_agent'):
            agents_involved = [pattern.primary_agent.agent_id] if pattern.primary_agent else []
            if hasattr(pattern, 'critic_agents'):
                agents_involved.extend([agent.agent_id for agent in pattern.critic_agents])
        
        return PatternResponse(
            pattern_id=pattern.pattern_id,
            name=pattern.name,
            pattern_type=type(pattern).__name__.replace('Pattern', '').lower(),
            configuration=configuration,
            agents_involved=agents_involved,
            status="active",
            created_at=datetime.now(),
            execution_count=0  # Would be tracked in real implementation
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get pattern: {str(e)}"
        )


@router.delete(
    "/patterns/{pattern_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete Pattern",
    description="Remove a pattern from the platform"
)
async def delete_pattern(
    pattern_id: str = Path(..., description="Pattern unique identifier"),
    platform: MultiAgentPlatform = Depends(get_platform)
):
    """Delete a pattern."""
    try:
        if pattern_id not in platform.active_patterns:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Pattern '{pattern_id}' not found"
            )
        
        # Remove pattern from platform
        del platform.active_patterns[pattern_id]
        
        return None
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete pattern: {str(e)}"
        )