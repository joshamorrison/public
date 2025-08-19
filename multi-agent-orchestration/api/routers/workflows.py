"""
Workflows API Router

FastAPI router for workflow execution endpoints.
Handles pattern execution, workflow management, and async processing.
"""

import sys
import os
import asyncio
import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime
import time

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'src'))

from fastapi import APIRouter, HTTPException, Depends, Query, Path, BackgroundTasks, status
from fastapi.responses import JSONResponse

from src.multi_agent_platform import MultiAgentPlatform
from ..models.request_models import ExecuteWorkflowRequest, BulkWorkflowRequest, MonitoringQuery
from ..models.response_models import (
    ExecutionResultResponse, WorkflowResponse, BulkOperationResponse, 
    MonitoringResponse
)


router = APIRouter()

# In-memory storage for async executions (use Redis in production)
active_executions: Dict[str, Dict[str, Any]] = {}
execution_history: List[Dict[str, Any]] = []


# Dependency to get platform instance
async def get_platform() -> MultiAgentPlatform:
    """Get platform instance - will be overridden by dependency injection."""
    raise HTTPException(status_code=500, detail="Platform dependency not properly configured")


@router.post(
    "/execute",
    response_model=ExecutionResultResponse,
    summary="Execute Workflow",
    description="Execute a workflow using a specified orchestration pattern"
)
async def execute_workflow(
    workflow_request: ExecuteWorkflowRequest,
    background_tasks: BackgroundTasks,
    platform: MultiAgentPlatform = Depends(get_platform)
):
    """Execute a workflow synchronously or asynchronously."""
    execution_id = f"exec_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    start_time = datetime.now()
    
    try:
        # Validate pattern exists
        pattern = platform.get_pattern(workflow_request.pattern_id)
        if not pattern:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Pattern '{workflow_request.pattern_id}' not found"
            )
        
        if workflow_request.async_execution:
            # Start async execution
            active_executions[execution_id] = {
                "execution_id": execution_id,
                "pattern_id": workflow_request.pattern_id,
                "task_id": workflow_request.task_id,
                "status": "running",
                "started_at": start_time,
                "task": workflow_request.task,
                "callback_url": workflow_request.callback_url
            }
            
            # Schedule background execution
            background_tasks.add_task(
                _execute_workflow_async,
                platform,
                execution_id,
                workflow_request
            )
            
            return ExecutionResultResponse(
                execution_id=execution_id,
                pattern_id=workflow_request.pattern_id,
                task_id=workflow_request.task_id,
                status="running",
                success=True,
                started_at=start_time,
                execution_time=0.0,
                result={"message": "Workflow execution started asynchronously"},
                metadata={
                    "async_execution": True,
                    "callback_url": workflow_request.callback_url,
                    "status_endpoint": f"/api/v1/workflows/executions/{execution_id}"
                }
            )
        
        else:
            # Execute synchronously
            execution_start = time.time()
            
            result = await platform.execute_pattern(
                workflow_request.pattern_id,
                workflow_request.task,
                workflow_request.execution_config
            )
            
            execution_time = time.time() - execution_start
            completed_at = datetime.now()
            
            # Store execution result
            execution_record = {
                "execution_id": execution_id,
                "pattern_id": workflow_request.pattern_id,
                "task_id": workflow_request.task_id,
                "status": "completed" if result.get("success") else "failed",
                "success": result.get("success", False),
                "started_at": start_time,
                "completed_at": completed_at,
                "execution_time": execution_time,
                "result": result,
                "error": result.get("error") if not result.get("success") else None
            }
            execution_history.append(execution_record)
            
            return ExecutionResultResponse(
                execution_id=execution_id,
                pattern_id=workflow_request.pattern_id,
                task_id=workflow_request.task_id,
                status="completed" if result.get("success") else "failed",
                success=result.get("success", False),
                started_at=start_time,
                completed_at=completed_at,
                execution_time=execution_time,
                result=result,
                error=result.get("error") if not result.get("success") else None,
                metadata={
                    "pattern_type": type(pattern).__name__.replace('Pattern', '').lower(),
                    "sync_execution": True
                }
            )
            
    except HTTPException:
        raise
    except Exception as e:
        return ExecutionResultResponse(
            execution_id=execution_id,
            pattern_id=workflow_request.pattern_id,
            task_id=workflow_request.task_id,
            status="failed",
            success=False,
            started_at=start_time,
            execution_time=0.0,
            error=f"Execution failed: {str(e)}",
            metadata={"error_type": type(e).__name__}
        )


@router.post(
    "/bulk",
    response_model=BulkOperationResponse,
    summary="Execute Multiple Workflows",
    description="Execute multiple workflows in parallel or sequential mode"
)
async def execute_bulk_workflows(
    bulk_request: BulkWorkflowRequest,
    background_tasks: BackgroundTasks,
    platform: MultiAgentPlatform = Depends(get_platform)
):
    """Execute multiple workflows."""
    start_time = time.time()
    results = []
    
    try:
        if bulk_request.execution_mode == "parallel":
            # Execute workflows in parallel
            tasks = []
            for workflow_req in bulk_request.workflows[:bulk_request.max_concurrent]:
                task = _execute_single_workflow(platform, workflow_req)
                tasks.append(task)
            
            # Wait for all tasks to complete
            parallel_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for i, result in enumerate(parallel_results):
                if isinstance(result, Exception):
                    results.append({
                        "workflow_index": i,
                        "status": "failed",
                        "error": str(result)
                    })
                else:
                    results.append({
                        "workflow_index": i,
                        "status": "success",
                        "result": result
                    })
        
        else:
            # Execute workflows sequentially
            for i, workflow_req in enumerate(bulk_request.workflows):
                try:
                    result = await _execute_single_workflow(platform, workflow_req)
                    results.append({
                        "workflow_index": i,
                        "status": "success", 
                        "result": result
                    })
                except Exception as e:
                    results.append({
                        "workflow_index": i,
                        "status": "failed",
                        "error": str(e)
                    })
        
        # Calculate summary statistics
        successful = len([r for r in results if r["status"] == "success"])
        failed = len([r for r in results if r["status"] == "failed"])
        execution_time = time.time() - start_time
        
        return BulkOperationResponse(
            total_requests=len(bulk_request.workflows),
            successful=successful,
            failed=failed,
            results=results,
            execution_time=execution_time
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Bulk execution failed: {str(e)}"
        )


@router.get(
    "/executions",
    response_model=List[ExecutionResultResponse],
    summary="List Workflow Executions",
    description="Get list of workflow executions with filtering options"
)
async def list_executions(
    pattern_id: Optional[str] = Query(None, description="Filter by pattern ID"),
    status_filter: Optional[str] = Query(None, alias="status", description="Filter by execution status"),
    limit: int = Query(50, ge=1, le=1000, description="Maximum number of executions to return"),
    offset: int = Query(0, ge=0, description="Number of executions to skip"),
    include_running: bool = Query(True, description="Include currently running executions")
):
    """List workflow executions with filtering and pagination."""
    try:
        all_executions = []
        
        # Add completed executions
        for exec_record in execution_history:
            if pattern_id and exec_record["pattern_id"] != pattern_id:
                continue
            if status_filter and exec_record["status"] != status_filter:
                continue
                
            all_executions.append(ExecutionResultResponse(**exec_record))
        
        # Add running executions if requested
        if include_running:
            for exec_id, exec_data in active_executions.items():
                if pattern_id and exec_data["pattern_id"] != pattern_id:
                    continue
                if status_filter and exec_data["status"] != status_filter:
                    continue
                
                all_executions.append(ExecutionResultResponse(
                    execution_id=exec_data["execution_id"],
                    pattern_id=exec_data["pattern_id"],
                    task_id=exec_data.get("task_id"),
                    status=exec_data["status"],
                    success=exec_data["status"] == "completed",
                    started_at=exec_data["started_at"],
                    execution_time=(datetime.now() - exec_data["started_at"]).total_seconds(),
                    result=exec_data.get("result"),
                    metadata={"currently_running": True}
                ))
        
        # Sort by start time (most recent first)
        all_executions.sort(key=lambda x: x.started_at, reverse=True)
        
        # Apply pagination
        return all_executions[offset:offset + limit]
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list executions: {str(e)}"
        )


@router.get(
    "/executions/{execution_id}",
    response_model=ExecutionResultResponse,
    summary="Get Execution Details",
    description="Get detailed information about a specific workflow execution"
)
async def get_execution(
    execution_id: str = Path(..., description="Execution unique identifier")
):
    """Get details of a specific execution."""
    try:
        # Check active executions first
        if execution_id in active_executions:
            exec_data = active_executions[execution_id]
            return ExecutionResultResponse(
                execution_id=exec_data["execution_id"],
                pattern_id=exec_data["pattern_id"],
                task_id=exec_data.get("task_id"),
                status=exec_data["status"],
                success=exec_data["status"] == "completed",
                started_at=exec_data["started_at"],
                execution_time=(datetime.now() - exec_data["started_at"]).total_seconds(),
                result=exec_data.get("result"),
                metadata={"currently_running": True}
            )
        
        # Check execution history
        for exec_record in execution_history:
            if exec_record["execution_id"] == execution_id:
                return ExecutionResultResponse(**exec_record)
        
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Execution '{execution_id}' not found"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get execution: {str(e)}"
        )


@router.delete(
    "/executions/{execution_id}",
    status_code=status.HTTP_200_OK,
    summary="Cancel Execution",
    description="Cancel a running workflow execution"
)
async def cancel_execution(
    execution_id: str = Path(..., description="Execution unique identifier")
):
    """Cancel a running execution."""
    try:
        if execution_id not in active_executions:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Active execution '{execution_id}' not found"
            )
        
        exec_data = active_executions[execution_id]
        if exec_data["status"] != "running":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Execution '{execution_id}' is not running (status: {exec_data['status']})"
            )
        
        # Mark as cancelled
        exec_data["status"] = "cancelled"
        exec_data["completed_at"] = datetime.now()
        exec_data["error"] = "Execution cancelled by user request"
        
        # Move to history
        execution_history.append(exec_data)
        del active_executions[execution_id]
        
        return {"message": f"Execution '{execution_id}' cancelled successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to cancel execution: {str(e)}"
        )


@router.get(
    "/templates",
    summary="Get Workflow Templates",
    description="Get pre-built workflow templates for each pattern type"
)
async def get_workflow_templates():
    """Get workflow templates for different patterns."""
    return {
        "templates": {
            "pipeline": {
                "name": "Research Pipeline Template",
                "description": "Sequential research workflow with quality gates",
                "task_template": {
                    "type": "research",
                    "description": "Research topic with comprehensive analysis",
                    "requirements": {
                        "depth": "comprehensive",
                        "quality_level": "publication_ready"
                    }
                },
                "example": {
                    "type": "research",
                    "description": "Market analysis for AI orchestration platforms",
                    "scope": "global",
                    "focus": "competitive_landscape"
                }
            },
            "supervisor": {
                "name": "Strategic Analysis Template",
                "description": "Hierarchical coordination for complex analysis",
                "task_template": {
                    "type": "strategic_analysis",
                    "description": "Multi-dimensional strategic analysis",
                    "requirements": {
                        "scope": "comprehensive",
                        "stakeholders": "leadership_team"
                    }
                },
                "example": {
                    "type": "strategic_analysis",
                    "description": "Business strategy for AI platform expansion",
                    "timeframe": "2024-2026",
                    "focus_areas": ["market_expansion", "technology_roadmap"]
                }
            },
            "parallel": {
                "name": "Competitive Analysis Template",
                "description": "Concurrent multi-dimensional analysis",
                "task_template": {
                    "type": "competitive_analysis",
                    "description": "Parallel competitive intelligence gathering",
                    "requirements": {
                        "dimensions": ["technology", "market", "pricing"],
                        "concurrency": "maximum"
                    }
                },
                "example": {
                    "type": "competitive_analysis",
                    "description": "AI platform competitive landscape",
                    "competitors": ["OpenAI", "Anthropic", "Microsoft", "Google"],
                    "analysis_depth": "detailed"
                }
            },
            "reflective": {
                "name": "Content Optimization Template",
                "description": "Self-improving iterative refinement",
                "task_template": {
                    "type": "content_optimization",
                    "description": "Iterative content improvement with feedback",
                    "requirements": {
                        "quality_level": "publication_ready",
                        "iteration_focus": "clarity_and_impact"
                    }
                },
                "example": {
                    "type": "content_optimization",
                    "description": "Strategic whitepaper optimization",
                    "target_audience": "enterprise_architects",
                    "content_type": "technical_whitepaper"
                }
            }
        }
    }


# Helper functions

async def _execute_workflow_async(
    platform: MultiAgentPlatform,
    execution_id: str,
    workflow_request: ExecuteWorkflowRequest
):
    """Execute workflow asynchronously in background."""
    try:
        execution_start = time.time()
        
        result = await platform.execute_pattern(
            workflow_request.pattern_id,
            workflow_request.task,
            workflow_request.execution_config
        )
        
        execution_time = time.time() - execution_start
        completed_at = datetime.now()
        
        # Update execution record
        if execution_id in active_executions:
            exec_data = active_executions[execution_id]
            exec_data.update({
                "status": "completed" if result.get("success") else "failed",
                "success": result.get("success", False),
                "completed_at": completed_at,
                "execution_time": execution_time,
                "result": result,
                "error": result.get("error") if not result.get("success") else None
            })
            
            # Move to history
            execution_history.append(exec_data.copy())
            del active_executions[execution_id]
            
            # TODO: Send callback if URL provided
            if exec_data.get("callback_url"):
                print(f"[WORKFLOW] Would send callback to {exec_data['callback_url']} for {execution_id}")
        
    except Exception as e:
        # Handle async execution error
        if execution_id in active_executions:
            exec_data = active_executions[execution_id]
            exec_data.update({
                "status": "failed",
                "success": False,
                "completed_at": datetime.now(),
                "execution_time": time.time() - time.mktime(exec_data["started_at"].timetuple()),
                "error": f"Async execution failed: {str(e)}"
            })
            
            execution_history.append(exec_data.copy())
            del active_executions[execution_id]


async def _execute_single_workflow(
    platform: MultiAgentPlatform,
    workflow_request: ExecuteWorkflowRequest
) -> Dict[str, Any]:
    """Execute a single workflow and return result."""
    result = await platform.execute_pattern(
        workflow_request.pattern_id,
        workflow_request.task,
        workflow_request.execution_config
    )
    return result