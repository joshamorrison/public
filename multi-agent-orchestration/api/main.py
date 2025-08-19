#!/usr/bin/env python3
"""
Multi-Agent Orchestration Platform - FastAPI Application

Main FastAPI application providing RESTful endpoints for multi-agent workflows.
Supports all four orchestration patterns: Pipeline, Supervisor, Parallel, Reflective.
"""

import sys
import os
from contextlib import asynccontextmanager
from typing import Dict, Any

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
import uvicorn

# Import platform and routers
from src.multi_agent_platform import MultiAgentPlatform, create_platform
from .routers import agents, workflows, monitoring
from .middleware import error_handling, rate_limiting
from .models.response_models import HealthCheckResponse, PlatformStatusResponse


# Global platform instance
platform: MultiAgentPlatform = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle - startup and shutdown events."""
    # Startup
    global platform
    platform = create_platform("api_platform")
    print(f"[API] Platform initialized: {platform.platform_id}")
    
    # Set platform instance in routers
    agents.set_platform_instance(platform)
    monitoring.set_platform_instance(platform)
    # workflows will need similar updates
    
    yield
    
    # Shutdown
    if platform:
        platform.cleanup_resources()
        print("[API] Platform resources cleaned up")


# Create FastAPI application
app = FastAPI(
    title="Multi-Agent Orchestration Platform API",
    description="""
    **Revolutionary Multi-Agent AI Orchestration Service**
    
    This API provides comprehensive multi-agent workflow orchestration using four core patterns:
    
    ## 🔄 **Pipeline Pattern**
    - Sequential agent workflows with quality gates
    - Use cases: Content creation, data processing, report generation
    
    ## 👥 **Supervisor Pattern** 
    - Hierarchical coordination with specialist agents
    - Use cases: Research projects, strategic analysis, complex problem solving
    
    ## ⚡ **Parallel Pattern**
    - Concurrent agent execution with intelligent result fusion
    - Use cases: Market analysis, competitive intelligence, scenario modeling
    
    ## 🔍 **Reflective Pattern**
    - Self-improving agents with feedback loops and meta-cognition
    - Use cases: Strategic planning, creative tasks, iterative refinement
    
    ## Features
    - **Production-ready**: Built with FastAPI for high performance
    - **Scalable**: Async processing with concurrent agent execution  
    - **Observable**: Complete monitoring and tracing capabilities
    - **Flexible**: Dynamic pattern composition and configuration
    """,
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/api/v1/openapi.json"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add custom middleware
app.add_middleware(error_handling.ErrorHandlingMiddleware)
app.add_middleware(rate_limiting.RateLimitingMiddleware)


# Dependency to get platform instance
async def get_platform() -> MultiAgentPlatform:
    """Get the global platform instance."""
    if platform is None:
        raise HTTPException(status_code=500, detail="Platform not initialized")
    return platform


# Root endpoint
@app.get(
    "/", 
    response_model=Dict[str, Any],
    tags=["Root"],
    summary="API Root Information"
)
async def root():
    """Get API root information and available endpoints."""
    return {
        "name": "Multi-Agent Orchestration Platform API",
        "version": "1.0.0",
        "description": "Production-ready multi-agent workflow orchestration service",
        "patterns_supported": ["pipeline", "supervisor", "parallel", "reflective"],
        "endpoints": {
            "health": "/health",
            "status": "/status", 
            "docs": "/docs",
            "agents": "/api/v1/agents",
            "workflows": "/api/v1/workflows",
            "monitoring": "/api/v1/monitoring"
        },
        "links": {
            "documentation": "/docs",
            "openapi_spec": "/api/v1/openapi.json",
            "github": "https://github.com/your-org/multi-agent-orchestration"
        }
    }


# Health check endpoint
@app.get(
    "/health",
    response_model=HealthCheckResponse,
    tags=["Health"],
    summary="Health Check"
)
async def health_check(platform: MultiAgentPlatform = Depends(get_platform)):
    """Check API and platform health status."""
    try:
        # Basic platform health check
        status = platform.get_platform_status()
        
        return HealthCheckResponse(
            status="healthy",
            timestamp="2024-01-01T00:00:00Z",  # Will be set properly by response model
            platform_id=platform.platform_id,
            version=platform.version,
            uptime_seconds=status["platform_info"]["uptime_seconds"],
            components={
                "platform": "healthy",
                "workflow_engine": "healthy",
                "agents": "healthy",
                "patterns": "healthy"
            }
        )
    except Exception as e:
        return HealthCheckResponse(
            status="unhealthy",
            timestamp="2024-01-01T00:00:00Z",
            platform_id="unknown",
            version="unknown",
            uptime_seconds=0,
            components={
                "platform": "unhealthy",
                "error": str(e)
            }
        )


# Platform status endpoint
@app.get(
    "/status",
    response_model=PlatformStatusResponse,
    tags=["Status"],
    summary="Platform Status"
)
async def platform_status(platform: MultiAgentPlatform = Depends(get_platform)):
    """Get comprehensive platform status and metrics."""
    try:
        status = platform.get_platform_status()
        performance = platform.get_performance_analytics()
        
        return PlatformStatusResponse(
            platform_info=status["platform_info"],
            component_status=status["component_status"],
            registry_status=status["registry_status"],
            platform_metrics=status["platform_metrics"],
            performance_summary=performance["performance_summary"],
            agent_count=len(platform.registered_agents),
            pattern_count=len(platform.active_patterns)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get platform status: {str(e)}")



# Include routers
app.include_router(
    agents.router,
    prefix="/api/v1/agents",
    tags=["Agents"]
)

app.include_router(
    workflows.router,
    prefix="/api/v1/workflows", 
    tags=["Workflows"]
)

app.include_router(
    monitoring.router,
    prefix="/api/v1/monitoring",
    tags=["Monitoring"]
)


# Custom OpenAPI schema
def custom_openapi():
    """Generate custom OpenAPI schema with enhanced documentation."""
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title="Multi-Agent Orchestration Platform API",
        version="1.0.0",
        description="""
        Production-ready FastAPI service for orchestrating multi-agent AI workflows.
        
        Built with four core architectural patterns that enable sophisticated agent coordination:
        - Pipeline: Sequential workflows with handoffs
        - Supervisor: Hierarchical delegation to specialists  
        - Parallel: Concurrent execution with result fusion
        - Reflective: Self-improving feedback loops
        """,
        routes=app.routes,
    )
    
    # Add custom schema extensions
    openapi_schema["info"]["x-logo"] = {
        "url": "https://fastapi.tiangolo.com/img/logo-margin/logo-teal.png"
    }
    
    openapi_schema["servers"] = [
        {"url": "http://localhost:8000", "description": "Development server"},
        {"url": "https://api.yourdomain.com", "description": "Production server"},
    ]
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi


# Exception handlers
@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    """Handle ValueError exceptions."""
    return JSONResponse(
        status_code=400,
        content={
            "error": "Bad Request",
            "detail": str(exc),
            "type": "ValueError"
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions."""
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error", 
            "detail": "An unexpected error occurred",
            "type": type(exc).__name__
        }
    )


# Main execution
if __name__ == "__main__":
    print("[API] Starting Multi-Agent Orchestration Platform API...")
    print("[API] Visit http://localhost:8000/docs for interactive API documentation")
    
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
        access_log=True
    )