"""
Health check endpoints for the Media Mix Modeling API.

Provides health, readiness, and status endpoints for monitoring and load balancing.
"""

import os
import sys
import psutil
import logging
from datetime import datetime
from typing import Dict, Any
from fastapi import APIRouter, HTTPException

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from api.models.response_models import HealthResponse

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/status", response_model=HealthResponse)
async def health_status():
    """
    Get detailed health status of the API and its dependencies.
    
    Returns comprehensive health information including:
    - API status and version
    - System resource usage
    - Dependent service status
    - Model availability
    """
    try:
        # Get system information
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # Check dependent services
        services = await _check_dependent_services()
        
        # Check model availability
        model_status = await _check_model_availability()
        
        return HealthResponse(
            status="healthy",
            timestamp=datetime.utcnow().isoformat(),
            version="1.0.0",
            services=services,
            system_info={
                "cpu_usage_percent": cpu_percent,
                "memory_usage_percent": memory.percent,
                "memory_available_gb": round(memory.available / (1024**3), 2),
                "disk_usage_percent": disk.percent,
                "disk_available_gb": round(disk.free / (1024**3), 2),
                "uptime_seconds": _get_uptime_seconds(),
                "models_available": model_status
            }
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(
            status_code=503,
            detail=f"Health check failed: {str(e)}"
        )

@router.get("/ready")
async def readiness_probe():
    """
    Kubernetes-style readiness probe.
    
    Returns 200 if the service is ready to accept requests,
    503 if it's still starting up or has issues.
    """
    try:
        # Quick checks for service readiness
        checks = {
            "data_clients": await _check_data_clients(),
            "models": await _check_basic_model_loading(),
            "memory": _check_memory_availability(),
            "disk_space": _check_disk_space()
        }
        
        # Service is ready if all checks pass
        if all(checks.values()):
            return {"status": "ready", "checks": checks}
        else:
            raise HTTPException(
                status_code=503,
                detail={"status": "not_ready", "checks": checks}
            )
            
    except Exception as e:
        logger.error(f"Readiness check failed: {str(e)}")
        raise HTTPException(
            status_code=503,
            detail=f"Readiness check failed: {str(e)}"
        )

@router.get("/live")
async def liveness_probe():
    """
    Kubernetes-style liveness probe.
    
    Returns 200 if the service is alive (basic health check),
    500 if the service should be restarted.
    """
    try:
        # Very basic checks - if this fails, restart the service
        return {
            "status": "alive",
            "timestamp": datetime.utcnow().isoformat(),
            "process_id": os.getpid()
        }
    except Exception as e:
        logger.error(f"Liveness check failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Liveness check failed: {str(e)}"
        )

@router.get("/metrics")
async def get_metrics():
    """
    Get basic metrics for monitoring.
    
    Returns system and application metrics suitable for monitoring systems.
    """
    try:
        memory = psutil.virtual_memory()
        
        return {
            "system": {
                "cpu_usage_percent": psutil.cpu_percent(interval=0.1),
                "memory_usage_percent": memory.percent,
                "memory_used_bytes": memory.used,
                "memory_total_bytes": memory.total,
                "uptime_seconds": _get_uptime_seconds()
            },
            "application": {
                "version": "1.0.0",
                "process_id": os.getpid(),
                "python_version": sys.version.split()[0]
            }
        }
        
    except Exception as e:
        logger.error(f"Metrics collection failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Metrics collection failed: {str(e)}"
        )

# Helper functions
async def _check_dependent_services() -> Dict[str, str]:
    """Check status of dependent services."""
    services = {}
    
    try:
        # Check data clients
        services["data_clients"] = "healthy" if await _check_data_clients() else "unhealthy"
    except:
        services["data_clients"] = "unknown"
    
    try:
        # Check if models can be loaded
        services["models"] = "healthy" if await _check_basic_model_loading() else "unhealthy"
    except:
        services["models"] = "unknown"
    
    # Add more service checks as needed
    services["database"] = "not_configured"  # Placeholder
    services["cache"] = "not_configured"     # Placeholder
    services["mlflow"] = "configured"        # MLflow is used in this project
    
    return services

async def _check_model_availability() -> Dict[str, bool]:
    """Check availability of different model components."""
    models = {
        "attribution_models": False,
        "budget_optimization": False,
        "mmm_models": False,
        "r_integration": False
    }
    
    try:
        # Check if we can import core modules
        from models.mmm.attribution_models import AttributionModels
        models["attribution_models"] = True
        
        from models.mmm.budget_optimizer import BudgetOptimizer
        models["budget_optimization"] = True
        
        from models.mmm.econometric_mmm import EconometricMMM
        models["mmm_models"] = True
        
        # Check R integration
        try:
            from models.r_integration.r_mmm_models import RMMModels
            models["r_integration"] = True
        except:
            pass
            
    except Exception as e:
        logger.warning(f"Model availability check failed: {str(e)}")
    
    return models

async def _check_data_clients() -> bool:
    """Quick check if data clients are available."""
    try:
        from data.media_data_client import MediaDataClient
        from data.facebook_ads_client import FacebookAdsClient
        from data.google_ads_client import GoogleAdsClient
        # Don't actually make API calls, just check if we can import
        return True
    except ImportError:
        return False

async def _check_basic_model_loading() -> bool:
    """Check if basic models can be loaded."""
    try:
        from models.mmm.attribution_models import AttributionModels
        # Quick instantiation test (doesn't load actual models)
        attribution = AttributionModels()
        return True
    except Exception:
        return False

def _check_memory_availability() -> bool:
    """Check if sufficient memory is available."""
    memory = psutil.virtual_memory()
    # Consider service ready if less than 90% memory used
    return memory.percent < 90

def _check_disk_space() -> bool:
    """Check if sufficient disk space is available."""
    disk = psutil.disk_usage('/')
    # Consider service ready if less than 90% disk used
    return disk.percent < 90

def _get_uptime_seconds() -> float:
    """Get approximate uptime in seconds."""
    try:
        # This is a simple approximation
        import time
        if not hasattr(_get_uptime_seconds, 'start_time'):
            _get_uptime_seconds.start_time = time.time()
        return time.time() - _get_uptime_seconds.start_time
    except:
        return 0.0