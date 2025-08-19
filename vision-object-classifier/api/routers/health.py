"""
Health check endpoints for the Vision Object Classifier API
"""

import time
import psutil
from pathlib import Path
from fastapi import APIRouter, HTTPException
from api.models.response_models import HealthResponse, ModelInfoResponse

router = APIRouter()

# Track startup time for uptime calculation
startup_time = time.time()

@router.get("/status", response_model=HealthResponse)
async def health_check():
    """
    Basic health check endpoint
    Returns service status and basic metrics
    """
    try:
        # Check if model files exist
        models_dir = Path(__file__).parent.parent.parent / "models"
        model_status = {}
        
        for model_type in ["fast", "balanced"]:
            model_file = models_dir / f"{model_type}_model.pth"
            model_status[model_type] = "available" if model_file.exists() else "missing"
        
        uptime = time.time() - startup_time
        
        # Determine overall status
        status = "healthy" if any(status == "available" for status in model_status.values()) else "unhealthy"
        
        return HealthResponse(
            status=status,
            version="1.0.0",
            uptime_seconds=uptime,
            model_status=model_status
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

@router.get("/detailed", response_model=dict)
async def detailed_health_check():
    """
    Detailed health check with system metrics
    """
    try:
        # System metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Model status
        models_dir = Path(__file__).parent.parent.parent / "models"
        model_files = {}
        for model_file in models_dir.glob("*.pth"):
            model_files[model_file.stem] = {
                "size_mb": model_file.stat().st_size / (1024 * 1024),
                "modified": model_file.stat().st_mtime
            }
        
        return {
            "service": {
                "status": "healthy",
                "version": "1.0.0",
                "uptime_seconds": time.time() - startup_time
            },
            "system": {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_available_gb": memory.available / (1024**3),
                "disk_percent": disk.percent,
                "disk_free_gb": disk.free / (1024**3)
            },
            "models": model_files
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Detailed health check failed: {str(e)}")

@router.get("/models", response_model=ModelInfoResponse)
async def model_info():
    """
    Get information about available models
    """
    try:
        models_dir = Path(__file__).parent.parent.parent / "models"
        available_models = []
        model_details = {}
        
        # Check for model files
        for model_type in ["fast", "balanced"]:
            model_file = models_dir / f"{model_type}_model.pth"
            config_file = models_dir / f"{model_type}_config.json"
            
            if model_file.exists():
                available_models.append(model_type)
                
                # Get model details
                details = {
                    "file_size_mb": model_file.stat().st_size / (1024 * 1024),
                    "last_modified": model_file.stat().st_mtime,
                    "config_available": config_file.exists()
                }
                
                # Add performance characteristics
                if model_type == "fast":
                    details.update({
                        "speed": "high",
                        "accuracy": "medium",
                        "memory_usage": "low"
                    })
                elif model_type == "balanced":
                    details.update({
                        "speed": "medium",
                        "accuracy": "high", 
                        "memory_usage": "medium"
                    })
                
                model_details[model_type] = details
        
        return ModelInfoResponse(
            available_models=available_models,
            default_model="balanced",
            model_details=model_details,
            supported_formats=["jpg", "jpeg", "png"],
            max_file_size_mb=10
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model info failed: {str(e)}")

@router.get("/ready")
async def readiness_check():
    """
    Kubernetes-style readiness probe
    Returns 200 if service is ready to serve requests
    """
    try:
        # Check if at least one model is available
        models_dir = Path(__file__).parent.parent.parent / "models"
        has_model = any((models_dir / f"{model}_model.pth").exists() 
                       for model in ["fast", "balanced"])
        
        if has_model:
            return {"status": "ready"}
        else:
            raise HTTPException(status_code=503, detail="No models available")
            
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service not ready: {str(e)}")

@router.get("/live")
async def liveness_check():
    """
    Kubernetes-style liveness probe
    Returns 200 if service is alive
    """
    return {"status": "alive", "timestamp": time.time()}