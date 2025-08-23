from fastapi import APIRouter
from datetime import datetime
from ..models.response_models import HealthCheckResponse

router = APIRouter()

@router.get("/health", response_model=HealthCheckResponse)
async def health_check():
    return HealthCheckResponse(
        status="healthy",
        timestamp=datetime.now(),
        version="1.0.0",
        active_jobs=0,
        system_metrics={"status": "operational"}
    )