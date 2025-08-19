"""
Vision Object Classifier API

FastAPI application for classifying images as clean or dirty objects.
Supports single image upload and batch processing.
"""

import sys
from pathlib import Path
from typing import List
import uvicorn

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.routers import classification, health, batch
from api.middleware.error_handling import error_handler
from api.middleware.rate_limiting import RateLimitMiddleware

# Create FastAPI application
app = FastAPI(
    title="Vision Object Classifier API",
    description="AI-powered image classification for clean/dirty object detection",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add rate limiting middleware
app.add_middleware(RateLimitMiddleware, calls=100, period=60)  # 100 requests per minute

# Add error handling middleware
app.middleware("http")(error_handler)

# Include routers
app.include_router(health.router, prefix="/health", tags=["health"])
app.include_router(classification.router, prefix="/api/v1/classify", tags=["classification"])
app.include_router(batch.router, prefix="/api/v1/batch", tags=["batch"])

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "Vision Object Classifier API",
        "version": "1.0.0",
        "description": "AI-powered clean/dirty object classification",
        "docs": "/docs",
        "health": "/health/status",
        "endpoints": {
            "classify_single": "POST /api/v1/classify/single",
            "classify_batch": "POST /api/v1/batch/classify",
            "model_info": "GET /api/v1/classify/model-info"
        }
    }

@app.get("/api/v1/info")
async def api_info():
    """Detailed API information"""
    return {
        "api_name": "Vision Object Classifier",
        "version": "1.0.0",
        "supported_formats": ["jpg", "jpeg", "png"],
        "max_file_size": "10MB",
        "classification_classes": ["clean", "dirty"],
        "model_types": ["fast", "balanced", "accurate"],
        "features": [
            "Single image classification",
            "Batch processing",
            "Confidence scoring",
            "Multiple model variants",
            "Real-time processing"
        ]
    }

if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )