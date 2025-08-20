"""
FastAPI application for Media Mix Modeling Platform

Provides REST API endpoints for attribution analysis, budget optimization, and media performance insights.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from contextlib import asynccontextmanager
import logging
import sys
import os
from typing import Dict, Any

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.middleware.error_handling import error_handler_middleware
from api.middleware.rate_limiting import RateLimitingMiddleware
from api.routers import attribution, optimization, health, performance

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    # Startup
    logger.info("Starting Media Mix Modeling API...")
    
    # Initialize any necessary services here
    # e.g., model loading, database connections
    
    logger.info("API startup complete")
    yield
    
    # Shutdown
    logger.info("Shutting down API...")


# Create FastAPI application
app = FastAPI(
    title="Media Mix Modeling API",
    description="Advanced attribution analysis and budget optimization for marketing campaigns",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
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
rate_limiter = RateLimitingMiddleware()
app.add_middleware(type(rate_limiter), dispatch=rate_limiter.dispatch)

# Add error handling middleware
app.middleware("http")(error_handler_middleware)

# Include routers
app.include_router(health.router, prefix="/health", tags=["health"])
app.include_router(attribution.router, prefix="/api/v1/attribution", tags=["attribution"])
app.include_router(optimization.router, prefix="/api/v1/optimization", tags=["optimization"])
app.include_router(performance.router, prefix="/api/v1/performance", tags=["performance"])

@app.get("/", include_in_schema=False)
async def root():
    """Redirect root to documentation."""
    return RedirectResponse(url="/docs")

@app.get("/info")
async def info() -> Dict[str, Any]:
    """Get API information."""
    return {
        "name": "Media Mix Modeling API",
        "version": "1.0.0",
        "description": "Advanced attribution analysis and budget optimization platform",
        "features": [
            "Multi-touch attribution modeling",
            "Budget optimization algorithms",
            "Cross-channel performance analysis",
            "Campaign ROI measurement",
            "Media saturation curves",
            "Incrementality testing",
            "Real-time performance monitoring"
        ],
        "endpoints": {
            "health": "/health/status",
            "attribution": "/api/v1/attribution/",
            "optimization": "/api/v1/optimization/",
            "performance": "/api/v1/performance/",
            "documentation": "/docs"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )