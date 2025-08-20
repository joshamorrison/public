"""
FastAPI application for Generative Econometric Forecasting Platform

Provides REST API endpoints for economic forecasting, analysis, and reporting.
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
from api.routers import forecasting, health, analysis

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
    logger.info("Starting Econometric Forecasting API...")
    
    # Initialize any necessary services here
    # e.g., model loading, database connections
    
    logger.info("API startup complete")
    yield
    
    # Shutdown
    logger.info("Shutting down API...")


# Create FastAPI application
app = FastAPI(
    title="Generative Econometric Forecasting API",
    description="AI-powered economic forecasting platform with foundation models and advanced analytics",
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
app.include_router(forecasting.router, prefix="/api/v1/forecast", tags=["forecasting"])
app.include_router(analysis.router, prefix="/api/v1/analysis", tags=["analysis"])

@app.get("/", include_in_schema=False)
async def root():
    """Redirect root to documentation."""
    return RedirectResponse(url="/docs")

@app.get("/info")
async def info() -> Dict[str, Any]:
    """Get API information."""
    return {
        "name": "Generative Econometric Forecasting API",
        "version": "1.0.0",
        "description": "AI-powered economic forecasting platform",
        "features": [
            "Three-tier foundation models",
            "Real-time FRED data integration",
            "30+ neural forecasting models",
            "Sentiment-adjusted forecasting",
            "Causal inference analysis",
            "Scenario analysis engine",
            "Executive report generation"
        ],
        "endpoints": {
            "health": "/health/status",
            "forecasting": "/api/v1/forecast/",
            "analysis": "/api/v1/analysis/",
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