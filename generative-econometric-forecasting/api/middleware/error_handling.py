"""
Error handling middleware for the Econometric Forecasting API.

Provides centralized error handling, logging, and standardized error responses.
"""

import logging
import traceback
import uuid
from typing import Any, Dict
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
import time

logger = logging.getLogger(__name__)

class APIError(Exception):
    """Custom API error with structured information."""
    
    def __init__(self, message: str, error_code: str = "API_ERROR", details: Dict[str, Any] = None):
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        super().__init__(self.message)

class ForecastingError(APIError):
    """Error in forecasting operations."""
    
    def __init__(self, message: str, details: Dict[str, Any] = None):
        super().__init__(message, "FORECASTING_ERROR", details)

class DataError(APIError):
    """Error in data operations."""
    
    def __init__(self, message: str, details: Dict[str, Any] = None):
        super().__init__(message, "DATA_ERROR", details)

async def error_handler_middleware(request: Request, call_next):
    """
    Middleware for centralized error handling and request logging.
    
    Args:
        request: The incoming HTTP request
        call_next: The next middleware or route handler
    
    Returns:
        HTTP response with error handling
    """
    # Generate request ID for tracking
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    
    start_time = time.time()
    
    try:
        # Log incoming request
        logger.info(
            f"Request {request_id}: {request.method} {request.url.path}",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "query_params": dict(request.query_params)
            }
        )
        
        # Process request
        response = await call_next(request)
        
        # Log successful response
        duration = time.time() - start_time
        logger.info(
            f"Request {request_id} completed in {duration:.3f}s with status {response.status_code}",
            extra={
                "request_id": request_id,
                "status_code": response.status_code,
                "duration": duration
            }
        )
        
        return response
        
    except HTTPException as e:
        # Handle FastAPI HTTPExceptions
        duration = time.time() - start_time
        logger.warning(
            f"Request {request_id} failed with HTTP {e.status_code}: {e.detail}",
            extra={
                "request_id": request_id,
                "status_code": e.status_code,
                "error": e.detail,
                "duration": duration
            }
        )
        
        return JSONResponse(
            status_code=e.status_code,
            content={
                "success": False,
                "error": {
                    "code": f"HTTP_{e.status_code}",
                    "message": e.detail,
                    "request_id": request_id
                },
                "data": None
            }
        )
        
    except APIError as e:
        # Handle custom API errors
        duration = time.time() - start_time
        logger.error(
            f"Request {request_id} failed with API error: {e.message}",
            extra={
                "request_id": request_id,
                "error_code": e.error_code,
                "error_message": e.message,
                "error_details": e.details,
                "duration": duration
            }
        )
        
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "error": {
                    "code": e.error_code,
                    "message": e.message,
                    "details": e.details,
                    "request_id": request_id
                },
                "data": None
            }
        )
        
    except Exception as e:
        # Handle unexpected errors
        duration = time.time() - start_time
        error_trace = traceback.format_exc()
        
        logger.error(
            f"Request {request_id} failed with unexpected error: {str(e)}",
            extra={
                "request_id": request_id,
                "error_type": type(e).__name__,
                "error_message": str(e),
                "traceback": error_trace,
                "duration": duration
            }
        )
        
        # Don't expose internal error details in production
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": {
                    "code": "INTERNAL_SERVER_ERROR",
                    "message": "An unexpected error occurred. Please try again later.",
                    "request_id": request_id
                },
                "data": None
            }
        )