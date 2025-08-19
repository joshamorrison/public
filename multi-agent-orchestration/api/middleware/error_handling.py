"""
Error Handling Middleware

Comprehensive error handling middleware for the FastAPI application.
Provides structured error responses, logging, and error tracking.
"""

import time
import uuid
from typing import Dict, Any
import traceback
from datetime import datetime

from fastapi import Request, Response, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """
    Comprehensive error handling middleware.
    
    Features:
    - Structured error responses
    - Request ID tracking for debugging
    - Error logging and monitoring
    - Performance timing
    - Graceful error handling for different exception types
    """
    
    async def dispatch(self, request: Request, call_next):
        """Process request and handle any errors that occur."""
        request_id = str(uuid.uuid4())[:8]
        start_time = time.time()
        
        # Add request ID to request state
        request.state.request_id = request_id
        
        try:
            # Add request timing
            response = await call_next(request)
            
            # Add request ID and timing headers
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Response-Time"] = f"{(time.time() - start_time):.3f}s"
            
            return response
            
        except Exception as exc:
            # Log the error (in production, use proper logging)
            error_details = {
                "request_id": request_id,
                "method": request.method,
                "url": str(request.url),
                "timestamp": datetime.now().isoformat(),
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "traceback": traceback.format_exc()
            }
            
            print(f"[ERROR] Request {request_id} failed: {error_details}")
            
            # Determine appropriate error response
            if isinstance(exc, ValueError):
                return self._create_error_response(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    error_type="ValidationError",
                    message=str(exc),
                    request_id=request_id
                )
            elif isinstance(exc, FileNotFoundError):
                return self._create_error_response(
                    status_code=status.HTTP_404_NOT_FOUND,
                    error_type="NotFoundError",
                    message="Requested resource not found",
                    request_id=request_id
                )
            elif isinstance(exc, PermissionError):
                return self._create_error_response(
                    status_code=status.HTTP_403_FORBIDDEN,
                    error_type="PermissionError", 
                    message="Insufficient permissions",
                    request_id=request_id
                )
            elif isinstance(exc, TimeoutError):
                return self._create_error_response(
                    status_code=status.HTTP_408_REQUEST_TIMEOUT,
                    error_type="TimeoutError",
                    message="Request timeout",
                    request_id=request_id
                )
            else:
                # Generic server error
                return self._create_error_response(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    error_type="InternalServerError",
                    message="An unexpected error occurred",
                    request_id=request_id,
                    include_traceback=False  # Don't expose internal details
                )
    
    def _create_error_response(
        self, 
        status_code: int, 
        error_type: str, 
        message: str, 
        request_id: str,
        additional_details: Dict[str, Any] = None,
        include_traceback: bool = False
    ) -> JSONResponse:
        """Create a structured error response."""
        
        error_content = {
            "error": error_type,
            "detail": message,
            "timestamp": datetime.now().isoformat(),
            "request_id": request_id,
            "status_code": status_code
        }
        
        if additional_details:
            error_content.update(additional_details)
        
        # Include traceback for development (configure based on environment)
        if include_traceback:
            error_content["traceback"] = traceback.format_exc()
        
        # Add helpful links for common errors
        if status_code == status.HTTP_400_BAD_REQUEST:
            error_content["help"] = "Check the request format and required parameters"
        elif status_code == status.HTTP_404_NOT_FOUND:
            error_content["help"] = "Verify the requested resource exists and the URL is correct"
        elif status_code == status.HTTP_500_INTERNAL_SERVER_ERROR:
            error_content["help"] = "Please try again or contact support with the request ID"
        
        return JSONResponse(
            status_code=status_code,
            content=error_content,
            headers={
                "X-Request-ID": request_id,
                "X-Error-Type": error_type
            }
        )


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Request logging middleware for monitoring and debugging.
    """
    
    async def dispatch(self, request: Request, call_next):
        """Log request and response information."""
        start_time = time.time()
        request_id = getattr(request.state, 'request_id', 'unknown')
        
        # Log request
        print(f"[REQUEST] {request_id} - {request.method} {request.url}")
        
        try:
            response = await call_next(request)
            
            # Log response
            duration = time.time() - start_time
            print(f"[RESPONSE] {request_id} - {response.status_code} - {duration:.3f}s")
            
            return response
            
        except Exception as exc:
            # Log error
            duration = time.time() - start_time
            print(f"[ERROR] {request_id} - {type(exc).__name__}: {str(exc)} - {duration:.3f}s")
            raise


class CORSSecurityMiddleware(BaseHTTPMiddleware):
    """
    Enhanced CORS middleware with security headers.
    """
    
    def __init__(self, app, allowed_origins: list = None):
        super().__init__(app)
        self.allowed_origins = allowed_origins or ["*"]
    
    async def dispatch(self, request: Request, call_next):
        """Add security headers to responses."""
        response = await call_next(request)
        
        # Security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        
        # API-specific headers
        response.headers["X-API-Version"] = "1.0.0"
        response.headers["X-Powered-By"] = "Multi-Agent Orchestration Platform"
        
        return response