"""
Error handling middleware for the Vision Object Classifier API
"""

import time
import uuid
import logging
from typing import Dict, Any
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """
    Middleware to handle errors and provide consistent error responses
    """
    
    async def dispatch(self, request: Request, call_next):
        # Generate request ID for tracking
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        start_time = time.time()
        
        try:
            response = await call_next(request)
            
            # Log successful requests
            processing_time = time.time() - start_time
            logger.info(
                f"Request {request_id} - {request.method} {request.url.path} "
                f"- Status: {response.status_code} - Time: {processing_time:.3f}s"
            )
            
            # Add request ID to response headers
            response.headers["X-Request-ID"] = request_id
            
            return response
            
        except HTTPException as exc:
            # Handle FastAPI HTTP exceptions
            processing_time = time.time() - start_time
            
            logger.warning(
                f"Request {request_id} - HTTPException - "
                f"Status: {exc.status_code} - Detail: {exc.detail} - "
                f"Time: {processing_time:.3f}s"
            )
            
            error_response = {
                "success": False,
                "error": exc.detail,
                "error_code": f"HTTP_{exc.status_code}",
                "timestamp": time.time(),
                "request_id": request_id
            }
            
            return JSONResponse(
                status_code=exc.status_code,
                content=error_response,
                headers={"X-Request-ID": request_id}
            )
            
        except Exception as exc:
            # Handle unexpected exceptions
            processing_time = time.time() - start_time
            
            logger.error(
                f"Request {request_id} - Unhandled Exception - "
                f"Type: {type(exc).__name__} - Message: {str(exc)} - "
                f"Time: {processing_time:.3f}s",
                exc_info=True
            )
            
            # Don't expose internal error details in production
            error_message = "Internal server error occurred"
            
            error_response = {
                "success": False,
                "error": error_message,
                "error_code": "INTERNAL_ERROR",
                "timestamp": time.time(),
                "request_id": request_id
            }
            
            return JSONResponse(
                status_code=500,
                content=error_response,
                headers={"X-Request-ID": request_id}
            )

async def error_handler(request: Request, call_next):
    """
    Simple error handler function (alternative to middleware class)
    """
    request_id = getattr(request.state, 'request_id', str(uuid.uuid4()))
    start_time = time.time()
    
    try:
        response = await call_next(request)
        
        # Add processing time header
        processing_time = time.time() - start_time
        response.headers["X-Processing-Time"] = f"{processing_time:.3f}"
        response.headers["X-Request-ID"] = request_id
        
        return response
        
    except Exception as exc:
        processing_time = time.time() - start_time
        
        # Log the error
        logger.error(
            f"Request {request_id} failed after {processing_time:.3f}s: {str(exc)}",
            exc_info=True
        )
        
        # Return standardized error response
        if isinstance(exc, HTTPException):
            status_code = exc.status_code
            error_message = exc.detail
            error_code = f"HTTP_{exc.status_code}"
        else:
            status_code = 500
            error_message = "Internal server error"
            error_code = "INTERNAL_ERROR"
        
        error_response = {
            "success": False,
            "error": error_message,
            "error_code": error_code,
            "timestamp": time.time(),
            "request_id": request_id,
            "processing_time_ms": processing_time * 1000
        }
        
        return JSONResponse(
            status_code=status_code,
            content=error_response,
            headers={
                "X-Request-ID": request_id,
                "X-Processing-Time": f"{processing_time:.3f}"
            }
        )

def get_error_response(
    error_message: str,
    error_code: str = "UNKNOWN_ERROR",
    request_id: str = None,
    status_code: int = 500
) -> Dict[str, Any]:
    """
    Generate standardized error response
    """
    return {
        "success": False,
        "error": error_message,
        "error_code": error_code,
        "timestamp": time.time(),
        "request_id": request_id or str(uuid.uuid4())
    }

def log_error(
    request_id: str,
    error: Exception,
    context: Dict[str, Any] = None
):
    """
    Log error with context
    """
    context_str = ""
    if context:
        context_items = [f"{k}={v}" for k, v in context.items()]
        context_str = f" - Context: {', '.join(context_items)}"
    
    logger.error(
        f"Request {request_id} - Error: {type(error).__name__}: {str(error)}{context_str}",
        exc_info=True
    )