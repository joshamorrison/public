from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
import logging
import traceback
from datetime import datetime

logger = logging.getLogger(__name__)

class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        try:
            response = await call_next(request)
            return response
        except HTTPException as e:
            # Let FastAPI handle HTTP exceptions
            raise e
        except Exception as e:
            # Log the error
            logger.error(f"Unhandled error in {request.method} {request.url}: {str(e)}")
            logger.error(traceback.format_exc())
            
            # Return a generic error response
            return JSONResponse(
                status_code=500,
                content={
                    "error": "Internal server error",
                    "detail": "An unexpected error occurred",
                    "timestamp": datetime.now().isoformat()
                }
            )