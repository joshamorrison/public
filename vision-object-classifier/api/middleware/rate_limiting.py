"""
Rate limiting middleware for the Vision Object Classifier API
"""

import time
import asyncio
from typing import Dict, Any
from collections import defaultdict
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

class InMemoryRateLimiter:
    """
    Simple in-memory rate limiter using token bucket algorithm
    """
    
    def __init__(self):
        self.clients = defaultdict(dict)
        self.cleanup_interval = 300  # 5 minutes
        self.last_cleanup = time.time()
    
    def is_allowed(self, client_id: str, max_calls: int, window_seconds: int) -> tuple[bool, Dict[str, Any]]:
        """
        Check if client is allowed to make request
        
        Returns:
            (allowed, info) where info contains rate limit details
        """
        current_time = time.time()
        
        # Cleanup old entries periodically
        if current_time - self.last_cleanup > self.cleanup_interval:
            self._cleanup_old_entries(current_time)
        
        # Get or create client info
        if client_id not in self.clients:
            self.clients[client_id] = {
                'calls': 0,
                'window_start': current_time,
                'last_request': current_time
            }
        
        client_info = self.clients[client_id]
        
        # Check if we need to reset the window
        if current_time - client_info['window_start'] >= window_seconds:
            client_info['calls'] = 0
            client_info['window_start'] = current_time
        
        # Calculate rate limit info
        window_remaining = window_seconds - (current_time - client_info['window_start'])
        calls_remaining = max(0, max_calls - client_info['calls'])
        
        rate_limit_info = {
            'limit': max_calls,
            'remaining': calls_remaining,
            'reset_time': client_info['window_start'] + window_seconds,
            'window_seconds': window_seconds
        }
        
        # Check if request is allowed
        if client_info['calls'] >= max_calls:
            return False, rate_limit_info
        
        # Allow request and update counters
        client_info['calls'] += 1
        client_info['last_request'] = current_time
        
        rate_limit_info['remaining'] = max_calls - client_info['calls']
        
        return True, rate_limit_info
    
    def _cleanup_old_entries(self, current_time: float):
        """Remove old client entries to prevent memory leaks"""
        cutoff_time = current_time - 3600  # Remove entries older than 1 hour
        
        clients_to_remove = []
        for client_id, info in self.clients.items():
            if info['last_request'] < cutoff_time:
                clients_to_remove.append(client_id)
        
        for client_id in clients_to_remove:
            del self.clients[client_id]
        
        self.last_cleanup = current_time

# Global rate limiter instance
rate_limiter = InMemoryRateLimiter()

class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Rate limiting middleware
    """
    
    def __init__(self, app, calls: int = 100, period: int = 60, per_ip: bool = True):
        """
        Initialize rate limiting middleware
        
        Args:
            app: FastAPI application
            calls: Maximum number of calls per period
            period: Time period in seconds
            per_ip: Whether to limit per IP address (True) or globally (False)
        """
        super().__init__(app)
        self.calls = calls
        self.period = period
        self.per_ip = per_ip
    
    async def dispatch(self, request: Request, call_next):
        # Determine client identifier
        if self.per_ip:
            client_ip = request.client.host if request.client else "unknown"
            client_id = f"ip:{client_ip}"
        else:
            client_id = "global"
        
        # Check rate limit
        allowed, rate_info = rate_limiter.is_allowed(client_id, self.calls, self.period)
        
        if not allowed:
            # Rate limit exceeded
            error_response = {
                "success": False,
                "error": "Rate limit exceeded",
                "error_code": "RATE_LIMIT_EXCEEDED",
                "timestamp": time.time(),
                "rate_limit": {
                    "limit": rate_info['limit'],
                    "remaining": rate_info['remaining'],
                    "reset_time": rate_info['reset_time'],
                    "retry_after": int(rate_info['reset_time'] - time.time())
                }
            }
            
            headers = {
                "X-RateLimit-Limit": str(rate_info['limit']),
                "X-RateLimit-Remaining": str(rate_info['remaining']),
                "X-RateLimit-Reset": str(int(rate_info['reset_time'])),
                "Retry-After": str(int(rate_info['reset_time'] - time.time()))
            }
            
            return JSONResponse(
                status_code=429,
                content=error_response,
                headers=headers
            )
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers to response
        response.headers["X-RateLimit-Limit"] = str(rate_info['limit'])
        response.headers["X-RateLimit-Remaining"] = str(rate_info['remaining'])
        response.headers["X-RateLimit-Reset"] = str(int(rate_info['reset_time']))
        
        return response

def get_client_identifier(request: Request, use_api_key: bool = False) -> str:
    """
    Get client identifier for rate limiting
    
    Args:
        request: FastAPI request object
        use_api_key: Whether to use API key for identification
    
    Returns:
        Client identifier string
    """
    if use_api_key:
        # Check for API key in headers
        api_key = request.headers.get("X-API-Key") or request.headers.get("Authorization")
        if api_key:
            # Use hash of API key for privacy
            import hashlib
            return f"api_key:{hashlib.sha256(api_key.encode()).hexdigest()[:16]}"
    
    # Fall back to IP address
    client_ip = request.client.host if request.client else "unknown"
    return f"ip:{client_ip}"

class AdvancedRateLimitMiddleware(BaseHTTPMiddleware):
    """
    Advanced rate limiting with different limits for different endpoints
    """
    
    def __init__(self, app, endpoint_limits: Dict[str, Dict[str, Any]] = None):
        """
        Initialize advanced rate limiting
        
        Args:
            app: FastAPI application
            endpoint_limits: Dict mapping endpoint patterns to rate limit configs
        """
        super().__init__(app)
        self.endpoint_limits = endpoint_limits or {
            "/api/v1/classify/single": {"calls": 50, "period": 60},
            "/api/v1/batch/classify": {"calls": 10, "period": 60},
            "default": {"calls": 100, "period": 60}
        }
    
    def get_endpoint_config(self, path: str) -> Dict[str, Any]:
        """Get rate limit config for specific endpoint"""
        for pattern, config in self.endpoint_limits.items():
            if pattern != "default" and pattern in path:
                return config
        return self.endpoint_limits.get("default", {"calls": 100, "period": 60})
    
    async def dispatch(self, request: Request, call_next):
        # Get endpoint-specific configuration
        config = self.get_endpoint_config(request.url.path)
        
        # Get client identifier
        client_id = get_client_identifier(request, use_api_key=False)
        
        # Create endpoint-specific client ID
        endpoint_client_id = f"{client_id}:{request.url.path}"
        
        # Check rate limit
        allowed, rate_info = rate_limiter.is_allowed(
            endpoint_client_id, 
            config["calls"], 
            config["period"]
        )
        
        if not allowed:
            # Rate limit exceeded
            error_response = {
                "success": False,
                "error": f"Rate limit exceeded for endpoint {request.url.path}",
                "error_code": "ENDPOINT_RATE_LIMIT_EXCEEDED",
                "timestamp": time.time(),
                "endpoint": request.url.path,
                "rate_limit": {
                    "limit": rate_info['limit'],
                    "remaining": rate_info['remaining'],
                    "reset_time": rate_info['reset_time'],
                    "retry_after": int(rate_info['reset_time'] - time.time())
                }
            }
            
            headers = {
                "X-RateLimit-Limit": str(rate_info['limit']),
                "X-RateLimit-Remaining": str(rate_info['remaining']),
                "X-RateLimit-Reset": str(int(rate_info['reset_time'])),
                "X-RateLimit-Endpoint": request.url.path,
                "Retry-After": str(int(rate_info['reset_time'] - time.time()))
            }
            
            return JSONResponse(
                status_code=429,
                content=error_response,
                headers=headers
            )
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers
        response.headers["X-RateLimit-Limit"] = str(rate_info['limit'])
        response.headers["X-RateLimit-Remaining"] = str(rate_info['remaining'])
        response.headers["X-RateLimit-Reset"] = str(int(rate_info['reset_time']))
        response.headers["X-RateLimit-Endpoint"] = request.url.path
        
        return response