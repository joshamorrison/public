"""
Rate Limiting Middleware

Simple in-memory rate limiting middleware for the FastAPI application.
In production, use Redis-based rate limiting for distributed systems.
"""

import time
from collections import defaultdict, deque
from typing import Dict, Tuple
from datetime import datetime, timedelta

from fastapi import Request, Response, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware


class RateLimitingMiddleware(BaseHTTPMiddleware):
    """
    Token bucket rate limiting middleware.
    
    Features:
    - Per-IP rate limiting
    - Different limits for different endpoints
    - Sliding window rate limiting
    - Rate limit headers in responses
    """
    
    def __init__(
        self, 
        app,
        requests_per_minute: int = 60,
        requests_per_hour: int = 1000,
        burst_size: int = 10
    ):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.requests_per_hour = requests_per_hour 
        self.burst_size = burst_size
        
        # In-memory storage (use Redis in production)
        self.request_times: Dict[str, deque] = defaultdict(deque)
        self.burst_tokens: Dict[str, Tuple[int, float]] = {}
    
    async def dispatch(self, request: Request, call_next):
        """Apply rate limiting to requests."""
        client_ip = self._get_client_ip(request)
        current_time = time.time()
        
        # Check rate limits
        if not self._check_rate_limit(client_ip, current_time):
            return self._create_rate_limit_response(client_ip)
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers
        self._add_rate_limit_headers(response, client_ip, current_time)
        
        return response
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP address from request."""
        # Check for forwarded headers (when behind proxy)
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(',')[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
            
        # Fallback to direct connection IP
        return request.client.host if request.client else "unknown"
    
    def _check_rate_limit(self, client_ip: str, current_time: float) -> bool:
        """Check if request should be rate limited."""
        # Clean old requests outside the time window
        self._cleanup_old_requests(client_ip, current_time)
        
        # Check hourly limit
        hourly_count = len([
            req_time for req_time in self.request_times[client_ip]
            if current_time - req_time <= 3600  # 1 hour
        ])
        
        if hourly_count >= self.requests_per_hour:
            return False
        
        # Check per-minute limit
        minute_count = len([
            req_time for req_time in self.request_times[client_ip] 
            if current_time - req_time <= 60  # 1 minute
        ])
        
        if minute_count >= self.requests_per_minute:
            # Check if we have burst tokens available
            if not self._consume_burst_token(client_ip, current_time):
                return False
        
        # Record this request
        self.request_times[client_ip].append(current_time)
        return True
    
    def _consume_burst_token(self, client_ip: str, current_time: float) -> bool:
        """Try to consume a burst token."""
        if client_ip not in self.burst_tokens:
            self.burst_tokens[client_ip] = (self.burst_size, current_time)
        
        tokens, last_refill = self.burst_tokens[client_ip]
        
        # Refill tokens (1 token per minute)
        time_passed = current_time - last_refill
        tokens_to_add = int(time_passed / 60)  # 1 token per minute
        tokens = min(self.burst_size, tokens + tokens_to_add)
        
        if tokens > 0:
            tokens -= 1
            self.burst_tokens[client_ip] = (tokens, current_time)
            return True
        
        return False
    
    def _cleanup_old_requests(self, client_ip: str, current_time: float):
        """Remove old request timestamps outside the tracking window."""
        if client_ip in self.request_times:
            # Keep only requests from last hour
            while (self.request_times[client_ip] and 
                   current_time - self.request_times[client_ip][0] > 3600):
                self.request_times[client_ip].popleft()
    
    def _add_rate_limit_headers(self, response: Response, client_ip: str, current_time: float):
        """Add rate limiting headers to response."""
        # Calculate remaining requests
        minute_count = len([
            req_time for req_time in self.request_times[client_ip]
            if current_time - req_time <= 60
        ])
        
        hourly_count = len([
            req_time for req_time in self.request_times[client_ip]
            if current_time - req_time <= 3600
        ])
        
        # Get burst tokens
        tokens, _ = self.burst_tokens.get(client_ip, (self.burst_size, current_time))
        
        # Add headers
        response.headers["X-RateLimit-Limit-Minute"] = str(self.requests_per_minute)
        response.headers["X-RateLimit-Remaining-Minute"] = str(max(0, self.requests_per_minute - minute_count))
        response.headers["X-RateLimit-Limit-Hour"] = str(self.requests_per_hour)
        response.headers["X-RateLimit-Remaining-Hour"] = str(max(0, self.requests_per_hour - hourly_count))
        response.headers["X-RateLimit-Burst-Tokens"] = str(tokens)
        
        # Reset times
        next_minute_reset = int(current_time / 60 + 1) * 60
        next_hour_reset = int(current_time / 3600 + 1) * 3600
        response.headers["X-RateLimit-Reset-Minute"] = str(next_minute_reset)
        response.headers["X-RateLimit-Reset-Hour"] = str(next_hour_reset)
    
    def _create_rate_limit_response(self, client_ip: str) -> JSONResponse:
        """Create rate limit exceeded response."""
        # Calculate retry after time
        current_time = time.time()
        minute_requests = [
            req_time for req_time in self.request_times[client_ip]
            if current_time - req_time <= 60
        ]
        
        if minute_requests:
            oldest_in_minute = min(minute_requests)
            retry_after = int(60 - (current_time - oldest_in_minute)) + 1
        else:
            retry_after = 60
        
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content={
                "error": "RateLimitExceeded",
                "detail": "Too many requests. Please slow down.",
                "retry_after": retry_after,
                "timestamp": datetime.now().isoformat(),
                "limits": {
                    "per_minute": self.requests_per_minute,
                    "per_hour": self.requests_per_hour,
                    "burst_size": self.burst_size
                }
            },
            headers={
                "Retry-After": str(retry_after),
                "X-RateLimit-Limit-Minute": str(self.requests_per_minute),
                "X-RateLimit-Limit-Hour": str(self.requests_per_hour)
            }
        )


class EndpointSpecificRateLimitingMiddleware(RateLimitingMiddleware):
    """
    Rate limiting with different limits for different endpoints.
    """
    
    def __init__(self, app):
        super().__init__(app)
        
        # Define endpoint-specific limits
        self.endpoint_limits = {
            "/api/v1/workflows/execute": {
                "requests_per_minute": 10,
                "requests_per_hour": 100,
                "burst_size": 3
            },
            "/api/v1/workflows/bulk": {
                "requests_per_minute": 5,
                "requests_per_hour": 50,
                "burst_size": 2
            },
            "/health": {
                "requests_per_minute": 300,
                "requests_per_hour": 3600,
                "burst_size": 50
            },
            "/status": {
                "requests_per_minute": 60,
                "requests_per_hour": 600,
                "burst_size": 10
            }
        }
    
    def _get_endpoint_limits(self, path: str) -> Dict[str, int]:
        """Get rate limits for specific endpoint."""
        for endpoint_pattern, limits in self.endpoint_limits.items():
            if path.startswith(endpoint_pattern):
                return limits
        
        # Default limits
        return {
            "requests_per_minute": self.requests_per_minute,
            "requests_per_hour": self.requests_per_hour,
            "burst_size": self.burst_size
        }
    
    async def dispatch(self, request: Request, call_next):
        """Apply endpoint-specific rate limiting."""
        limits = self._get_endpoint_limits(request.url.path)
        
        # Temporarily override instance limits
        original_limits = (self.requests_per_minute, self.requests_per_hour, self.burst_size)
        self.requests_per_minute = limits["requests_per_minute"]
        self.requests_per_hour = limits["requests_per_hour"] 
        self.burst_size = limits["burst_size"]
        
        try:
            result = await super().dispatch(request, call_next)
            return result
        finally:
            # Restore original limits
            self.requests_per_minute, self.requests_per_hour, self.burst_size = original_limits