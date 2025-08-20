"""
Rate limiting middleware for the Econometric Forecasting API.

Implements token bucket algorithm for rate limiting API requests.
"""

import time
import logging
from typing import Dict, Optional
from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

logger = logging.getLogger(__name__)

class TokenBucket:
    """Token bucket implementation for rate limiting."""
    
    def __init__(self, max_tokens: int, refill_rate: float):
        """
        Initialize token bucket.
        
        Args:
            max_tokens: Maximum number of tokens in bucket
            refill_rate: Tokens added per second
        """
        self.max_tokens = max_tokens
        self.refill_rate = refill_rate
        self.tokens = max_tokens
        self.last_refill = time.time()
    
    def consume(self, tokens: int = 1) -> bool:
        """
        Attempt to consume tokens from bucket.
        
        Args:
            tokens: Number of tokens to consume
            
        Returns:
            True if tokens were consumed, False otherwise
        """
        now = time.time()
        time_passed = now - self.last_refill
        
        # Refill tokens based on time passed
        self.tokens = min(
            self.max_tokens,
            self.tokens + (time_passed * self.refill_rate)
        )
        self.last_refill = now
        
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False

class InMemoryRateLimiter:
    """In-memory rate limiter using token buckets."""
    
    def __init__(self):
        self.buckets: Dict[str, TokenBucket] = {}
        self.cleanup_interval = 3600  # Clean up old buckets every hour
        self.last_cleanup = time.time()
    
    def is_allowed(self, client_id: str, max_calls: int, window_seconds: int) -> bool:
        """
        Check if client is allowed to make request.
        
        Args:
            client_id: Unique identifier for client
            max_calls: Maximum calls allowed
            window_seconds: Time window in seconds
            
        Returns:
            True if request is allowed, False otherwise
        """
        # Cleanup old buckets periodically
        if time.time() - self.last_cleanup > self.cleanup_interval:
            self._cleanup_old_buckets()
        
        # Get or create bucket for client
        if client_id not in self.buckets:
            refill_rate = max_calls / window_seconds
            self.buckets[client_id] = TokenBucket(max_calls, refill_rate)
        
        return self.buckets[client_id].consume(1)
    
    def _cleanup_old_buckets(self):
        """Remove old, unused buckets to prevent memory leaks."""
        current_time = time.time()
        old_buckets = []
        
        for client_id, bucket in self.buckets.items():
            # Remove buckets that haven't been accessed in the last hour
            if current_time - bucket.last_refill > 3600:
                old_buckets.append(client_id)
        
        for client_id in old_buckets:
            del self.buckets[client_id]
        
        self.last_cleanup = current_time
        
        if old_buckets:
            logger.info(f"Cleaned up {len(old_buckets)} old rate limit buckets")

class RateLimitingMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware for rate limiting."""
    
    def __init__(self):
        """Initialize rate limiting middleware."""
        self.limiter = InMemoryRateLimiter()
        
        # Rate limit configuration
        self.rate_limits = {
            "/api/v1/forecast": {"max_calls": 10, "window_seconds": 60},  # 10 requests per minute
            "/api/v1/analysis": {"max_calls": 5, "window_seconds": 60},   # 5 requests per minute
            "default": {"max_calls": 100, "window_seconds": 60}           # 100 requests per minute
        }
    
    async def dispatch(self, request: Request, call_next) -> Response:
        """
        Process request with rate limiting.
        
        Args:
            request: The incoming HTTP request
            call_next: Next middleware or route handler
            
        Returns:
            HTTP response
        """
        # Skip rate limiting for health checks and documentation
        if request.url.path.startswith(("/health", "/docs", "/redoc", "/openapi.json")):
            return await call_next(request)
        
        # Get client identifier (IP address)
        client_ip = self._get_client_ip(request)
        
        # Determine rate limit based on endpoint
        rate_limit = self._get_rate_limit(request.url.path)
        
        # Check if request is allowed
        if not self.limiter.is_allowed(
            client_ip, 
            rate_limit["max_calls"], 
            rate_limit["window_seconds"]
        ):
            logger.warning(
                f"Rate limit exceeded for {client_ip} on {request.url.path}",
                extra={
                    "client_ip": client_ip,
                    "path": request.url.path,
                    "rate_limit": rate_limit
                }
            )
            
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded. Maximum {rate_limit['max_calls']} requests per {rate_limit['window_seconds']} seconds."
            )
        
        return await call_next(request)
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP address from request."""
        # Check for forwarded headers (when behind proxy)
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        # Fallback to direct connection IP
        if hasattr(request.client, "host"):
            return request.client.host
        
        return "unknown"
    
    def _get_rate_limit(self, path: str) -> Dict[str, int]:
        """Get rate limit configuration for path."""
        # Check for exact matches first
        if path in self.rate_limits:
            return self.rate_limits[path]
        
        # Check for prefix matches
        for endpoint, config in self.rate_limits.items():
            if endpoint != "default" and path.startswith(endpoint):
                return config
        
        # Return default rate limit
        return self.rate_limits["default"]