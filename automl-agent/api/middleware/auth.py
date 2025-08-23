from fastapi import HTTPException, Security, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from passlib.context import CryptContext
from jose import JWTError, jwt
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import secrets
import hashlib
import time
from functools import wraps
import asyncio
from collections import defaultdict, deque

# Configuration
SECRET_KEY = secrets.token_urlsafe(32)  # In production, use environment variable
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Security instances
security = HTTPBearer(auto_error=False)
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Rate limiting storage
rate_limit_storage: Dict[str, deque] = defaultdict(deque)

# Simple in-memory user store (replace with database in production)
users_db = {
    "admin": {
        "username": "admin",
        "email": "admin@automl.com",
        "hashed_password": pwd_context.hash("admin123"),
        "permissions": ["read", "write", "admin"],
        "api_quota": 1000,
        "api_usage": 0
    },
    "user": {
        "username": "user",
        "email": "user@automl.com", 
        "hashed_password": pwd_context.hash("user123"),
        "permissions": ["read", "write"],
        "api_quota": 100,
        "api_usage": 0
    },
    "readonly": {
        "username": "readonly",
        "email": "readonly@automl.com",
        "hashed_password": pwd_context.hash("readonly123"), 
        "permissions": ["read"],
        "api_quota": 50,
        "api_usage": 0
    }
}

# API Keys storage (replace with database in production)
api_keys_db = {
    "ak_demo_12345": {
        "key_id": "ak_demo_12345",
        "user": "demo_user",
        "permissions": ["read", "write"],
        "quota": 500,
        "usage": 0,
        "created_at": datetime.now(),
        "last_used": None,
        "active": True
    }
}

class RateLimiter:
    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
    
    def is_allowed(self, identifier: str) -> bool:
        now = time.time()
        window_start = now - self.window_seconds
        
        # Clean old requests
        requests = rate_limit_storage[identifier]
        while requests and requests[0] < window_start:
            requests.popleft()
        
        # Check rate limit
        if len(requests) >= self.max_requests:
            return False
        
        # Add current request
        requests.append(now)
        return True
    
    def get_remaining(self, identifier: str) -> int:
        now = time.time()
        window_start = now - self.window_seconds
        
        requests = rate_limit_storage[identifier]
        # Count requests in current window
        current_requests = sum(1 for req_time in requests if req_time >= window_start)
        return max(0, self.max_requests - current_requests)

# Rate limiters for different tiers
rate_limiters = {
    "default": RateLimiter(max_requests=60, window_seconds=60),  # 60 requests per minute
    "premium": RateLimiter(max_requests=300, window_seconds=60),  # 300 requests per minute
    "admin": RateLimiter(max_requests=1000, window_seconds=60)   # 1000 requests per minute
}

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

def authenticate_user(username: str, password: str) -> Optional[Dict[str, Any]]:
    user = users_db.get(username)
    if not user or not verify_password(password, user["hashed_password"]):
        return None
    return user

def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(token: str) -> Optional[Dict[str, Any]]:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            return None
        return payload
    except JWTError:
        return None

def verify_api_key(api_key: str) -> Optional[Dict[str, Any]]:
    key_data = api_keys_db.get(api_key)
    if not key_data or not key_data["active"]:
        return None
    
    # Update last used
    key_data["last_used"] = datetime.now()
    return key_data

async def get_current_user(credentials: Optional[HTTPAuthorizationCredentials] = Security(security)):
    if not credentials:
        return None
    
    # Try JWT token first
    if credentials.scheme.lower() == "bearer":
        payload = verify_token(credentials.credentials)
        if payload:
            username = payload.get("sub")
            user = users_db.get(username)
            if user:
                return {**user, "auth_type": "jwt"}
    
    # Try API key
    api_key_data = verify_api_key(credentials.credentials)
    if api_key_data:
        return {**api_key_data, "auth_type": "api_key"}
    
    return None

async def require_auth(current_user: Optional[Dict[str, Any]] = Depends(get_current_user)):
    if not current_user:
        raise HTTPException(
            status_code=401,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return current_user

def require_permissions(*required_permissions: str):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract current_user from kwargs or get it
            current_user = kwargs.get('current_user')
            if not current_user:
                current_user = await require_auth()
            
            user_permissions = current_user.get("permissions", [])
            
            # Check if user has required permissions
            if not any(perm in user_permissions for perm in required_permissions):
                raise HTTPException(
                    status_code=403,
                    detail=f"Insufficient permissions. Required: {required_permissions}"
                )
            
            return await func(*args, **kwargs)
        return wrapper
    return decorator

def rate_limit(tier: str = "default"):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract current_user and request info
            current_user = kwargs.get('current_user')
            if not current_user:
                current_user = await require_auth()
            
            # Determine rate limit tier
            if "admin" in current_user.get("permissions", []):
                limiter = rate_limiters["admin"]
            elif current_user.get("auth_type") == "api_key":
                limiter = rate_limiters["premium"]
            else:
                limiter = rate_limiters[tier]
            
            # Create identifier
            identifier = current_user.get("username") or current_user.get("key_id", "unknown")
            
            # Check rate limit
            if not limiter.is_allowed(identifier):
                remaining = limiter.get_remaining(identifier)
                raise HTTPException(
                    status_code=429,
                    detail="Rate limit exceeded",
                    headers={
                        "X-RateLimit-Remaining": str(remaining),
                        "X-RateLimit-Reset": str(int(time.time()) + limiter.window_seconds)
                    }
                )
            
            # Add rate limit headers to response
            remaining = limiter.get_remaining(identifier)
            # Note: In a real FastAPI app, you'd add these headers to the response
            
            return await func(*args, **kwargs)
        return wrapper
    return decorator

def check_api_quota(current_user: Dict[str, Any]) -> bool:
    """Check if user has remaining API quota"""
    if current_user.get("auth_type") == "jwt":
        user = users_db.get(current_user["username"])
        if user and user["api_usage"] >= user["api_quota"]:
            return False
    elif current_user.get("auth_type") == "api_key":
        key_data = api_keys_db.get(current_user["key_id"])
        if key_data and key_data["usage"] >= key_data["quota"]:
            return False
    
    return True

def increment_api_usage(current_user: Dict[str, Any]):
    """Increment API usage counter"""
    if current_user.get("auth_type") == "jwt":
        username = current_user["username"]
        if username in users_db:
            users_db[username]["api_usage"] += 1
    elif current_user.get("auth_type") == "api_key":
        key_id = current_user["key_id"]
        if key_id in api_keys_db:
            api_keys_db[key_id]["usage"] += 1

def api_quota_check():
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            current_user = kwargs.get('current_user')
            if not current_user:
                current_user = await require_auth()
            
            # Check quota
            if not check_api_quota(current_user):
                raise HTTPException(
                    status_code=429,
                    detail="API quota exceeded"
                )
            
            # Execute function
            result = await func(*args, **kwargs)
            
            # Increment usage after successful execution
            increment_api_usage(current_user)
            
            return result
        return wrapper
    return decorator

# Security middleware for FastAPI
async def security_middleware(request, call_next):
    # Add security headers
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    return response

# Utility functions for user management
def create_user(username: str, email: str, password: str, permissions: list = None) -> Dict[str, Any]:
    if username in users_db:
        raise ValueError("User already exists")
    
    permissions = permissions or ["read"]
    hashed_password = get_password_hash(password)
    
    user = {
        "username": username,
        "email": email,
        "hashed_password": hashed_password,
        "permissions": permissions,
        "api_quota": 100,
        "api_usage": 0
    }
    
    users_db[username] = user
    return user

def create_api_key(user: str, permissions: list = None, quota: int = 500) -> str:
    permissions = permissions or ["read"]
    key_id = f"ak_{secrets.token_urlsafe(16)}"
    
    api_keys_db[key_id] = {
        "key_id": key_id,
        "user": user,
        "permissions": permissions,
        "quota": quota,
        "usage": 0,
        "created_at": datetime.now(),
        "last_used": None,
        "active": True
    }
    
    return key_id

def revoke_api_key(key_id: str) -> bool:
    if key_id in api_keys_db:
        api_keys_db[key_id]["active"] = False
        return True
    return False