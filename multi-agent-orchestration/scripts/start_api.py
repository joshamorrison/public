#!/usr/bin/env python3
"""
API Startup Script

Properly configured startup script for the FastAPI application.
"""

import sys
import os

# Add project root and src to path for imports
project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, project_root)  # For api module
sys.path.insert(0, os.path.join(project_root, 'src'))  # For src modules

if __name__ == "__main__":
    print("[API] Starting Multi-Agent Orchestration Platform API...")
    print("[API] Setting up Python path...")
    
    # Import and start the API
    from api.main import app
    import uvicorn
    
    print("[API] FastAPI app loaded successfully")
    print("[API] Starting server...")
    print("[API] Visit http://localhost:8000/docs for interactive API documentation")
    print("[API] Visit http://localhost:8000/health for health check")
    print("[API] Visit http://localhost:8000/status for platform status")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=False,  # Disable reload to avoid path issues
        log_level="info",
        access_log=True
    )