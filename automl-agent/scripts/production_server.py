#!/usr/bin/env python3
"""
Production Server Script for AutoML Agent Platform

Starts the production-ready AutoML platform with:
- FastAPI backend with SageMaker Autopilot integration
- Streamlit web interface
- Monitoring and logging
- Health checks and graceful shutdown
"""

import os
import sys
import asyncio
import signal
import uvicorn
import subprocess
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import argparse
import logging

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

class AutoMLProductionServer:
    """Production server manager for AutoML platform"""
    
    def __init__(self, 
                 api_host: str = "0.0.0.0", 
                 api_port: int = 8000,
                 streamlit_port: int = 8501,
                 environment: str = "production"):
        
        self.api_host = api_host
        self.api_port = api_port
        self.streamlit_port = streamlit_port
        self.environment = environment
        
        # Process tracking
        self.api_process = None
        self.streamlit_process = None
        self.executor = ProcessPoolExecutor(max_workers=2)
        
        # Setup logging
        self.setup_logging()
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        self.logger.info(f"AutoML Production Server initialized")
        self.logger.info(f"Environment: {environment}")
        self.logger.info(f"API: {api_host}:{api_port}")
        self.logger.info(f"Streamlit: localhost:{streamlit_port}")
    
    def setup_logging(self):
        """Setup production logging"""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / f'automl_server_{self.environment}.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger = logging.getLogger('AutoMLServer')
    
    def check_prerequisites(self) -> bool:
        """Check if all prerequisites are met"""
        self.logger.info("Checking prerequisites...")
        
        # Check if required modules can be imported
        required_modules = [
            'fastapi', 'uvicorn', 'streamlit', 
            'pandas', 'sklearn', 'boto3'
        ]
        
        missing_modules = []
        for module in required_modules:
            try:
                __import__(module)
            except ImportError:
                missing_modules.append(module)
        
        if missing_modules:
            self.logger.error(f"Missing required modules: {missing_modules}")
            return False
        
        # Check if source code exists
        src_dir = Path(__file__).parent.parent / "src"
        if not src_dir.exists():
            self.logger.error(f"Source directory not found: {src_dir}")
            return False
        
        # Check if API main exists
        api_main = src_dir / "api" / "main.py"
        if not api_main.exists():
            self.logger.error(f"API main not found: {api_main}")
            return False
        
        # Check if Streamlit app exists
        streamlit_app = Path(__file__).parent.parent / "infrastructure" / "streamlit" / "app.py"
        if not streamlit_app.exists():
            self.logger.error(f"Streamlit app not found: {streamlit_app}")
            return False
        
        self.logger.info("✅ All prerequisites met")
        return True
    
    def setup_environment_variables(self):
        """Setup production environment variables"""
        env_vars = {
            'AUTOML_ENVIRONMENT': self.environment,
            'AUTOML_API_HOST': self.api_host,
            'AUTOML_API_PORT': str(self.api_port),
            'AUTOML_LOG_LEVEL': 'INFO' if self.environment == 'production' else 'DEBUG',
            'AUTOML_ENABLE_SAGEMAKER': 'true',
            'AWS_DEFAULT_REGION': os.getenv('AWS_DEFAULT_REGION', 'us-east-1')
        }
        
        for key, value in env_vars.items():
            os.environ[key] = value
            self.logger.info(f"Set {key}={value}")
    
    def start_api_server(self):
        """Start FastAPI server"""
        self.logger.info(f"Starting FastAPI server on {self.api_host}:{self.api_port}")
        
        # Configure uvicorn for production
        config = uvicorn.Config(
            "src.api.main:app",
            host=self.api_host,
            port=self.api_port,
            log_level="info" if self.environment == "production" else "debug",
            access_log=True,
            reload=False,  # No reload in production
            workers=1 if self.environment == "dev" else 4
        )
        
        server = uvicorn.Server(config)
        return server.serve()
    
    def start_streamlit_server(self):
        """Start Streamlit server"""
        self.logger.info(f"Starting Streamlit server on port {self.streamlit_port}")
        
        streamlit_app = Path(__file__).parent.parent / "infrastructure" / "streamlit" / "app.py"
        
        cmd = [
            sys.executable, "-m", "streamlit", "run", 
            str(streamlit_app),
            "--server.port", str(self.streamlit_port),
            "--server.address", "0.0.0.0",
            "--browser.gatherUsageStats", "false",
            "--server.headless", "true" if self.environment == "production" else "false"
        ]
        
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.streamlit_process = process
        
        return process
    
    async def health_check_loop(self):
        """Continuous health checking"""
        import aiohttp
        
        while True:
            try:
                # Check API health
                async with aiohttp.ClientSession() as session:
                    async with session.get(f"http://{self.api_host}:{self.api_port}/health") as response:
                        if response.status == 200:
                            self.logger.debug("API health check: OK")
                        else:
                            self.logger.warning(f"API health check failed: {response.status}")
                
                # Check Streamlit (basic process check)
                if self.streamlit_process and self.streamlit_process.poll() is None:
                    self.logger.debug("Streamlit health check: OK")
                else:
                    self.logger.warning("Streamlit process not running")
                    
            except Exception as e:
                self.logger.error(f"Health check error: {e}")
            
            await asyncio.sleep(30)  # Check every 30 seconds
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        self.logger.info(f"Received signal {signum}, shutting down gracefully...")
        
        # Stop Streamlit process
        if self.streamlit_process:
            self.logger.info("Stopping Streamlit server...")
            self.streamlit_process.terminate()
            try:
                self.streamlit_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.streamlit_process.kill()
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        self.logger.info("Shutdown complete")
        sys.exit(0)
    
    async def run_servers(self):
        """Run both servers concurrently"""
        self.logger.info("🚀 Starting AutoML Production Servers")
        
        # Setup environment
        self.setup_environment_variables()
        
        # Start health checking
        health_check_task = asyncio.create_task(self.health_check_loop())
        
        # Start Streamlit in subprocess
        streamlit_process = self.start_streamlit_server()
        
        # Wait a moment for Streamlit to start
        await asyncio.sleep(5)
        
        self.logger.info("✅ All servers started successfully")
        self.logger.info(f"🌐 API Documentation: http://{self.api_host}:{self.api_port}/docs")
        self.logger.info(f"🎨 Streamlit Interface: http://localhost:{self.streamlit_port}")
        
        # Run FastAPI server (this will block)
        try:
            await self.start_api_server()
        except KeyboardInterrupt:
            self.logger.info("Received shutdown signal")
        finally:
            # Cleanup
            health_check_task.cancel()
            if streamlit_process:
                streamlit_process.terminate()
    
    def run(self):
        """Main entry point"""
        if not self.check_prerequisites():
            sys.exit(1)
        
        try:
            asyncio.run(self.run_servers())
        except KeyboardInterrupt:
            self.logger.info("Server shutdown requested")
        except Exception as e:
            self.logger.error(f"Server error: {e}")
            sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description='AutoML Platform Production Server')
    parser.add_argument('--api-host', default='0.0.0.0', help='API server host')
    parser.add_argument('--api-port', type=int, default=8000, help='API server port')
    parser.add_argument('--streamlit-port', type=int, default=8501, help='Streamlit port')
    parser.add_argument('--environment', default='production', 
                       choices=['dev', 'staging', 'production'],
                       help='Environment configuration')
    parser.add_argument('--workers', type=int, default=4, help='Number of API workers (production)')
    parser.add_argument('--log-level', default='INFO', 
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    
    args = parser.parse_args()
    
    # Create and run server
    server = AutoMLProductionServer(
        api_host=args.api_host,
        api_port=args.api_port,
        streamlit_port=args.streamlit_port,
        environment=args.environment
    )
    
    server.run()

if __name__ == "__main__":
    main()