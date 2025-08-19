#!/usr/bin/env python3
"""
Docker Build Script

Builds and manages Docker containers for the Multi-Agent Orchestration Platform.
"""

import subprocess
import sys
import argparse
from pathlib import Path

def run_command(cmd, cwd=None):
    """Run shell command and return result."""
    try:
        result = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"❌ Command failed: {cmd}")
            print(f"Error: {result.stderr}")
            return False
        return True
    except Exception as e:
        print(f"❌ Error running command: {e}")
        return False

def build_development():
    """Build development environment."""
    print("🔨 Building development environment...")
    
    docker_dir = Path(__file__).parent.parent / "docker"
    
    # Build development containers
    if not run_command("docker-compose -f docker-compose.dev.yml build", cwd=docker_dir):
        return False
    
    print("✅ Development environment built successfully!")
    return True

def build_production():
    """Build production environment."""
    print("🔨 Building production environment...")
    
    docker_dir = Path(__file__).parent.parent / "docker"
    
    # Build production containers
    if not run_command("docker-compose build", cwd=docker_dir):
        return False
    
    print("✅ Production environment built successfully!")
    return True

def start_development():
    """Start development environment."""
    print("🚀 Starting development environment...")
    
    docker_dir = Path(__file__).parent.parent / "docker"
    
    # Start development containers
    if not run_command("docker-compose -f docker-compose.dev.yml up -d", cwd=docker_dir):
        return False
    
    print("✅ Development environment started!")
    print("📡 API available at: http://localhost:8000")
    print("🔍 Health check: http://localhost:8000/health")
    return True

def start_production():
    """Start production environment."""
    print("🚀 Starting production environment...")
    
    docker_dir = Path(__file__).parent.parent / "docker"
    
    # Start production containers
    if not run_command("docker-compose up -d", cwd=docker_dir):
        return False
    
    print("✅ Production environment started!")
    print("📡 API available at: http://localhost:8000")
    print("🔍 Health check: http://localhost:8000/health")
    return True

def stop_services():
    """Stop all Docker services."""
    print("🛑 Stopping Docker services...")
    
    docker_dir = Path(__file__).parent.parent / "docker"
    
    # Stop development services
    run_command("docker-compose -f docker-compose.dev.yml down", cwd=docker_dir)
    
    # Stop production services
    run_command("docker-compose down", cwd=docker_dir)
    
    print("✅ All services stopped!")

def show_logs():
    """Show logs from Docker services."""
    docker_dir = Path(__file__).parent.parent / "docker"
    
    print("📋 Showing logs (Ctrl+C to exit)...")
    
    # Try development first, then production
    if not run_command("docker-compose -f docker-compose.dev.yml logs -f", cwd=docker_dir):
        run_command("docker-compose logs -f", cwd=docker_dir)

def show_status():
    """Show status of Docker services."""
    docker_dir = Path(__file__).parent.parent / "docker"
    
    print("📊 Docker Services Status:")
    print("-" * 40)
    
    # Check development services
    result = subprocess.run(
        "docker-compose -f docker-compose.dev.yml ps", 
        shell=True, cwd=docker_dir, capture_output=True, text=True
    )
    if result.returncode == 0 and result.stdout.strip():
        print("🔧 Development Services:")
        print(result.stdout)
    
    # Check production services
    result = subprocess.run(
        "docker-compose ps", 
        shell=True, cwd=docker_dir, capture_output=True, text=True
    )
    if result.returncode == 0 and result.stdout.strip():
        print("🏭 Production Services:")
        print(result.stdout)

def main():
    """Main script entry point."""
    parser = argparse.ArgumentParser(description="Docker management for Multi-Agent Platform")
    parser.add_argument("action", choices=[
        "build-dev", "build-prod", 
        "start-dev", "start-prod",
        "stop", "logs", "status"
    ], help="Action to perform")
    
    args = parser.parse_args()
    
    print("🐳 Multi-Agent Platform Docker Manager")
    print("=" * 50)
    
    if args.action == "build-dev":
        success = build_development()
    elif args.action == "build-prod":
        success = build_production()
    elif args.action == "start-dev":
        success = build_development() and start_development()
    elif args.action == "start-prod":
        success = build_production() and start_production()
    elif args.action == "stop":
        stop_services()
        success = True
    elif args.action == "logs":
        show_logs()
        success = True
    elif args.action == "status":
        show_status()
        success = True
    else:
        print(f"❌ Unknown action: {args.action}")
        success = False
    
    if not success:
        sys.exit(1)
    
    print("\n🎉 Operation completed successfully!")

if __name__ == "__main__":
    main()