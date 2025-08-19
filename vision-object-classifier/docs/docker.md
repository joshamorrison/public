# Docker Setup and Usage

## Overview

The Vision Object Classifier includes Docker configurations for both development and production deployments, providing consistent environments across different systems.

## Docker Files

### Production Dockerfile (`docker/Dockerfile`)
- Optimized for production deployment
- Minimal image size with security best practices
- Non-root user for security
- Health checks included

### Development Dockerfile (`docker/Dockerfile.dev`)
- Includes development tools (Jupyter, pytest, etc.)
- Full source code mounting for hot reload
- Additional debugging utilities

## Quick Start

### Production Deployment
```bash
# Start production services
docker-compose up -d

# View logs
docker-compose logs -f vision-classifier

# Stop services
docker-compose down
```

### Development Environment
```bash
# Start development environment
docker-compose -f docker/docker-compose.dev.yml up

# Access development container
docker-compose -f docker/docker-compose.dev.yml exec vision-classifier-dev bash

# Run Jupyter notebook
docker-compose -f docker/docker-compose.dev.yml exec vision-classifier-dev jupyter notebook --ip=0.0.0.0
```

## Manual Docker Commands

### Building Images
```bash
# Build production image
docker build -f docker/Dockerfile -t vision-classifier:latest .

# Build development image  
docker build -f docker/Dockerfile.dev -t vision-classifier:dev .

# Build with custom tag
docker build -f docker/Dockerfile -t vision-classifier:v1.0.0 .
```

### Running Containers

#### Basic Container
```bash
docker run -d \
  --name vision-classifier \
  -p 8000:8000 \
  vision-classifier:latest
```

#### Container with Volume Mounts
```bash
docker run -d \
  --name vision-classifier \
  -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/outputs:/app/outputs \
  -v $(pwd)/logs:/app/logs \
  vision-classifier:latest
```

#### Container with Environment Variables
```bash
docker run -d \
  --name vision-classifier \
  -p 8000:8000 \
  -e LOG_LEVEL=DEBUG \
  -e API_PORT=8000 \
  -e WORKERS=4 \
  vision-classifier:latest
```

#### Development Container with Source Mount
```bash
docker run -it \
  --name vision-classifier-dev \
  -p 8000:8000 \
  -p 8888:8888 \
  -v $(pwd):/app \
  -e PYTHONPATH=/app \
  vision-classifier:dev bash
```

## Environment Variables

### Production Configuration
```bash
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
WORKERS=4

# Model Configuration
DEFAULT_MODEL_PATH=/app/models/balanced_model.pth
DEFAULT_CONFIG_PATH=/app/models/balanced_config.json

# Logging
LOG_LEVEL=INFO

# Performance
TIMEOUT=30
KEEP_ALIVE=2
```

### Development Configuration
```bash
# Development specific
FLASK_ENV=development
DEBUG_MODE=true
PYTHONPATH=/app

# Tools
JUPYTER_PORT=8888
PROFILING_ENABLED=true
```

## Volume Mounts

### Recommended Volume Structure
```bash
docker run -d \
  -v $(pwd)/models:/app/models:ro \          # Read-only models
  -v $(pwd)/data/samples:/app/data/samples:ro \ # Read-only sample data
  -v $(pwd)/outputs:/app/outputs \           # Writable outputs
  -v $(pwd)/logs:/app/logs \                 # Writable logs
  -v vision-cache:/app/.cache \              # Persistent cache
  vision-classifier:latest
```

### Development Volume Mounts
```bash
docker run -it \
  -v $(pwd):/app \                          # Full source mount
  -v /app/venv \                            # Exclude venv
  -v /app/__pycache__ \                     # Exclude cache
  -v /app/.pytest_cache \                   # Exclude test cache
  vision-classifier:dev
```

## Docker Compose Configurations

### Production Stack (`docker-compose.yml`)
```yaml
services:
  vision-classifier:
    build:
      context: .
      dockerfile: docker/Dockerfile
    ports:
      - "8000:8000"
    volumes:
      - ./outputs:/app/outputs
      - ./logs:/app/logs
    environment:
      - LOG_LEVEL=INFO
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health/status"]
      interval: 30s
      timeout: 10s
      retries: 3
```

### Development Stack (`docker-compose.dev.yml`)
```yaml
services:
  vision-classifier-dev:
    build:
      context: .
      dockerfile: docker/Dockerfile.dev
    ports:
      - "8000:8000"
      - "8888:8888"
    volumes:
      - .:/app
      - /app/venv
    environment:
      - PYTHONPATH=/app
      - LOG_LEVEL=DEBUG
    stdin_open: true
    tty: true
```

## Health Checks

### Built-in Health Check
The production Dockerfile includes a health check:
```dockerfile
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health/status || exit 1
```

### Manual Health Check
```bash
# Check container health
docker inspect --format='{{.State.Health.Status}}' vision-classifier

# View health check logs
docker inspect vision-classifier | jq '.[0].State.Health'

# Test health endpoint directly
curl http://localhost:8000/health/status
```

## Troubleshooting Docker Issues

### Common Problems

#### Container Exits Immediately
```bash
# Check logs
docker logs vision-classifier

# Run interactively
docker run -it --entrypoint /bin/bash vision-classifier:latest

# Check health status
docker inspect vision-classifier
```

#### Port Conflicts
```bash
# Check port usage
netstat -an | grep 8000

# Use different port
docker run -p 8080:8000 vision-classifier:latest

# Or in compose file
ports:
  - "8080:8000"
```

#### Volume Mount Issues
```bash
# Check permissions
ls -la models/
chmod 644 models/*.pth

# Test with absolute paths
docker run -v /full/path/to/models:/app/models vision-classifier:latest
```

#### Memory Issues
```bash
# Monitor container resources
docker stats vision-classifier

# Limit memory usage
docker run --memory=2g vision-classifier:latest

# Or in compose file
deploy:
  resources:
    limits:
      memory: 2G
```

### Debug Commands

#### Container Inspection
```bash
# Container details
docker inspect vision-classifier

# Process list
docker exec vision-classifier ps aux

# File system
docker exec vision-classifier ls -la /app

# Network info
docker exec vision-classifier netstat -tlnp
```

#### Log Analysis
```bash
# Container logs
docker logs -f vision-classifier

# Application logs inside container
docker exec vision-classifier tail -f logs/app.log

# System logs
docker exec vision-classifier dmesg
```

## Performance Optimization

### Production Optimizations
```dockerfile
# Multi-stage build for smaller images
FROM python:3.11-slim as builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --user -r requirements.txt

FROM python:3.11-slim
COPY --from=builder /root/.local /root/.local
COPY . /app
WORKDIR /app
```

### Resource Limits
```yaml
# docker-compose.yml
services:
  vision-classifier:
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
        reservations:
          cpus: '1'
          memory: 2G
```

### Caching Strategies
```bash
# Use BuildKit for better caching
export DOCKER_BUILDKIT=1
docker build -f docker/Dockerfile .

# Multi-stage caching
docker build --target builder -t vision-classifier:builder .
docker build --cache-from vision-classifier:builder .
```

## Security Best Practices

### Container Security
```dockerfile
# Use non-root user
RUN useradd -m -u 1000 appuser
USER appuser

# Set proper permissions
COPY --chown=appuser:appuser . /app

# Use specific base image versions
FROM python:3.11.5-slim
```

### Runtime Security
```bash
# Run with read-only root filesystem
docker run --read-only -v /tmp --tmpfs /app/tmp vision-classifier:latest

# Drop capabilities
docker run --cap-drop ALL vision-classifier:latest

# Use security profiles
docker run --security-opt seccomp=seccomp-profile.json vision-classifier:latest
```

## Integration Examples

### CI/CD Pipeline
```yaml
# .github/workflows/docker.yml
name: Docker Build
on: [push]
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Build Docker image
        run: docker build -f docker/Dockerfile -t vision-classifier:${{ github.sha }} .
      - name: Test container
        run: |
          docker run -d --name test-container -p 8000:8000 vision-classifier:${{ github.sha }}
          sleep 10
          curl -f http://localhost:8000/health/status
```

### Production Deployment
```bash
# Blue-green deployment script
#!/bin/bash
NEW_VERSION=$1
OLD_CONTAINER=$(docker ps --format "table {{.Names}}" | grep vision-classifier)

# Start new version
docker run -d --name vision-classifier-new -p 8001:8000 vision-classifier:$NEW_VERSION

# Health check
sleep 30
if curl -f http://localhost:8001/health/status; then
    # Switch traffic
    docker stop $OLD_CONTAINER
    docker rename vision-classifier-new vision-classifier
    docker port update vision-classifier 8000:8000
    echo "Deployment successful"
else
    docker stop vision-classifier-new
    echo "Deployment failed"
    exit 1
fi
```