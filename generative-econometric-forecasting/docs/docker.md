# Docker Deployment Guide

## Overview

The Generative Econometric Forecasting Platform provides comprehensive Docker support for both development and production environments. This guide covers containerized deployment options, configuration, and best practices.

## Quick Start

### Production Deployment
```bash
# Navigate to project directory
cd generative-econometric-forecasting

# Start all services
docker-compose up -d

# API available at: http://localhost:8000
# Redis cache: localhost:6379
# PostgreSQL: localhost:5432
```

### Development Environment
```bash
# Start development environment with live reload
docker-compose -f docker/docker-compose.dev.yml up

# Includes Jupyter notebook at: http://localhost:8888
```

## Container Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Nginx Proxy   │    │  Econometric    │    │   PostgreSQL    │
│   (Port 80)     │────│   API Server    │────│   (Metadata)    │
└─────────────────┘    │  (Port 8000)    │    └─────────────────┘
                       └─────────────────┘              │
                                │                       │
                       ┌─────────────────┐              │
                       │   Redis Cache   │──────────────┘
                       │  (Port 6379)    │
                       └─────────────────┘
```

## Images and Services

### 1. Production API Server (`econometric-api`)
**Base**: `python:3.11-slim`
**Purpose**: Main FastAPI application with R integration
**Ports**: 8000
**Features**:
- R statistical packages pre-installed
- Non-root user for security
- Health checks configured
- Volume mounts for data and outputs

### 2. Development Environment (`econometric-api-dev`)
**Base**: `python:3.11-slim` + dev tools
**Purpose**: Development with live reload and debugging
**Ports**: 8000 (API), 8888 (Jupyter), 5678 (Debugger)
**Features**:
- Live code mounting
- Jupyter notebook server
- Development dependencies (pytest, black, flake8)
- Debug port for remote debugging

### 3. Supporting Services

#### Nginx Reverse Proxy
- Load balancing and SSL termination
- Request routing and rate limiting
- Static file serving
- Production-ready configuration

#### Redis Cache
- Model result caching
- Session storage
- Background task queuing

#### PostgreSQL Database
- Forecast metadata storage
- User session management
- Audit logging

## Environment Configuration

### Production Environment Variables
```bash
# Core API Configuration
LOG_LEVEL=INFO
PYTHONPATH=/app
ENVIRONMENT=production

# Economic Data APIs
FRED_API_KEY=your_fred_api_key_here
NIXTLA_API_KEY=your_nixtla_api_key_here

# AI & LLM Configuration
OPENAI_API_KEY=your_openai_api_key_here
LANGCHAIN_API_KEY=your_langchain_api_key_here
LANGSMITH_PROJECT=econometric-forecasting-prod

# Database Configuration
DATABASE_URL=postgresql://econometric_user:econometric_pass@postgres:5432/econometric_db
REDIS_URL=redis://redis:6379/0

# R Configuration
R_HOME=/usr/lib/R
R_PACKAGES_REQUIRED=vars,forecast,urca,tseries

# Security
API_KEY_REQUIRED=true
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_PERIOD=60
```

### Development Environment Variables
```bash
# Development overrides
LOG_LEVEL=DEBUG
ENVIRONMENT=development
API_KEY_REQUIRED=false

# Jupyter Configuration
JUPYTER_ENABLE_LAB=yes
JUPYTER_TOKEN=dev_token_123

# Debug Configuration
ENABLE_DEBUGGER=true
DEBUG_PORT=5678
```

## Volume Mounts

### Production Volumes
```yaml
volumes:
  - ../data:/app/data                    # Economic data and samples
  - ../outputs:/app/outputs              # Generated reports and forecasts
  - ../models:/app/models:ro             # Pre-trained models (read-only)
  - postgres_data:/var/lib/postgresql/data
  - redis_data:/data
```

### Development Volumes
```yaml
volumes:
  - ..:/app                             # Full project mount for live reload
  - /app/venv                           # Exclude virtual environment
  - jupyter_data:/home/appuser/.jupyter  # Jupyter configuration
```

## Docker Commands

### Build and Deploy
```bash
# Build production image
docker build -f docker/Dockerfile -t econometric-forecasting:latest .

# Build development image
docker build -f docker/Dockerfile.dev -t econometric-forecasting:dev .

# Deploy production stack
docker-compose up -d

# Deploy development stack
docker-compose -f docker/docker-compose.dev.yml up
```

### Management Commands
```bash
# View logs
docker-compose logs -f econometric-api

# Execute commands in container
docker-compose exec econometric-api python quick_start.py

# Scale API servers
docker-compose up -d --scale econometric-api=3

# Stop all services
docker-compose down

# Clean up (removes volumes)
docker-compose down -v
```

### Development Workflows
```bash
# Start development environment
docker-compose -f docker/docker-compose.dev.yml up -d

# Access Jupyter notebook
open http://localhost:8888  # Token: dev_token_123

# Run tests in container
docker-compose exec econometric-api-dev pytest tests/

# Format code
docker-compose exec econometric-api-dev black src/ api/

# Type checking
docker-compose exec econometric-api-dev mypy src/
```

## Health Checks

### API Health Check
```bash
# Production health check
curl -f http://localhost:8000/health/status

# Development health check
curl -f http://localhost:8000/health/ready
```

### Service Health Monitoring
```bash
# Check all services
docker-compose ps

# Expected output:
# econometric-forecasting-api   Up (healthy)
# econometric-redis            Up
# econometric-postgres         Up (healthy)
# econometric-nginx           Up
```

## Performance Tuning

### Memory Optimization
```yaml
# In docker-compose.yml
services:
  econometric-api:
    deploy:
      resources:
        limits:
          memory: 4G
          cpus: '2'
        reservations:
          memory: 2G
          cpus: '1'
```

### R Configuration for Containers
```dockerfile
# Optimize R for containers
ENV R_COMPILE_AND_INSTALL_PACKAGES=always
ENV R_KEEP_PKG_SOURCE=no

# Multi-stage build for smaller images
FROM r-base as r-packages
RUN R -e "install.packages(c('vars', 'forecast', 'urca', 'tseries'))"

FROM python:3.11-slim
COPY --from=r-packages /usr/local/lib/R /usr/local/lib/R
```

## Security Configuration

### Non-Root User Setup
```dockerfile
# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Set appropriate permissions
COPY --chown=appuser:appuser . .
```

### Network Security
```yaml
# Custom network isolation
networks:
  econometric-network:
    driver: bridge
    ipam:
      config:
        - subnet: 172.20.0.0/16
```

### Secrets Management
```yaml
# Using Docker secrets (production)
secrets:
  fred_api_key:
    external: true
  openai_api_key:
    external: true

services:
  econometric-api:
    secrets:
      - fred_api_key
      - openai_api_key
```

## Production Deployment

### Docker Swarm Deployment
```bash
# Initialize swarm
docker swarm init

# Create secrets
echo "your_fred_key" | docker secret create fred_api_key -
echo "your_openai_key" | docker secret create openai_api_key -

# Deploy stack
docker stack deploy -c docker-compose.yml econometric-stack
```

### Kubernetes Deployment
```yaml
# kubernetes/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: econometric-forecasting
spec:
  replicas: 3
  selector:
    matchLabels:
      app: econometric-forecasting
  template:
    metadata:
      labels:
        app: econometric-forecasting
    spec:
      containers:
      - name: api
        image: econometric-forecasting:latest
        ports:
        - containerPort: 8000
        env:
        - name: FRED_API_KEY
          valueFrom:
            secretKeyRef:
              name: api-secrets
              key: fred-api-key
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /health/live
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health/ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
```

## Monitoring and Logging

### Log Aggregation
```yaml
# Add to docker-compose.yml
services:
  econometric-api:
    logging:
      driver: "json-file"
      options:
        max-size: "100m"
        max-file: "5"
        tag: "econometric-api"
```

### Metrics Collection
```bash
# Prometheus metrics endpoint
curl http://localhost:8000/metrics

# Health metrics
curl http://localhost:8000/health/metrics
```

## Troubleshooting

### Common Issues

#### Container Won't Start
```bash
# Check container logs
docker-compose logs econometric-api

# Common fixes:
# 1. Port conflicts
netstat -tulpn | grep :8000

# 2. Memory issues
docker system df
docker system prune
```

#### R Packages Not Found
```bash
# Rebuild with R packages
docker-compose build --no-cache econometric-api

# Verify R installation
docker-compose exec econometric-api R --version
```

#### Performance Issues
```bash
# Check resource usage
docker stats

# Monitor API performance
docker-compose exec econometric-api htop
```

### Debug Mode
```yaml
# Enable debug mode in docker-compose.dev.yml
services:
  econometric-api-dev:
    environment:
      - LOG_LEVEL=DEBUG
      - ENABLE_PROFILING=true
    command: python -m debugpy --listen 0.0.0.0:5678 --wait-for-client -m api.main
```

## Best Practices

### Image Optimization
```dockerfile
# Multi-stage build
FROM python:3.11-slim as builder
RUN pip install --user -r requirements.txt

FROM python:3.11-slim
COPY --from=builder /root/.local /root/.local
```

### Data Persistence
```bash
# Regular backups
docker-compose exec postgres pg_dump -U econometric_user econometric_db > backup.sql

# Volume backup
docker run --rm -v econometric_postgres_data:/data -v $(pwd):/backup alpine tar czf /backup/postgres_backup.tar.gz /data
```

### High Availability
```yaml
# Health check configuration
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost:8000/health/status"]
  interval: 30s
  timeout: 10s
  retries: 3
  start_period: 60s
  
# Restart policy
restart: unless-stopped
```

This Docker setup provides a robust, scalable foundation for deploying the Generative Econometric Forecasting Platform in any environment.