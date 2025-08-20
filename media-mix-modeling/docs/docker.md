# Docker Deployment Guide

## 🐳 Containerized Media Mix Modeling Platform

This guide covers deploying and running the Media Mix Modeling platform using Docker for development, testing, and production environments.

## Prerequisites

- Docker Engine 20.10+
- Docker Compose 2.0+
- At least 4GB RAM available for containers
- 10GB free disk space

## Quick Start

### 1. Development Environment

Start the development environment with hot reloading:

```bash
# Start all services
docker-compose -f docker-compose.dev.yml up -d

# View logs
docker-compose -f docker-compose.dev.yml logs -f

# Access the API
curl http://localhost:8000/health
```

### 2. Production Environment

Deploy the production-ready stack:

```bash
# Build and start production services
docker-compose up -d

# Check service status
docker-compose ps

# View logs
docker-compose logs -f api
```

## Container Architecture

### Services Overview

```yaml
┌─────────────────────────────────────────┐
│              Load Balancer              │
│                (nginx)                  │
└─────────────┬───────────────────────────┘
              │
┌─────────────▼───────────────────────────┐
│              API Gateway                │
│            (FastAPI App)                │
└─────────────┬───────────────────────────┘
              │
┌─────────────▼───────────────────────────┐
│             Database                    │
│            (PostgreSQL)                 │
└─────────────────────────────────────────┘
```

### Service Details

| Service | Port | Description |
|---------|------|-------------|
| `api` | 8000 | Main FastAPI application |
| `postgres` | 5432 | PostgreSQL database |
| `redis` | 6379 | Caching and session storage |
| `nginx` | 80/443 | Reverse proxy and load balancer |
| `airflow-webserver` | 8080 | Airflow web interface |
| `airflow-scheduler` | - | Airflow task scheduler |

## Configuration Files

### docker-compose.yml (Production)

The production configuration includes:
- Multi-stage builds for optimized images
- Health checks for all services
- Resource limits
- Production-ready networking
- Volume management for data persistence

### docker-compose.dev.yml (Development)

The development configuration includes:
- Hot reloading for code changes
- Debug mode enabled
- Exposed database ports
- Development-friendly logging
- Volume mounts for source code

## Environment Variables

### Required Variables

Create a `.env` file in the project root:

```bash
# Database Configuration
DATABASE_URL=postgresql://mmm_user:mmm_password@postgres:5432/mmm_db
POSTGRES_DB=mmm_db
POSTGRES_USER=mmm_user
POSTGRES_PASSWORD=mmm_password

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=false
WORKERS=4

# Redis Configuration
REDIS_URL=redis://redis:6379

# Airflow Configuration
AIRFLOW_UID=50000
AIRFLOW_GID=0
_AIRFLOW_WWW_USER_USERNAME=admin
_AIRFLOW_WWW_USER_PASSWORD=admin

# AWS Configuration (if using)
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_DEFAULT_REGION=us-east-1

# Model Configuration
DEFAULT_MODEL_PATH=/app/models
MODEL_CACHE_SIZE=1000

# Monitoring
LOG_LEVEL=INFO
METRICS_ENABLED=true
```

## Building Images

### Development Build

```bash
# Build development image
docker build -f docker/Dockerfile.dev -t mmm-platform:dev .

# Build with no cache
docker build --no-cache -f docker/Dockerfile.dev -t mmm-platform:dev .
```

### Production Build

```bash
# Build production image
docker build -f docker/Dockerfile -t mmm-platform:latest .

# Build with build args
docker build \
  --build-arg PYTHON_VERSION=3.9 \
  --build-arg INSTALL_DEV=false \
  -f docker/Dockerfile \
  -t mmm-platform:latest .
```

## Running Services

### Individual Services

```bash
# Start database only
docker-compose up -d postgres

# Start API with dependencies
docker-compose up -d postgres redis api

# Start all services
docker-compose up -d
```

### Service Management

```bash
# Stop all services
docker-compose down

# Stop and remove volumes
docker-compose down -v

# Restart specific service
docker-compose restart api

# Scale API service
docker-compose up -d --scale api=3
```

## Health Checks

All services include health checks:

### API Health Check

```bash
# Check API health
curl http://localhost:8000/health

# Detailed health check
curl http://localhost:8000/monitoring/health
```

### Database Health Check

```bash
# Check database connection
docker-compose exec postgres pg_isready -U mmm_user -d mmm_db

# Connect to database
docker-compose exec postgres psql -U mmm_user -d mmm_db
```

## Data Management

### Database Initialization

The database is automatically initialized with:
- Required schemas and tables
- Sample data for testing
- Database migrations

### Data Persistence

```bash
# Backup database
docker-compose exec postgres pg_dump -U mmm_user mmm_db > backup.sql

# Restore database
docker-compose exec -T postgres psql -U mmm_user mmm_db < backup.sql

# Access database shell
docker-compose exec postgres psql -U mmm_user mmm_db
```

### Volume Management

```bash
# List volumes
docker volume ls | grep mmm

# Remove unused volumes
docker volume prune

# Backup volume
docker run --rm -v mmm_postgres_data:/data -v $(pwd):/backup alpine tar czf /backup/postgres-backup.tar.gz -C /data .
```

## Development Workflow

### Hot Reloading

When using the development configuration:
1. Code changes are automatically detected
2. API server restarts with new changes
3. No need to rebuild containers

### Debugging

```bash
# View real-time logs
docker-compose -f docker-compose.dev.yml logs -f api

# Access container shell
docker-compose exec api /bin/bash

# Debug specific service
docker-compose -f docker-compose.dev.yml up api --no-deps
```

### Testing in Containers

```bash
# Run tests in container
docker-compose exec api pytest

# Run specific test file
docker-compose exec api pytest tests/test_attribution.py

# Run tests with coverage
docker-compose exec api pytest --cov=src
```

## Production Deployment

### Security Considerations

1. **Environment Variables**: Use Docker secrets or external secret management
2. **Network Security**: Configure proper firewall rules
3. **SSL/TLS**: Enable HTTPS with proper certificates
4. **User Permissions**: Run containers with non-root users
5. **Image Security**: Scan images for vulnerabilities

### Resource Management

```yaml
# Resource limits in docker-compose.yml
services:
  api:
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: '1.0'
        reservations:
          memory: 1G
          cpus: '0.5'
```

### Monitoring

```bash
# Monitor resource usage
docker stats

# Monitor specific container
docker stats mmm-api

# Export metrics
docker-compose exec api curl http://localhost:8000/metrics
```

## Troubleshooting

### Common Issues

1. **Port conflicts**:
   ```bash
   # Check port usage
   netstat -tulpn | grep :8000
   
   # Use different ports
   docker-compose -f docker-compose.dev.yml up -d
   ```

2. **Memory issues**:
   ```bash
   # Increase Docker memory limit
   # Docker Desktop -> Resources -> Advanced -> Memory: 8GB
   
   # Check container memory usage
   docker stats --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}"
   ```

3. **Database connection fails**:
   ```bash
   # Check database logs
   docker-compose logs postgres
   
   # Restart database
   docker-compose restart postgres
   ```

### Log Management

```bash
# View logs by service
docker-compose logs api
docker-compose logs postgres
docker-compose logs redis

# Follow logs in real-time
docker-compose logs -f --tail=100 api

# Export logs
docker-compose logs api > api-logs.txt
```

### Container Cleanup

```bash
# Remove stopped containers
docker container prune

# Remove unused images
docker image prune -a

# Complete cleanup
docker system prune -a --volumes
```

## Performance Optimization

### Image Optimization

1. **Multi-stage builds**: Reduce final image size
2. **Layer caching**: Optimize Dockerfile layer order
3. **Base image selection**: Use Alpine or slim variants
4. **.dockerignore**: Exclude unnecessary files

### Runtime Optimization

```bash
# Use production WSGI server
gunicorn --workers 4 --worker-class uvicorn.workers.UvicornWorker

# Enable caching
REDIS_URL=redis://redis:6379

# Database connection pooling
DATABASE_POOL_SIZE=20
```

## CI/CD Integration

### GitHub Actions

```yaml
name: Docker Build and Deploy

on:
  push:
    branches: [main]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Build Docker image
        run: docker build -t mmm-platform:${{ github.sha }} .
      - name: Run tests
        run: docker run mmm-platform:${{ github.sha }} pytest
```

### Automated Deployment

```bash
# Build and push to registry
docker build -t your-registry/mmm-platform:latest .
docker push your-registry/mmm-platform:latest

# Deploy to production
docker-compose pull
docker-compose up -d
```

This Docker deployment guide provides comprehensive coverage for containerizing and deploying the Media Mix Modeling platform across different environments.