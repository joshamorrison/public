# Docker Deployment for AutoML Agent Platform

This directory contains all Docker-related files for deploying the AutoML Agent Platform in containerized environments.

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Nginx Proxy   │    │  Streamlit UI   │    │   FastAPI       │
│   Port: 80/443  │────│   Port: 8501    │────│   Port: 8000    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                               │
                       ┌─────────────────┐    │    ┌─────────────────┐
                       │   PostgreSQL    │────┘    │     Redis       │
                       │   Port: 5432    │         │   Port: 6379    │
                       └─────────────────┘         └─────────────────┘
                                               │
                       ┌─────────────────┐    │    ┌─────────────────┐
                       │   Prometheus    │────┘    │    Grafana      │
                       │   Port: 9090    │         │   Port: 3000    │
                       └─────────────────┘         └─────────────────┘
```

## 📁 Files Overview

- **`api.Dockerfile`** - Production-ready API service container
- **`streamlit.Dockerfile`** - Streamlit interface container  
- **`docker-compose.yml`** - Full stack orchestration
- **`nginx.conf`** - Reverse proxy configuration
- **`init-db.sql`** - PostgreSQL database initialization
- **`monitoring/`** - Prometheus and Grafana configurations
- **`.dockerignore`** - Files to exclude from Docker builds

## 🚀 Quick Start

### 1. Development Environment

```bash
# Start all services
docker-compose up -d

# Check service status
docker-compose ps

# View logs
docker-compose logs -f api
```

### 2. Production Deployment

```bash
# Build and start in production mode
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# Scale API workers
docker-compose up -d --scale api=3
```

### 3. Individual Services

```bash
# API service only
docker build -f docker/api.Dockerfile -t automl-api .
docker run -p 8000:8000 automl-api

# Streamlit interface only  
docker build -f docker/streamlit.Dockerfile -t automl-streamlit .
docker run -p 8501:8501 automl-streamlit
```

## 🔧 Configuration

### Environment Variables

Create `.env` file in project root:

```env
# Database
DATABASE_URL=postgresql://automl:automl_password@postgres:5432/automl
REDIS_URL=redis://redis:6379/0

# Security
API_SECRET_KEY=your-super-secret-key-here
JWT_SECRET_KEY=another-secret-key

# API Configuration
API_WORKERS=4
API_HOST=0.0.0.0
API_PORT=8000

# Monitoring
PROMETHEUS_ENABLED=true
GRAFANA_ADMIN_PASSWORD=admin

# External Services
OPENAI_API_KEY=your-openai-key
WANDB_API_KEY=your-wandb-key
```

### Resource Limits

Update `docker-compose.yml` for production:

```yaml
services:
  api:
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 4G
        reservations:
          cpus: '1.0'
          memory: 2G
```

## 📊 Monitoring

### Access Points

- **API Documentation**: http://localhost/docs
- **Streamlit Interface**: http://localhost
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (admin/admin)

### Health Checks

All services include health checks:

```bash
# Check all service health
docker-compose ps

# Individual health check
curl http://localhost/health
curl http://localhost:8000/health
```

### Metrics

The platform exposes metrics at:
- API metrics: `/metrics`
- Custom AutoML metrics: Job completion rates, model performance, agent collaboration

## 🛠️ Development

### Local Development with Docker

```bash
# Development compose with hot reload
docker-compose -f docker-compose.dev.yml up

# Run tests in container
docker-compose exec api pytest

# Access container shell
docker-compose exec api bash
```

### Building Images

```bash
# Build all images
docker-compose build

# Build specific service
docker-compose build api

# Build with no cache
docker-compose build --no-cache
```

## 🔒 Security

### SSL/TLS Setup

1. Generate certificates:
```bash
mkdir -p docker/ssl
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout docker/ssl/nginx.key \
  -out docker/ssl/nginx.crt
```

2. Update nginx.conf for HTTPS

### Production Security

- Change default passwords
- Use Docker secrets for sensitive data
- Enable firewall rules
- Regular security updates

## 📈 Scaling

### Horizontal Scaling

```bash
# Scale API workers
docker-compose up -d --scale api=3

# Load balancer configuration
# Nginx automatically load balances across API instances
```

### Kubernetes Migration

Ready for K8s deployment:
- Multi-stage builds for efficiency
- Health checks for readiness/liveness
- ConfigMaps/Secrets support
- Resource limits defined

## 🐛 Troubleshooting

### Common Issues

1. **Port conflicts**:
   ```bash
   # Check port usage
   netstat -tlnp | grep :8000
   
   # Change ports in docker-compose.yml
   ```

2. **Database connection**:
   ```bash
   # Check database logs
   docker-compose logs postgres
   
   # Test connection
   docker-compose exec api python -c "from src.database import test_connection; test_connection()"
   ```

3. **Memory issues**:
   ```bash
   # Monitor resource usage
   docker stats
   
   # Increase memory limits
   ```

### Logs

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f api

# Follow recent logs
docker-compose logs -f --tail=100
```

## 🧹 Cleanup

```bash
# Stop all services
docker-compose down

# Remove volumes (caution: deletes data)
docker-compose down -v

# Remove images
docker-compose down --rmi all

# Full cleanup
docker system prune -a --volumes
```

## 📋 Maintenance

### Backups

```bash
# Database backup
docker-compose exec postgres pg_dump -U automl automl > backup.sql

# Volume backup
docker run --rm -v automl_postgres_data:/data -v $(pwd):/backup ubuntu tar czf /backup/postgres_backup.tar.gz /data
```

### Updates

```bash
# Pull latest images
docker-compose pull

# Rebuild and restart
docker-compose up -d --build
```

This Docker setup provides a production-ready, scalable deployment of the AutoML Agent Platform with monitoring, security, and operational best practices.