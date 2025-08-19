# Deployment Guide

## Overview

This guide covers deploying the Vision Object Classifier across different environments. We support two main deployment strategies optimized for different use cases.

**Quick Links:**
- 🚀 **[AWS Production Deployment](#aws-production-deployment)** - Scalable, enterprise-ready
- 🎨 **[Streamlit Interactive Deployment](#streamlit-demo-deployment)** - Interactive demos and prototypes  
- 📊 **[Deployment Strategies Comparison](deployment_strategies.md)** - Detailed comparison guide

## Deployment Strategy Selection

| Use Case | Recommended Strategy | Setup Time | Cost |
|----------|-------------------|------------|------|
| Demo/Prototype | Streamlit | 5 minutes | Free |
| Internal Tools | Streamlit Local | 10 minutes | Infrastructure only |
| Production API | AWS | 30 minutes | $50-300/month |
| Enterprise | AWS + Streamlit | 45 minutes | $100-500/month |

## 🎨 Streamlit Demo Deployment

### Quick Start (5 minutes)
```bash
# Navigate to Streamlit infrastructure
cd infrastructure/streamlit/

# Deploy locally with API backend
./deploy.sh local

# Access at: http://localhost:8501
```

### Features
- Interactive web interface with drag-and-drop uploads
- Real-time classification with confidence visualization  
- Batch processing capabilities
- Demo image gallery for quick testing
- Results export to CSV

### Deployment Options
- **Local**: Development and testing (`./deploy.sh local`)
- **Cloud**: Free Streamlit Cloud hosting (`./deploy.sh cloud`)  
- **Docker**: Container deployment (`./deploy.sh docker`)

---

## Quick Start Deployment

### Local Installation

```bash
# Clone and setup
git clone <repository>
cd vision-object-classifier

# Install dependencies
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt

# Start API server
python -m api.main
```

### Docker Deployment

```bash
# Production deployment
docker-compose up -d

# Development deployment  
docker-compose -f docker/docker-compose.dev.yml up
```

## 🚀 AWS Production Deployment

### Infrastructure Overview
Complete production deployment with auto-scaling, load balancing, and monitoring.

```
Internet → ALB → ECS Fargate → S3 (Models/Outputs)
                     ↓
               CloudWatch Logs
```

### Quick Start (30 minutes)
```bash
# Navigate to AWS infrastructure
cd infrastructure/aws/

# Set environment (optional)
export AWS_REGION=us-west-2
export ENVIRONMENT=prod

# Deploy complete infrastructure
./deploy.sh

# API available at ALB DNS name (output)
```

### Option 1: Docker Container (Recommended)

**Advantages:**
- Consistent environment
- Easy scaling
- Simplified dependencies
- Portable across platforms

**Setup:**
```bash
# Build production image
docker build -f docker/Dockerfile -t vision-classifier:latest .

# Run with environment variables
docker run -d \
  --name vision-classifier \
  -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/outputs:/app/outputs \
  -e LOG_LEVEL=INFO \
  vision-classifier:latest
```

### Option 2: Cloud Deployment

#### AWS ECS/Fargate
```bash
# Build and push to ECR
aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin 123456789012.dkr.ecr.us-west-2.amazonaws.com
docker tag vision-classifier:latest 123456789012.dkr.ecr.us-west-2.amazonaws.com/vision-classifier:latest
docker push 123456789012.dkr.ecr.us-west-2.amazonaws.com/vision-classifier:latest

# Deploy with ECS task definition
aws ecs create-service --cluster production --service-name vision-classifier --task-definition vision-classifier:1 --desired-count 2
```

#### Google Cloud Run
```bash
# Deploy to Cloud Run
gcloud run deploy vision-classifier \
  --image gcr.io/PROJECT_ID/vision-classifier \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --port 8000 \
  --memory 2Gi \
  --cpu 1
```

#### Azure Container Instances
```bash
# Deploy to ACI
az container create \
  --resource-group myResourceGroup \
  --name vision-classifier \
  --image myregistry.azurecr.io/vision-classifier:latest \
  --cpu 1 --memory 2 \
  --ports 8000 \
  --environment-variables LOG_LEVEL=INFO
```

### Option 3: Kubernetes Deployment

```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vision-classifier
spec:
  replicas: 3
  selector:
    matchLabels:
      app: vision-classifier
  template:
    metadata:
      labels:
        app: vision-classifier
    spec:
      containers:
      - name: vision-classifier
        image: vision-classifier:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        env:
        - name: LOG_LEVEL
          value: "INFO"
        livenessProbe:
          httpGet:
            path: /health/status
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health/ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: vision-classifier-service
spec:
  selector:
    app: vision-classifier
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer
```

## Configuration Management

### Environment Variables

**Required:**
```bash
# Model configuration
DEFAULT_MODEL_PATH=models/balanced_model.pth
DEFAULT_CONFIG_PATH=models/balanced_config.json

# API configuration
API_HOST=0.0.0.0
API_PORT=8000
LOG_LEVEL=INFO
```

**Optional:**
```bash
# Performance tuning
WORKERS=4
TIMEOUT=30
KEEP_ALIVE=2

# Security
API_KEY_REQUIRED=true
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_PERIOD=60

# Monitoring
PROMETHEUS_ENABLED=true
JAEGER_ENABLED=true
```

### Configuration Files

**production.yaml:**
```yaml
api:
  host: 0.0.0.0
  port: 8000
  workers: 4
  
models:
  default: balanced_model.pth
  cache_size: 3
  
logging:
  level: INFO
  format: json
  
monitoring:
  enabled: true
  prometheus_port: 9090
```

## Scaling Considerations

### Vertical Scaling
- **CPU**: 2-4 cores recommended
- **Memory**: 2-4GB minimum, 8GB+ for high load
- **Storage**: 10GB+ for models and logs
- **GPU**: Optional but improves performance 2-5x

### Horizontal Scaling
- **Load Balancer**: Distribute requests across instances
- **Auto-scaling**: Scale based on CPU/memory usage
- **Health Checks**: Implement proper liveness/readiness probes

### Performance Optimization
```bash
# Optimize for high throughput
export WORKERS=8
export WORKER_CONNECTIONS=1000
export KEEP_ALIVE=5

# Memory optimization
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export OMP_NUM_THREADS=4
```

## Security Configuration

### Network Security
- Use HTTPS in production
- Implement API authentication
- Configure rate limiting
- Set up WAF protection

### Container Security
```dockerfile
# Security best practices in Dockerfile
RUN useradd -m -u 1000 appuser
USER appuser
COPY --chown=appuser:appuser . .
```

### Secret Management
```bash
# Use secret management systems
docker run -d \
  --name vision-classifier \
  -e API_KEY_FILE=/run/secrets/api_key \
  --secret api_key \
  vision-classifier:latest
```

## Monitoring and Logging

### Health Checks
- **Liveness**: `GET /health/live`
- **Readiness**: `GET /health/ready`
- **Status**: `GET /health/status`

### Metrics Collection
```python
# Prometheus metrics endpoint
GET /metrics

# Custom metrics
- classification_requests_total
- classification_duration_seconds
- model_load_time_seconds
- error_rate
```

### Logging Configuration
```json
{
  "version": 1,
  "formatters": {
    "detailed": {
      "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    }
  },
  "handlers": {
    "console": {
      "class": "logging.StreamHandler",
      "formatter": "detailed"
    }
  },
  "root": {
    "level": "INFO",
    "handlers": ["console"]
  }
}
```

## Troubleshooting Deployment

### Common Issues

**Model Loading Errors:**
```bash
# Check model file permissions
ls -la models/
chmod 644 models/*.pth

# Verify model compatibility
python -c "import torch; print(torch.load('models/balanced_model.pth'))"
```

**Memory Issues:**
```bash
# Monitor memory usage
docker stats vision-classifier

# Adjust container limits
docker run --memory=4g --cpus=2 vision-classifier:latest
```

**Performance Issues:**
```bash
# Enable performance profiling
export PROFILING_ENABLED=true

# Check resource utilization
htop
nvidia-smi  # If using GPU
```

### Deployment Checklist

- [ ] Model files accessible and correct permissions
- [ ] Environment variables configured
- [ ] Health checks responding
- [ ] Logging configured and working
- [ ] Monitoring metrics available
- [ ] Security settings applied
- [ ] Load testing completed
- [ ] Backup and recovery plan in place
- [ ] Documentation updated with deployment details