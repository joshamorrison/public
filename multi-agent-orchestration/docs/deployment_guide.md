# Deployment Guide

## 🚀 Overview

This guide covers deployment options from local development to enterprise-scale production environments. The platform supports Docker containerization, cloud deployment, and auto-scaling configurations.

## 🏠 Local Development

### **Quick Setup**
```bash
# Clone and setup
git clone https://github.com/joshamorrison/public.git
cd public/multi-agent-orchestration

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your configurations

# Run the platform
python quick_start.py
```

### **Development Server**
```bash
# Start FastAPI development server
uvicorn src.api.main:app --reload --port 8000

# Access API documentation
open http://localhost:8000/docs
```

### **Environment Configuration**
```bash
# .env file configuration
DATABASE_URL=sqlite:///./agents.db
REDIS_URL=redis://localhost:6379
LANGCHAIN_API_KEY=your-langsmith-key
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=multi-agent-dev
```

## 🐳 Docker Deployment

### **Development Container**
```bash
# Build development image
docker build -f docker/Dockerfile.dev -t multi-agent-platform:dev .

# Run with development settings
docker run -p 8000:8000 -v $(pwd):/app multi-agent-platform:dev
```

### **Production Container**
```bash
# Build production image
docker build -f docker/Dockerfile -t multi-agent-platform:latest .

# Run production container
docker run -p 8000:8000 -e DATABASE_URL=your-db-url multi-agent-platform:latest
```

### **Docker Compose**
```yaml
# docker-compose.yml
version: '3.8'

services:
  app:
    build:
      context: .
      dockerfile: docker/Dockerfile
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://postgres:password@db:5432/agents
      - REDIS_URL=redis://redis:6379
    depends_on:
      - db
      - redis

  db:
    image: postgres:15
    environment:
      POSTGRES_DB: agents
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: password
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data

volumes:
  postgres_data:
  redis_data:
```

### **Start Full Stack**
```bash
# Start all services
docker-compose up -d

# Check service status
docker-compose ps

# Scale application
docker-compose up -d --scale app=3
```

## ☁️ AWS Deployment

### **ECS Task Definition**
```json
{
  "family": "multi-agent-platform",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "512",
  "memory": "1024",
  "containerDefinitions": [
    {
      "name": "multi-agent-app",
      "image": "your-account.dkr.ecr.us-east-1.amazonaws.com/multi-agent-platform:latest",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "DATABASE_URL",
          "value": "postgresql://user:pass@rds-endpoint:5432/agents"
        }
      ]
    }
  ]
}
```

### **Auto Scaling**
```json
{
  "serviceNamespace": "ecs",
  "resourceId": "service/multi-agent-cluster/multi-agent-service",
  "scalableDimension": "ecs:service:DesiredCount",
  "minCapacity": 2,
  "maxCapacity": 10,
  "targetTrackingScalingPolicies": [
    {
      "targetValue": 70.0,
      "predefinedMetricSpecification": {
        "predefinedMetricType": "ECSServiceAverageCPUUtilization"
      }
    }
  ]
}
```

## 🔧 Infrastructure as Code

### **CloudFormation**
```yaml
# infrastructure/aws/cloudformation/main.yaml
AWSTemplateFormatVersion: '2010-09-09'
Description: 'Multi-Agent Platform Infrastructure'

Resources:
  ECSCluster:
    Type: AWS::ECS::Cluster
    Properties:
      ClusterName: multi-agent-cluster

  Database:
    Type: AWS::RDS::DBInstance
    Properties:
      DBInstanceClass: db.t3.micro
      Engine: postgres
      AllocatedStorage: 20
      DatabaseName: agents

  LoadBalancer:
    Type: AWS::ElasticLoadBalancingV2::LoadBalancer
    Properties:
      Scheme: internet-facing
      Type: application
```

### **Terraform**
```hcl
# infrastructure/aws/terraform/main.tf
resource "aws_ecs_cluster" "main" {
  name = "multi-agent-cluster"
}

resource "aws_db_instance" "main" {
  identifier     = "multi-agent-db"
  instance_class = "db.t3.micro"
  engine         = "postgres"
  allocated_storage = 20
}
```

## 📊 Monitoring

### **Health Checks**
```python
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "components": {
            "database": "healthy",
            "redis": "healthy"
        }
    }
```

### **Prometheus Metrics**
```yaml
# docker-compose.monitoring.yml
services:
  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml

  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
    volumes:
      - grafana_data:/var/lib/grafana
```

## 🔐 Security

### **SSL Configuration**
```nginx
# nginx.conf
server {
    listen 443 ssl;
    server_name your-domain.com;
    
    ssl_certificate /etc/letsencrypt/live/your-domain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/your-domain.com/privkey.pem;
    
    location / {
        proxy_pass http://app:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### **Environment Variables**
```bash
# Production environment
DATABASE_URL=postgresql://user:pass@db:5432/agents
REDIS_URL=redis://redis:6379
LANGCHAIN_API_KEY=your-production-key
API_RATE_LIMIT=1000
CORS_ORIGINS=https://your-domain.com
```

## 🚨 Troubleshooting

### **Common Issues**
1. **Database Connection**: Check DATABASE_URL format and network access
2. **Memory Issues**: Increase container memory limits
3. **SSL Problems**: Verify certificate renewal and nginx config

### **Debug Commands**
```bash
# Check container logs
docker logs multi-agent-platform

# Database connection test
psql $DATABASE_URL -c "SELECT 1;"

# Redis connection test
redis-cli -u $REDIS_URL ping
```

## 🎯 Production Checklist

- [ ] SSL certificates configured
- [ ] Database backups automated
- [ ] Monitoring and alerting setup
- [ ] Auto-scaling configured
- [ ] Load balancing implemented
- [ ] Security groups restricted
- [ ] Environment variables secured
- [ ] Health checks enabled

For additional troubleshooting, see [troubleshooting.md](troubleshooting.md).

---

This deployment guide provides comprehensive coverage from development to production. For specific deployment scenarios, refer to the infrastructure templates in the `infrastructure/` directory.