# Docker Setup & Usage

## 🐳 Docker Configuration

The Multi-Agent Orchestration Platform includes comprehensive Docker support for development, testing, and production deployment.

## 📁 Docker Files Overview

```
docker/
├── Dockerfile              # Production image
├── Dockerfile.dev          # Development image
├── docker-compose.yml      # Production services
├── docker-compose.dev.yml  # Development services
└── nginx.conf              # Nginx configuration
```

## 🏗️ Building Images

### **Production Image**
```bash
# Build production image
docker build -f docker/Dockerfile -t multi-agent-platform:latest .

# Build with specific version tag
docker build -f docker/Dockerfile -t multi-agent-platform:v1.0.0 .

# Build with build arguments
docker build \
  --build-arg PYTHON_VERSION=3.9 \
  -f docker/Dockerfile \
  -t multi-agent-platform:latest .
```

### **Development Image**
```bash
# Build development image with hot reload
docker build -f docker/Dockerfile.dev -t multi-agent-platform:dev .

# Run development container with volume mounts
docker run -p 8000:8000 -v $(pwd):/app multi-agent-platform:dev
```

## 🐳 Production Dockerfile

```dockerfile
# docker/Dockerfile
FROM python:3.9-slim as builder

# Set build arguments
ARG PYTHON_VERSION=3.9
ARG APP_VERSION=latest

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /venv
ENV PATH="/venv/bin:$PATH"

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.9-slim

# Copy virtual environment from builder
COPY --from=builder /venv /venv
ENV PATH="/venv/bin:$PATH"

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home --shell /bin/bash app
USER app
WORKDIR /app

# Copy application code
COPY --chown=app:app src/ ./src/
COPY --chown=app:app *.py ./
COPY --chown=app:app requirements.txt ./

# Create necessary directories
RUN mkdir -p /app/outputs /app/logs /app/data

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Set labels
LABEL maintainer="Joshua Morrison <joshamorrison@gmail.com>"
LABEL version="${APP_VERSION}"
LABEL description="Multi-Agent Orchestration Platform"

# Default command
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## 🛠️ Development Dockerfile

```dockerfile
# docker/Dockerfile.dev
FROM python:3.9-slim

# Install development dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    postgresql-client \
    curl \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home --shell /bin/bash app
USER app
WORKDIR /app

# Install Python dependencies
COPY requirements.txt requirements-dev.txt ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir -r requirements-dev.txt

# Copy application code (will be overridden by volume mount)
COPY --chown=app:app . .

# Expose port
EXPOSE 8000

# Development command with hot reload
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
```

## 🎼 Docker Compose

### **Production Configuration**
```yaml
# docker-compose.yml
version: '3.8'

services:
  app:
    build:
      context: .
      dockerfile: docker/Dockerfile
    image: multi-agent-platform:latest
    container_name: multi-agent-app
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://postgres:password@db:5432/agents
      - REDIS_URL=redis://redis:6379
      - LANGCHAIN_TRACING_V2=true
      - LANGCHAIN_API_KEY=${LANGCHAIN_API_KEY}
      - LANGCHAIN_PROJECT=multi-agent-production
    depends_on:
      - db
      - redis
    volumes:
      - ./outputs:/app/outputs
      - ./logs:/app/logs
    networks:
      - app-network
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s

  db:
    image: postgres:15
    container_name: multi-agent-db
    environment:
      POSTGRES_DB: agents
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: password
      POSTGRES_INITDB_ARGS: "--encoding=UTF-8"
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./infrastructure/database/init.sql:/docker-entrypoint-initdb.d/init.sql
    ports:
      - "5432:5432"
    networks:
      - app-network
    restart: unless-stopped
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 10s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    container_name: multi-agent-redis
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    networks:
      - app-network
    restart: unless-stopped
    command: redis-server --appendonly yes --maxmemory 512mb --maxmemory-policy allkeys-lru
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 3s
      retries: 3

  nginx:
    image: nginx:alpine
    container_name: multi-agent-nginx
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./docker/nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/nginx/ssl:ro
    depends_on:
      - app
    networks:
      - app-network
    restart: unless-stopped

volumes:
  postgres_data:
    driver: local
  redis_data:
    driver: local

networks:
  app-network:
    driver: bridge
```

### **Development Configuration**
```yaml
# docker-compose.dev.yml
version: '3.8'

services:
  app:
    build:
      context: .
      dockerfile: docker/Dockerfile.dev
    image: multi-agent-platform:dev
    container_name: multi-agent-app-dev
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://postgres:password@db:5432/agents
      - REDIS_URL=redis://redis:6379
      - LANGCHAIN_TRACING_V2=true
      - LANGCHAIN_API_KEY=${LANGCHAIN_API_KEY}
      - LANGCHAIN_PROJECT=multi-agent-development
      - PYTHONPATH=/app
    depends_on:
      - db
      - redis
    volumes:
      - .:/app  # Mount entire project for development
      - /app/.venv  # Exclude virtual environment
    networks:
      - app-network
    stdin_open: true
    tty: true

  db:
    image: postgres:15
    container_name: multi-agent-db-dev
    environment:
      POSTGRES_DB: agents
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: password
    volumes:
      - postgres_dev_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"
    networks:
      - app-network

  redis:
    image: redis:7-alpine
    container_name: multi-agent-redis-dev
    ports:
      - "6379:6379"
    volumes:
      - redis_dev_data:/data
    networks:
      - app-network

  adminer:
    image: adminer
    container_name: multi-agent-adminer
    ports:
      - "8080:8080"
    networks:
      - app-network

volumes:
  postgres_dev_data:
  redis_dev_data:

networks:
  app-network:
    driver: bridge
```

## 🌐 Nginx Configuration

```nginx
# docker/nginx.conf
events {
    worker_connections 1024;
}

http {
    upstream app {
        server app:8000;
    }

    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;

    # Gzip compression
    gzip on;
    gzip_types text/plain application/json application/javascript text/css;

    server {
        listen 80;
        server_name _;

        # Security headers
        add_header X-Frame-Options DENY;
        add_header X-Content-Type-Options nosniff;
        add_header X-XSS-Protection "1; mode=block";

        # Health check endpoint (bypasses rate limiting)
        location /health {
            proxy_pass http://app;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        }

        # API endpoints with rate limiting
        location /api/ {
            limit_req zone=api burst=20 nodelay;
            
            proxy_pass http://app;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            
            # Timeouts
            proxy_connect_timeout 60s;
            proxy_send_timeout 60s;
            proxy_read_timeout 60s;
        }

        # Documentation
        location /docs {
            proxy_pass http://app;
            proxy_set_header Host $host;
        }

        # Static files
        location /static/ {
            alias /app/static/;
            expires 1y;
            add_header Cache-Control "public, immutable";
        }
    }

    # HTTPS configuration (if SSL certificates are available)
    server {
        listen 443 ssl;
        server_name _;

        ssl_certificate /etc/nginx/ssl/cert.pem;
        ssl_certificate_key /etc/nginx/ssl/key.pem;
        ssl_protocols TLSv1.2 TLSv1.3;
        ssl_ciphers HIGH:!aNULL:!MD5;

        location / {
            proxy_pass http://app;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto https;
        }
    }
}
```

## 🚀 Usage Commands

### **Development Workflow**
```bash
# Start development environment
docker-compose -f docker-compose.dev.yml up -d

# View logs
docker-compose -f docker-compose.dev.yml logs -f app

# Run tests in container
docker-compose -f docker-compose.dev.yml exec app pytest tests/

# Access database
docker-compose -f docker-compose.dev.yml exec db psql -U postgres -d agents

# Stop development environment
docker-compose -f docker-compose.dev.yml down
```

### **Production Deployment**
```bash
# Start production environment
docker-compose up -d

# Check service status
docker-compose ps

# View application logs
docker-compose logs -f app

# Scale application
docker-compose up -d --scale app=3

# Update application
docker-compose pull app
docker-compose up -d app

# Backup database
docker-compose exec db pg_dump -U postgres agents > backup.sql

# Stop production environment
docker-compose down
```

### **Maintenance Commands**
```bash
# Clean up unused images
docker image prune -a

# Clean up unused volumes
docker volume prune

# View container resource usage
docker stats

# Execute commands in running container
docker-compose exec app bash

# Restart specific service
docker-compose restart app
```

## 📊 Monitoring & Debugging

### **Container Health Checks**
```bash
# Check container health
docker-compose ps

# Inspect health check details
docker inspect multi-agent-app | grep -A 20 Health

# Test health endpoint
curl http://localhost:8000/health
```

### **Log Management**
```bash
# View logs with timestamps
docker-compose logs -t app

# Follow logs for specific service
docker-compose logs -f --tail=100 app

# View logs for specific time range
docker-compose logs --since="1h" app
```

### **Performance Monitoring**
```bash
# Resource usage
docker stats multi-agent-app

# Container metrics
docker-compose exec app top

# Database performance
docker-compose exec db pg_stat_activity
```

## 🔧 Troubleshooting

### **Common Issues**

#### **Port Already in Use**
```bash
# Find process using port
lsof -i :8000
netstat -tulpn | grep 8000

# Kill process or use different port
docker-compose up -d --force-recreate
```

#### **Permission Errors**
```bash
# Fix file permissions
sudo chown -R $USER:$USER ./outputs ./logs

# Rebuild with correct user
docker-compose build --no-cache app
```

#### **Database Connection Issues**
```bash
# Check database status
docker-compose exec db pg_isready -U postgres

# Reset database
docker-compose down -v
docker-compose up -d db
```

### **Development Tips**

#### **Live Code Reloading**
- Use `docker-compose.dev.yml` for automatic code reload
- Mount source code as volume: `-v .:/app`
- Exclude virtual environment: `-v /app/.venv`

#### **Database Access**
```bash
# Access PostgreSQL
docker-compose exec db psql -U postgres -d agents

# Access Adminer (web interface)
open http://localhost:8080
```

#### **Debugging**
```bash
# Interactive debugging
docker-compose exec app python -c "import pdb; pdb.set_trace()"

# Install additional packages
docker-compose exec app pip install package-name
```

## 🔐 Security Considerations

### **Production Security**
- Use non-root user in containers
- Scan images for vulnerabilities
- Keep base images updated
- Use secrets management for sensitive data
- Enable SSL/TLS for external traffic

### **Environment Variables**
```bash
# Use .env file for sensitive data
echo "LANGCHAIN_API_KEY=your-key" >> .env

# Or use Docker secrets
docker secret create langchain_key /path/to/key/file
```

### **Network Security**
- Use custom networks for service isolation
- Limit exposed ports
- Implement proper firewall rules
- Use reverse proxy for SSL termination

---

This Docker configuration provides a robust foundation for both development and production deployment of the Multi-Agent Orchestration Platform.