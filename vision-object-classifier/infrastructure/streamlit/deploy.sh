#!/bin/bash

# Vision Object Classifier - Streamlit Deployment Script
set -e

# Configuration
PROJECT_NAME="vision-classifier"
STREAMLIT_PORT="${STREAMLIT_PORT:-8501}"
API_PORT="${API_PORT:-8000}"
ENVIRONMENT="${ENVIRONMENT:-dev}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
}

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."
    
    # Check Python
    if ! command -v python &> /dev/null; then
        error "Python is not installed"
        exit 1
    fi
    
    # Check if we're in project root
    if [ ! -f "../../pyproject.toml" ] && [ ! -f "../../requirements.txt" ]; then
        error "Please run this script from infrastructure/streamlit/ directory"
        exit 1
    fi
    
    log "Prerequisites check passed"
}

# Setup environment
setup_environment() {
    log "Setting up Python environment..."
    
    cd ../../  # Go to project root
    
    # Create virtual environment if it doesn't exist
    if [ ! -d "venv" ]; then
        python -m venv venv
    fi
    
    # Activate virtual environment
    source venv/bin/activate || source venv/Scripts/activate
    
    # Install core requirements
    pip install -r requirements.txt
    
    # Install Streamlit requirements
    pip install -r infrastructure/streamlit/requirements.txt
    
    log "Environment setup complete"
}

# Start FastAPI backend
start_backend() {
    log "Starting FastAPI backend on port $API_PORT..."
    
    # Check if API is already running
    if curl -s "http://localhost:$API_PORT/health/status" &> /dev/null; then
        log "FastAPI backend already running on port $API_PORT"
        return
    fi
    
    # Start API in background
    nohup python -m api.main --host 0.0.0.0 --port $API_PORT > logs/api.log 2>&1 &
    API_PID=$!
    echo $API_PID > infrastructure/streamlit/api.pid
    
    # Wait for API to start
    log "Waiting for API to start..."
    for i in {1..30}; do
        if curl -s "http://localhost:$API_PORT/health/status" &> /dev/null; then
            log "FastAPI backend started successfully (PID: $API_PID)"
            return
        fi
        sleep 2
    done
    
    error "Failed to start FastAPI backend"
    exit 1
}

# Start Streamlit app
start_streamlit() {
    log "Starting Streamlit app on port $STREAMLIT_PORT..."
    
    # Create logs directory
    mkdir -p logs
    
    # Set Streamlit config
    export STREAMLIT_SERVER_PORT=$STREAMLIT_PORT
    export STREAMLIT_SERVER_ADDRESS="0.0.0.0"
    
    # Start Streamlit
    streamlit run infrastructure/streamlit/app.py \
        --server.port $STREAMLIT_PORT \
        --server.address 0.0.0.0 \
        --server.headless true \
        --browser.gatherUsageStats false \
        --client.toolbarMode minimal
}

# Deploy to Streamlit Cloud
deploy_to_cloud() {
    log "Preparing for Streamlit Cloud deployment..."
    
    # Check if git repo is clean
    if ! git diff-index --quiet HEAD --; then
        warn "Git repository has uncommitted changes"
        echo "Please commit your changes before deploying to Streamlit Cloud"
        return 1
    fi
    
    # Create streamlit secrets file template
    cat > infrastructure/streamlit/secrets.toml << EOF
# Streamlit Cloud secrets configuration
# Add your secrets in the Streamlit Cloud dashboard

# API Configuration
API_URL = "http://localhost:8000"

# Optional: External API URL for production
# API_URL = "https://your-api-domain.com"

# Model Configuration  
DEFAULT_MODEL_TYPE = "balanced"
MIN_CONFIDENCE = 0.0

# Feature flags
ENABLE_BATCH_PROCESSING = true
ENABLE_API_MODE = true
SHOW_DEBUG_INFO = false
EOF

    log "Created secrets template at infrastructure/streamlit/secrets.toml"
    log "Deploy instructions:"
    echo "1. Push your code to GitHub"
    echo "2. Go to https://share.streamlit.io/"
    echo "3. Connect your GitHub repository"
    echo "4. Set the app path to: infrastructure/streamlit/app.py"
    echo "5. Add secrets in the Streamlit Cloud dashboard using secrets.toml as template"
}

# Create Docker setup for Streamlit
create_docker_setup() {
    log "Creating Docker setup for Streamlit..."
    
    # Dockerfile for Streamlit
    cat > infrastructure/streamlit/Dockerfile << EOF
# Streamlit Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    libgl1-mesa-glx \\
    libglib2.0-0 \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
COPY infrastructure/streamlit/requirements.txt ./streamlit-requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -r streamlit-requirements.txt

# Copy application
COPY . .

# Create non-root user
RUN useradd -m -u 1000 streamlit && chown -R streamlit:streamlit /app
USER streamlit

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Run Streamlit
CMD ["streamlit", "run", "infrastructure/streamlit/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
EOF

    # Docker compose for Streamlit + API
    cat > infrastructure/streamlit/docker-compose.yml << EOF
version: '3.8'

services:
  vision-api:
    build:
      context: ../../
      dockerfile: docker/Dockerfile
    ports:
      - "8000:8000"
    environment:
      - LOG_LEVEL=INFO
    volumes:
      - ../../models:/app/models:ro
      - ../../outputs:/app/outputs
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health/status"]
      interval: 30s
      timeout: 10s
      retries: 3

  vision-streamlit:
    build:
      context: ../../
      dockerfile: infrastructure/streamlit/Dockerfile
    ports:
      - "8501:8501"
    environment:
      - API_URL=http://vision-api:8000
    depends_on:
      - vision-api
    volumes:
      - ../../data/samples:/app/data/samples:ro
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8501/_stcore/health"]
      interval: 30s
      timeout: 10s
      retries: 3

networks:
  default:
    driver: bridge
EOF

    log "Docker setup created"
    log "Use: docker-compose -f infrastructure/streamlit/docker-compose.yml up"
}

# Stop services
stop_services() {
    log "Stopping services..."
    
    cd ../../
    
    # Stop API if PID file exists
    if [ -f "infrastructure/streamlit/api.pid" ]; then
        API_PID=$(cat infrastructure/streamlit/api.pid)
        if kill -0 $API_PID 2>/dev/null; then
            kill $API_PID
            log "Stopped FastAPI backend (PID: $API_PID)"
        fi
        rm infrastructure/streamlit/api.pid
    fi
    
    # Kill any remaining Streamlit processes
    pkill -f "streamlit run" 2>/dev/null || true
    
    log "Services stopped"
}

# Main deployment function
main() {
    log "Starting Streamlit deployment for $PROJECT_NAME"
    
    check_prerequisites
    setup_environment
    
    case "${1:-local}" in
        local)
            start_backend
            log "FastAPI backend: http://localhost:$API_PORT"
            log "Streamlit app starting..."
            log "Will be available at: http://localhost:$STREAMLIT_PORT"
            start_streamlit
            ;;
        cloud)
            deploy_to_cloud
            ;;
        docker)
            create_docker_setup
            ;;
        stop)
            stop_services
            ;;
        *)
            echo "Usage: $0 [local|cloud|docker|stop]"
            echo "  local  - Start locally with API backend"
            echo "  cloud  - Prepare for Streamlit Cloud deployment" 
            echo "  docker - Create Docker setup"
            echo "  stop   - Stop running services"
            exit 1
            ;;
    esac
}

# Handle script execution
main "$@"