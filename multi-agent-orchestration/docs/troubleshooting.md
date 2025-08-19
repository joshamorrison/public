# Troubleshooting Guide

## 🔍 Common Issues & Solutions

This guide covers common problems you might encounter when setting up, developing, or deploying the Multi-Agent Orchestration Platform.

## 🚀 Setup Issues

### **Installation Problems**

#### **Python Version Compatibility**
```bash
# Error: Python version not supported
# Solution: Check Python version
python --version  # Should be 3.8+

# Install correct Python version
pyenv install 3.9.0
pyenv global 3.9.0
```

#### **Dependency Installation Failures**
```bash
# Error: pip install fails with build errors
# Solution: Update pip and install build tools
pip install --upgrade pip setuptools wheel

# On Ubuntu/Debian
sudo apt-get install python3-dev build-essential

# On macOS
xcode-select --install

# On Windows
# Install Microsoft C++ Build Tools
```

#### **Virtual Environment Issues**
```bash
# Error: Virtual environment not activating
# Solution: Create fresh virtual environment
rm -rf .venv
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows
```

### **Environment Configuration**

#### **Missing Environment Variables**
```bash
# Error: KeyError: 'LANGCHAIN_API_KEY'
# Solution: Copy and configure .env file
cp .env.example .env
# Edit .env file with your actual API keys

# Verify environment variables are loaded
python -c "import os; print(os.getenv('LANGCHAIN_API_KEY'))"
```

#### **Database Connection Issues**
```bash
# Error: Connection to database failed
# Solution: Check database URL format
DATABASE_URL=postgresql://username:password@host:port/database

# Test database connection
psql $DATABASE_URL -c "SELECT 1;"

# For SQLite (development)
DATABASE_URL=sqlite:///./agents.db
```

## 🔧 Development Issues

### **Import Errors**

#### **Module Not Found**
```bash
# Error: ModuleNotFoundError: No module named 'src'
# Solution: Add project root to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Or run from project root
python -m src.main

# In Python scripts
import sys
sys.path.append('.')
from src.agents.base_agent import BaseAgent
```

#### **Circular Import Issues**
```python
# Error: ImportError: cannot import name 'AgentA' from partially initialized module
# Solution: Use lazy imports or restructure code

# Instead of:
from src.agents.agent_b import AgentB

# Use:
def get_agent_b():
    from src.agents.agent_b import AgentB
    return AgentB()
```

### **Agent Execution Issues**

#### **Agent Not Responding**
```python
# Error: Agent hangs indefinitely
# Solution: Add timeout and retry logic

import asyncio

async def execute_with_timeout(agent, task, timeout=60):
    try:
        return await asyncio.wait_for(
            agent.process_task(task), 
            timeout=timeout
        )
    except asyncio.TimeoutError:
        print(f"Agent timed out after {timeout} seconds")
        return None
```

#### **Low Confidence Scores**
```python
# Issue: Agent consistently returns low confidence
# Solution: Improve prompts and add validation

class ImprovedAgent(BaseAgent):
    async def process_task(self, task):
        # Add task validation
        if not self.validate_task(task):
            return AgentResult(
                content="Invalid task format",
                confidence=0.0
            )
        
        # Improve prompt engineering
        enhanced_prompt = self.enhance_prompt(task)
        result = await self.llm.process(enhanced_prompt)
        
        # Add confidence calibration
        confidence = self.calibrate_confidence(result, task)
        
        return AgentResult(
            content=result,
            confidence=confidence
        )
```

### **Performance Issues**

#### **Slow Agent Responses**
```python
# Issue: Agents taking too long to respond
# Solution: Optimize prompts and add caching

import functools
import hashlib

@functools.lru_cache(maxsize=128)
def cached_llm_call(prompt_hash, prompt):
    return llm.generate(prompt)

class OptimizedAgent(BaseAgent):
    async def process_task(self, task):
        # Create prompt hash for caching
        prompt = self.create_prompt(task)
        prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
        
        # Use cached result if available
        result = cached_llm_call(prompt_hash, prompt)
        
        return AgentResult(content=result, confidence=0.85)
```

#### **Memory Usage Issues**
```python
# Issue: High memory consumption
# Solution: Implement proper cleanup

import gc
import weakref

class MemoryEfficientAgent(BaseAgent):
    def __init__(self):
        super().__init__()
        self._cache = weakref.WeakValueDictionary()
    
    async def process_task(self, task):
        try:
            result = await self._execute_task(task)
            return result
        finally:
            # Force garbage collection
            gc.collect()
```

## 🐳 Docker Issues

### **Container Build Problems**

#### **Docker Build Fails**
```bash
# Error: Docker build fails with dependency errors
# Solution: Check Dockerfile and requirements.txt

# Dockerfile debugging
docker build --no-cache -t debug-image .

# Check requirements.txt format
pip-compile requirements.in
```

#### **Container Won't Start**
```bash
# Error: Container exits immediately
# Solution: Check logs and entrypoint

# View container logs
docker logs container-name

# Run container in interactive mode
docker run -it --entrypoint /bin/bash image-name

# Check file permissions
ls -la /app/
chmod +x /app/entrypoint.sh
```

### **Container Runtime Issues**

#### **Port Binding Problems**
```bash
# Error: Port already in use
# Solution: Find and kill process or use different port

# Find process using port
lsof -i :8000
netstat -tulpn | grep 8000

# Kill process
kill -9 PID

# Use different port
docker run -p 8001:8000 image-name
```

#### **Volume Mounting Issues**
```bash
# Error: Volume mount not working
# Solution: Check paths and permissions

# Verify absolute paths
docker run -v /absolute/path:/app/data image-name

# Fix permissions
sudo chown -R $USER:$USER ./data
chmod -R 755 ./data
```

## ☁️ AWS Deployment Issues

### **ECS Service Problems**

#### **Service Won't Start**
```bash
# Error: ECS service stuck in PENDING state
# Solution: Check task definition and logs

# View service events
aws ecs describe-services --cluster cluster-name --services service-name

# Check task logs
aws logs get-log-events --log-group-name /ecs/multi-agent --log-stream-name stream-name
```

#### **Load Balancer Health Check Failures**
```bash
# Error: ALB health checks failing
# Solution: Verify health endpoint and security groups

# Test health endpoint locally
curl http://localhost:8000/health

# Check security group rules
aws ec2 describe-security-groups --group-ids sg-xxxxxx

# Verify health check configuration
aws elbv2 describe-target-health --target-group-arn arn:aws:elasticloadbalancing:...
```

### **Database Connection Issues**

#### **RDS Connection Timeout**
```bash
# Error: Database connection timeout
# Solution: Check security groups and network configuration

# Test connectivity from ECS task
aws ecs execute-command --cluster cluster-name --task task-id --command "telnet rds-endpoint 5432"

# Check RDS security group
aws rds describe-db-instances --db-instance-identifier db-name
```

#### **Connection Pool Exhaustion**
```python
# Error: Too many database connections
# Solution: Configure connection pooling

from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool

engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,
    pool_recycle=3600
)
```

## 🔐 Authentication Issues

### **API Key Problems**

#### **Invalid API Key Errors**
```bash
# Error: 401 Unauthorized
# Solution: Verify API key format and permissions

# Check API key format
echo $LANGCHAIN_API_KEY | wc -c  # Should be expected length

# Test API key
curl -H "Authorization: Bearer $LANGCHAIN_API_KEY" https://api.smith.langchain.com/

# Rotate API key if compromised
# Generate new key from provider dashboard
```

#### **Rate Limiting Issues**
```python
# Error: 429 Too Many Requests
# Solution: Implement exponential backoff

import asyncio
import random

async def retry_with_backoff(func, max_retries=3):
    for attempt in range(max_retries):
        try:
            return await func()
        except RateLimitError:
            if attempt == max_retries - 1:
                raise
            
            wait_time = (2 ** attempt) + random.uniform(0, 1)
            await asyncio.sleep(wait_time)
```

## 📊 Monitoring Issues

### **LangSmith Integration Problems**

#### **Traces Not Appearing**
```python
# Issue: LangSmith traces not showing up
# Solution: Verify configuration and connectivity

import os
from langchain.callbacks import LangChainTracer

# Check environment variables
print(f"Tracing enabled: {os.getenv('LANGCHAIN_TRACING_V2')}")
print(f"Project: {os.getenv('LANGCHAIN_PROJECT')}")

# Test tracer
tracer = LangChainTracer()
print(f"Tracer initialized: {tracer is not None}")
```

#### **High Trace Volume**
```python
# Issue: Too many traces causing performance issues
# Solution: Implement sampling

import random

class SamplingTracer:
    def __init__(self, sample_rate=0.1):
        self.sample_rate = sample_rate
        self.tracer = LangChainTracer() if random.random() < sample_rate else None
    
    def trace_run(self, *args, **kwargs):
        if self.tracer:
            return self.tracer.trace_run(*args, **kwargs)
```

## 🔄 Workflow Issues

### **Pattern Execution Problems**

#### **Pipeline Stages Failing**
```python
# Issue: Pipeline breaks at specific stage
# Solution: Add error handling and recovery

class RobustPipeline:
    async def execute_stage(self, stage, data):
        max_retries = 3
        for attempt in range(max_retries):
            try:
                return await stage.process(data)
            except Exception as e:
                if attempt == max_retries - 1:
                    # Log failure and return degraded result
                    logger.error(f"Stage {stage.name} failed: {e}")
                    return self.create_fallback_result(stage, data)
                
                await asyncio.sleep(2 ** attempt)
```

#### **Supervisor Coordination Issues**
```python
# Issue: Supervisor can't coordinate specialists
# Solution: Improve task delegation logic

class ImprovedSupervisor:
    async def delegate_task(self, task, specialists):
        # Validate specialists are available
        available = [s for s in specialists if await s.is_available()]
        
        if not available:
            raise NoAvailableSpecialistsError("No specialists available")
        
        # Select best specialist for task
        best_specialist = self.select_specialist(task, available)
        
        # Delegate with timeout and fallback
        try:
            return await asyncio.wait_for(
                best_specialist.process_task(task),
                timeout=300
            )
        except asyncio.TimeoutError:
            # Try with next best specialist
            return await self.fallback_delegation(task, available[1:])
```

## 📋 Diagnostic Commands

### **System Health Check**
```bash
#!/bin/bash
echo "=== System Health Check ==="

# Python environment
echo "Python version: $(python --version)"
echo "Virtual environment: $VIRTUAL_ENV"

# Dependencies
echo "Key packages:"
pip show langchain langsmith fastapi

# Environment variables
echo "Environment:"
echo "LANGCHAIN_TRACING_V2: $LANGCHAIN_TRACING_V2"
echo "DATABASE_URL: ${DATABASE_URL:0:20}..."

# Network connectivity
echo "Network tests:"
curl -s -o /dev/null -w "LangSmith API: %{http_code}\n" https://api.smith.langchain.com/

# Database
echo "Database connection:"
python -c "
import os
from sqlalchemy import create_engine
try:
    engine = create_engine(os.getenv('DATABASE_URL'))
    with engine.connect() as conn:
        conn.execute('SELECT 1')
    print('Database: Connected')
except Exception as e:
    print(f'Database: Error - {e}')
"
```

### **Performance Profiling**
```python
# Profile agent performance
import cProfile
import pstats

def profile_agent(agent, task):
    profiler = cProfile.Profile()
    profiler.enable()
    
    result = asyncio.run(agent.process_task(task))
    
    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(10)
    
    return result
```

## 📞 Getting Help

### **Log Collection**
```bash
# Collect comprehensive logs for support
mkdir -p debug-logs
docker logs multi-agent-app > debug-logs/app.log 2>&1
docker logs multi-agent-db > debug-logs/db.log 2>&1
docker logs multi-agent-redis > debug-logs/redis.log 2>&1

# System info
docker info > debug-logs/docker-info.txt
python --version > debug-logs/python-info.txt
pip freeze > debug-logs/requirements.txt

# Create support bundle
tar -czf debug-bundle-$(date +%Y%m%d).tar.gz debug-logs/
```

### **Support Channels**
- **GitHub Issues**: [Report bugs and feature requests](https://github.com/joshamorrison/public/issues)
- **Email Support**: [joshamorrison@gmail.com](mailto:joshamorrison@gmail.com)
- **LinkedIn**: [linkedin.com/in/joshamorrison](https://www.linkedin.com/in/joshamorrison)

When reporting issues, please include:
- Python version and operating system
- Complete error messages and stack traces
- Steps to reproduce the problem
- Relevant configuration (with sensitive data redacted)

---

This troubleshooting guide covers the most common issues. For deployment-specific problems, see [deployment_guide.md](deployment_guide.md).