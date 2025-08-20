# Troubleshooting Guide

## 🚨 Common Issues and Solutions

This guide covers common issues you might encounter when working with the Media Mix Modeling platform and their solutions.

## Installation Issues

### Python Environment Problems

**Issue**: Package conflicts or version incompatibility
```bash
ERROR: pip's dependency resolver does not currently consider all the packages that are installed
```

**Solutions**:
1. **Use virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Clear pip cache**:
   ```bash
   pip cache purge
   pip install --no-cache-dir -r requirements.txt
   ```

3. **Use conda environment**:
   ```bash
   conda create -n mmm-env python=3.9
   conda activate mmm-env
   pip install -r requirements.txt
   ```

### R Integration Issues

**Issue**: R packages not found or rpy2 installation fails
```bash
ModuleNotFoundError: No module named 'rpy2'
```

**Solutions**:
1. **Install R first**:
   ```bash
   # Ubuntu/Debian
   sudo apt-get install r-base r-base-dev
   
   # macOS
   brew install r
   
   # Windows: Download from https://cran.r-project.org/
   ```

2. **Install required R packages**:
   ```r
   install.packages(c("prophet", "CausalImpact", "bayesm"))
   ```

3. **Install rpy2**:
   ```bash
   pip install rpy2
   ```

## Data Issues

### Data Loading Problems

**Issue**: Data files not found
```bash
FileNotFoundError: [Errno 2] No such file or directory: 'data/samples/campaign_data.csv'
```

**Solutions**:
1. **Check file paths**:
   ```python
   import os
   print(os.getcwd())  # Verify working directory
   print(os.path.exists('data/samples/campaign_data.csv'))
   ```

2. **Run from project root**:
   ```bash
   cd /path/to/media-mix-modeling
   python quick_start.py
   ```

3. **Generate sample data**:
   ```python
   from src.synthetic.campaign_data_generator import generate_sample_data
   generate_sample_data()
   ```

### Data Format Issues

**Issue**: Incorrect data format or missing columns
```bash
KeyError: 'spend' column not found in dataframe
```

**Solutions**:
1. **Check data schema**:
   ```python
   import pandas as pd
   df = pd.read_csv('data/samples/campaign_data.csv')
   print(df.columns.tolist())
   print(df.dtypes)
   ```

2. **Validate against schema**:
   ```python
   from data.schemas import validate_campaign_data
   validate_campaign_data(df)
   ```

3. **Required columns for MMM**:
   - `date`: Date column (YYYY-MM-DD format)
   - `channel`: Media channel name
   - `spend`: Media spend amount
   - `impressions`: Number of impressions
   - `clicks`: Number of clicks
   - `conversions`: Number of conversions

## Model Training Issues

### Memory Issues

**Issue**: Out of memory during model training
```bash
MemoryError: Unable to allocate array with shape (1000000, 1000)
```

**Solutions**:
1. **Reduce batch size**:
   ```python
   model_config = {
       'batch_size': 1000,  # Reduce from default
       'max_iterations': 1000
   }
   ```

2. **Use data sampling**:
   ```python
   df_sample = df.sample(frac=0.5)  # Use 50% of data
   ```

3. **Increase swap space** (Linux/macOS):
   ```bash
   sudo swapon --show
   sudo fallocate -l 4G /swapfile
   ```

### Model Convergence Issues

**Issue**: Model fails to converge
```bash
Warning: Model did not converge after 1000 iterations
```

**Solutions**:
1. **Increase iterations**:
   ```python
   model_config['max_iterations'] = 5000
   ```

2. **Adjust learning rate**:
   ```python
   model_config['learning_rate'] = 0.001  # Reduce from default
   ```

3. **Check data quality**:
   ```python
   # Remove outliers
   df = df[df['spend'] < df['spend'].quantile(0.99)]
   
   # Handle missing values
   df = df.fillna(method='forward')
   ```

## API Issues

### FastAPI Server Problems

**Issue**: Server won't start
```bash
ImportError: cannot import name 'Annotated' from 'typing'
```

**Solutions**:
1. **Update Python version** (requires Python 3.9+):
   ```bash
   python --version  # Should be 3.9 or higher
   ```

2. **Install typing extensions**:
   ```bash
   pip install typing-extensions
   ```

3. **Check port availability**:
   ```bash
   lsof -i :8000  # Check if port 8000 is in use
   ```

### Database Connection Issues

**Issue**: Database connection fails
```bash
sqlalchemy.exc.OperationalError: (psycopg2.OperationalError) connection to server failed
```

**Solutions**:
1. **Check database status**:
   ```bash
   docker ps  # If using Docker
   pg_isready -h localhost -p 5432  # For PostgreSQL
   ```

2. **Verify connection settings**:
   ```python
   # Check .env file
   DATABASE_URL=postgresql://user:password@localhost:5432/mmm_db
   ```

3. **Reset database**:
   ```bash
   docker-compose down
   docker-compose up -d postgres
   ```

## Docker Issues

### Container Build Failures

**Issue**: Docker build fails
```bash
ERROR [3/5] RUN pip install -r requirements.txt
```

**Solutions**:
1. **Clear Docker cache**:
   ```bash
   docker system prune -a
   docker build --no-cache -t mmm-platform .
   ```

2. **Check Dockerfile syntax**:
   ```dockerfile
   # Ensure proper base image
   FROM python:3.9-slim
   ```

3. **Build with verbose output**:
   ```bash
   docker build --progress=plain -t mmm-platform .
   ```

### Container Runtime Issues

**Issue**: Container exits immediately
```bash
docker: Error response from daemon: container exited with code 1
```

**Solutions**:
1. **Check logs**:
   ```bash
   docker logs <container-id>
   ```

2. **Run interactively**:
   ```bash
   docker run -it mmm-platform /bin/bash
   ```

3. **Check environment variables**:
   ```bash
   docker run --env-file .env mmm-platform
   ```

## Performance Issues

### Slow Model Training

**Issue**: Model training takes too long

**Solutions**:
1. **Use parallel processing**:
   ```python
   import multiprocessing
   n_jobs = multiprocessing.cpu_count() - 1
   ```

2. **Enable GPU acceleration** (if available):
   ```python
   import tensorflow as tf
   print(tf.config.list_physical_devices('GPU'))
   ```

3. **Optimize data loading**:
   ```python
   # Use chunked reading for large files
   chunks = pd.read_csv('large_file.csv', chunksize=10000)
   ```

### Memory Leaks

**Issue**: Memory usage grows over time

**Solutions**:
1. **Monitor memory usage**:
   ```python
   import psutil
   import os
   process = psutil.Process(os.getpid())
   print(f"Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")
   ```

2. **Clear variables explicitly**:
   ```python
   del large_dataframe
   import gc
   gc.collect()
   ```

## Deployment Issues

### AWS Deployment Problems

**Issue**: AWS credentials not configured
```bash
NoCredentialsError: Unable to locate credentials
```

**Solutions**:
1. **Configure AWS CLI**:
   ```bash
   aws configure
   ```

2. **Use environment variables**:
   ```bash
   export AWS_ACCESS_KEY_ID=your_key
   export AWS_SECRET_ACCESS_KEY=your_secret
   export AWS_DEFAULT_REGION=us-east-1
   ```

3. **Use IAM roles** (recommended for EC2):
   ```bash
   # Attach appropriate IAM role to EC2 instance
   ```

### Airflow Issues

**Issue**: DAG not appearing in Airflow UI
```bash
DAG 'mmm_pipeline' not found
```

**Solutions**:
1. **Check DAG syntax**:
   ```bash
   python dags/mmm_pipeline.py  # Should run without errors
   ```

2. **Restart Airflow**:
   ```bash
   docker-compose restart airflow-webserver airflow-scheduler
   ```

3. **Check Airflow logs**:
   ```bash
   docker logs airflow-scheduler
   ```

## Getting Help

### Log Analysis

**Enable detailed logging**:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
```

### Debug Mode

**Run in debug mode**:
```bash
export DEBUG=true
python quick_start.py
```

### Community Support

1. **Check GitHub Issues**: Search existing issues for solutions
2. **Stack Overflow**: Tag questions with `media-mix-modeling`
3. **Documentation**: Refer to API documentation and examples

### Reporting Bugs

When reporting bugs, include:
1. **Environment details**:
   ```bash
   python --version
   pip freeze > requirements-actual.txt
   ```

2. **Error messages**: Full stack trace
3. **Sample data**: Minimal reproducible example
4. **Steps to reproduce**: Clear sequence of actions

## Quick Diagnostic Commands

```bash
# System information
python --version
pip --version
docker --version

# Package versions
pip show pandas numpy scikit-learn

# Disk space
df -h

# Memory usage
free -h

# Port usage
netstat -tulpn | grep :8000

# Process monitoring
top -p $(pgrep -f python)
```

## Emergency Procedures

### Complete Reset

If all else fails:
```bash
# Stop all services
docker-compose down

# Remove containers and volumes
docker-compose down -v

# Clean Python cache
find . -type d -name __pycache__ -exec rm -r {} +
find . -name "*.pyc" -delete

# Recreate virtual environment
rm -rf venv
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Restart services
docker-compose up -d
python quick_start.py
```

This should resolve most issues and get you back to a working state.