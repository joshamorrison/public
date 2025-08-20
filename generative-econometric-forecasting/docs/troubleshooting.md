# Troubleshooting Guide

## Common Issues and Solutions

### Installation Issues

#### Missing Dependencies
**Problem**: `ModuleNotFoundError` when importing packages
**Solution**:
```bash
# Activate virtual environment
cd generative-econometric-forecasting
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install all dependencies
pip install -r requirements.txt
```

#### R Integration Errors
**Problem**: `rpy2` or R packages not found
**Solution**:
```bash
# Install R (required for advanced econometric models)
# Ubuntu/Debian:
sudo apt-get install r-base r-base-dev

# Windows: Download R from https://cran.r-project.org/
# Then install required R packages:
R -e "install.packages(c('vars', 'forecast', 'urca', 'tseries'), repos='http://cran.rstudio.com/')"

# Set R_HOME environment variable if needed
export R_HOME=/usr/lib/R  # Linux
# or
set R_HOME=C:\Program Files\R\R-4.3.0  # Windows
```

### API Issues

#### FRED API Key Problems
**Problem**: "No data available for indicator" errors
**Solution**:
1. Get a free FRED API key from https://fred.stlouisfed.org/
2. Set environment variable:
```bash
export FRED_API_KEY=your_actual_api_key_here
```
3. Or create `.env` file:
```bash
echo "FRED_API_KEY=your_actual_api_key_here" > .env
```

#### Rate Limit Exceeded
**Problem**: FRED API rate limit reached (120 requests per minute)
**Solution**:
- Wait 60 seconds and retry
- Use cached data for development: set `use_cache_fallback=True`
- For production, implement request queuing

### Model Performance Issues

#### Slow Forecasting
**Problem**: Forecasts taking too long to generate
**Solution**:
```python
# Use lighter models for development
from models.forecasting_models import EconometricForecaster
forecaster = EconometricForecaster(max_models=5)  # Limit ensemble size

# Or use tier 3 (fastest) models
request.model_tier = "tier3"
```

#### Memory Errors
**Problem**: Out of memory when training models
**Solution**:
```python
# Reduce data size
data = data.tail(500)  # Use last 500 observations only

# Or use memory-efficient models
from models.foundation_models.nixtla_opensource import NixtlaOSS
model = NixtlaOSS(memory_efficient=True)
```

### Data Quality Issues

#### Missing Data Points
**Problem**: Gaps in economic time series
**Solution**:
```python
# Forward fill missing values
data = data.fillna(method='ffill')

# Or interpolate
data = data.interpolate(method='linear')

# Check data quality
from data.fred_client import validate_data_quality
quality_report = validate_data_quality(data)
print(quality_report)
```

#### Irregular Frequencies
**Problem**: Data series with different frequencies (daily, monthly, quarterly)
**Solution**:
```python
# Resample to common frequency
monthly_data = data.resample('M').last()  # Convert to monthly

# Align multiple series
aligned_data = pd.concat([series1, series2], axis=1).fillna(method='ffill')
```

### Foundation Model Issues

#### TimeGPT Authentication
**Problem**: TimeGPT API errors
**Solution**:
```bash
# Set Nixtla API key
export NIXTLA_API_KEY=your_nixtla_api_key

# Check authentication
python -c "from models.foundation_models.timegpt_client import TimeGPTClient; client = TimeGPTClient(); print('Auth OK' if client.test_auth() else 'Auth Failed')"
```

#### HuggingFace Model Loading
**Problem**: HuggingFace models fail to load
**Solution**:
```python
# Clear HuggingFace cache
import shutil
shutil.rmtree('~/.cache/huggingface', ignore_errors=True)

# Use smaller models
from models.foundation_models.huggingface_forecaster import HybridFoundationEnsemble
ensemble = HybridFoundationEnsemble(model_size='small')
```

### LangSmith Monitoring Issues

#### Missing Traces
**Problem**: LangSmith traces not appearing
**Solution**:
```bash
# Check environment variables
echo $LANGCHAIN_API_KEY
echo $LANGSMITH_PROJECT

# Test connection
python scripts/test_langsmith_tracing.py
```

#### High LLM Costs
**Problem**: Unexpected high costs from LLM usage
**Solution**:
```python
# Monitor costs
from scripts.langsmith_enhanced_monitoring import CostMonitor
monitor = CostMonitor()
monitor.set_daily_limit(50.00)  # $50 daily limit

# Use cheaper models
import os
os.environ['OPENAI_MODEL'] = 'gpt-3.5-turbo'  # Instead of gpt-4
```

### Docker Issues

#### Container Won't Start
**Problem**: Docker container fails to start
**Solution**:
```bash
# Check logs
docker logs econometric-forecasting-api

# Rebuild with no cache
docker-compose build --no-cache

# Check port conflicts
netstat -an | grep 8000
```

#### R Not Found in Container
**Problem**: R packages not available in Docker
**Solution**:
```dockerfile
# Add to Dockerfile
RUN apt-get update && apt-get install -y r-base r-base-dev
RUN R -e "install.packages(c('vars', 'forecast', 'urca', 'tseries'))"
```

### Performance Optimization

#### Speed Up Forecasting
**Quick fixes**:
```python
# Use parallel processing
from multiprocessing import cpu_count
n_jobs = min(cpu_count(), 4)

# Reduce forecast horizon
horizon = min(horizon, 12)  # Max 12 months

# Cache model results
from functools import lru_cache
@lru_cache(maxsize=32)
def cached_forecast(indicator, start_date):
    return generate_forecast(indicator, start_date)
```

#### Reduce Memory Usage
```python
# Use generators for large datasets
def data_chunks(data, chunk_size=1000):
    for i in range(0, len(data), chunk_size):
        yield data[i:i + chunk_size]

# Clear unused variables
import gc
del large_dataframe
gc.collect()
```

## Debug Mode

Enable debug logging for detailed troubleshooting:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Or via environment
export LOG_LEVEL=DEBUG
```

## Getting Help

### Log Analysis
Check logs for specific error patterns:
```bash
# API logs
tail -f logs/api.log | grep ERROR

# Model performance logs
grep "processing_time" logs/api.log | tail -20
```

### System Information
Collect system info for support:
```python
import platform
import sys
import pkg_resources

print("Platform:", platform.platform())
print("Python:", sys.version)
print("Packages:")
for pkg in ['pandas', 'numpy', 'torch', 'langchain']:
    try:
        version = pkg_resources.get_distribution(pkg).version
        print(f"  {pkg}: {version}")
    except:
        print(f"  {pkg}: Not installed")
```

### Community Support
- **GitHub Issues**: Report bugs and feature requests
- **Documentation**: Check [API Reference](API_REFERENCE.md) and [Architecture](ARCHITECTURE.md)
- **Examples**: Review working examples in `examples/` folder

### Emergency Fallbacks
If all else fails, use basic statistical models:
```python
# Simple ARIMA fallback
from models.forecasting_models import EconometricForecaster
forecaster = EconometricForecaster()
result = forecaster.fit_arima(data, fallback_simple=True)
```

## Performance Benchmarks

Expected performance on standard hardware:
- **Single forecast**: 2-10 seconds
- **Batch forecasts (5 indicators)**: 30-60 seconds  
- **Complex scenario analysis**: 2-5 minutes
- **Memory usage**: 500MB-2GB typical

If your performance is significantly worse, check:
1. Available system memory
2. R installation completeness
3. Network connectivity to APIs
4. Model complexity settings