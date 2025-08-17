# API Reference

Complete reference for all classes, functions, and command-line interfaces.

## Command Line Interface

### quick_start.py
Main entry point for the platform demonstration and basic usage.

```bash
python quick_start.py [OPTIONS]
```

#### Options
- `--indicators TEXT`: Economic indicators to forecast (space-separated)
- `--forecast-horizon INT`: Number of periods to forecast (default: 12)
- `--start-date TEXT`: Start date for historical data (YYYY-MM-DD)
- `--output-dir TEXT`: Directory to save outputs (default: outputs)
- `--industry TEXT`: Industry context for demand planning (default: retail)
- `--include-demand-planning`: Include GenAI demand planning analysis (default: True)
- `--no-demand-planning`: Skip demand planning analysis
- `--no-save`: Do not save outputs to files

#### Examples
```bash
# Complete analysis with demand planning
python quick_start.py --indicators gdp unemployment inflation

# Manufacturing industry focus
python quick_start.py --indicators gdp unemployment --industry manufacturing

# Economic forecasting only
python quick_start.py --indicators gdp unemployment --no-demand-planning
```

## Core Data Classes

### FredDataClient
FRED API integration for economic data retrieval.

```python
from data.fred_client import FredDataClient

client = FredDataClient(api_key="your_fred_key")
```

#### Methods

##### `fetch_indicator(indicator, start_date=None, end_date=None)`
Fetch economic indicator data from FRED.

**Parameters:**
- `indicator` (str): Economic indicator code ('gdp', 'unemployment', 'inflation')
- `start_date` (str, optional): Start date in 'YYYY-MM-DD' format
- `end_date` (str, optional): End date in 'YYYY-MM-DD' format

**Returns:**
- `pandas.Series`: Time series data with datetime index

**Example:**
```python
gdp_data = client.fetch_indicator('gdp', start_date='2010-01-01')
print(f"Latest GDP: {gdp_data.iloc[-1]}")
```

##### `validate_data(series, min_periods=24)`
Validate time series data quality.

**Parameters:**
- `series` (pandas.Series): Time series to validate
- `min_periods` (int): Minimum required observations

**Returns:**
- `dict`: Validation results with quality metrics

### NewsClient
News data collection and economic relevance filtering.

```python
from data.unstructured.news_client import NewsClient

news_client = NewsClient(newsapi_key="your_key")
```

#### Methods

##### `fetch_rss_feeds(max_articles_per_feed=10)`
Fetch recent economic news from RSS feeds.

**Parameters:**
- `max_articles_per_feed` (int): Maximum articles per RSS feed

**Returns:**
- `list`: List of news article dictionaries

##### `filter_economic_articles(articles, min_relevance_score=0.1)`
Filter articles for economic relevance.

**Parameters:**
- `articles` (list): List of news articles
- `min_relevance_score` (float): Minimum relevance threshold

**Returns:**
- `list`: Filtered list of economically relevant articles

## Forecasting Models

### EconometricForecaster
Main forecasting engine with multiple model support.

```python
from models.forecasting_models import EconometricForecaster

forecaster = EconometricForecaster()
```

#### Methods

##### `fit_arima(series, max_p=5, max_d=2, max_q=5)`
Fit ARIMA model with automatic order selection.

**Parameters:**
- `series` (pandas.Series): Time series data
- `max_p` (int): Maximum AR order
- `max_d` (int): Maximum differencing order  
- `max_q` (int): Maximum MA order

**Returns:**
- `dict`: Model results with fitted model and diagnostics

##### `fit_prophet(series, yearly_seasonality=True, weekly_seasonality=False)`
Fit Prophet forecasting model.

**Parameters:**
- `series` (pandas.Series): Time series data
- `yearly_seasonality` (bool): Include yearly seasonal component
- `weekly_seasonality` (bool): Include weekly seasonal component

**Returns:**
- `dict`: Model results with fitted model and components

##### `generate_forecast(model_key, periods=12, confidence_level=0.95)`
Generate forecast from fitted model.

**Parameters:**
- `model_key` (str): Model identifier from fitting methods
- `periods` (int): Number of periods to forecast
- `confidence_level` (float): Confidence level for intervals

**Returns:**
- `dict`: Forecast with values, confidence intervals, and metadata

### NeuralModelEnsemble
Neural network forecasting with multiple model types.

```python
from models.neural_forecasting import NeuralModelEnsemble

ensemble = NeuralModelEnsemble()
```

#### Methods

##### `get_available_models()`
Get list of available neural models.

**Returns:**
- `list`: Available model names

##### `fit_ensemble(data, target_column, model_names=None)`
Fit ensemble of neural models.

**Parameters:**
- `data` (pandas.DataFrame): Time series data
- `target_column` (str): Name of target variable column
- `model_names` (list, optional): Specific models to use

**Returns:**
- `dict`: Fitted ensemble with model weights and performance

##### `predict(horizon=12, return_intervals=True)`
Generate ensemble forecast.

**Parameters:**
- `horizon` (int): Number of periods to forecast
- `return_intervals` (bool): Include confidence intervals

**Returns:**
- `dict`: Ensemble forecast with individual model contributions

### SentimentAdjustedForecaster
Sentiment-enhanced forecasting with news integration.

```python
from models.sentiment_adjusted_forecasting import SentimentAdjustedForecaster

forecaster = SentimentAdjustedForecaster(
    sentiment_weight=0.1,
    sentiment_decay=0.8
)
```

#### Methods

##### `get_current_sentiment(days_back=7)`
Analyze current economic sentiment from news.

**Parameters:**
- `days_back` (int): Days of news history to analyze

**Returns:**
- `dict`: Sentiment metrics and analysis

##### `adjust_forecast(base_forecast, indicator, sentiment_data=None)`
Adjust forecast using sentiment analysis.

**Parameters:**
- `base_forecast` (numpy.ndarray): Original forecast values
- `indicator` (str): Economic indicator name
- `sentiment_data` (dict, optional): Pre-calculated sentiment data

**Returns:**
- `dict`: Adjusted forecast with sentiment impact analysis

## Reporting Classes

### SimpleEconomicReporter
Multi-format report generation system.

```python
from src.reports.simple_reporting import SimpleEconomicReporter

reporter = SimpleEconomicReporter(output_dir="outputs/reports")
```

#### Methods

##### `generate_reports(economic_data, forecast_results, sentiment_analysis=None, ai_analysis=None)`
Generate comprehensive reports in multiple formats.

**Parameters:**
- `economic_data` (dict): Dictionary of economic time series
- `forecast_results` (dict): Forecasting results from models
- `sentiment_analysis` (dict, optional): Sentiment analysis results
- `ai_analysis` (str, optional): AI-generated analysis text

**Returns:**
- `dict`: Dictionary with paths to generated report files

##### `save_json_report(data, filename)`
Save structured data as JSON report.

**Parameters:**
- `data` (dict): Report data to save
- `filename` (str): Output filename

**Returns:**
- `str`: Path to saved JSON file

##### `save_csv_summary(data, filename)`
Save forecast summary as CSV.

**Parameters:**
- `data` (dict): Forecast data to summarize
- `filename` (str): Output filename

**Returns:**
- `str`: Path to saved CSV file

## Foundation Models

### HybridFoundationEnsemble
Multi-tier foundation model system.

```python
from models.foundation_models.huggingface_forecaster import HybridFoundationEnsemble

ensemble = HybridFoundationEnsemble(
    nixtla_api_key="your_key",
    include_nixtla_oss=True,
    prefer_paid=True
)
```

#### Methods

##### `forecast(series, horizon=12, frequency='M')`
Generate forecast using best available model.

**Parameters:**
- `series` (pandas.Series): Time series data
- `horizon` (int): Forecast horizon
- `frequency` (str): Data frequency ('M', 'Q', 'D')

**Returns:**
- `dict`: Forecast results with model information

##### `get_model_hierarchy()`
Get current model tier hierarchy.

**Returns:**
- `list`: Ordered list of available models by preference

## Environment Variables

### Required API Keys
```bash
# Economic data access
FRED_API_KEY=your_fred_api_key_here

# AI-powered analysis (optional)
OPENAI_API_KEY=your_openai_api_key_here

# Premium forecasting (optional)
NIXTLA_API_KEY=your_nixtla_api_key_here

# LangSmith monitoring (optional)
LANGCHAIN_API_KEY=your_langchain_api_key_here
```

### Configuration Settings
```bash
# Default forecasting parameters
DEFAULT_FORECAST_HORIZON=12
DEFAULT_INDICATORS=gdp,unemployment,inflation
ENSEMBLE_MODELS=arima,prophet

# AI settings
NARRATIVE_MODEL=gpt-3.5-turbo
NARRATIVE_TEMPERATURE=0.3
SCENARIO_COUNT=3

# Output configuration
OUTPUT_DIRECTORY=outputs
SAVE_CHARTS=True
SAVE_JSON_REPORTS=True
SAVE_EXECUTIVE_SUMMARIES=True
```

## Economic Indicators

### Supported Indicators

| Indicator | FRED Code | Description | Frequency |
|-----------|-----------|-------------|-----------|
| `gdp` | GDPC1 | Real Gross Domestic Product | Quarterly |
| `unemployment` | UNRATE | Unemployment Rate | Monthly |
| `inflation` | CPIAUCSL | Consumer Price Index | Monthly |
| `interest_rate` | DGS10 | 10-Year Treasury Rate | Daily |
| `consumer_confidence` | UMCSENT | Consumer Sentiment Index | Monthly |
| `housing_starts` | HOUST | Housing Starts | Monthly |
| `industrial_production` | INDPRO | Industrial Production Index | Monthly |
| `retail_sales` | RSAFS | Retail Sales | Monthly |

### Custom Indicators
```python
# Add custom FRED series
custom_indicators = {
    'labor_force': 'CLF16OV',
    'real_earnings': 'AHETPI',
    'capacity_utilization': 'TCU'
}

client = FredDataClient(api_key="your_key")
for name, fred_code in custom_indicators.items():
    data = client.fetch_series(fred_code)
    # Process custom indicator
```

## Error Handling

### Common Exceptions

#### `FredAPIError`
Raised when FRED API requests fail.

```python
try:
    data = fred_client.fetch_indicator('gdp')
except FredAPIError as e:
    print(f"FRED API error: {e}")
    # Use cached data or synthetic fallback
```

#### `ModelFittingError`
Raised when model fitting fails.

```python
try:
    model = forecaster.fit_arima(series)
except ModelFittingError as e:
    print(f"Model fitting failed: {e}")
    # Try alternative model or preprocessing
```

#### `InsufficientDataError`
Raised when time series has insufficient observations.

```python
try:
    forecast = forecaster.generate_forecast(model_key, periods=12)
except InsufficientDataError as e:
    print(f"Insufficient data: {e}")
    # Reduce forecast horizon or get more data
```

### Error Recovery Patterns

#### Graceful Degradation
```python
def robust_forecast(series, horizon=12):
    try:
        # Try premium TimeGPT
        return timegpt_forecast(series, horizon)
    except APIError:
        try:
            # Fall back to Nixtla OSS
            return nixtla_oss_forecast(series, horizon)
        except ImportError:
            # Final fallback to simple methods
            return exponential_smoothing(series, horizon)
```

#### Retry Logic
```python
import time
from functools import wraps

def retry_on_failure(max_retries=3, delay=1):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise e
                    time.sleep(delay * (2 ** attempt))
            return None
        return wrapper
    return decorator
```

## Performance Optimization

### Batch Processing
```python
# Process multiple indicators efficiently
indicators = ['gdp', 'unemployment', 'inflation']
batch_results = {}

for indicator in indicators:
    data = fred_client.fetch_indicator(indicator)
    batch_results[indicator] = forecaster.fit_arima(data)

# Generate all forecasts
forecasts = {
    indicator: forecaster.generate_forecast(result['model_key'], 12)
    for indicator, result in batch_results.items()
}
```

### Caching
```python
import pickle
from pathlib import Path

def cache_model(model, indicator, cache_dir="cache"):
    """Cache fitted model for reuse."""
    Path(cache_dir).mkdir(exist_ok=True)
    cache_file = Path(cache_dir) / f"{indicator}_model.pkl"
    with open(cache_file, 'wb') as f:
        pickle.dump(model, f)

def load_cached_model(indicator, cache_dir="cache"):
    """Load cached model if available."""
    cache_file = Path(cache_dir) / f"{indicator}_model.pkl"
    if cache_file.exists():
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    return None
```

### Memory Management
```python
import gc

def memory_efficient_ensemble(data, models):
    """Run ensemble with memory cleanup."""
    results = []
    
    for model_name in models:
        # Fit individual model
        model = fit_model(data, model_name)
        results.append(model.predict())
        
        # Clean up model from memory
        del model
        gc.collect()
    
    return combine_results(results)
```