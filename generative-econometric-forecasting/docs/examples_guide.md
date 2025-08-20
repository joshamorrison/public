# Examples Guide

## Overview

This guide provides comprehensive examples for using the Generative Econometric Forecasting Platform. Examples are organized by complexity and use case, from basic forecasting to advanced economic analysis.

## Quick Start Examples

### Basic GDP Forecast
```python
from data.fred_client import FredDataClient
from models.forecasting_models import EconometricForecaster

# Initialize clients
fred_client = FredDataClient()
forecaster = EconometricForecaster()

# Fetch GDP data
gdp_data = fred_client.fetch_indicator('gdp', start_date='2020-01-01')

# Generate 12-month forecast
forecast = forecaster.fit_arima(gdp_data)
results = forecaster.generate_forecast(forecast['model_key'], periods=12)

print(f"Next quarter GDP forecast: ${results['forecast'][0]:.1f}B")
```

### Multi-Indicator Dashboard
```python
from data.fred_client import FredDataClient
import matplotlib.pyplot as plt

# Get dashboard data
client = FredDataClient()
dashboard_data = client.get_economic_dashboard_data(start_date='2020-01-01')

# Plot key indicators
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# GDP
dashboard_data['gdp'].plot(ax=axes[0,0], title='Real GDP')
axes[0,0].set_ylabel('Billions $')

# Unemployment
dashboard_data['unemployment'].plot(ax=axes[0,1], title='Unemployment Rate', color='red')
axes[0,1].set_ylabel('Percent')

# Inflation
dashboard_data['inflation'].plot(ax=axes[1,0], title='Inflation Rate', color='orange')
axes[1,0].set_ylabel('Percent')

# Interest Rates
dashboard_data['interest_rate'].plot(ax=axes[1,1], title='10-Year Treasury Rate', color='green')
axes[1,1].set_ylabel('Percent')

plt.tight_layout()
plt.show()
```

## Foundation Model Examples

### Using TimeGPT (Tier 1)
```python
from models.foundation_models.timegpt_client import TimeGPTClient

# Initialize TimeGPT client (requires API key)
timegpt = TimeGPTClient(api_key='your_nixtla_api_key')

# Fetch and forecast unemployment
unemployment_data = fred_client.fetch_indicator('unemployment', start_date='2020-01-01')

# Generate forecast with confidence intervals
forecast_result = timegpt.forecast(
    data=unemployment_data,
    horizon=12,
    confidence_intervals=[80, 95]
)

print("TimeGPT Unemployment Forecast:")
print(forecast_result['forecast'].head())
print(f"Model confidence: {forecast_result['model_info']['confidence']}")
```

## API Integration Examples

### REST API Usage
```python
import requests
import json

# API base URL
API_BASE = "http://localhost:8000"

# Single forecast request
forecast_request = {
    "indicators": ["gdp", "unemployment"],
    "horizon": 12,
    "method": "foundation",
    "model_tier": "auto",
    "confidence_interval": 0.95,
    "generate_report": True
}

response = requests.post(
    f"{API_BASE}/api/v1/forecast/single",
    json=forecast_request
)

if response.status_code == 200:
    result = response.json()
    print(f"Forecast completed in {result['processing_time']:.2f}s")
    
    for forecast in result['forecasts']:
        indicator = forecast['indicator']
        next_value = forecast['forecast_points'][0]['value']
        print(f"{indicator.upper()}: {next_value:.2f}")
else:
    print(f"API Error: {response.status_code}")
    print(response.json())
```

These examples provide a comprehensive foundation for using the platform across different scenarios and complexity levels.