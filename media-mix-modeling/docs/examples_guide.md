# Examples Guide

## 📚 Comprehensive Guide to Media Mix Modeling Examples

This guide walks through all the examples provided with the Media Mix Modeling platform, from basic use cases to advanced enterprise scenarios.

## Getting Started

Before running examples, ensure you have:
1. **Installed dependencies**: `pip install -r requirements.txt`
2. **Sample data available**: Check `data/samples/` directory
3. **Environment configured**: Copy `.env.example` to `.env`

## Examples Structure

```
examples/
├── basic_examples/          # Simple, single-concept demos
│   ├── simple_attribution.py      # Basic attribution modeling
│   └── budget_optimizer.py        # Simple budget optimization
├── advanced_examples/       # Complex, multi-step scenarios  
│   └── cross_channel_synergy.py   # Advanced cross-channel analysis
└── integration_examples/    # Real-world integration patterns
    └── api_integration_example.py # API usage patterns
```

## Basic Examples

### 1. Simple Attribution Analysis

**File**: `examples/basic_examples/simple_attribution.py`

**What it demonstrates**:
- Loading campaign data
- Basic attribution modeling
- Calculating channel contributions
- Visualizing results

**Usage**:
```bash
cd examples/basic_examples
python simple_attribution.py
```

**Key concepts**:
- Data preparation for MMM
- Attribution model selection
- Performance metrics calculation
- Basic visualization

**Expected output**:
- Channel contribution percentages
- Attribution scores by channel
- Simple visualization plots
- Performance summary statistics

### 2. Budget Optimization

**File**: `examples/basic_examples/budget_optimizer.py`

**What it demonstrates**:
- Budget allocation optimization
- ROI calculation by channel
- Constraint-based optimization
- Budget reallocation recommendations

**Usage**:
```bash
cd examples/basic_examples  
python budget_optimizer.py
```

**Key concepts**:
- Optimization algorithms
- Budget constraints
- ROI maximization
- Sensitivity analysis

**Expected output**:
- Optimized budget allocation
- ROI improvement projections
- Channel-wise budget recommendations
- Performance lift estimates

## Advanced Examples

### 1. Cross-Channel Synergy Analysis

**File**: `examples/advanced_examples/cross_channel_synergy.py`

**What it demonstrates**:
- Multi-touch attribution modeling
- Channel interaction effects
- Synergy quantification
- Advanced statistical modeling

**Usage**:
```bash
cd examples/advanced_examples
python cross_channel_synergy.py
```

**Key concepts**:
- Cross-channel effects
- Interaction modeling
- Advanced attribution methods
- Synergy measurement

**Expected output**:
- Channel synergy matrix
- Interaction effect quantification
- Optimized channel combinations
- Advanced performance metrics

**Sample output**:
```
Channel Synergy Analysis Results:
=====================================
Direct interactions found:
- Social Media + Search: +23% lift
- Display + Video: +18% lift  
- Email + Social: +15% lift

Recommended channel combinations:
1. Social + Search + Email: 34% total lift
2. Display + Video + Social: 28% total lift
```

## Integration Examples

### 1. API Integration Example

**File**: `examples/integration_examples/api_integration_example.py`

**What it demonstrates**:
- FastAPI endpoint usage
- Programmatic model training
- Automated reporting
- Real-time predictions

**Usage**:
```bash
# Start API server first
uvicorn api.main:app --reload

# Run integration example
cd examples/integration_examples
python api_integration_example.py
```

**Key concepts**:
- REST API integration
- Asynchronous processing
- Error handling
- Response parsing

## Running Examples with Custom Data

### Using Your Own Data

1. **Prepare your data** in the required format:
   ```python
   # Required columns
   df.columns = ['date', 'channel', 'spend', 'impressions', 'clicks', 'conversions']
   ```

2. **Modify example scripts** to point to your data:
   ```python
   # In any example script
   data_path = 'path/to/your/data.csv'
   df = pd.read_csv(data_path)
   ```

3. **Validate data format**:
   ```python
   from src.data_utils import validate_mmm_data
   validate_mmm_data(df)
   ```

### Example with Custom Data

```python
import pandas as pd
from examples.basic_examples.simple_attribution import run_attribution_analysis

# Load your data
custom_data = pd.read_csv('your_campaign_data.csv')

# Run attribution analysis
results = run_attribution_analysis(custom_data)
print(f"Attribution results: {results}")
```

## Example Configurations

### Model Configuration Options

```python
# Basic configuration
basic_config = {
    'model_type': 'linear',
    'attribution_method': 'first_touch',
    'time_window': 30
}

# Advanced configuration
advanced_config = {
    'model_type': 'bayesian',
    'attribution_method': 'data_driven',
    'time_window': 60,
    'include_interactions': True,
    'seasonality': True,
    'trend_adjustment': True
}
```

### Data Requirements

**Minimum required data**:
- At least 90 days of historical data
- Daily or weekly granularity
- Minimum 3 media channels
- Spend and outcome metrics

**Optimal data setup**:
- 2+ years of historical data  
- Daily granularity preferred
- 5+ media channels
- Multiple outcome metrics
- External factors (seasonality, promotions)

## Troubleshooting Examples

### Common Issues

1. **Data loading errors**:
   ```python
   # Debug data issues
   import pandas as pd
   df = pd.read_csv('data/samples/campaign_data.csv')
   print(f"Data shape: {df.shape}")
   print(f"Columns: {df.columns.tolist()}")
   print(f"Date range: {df['date'].min()} to {df['date'].max()}")
   ```

2. **Model convergence issues**:
   ```python
   # Adjust model parameters
   model_config = {
       'max_iterations': 5000,  # Increase iterations
       'learning_rate': 0.001,  # Reduce learning rate
       'regularization': 0.1    # Add regularization
   }
   ```

3. **Memory issues with large datasets**:
   ```python
   # Process data in chunks
   chunk_size = 10000
   for chunk in pd.read_csv('large_file.csv', chunksize=chunk_size):
       process_chunk(chunk)
   ```

## Performance Benchmarks

### Expected Runtime

| Example | Dataset Size | Runtime | Memory Usage |
|---------|--------------|---------|--------------|
| Simple Attribution | 1K rows | 2-5 seconds | 50MB |
| Budget Optimizer | 5K rows | 10-30 seconds | 100MB |
| Cross-Channel Synergy | 10K rows | 1-5 minutes | 200MB |

### Optimization Tips

1. **Use sampling for exploration**:
   ```python
   # Sample 10% of data for quick testing
   df_sample = df.sample(frac=0.1)
   ```

2. **Cache intermediate results**:
   ```python
   import joblib
   
   # Save processed data
   joblib.dump(processed_data, 'cache/processed_data.pkl')
   
   # Load cached data
   processed_data = joblib.load('cache/processed_data.pkl')
   ```

3. **Parallel processing**:
   ```python
   from multiprocessing import Pool
   
   with Pool() as pool:
       results = pool.map(process_channel, channels)
   ```

## Creating Custom Examples

### Example Template

```python
"""
Custom MMM Example Template
"""
import pandas as pd
import numpy as np
from src.attribution.attribution_engine import AttributionEngine
from src.optimization.budget_optimizer import BudgetOptimizer

def main():
    """Main example function."""
    
    # 1. Load and prepare data
    print("Loading data...")
    data = load_example_data()
    
    # 2. Initialize models
    print("Initializing models...")
    attribution_model = AttributionEngine()
    optimizer = BudgetOptimizer()
    
    # 3. Run analysis
    print("Running analysis...")
    results = run_analysis(data, attribution_model, optimizer)
    
    # 4. Display results
    print("Results:")
    display_results(results)
    
    return results

def load_example_data():
    """Load and validate example data."""
    # Your data loading logic here
    pass

def run_analysis(data, attribution_model, optimizer):
    """Run the main analysis."""
    # Your analysis logic here
    pass

def display_results(results):
    """Display analysis results."""
    # Your result display logic here
    pass

if __name__ == "__main__":
    results = main()
```

### Best Practices for Custom Examples

1. **Clear documentation**: Add docstrings and comments
2. **Error handling**: Include try/catch blocks
3. **Validation**: Validate inputs and outputs
4. **Visualization**: Include plots and charts
5. **Modularity**: Break into reusable functions

## Integration Patterns

### Jupyter Notebook Integration

```python
# Install Jupyter support
pip install jupyter

# Start Jupyter
jupyter notebook examples/

# Run examples interactively
%run basic_examples/simple_attribution.py
```

### API Integration Patterns

```python
import requests

# Basic API call
response = requests.post('http://localhost:8000/attribution/analyze', 
                        json={'data': data_dict})

# Async API usage
import asyncio
import aiohttp

async def run_analysis():
    async with aiohttp.ClientSession() as session:
        async with session.post('http://localhost:8000/attribution/analyze',
                               json={'data': data_dict}) as response:
            return await response.json()
```

### Airflow DAG Integration

```python
from airflow import DAG
from airflow.operators.python_operator import PythonOperator

def run_mmm_example():
    from examples.basic_examples.simple_attribution import main
    return main()

dag = DAG('mmm_example_pipeline', schedule_interval='@daily')

example_task = PythonOperator(
    task_id='run_attribution_example',
    python_callable=run_mmm_example,
    dag=dag
)
```

## Next Steps

After working through the examples:

1. **Modify examples** with your own data
2. **Combine techniques** from multiple examples
3. **Create custom workflows** based on your needs
4. **Integrate with your systems** using API patterns
5. **Scale up** to production environments

For more advanced usage, see:
- **API Reference**: Complete API documentation
- **Architecture Guide**: System design patterns
- **Deployment Guide**: Production deployment
- **Business Applications**: Industry-specific use cases

## Getting Help

- **GitHub Issues**: Report problems or request features
- **Documentation**: Comprehensive API and system docs
- **Examples**: Working code demonstrations
- **Community**: Discussion forums and support channels