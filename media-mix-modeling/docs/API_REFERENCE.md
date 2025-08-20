# 📚 Media Mix Modeling - API Reference

Complete API documentation for the MMM platform components, models, and integrations.

## ⚡ FastAPI Endpoints

### Starting the Server

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

Access interactive documentation at: http://localhost:8000/docs

### Health Endpoints

#### GET /health
Returns service status and dependency health checks.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-08-20T13:11:54.123456",
  "dependencies": {
    "pandas": "available",
    "sklearn": "available",
    "mlflow": "available"
  }
}
```

### Attribution Endpoints

#### POST /attribution/analyze
Performs multi-touch attribution analysis on customer journey data.

**Request Body:**
```json
{
  "customer_journeys": [...],
  "attribution_model": "time_decay",
  "lookback_window": 30,
  "include_view_through": true
}
```

#### POST /attribution/incrementality
Runs incrementality testing to measure causal impact of campaigns.

### Optimization Endpoints

#### POST /optimization/optimize
Multi-objective budget optimization across channels.

**Request Body:**
```json
{
  "current_budget": {
    "tv": 45000,
    "digital": 30000,
    "social": 15000
  },
  "constraints": {
    "total_budget": 100000,
    "min_channel_spend": 5000
  },
  "objective": "roi"
}
```

### Performance Endpoints

#### POST /performance/analyze
Comprehensive performance analysis with trend detection and anomaly identification.

## 🏗️ Core Models

### EconometricMMM

The primary media mix modeling class with adstock, saturation, and synergy effects.

```python
from models.mmm.econometric_mmm import EconometricMMM

# Initialize model
mmm = EconometricMMM(
    adstock_rate=0.5,
    saturation_param=0.6,
    regularization_alpha=0.1,
    include_baseline=True
)
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `adstock_rate` | float | 0.5 | Carryover effect rate (0-1) |
| `saturation_param` | float | 0.6 | Diminishing returns parameter |
| `regularization_alpha` | float | 0.1 | Ridge regression regularization |
| `include_baseline` | bool | True | Include baseline revenue |

#### Methods

##### `fit(data, target_column, spend_columns, include_synergies=False)`

Train the media mix model on historical data.

**Parameters:**
- `data` (DataFrame): Historical marketing and revenue data
- `target_column` (str): Target variable column name (e.g., 'revenue')
- `spend_columns` (list): List of media spend column names
- `include_synergies` (bool): Include cross-channel interaction effects

**Returns:**
- `dict`: Model results containing performance metrics and analysis

**Example:**
```python
# Prepare data
spend_columns = ['tv_spend', 'digital_spend', 'radio_spend']

# Fit model
results = mmm.fit(
    data=marketing_data,
    target_column='revenue',
    spend_columns=spend_columns,
    include_synergies=True
)

# Access performance
r2_score = results['performance']['r2_score']
mape = results['performance']['mape']
```

##### `predict(data)`

Generate revenue predictions for new media spend scenarios.

**Parameters:**
- `data` (DataFrame): Media spend data for prediction

**Returns:**
- `numpy.ndarray`: Predicted revenue values

##### `get_attribution(data)`

Calculate channel attribution for media spend.

**Parameters:**
- `data` (DataFrame): Historical media spend data

**Returns:**
- `dict`: Attribution percentages by channel

---

### BudgetOptimizer

Optimize budget allocation across media channels for maximum ROI.

```python
from models.mmm.budget_optimizer import BudgetOptimizer

# Initialize optimizer
optimizer = BudgetOptimizer(
    objective='roi',
    max_iterations=1000,
    tolerance=1e-6
)
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `objective` | str | 'roi' | Optimization objective ('roi', 'revenue', 'reach') |
| `max_iterations` | int | 1000 | Maximum optimization iterations |
| `tolerance` | float | 1e-6 | Convergence tolerance |

#### Methods

##### `optimize(mmm_model, total_budget, constraints=None)`

Optimize budget allocation across channels.

**Parameters:**
- `mmm_model`: Trained EconometricMMM instance
- `total_budget` (float): Total budget to allocate
- `constraints` (dict): Channel-specific constraints

**Returns:**
- `dict`: Optimized allocation and performance projections

**Example:**
```python
# Set constraints
constraints = {
    'tv_spend': {'min': 5000, 'max': 50000},
    'digital_spend': {'min': 10000, 'max': 60000}
}

# Optimize budget
result = optimizer.optimize(
    mmm_model=trained_mmm,
    total_budget=100000,
    constraints=constraints
)

# Access results
optimal_allocation = result['allocation']
projected_roi = result['projected_roi']
```

---

## 📊 Data Integration

### MediaDataClient

Handle multi-source data integration with progressive enhancement.

```python
from data.media_data_client import MediaDataClient

# Initialize client
client = MediaDataClient()
```

#### Methods

##### `get_best_available_data()`

Retrieve the highest quality data source available.

**Returns:**
- `tuple`: (data_df, data_info, source_type)

**Data Source Priority:**
1. Kaggle Enterprise Marketing Analytics
2. HuggingFace Professional Datasets
3. Synthetic Marketing Data (fallback)

##### `get_kaggle_data(dataset_name)`

Load marketing data from Kaggle.

**Parameters:**
- `dataset_name` (str): Kaggle dataset identifier

**Returns:**
- `DataFrame`: Marketing campaign data

##### `get_synthetic_data(weeks=52, channels=5)`

Generate synthetic marketing mix data.

**Parameters:**
- `weeks` (int): Number of weeks to generate
- `channels` (int): Number of media channels

**Returns:**
- `DataFrame`: Synthetic marketing data

---

## 🏭 Attribution Models

### AttributionAnalyzer

Advanced attribution modeling with multiple methodologies.

```python
from src.attribution.attribution_analyzer import AttributionAnalyzer

# Initialize analyzer
analyzer = AttributionAnalyzer()
```

#### Methods

##### `run_attribution_analysis(data, spend_columns, revenue_column)`

Run comprehensive attribution analysis.

**Parameters:**
- `data` (DataFrame): Historical marketing data
- `spend_columns` (list): Media spend column names
- `revenue_column` (str): Revenue column name

**Returns:**
- `dict`: Attribution results across multiple methods

**Attribution Methods:**
- Last-touch attribution
- First-touch attribution  
- Linear attribution
- Time-decay attribution
- Data-driven attribution (MMM-based)

---

## ☁️ AWS Integration

### MMMSageMakerDeployment

Deploy MMM models to AWS SageMaker for production serving.

```python
from infrastructure.aws.sagemaker_deployment import MMMSageMakerDeployment

# Initialize deployment
deployer = MMMSageMakerDeployment(
    region_name='us-east-1',
    bucket_name='my-mmm-bucket'
)
```

#### Methods

##### `prepare_model_artifacts(mmm_model, model_name)`

Prepare model for SageMaker deployment.

**Parameters:**
- `mmm_model`: Trained MMM model instance
- `model_name` (str): Model identifier

**Returns:**
- `str`: S3 URI of uploaded model artifacts

##### `deploy_model(s3_model_uri, model_name, endpoint_name=None)`

Deploy model to SageMaker endpoint.

**Parameters:**
- `s3_model_uri` (str): S3 location of model artifacts
- `model_name` (str): Model identifier
- `endpoint_name` (str): Custom endpoint name

**Returns:**
- `dict`: Deployment configuration and endpoint details

---

## 🔄 dbt Integration

### DBTIntegration

Integrate with dbt for data transformation and attribution modeling.

```python
from src.dbt_integration.dbt_runner import DBTIntegration

# Initialize dbt integration
dbt = DBTIntegration(project_dir='./infrastructure/dbt')
```

#### Methods

##### `run_attribution_models()`

Execute dbt attribution models.

**Returns:**
- `dict`: Model execution results

##### `run_data_quality_tests()`

Run dbt data quality tests.

**Returns:**
- `dict`: Test results and data quality metrics

---

## 📈 Reporting

### ExecutiveReporting

Generate executive-level reports and summaries.

```python
from src.reports.executive_reporting import ExecutiveReporting

# Initialize reporting
reporter = ExecutiveReporting()
```

#### Methods

##### `generate_mmm_report(mmm_results, filename_prefix=None)`

Generate comprehensive MMM analysis report.

**Parameters:**
- `mmm_results` (dict): Results from MMM model fitting
- `filename_prefix` (str): Custom filename prefix

**Returns:**
- `dict`: Generated report file paths

**Generated Files:**
- Executive summary (JSON)
- Detailed CSV report
- Executive text summary

---

## 🧪 Testing Utilities

### Test Fixtures

Common test fixtures for MMM testing.

```python
import pytest
from tests.conftest import sample_marketing_data

@pytest.fixture
def sample_marketing_data():
    """Generate sample marketing data for testing"""
    # Returns standardized test dataset
```

### Performance Testing

```python
from tests.test_performance import benchmark_mmm_training

def test_mmm_performance():
    """Test MMM training performance"""
    # Benchmark model training time
```

---

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `KAGGLE_USERNAME` | Kaggle API username | None |
| `KAGGLE_KEY` | Kaggle API key | None |
| `HF_TOKEN` | HuggingFace API token | None |
| `AWS_REGION` | AWS deployment region | us-east-1 |
| `MLFLOW_TRACKING_URI` | MLflow server URI | None |

### Model Parameters

```python
# Default MMM configuration
MMM_DEFAULT_CONFIG = {
    'adstock_rate': 0.5,
    'saturation_param': 0.6,
    'regularization_alpha': 0.1,
    'include_baseline': True,
    'max_iterations': 1000
}
```

---

## 📊 Response Formats

### Model Performance Metrics

```json
{
  "performance": {
    "r2_score": 0.85,
    "mape": 8.2,
    "rmse": 1250.5,
    "mean_actual": 50000.0,
    "mean_predicted": 49800.0
  }
}
```

### Attribution Results

```json
{
  "attribution": {
    "tv_spend": 0.35,
    "digital_spend": 0.28,
    "radio_spend": 0.20,
    "print_spend": 0.10,
    "social_spend": 0.07
  },
  "method": "data_driven"
}
```

### Budget Optimization

```json
{
  "allocation": {
    "tv_spend": 35000,
    "digital_spend": 30000,
    "radio_spend": 20000,
    "print_spend": 10000,
    "social_spend": 5000
  },
  "projected_roi": 1.85,
  "projected_revenue": 185000,
  "optimization_objective": "roi"
}
```

---

## 🚨 Error Handling

### Common Exceptions

#### `MMModelError`
Base exception for MMM-related errors.

#### `DataIntegrationError`
Raised when data source integration fails.

#### `OptimizationError`
Raised during budget optimization failures.

#### `DeploymentError`
Raised during AWS deployment issues.

### Error Response Format

```json
{
  "error": {
    "type": "MMModelError",
    "message": "Model training failed: insufficient data",
    "code": "MMM_001",
    "timestamp": "2025-08-17T22:15:00Z"
  }
}
```

---

## 🔗 Integration Examples

### End-to-End Workflow

```python
from data.media_data_client import MediaDataClient
from models.mmm.econometric_mmm import EconometricMMM
from models.mmm.budget_optimizer import BudgetOptimizer

# 1. Load data
client = MediaDataClient()
data, info, source = client.get_best_available_data()

# 2. Train model
mmm = EconometricMMM()
results = mmm.fit(
    data=data,
    target_column='revenue',
    spend_columns=['tv_spend', 'digital_spend', 'radio_spend'],
    include_synergies=True
)

# 3. Optimize budget
optimizer = BudgetOptimizer(objective='roi')
allocation = optimizer.optimize(
    mmm_model=mmm,
    total_budget=100000
)

# 4. Generate report
from src.reports.executive_reporting import ExecutiveReporting
reporter = ExecutiveReporting()
report_files = reporter.generate_mmm_report(results)
```

---

## 📝 Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2025-08-17 | Initial API release |

---

*For implementation details, see [ARCHITECTURE.md](ARCHITECTURE.md)*
*For deployment instructions, see [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)*