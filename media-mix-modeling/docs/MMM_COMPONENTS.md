# 🧩 Media Mix Modeling - Component Documentation

Detailed documentation for individual components of the MMM platform, covering each module's functionality, parameters, and integration patterns.

## 📊 Core Models

### EconometricMMM

**Location:** `models/mmm/econometric_mmm.py`

The primary media mix modeling engine that implements econometric principles for marketing attribution and optimization.

#### Core Features

##### Adstock Transformation
Models carryover effects of media spending across time periods.

```python
def _apply_adstock(self, spend_series, adstock_rate):
    """
    Apply adstock transformation to media spend
    
    Mathematical Formula:
    adstocked_spend[t] = spend[t] + adstock_rate * adstocked_spend[t-1]
    """
    adstocked = np.zeros_like(spend_series)
    adstocked[0] = spend_series[0]
    
    for t in range(1, len(spend_series)):
        adstocked[t] = spend_series[t] + adstock_rate * adstocked[t-1]
    
    return adstocked
```

**Parameters:**
- `adstock_rate` (float, 0-1): Controls strength of carryover effect
- Higher values = longer lasting impact of media spend

##### Saturation Curves
Implements diminishing returns at high spend levels using Hill transformation.

```python
def _apply_saturation(self, adstocked_spend, saturation_param):
    """
    Apply saturation transformation (Hill curve)
    
    Mathematical Formula:
    saturated_spend = spend^α / (spend^α + saturation_param^α)
    """
    alpha = 2.0  # Shape parameter
    return np.power(adstocked_spend, alpha) / (
        np.power(adstocked_spend, alpha) + np.power(saturation_param, alpha)
    )
```

**Parameters:**
- `saturation_param` (float): Controls saturation point
- Lower values = earlier saturation (diminishing returns)

##### Synergy Effects
Models cross-channel interaction effects and amplification.

```python
def _add_synergy_effects(self, transformed_spend):
    """
    Add interaction terms between channels
    Creates multiplicative effects between channel pairs
    """
    synergy_features = []
    channels = transformed_spend.columns
    
    for i, channel1 in enumerate(channels):
        for j, channel2 in enumerate(channels[i+1:], i+1):
            interaction = transformed_spend[channel1] * transformed_spend[channel2]
            synergy_features.append(interaction)
    
    return pd.concat([transformed_spend] + synergy_features, axis=1)
```

#### Model Training Process

```python
# Complete training workflow
def fit(self, data, target_column, spend_columns, include_synergies=False):
    """
    Training Steps:
    1. Data validation and preprocessing
    2. Apply adstock transformation to all spend columns
    3. Apply saturation curves to adstocked spend
    4. Add synergy effects (if enabled)
    5. Include baseline factors (seasonality, trend)
    6. Train Ridge regression model
    7. Calculate performance metrics
    8. Generate attribution analysis
    """
```

#### Performance Metrics

- **R² Score:** Coefficient of determination (model fit quality)
- **MAPE:** Mean Absolute Percentage Error (prediction accuracy)
- **RMSE:** Root Mean Square Error (prediction precision)

---

### BudgetOptimizer

**Location:** `models/mmm/budget_optimizer.py`

Optimization engine for allocating marketing budget across channels to maximize business objectives.

#### Optimization Objectives

```python
class OptimizationObjectives:
    ROI = 'roi'           # Maximize return on investment
    REVENUE = 'revenue'   # Maximize total revenue
    REACH = 'reach'       # Maximize audience reach
    EFFICIENCY = 'efficiency'  # Minimize cost per acquisition
```

#### Constraint Handling

```python
def _apply_constraints(self, allocation, constraints):
    """
    Apply business constraints to budget allocation
    
    Constraint Types:
    - min_spend: Minimum spend per channel
    - max_spend: Maximum spend per channel
    - spend_ratio: Relative spend ratios between channels
    - total_budget: Total budget constraint
    """
    for channel, constraint in constraints.items():
        if 'min' in constraint:
            allocation[channel] = max(allocation[channel], constraint['min'])
        if 'max' in constraint:
            allocation[channel] = min(allocation[channel], constraint['max'])
    
    return allocation
```

#### Optimization Algorithm

Uses scipy.optimize with gradient-based methods for efficient convergence.

```python
def optimize(self, mmm_model, total_budget, constraints=None):
    """
    Optimization Process:
    1. Define objective function using trained MMM model
    2. Set up constraint functions
    3. Run scipy.optimize.minimize with gradient descent
    4. Validate solution feasibility
    5. Return optimal allocation with performance projections
    """
```

---

## 📊 Data Integration

### MediaDataClient

**Location:** `data/media_data_client.py`

Progressive data integration client with fallback mechanisms across multiple data sources.

#### Data Source Hierarchy

```python
class DataSourcePriority:
    """Data source priority order (best to fallback)"""
    KAGGLE = 1        # Enterprise marketing analytics datasets
    HUGGINGFACE = 2   # Professional advertising datasets  
    SYNTHETIC = 3     # Generated marketing data (always available)
```

#### Data Validation

```python
def _validate_data_quality(self, data):
    """
    Data Quality Checks:
    1. Required columns present (spend columns, revenue)
    2. No null values in critical columns
    3. Positive values for spend and revenue
    4. Sufficient time series length (>= 26 weeks)
    5. Reasonable data ranges (spend/revenue ratios)
    """
    validation_results = {
        'columns_check': self._check_required_columns(data),
        'nulls_check': self._check_null_values(data),
        'values_check': self._check_positive_values(data),
        'length_check': len(data) >= 26,
        'ranges_check': self._check_reasonable_ranges(data)
    }
    
    return all(validation_results.values())
```

#### Synthetic Data Generation

**Location:** `data/synthetic/campaign_data_generator.py`

```python
class CampaignDataGenerator:
    """
    Generates realistic marketing mix data with:
    - Seasonal patterns
    - Cross-channel correlations
    - Realistic spend/revenue relationships
    - Noise and variance patterns
    """
    
    def generate_mmm_data(self, weeks=52, channels=5):
        """
        Generation Process:
        1. Create base seasonal patterns
        2. Generate correlated spend across channels
        3. Apply media effectiveness curves
        4. Add external factors (holidays, economic)
        5. Generate realistic revenue with noise
        """
```

---

## 🎯 Attribution Analysis

### AttributionAnalyzer

**Location:** `src/attribution/attribution_analyzer.py`

Multi-methodology attribution analysis for comprehensive channel evaluation.

#### Attribution Methods

```python
class AttributionMethods:
    """Available attribution methodologies"""
    
    LAST_TOUCH = 'last_touch'      # 100% credit to last touchpoint
    FIRST_TOUCH = 'first_touch'    # 100% credit to first touchpoint
    LINEAR = 'linear'              # Equal credit distribution
    TIME_DECAY = 'time_decay'      # Exponential decay by recency
    DATA_DRIVEN = 'data_driven'    # MMM-based attribution
```

#### Data-Driven Attribution

```python
def calculate_data_driven_attribution(self, mmm_model, data):
    """
    MMM-Based Attribution Process:
    1. Use trained MMM model coefficients
    2. Calculate incremental contribution per channel
    3. Account for adstock and saturation effects
    4. Normalize to total revenue attribution
    5. Apply confidence intervals
    """
    
    # Get model coefficients
    coefficients = mmm_model.model.coef_
    
    # Calculate base contributions
    contributions = {}
    for i, channel in enumerate(self.spend_columns):
        base_contribution = coefficients[i] * data[channel].sum()
        contributions[channel] = base_contribution
    
    # Normalize to 100%
    total_contribution = sum(contributions.values())
    attribution = {
        channel: contrib / total_contribution 
        for channel, contrib in contributions.items()
    }
    
    return attribution
```

---

## 🔄 Infrastructure Components

### dbt Integration

**Location:** `infrastructure/dbt/`

Data transformation pipeline using dbt for marketing data processing.

#### Model Structure

```sql
-- staging/stg_media_spend.sql
-- Standardize and clean media spend data
{{ config(materialized='view') }}

select
    date_trunc('week', spend_date) as week_start_date,
    channel_name,
    sum(spend_amount) as weekly_spend,
    sum(impressions) as weekly_impressions,
    sum(clicks) as weekly_clicks,
    current_timestamp() as processed_at
from {{ source('raw_data', 'media_spend') }}
where spend_date >= current_date - interval '2 years'
group by 1, 2
```

```sql
-- intermediate/int_attribution.sql  
-- Calculate attribution using multiple methods
{{ config(materialized='ephemeral') }}

with attribution_base as (
    select
        week_start_date,
        channel_name,
        weekly_spend,
        revenue,
        -- Linear attribution
        revenue / count(*) over (partition by week_start_date) as linear_attribution,
        -- Time decay attribution  
        revenue * exp(-0.1 * row_number() over (partition by week_start_date order by channel_name desc)) as time_decay_attribution
    from {{ ref('stg_media_spend') }}
    join {{ ref('stg_revenue') }} using (week_start_date)
)

select * from attribution_base
```

#### Data Quality Tests

```sql
-- tests/assert_positive_spend.sql
-- Ensure all spend values are positive
select *
from {{ ref('stg_media_spend') }}
where weekly_spend < 0
```

```sql
-- tests/assert_revenue_completeness.sql  
-- Ensure revenue data completeness
select week_start_date
from {{ ref('stg_revenue') }}
where revenue is null
```

### SageMaker Deployment

**Location:** `infrastructure/aws/sagemaker_deployment.py`

Production model serving on AWS SageMaker with auto-scaling and monitoring.

#### Model Serving Architecture

```python
class MMMInferenceHandler:
    """Custom inference handler for SageMaker"""
    
    def model_fn(self, model_dir):
        """Load model artifacts"""
        model_path = os.path.join(model_dir, 'model.joblib')
        metadata_path = os.path.join(model_dir, 'metadata.json')
        
        return {
            'model': joblib.load(model_path),
            'metadata': json.load(open(metadata_path))
        }
    
    def input_fn(self, request_body, request_content_type):
        """Parse incoming requests"""
        if request_content_type == 'application/json':
            data = json.loads(request_body)
            return pd.DataFrame(data)
        else:
            raise ValueError(f"Unsupported content type: {request_content_type}")
    
    def predict_fn(self, input_data, model_artifacts):
        """Generate predictions"""
        model = model_artifacts['model']
        
        try:
            predictions = model.predict(input_data)
            return {
                'predictions': predictions.tolist(),
                'model_version': model_artifacts['metadata']['version'],
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            return {'error': str(e)}
```

#### Auto-Scaling Configuration

```python
def setup_autoscaling(self, endpoint_name):
    """Configure SageMaker endpoint auto-scaling"""
    autoscaling_config = {
        'min_capacity': 1,
        'max_capacity': 10,
        'target_cpu_utilization': 70,
        'scale_out_cooldown': 300,  # 5 minutes
        'scale_in_cooldown': 300
    }
    
    # Register scalable target
    self.autoscaling_client.register_scalable_target(
        ServiceNamespace='sagemaker',
        ResourceId=f'endpoint/{endpoint_name}/variant/AllTraffic',
        ScalableDimension='sagemaker:variant:DesiredInstanceCount',
        **autoscaling_config
    )
```

---

## 📈 Reporting Components

### ExecutiveReporting

**Location:** `src/reports/executive_reporting.py`

Automated generation of executive-level reports and insights.

#### Report Types

```python
class ReportTypes:
    EXECUTIVE_SUMMARY = 'executive_summary'    # High-level business insights
    DETAILED_ANALYSIS = 'detailed_analysis'    # Complete MMM analysis
    OPTIMIZATION_RECOMMENDATIONS = 'optimization'  # Budget allocation recommendations
    PERFORMANCE_MONITORING = 'monitoring'      # Ongoing performance tracking
```

#### Report Generation Pipeline

```python
def generate_mmm_report(self, mmm_results, filename_prefix=None):
    """
    Report Generation Process:
    1. Extract key insights from MMM results
    2. Calculate business impact metrics
    3. Generate optimization recommendations
    4. Create visualizations and charts
    5. Export to multiple formats (JSON, CSV, PDF)
    """
    
    # Key insights extraction
    insights = self._extract_key_insights(mmm_results)
    
    # Business impact calculation
    impact_metrics = self._calculate_business_impact(mmm_results)
    
    # Recommendations generation
    recommendations = self._generate_recommendations(mmm_results)
    
    # Export reports
    report_files = {
        'executive_summary': self._export_executive_summary(insights, impact_metrics),
        'detailed_csv': self._export_detailed_csv(mmm_results),
        'recommendations': self._export_recommendations(recommendations)
    }
    
    return report_files
```

#### Insight Generation

```python
def _extract_key_insights(self, mmm_results):
    """
    Extract actionable insights from MMM results:
    - Top performing channels by ROI
    - Channels approaching saturation
    - Optimization opportunities
    - Attribution shifts vs. previous period
    """
    
    performance = mmm_results['performance']
    attribution = mmm_results.get('attribution', {})
    
    insights = {
        'model_quality': {
            'r2_score': performance['r2_score'],
            'accuracy_grade': self._grade_model_accuracy(performance['r2_score'])
        },
        'top_channels': self._identify_top_channels(attribution),
        'optimization_opportunities': self._identify_optimization_opportunities(mmm_results),
        'saturation_analysis': self._analyze_saturation_levels(mmm_results)
    }
    
    return insights
```

---

## 🧪 Testing Components

### Performance Testing

**Location:** `tests/test_performance.py`

Comprehensive performance benchmarking for production readiness.

#### Benchmark Categories

```python
class PerformanceBenchmarks:
    """Performance benchmark thresholds"""
    
    MAX_TRAINING_TIME_SECONDS = 30      # Model training time limit
    MAX_PREDICTION_TIME_MS = 100        # Prediction latency limit  
    MAX_MEMORY_USAGE_MB = 500          # Memory usage limit
    MIN_MODEL_ACCURACY = 0.7           # Minimum R² score
    MAX_OPTIMIZATION_TIME_SECONDS = 10  # Budget optimization time limit
```

#### Load Testing

```python
def test_concurrent_predictions(self, sample_marketing_data):
    """Test model thread safety and concurrent access"""
    
    def make_predictions():
        for _ in range(10):
            prediction = model.predict(test_data)
            results.put(prediction)
    
    # Start multiple threads
    threads = [threading.Thread(target=make_predictions) for _ in range(5)]
    [thread.start() for thread in threads]
    [thread.join() for thread in threads]
    
    # Verify consistent results
    all_predictions = list(results.queue)
    assert all(np.array_equal(pred, all_predictions[0]) for pred in all_predictions)
```

### Integration Testing

**Location:** `tests/test_integration.py`

End-to-end workflow testing for complete system validation.

```python
def test_complete_mmm_workflow(self):
    """Test complete MMM workflow from data loading to deployment"""
    
    # 1. Data loading
    client = MediaDataClient()
    data, info, source = client.get_best_available_data()
    
    # 2. Model training
    mmm = EconometricMMM()
    results = mmm.fit(data, 'revenue', spend_columns)
    
    # 3. Budget optimization
    optimizer = BudgetOptimizer()
    allocation = optimizer.optimize(mmm, 100000)
    
    # 4. Report generation
    reporter = ExecutiveReporting()
    reports = reporter.generate_mmm_report(results)
    
    # 5. Validation
    assert results['performance']['r2_score'] > 0
    assert 'allocation' in allocation
    assert len(reports) > 0
```

---

## 🔧 Configuration Components

### Environment Configuration

**Location:** `.env.example`

```bash
# Data Source Configuration
KAGGLE_USERNAME=your_kaggle_username
KAGGLE_KEY=your_kaggle_api_key
HF_TOKEN=your_huggingface_token

# AWS Configuration  
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key

# MLflow Configuration
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_EXPERIMENT_NAME=mmm-experiments

# Model Configuration
MMM_DEFAULT_ADSTOCK_RATE=0.5
MMM_DEFAULT_SATURATION_PARAM=0.6
MMM_DEFAULT_REGULARIZATION_ALPHA=0.1

# Deployment Configuration
SAGEMAKER_INSTANCE_TYPE=ml.m5.large
SAGEMAKER_INSTANCE_COUNT=1
SAGEMAKER_MAX_CAPACITY=5
```

### Model Parameters

**Location:** `models/mmm/config.py`

```python
class MMMConfig:
    """Default MMM model configuration"""
    
    # Adstock parameters
    DEFAULT_ADSTOCK_RATE = 0.5
    MIN_ADSTOCK_RATE = 0.0
    MAX_ADSTOCK_RATE = 0.9
    
    # Saturation parameters  
    DEFAULT_SATURATION_PARAM = 0.6
    MIN_SATURATION_PARAM = 0.1
    MAX_SATURATION_PARAM = 2.0
    
    # Regularization parameters
    DEFAULT_REGULARIZATION_ALPHA = 0.1
    MIN_REGULARIZATION_ALPHA = 0.001
    MAX_REGULARIZATION_ALPHA = 1.0
    
    # Training parameters
    MIN_TRAINING_WEEKS = 26
    RECOMMENDED_TRAINING_WEEKS = 52
    MAX_TRAINING_WEEKS = 208  # 4 years
    
    # Performance thresholds
    MIN_ACCEPTABLE_R2 = 0.5
    TARGET_R2 = 0.8
    MAX_ACCEPTABLE_MAPE = 20.0
```

---

## 🔍 Monitoring Components

### Model Performance Monitoring

**Location:** `src/monitoring/model_monitor.py`

```python
class ModelPerformanceMonitor:
    """Monitor model performance and drift in production"""
    
    def check_model_drift(self, recent_predictions, historical_performance):
        """
        Detect model performance drift:
        1. Compare recent accuracy vs. historical baseline
        2. Analyze prediction distribution shifts
        3. Monitor feature importance changes
        4. Alert on significant performance degradation
        """
        
        drift_metrics = {
            'accuracy_drift': self._calculate_accuracy_drift(recent_predictions, historical_performance),
            'distribution_drift': self._calculate_distribution_drift(recent_predictions),
            'feature_drift': self._calculate_feature_importance_drift(recent_predictions)
        }
        
        # Alert if significant drift detected
        if any(metric > self.drift_threshold for metric in drift_metrics.values()):
            self._send_drift_alert(drift_metrics)
        
        return drift_metrics
```

### Business Metrics Monitoring

```python
class BusinessMetricsMonitor:
    """Monitor business impact of MMM implementation"""
    
    def track_roi_improvement(self, current_period, baseline_period):
        """
        Track business impact metrics:
        - ROAS improvement vs. baseline
        - CAC reduction
        - Revenue attribution accuracy
        - Budget optimization effectiveness
        """
        
        roi_metrics = {
            'roas_improvement': self._calculate_roas_improvement(current_period, baseline_period),
            'cac_reduction': self._calculate_cac_reduction(current_period, baseline_period),
            'attribution_accuracy': self._calculate_attribution_accuracy(current_period),
            'optimization_effectiveness': self._calculate_optimization_effectiveness(current_period)
        }
        
        return roi_metrics
```

---

*Component Documentation version: 1.0.0 | Last updated: 2025-08-17*