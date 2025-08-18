# 🏗️ Media Mix Modeling - System Architecture

Comprehensive architecture documentation for the MMM platform, covering design principles, component interactions, and deployment patterns.

## 🎯 Architecture Overview

The Media Mix Modeling platform follows a **layered architecture** with clear separation of concerns, enabling scalability, maintainability, and production deployment.

```
┌─────────────────────────────────────────────────────────────┐
│                    🌐 Presentation Layer                     │
├─────────────────────────────────────────────────────────────┤
│  📊 Executive Reports  │  🔌 REST APIs  │  📈 Dashboards    │
└─────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────┐
│                    🧠 Business Logic Layer                   │
├─────────────────────────────────────────────────────────────┤
│  🎯 MMM Models  │  💰 Optimization  │  📊 Attribution     │
└─────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────┐
│                    📊 Data Integration Layer                 │
├─────────────────────────────────────────────────────────────┤
│  🏢 Kaggle APIs  │  🤗 HuggingFace  │  🔄 dbt Models     │
└─────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────┐
│                    ☁️ Infrastructure Layer                  │
├─────────────────────────────────────────────────────────────┤
│  🚀 AWS Services  │  🌪️ Airflow  │  📦 Docker  │  🔧 MLflow │
└─────────────────────────────────────────────────────────────┘
```

## 📁 Project Structure

```
media-mix-modeling/
├── 🚀 quick_start.py              # Entry point & demo
├── 📋 requirements.txt            # Dependencies
├── ⚙️ pyproject.toml             # Package configuration
├── 🧪 pytest.ini                 # Test configuration
│
├── 📊 data/                       # Data integration layer
│   ├── 🔌 media_data_client.py    # Multi-source data client
│   ├── 📱 google_ads_client.py    # Google Ads integration
│   ├── 📘 facebook_ads_client.py  # Facebook Ads integration
│   └── 🎲 synthetic/              # Synthetic data generation
│       └── 📈 campaign_data_generator.py
│
├── 🧠 models/                     # Core modeling layer
│   ├── 📊 mmm/                    # Media mix models
│   │   ├── 🎯 econometric_mmm.py  # Primary MMM model
│   │   ├── 📊 attribution_models.py # Attribution analysis
│   │   └── 💰 budget_optimizer.py # Budget optimization
│   └── 📈 r_integration/          # R statistical models
│       └── 🔬 r_mmm_models.py
│
├── 🔧 src/                        # Application logic
│   ├── 📊 attribution/            # Attribution modeling
│   ├── 💰 optimization/           # Budget optimization
│   ├── 🔄 dbt_integration/        # dbt transformations
│   ├── 📈 reports/                # Executive reporting
│   └── 🔌 mlflow_integration.py   # MLflow tracking
│
├── ☁️ infrastructure/             # Deployment & orchestration
│   ├── 🚀 aws/                    # AWS deployment
│   │   ├── 📦 deploy_to_aws.py    # Main deployment script
│   │   └── 🎯 sagemaker_deployment.py # SageMaker integration
│   ├── 🌪️ airflow/               # Workflow orchestration
│   │   └── 📅 dags/               # Airflow DAGs
│   └── 🔄 dbt/                    # Data transformations
│       ├── 📊 models/             # dbt models
│       └── 🧪 tests/              # Data quality tests
│
├── 🧪 tests/                      # Test suite
│   ├── 🔧 conftest.py             # Test configuration
│   ├── 🎯 test_mmm_models.py      # Model tests
│   ├── 📊 test_data_client.py     # Data integration tests
│   ├── 💰 test_budget_optimization.py # Optimization tests
│   ├── ☁️ test_aws_deployment.py  # AWS deployment tests
│   └── 🔄 test_integration.py     # End-to-end tests
│
├── 📚 docs/                       # Documentation
│   ├── 📖 API_REFERENCE.md        # API documentation
│   ├── 🏗️ ARCHITECTURE.md        # This file
│   ├── 🚀 DEPLOYMENT_GUIDE.md     # Deployment instructions
│   └── 📊 BUSINESS_APPLICATIONS.md # Business use cases
│
├── 📈 outputs/                    # Generated artifacts
│   └── 📊 reports/                # Executive reports
│
└── 🏭 production/                 # Production deployment
    ├── 🎯 mmm_production_model.joblib # Trained model
    ├── 📊 deployment_metadata.json    # Model metadata
    └── 📋 deployment_summary.json     # Deployment info
```

## 🔧 Core Components

### 1. 📊 Data Integration Layer

**Progressive Enhancement Strategy:**
```python
# Data source priority hierarchy
1. 🏆 Kaggle Enterprise Marketing Analytics (Premium)
2. 🥇 HuggingFace Professional Datasets (Good)  
3. 🥈 Synthetic Marketing Data (Fallback)
```

**Key Components:**
- **MediaDataClient**: Orchestrates multi-source data integration
- **Progressive Fallback**: Automatically downgrades to available data sources
- **Data Validation**: Ensures data quality and consistency
- **Schema Standardization**: Normalizes data across sources

### 2. 🧠 Media Mix Modeling Engine

**Econometric Foundation:**
```python
# MMM model architecture
Revenue = f(
    adstock_transformed_spend,     # Carryover effects
    saturation_curves,             # Diminishing returns
    synergy_interactions,          # Cross-channel effects
    baseline_factors              # Organic/non-media drivers
)
```

**Core Features:**
- **Adstock Transformation**: Models carryover effects between periods
- **Saturation Curves**: Captures diminishing returns at high spend levels
- **Synergy Effects**: Models cross-channel interaction effects
- **Regularization**: Ridge regression prevents overfitting

### 3. 💰 Budget Optimization Engine

**Multi-Objective Optimization:**
```python
# Optimization objectives
objectives = {
    'roi': maximize_return_on_investment,
    'revenue': maximize_total_revenue,
    'reach': maximize_audience_reach
}
```

**Constraint Handling:**
- Channel-specific budget limits
- Minimum/maximum spend thresholds
- Business rule constraints
- Competitive parity requirements

### 4. 📊 Attribution Analysis

**Attribution Methodologies:**
```python
attribution_methods = [
    'last_touch',       # Credit last touchpoint
    'first_touch',      # Credit first touchpoint  
    'linear',           # Equal credit distribution
    'time_decay',       # Decay based on recency
    'data_driven'       # MMM-based attribution
]
```

### 5. 🔄 dbt Integration

**Data Transformation Pipeline:**
```sql
-- dbt model structure
staging/
├── stg_media_spend.sql      # Clean spend data
├── stg_revenue.sql          # Clean revenue data
└── stg_external_factors.sql # Economic indicators

intermediate/
├── int_attribution.sql      # Attribution logic
├── int_media_transforms.sql # Adstock/saturation
└── int_performance_agg.sql  # Aggregated metrics

marts/
├── media_performance.sql    # Final performance mart
├── attribution_summary.sql  # Attribution results
└── optimization_inputs.sql  # Optimization ready data
```

## 🚀 Production Architecture

### AWS Deployment Pattern

```
┌─────────────────────────────────────────────────────────────┐
│                        🌐 Internet                          │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│                   🔲 Application Load Balancer              │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│  🎯 SageMaker Endpoints  │  📊 Lambda Functions  │  🔌 API Gateway │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│  📦 S3 Data Lake  │  📊 CloudWatch  │  🔧 Parameter Store    │
└─────────────────────────────────────────────────────────────┘
```

### Deployment Components

#### 🎯 SageMaker Integration
```python
# Model serving architecture
sagemaker_endpoint = {
    'instance_type': 'ml.m5.large',
    'instance_count': 1,
    'auto_scaling': {
        'min_capacity': 1,
        'max_capacity': 5,
        'target_cpu_utilization': 70
    }
}
```

#### 🌪️ Airflow Orchestration
```python
# DAG structure for automated workflows
dag_config = {
    'data_ingestion': {
        'schedule': '0 6 * * *',  # Daily at 6 AM
        'tasks': ['fetch_kaggle', 'validate_data', 'store_s3']
    },
    'model_training': {
        'schedule': '0 2 * * 1',  # Weekly on Monday at 2 AM
        'tasks': ['prepare_data', 'train_mmm', 'validate_model', 'deploy']
    },
    'optimization': {
        'schedule': '0 8 * * 1',  # Weekly on Monday at 8 AM
        'tasks': ['run_optimization', 'generate_reports', 'notify_stakeholders']
    }
}
```

## 🔌 Integration Patterns

### 1. 📊 Real-time Data Integration

```python
# Streaming data architecture
data_flow = {
    'sources': ['Google Ads API', 'Facebook Ads API', 'GA4'],
    'ingestion': 'Apache Kafka / AWS Kinesis',
    'processing': 'Apache Spark / AWS Lambda',
    'storage': 'S3 Data Lake',
    'serving': 'dbt transformations'
}
```

### 2. 🔄 Model Lifecycle Management

```python
# MLflow integration pattern
mlflow_workflow = {
    'experiment_tracking': 'Track model parameters and metrics',
    'model_registry': 'Version and stage model artifacts',
    'deployment': 'Automated model deployment pipeline',
    'monitoring': 'Model performance and drift detection'
}
```

### 3. 📈 Executive Reporting Pipeline

```python
# Report generation workflow
reporting_pipeline = {
    'data_preparation': 'dbt models aggregate performance data',
    'analysis': 'MMM models generate insights',
    'visualization': 'Executive dashboards and reports',
    'distribution': 'Automated email/Slack notifications'
}
```

## 🧪 Testing Architecture

### Test Strategy Pyramid

```
┌─────────────────────────────────────────┐
│            🔧 E2E Tests                 │  ← Few, high-value
├─────────────────────────────────────────┤
│         📊 Integration Tests            │  ← Medium coverage
├─────────────────────────────────────────┤
│           🎯 Unit Tests                 │  ← Many, fast
└─────────────────────────────────────────┘
```

### Test Categories

#### 🎯 Unit Tests
- Model component validation
- Data transformation logic
- Optimization algorithm correctness
- API endpoint functionality

#### 📊 Integration Tests
- End-to-end model training
- Multi-source data integration
- AWS service integration
- dbt model execution

#### 🔧 E2E Tests
- Complete workflow validation
- Production deployment testing
- Performance benchmarking
- User acceptance scenarios

## 📊 Performance Considerations

### Scalability Patterns

#### 🚀 Model Training Optimization
```python
performance_config = {
    'data_sampling': 'Smart sampling for large datasets',
    'parallel_processing': 'Multi-core model training',
    'caching': 'Redis for intermediate results',
    'batch_processing': 'Airflow for scheduled training'
}
```

#### 📈 Inference Optimization
```python
inference_config = {
    'model_caching': 'In-memory model serving',
    'auto_scaling': 'Kubernetes horizontal pod autoscaling',
    'load_balancing': 'AWS Application Load Balancer',
    'cdn': 'CloudFront for static assets'
}
```

### Monitoring & Observability

#### 📊 Application Metrics
- Model prediction latency
- API response times
- Data pipeline success rates
- Model accuracy drift

#### 🔍 Business Metrics
- Attribution model accuracy
- Budget optimization ROI lift
- Report generation times
- User engagement metrics

## 🔒 Security Architecture

### Data Protection
```python
security_measures = {
    'encryption': {
        'at_rest': 'S3 server-side encryption',
        'in_transit': 'TLS 1.3 for all communications',
        'database': 'RDS encryption with KMS'
    },
    'access_control': {
        'authentication': 'AWS IAM with MFA',
        'authorization': 'Role-based access control',
        'api_security': 'API Gateway with rate limiting'
    },
    'compliance': {
        'data_governance': 'Data lineage tracking',
        'audit_logging': 'CloudTrail comprehensive logging',
        'privacy': 'GDPR-compliant data handling'
    }
}
```

## 🔄 DevOps & CI/CD

### Deployment Pipeline

```yaml
# CI/CD pipeline stages
stages:
  - 🧪 test:
      - unit_tests
      - integration_tests
      - security_scanning
  
  - 📦 build:
      - docker_image_build
      - artifact_packaging
      - vulnerability_scanning
  
  - 🚀 deploy:
      - staging_deployment
      - smoke_tests
      - production_deployment
  
  - 📊 monitor:
      - health_checks
      - performance_monitoring
      - alert_validation
```

### Infrastructure as Code

```python
# Terraform/CloudFormation resources
infrastructure_components = {
    'compute': ['SageMaker endpoints', 'Lambda functions', 'EC2 instances'],
    'storage': ['S3 buckets', 'RDS databases', 'ElastiCache'],
    'networking': ['VPC', 'subnets', 'security groups'],
    'monitoring': ['CloudWatch', 'X-Ray', 'Config']
}
```

## 🔮 Future Architecture Considerations

### Planned Enhancements

#### 🤖 AI/ML Capabilities
- AutoML model selection
- Neural MMM models
- Real-time attribution
- Causal inference integration

#### 📊 Data & Analytics
- Streaming data processing
- Graph-based attribution
- Multi-touch attribution
- Cross-device tracking

#### ☁️ Cloud-Native Features
- Serverless architecture migration
- Multi-region deployment
- Edge computing integration
- Container orchestration with K8s

---

## 📚 Related Documentation

- [API Reference](API_REFERENCE.md) - Detailed API documentation
- [Deployment Guide](DEPLOYMENT_GUIDE.md) - Production deployment instructions
- [Business Applications](BUSINESS_APPLICATIONS.md) - Use cases and ROI examples

---

*Architecture version: 1.0.0 | Last updated: 2025-08-17*