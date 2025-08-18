# 🚀 Media Mix Modeling - Deployment Guide

Complete production deployment guide for the MMM platform, covering local setup through cloud deployment with monitoring and scaling.

## 🎯 Deployment Overview

The MMM platform supports **progressive deployment** from local development to cloud production:

1. **🖥️ Local Development** - Quick start and testing
2. **🔧 Staging Environment** - Pre-production validation  
3. **☁️ Cloud Production** - AWS-based scalable deployment
4. **📊 Monitoring & Optimization** - Performance tracking and scaling

---

## 🖥️ Local Development Setup

### Prerequisites

- **Python 3.8+** (Python 3.11 recommended)
- **Git** for version control
- **Docker** (optional, for containerized deployment)
- **AWS CLI** (for cloud deployment)

### Quick Start (5 minutes)

```bash
# 1. Clone repository
git clone https://github.com/joshamorrison/public.git
cd public/media-mix-modeling

# 2. Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run demo
python quick_start.py
```

### Expected Output
```
🚀 MEDIA MIX MODELING & OPTIMIZATION - QUICK START DEMO
============================================================
[OK] Core dependencies available
[DATA] Using SYNTHETIC data: DEMO quality
[MMM] Advanced media mix model trained (adstock + saturation)
[OPT] Budget optimization complete: +18.3% projected ROAS improvement
[RPT] Executive reports generated (JSON, CSV, executive summary)
🎉 DEMO COMPLETE - Ready for real media data integration!
```

### Development Environment Configuration

#### Environment Variables (.env)
```bash
# Create .env file for configuration
cp .env.example .env

# Edit with your API keys (optional for local development)
KAGGLE_USERNAME=your_kaggle_username
KAGGLE_KEY=your_kaggle_api_key
HF_TOKEN=your_huggingface_token
AWS_REGION=us-east-1
MLFLOW_TRACKING_URI=http://localhost:5000
```

#### Advanced Dependencies (Optional)
```bash
# For full feature support
pip install -e ".[full]"

# For development tools
pip install -e ".[dev]"

# For documentation generation
pip install -e ".[docs]"
```

---

## 🔧 Staging Environment

### Docker Deployment

#### Build Container
```bash
# Build MMM container
docker build -t mmm-platform:latest .

# Run container
docker run -p 8000:8000 \
  -e AWS_REGION=us-east-1 \
  -e KAGGLE_USERNAME=${KAGGLE_USERNAME} \
  -e KAGGLE_KEY=${KAGGLE_KEY} \
  mmm-platform:latest
```

#### Docker Compose (Recommended)
```yaml
# docker-compose.yml
version: '3.8'

services:
  mmm-platform:
    build: .
    ports:
      - "8000:8000"
    environment:
      - AWS_REGION=us-east-1
      - KAGGLE_USERNAME=${KAGGLE_USERNAME}
      - KAGGLE_KEY=${KAGGLE_KEY}
    volumes:
      - ./outputs:/app/outputs
      - ./production:/app/production

  mlflow:
    image: python:3.11-slim
    ports:
      - "5000:5000"
    command: >
      sh -c "pip install mlflow && 
             mlflow server 
             --backend-store-uri sqlite:///mlflow.db 
             --default-artifact-root ./artifacts 
             --host 0.0.0.0"
    volumes:
      - mlflow_data:/mlflow

volumes:
  mlflow_data:
```

```bash
# Start staging environment
docker-compose up -d

# Verify deployment
curl http://localhost:8000/health
```

### Staging Validation

#### Run Test Suite
```bash
# Run all tests
python scripts/run_tests.py --suite all

# Run integration tests specifically
python scripts/run_tests.py --suite integration

# Generate coverage report
python scripts/run_tests.py --suite coverage
```

#### Performance Benchmarking
```bash
# Benchmark model training
python -c "
from tests.test_performance import benchmark_mmm_training
results = benchmark_mmm_training()
print(f'Training time: {results[\"training_time\"]:.2f}s')
print(f'Memory usage: {results[\"memory_mb\"]:.1f}MB')
"
```

---

## ☁️ AWS Cloud Production Deployment

### AWS Prerequisites

#### 1. AWS Account Setup
```bash
# Install AWS CLI
pip install awscli

# Configure AWS credentials
aws configure
# AWS Access Key ID: [Your Access Key]
# AWS Secret Access Key: [Your Secret Key]
# Default region name: us-east-1
# Default output format: json
```

#### 2. IAM Roles and Permissions
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "sagemaker:*",
        "s3:*",
        "lambda:*",
        "iam:PassRole",
        "cloudwatch:*",
        "logs:*"
      ],
      "Resource": "*"
    }
  ]
}
```

#### 3. S3 Bucket Creation
```bash
# Create S3 bucket for MMM artifacts
aws s3 mb s3://your-mmm-bucket-name --region us-east-1

# Enable versioning
aws s3api put-bucket-versioning \
  --bucket your-mmm-bucket-name \
  --versioning-configuration Status=Enabled
```

### Production Deployment Methods

#### Method 1: Automated AWS Deployment (Recommended)

```bash
# Run complete AWS deployment
python infrastructure/aws/deploy_to_aws.py

# Expected output:
# [AWS] Setting up infrastructure...
# [MODELS] Training and deploying MMM models...
# [MONITORING] Setting up model monitoring...
# ✅ AWS DEPLOYMENT COMPLETED SUCCESSFULLY!
```

#### Method 2: SageMaker-Only Deployment

```python
# Deploy model to SageMaker only
from infrastructure.aws.sagemaker_deployment import deploy_mmm_to_sagemaker
from production.mmm_production_model import load_production_model

# Load trained model
mmm_model = load_production_model('production/mmm_production_model.joblib')

# Deploy to SageMaker
deployment_info = deploy_mmm_to_sagemaker(
    mmm_model=mmm_model,
    model_name='mmm-production-v1',
    region_name='us-east-1'
)

print(f"Endpoint URL: {deployment_info['endpoint_url']}")
```

#### Method 3: Infrastructure as Code (CloudFormation)

```bash
# Deploy using CloudFormation
aws cloudformation create-stack \
  --stack-name mmm-platform \
  --template-body file://infrastructure/aws/cloudformation/main.yaml \
  --parameters ParameterKey=BucketName,ParameterValue=your-mmm-bucket \
  --capabilities CAPABILITY_IAM
```

### Post-Deployment Validation

#### 1. Health Checks
```bash
# Test SageMaker endpoint
python -c "
import boto3
import json

# Create SageMaker runtime client
runtime = boto3.client('sagemaker-runtime', region_name='us-east-1')

# Test prediction
test_data = {
    'tv_spend': [10000],
    'digital_spend': [15000],
    'radio_spend': [5000],
    'print_spend': [3000],
    'social_spend': [8000]
}

response = runtime.invoke_endpoint(
    EndpointName='your-endpoint-name',
    ContentType='application/json',
    Body=json.dumps(test_data)
)

result = json.loads(response['Body'].read().decode())
print(f'Prediction: {result}')
"
```

#### 2. Performance Testing
```bash
# Load test the endpoint
python scripts/load_test_endpoint.py \
  --endpoint-name your-endpoint-name \
  --concurrent-requests 10 \
  --duration 60
```

---

## 🌪️ Airflow Orchestration Setup

### Local Airflow Setup

```bash
# Install Apache Airflow
pip install apache-airflow[aws,postgres]

# Initialize Airflow database
airflow db init

# Create admin user
airflow users create \
  --username admin \
  --firstname Admin \
  --lastname User \
  --role Admin \
  --email admin@example.com \
  --password admin

# Start Airflow webserver
airflow webserver --port 8080 &

# Start Airflow scheduler
airflow scheduler &
```

### Deploy MMM DAGs

```bash
# Copy DAGs to Airflow
cp infrastructure/airflow/dags/* $AIRFLOW_HOME/dags/

# List available DAGs
airflow dags list | grep mmm

# Test DAG
airflow dags test mmm_daily_pipeline 2025-08-17
```

### Airflow Configuration

```python
# airflow.cfg modifications for MMM
[core]
dags_folder = /path/to/media-mix-modeling/infrastructure/airflow/dags
parallelism = 32
max_active_runs_per_dag = 16

[webserver]
base_url = http://localhost:8080
expose_config = False

[email]
email_backend = airflow.providers.sendgrid.hooks.sendgrid.SendGridHook
```

---

## 🔄 dbt Integration Setup

### Local dbt Setup

```bash
# Install dbt
pip install dbt-core dbt-sqlite dbt-bigquery dbt-snowflake

# Initialize dbt project
cd infrastructure/dbt
dbt init mmm_project

# Configure profiles.yml
# ~/.dbt/profiles.yml
mmm_project:
  target: dev
  outputs:
    dev:
      type: sqlite
      database: './mmm_dev.db'
      schema: main
    prod:
      type: bigquery
      project: your-gcp-project
      dataset: mmm_prod
      keyfile: /path/to/service-account.json
```

### Run dbt Models

```bash
# Install dependencies
dbt deps

# Test database connection
dbt debug

# Run data quality tests
dbt test

# Build all models
dbt run

# Generate documentation
dbt docs generate
dbt docs serve
```

### dbt Cloud Deployment

```yaml
# dbt_project.yml configuration for production
name: 'mmm_platform'
version: '1.0.0'

model-paths: ["models"]
analysis-paths: ["analysis"]
test-paths: ["tests"]
seed-paths: ["data"]

clean-targets:
  - "target"
  - "dbt_packages"

models:
  mmm_platform:
    staging:
      +materialized: view
    intermediate:
      +materialized: ephemeral
    marts:
      +materialized: table
      +pre-hook: "{{ log('Running MMM mart model: ' ~ this.name, info=true) }}"
```

---

## 📊 Monitoring & Observability

### CloudWatch Setup

#### Custom Metrics
```python
# Set up custom CloudWatch metrics
import boto3

cloudwatch = boto3.client('cloudwatch')

# Model performance metrics
cloudwatch.put_metric_data(
    Namespace='MMM/Model',
    MetricData=[
        {
            'MetricName': 'ModelAccuracy',
            'Value': 0.85,
            'Unit': 'Percent'
        },
        {
            'MetricName': 'PredictionLatency',
            'Value': 150,
            'Unit': 'Milliseconds'
        }
    ]
)
```

#### CloudWatch Alarms
```bash
# Create model performance alarm
aws cloudwatch put-metric-alarm \
  --alarm-name "MMM-Model-Low-Accuracy" \
  --alarm-description "Alert when model accuracy drops below 80%" \
  --metric-name ModelAccuracy \
  --namespace MMM/Model \
  --statistic Average \
  --period 300 \
  --threshold 80 \
  --comparison-operator LessThanThreshold \
  --evaluation-periods 2
```

### MLflow Tracking Setup

#### Production MLflow Server
```bash
# Start MLflow server with database backend
mlflow server \
  --backend-store-uri postgresql://user:password@host:port/mlflow \
  --default-artifact-root s3://your-mlflow-bucket/artifacts \
  --host 0.0.0.0 \
  --port 5000
```

#### Model Registry Integration
```python
# Register production model
import mlflow
import mlflow.sklearn

# Set tracking URI
mlflow.set_tracking_uri("http://your-mlflow-server:5000")

# Register model
model_name = "MMM-Production"
model_version = mlflow.register_model(
    model_uri=f"runs:/{run_id}/model",
    name=model_name
)

# Transition to production
client = mlflow.tracking.MlflowClient()
client.transition_model_version_stage(
    name=model_name,
    version=model_version.version,
    stage="Production"
)
```

### Log Aggregation

#### CloudWatch Logs
```python
# Configure application logging
import logging
import watchtower

# Set up CloudWatch handler
handler = watchtower.CloudWatchLogsHandler(
    log_group='mmm-platform',
    stream_name='application-logs'
)

logger = logging.getLogger(__name__)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Log model performance
logger.info(f"Model training completed - R²: {r2_score:.3f}")
```

---

## 🔒 Security Configuration

### Environment Security

#### Secrets Management
```bash
# Store secrets in AWS Parameter Store
aws ssm put-parameter \
  --name "/mmm/kaggle/username" \
  --value "your-kaggle-username" \
  --type "SecureString"

aws ssm put-parameter \
  --name "/mmm/kaggle/key" \
  --value "your-kaggle-api-key" \
  --type "SecureString"
```

#### IAM Role Configuration
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "ssm:GetParameter",
        "ssm:GetParameters"
      ],
      "Resource": "arn:aws:ssm:*:*:parameter/mmm/*"
    }
  ]
}
```

### Network Security

#### VPC Configuration
```bash
# Create VPC for MMM platform
aws ec2 create-vpc \
  --cidr-block 10.0.0.0/16 \
  --tag-specifications 'ResourceType=vpc,Tags=[{Key=Name,Value=mmm-vpc}]'

# Create private subnets for SageMaker
aws ec2 create-subnet \
  --vpc-id vpc-12345678 \
  --cidr-block 10.0.1.0/24 \
  --availability-zone us-east-1a
```

#### Security Groups
```bash
# Create security group for SageMaker endpoints
aws ec2 create-security-group \
  --group-name mmm-sagemaker-sg \
  --description "Security group for MMM SageMaker endpoints" \
  --vpc-id vpc-12345678

# Add HTTPS inbound rule
aws ec2 authorize-security-group-ingress \
  --group-id sg-12345678 \
  --protocol tcp \
  --port 443 \
  --cidr 10.0.0.0/16
```

---

## 📈 Scaling & Performance Optimization

### Auto Scaling Configuration

#### SageMaker Auto Scaling
```python
# Configure SageMaker endpoint auto scaling
import boto3

autoscaling = boto3.client('application-autoscaling')

# Register scalable target
autoscaling.register_scalable_target(
    ServiceNamespace='sagemaker',
    ResourceId='endpoint/mmm-production/variant/AllTraffic',
    ScalableDimension='sagemaker:variant:DesiredInstanceCount',
    MinCapacity=1,
    MaxCapacity=10
)

# Create scaling policy
autoscaling.put_scaling_policy(
    PolicyName='mmm-scaling-policy',
    ServiceNamespace='sagemaker',
    ResourceId='endpoint/mmm-production/variant/AllTraffic',
    ScalableDimension='sagemaker:variant:DesiredInstanceCount',
    PolicyType='TargetTrackingScaling',
    TargetTrackingScalingPolicyConfiguration={
        'TargetValue': 70.0,
        'PredefinedMetricSpecification': {
            'PredefinedMetricType': 'SageMakerVariantInvocationsPerInstance'
        }
    }
)
```

### Performance Optimization

#### Model Optimization
```python
# Optimize model for inference
import joblib
from sklearn.pipeline import Pipeline

# Create optimized inference pipeline
optimized_pipeline = Pipeline([
    ('preprocessor', data_preprocessor),
    ('model', trained_mmm_model)
])

# Save optimized model
joblib.dump(optimized_pipeline, 'production/optimized_mmm_model.joblib')
```

#### Caching Strategy
```python
# Implement Redis caching for frequent predictions
import redis
import json
import hashlib

redis_client = redis.Redis(host='elasticache-endpoint', port=6379, db=0)

def cached_prediction(input_data):
    # Create cache key
    cache_key = hashlib.md5(json.dumps(input_data, sort_keys=True).encode()).hexdigest()
    
    # Check cache
    cached_result = redis_client.get(cache_key)
    if cached_result:
        return json.loads(cached_result)
    
    # Generate prediction
    prediction = model.predict(input_data)
    
    # Cache result (expire in 1 hour)
    redis_client.setex(cache_key, 3600, json.dumps(prediction.tolist()))
    
    return prediction
```

---

## 🚨 Troubleshooting

### Common Issues

#### 1. SageMaker Deployment Failures
```bash
# Check SageMaker endpoint status
aws sagemaker describe-endpoint --endpoint-name your-endpoint-name

# View CloudWatch logs
aws logs describe-log-groups --log-group-name-prefix /aws/sagemaker

# Debug inference container
docker run -it your-inference-image:latest /bin/bash
```

#### 2. Data Integration Issues
```python
# Debug data source connectivity
from data.media_data_client import MediaDataClient

client = MediaDataClient()
try:
    data, info, source = client.get_best_available_data()
    print(f"Successfully loaded {source} data: {len(data)} records")
except Exception as e:
    print(f"Data loading failed: {e}")
```

#### 3. Model Training Failures
```bash
# Run model training with debug mode
python -c "
import logging
logging.basicConfig(level=logging.DEBUG)

from models.mmm.econometric_mmm import EconometricMMM
# ... training code with verbose logging
"
```

### Performance Issues

#### Memory Optimization
```python
# Monitor memory usage during training
import psutil
import os

def monitor_memory():
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / 1024 / 1024
    print(f"Memory usage: {memory_mb:.1f} MB")
    return memory_mb

# Use during model training
monitor_memory()  # Before training
mmm_results = mmm.fit(data, target_column='revenue', spend_columns=spend_columns)
monitor_memory()  # After training
```

#### Latency Optimization
```python
# Profile prediction latency
import time

def profile_prediction_latency(model, test_data, num_iterations=100):
    latencies = []
    
    for _ in range(num_iterations):
        start_time = time.time()
        prediction = model.predict(test_data)
        end_time = time.time()
        latencies.append((end_time - start_time) * 1000)  # Convert to ms
    
    return {
        'mean_latency_ms': np.mean(latencies),
        'p95_latency_ms': np.percentile(latencies, 95),
        'p99_latency_ms': np.percentile(latencies, 99)
    }
```

---

## 📋 Deployment Checklist

### Pre-Deployment
- [ ] Code review completed
- [ ] All tests passing
- [ ] Security scan completed
- [ ] Performance benchmarks met
- [ ] Documentation updated

### Production Deployment
- [ ] AWS credentials configured
- [ ] S3 buckets created and configured
- [ ] IAM roles and policies set up
- [ ] SageMaker endpoints deployed
- [ ] CloudWatch monitoring configured
- [ ] Auto-scaling policies activated

### Post-Deployment
- [ ] Health checks passing
- [ ] Model performance validated
- [ ] Monitoring dashboards operational
- [ ] Alert notifications working
- [ ] Documentation updated with endpoints

### Rollback Plan
- [ ] Previous model version available
- [ ] Rollback procedure documented
- [ ] Monitoring in place for issues
- [ ] Team notification process ready

---

## 🔗 Additional Resources

- [AWS SageMaker Documentation](https://docs.aws.amazon.com/sagemaker/)
- [Apache Airflow Documentation](https://airflow.apache.org/docs/)
- [dbt Documentation](https://docs.getdbt.com/)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)

---

*Deployment Guide version: 1.0.0 | Last updated: 2025-08-17*