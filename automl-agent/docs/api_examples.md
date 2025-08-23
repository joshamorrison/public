# AutoML Agent Platform API Examples

## Authentication

### JWT Token Authentication

```bash
# Login to get JWT token
curl -X POST "http://localhost:8000/auth/login" \
  -H "Content-Type: application/json" \
  -d '{
    "username": "admin",
    "password": "admin123"
  }'

# Response
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 1800
}

# Use token in subsequent requests
curl -X GET "http://localhost:8000/agents" \
  -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
```

### API Key Authentication

```bash
# Use API key directly
curl -X GET "http://localhost:8000/agents" \
  -H "Authorization: Bearer ak_demo_12345"
```

## Basic Operations

### List Available Agents

```bash
curl -X GET "http://localhost:8000/agents" \
  -H "Authorization: Bearer YOUR_TOKEN"
```

**Response:**
```json
{
  "agents": {
    "eda": {
      "name": "EDAAgent",
      "description": "Comprehensive exploratory data analysis",
      "capabilities": ["statistical_analysis", "visualization", "pattern_discovery"],
      "task_types": ["exploratory_analysis", "data_profiling"]
    },
    "classification": {
      "name": "ClassificationAgent", 
      "description": "Advanced classification with multiple algorithms",
      "capabilities": ["model_selection", "hyperparameter_optimization"],
      "task_types": ["binary_classification", "multiclass_classification"]
    }
  }
}
```

### Upload Data

```bash
curl -X POST "http://localhost:8000/upload-data" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -F "file=@data.csv"
```

**Response:**
```json
{
  "data_id": "550e8400-e29b-41d4-a716-446655440000",
  "filename": "data.csv",
  "shape": [1000, 15],
  "columns": ["feature_0", "feature_1", "target"],
  "dtypes": {"feature_0": "float64", "target": "int64"},
  "sample": [
    {"feature_0": 1.23, "feature_1": 4.56, "target": 1}
  ]
}
```

## Agent Execution

### Execute Single Agent

```bash
curl -X POST "http://localhost:8000/execute/eda" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "task_description": "Analyze customer data for patterns and insights",
    "parameters": {
      "target_column": "churn",
      "include_visualizations": true
    }
  }'
```

**Response:**
```json
{
  "job_id": "job_123456",
  "status": "queued"
}
```

### Execute Classification Workflow

```bash
curl -X POST "http://localhost:8000/execute/classification" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "task_description": "Build a model to predict customer churn",
    "parameters": {
      "target_column": "churn",
      "quality_threshold": 0.85,
      "algorithms": ["random_forest", "xgboost", "lightgbm"]
    }
  }'
```

## Multi-Agent Workflows

### Sequential Workflow

```bash
curl -X POST "http://localhost:8000/workflow" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "task_description": "Complete ML pipeline for customer churn prediction",
    "agents": ["eda", "data_hygiene", "feature_engineering", "classification"],
    "quality_threshold": 0.8,
    "collaboration_mode": "sequential",
    "max_iterations": 3
  }'
```

### Parallel Processing

```bash
curl -X POST "http://localhost:8000/workflow" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "task_description": "Compare multiple modeling approaches",
    "agents": ["classification", "regression", "time_series"],
    "collaboration_mode": "parallel",
    "quality_threshold": 0.75
  }'
```

## Complete AutoML Workflow

### Full AutoML Pipeline

Upload data and run complete AutoML workflow:

```bash
curl -X POST "http://localhost:8000/automl" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -F "file=@customer_data.csv" \
  -F "target_column=churn" \
  -F "task_type=classification"
```

**Response:**
```json
{
  "success": true,
  "data_info": {
    "original_shape": [5000, 20],
    "processed_shape": [5000, 19],
    "data_quality_score": 0.95,
    "splits": {
      "train_size": 3000,
      "validation_size": 1000,
      "test_size": 1000
    }
  },
  "eda_results": {
    "success": true,
    "quality_score": 0.95,
    "visualizations_count": 8,
    "recommendations": [
      "Consider feature engineering on tenure data",
      "Address class imbalance in target variable"
    ]
  },
  "model_results": {
    "success": true,
    "best_model": "random_forest",
    "best_score": 0.89,
    "models_trained": 3
  },
  "processing_time": 45.2
}
```

This endpoint performs the complete AutoML pipeline:
1. Data upload and validation
2. Data processing and cleaning
3. Exploratory Data Analysis (EDA)
4. Feature engineering
5. Model training and evaluation
6. Results generation

## Quick Analysis

### Auto-Detect Task Type

```bash
curl -X POST "http://localhost:8000/analyze" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "task_description": "Predict stock prices based on historical data and market indicators",
    "target_column": "close_price"
  }'
```

This automatically detects it's a regression/time series task and runs appropriate agents.

### Text Analysis

```bash
curl -X POST "http://localhost:8000/analyze" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "task_description": "Analyze customer reviews for sentiment and extract key topics"
  }'
```

This automatically detects it's an NLP task and runs the NLP agent.

## Job Management

### Check Job Status

```bash
curl -X GET "http://localhost:8000/jobs/job_123456" \
  -H "Authorization: Bearer YOUR_TOKEN"
```

**Response:**
```json
{
  "job_id": "job_123456",
  "status": "running",
  "progress": 0.65,
  "result": null,
  "error": null,
  "created_at": "2024-01-15T10:30:00Z",
  "updated_at": "2024-01-15T10:35:30Z",
  "estimated_completion": "2024-01-15T10:45:00Z"
}
```

### List All Jobs

```bash
curl -X GET "http://localhost:8000/jobs?limit=10&status=running" \
  -H "Authorization: Bearer YOUR_TOKEN"
```

### Cancel Job

```bash
curl -X DELETE "http://localhost:8000/jobs/job_123456" \
  -H "Authorization: Bearer YOUR_TOKEN"
```

## Specialized Agent Examples

### EDA Agent - Comprehensive Analysis

```bash
curl -X POST "http://localhost:8000/execute/eda" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "task_description": "Perform comprehensive exploratory data analysis on sales data",
    "parameters": {
      "target_column": "revenue",
      "include_correlations": true,
      "generate_visualizations": true,
      "statistical_tests": ["normality", "stationarity"],
      "outlier_detection": true
    }
  }'
```

### Feature Engineering Agent

```bash
curl -X POST "http://localhost:8000/execute/feature_engineering" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "task_description": "Create advanced features for customer behavior prediction",
    "parameters": {
      "feature_types": ["polynomial", "interaction", "temporal"],
      "target_column": "conversion",
      "max_features": 50,
      "selection_method": "mutual_info"
    }
  }'
```

### NLP Agent - Text Classification

```bash
curl -X POST "http://localhost:8000/execute/nlp" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "task_description": "Classify customer support tickets by urgency and category",
    "parameters": {
      "text_column": "description",
      "target_column": "category",
      "preprocessing": ["lowercase", "remove_stopwords", "lemmatize"],
      "vectorization": "tfidf",
      "max_features": 5000
    }
  }'
```

### Computer Vision Agent

```bash
curl -X POST "http://localhost:8000/execute/computer_vision" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "task_description": "Classify product images for automated inventory management",
    "parameters": {
      "image_size": [224, 224],
      "model_type": "transfer_learning",
      "base_model": "resnet50",
      "augmentation": true,
      "batch_size": 32
    }
  }'
```

### Time Series Agent

```bash
curl -X POST "http://localhost:8000/execute/time_series" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "task_description": "Forecast daily sales for the next 30 days",
    "parameters": {
      "date_column": "date",
      "value_column": "sales",
      "forecast_horizon": 30,
      "seasonality": "daily",
      "models": ["arima", "prophet", "lstm"]
    }
  }'
```

## Advanced Workflows

### Iterative Refinement

```bash
curl -X POST "http://localhost:8000/workflow" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "task_description": "Build high-accuracy fraud detection model with iterative improvement",
    "agents": ["data_hygiene", "feature_engineering", "classification"],
    "quality_threshold": 0.95,
    "collaboration_mode": "sequential",
    "max_iterations": 5,
    "refinement_strategy": "quality_driven"
  }'
```

### Multi-Domain Analysis

```bash
curl -X POST "http://localhost:8000/workflow" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "task_description": "Comprehensive analysis of multi-modal customer data",
    "agents": ["eda", "nlp", "classification", "time_series"],
    "collaboration_mode": "adaptive",
    "quality_threshold": 0.8,
    "data_sources": {
      "structured": "customer_metrics.csv",
      "text": "reviews.json", 
      "temporal": "interaction_history.csv"
    }
  }'
```

## Response Examples

### Successful Agent Execution

```json
{
  "job_id": "job_789",
  "status": "completed",
  "progress": 1.0,
  "result": {
    "agent_type": "classification",
    "task_description": "Customer churn prediction",
    "status": "completed",
    "performance_metrics": {
      "accuracy": 0.87,
      "precision": 0.85,
      "recall": 0.89,
      "f1_score": 0.87,
      "auc_roc": 0.93
    },
    "quality_score": 0.87,
    "execution_time": 45.2,
    "best_model": "XGBoost",
    "feature_importance": {
      "tenure": 0.25,
      "monthly_charges": 0.18,
      "total_charges": 0.15
    },
    "recommendations": [
      "Consider feature engineering on customer tenure",
      "Collect more data on customer satisfaction",
      "Monitor model drift monthly"
    ]
  },
  "created_at": "2024-01-15T10:30:00Z",
  "updated_at": "2024-01-15T11:15:20Z"
}
```

### Workflow Completion

```json
{
  "job_id": "workflow_456",
  "status": "completed", 
  "progress": 1.0,
  "result": {
    "workflow_results": {
      "eda": {
        "insights_found": 15,
        "quality_score": 0.92,
        "key_findings": ["Strong correlation between tenure and churn"]
      },
      "feature_engineering": {
        "features_created": 23,
        "features_selected": 12,
        "quality_improvement": 0.08
      },
      "classification": {
        "best_model": "LightGBM",
        "accuracy": 0.91,
        "quality_score": 0.91
      }
    },
    "overall_quality": 0.91,
    "total_execution_time": 127.5,
    "refinement_iterations": 2,
    "summary": "Successfully built high-quality churn prediction model"
  }
}
```

## Error Handling

### Authentication Error

```json
{
  "detail": "Authentication required",
  "error_code": "AUTH_001"
}
```

### Rate Limit Error

```json
{
  "detail": "Rate limit exceeded", 
  "error_code": "RATE_001",
  "headers": {
    "X-RateLimit-Remaining": "0",
    "X-RateLimit-Reset": "1642248600"
  }
}
```

### Job Error

```json
{
  "job_id": "job_error_123",
  "status": "failed",
  "error": "Insufficient data quality for reliable model training",
  "suggestions": [
    "Clean missing values in target column",
    "Remove outliers beyond 3 standard deviations",
    "Increase dataset size to minimum 1000 samples"
  ]
}
```

## WebSocket Real-time Updates

### Connect to Job Updates

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/jobs/job_123456');

ws.onmessage = function(event) {
    const update = JSON.parse(event.data);
    console.log(`Job ${update.job_id}: ${update.status} (${update.progress}%)`);
    
    if (update.status === 'completed') {
        console.log('Results:', update.result);
    }
};
```

### Agent Communication Stream

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/agent-communication');

ws.onmessage = function(event) {
    const message = JSON.parse(event.data);
    console.log(`${message.sender_agent} → ${message.receiver_agent}: ${message.content}`);
};
```

## Python SDK Examples

### Basic Usage

```python
from automl_client import AutoMLClient

# Initialize client
client = AutoMLClient(
    base_url="http://localhost:8000",
    api_key="ak_demo_12345"
)

# Execute single agent
job = client.execute_agent(
    agent_type="classification",
    task_description="Predict customer churn",
    data="customer_data.csv"
)

# Wait for completion
result = client.wait_for_completion(job.job_id)
print(f"Model accuracy: {result.performance_metrics['accuracy']}")
```

### Workflow Execution

```python
# Execute full workflow
workflow = client.execute_workflow(
    task_description="End-to-end customer analysis",
    agents=["eda", "feature_engineering", "classification"],
    quality_threshold=0.85
)

# Monitor progress
for update in client.stream_progress(workflow.job_id):
    print(f"Progress: {update.progress:.1%}")
    if update.status == "completed":
        break

# Get final results
results = client.get_job_result(workflow.job_id)
```

This comprehensive API documentation provides examples for all major functionality of the AutoML Agent Platform, from basic authentication to complex multi-agent workflows.