# REST API Examples

## AI-Driven Agent Recommendations via API

### 1. Customer Churn Prediction

**Request:**
```bash
curl -X POST "http://localhost:8000/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "task_description": "Predict customer churn from usage patterns and demographics",
    "target_column": "churn"
  }'
```

**Response:**
```json
{
  "job_id": "job_churn_001",
  "status": "queued",
  "ai_recommendation": {
    "agents": ["eda", "classification"],
    "workflow": "eda → classification",
    "reasoning": [
      "EDA: Understand customer behavior patterns",
      "Classification: Binary prediction task (churn/no churn)"
    ],
    "confidence": 0.95
  },
  "estimated_duration": "3-5 minutes"
}
```

### 2. Sales Forecasting with Data Quality Issues

**Request:**
```bash
curl -X POST "http://localhost:8000/automl" \
  -F "file=@sales_data.csv" \
  -F "target_column=daily_sales" \
  -F "task_type=time_series" \
  -F "task_description=Forecast daily sales for next 30 days with missing data"
```

**Response:**
```json
{
  "success": true,
  "workflow_decision": {
    "ai_recommendation": ["eda", "data_hygiene", "time_series"],
    "reasoning": "Temporal forecasting with data quality concerns detected",
    "agents_executed": ["eda", "data_hygiene", "time_series"]
  },
  "data_info": {
    "original_shape": [365, 8],
    "processed_shape": [350, 12],
    "missing_values_fixed": 15,
    "data_quality_score": 0.87
  },
  "results": {
    "forecast_accuracy": "MAPE: 8.3%",
    "model_type": "ARIMA + Prophet ensemble",
    "seasonal_patterns": "Weekly and monthly seasonality detected"
  },
  "processing_time": 45.2
}
```

### 3. Sentiment Analysis

**Request:**
```bash
curl -X POST "http://localhost:8000/execute/nlp" \
  -H "Content-Type: application/json" \
  -d '{
    "task_description": "Analyze customer reviews for sentiment and extract topics",
    "parameters": {
      "text_column": "review_text",
      "include_topics": true,
      "sentiment_classes": ["positive", "negative", "neutral"]
    }
  }'
```

**Response:**
```json
{
  "job_id": "nlp_sentiment_789",
  "status": "completed",
  "ai_analysis": {
    "detected_task": "sentiment_analysis + topic_extraction",
    "recommended_workflow": ["eda", "nlp"],
    "text_preprocessing": ["tokenization", "stopword_removal", "lemmatization"]
  },
  "results": {
    "sentiment_accuracy": 0.921,
    "topics_discovered": 5,
    "top_topics": [
      {"topic": "product_quality", "weight": 0.28},
      {"topic": "customer_service", "weight": 0.24},
      {"topic": "pricing", "weight": 0.19}
    ],
    "sentiment_distribution": {
      "positive": 0.45,
      "neutral": 0.32,
      "negative": 0.23
    }
  }
}
```

### 4. Image Classification

**Request:**
```bash
curl -X POST "http://localhost:8000/execute/computer_vision" \
  -H "Content-Type: application/json" \
  -d '{
    "task_description": "Classify product images into categories with feature engineering",
    "parameters": {
      "image_size": [224, 224],
      "augmentation": true,
      "transfer_learning": "resnet50"
    }
  }'
```

**Response:**
```json
{
  "job_id": "cv_product_456",
  "status": "completed",
  "ai_recommendation": {
    "agents": ["computer_vision", "feature_engineering"],
    "reasoning": "Image classification with explicit feature enhancement",
    "architecture": "CNN with transfer learning + custom features"
  },
  "results": {
    "classification_accuracy": 0.947,
    "classes_detected": 8,
    "model_architecture": "ResNet50 + custom feature layers",
    "training_images": 2500,
    "validation_accuracy": 0.923,
    "top_features": [
      "color_histogram",
      "texture_patterns", 
      "edge_density",
      "shape_features"
    ]
  }
}
```

### 5. Customer Segmentation

**Request:**
```bash
curl -X POST "http://localhost:8000/workflow" \
  -H "Content-Type: application/json" \
  -d '{
    "task_description": "Group customers into segments based on purchasing behavior",
    "agents": ["eda", "feature_engineering", "clustering"],
    "collaboration_mode": "sequential",
    "quality_threshold": 0.8
  }'
```

**Response:**
```json
{
  "job_id": "clustering_segments_123",
  "status": "completed",
  "workflow_results": {
    "eda": {
      "behavioral_features_found": 15,
      "customer_profiles": "4 distinct purchasing patterns identified"
    },
    "feature_engineering": {
      "features_created": 23,
      "interaction_features": 8,
      "temporal_features": 6
    },
    "clustering": {
      "optimal_clusters": 4,
      "silhouette_score": 0.73,
      "cluster_characteristics": {
        "high_value": {"size": 0.15, "avg_spend": 2400},
        "frequent_buyers": {"size": 0.28, "purchase_frequency": 2.3},
        "price_sensitive": {"size": 0.35, "discount_usage": 0.82},
        "occasional": {"size": 0.22, "avg_spend": 320}
      }
    }
  },
  "total_execution_time": 127.5
}
```

## Intelligent Workflow API

### Auto-Recommend Endpoint

**Request:**
```bash
curl -X POST "http://localhost:8000/recommend-workflow" \
  -H "Content-Type: application/json" \
  -d '{
    "task_description": "Build best fraud detection model with dirty data",
    "data_info": {
      "rows": 50000,
      "columns": 25,
      "missing_values": 0.15,
      "target_balance": 0.02
    },
    "requirements": {
      "max_time_minutes": 60,
      "min_accuracy": 0.95,
      "interpretability": "medium"
    }
  }'
```

**Response:**
```json
{
  "recommended_workflow": {
    "agents": ["eda", "data_hygiene", "feature_engineering", "classification", "ensemble"],
    "sequence": "eda → data_hygiene → feature_engineering → classification → ensemble",
    "reasoning": {
      "eda": "Understand fraud patterns and class imbalance",
      "data_hygiene": "Clean missing values (15% detected)",
      "feature_engineering": "Create fraud indicators and interactions",
      "classification": "Binary fraud detection task",
      "ensemble": "Boost performance for high accuracy requirement"
    },
    "estimated_time": 45,
    "expected_accuracy": "0.96-0.98",
    "alternative_workflows": [
      {
        "workflow": ["eda", "classification", "hyperparameter_tuning"],
        "time": 25,
        "accuracy": "0.93-0.95",
        "use_case": "Faster development"
      }
    ]
  }
}
```

## Real-time Job Monitoring

### WebSocket Connection
```javascript
const ws = new WebSocket('ws://localhost:8000/ws/jobs/job_123456');

ws.onmessage = function(event) {
    const update = JSON.parse(event.data);
    console.log(`${update.agent}: ${update.message} (${update.progress}%)`);
};

// Example messages:
// "EDA Agent: Analyzing data patterns (20%)"
// "Classification Agent: Training RandomForest model (60%)"
// "Workflow complete: 94.2% accuracy achieved (100%)"
```

### Job Status Polling
```bash
curl -X GET "http://localhost:8000/jobs/job_123456"
```

**Response:**
```json
{
  "job_id": "job_123456",
  "status": "running",
  "progress": 0.65,
  "current_agent": "Classification Agent",
  "current_task": "Hyperparameter optimization",
  "agents_completed": ["EDA Agent", "Data Hygiene Agent"],
  "agents_remaining": ["Classification Agent", "Ensemble Agent"],
  "estimated_completion": "2024-08-23T10:45:00Z",
  "partial_results": {
    "data_quality_score": 0.91,
    "features_engineered": 23,
    "best_model_so_far": "RandomForest (0.89 accuracy)"
  }
}
```

## Batch Processing API

### Multiple Dataset Analysis
```bash
curl -X POST "http://localhost:8000/batch-analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "datasets": [
      {"name": "region_east", "file": "data/east.csv", "target": "churn"},
      {"name": "region_west", "file": "data/west.csv", "target": "churn"},
      {"name": "region_north", "file": "data/north.csv", "target": "churn"}
    ],
    "task_description": "Regional customer churn analysis",
    "shared_workflow": true
  }'
```

**Response:**
```json
{
  "batch_id": "batch_regional_001",
  "datasets_queued": 3,
  "shared_workflow": ["eda", "classification"],
  "individual_jobs": [
    {"dataset": "region_east", "job_id": "job_east_001"},
    {"dataset": "region_west", "job_id": "job_west_002"}, 
    {"dataset": "region_north", "job_id": "job_north_003"}
  ],
  "comparison_report": "Will be generated when all jobs complete"
}
```

## Error Handling Examples

### Invalid Task Description
```bash
curl -X POST "http://localhost:8000/analyze" \
  -d '{"task_description": ""}'
```

**Response:**
```json
{
  "error": "Task description is required",
  "code": "MISSING_TASK_DESCRIPTION",
  "suggestions": [
    "Provide a clear description of your ML task",
    "Include details about what you want to predict",
    "Mention your data type (text, images, tabular, etc.)"
  ]
}
```

### Unsupported Task Type
```bash
curl -X POST "http://localhost:8000/analyze" \
  -d '{"task_description": "Generate realistic human faces"}'
```

**Response:**
```json
{
  "error": "Task type not supported",
  "detected_task": "generative_modeling",
  "code": "UNSUPPORTED_TASK",
  "supported_tasks": [
    "classification", "regression", "clustering", 
    "time_series", "nlp", "computer_vision"
  ],
  "alternatives": [
    "Try image classification instead of generation",
    "Consider using specialized generative AI tools"
  ]
}
```

## API Performance Tips

1. **Use Async Endpoints**: Long-running tasks return job IDs immediately
2. **Monitor with WebSockets**: Real-time progress updates
3. **Batch Similar Tasks**: More efficient than individual requests
4. **Cache Recommendations**: Same task descriptions return consistent workflows
5. **Specify Requirements**: Time/accuracy constraints improve recommendations
6. **Use Streaming**: Large datasets benefit from streaming uploads

The REST API provides enterprise-grade automation with intelligent agent recommendations!