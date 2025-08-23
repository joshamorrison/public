# AutoML Agent Platform Examples

This document contains comprehensive examples demonstrating the AI-driven agent recommendation system across all interfaces.

## 🤖 AI-Driven Agent Recommendations

The platform analyzes your task description and intelligently recommends the optimal agent workflow:

```
Task Description → NLP Analysis → Agent Recommendation → Workflow Execution
```

## Core Examples

### 1. Customer Churn Prediction
**Task:** "Predict customer churn from usage patterns and demographics"  
**AI Recommendation:** `eda → classification`

**CLI:**
```bash
python quick_start.py --task "Predict customer churn from usage patterns" --target churn
```

**API:**
```bash
curl -X POST "http://localhost:8000/analyze" \
  -d '{"task_description": "Predict customer churn from usage patterns", "target_column": "churn"}'
```

**Streamlit:** Navigate to "🎯 Custom Task" → Enter description → Get AI recommendation

### 2. Sales Forecasting with Data Issues
**Task:** "Forecast daily sales with missing data and optimize performance"  
**AI Recommendation:** `eda → data_hygiene → time_series → hyperparameter_tuning`

### 3. Sentiment Analysis  
**Task:** "Analyze customer reviews for sentiment"  
**AI Recommendation:** `eda → nlp`

### 4. Image Classification
**Task:** "Classify product images with feature engineering"  
**AI Recommendation:** `computer_vision → feature_engineering`

## Keyword Detection

| Category | Keywords | Triggers |
|----------|----------|----------|
| **Data Quality** | missing, dirty, clean, outlier | `data_hygiene` |
| **Feature Work** | feature, transform, encode | `feature_engineering` |
| **Classification** | classify, predict, category | `classification` |
| **Regression** | forecast value, continuous | `regression` |
| **Time Series** | forecast, temporal, daily | `time_series` |
| **NLP** | text, sentiment, reviews | `nlp` |
| **Computer Vision** | image, photo, visual | `computer_vision` |
| **Optimization** | optimize, best, accuracy | `hyperparameter_tuning` |

## Interface-Specific Usage

### Command Line
```bash
# Basic usage
python quick_start.py --task "Your ML task description" --target column_name

# Custom agents
python quick_start.py --agents "eda,classification,hyperparameter_tuning"

# Demo mode
python quick_start.py --demo --task "classification example"
```

### REST API
```bash
# Complete AutoML workflow
curl -X POST "http://localhost:8000/automl" \
  -F "file=@data.csv" \
  -F "target_column=target" \
  -F "task_description=Your ML task"

# Get agent recommendations
curl -X POST "http://localhost:8000/analyze" \
  -d '{"task_description": "Your task", "target_column": "target"}'
```

### Streamlit Interface
1. Navigate to "🎯 Custom Task"
2. Enter task description
3. Review AI recommendation with explanations  
4. Customize agent selection if needed
5. Upload data and run workflow

## Smart Routing Examples

### Simple Clean Data → Direct Classification
```
Input: "Classify clean customer data" (high quality, few features)
Routing: Skip EDA → classification (only)
Reason: Clean data doesn't need full pipeline
```

### Complex Data → Full Pipeline
```
Input: "Predict churn with missing values and optimize"  
Routing: eda → data_hygiene → classification → hyperparameter_tuning
Reason: Data issues + optimization request
```

## Best Practices

1. **Be Descriptive**: Include domain context and data characteristics
2. **Mention Issues**: Specify "missing data", "dirty data", "outliers"  
3. **State Goals**: Use "optimize", "best accuracy", "fast results"
4. **Trust AI First**: Start with recommendations, then customize
5. **Choose Right Interface**: CLI for automation, API for integration, Streamlit for exploration

For more detailed examples, see the `/examples/` directory with interface-specific documentation.