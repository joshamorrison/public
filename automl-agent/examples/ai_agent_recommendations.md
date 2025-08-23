# AI-Driven Agent Recommendation Examples

This document shows how the AutoML platform intelligently analyzes task descriptions and recommends appropriate agent workflows across all interfaces.

## How It Works

The AI analyzes your task description for keywords and context to recommend the optimal sequence of agents:

```
Task Description → NLP Analysis → Agent Recommendation → Workflow Execution
```

## Example Workflows

### 1. Customer Churn Prediction

**Task Description:** "Predict customer churn from usage patterns and demographics"

**AI Analysis:** 
- Detects: ["predict", "customer", "churn"]
- Task Type: Classification

**Recommended Workflow:** `eda → classification`

**Reasoning:**
- 📊 **EDA**: Understand customer behavior patterns
- 🎯 **Classification**: Binary prediction task (churn/no churn)

### 2. Sales Forecasting with Data Quality Issues

**Task Description:** "Forecast daily sales for next 30 days with missing data and optimize performance"

**AI Analysis:**
- Detects: ["forecast", "daily", "missing", "optimize"]
- Task Type: Time Series

**Recommended Workflow:** `eda → data_hygiene → time_series → hyperparameter_tuning`

**Reasoning:**
- 📊 **EDA**: Understand sales patterns and trends
- 🧹 **Data Hygiene**: Address missing data issues
- 📅 **Time Series**: Temporal forecasting task
- 🔧 **Hyperparameter Tuning**: Optimization requested

### 3. Sentiment Analysis

**Task Description:** "Analyze customer reviews for sentiment and extract key topics"

**AI Analysis:**
- Detects: ["reviews", "sentiment", "text", "language"]
- Task Type: NLP

**Recommended Workflow:** `eda → nlp`

**Reasoning:**
- 📊 **EDA**: Basic text statistics and patterns
- 💬 **NLP**: Text processing and sentiment analysis

### 4. Product Image Classification

**Task Description:** "Classify product images with feature engineering for better accuracy"

**AI Analysis:**
- Detects: ["classify", "images", "feature", "engineering"]
- Task Type: Computer Vision

**Recommended Workflow:** `computer_vision → feature_engineering`

**Reasoning:**
- 👁️ **Computer Vision**: Image classification task
- ⚙️ **Feature Engineering**: Explicit feature enhancement request

### 5. Customer Segmentation

**Task Description:** "Group customers into segments based on purchasing behavior patterns"

**AI Analysis:**
- Detects: ["group", "segment", "behavior", "patterns"]
- Task Type: Clustering

**Recommended Workflow:** `eda → feature_engineering → clustering`

**Reasoning:**
- 📊 **EDA**: Understand customer behavior patterns
- ⚙️ **Feature Engineering**: Create behavioral features
- 🎨 **Clustering**: Unsupervised grouping task

### 6. High-Performance Fraud Detection

**Task Description:** "Build best possible fraud detection model with ensemble methods and dirty data"

**AI Analysis:**
- Detects: ["fraud", "best", "ensemble", "dirty"]
- Task Type: Classification

**Recommended Workflow:** `eda → data_hygiene → feature_engineering → classification → ensemble → hyperparameter_tuning`

**Reasoning:**
- 📊 **EDA**: Understand fraud patterns
- 🧹 **Data Hygiene**: Clean dirty data
- ⚙️ **Feature Engineering**: Create fraud indicators
- 🎯 **Classification**: Binary fraud/legitimate classification
- 🤝 **Ensemble**: Multiple model combination
- 🔧 **Hyperparameter Tuning**: Maximize performance

## Usage Across Interfaces

### Command Line Interface
```bash
python quick_start.py --task "Predict customer churn from usage patterns" --target churn
# AI recommends: eda → classification
```

### REST API
```bash
curl -X POST "http://localhost:8000/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "task_description": "Forecast daily sales for next 30 days",
    "target_column": "sales"
  }'
# Returns: {"recommended_agents": ["eda", "time_series"]}
```

### Streamlit Interface
1. Navigate to "🎯 Custom Task" 
2. Enter: "Analyze customer reviews for sentiment"
3. See AI recommendation: `eda → nlp`
4. Customize if needed and run workflow

## Keyword Detection Categories

| Category | Keywords | Triggers |
|----------|----------|----------|
| **Data Quality** | missing, dirty, clean, outlier, quality | data_hygiene |
| **Feature Engineering** | feature, transform, encode, scale, engineer | feature_engineering |
| **Classification** | predict, classify, category, class, label | classification |
| **Regression** | forecast value, estimate, regression, continuous | regression |
| **Time Series** | forecast, temporal, daily, monthly, trend | time_series |
| **NLP** | text, sentiment, language, review, comment | nlp |
| **Computer Vision** | image, photo, picture, vision, visual | computer_vision |
| **Clustering** | cluster, group, segment, unsupervised | clustering |
| **Optimization** | optimize, best, high accuracy, performance | hyperparameter_tuning |
| **Ensemble** | ensemble, combine models, voting, stacking | ensemble |

## Best Practices

1. **Be Descriptive**: Include specific details about your data and goals
2. **Mention Data Issues**: If you have missing values, outliers, etc.
3. **Specify Performance Needs**: Use words like "optimize", "best", "high accuracy"
4. **Include Domain Context**: "customer", "sales", "fraud", "sentiment" help with recommendations
5. **Review and Customize**: AI recommendations are starting points - adjust as needed

## Advanced Examples

### Multi-Domain Analysis
**Task:** "Comprehensive analysis of customer data including transaction history, review text, and profile images"

**Recommendation:** `eda → nlp → computer_vision → feature_engineering → classification`

### Real-Time Prediction
**Task:** "Real-time fraud detection with feature engineering and ensemble methods for high accuracy"

**Recommendation:** `eda → feature_engineering → classification → ensemble → hyperparameter_tuning`

### Experimental Research
**Task:** "Explore unsupervised patterns in gene expression data with dimensionality reduction"

**Recommendation:** `eda → feature_engineering → clustering`

The AI recommendation system makes the AutoML platform accessible to both beginners and experts by intelligently interpreting natural language requirements and suggesting optimal workflows.