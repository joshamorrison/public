# Command Line Interface Examples

## Quick Start Examples

### 1. Customer Churn Prediction
```bash
# Basic classification task
python quick_start.py --task "Predict customer churn from usage patterns" --target churn_flag

# Expected output:
# AI Recommendation: eda → classification
# [EDA Agent] Starting analysis...
# [Classification Agent] Training models...
# Result: 89.2% accuracy achieved
```

### 2. Sales Forecasting
```bash
# Time series forecasting
python quick_start.py --task "Forecast daily sales for next 30 days" --target daily_sales

# Expected output:
# AI Recommendation: eda → time_series
# [EDA Agent] Analyzing temporal patterns...
# [Time Series Agent] Building forecasting model...
# Result: Forecast generated with MAPE: 8.3%
```

### 3. Sentiment Analysis
```bash
# NLP task
python quick_start.py --task "Analyze customer reviews for sentiment" --target sentiment --data reviews.csv

# Expected output:
# AI Recommendation: eda → nlp
# [EDA Agent] Text statistics: 5000 reviews, avg length 45 words
# [NLP Agent] Training sentiment classifier...
# Result: 92.1% sentiment classification accuracy
```

### 4. Image Classification
```bash
# Computer vision task
python quick_start.py --task "Classify product images into categories" --target product_type --data images/

# Expected output:
# AI Recommendation: computer_vision
# [Computer Vision Agent] Processing 1000 images...
# Result: 94.7% image classification accuracy
```

### 5. Customer Segmentation
```bash
# Clustering task
python quick_start.py --task "Group customers into segments based on behavior" --data customer_behavior.csv

# Expected output:
# AI Recommendation: eda → feature_engineering → clustering
# [EDA Agent] Found 15 behavioral features...
# [Feature Engineering Agent] Created interaction features...
# [Clustering Agent] Identified 4 distinct customer segments
```

## Advanced Command Line Usage

### High-Performance Mode
```bash
# Optimize for best accuracy
python quick_start.py \
  --task "Build best fraud detection model with ensemble methods" \
  --target is_fraud \
  --optimize-performance \
  --max-time 30

# Expected output:
# AI Recommendation: eda → data_hygiene → classification → ensemble → hyperparameter_tuning
# [Ensemble Agent] Combining 5 models...
# Result: 96.8% fraud detection accuracy
```

### Custom Workflow
```bash
# Override AI recommendation
python quick_start.py \
  --task "Predict housing prices" \
  --target price \
  --agents "eda,feature_engineering,regression,hyperparameter_tuning"

# Skips AI recommendation, uses specified agents
```

### Batch Processing
```bash
# Process multiple datasets
python quick_start.py \
  --task "Customer analysis across regions" \
  --batch-data "data/region_*.csv" \
  --target churn

# Processes all matching files with same workflow
```

## Environment-Specific Examples

### Development Mode
```bash
# Quick testing with sample data
python quick_start.py --demo --task "classification example"
# Uses built-in sample data
```

### Production Mode
```bash
# Full pipeline with logging
python quick_start.py \
  --task "Production churn prediction" \
  --target churn \
  --data production_data.csv \
  --output-dir results/ \
  --log-level INFO \
  --save-models

# Saves models and detailed logs
```

### Distributed Processing
```bash
# Use multiple cores
python quick_start.py \
  --task "Large dataset classification" \
  --target outcome \
  --data big_data.csv \
  --n-jobs 8 \
  --memory-efficient

# Optimizes for large datasets
```

## Output Examples

### Successful Completion
```
[2024-08-23 10:30:15] Starting AutoML workflow...
[2024-08-23 10:30:16] AI Recommendation: eda → classification
[2024-08-23 10:30:16] [EDA Agent] Analyzing dataset: 5000 rows, 15 features
[2024-08-23 10:30:18] [EDA Agent] Quality score: 0.92, no missing values
[2024-08-23 10:30:20] [Classification Agent] Training RandomForest, LogisticRegression
[2024-08-23 10:30:25] [Classification Agent] Best model: RandomForest (89.2% accuracy)
[2024-08-23 10:30:25] ✅ Workflow completed successfully!

Results saved to: outputs/results_20240823_103025/
- model.pkl (trained model)
- results.json (performance metrics)
- eda_report.html (data analysis)
- feature_importance.png (visualizations)
```

### Error Handling
```
[2024-08-23 10:30:15] Starting AutoML workflow...
[2024-08-23 10:30:16] ❌ Error: Target column 'churn' not found in dataset
Available columns: ['customer_id', 'usage_hours', 'subscription_type']

Suggestions:
- Check column name spelling
- Use --list-columns to see all columns
- Update target column parameter
```

## Command Line Arguments Reference

```bash
python quick_start.py [OPTIONS]

Core Arguments:
  --task TEXT           Task description for AI analysis
  --target TEXT         Target column name
  --data PATH          Data file path (CSV, Excel, etc.)
  --agents TEXT        Comma-separated agent list (overrides AI)

AI Options:
  --auto-recommend     Use AI agent recommendations (default)
  --task-type TEXT     Override detected task type
  --quality-threshold  Minimum quality threshold (0.0-1.0)

Performance:
  --optimize           Enable performance optimization
  --n-jobs INTEGER     Number of parallel jobs
  --max-time INTEGER   Maximum runtime in minutes
  --memory-efficient   Optimize for large datasets

Output:
  --output-dir PATH    Results output directory
  --save-models        Save trained models
  --log-level TEXT     Logging level (INFO, DEBUG, WARNING)
  --format TEXT        Output format (json, csv, html)

Demo/Testing:
  --demo              Use sample data
  --list-agents       Show available agents
  --list-columns      Show dataset columns
  --dry-run           Plan workflow without execution
```

## Tips and Best Practices

1. **Start Simple**: Begin with basic task descriptions, let AI recommend
2. **Be Specific**: Include domain context for better recommendations
3. **Use Demo Mode**: Test workflows with `--demo` before real data
4. **Check Columns**: Use `--list-columns` to verify target column exists
5. **Save Results**: Always use `--output-dir` for important runs
6. **Monitor Progress**: Use `--log-level INFO` to see detailed progress
7. **Optimize When Needed**: Add `--optimize` for production models

The command line interface provides powerful automation while remaining simple for quick experimentation!