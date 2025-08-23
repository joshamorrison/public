# Streamlit Interface Examples

## AI-Driven Custom Task Creation

### 1. Customer Churn Prediction

**Step 1: Navigate to Custom Task**
- Go to "🎯 Custom Task" in sidebar

**Step 2: Fill Task Details**
```
Task Name: Customer Retention Analysis
Task Description: Predict customer churn from usage patterns and demographics to improve retention strategies
Target Column: churn_flag
Task Type: classification
Quality Threshold: 0.85
```

**Step 3: AI Recommendation**
✅ **AI Recommendation:** `eda → classification`

**Reasoning shown:**
- 📊 **EDA**: Understand customer behavior patterns
- 🎯 **Classification**: Binary prediction task (churn/no churn)

**Step 4: Review and Customize**
- AI suggests: `["eda", "classification"]`
- User can add: `"hyperparameter_tuning"` for better performance
- Final workflow: `eda → classification → hyperparameter_tuning`

### 2. Sales Forecasting with Data Issues

**Task Details:**
```
Task Name: Monthly Sales Forecast
Task Description: Forecast monthly sales for next 12 months with missing data and optimize performance
Target Column: monthly_sales
Task Type: time_series
Quality Threshold: 0.80
```

**AI Recommendation:** `eda → data_hygiene → time_series → hyperparameter_tuning`

**Reasoning displayed:**
- 📊 **EDA**: Understand sales patterns and trends
- 🧹 **Data Hygiene**: Your description suggests data quality concerns
- 📅 **Time Series**: Detected temporal/forecasting requirements
- 🔧 **Hyperparameter Tuning**: Performance optimization requested

### 3. Sentiment Analysis

**Task Details:**
```
Task Name: Review Sentiment Analysis
Task Description: Analyze customer reviews for sentiment and extract key topics from text
Target Column: sentiment
Task Type: nlp
```

**AI Recommendation:** `eda → nlp`

**Reasoning:**
- 📊 **EDA**: Basic text statistics and patterns
- 💬 **NLP**: Text processing task identified

### 4. Product Image Classification

**Task Details:**
```
Task Name: Product Category Classification
Task Description: Classify product images into categories using computer vision with feature engineering
Target Column: product_category
Task Type: computer_vision
```

**AI Recommendation:** `computer_vision → feature_engineering`

**Reasoning:**
- 👁️ **Computer Vision**: Image classification task
- ⚙️ **Feature Engineering**: Detected need for feature transformation

### 5. Customer Segmentation

**Task Details:**
```
Task Name: Behavioral Customer Segments
Task Description: Group customers into segments based on purchasing behavior patterns for marketing
Task Type: clustering
```

**AI Recommendation:** `eda → feature_engineering → clustering`

**Reasoning:**
- 📊 **EDA**: Understand customer behavior patterns
- ⚙️ **Feature Engineering**: Create behavioral features
- 🎨 **Clustering**: Unsupervised grouping task

## Interactive Workflow Customization

### Scenario: Fraud Detection System

**Initial Task:**
```
Task Description: Build best possible fraud detection model with ensemble methods and dirty data for production use
```

**AI Analysis & Recommendation:**
```
🤖 AI Recommendation based on your description: 
eda → data_hygiene → feature_engineering → classification → ensemble → hyperparameter_tuning
```

**Expandable Reasoning:**
```
Why these agents?
📊 EDA: Understand data patterns and characteristics
🧹 Data Hygiene: Your description suggests data quality concerns  
⚙️ Feature Engineering: Detected need for feature transformation
🎯 Classification: Task involves predicting categories/classes
🤝 Ensemble: Multiple model combination requested
🔧 Hyperparameter Tuning: Performance optimization requested
```

**User Customization Options:**
- ✅ Keep all recommended agents
- ➕ Add "validation" for extra model validation
- ➖ Remove "hyperparameter_tuning" for faster development
- 🔄 Reorder agents (drag and drop interface)

## Live Demo Workflows

### Quick Start Demo

**Demonstration 1: Telco Churn (Real Data)**
```
📊 Dataset: Telco Customer Churn (7,032 customers, 21 features)
🎯 Task: Binary classification (churn/no churn)
⚡ Workflow: eda → classification (intelligent routing)
📈 Result: 89.2% accuracy with RandomForest

Key Insights:
- Monthly charges most important feature (0.28 importance)
- Contract type strongly correlates with churn
- 26.6% churn rate in dataset
```

**Demonstration 2: California Housing (Real Data)**  
```
🏠 Dataset: California Housing (20,640 houses, 9 features)
🎯 Task: Regression (house price prediction)  
⚡ Workflow: eda → regression
📈 Result: R² = 0.85, RMSE = $45,000

Key Insights:
- Median income most predictive feature
- Geographic location clusters visible
- Price range: $15k - $500k
```

### Agent Analytics Dashboard

**Performance Metrics Display:**
```
Active Agents: 8 (+2)
Tasks Completed: 1,247 (+156) 
Average Accuracy: 87.3% (+2.1%)
Success Rate: 94.2% (+0.8%)
```

**Agent Performance Charts:**
- Line chart showing success rates over time for each agent
- Processing time trends by agent type
- Task completion distribution

**Current Agent Status Table:**
| Agent | Status | Current Tasks | Queue Length | Last Activity |
|-------|---------|---------------|--------------|---------------|
| EDA Agent | Active | 3 | 5 | 2 min ago |
| Classification Agent | Active | 2 | 3 | 1 min ago |
| Feature Engineering | Active | 1 | 2 | 5 min ago |
| Data Hygiene | Idle | 0 | 0 | 1 hour ago |

### Agent Communication Hub

**Real-time Communication Log:**
```
10:32:15 | EDA Agent → Feature Engineering | "Data profile complete, found 3 categorical features"
10:32:18 | Feature Engineering → Classification Agent | "Features encoded, ready for training"
10:32:45 | Classification Agent → EDA Agent | "Model accuracy: 89.2%, requesting feature importance"
10:33:02 | Data Hygiene → All Agents | "Data quality check complete, no issues found"
10:33:15 | Classification Agent → Router Agent | "Classification task completed successfully"
```

**Communication Statistics:**
```
Messages Today: 1,423 (+234)
Active Conversations: 12 (+3)
Avg Response Time: 0.8s (-0.2s)
```

**Message Type Distribution (Pie Chart):**
- Data Transfer: 35%
- Status Updates: 28%
- Error Reports: 8%
- Task Requests: 22%
- Results Sharing: 7%

## Interactive Features

### 1. Dynamic Agent Selection
- **Task Type Change**: Selecting "time_series" auto-updates suggested agents
- **Real-time Validation**: Shows warnings if incompatible agents selected
- **Workflow Preview**: Visual representation of agent sequence

### 2. Data Upload Interface
```
📁 Supported Formats: CSV, Excel (.xlsx, .xls)
📊 Preview: Shows first 5 rows after upload
🔍 Column Detection: Auto-identifies potential target columns
⚡ Quick Stats: Rows, columns, data types, missing values
```

### 3. Progress Monitoring
```
🚀 Workflow Status: Running...
📊 Current Agent: EDA Agent
⏳ Progress: 45% complete
🕐 Estimated Time: 2 minutes remaining
📝 Live Log: "Generating correlation matrix..."
```

### 4. Results Visualization
```
📈 Model Performance:
- Accuracy: 89.2%
- Precision: 87.5%  
- Recall: 91.1%
- F1-Score: 89.2%

📊 Feature Importance:
[Interactive bar chart showing top 10 features]

🎯 Confusion Matrix:
[Interactive heatmap with hover details]

💡 Recommendations:
- "Consider feature engineering on tenure data"
- "Address slight class imbalance"
- "Monitor model performance monthly"
```

## Tips for Using Streamlit Interface

### 1. Task Description Best Practices
- **Be Specific**: "Predict customer churn from usage data" vs "classification"
- **Include Context**: Mention domain, data type, performance needs
- **Use Keywords**: "optimize", "missing data", "ensemble" trigger relevant agents

### 2. Agent Selection
- **Trust AI First**: Start with AI recommendation, then customize
- **Check Dependencies**: Some agents require others (ensemble needs classification)
- **Consider Data Size**: Large datasets may need data_hygiene first

### 3. Performance Optimization
- **Quality Threshold**: Lower (0.7) for exploration, higher (0.9) for production
- **Max Iterations**: More iterations = better results but longer time
- **Agent Order**: EDA first, then domain-specific agents

### 4. Data Upload
- **Column Names**: Use descriptive names, avoid spaces/special characters
- **Target Column**: Ensure target exists and has reasonable distribution
- **Sample Size**: Start with subset for testing, full data for final models

### 5. Results Interpretation
- **Check All Tabs**: EDA insights, model performance, agent logs
- **Download Results**: Save models, reports, visualizations
- **Review Recommendations**: Agent suggestions for improvement

## Advanced Use Cases

### A/B Testing Framework
```
Task: Compare model performance across customer segments
Workflow: eda → feature_engineering → classification → ensemble
Special: Run parallel workflows for different segments
```

### Real-time Model Deployment
```
Task: Deploy fraud detection model with monitoring
Workflow: eda → classification → validation → deployment
Features: Model versioning, performance monitoring, alerts
```

### Multi-modal Analysis
```
Task: Combine text reviews, images, and structured data
Workflow: eda → nlp → computer_vision → feature_engineering → classification
Challenge: Integrate different data types effectively
```

The Streamlit interface makes AI-driven AutoML accessible through an intuitive, interactive web application!