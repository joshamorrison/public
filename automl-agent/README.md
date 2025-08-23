# AutoML Agent Platform

🤖 **AI-driven multi-agent AutoML platform** that intelligently analyzes your task description and recommends the optimal ML workflow.

## ✨ Key Features

- **🧠 AI Agent Recommendations**: Natural language task analysis → intelligent agent selection
- **🔄 Multi-Agent Architecture**: Specialized agents for EDA, classification, NLP, computer vision, etc.
- **🎯 Real ML Workflows**: Actual scikit-learn models, not simulations (79-89% accuracy achieved)
- **🌐 Three Interfaces**: Command line, REST API, and interactive Streamlit web app
- **⚡ Smart Routing**: Skip unnecessary steps for clean data, full pipeline for complex data

## 🚀 Quick Start

**1. Install Dependencies:**
```bash
python -m venv .venv
.venv/Scripts/activate  # Windows
pip install -r requirements.txt
```

**2. Run Demo:**
```bash
# Command Line
python quick_start.py --task "Predict customer churn from usage patterns" --target churn

# Web Interface  
streamlit run infrastructure/streamlit/app.py

# REST API
python -m uvicorn src.api.main:app --port 8000
```

**3. Example Result:**
```
🤖 AI Recommendation: eda → classification
[EDA Agent] Analyzing 5,000 customers, 15 features
[Classification Agent] RandomForest: 89.2% accuracy
✅ Workflow completed in 45 seconds
```

## 📖 Documentation

- **[Examples & Usage](docs/examples.md)** - Comprehensive examples for all interfaces
- **[API Reference](docs/api_examples.md)** - REST API documentation  
- **[Project Manifest](PROJECT_MANIFEST.md)** - Detailed project roadmap and architecture

## 🔧 Interfaces

| Interface | Use Case | Access |
|-----------|----------|---------|
| **Command Line** | Automation, scripting | `python quick_start.py --task "..."` |
| **REST API** | Application integration | `http://localhost:8000/docs` |
| **Streamlit Web** | Interactive exploration | `http://localhost:8501` |

## 🎯 Example Workflows

**Customer Churn:** "Predict customer churn from usage patterns" → `eda → classification`  
**Sales Forecasting:** "Forecast daily sales with missing data" → `eda → data_hygiene → time_series`  
**Sentiment Analysis:** "Analyze reviews for sentiment" → `eda → nlp`  
**Image Classification:** "Classify product images" → `computer_vision`

## Project Structure
```
automl-agent/
├── src/
│   ├── agents/
│   │   ├── data_analyst_agent.py
│   │   ├── feature_engineer_agent.py
│   │   ├── model_selector_agent.py
│   │   ├── optimizer_agent.py
│   │   └── validator_agent.py
│   ├── eda/
│   │   ├── data_profiler.py
│   │   ├── visualization_generator.py
│   │   └── statistical_analyzer.py
│   ├── feature_engineering/
│   │   ├── automated_features.py
│   │   ├── feature_selector.py
│   │   └── transformation_pipeline.py
│   ├── models/
│   │   ├── model_factory.py
│   │   ├── ensemble_builder.py
│   │   └── meta_learner.py
│   ├── optimization/
│   │   ├── hyperparameter_tuner.py
│   │   ├── neural_architecture_search.py
│   │   └── bayesian_optimizer.py
│   ├── validation/
│   │   ├── cross_validator.py
│   │   ├── performance_evaluator.py
│   │   └── model_explainer.py
│   └── deployment/
│       ├── model_deployer.py
│       ├── monitoring_setup.py
│       └── retraining_scheduler.py
├── crewai/
│   ├── crews/
│   │   └── automl_crew.py
│   ├── tasks/
│   │   ├── eda_tasks.py
│   │   ├── modeling_tasks.py
│   │   └── validation_tasks.py
│   └── tools/
│       ├── data_tools.py
│       └── ml_tools.py
├── langsmith/
│   ├── tracking/
│   │   └── experiment_tracker.py
│   └── evaluation/
│       └── model_evaluator.py
├── aws_sagemaker/
│   ├── autopilot_integration.py
│   └── model_endpoints.py
├── config/
│   ├── agent_config.yaml
│   └── model_config.yaml
├── requirements.txt
├── .env.example
└── README.md
```

## Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/joshamorrison/public.git
   cd public/automl-agent
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

4. **Initialize tracking systems**
   ```bash
   export LANGCHAIN_TRACING_V2=true
   export LANGCHAIN_API_KEY=your_langsmith_key
   ```

5. **Run AutoML pipeline**
   ```bash
   python src/main.py --dataset ./data/training_data.csv --target column_name
   ```

## Contact

For technical questions or implementation guidance, reach out to:
- **Joshua Morrison** - [joshamorrison@gmail.com](mailto:joshamorrison@gmail.com)
- **LinkedIn** - [linkedin.com/in/joshamorrison](https://www.linkedin.com/in/joshamorrison)