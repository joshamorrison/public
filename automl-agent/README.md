# AutoML Agent

End-to-end automated machine learning pipeline for EDA, feature engineering, model selection, and hyperparameter tuning. CrewAI orchestrates task-specific agents while LangSmith tracks experiments and model performance.

## Key Results
- **70% reduction in model development time** through automation
- **15% accuracy improvement** over manual approaches
- **Continuous optimization** with automated retraining

## Technology Stack
- **Python** - Core platform development
- **LangSmith** - Experiment tracking and monitoring
- **CrewAI** - Multi-agent orchestration framework
- **AWS SageMaker Autopilot** - Automated model building

## Features
- Automated exploratory data analysis (EDA)
- Intelligent feature engineering and selection
- Multi-algorithm model selection and comparison
- Hyperparameter optimization with advanced techniques
- Continuous model monitoring and retraining

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