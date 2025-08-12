# Media Mix Modeling & Optimization

Advanced media mix models and econometric forecasting to optimize campaign spend and channel allocation. Incorporates real-time performance data and predictive analytics to maximize ROI.

## Key Results
- **+18% ROAS** improvement across campaigns
- **12% lower CAC** through optimized channel allocation
- **Optimized cross-channel allocation** with real-time adjustments

## Technology Stack
- **Python** - Core modeling and data processing
- **R** - Econometric modeling and statistical analysis
- **AWS SageMaker** - Model training and deployment
- **MLflow** - Experiment tracking and model versioning
- **Apache Airflow** - Workflow orchestration and scheduling

## Features
- Advanced econometric models for media attribution
- Real-time performance data integration
- Cross-channel budget optimization
- Automated retraining pipelines
- ROI prediction and scenario modeling

## Project Structure
```
media-mix-modeling/
├── src/
│   ├── models/
│   │   ├── econometric_models.py
│   │   └── attribution_models.py
│   ├── data/
│   │   ├── ingestion.py
│   │   └── preprocessing.py
│   ├── optimization/
│   │   └── channel_optimizer.py
│   └── utils/
│       └── roi_calculator.py
├── airflow/
│   └── dags/
│       └── media_mix_pipeline.py
├── config/
│   └── model_config.yaml
├── requirements.txt
├── .env.example
└── README.md
```

## Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/joshamorrison/public.git
   cd public/media-mix-modeling
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

4. **Initialize MLflow**
   ```bash
   mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./artifacts
   ```

5. **Run the pipeline**
   ```bash
   python src/main.py
   ```

## Model Architecture

### Econometric Foundation
- Adstock transformation for carryover effects
- Saturation curves for diminishing returns
- Base vs. incremental lift decomposition
- Seasonality and trend adjustments

### Optimization Engine
- Multi-objective optimization (ROI, reach, frequency)
- Constraint handling for budget limits
- Real-time bid adjustment algorithms
- Cross-channel synergy modeling

## Business Impact

This solution enables marketing teams to:
- **Optimize budget allocation** across 10+ channels
- **Predict campaign performance** with 95% accuracy
- **Automate media planning** reducing manual effort by 60%
- **Measure true incremental lift** from each channel

## Contact

For technical questions or implementation guidance, reach out to:
- **Joshua Morrison** - [joshamorrison@gmail.com](mailto:joshamorrison@gmail.com)
- **LinkedIn** - [linkedin.com/in/joshamorrison](https://www.linkedin.com/in/joshamorrison)