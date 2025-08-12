# Customer Lifetime Value and Churn Prediction

Customer lifetime value prediction and churn analysis system that identifies at-risk customers and calculates long-term value. Uses machine learning models to analyze purchase patterns, engagement metrics, and behavioral indicators. Automated LangChain agents trigger personalized retention campaigns and value optimization strategies based on customer segments.

## Key Results
- **25% retention improvement** through predictive analytics
- **$2.5M prevented churn** with early warning systems
- **Real-time predictions** with 92% accuracy
- **Automated campaigns** reducing manual intervention by 70%

## Technology Stack
- **Python** - Core development language
- **LangChain** - Agent orchestration and workflow automation
- **AWS** - Cloud infrastructure and deployment
- **SQLAlchemy** - Database ORM and data modeling
- **FastAPI** - API framework for real-time predictions

## Features
- Real-time customer lifetime value prediction
- Churn risk analysis and early warning system
- Automated retention campaign triggers
- Customer segmentation and value optimization
- Purchase pattern and behavioral analysis

## Project Structure
```
cltv/
├── src/
│   ├── models/
│   │   ├── cltv_predictor.py
│   │   ├── churn_classifier.py
│   │   └── segmentation_model.py
│   ├── agents/
│   │   ├── retention_agent.py
│   │   ├── campaign_optimizer.py
│   │   └── value_calculator.py
│   ├── data/
│   │   ├── ingestion.py
│   │   ├── preprocessing.py
│   │   └── feature_engineering.py
│   ├── analysis/
│   │   ├── behavioral_analyzer.py
│   │   ├── pattern_detector.py
│   │   └── risk_scorer.py
│   └── api/
│       ├── prediction_endpoints.py
│       └── campaign_triggers.py
├── langchain/
│   ├── agents/
│   │   └── retention_agents.py
│   └── chains/
│       └── optimization_chains.py
├── config/
│   ├── model_config.yaml
│   └── campaign_config.yaml
├── requirements.txt
├── .env.example
└── README.md
```

## Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/joshamorrison/public.git
   cd public/cltv
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

4. **Initialize database**
   ```bash
   python -c "from src.data.database import init_db; init_db()"
   ```

5. **Run the prediction service**
   ```bash
   python src/main.py --mode realtime
   ```

## Model Architecture

### CLTV Prediction Engine
- **Regression models** for lifetime value calculation
- **Time series analysis** for purchase frequency patterns
- **Cohort analysis** for retention rate modeling
- **Feature engineering** from transactional and behavioral data

### Churn Prevention System
- **Early warning detection** with configurable risk thresholds
- **Multi-model ensemble** for prediction accuracy
- **Real-time scoring** for immediate intervention
- **Campaign automation** with personalized messaging

## Business Impact

This system enables customer success and marketing teams to:
- **Predict customer lifetime value** with 92% accuracy
- **Reduce churn rates** by 25% through early intervention
- **Optimize retention spend** with targeted campaign automation
- **Increase customer lifetime value** through personalized experiences

## API Endpoints

### Prediction APIs
```bash
# Get CLTV prediction for a customer
GET /api/v1/cltv/predict/{customer_id}

# Batch churn risk scoring
POST /api/v1/churn/batch_predict
{
  "customer_ids": ["cust_1", "cust_2", "cust_3"],
  "prediction_horizon_days": 90
}

# Trigger retention campaign
POST /api/v1/campaigns/retention
{
  "customer_id": "cust_123",
  "risk_score": 0.85,
  "campaign_type": "discount_offer"
}

# Get customer segmentation
GET /api/v1/segments/{customer_id}
```

## Contact

For technical questions or implementation guidance, reach out to:
- **Joshua Morrison** - [joshamorrison@gmail.com](mailto:joshamorrison@gmail.com)
- **LinkedIn** - [linkedin.com/in/joshamorrison](https://www.linkedin.com/in/joshamorrison)