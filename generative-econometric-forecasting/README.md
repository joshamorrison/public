# 🚀 Generative Econometric Forecasting Platform

**Revolutionary AI-powered economic forecasting with three-tier foundation model system**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![LangSmith](https://img.shields.io/badge/LangSmith-Enabled-green.svg)](docs/LANGSMITH_INTEGRATION.md)

Advanced econometric forecasting platform featuring the world's first **three-tier foundation model ecosystem** - from completely free professional-grade models to cutting-edge paid APIs. Combines rigorous statistical models with generative AI to produce accurate forecasts and executive-ready business insights.

## ✨ Key Features

- **🏆 Three-tier foundation models** - Free professional to premium AI (MLForecast + HuggingFace + TimeGPT)
- **📊 Real-time FRED data** - Live economic data integration (GDP, unemployment, inflation)
- **🤖 AI-powered analysis** - OpenAI → HuggingFace → Smart templates with intelligent fallbacks
- **🧠 30+ neural models** - Advanced neural forecasting with ensemble capabilities
- **📰 Sentiment integration** - News sentiment analysis for adjusted predictions
- **📈 Executive reporting** - Professional JSON, CSV, and executive summary generation
- **🔄 Always works** - Bulletproof fallback system ensures platform never fails

## 🏗️ Three-Tier Foundation Model System

| **Tier** | **Provider** | **Cost** | **Performance** | **Best For** |
|-----------|--------------|----------|-----------------|--------------|
| **🏆 Tier 1** | Nixtla TimeGPT | 💰 Paid API | ⭐⭐⭐⭐⭐ Premium | Production systems |
| **🥇 Tier 2** | Nixtla Open Source | 🆓 Free | ⭐⭐⭐⭐ Professional | Most users |
| **🥈 Tier 3** | HuggingFace + Fallbacks | 🆓 Free | ⭐⭐⭐ Good | Always available |

## 🚀 Quick Start

Get from clone to AI-powered forecasting in under 5 minutes:

```bash
# 1. Clone and setup
git clone https://github.com/joshamorrison/public.git
cd public/generative-econometric-forecasting

# 2. Create virtual environment (recommended)
python -m venv venv
source venv/Scripts/activate  # Windows
# source venv/bin/activate    # macOS/Linux

# 3. Install dependencies (~2 minutes)
pip install -r requirements.txt

# 4. Run the demo (~60 seconds)
python quick_start.py
```

**Expected Output:**
```
🚀 GENERATIVE ECONOMETRIC FORECASTING - QUICK START DEMO
======================================================================
[OK] FRED API: CONNECTED (Real economic data available)
[SUCCESS] Real economic data loaded (3 indicators)

[FORECAST] Forecasting GDP...
   [DATA] Last actual (Jul 2025): 23685.3
   [6M-AVG] 6-month forecast avg: 23915.1
   [JAN-2026] January 2026: 24283.0

[AI] AI Analysis Status: LOCAL (HuggingFace)
[TARGET] SYSTEM STATUS: [TIER2] PROFESSIONAL READY
🎉 QUICK START DEMO COMPLETE!
```

## 📁 Project Structure

```
generative-econometric-forecasting/
├── data/                          # Data handling & API clients
│   ├── fred_client.py             # FRED API integration
│   └── unstructured/              # News & sentiment analysis
├── models/                        # Forecasting models & algorithms
│   ├── forecasting_models.py      # ARIMA, Prophet, VAR models
│   ├── neural_forecasting.py      # 30+ neural network models
│   ├── sentiment_adjusted_forecasting.py # Sentiment-adjusted predictions
│   └── foundation_models/         # TimeGPT, Chronos, Nixtla OSS
├── src/                          # Core application logic
│   ├── agents/                   # AI agents for analysis & planning
│   ├── reports/                  # Report generation system
│   ├── synthetic/                # Data generation & augmentation
│   └── uncertainty/              # Bayesian & probabilistic forecasting
├── scripts/                      # Utility scripts & monitoring
├── docs/                         # Detailed documentation
├── outputs/                      # Generated reports and files
└── quick_start.py                # Main demo & entry point
```

## 💻 Usage Examples

### Basic Forecasting
```python
from data.fred_client import FredDataClient
from models.forecasting_models import EconometricForecaster

# Fetch data and generate forecast
fred_client = FredDataClient(api_key='your_fred_key')
forecaster = EconometricForecaster()

gdp_data = fred_client.fetch_indicator('gdp', start_date='2010-01-01')
arima_result = forecaster.fit_arima(gdp_data)
forecast = forecaster.generate_forecast(arima_result['model_key'], periods=12)
```

### Neural Ensemble Forecasting
```python
from models.neural_forecasting import NeuralModelEnsemble

ensemble = NeuralModelEnsemble()
models = ensemble.fit_ensemble(gdp_data, target_column='value')
forecast = ensemble.predict(horizon=12, return_intervals=True)
```

### Sentiment-Adjusted Forecasting
```python
from models.sentiment_adjusted_forecasting import SentimentAdjustedForecaster

forecaster = SentimentAdjustedForecaster(sentiment_weight=0.15)
sentiment_data = forecaster.get_current_sentiment()
adjusted_result = forecaster.adjust_forecast(base_forecast, 'gdp', sentiment_data)
```

### Foundation Models
```python
from models.foundation_models.huggingface_forecaster import HybridFoundationEnsemble

# Automatic model selection across all tiers
ensemble = HybridFoundationEnsemble(auto_select=True)
result = ensemble.forecast(series, horizon=12)
print(f"Used model: {result['ensemble_info']['primary_model']}")
```

## 🔧 Configuration

### Environment Variables
```bash
# Required for real economic data
FRED_API_KEY=your_fred_api_key_here

# Optional: AI-powered analysis
OPENAI_API_KEY=your_openai_api_key_here

# Optional: Premium forecasting
NIXTLA_API_KEY=your_nixtla_api_key_here

# Optional: Monitoring
LANGCHAIN_API_KEY=your_langchain_api_key_here
```

### Command Line Options
```bash
# Complete analysis with demand planning
python quick_start.py --indicators gdp unemployment inflation

# Manufacturing industry focus
python quick_start.py --indicators gdp unemployment --industry manufacturing

# Extended forecast horizon
python quick_start.py --indicators gdp --forecast-horizon 18

# Economic forecasting only (no AI analysis)
python quick_start.py --indicators gdp unemployment --no-demand-planning
```

## 📚 Documentation

- **[🏗️ Architecture](docs/ARCHITECTURE.md)** - Technical implementation and model details
- **[💼 Business Applications](docs/BUSINESS_APPLICATIONS.md)** - Use cases and ROI examples  
- **[🤖 Foundation Models Guide](docs/FOUNDATION_MODELS.md)** - Complete guide to three-tier system
- **[📋 API Reference](docs/API_REFERENCE.md)** - Detailed function and class documentation
- **[🔍 LangSmith Integration](docs/LANGSMITH_INTEGRATION.md)** - AI monitoring setup guide

## 🎯 What Makes This Revolutionary

### World's First Three-Tier Foundation Model System
Unlike traditional platforms that force you to choose between expensive paid APIs or basic free models, we offer **graduated tiers** that scale with your needs:

- **🆓 Start Free**: Professional-grade Nixtla open source models
- **📈 Scale Up**: Add TimeGPT foundation model for production
- **🔄 Always Works**: Intelligent fallback ensures zero downtime

### Performance Comparison
| **Metric** | **Our Platform** | **Traditional Tools** |
|------------|------------------|----------------------|
| **Speed** | 20x faster (StatsForecast) | pmdarima baseline |
| **Models** | 30+ neural models | Limited selection |
| **Cost** | Free tier available | Mostly paid only |
| **Reliability** | Always works (fallbacks) | Fails without APIs |

## 🛠️ Requirements

- **Python**: 3.8+
- **Memory**: 2GB RAM minimum
- **Storage**: 500MB for dependencies
- **Network**: Required for real-time data (works offline for demos)

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🌟 Support

- **📧 Email**: [joshamorrison@gmail.com](mailto:joshamorrison@gmail.com)
- **💼 LinkedIn**: [linkedin.com/in/joshamorrison](https://www.linkedin.com/in/joshamorrison)
- **🐙 GitHub**: [github.com/joshamorrison](https://github.com/joshamorrison)

---

**⭐ Found this valuable? Star the repo and share with your network!**

*Revolutionary AI-powered forecasting platform with three-tier foundation models - the future of time series forecasting is here!* 🚀✨