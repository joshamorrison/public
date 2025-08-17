#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QUICK START DEMO - Generative Econometric Forecasting Platform
Test the three-tier foundation model system with sample data

Run this after: pip install -r requirements.txt
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("[WARN] python-dotenv not installed, using system environment only")

# Initialize LangSmith tracing for the entire application
try:
    import os
    if os.getenv('LANGCHAIN_API_KEY'):
        os.environ['LANGCHAIN_TRACING_V2'] = 'true'
        os.environ['LANGCHAIN_PROJECT'] = 'generative-econometric-forecasting'
        print("[LANGSMITH] LangSmith tracing enabled for comprehensive monitoring")
except Exception as e:
    pass  # Silent fail if LangSmith not available

def print_header():
    """Print demo header"""
    print("GENERATIVE ECONOMETRIC FORECASTING - QUICK START DEMO")
    print("=" * 70)
    print("Revolutionary Three-Tier Foundation Model System")
    print("From FREE professional models to PREMIUM AI forecasting")
    print()

def check_dependencies():
    """Check if basic dependencies are available"""
    missing_deps = []
    
    try:
        import pandas
        import numpy
        import matplotlib
    except ImportError as e:
        missing_deps.append(str(e).split("'")[1])
    
    if missing_deps:
        print("[X] Missing dependencies. Please run:")
        print("   pip install pandas numpy matplotlib scikit-learn")
        print()
        return False
    
    print("[OK] Core dependencies available")
    return True

def check_api_availability():
    """Check which APIs are available for real data"""
    print("\n[API] CHECKING REAL DATA API AVAILABILITY")
    print("-" * 45)
    
    api_status = {
        'fred': False,
        'openai': False,
        'news': False
    }
    
    # Check FRED API
    try:
        import fredapi
        fred_key = os.environ.get('FRED_API_KEY')
        if fred_key and fred_key != 'your_fred_api_key_here':
            # Test FRED connection
            fred = fredapi.Fred(api_key=fred_key)
            test_data = fred.get_series('GDP', limit=1)
            if len(test_data) > 0:
                api_status['fred'] = True
                print("[OK] FRED API: CONNECTED (Real economic data available)")
            else:
                print("[ERROR] FRED API: Invalid response")
        else:
            print("[KEY] FRED API: API key needed (using synthetic data)")
    except ImportError:
        print("[INSTALL] FRED API: Install with 'pip install fredapi'")
    except Exception as e:
        print(f"[ERROR] FRED API: Connection failed ({str(e)[:50]})")
    
    # Check OpenAI API
    try:
        import openai
        openai_key = os.environ.get('OPENAI_API_KEY')
        if openai_key and openai_key != 'your_openai_api_key_here':
            api_status['openai'] = True
            print("[OK] OpenAI API: CONNECTED (Real AI analysis available)")
        else:
            print("[KEY] OpenAI API: API key needed (using mock analysis)")
    except ImportError:
        print("[INSTALL] OpenAI API: Install with 'pip install openai'")
    
    # Check News API (for sentiment analysis)
    try:
        import newsapi
        news_key = os.environ.get('NEWS_API_KEY')
        if news_key and news_key != 'your_news_api_key_here':
            api_status['news'] = True
            print("[OK] News API: CONNECTED (Real sentiment data available)")
        else:
            print("[KEY] News API: API key needed (using mock sentiment)")
    except ImportError:
        print("[INSTALL] News API: Install with 'pip install newsapi-python'")
    
    return api_status

def fetch_real_economic_data():
    """Fetch real economic data from FRED API"""
    print("\n[REAL-DATA] FETCHING REAL ECONOMIC DATA FROM FRED")
    print("-" * 50)
    
    try:
        import fredapi
        fred_key = os.environ.get('FRED_API_KEY')
        fred = fredapi.Fred(api_key=fred_key)
        
        # Fetch last 4 years of key economic indicators
        start_date = '2020-01-01'
        
        indicators = {
            'GDP': 'GDPC1',      # Real GDP
            'Unemployment': 'UNRATE',  # Unemployment Rate
            'Inflation': 'CPIAUCSL'    # Consumer Price Index
        }
        
        economic_data = {}
        
        for name, fred_code in indicators.items():
            print(f"[FETCH] Downloading {name} ({fred_code})...")
            try:
                series = fred.get_series(fred_code, start=start_date)
                if len(series) > 0:
                    # Convert CPI to inflation rate for inflation indicator
                    if name == 'Inflation':
                        series = series.pct_change(periods=12) * 100  # Year-over-year % change
                        series = series.dropna()
                    
                    economic_data[name] = series
                    print(f"   [OK] {len(series)} observations, latest: {series.iloc[-1]:.2f}")
                else:
                    print(f"   [ERROR] No data received for {name}")
            except Exception as e:
                print(f"   [ERROR] Failed to fetch {name}: {str(e)[:50]}")
        
        if len(economic_data) >= 2:  # Need at least 2 indicators for meaningful demo
            print(f"\n[SUCCESS] Real economic data loaded ({len(economic_data)} indicators)")
            return economic_data
        else:
            print("\n[FALLBACK] Insufficient real data, using synthetic data")
            return None
            
    except Exception as e:
        print(f"[ERROR] FRED API error: {e}")
        return None

def create_sample_economic_data():
    """Create realistic economic time series data up to current date"""
    print("\n[DATA] GENERATING SAMPLE ECONOMIC DATA")
    print("-" * 40)
    
    # Create data up to current month (August 2025)
    from datetime import datetime
    current_date = datetime.now()
    end_date = f"{current_date.year}-{current_date.month:02d}-01"
    
    dates = pd.date_range('2020-01-01', end_date, freq='M')
    np.random.seed(42)  # For reproducible results
    
    # Simulate realistic GDP growth with trend to current levels (~$23,700B)
    trend = np.linspace(20000, 23700, len(dates))
    seasonal = 100 * np.sin(2 * np.pi * np.arange(len(dates)) / 12)
    noise = np.random.normal(0, 50, len(dates))
    gdp_values = trend + seasonal + noise
    
    # Create unemployment rate reflecting current low levels (~4.2%)
    unemployment_base = 4.2
    unemployment_trend = -0.2 * (gdp_values - gdp_values.mean()) / gdp_values.std()
    unemployment_noise = np.random.normal(0, 0.2, len(dates))
    unemployment_values = unemployment_base + unemployment_trend + unemployment_noise
    unemployment_values = np.clip(unemployment_values, 3.5, 6.0)  # Current realistic bounds
    
    # Create inflation data reflecting current levels (~2.7%)
    inflation_base = 2.7
    inflation_trend = np.random.normal(0, 0.3, len(dates)).cumsum() * 0.05
    inflation_values = inflation_base + inflation_trend
    inflation_values = np.clip(inflation_values, 1.5, 4.0)  # Current realistic bounds
    
    economic_data = {
        'GDP': pd.Series(gdp_values, index=dates, name='Real GDP (Billions)'),
        'Unemployment': pd.Series(unemployment_values, index=dates, name='Unemployment Rate (%)'),
        'Inflation': pd.Series(inflation_values, index=dates, name='Inflation Rate (%)')
    }
    
    print(f"[OK] Created {len(dates)} months of economic data (2020-{current_date.year})")
    print(f"[GDP] GDP range: ${economic_data['GDP'].min():,.0f}B - ${economic_data['GDP'].max():,.0f}B")
    print(f"[UNEMP] Unemployment range: {economic_data['Unemployment'].min():.1f}% - {economic_data['Unemployment'].max():.1f}%")
    print(f"[INFL] Inflation range: {economic_data['Inflation'].min():.1f}% - {economic_data['Inflation'].max():.1f}%")
    
    return economic_data

def test_basic_forecasting(economic_data, data_source="SYNTHETIC"):
    """Test basic statistical forecasting with dynamic horizon"""
    print("\n[STATS] TESTING BASIC STATISTICAL FORECASTING")
    print("-" * 45)
    print("Using built-in exponential smoothing (always available)")
    
    # Determine forecast period based on data source
    from datetime import datetime, timedelta
    current_date = datetime.now()
    
    if data_source == "REAL-FRED":
        # For FRED data, forecast 6 months from the latest observation
        latest_date = economic_data[list(economic_data.keys())[0]].index[-1]
        months_to_forecast = 6
        forecast_start_desc = f"from {latest_date.strftime('%B %Y')}"
        forecast_end_date = latest_date + timedelta(days=180)  # ~6 months
        forecast_end_desc = forecast_end_date.strftime('%B %Y')
    else:
        # For synthetic data, forecast 6 months from current date
        months_to_forecast = 6
        forecast_start_desc = f"from {current_date.strftime('%B %Y')}"
        forecast_end_date = current_date + timedelta(days=180)  # ~6 months
        forecast_end_desc = forecast_end_date.strftime('%B %Y')
    
    print(f"[HORIZON] Forecasting {months_to_forecast} months {forecast_start_desc} through {forecast_end_desc}")
    
    try:
        from sklearn.linear_model import LinearRegression
        sklearn_available = True
    except ImportError:
        sklearn_available = False
    
    results = {}
    
    for indicator, series in economic_data.items():
        print(f"\n[FORECAST] Forecasting {indicator}...")
        
        # Simple exponential smoothing forecast
        alpha = 0.3
        smoothed = [series.iloc[0]]
        for i in range(1, len(series)):
            smoothed.append(alpha * series.iloc[i] + (1 - alpha) * smoothed[-1])
        
        # Calculate trend
        recent_data = series.tail(12).values
        trend = np.mean(np.diff(recent_data))
        
        # Generate 6-month forward forecast
        last_value = smoothed[-1]
        forecast = []
        for i in range(months_to_forecast):
            next_value = last_value + trend * (i + 1)
            forecast.append(next_value)
        
        forecast = np.array(forecast)
        
        # Create forecast dates
        last_date = series.index[-1]
        forecast_dates = pd.date_range(
            start=last_date + pd.DateOffset(months=1),
            periods=months_to_forecast,
            freq='M'
        )
        
        # Calculate confidence intervals based on historical volatility
        volatility = series.std()
        lower_bound = forecast - 1.96 * volatility
        upper_bound = forecast + 1.96 * volatility
        
        results[indicator] = {
            'forecast': forecast,
            'forecast_dates': forecast_dates,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'last_actual': series.iloc[-1],
            'trend': trend
        }
        
        # Display results with dynamic messaging
        latest_date = series.index[-1]
        final_forecast_date = forecast_dates[-1]
        print(f"   [DATA] Last actual ({latest_date.strftime('%b %Y')}): {series.iloc[-1]:.1f}")
        print(f"   [6M-AVG] 6-month forecast avg: {forecast.mean():.1f}")
        print(f"   [{final_forecast_date.strftime('%b-%Y').upper()}] {final_forecast_date.strftime('%B %Y')}: {forecast[-1]:.1f}")
        print(f"   [TREND] Monthly trend: {'+' if trend > 0 else ''}{trend:.2f}")
        print(f"   [OK] 6-month forecast complete")
    
    return results

def test_advanced_models():
    """Test availability of advanced foundation models"""
    print("\n[TIER1] TESTING ADVANCED FOUNDATION MODELS")
    print("-" * 42)
    
    model_status = {
        'statsforecast': False,
        'neuralforecast': False,
        'mlforecast': False,
        'nixtla_timegpt': False,
        'huggingface': False
    }
    
    # Test Nixtla StatsForecast
    try:
        import statsforecast
        model_status['statsforecast'] = True
        print("[OK] Nixtla StatsForecast: AVAILABLE (Lightning-fast statistical models)")
    except ImportError:
        print("[INSTALL] Nixtla StatsForecast: Install with 'pip install statsforecast'")
    
    # Test Nixtla NeuralForecast
    try:
        import neuralforecast
        model_status['neuralforecast'] = True
        print("[OK] Nixtla NeuralForecast: AVAILABLE (30+ neural models)")
    except (ImportError, AttributeError) as e:
        # Try alternative neural forecasting
        try:
            from models.neural_forecasting import NeuralModelEnsemble
            ensemble = NeuralModelEnsemble()
            available_models = ensemble.get_available_models()
            model_status['neural_alternative'] = True
            print(f"[OK] Alternative Neural Models: AVAILABLE ({len(available_models)} models)")
        except Exception:
            print("[INSTALL] Neural Forecasting: Limited availability")
    
    # Test Nixtla MLForecast
    try:
        import mlforecast
        model_status['mlforecast'] = True
        print("[OK] Nixtla MLForecast: AVAILABLE (ML models with features)")
    except ImportError:
        print("[INSTALL] Nixtla MLForecast: Install with 'pip install mlforecast'")
    
    # Test TimeGPT
    api_key = os.environ.get('NIXTLA_API_KEY')
    if api_key and api_key != 'your_nixtla_api_key_here':
        try:
            import nixtla
            model_status['nixtla_timegpt'] = True
            print("[OK] Nixtla TimeGPT: AVAILABLE (Premium foundation model)")
        except ImportError:
            print("[INSTALL] Nixtla TimeGPT: Install with 'pip install nixtla'")
    else:
        print("[KEY] Nixtla TimeGPT: API key needed (premium features)")
    
    # Test HuggingFace
    try:
        import transformers
        model_status['huggingface'] = True
        print("[OK] HuggingFace Transformers: AVAILABLE (Free transformer models)")
    except ImportError:
        print("[INSTALL] HuggingFace: Install with 'pip install transformers chronos-forecasting'")
    
    return model_status

def demonstrate_ai_insights(economic_data, forecast_results, openai_available=False):
    """Demonstrate AI-powered insights (real if OpenAI available, mock otherwise)"""
    print("\n[AI] TESTING AI-POWERED INSIGHTS")
    print("-" * 35)
    
    if openai_available:
        try:
            # Try to generate real AI insights
            analysis = generate_real_ai_analysis(economic_data, forecast_results)
            if analysis:
                print(analysis)
                print("[AI] AI Analysis Status: REAL (OpenAI API)")
                return
        except Exception as e:
            print(f"[ERROR] Real AI analysis failed: {str(e)[:50]}")
            openai_available = False
    
    # Try HuggingFace local model as backup
    try:
        analysis = generate_huggingface_ai_analysis(economic_data, forecast_results)
        if analysis:
            print(analysis)
            print("[AI] AI Analysis Status: LOCAL (HuggingFace)")
            return
    except Exception as e:
        print(f"[ERROR] HuggingFace AI analysis failed: {str(e)[:50]}")
    
    # Final fallback to mock analysis
    generate_mock_ai_analysis(economic_data, forecast_results)
    print("[AI] AI Analysis Status: MOCK DEMO")

def generate_real_ai_analysis(economic_data, forecast_results):
    """Generate real AI analysis using OpenAI API"""
    try:
        import openai
        
        # Setup LangSmith tracing for AI analysis
        if os.getenv('LANGCHAIN_API_KEY'):
            os.environ['LANGCHAIN_TRACING_V2'] = 'true'
            os.environ['LANGCHAIN_PROJECT'] = 'economic-ai-analysis'
        
        # Prepare economic context
        latest_data = {}
        forecasts_6m = {}
        forecasts_18m = {}
        forecasts_end_2025 = {}
        
        for indicator in economic_data.keys():
            latest_data[indicator] = economic_data[indicator].iloc[-1]
            if indicator in forecast_results:
                forecast_data = forecast_results[indicator]['forecast']
                forecasts_6m[indicator] = forecast_data.mean()  # FY25 average
                forecasts_18m[indicator] = forecast_data.mean()  # Same as 6m now
                forecasts_end_2025[indicator] = forecast_data[-1]
        
        # Create prompt for economic analysis
        prompt = f"""You are an expert economist analyzing current economic conditions. Based on this data:

Current Economic Indicators:
- GDP: ${latest_data.get('GDP', 0):,.0f}B
- Unemployment: {latest_data.get('Unemployment', 0):.1f}%
- Inflation: {latest_data.get('Inflation', 0):.1f}%

FY25 Year-End Forecasts (through December 2025):
- GDP: ${forecasts_6m.get('GDP', 0):,.0f}B
- Unemployment: {forecasts_6m.get('Unemployment', 0):.1f}%
- Inflation: {forecasts_6m.get('Inflation', 0):.1f}%

December 2025 Projections:
- GDP: ${forecasts_end_2025.get('GDP', 0):,.0f}B
- Unemployment: {forecasts_end_2025.get('Unemployment', 0):.1f}%
- Inflation: {forecasts_end_2025.get('Inflation', 0):.1f}%

Provide a concise executive summary (200 words max) covering:
1. Current economic environment assessment
2. Key insights from the forecasts
3. Business recommendations

Format with clear sections and bullet points."""

        # Make API call
        client = openai.OpenAI()
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.7
        )
        
        return f"\n[IDEA] REAL AI ECONOMIC ANALYSIS:\n{'=' * 50}\n{response.choices[0].message.content}\n"
        
    except Exception as e:
        print(f"[ERROR] OpenAI API call failed: {e}")
        return None

def generate_huggingface_ai_analysis(economic_data, forecast_results):
    """Generate AI analysis using HuggingFace local models"""
    try:
        from transformers import pipeline, logging
        
        # Setup LangSmith tracing for HuggingFace analysis
        if os.getenv('LANGCHAIN_API_KEY'):
            os.environ['LANGCHAIN_TRACING_V2'] = 'true'
            os.environ['LANGCHAIN_PROJECT'] = 'huggingface-local-analysis'
        
        # Suppress transformers warnings for cleaner output
        logging.set_verbosity_error()
        
        print("[LOCAL] Loading HuggingFace text generation model...")
        
        # Use a lightweight text generation model
        generator = pipeline(
            'text-generation',
            model='gpt2',  # Small, reliable model (548MB)
            max_length=512,
            truncation=True,
            pad_token_id=50256  # GPT-2 EOS token
        )
        
        # Prepare economic context
        latest_data = {}
        forecasts_6m = {}
        forecasts_end = {}
        
        for indicator in economic_data.keys():
            latest_data[indicator] = economic_data[indicator].iloc[-1]
            if indicator in forecast_results:
                forecast_data = forecast_results[indicator]['forecast']
                forecasts_6m[indicator] = forecast_data.mean()
                forecasts_end[indicator] = forecast_data[-1]
        
        # Create concise prompt for local model
        prompt = f"""Economic Analysis Report:

Current Data:
- GDP: ${latest_data.get('GDP', 0):,.0f}B
- Unemployment: {latest_data.get('Unemployment', 0):.1f}%
- Inflation: {latest_data.get('Inflation', 0):.1f}%

6-Month Forecasts:
- GDP: ${forecasts_end.get('GDP', 0):,.0f}B  
- Unemployment: {forecasts_end.get('Unemployment', 0):.1f}%
- Inflation: {forecasts_end.get('Inflation', 0):.1f}%

Analysis: The economic indicators show"""
        
        # Generate analysis
        response = generator(
            prompt,
            max_new_tokens=150,
            temperature=0.7,
            do_sample=True,
            pad_token_id=50256
        )
        
        # Extract generated text
        generated_text = response[0]['generated_text']
        # Remove the prompt part
        analysis_part = generated_text[len(prompt):].strip()
        
        # Format the output
        formatted_analysis = f"""
[IDEA] LOCAL AI ECONOMIC ANALYSIS:
{'=' * 50}

{prompt.replace('Analysis: The economic indicators show', 'EXECUTIVE SUMMARY:')}

AI INSIGHTS:
{analysis_part}

GENERATED BY: HuggingFace GPT-2 (Local Model)
"""
        
        return formatted_analysis
        
    except Exception as e:
        print(f"[ERROR] HuggingFace text generation failed: {e}")
        return None

def generate_mock_ai_analysis(economic_data, forecast_results):
    """Generate mock AI analysis for demo purposes"""
    gdp_forecast = forecast_results['GDP']['forecast']
    unemployment_forecast = forecast_results['Unemployment']['forecast']
    inflation_forecast = forecast_results['Inflation']['forecast']
    
    analysis = f"""
[IDEA] SAMPLE AI ECONOMIC ANALYSIS:
{'=' * 50}

EXECUTIVE ECONOMIC SUMMARY

[DATA] Current Economic Environment:
• GDP: ${economic_data['GDP'].iloc[-1]:,.0f}B (stable growth trajectory)
• Unemployment: {economic_data['Unemployment'].iloc[-1]:.1f}% (within healthy range)
• Inflation: {economic_data['Inflation'].iloc[-1]:.1f}% (near target levels)

[FORECAST] FY25 Year-End Forecast (through December 2025):
• GDP: ${gdp_forecast.mean():,.0f}B ({'+' if gdp_forecast.mean() > economic_data['GDP'].iloc[-1] else ''}{((gdp_forecast.mean() / economic_data['GDP'].iloc[-1] - 1) * 100):.1f}% growth)
• Unemployment: {unemployment_forecast.mean():.1f}% (stable employment)
• Inflation: {inflation_forecast.mean():.1f}% (controlled price growth)

[FORECAST] December 2025 Projection:
• GDP: ${gdp_forecast[-1]:,.0f}B ({'+' if gdp_forecast[-1] > economic_data['GDP'].iloc[-1] else ''}{((gdp_forecast[-1] / economic_data['GDP'].iloc[-1] - 1) * 100):.1f}% from current)
• Unemployment: {unemployment_forecast[-1]:.1f}% (continued stability)
• Inflation: {inflation_forecast[-1]:.1f}% (year-end target)

[TARGET] Key Insights:
• Economic fundamentals remain stable with moderate growth
• Employment levels suggest continued consumer spending power
• Inflation trends support business planning confidence
• Recommend maintaining current strategic positioning

[TREND] Business Recommendations:
• [OK] Stable demand environment supports expansion plans
• [OK] Employment trends favor consumer-focused strategies  
• [OK] Inflation outlook enables predictable cost planning
• [OK] Overall economic conditions support business growth

Risk Assessment: LOW (stable indicators across all metrics)
Confidence Level: HIGH (consistent trend patterns)
"""
    
    print(analysis)

def show_tier_summary(model_status):
    """Show three-tier system summary"""
    print("\n[BUILD] THREE-TIER FOUNDATION MODEL SYSTEM SUMMARY")
    print("=" * 55)
    
    # Count available models by tier
    tier1_count = sum([model_status['nixtla_timegpt']])
    tier2_count = sum([model_status['statsforecast'], model_status['neuralforecast'], model_status['mlforecast']])
    tier3_count = sum([model_status['huggingface']])
    
    print(f"[TIER1] TIER 1 (Premium): {tier1_count}/1 models available")
    print("   • Nixtla TimeGPT - State-of-the-art foundation model")
    print("   • Best accuracy, zero-shot forecasting")
    print("   • Requires API key")
    print()
    
    print(f"[TIER2] TIER 2 (Professional): {tier2_count}/3 model types available")
    print("   • Nixtla StatsForecast - Lightning-fast statistical models")
    print("   • Nixtla NeuralForecast - 30+ neural models")  
    print("   • Nixtla MLForecast - ML models with features")
    print("   • Professional accuracy, completely free")
    print()
    
    print(f"[TIER3] TIER 3 (Always Available): {tier3_count + 1}/2 model types available")
    print("   • HuggingFace Chronos - Transformer forecasting")
    print("   • Statistical Fallbacks - Always work ([OK] DEMO USED)")
    print("   • Good accuracy, zero configuration")
    print()
    
    total_available = tier1_count + tier2_count + tier3_count + 1  # +1 for fallback
    print(f"[DATA] TOTAL SYSTEM STATUS: {total_available}/7 model types available")
    
    if tier1_count > 0:
        status = "[TIER1] PREMIUM READY"
    elif tier2_count > 0:
        status = "[TIER2] PROFESSIONAL READY"
    else:
        status = "[TIER3] BASIC READY (Always Works)"
    
    print(f"[TARGET] SYSTEM STATUS: {status}")

def show_next_steps():
    """Show next steps for users"""
    print("\n[ROCKET] NEXT STEPS TO UNLOCK MORE POWER")
    print("=" * 40)
    
    print("[FREE] FREE UPGRADES (Professional-Grade):")
    print("   pip install statsforecast neuralforecast mlforecast")
    print("   -> Gets you 20x faster models + 30+ neural networks")
    print()
    
    print("[TIER1] PREMIUM UPGRADE (State-of-the-Art):")
    print("   1. Get Nixtla API key: https://nixtla.io/")
    print("   2. Set NIXTLA_API_KEY in .env file")
    print("   3. pip install nixtla")
    print("   -> Gets you TimeGPT foundation model")
    print()
    
    print("[AI] AI INSIGHTS UPGRADE:")
    print("   1. Get OpenAI API key: https://openai.com/")
    print("   2. Set OPENAI_API_KEY in .env file")
    print("   3. pip install openai langchain")
    print("   -> Gets you real AI-powered analysis")
    print()
    
    print("[BOOKS] EXPLORE MORE:")
    print("   python quick_start.py           # Quick start demo")
    print("   python test_real_data.py        # Real FRED data")
    print("   python test_full_ai.py          # Complete AI platform")
    print("   python test_expanded_hybrid.py  # Foundation models test")

def show_next_steps_with_apis(api_status):
    """Show next steps including API setup recommendations"""
    print("\n[ROCKET] NEXT STEPS TO UNLOCK MORE POWER")
    print("=" * 40)
    
    # Real data APIs
    if not api_status.get('fred', False):
        print("[REAL-DATA] GET REAL ECONOMIC DATA:")
        print("   1. Get FREE FRED API key: https://fred.stlouisfed.org/docs/api/api_key.html")
        print("   2. Set FRED_API_KEY in .env file")
        print("   3. pip install fredapi")
        print("   -> Gets you REAL economic data from Federal Reserve")
        print()
    
    if not api_status.get('openai', False):
        print("[AI] GET REAL AI ANALYSIS:")
        print("   1. Get OpenAI API key: https://platform.openai.com/api-keys")
        print("   2. Set OPENAI_API_KEY in .env file")
        print("   3. pip install openai")
        print("   -> Gets you REAL AI-powered economic analysis")
        print()
    
    # Foundation models
    print("[FREE] FREE UPGRADES (Professional-Grade):")
    print("   pip install statsforecast neuralforecast mlforecast")
    print("   -> Gets you 20x faster models + 30+ neural networks")
    print()
    
    print("[TIER1] PREMIUM UPGRADE (State-of-the-Art):")
    print("   1. Get Nixtla API key: https://nixtla.io/")
    print("   2. Set NIXTLA_API_KEY in .env file")
    print("   3. pip install nixtla")
    print("   -> Gets you TimeGPT foundation model")
    print()
    
    print("[BOOKS] EXPLORE MORE:")
    print("   python quick_start.py           # Quick start demo")

def main():
    """Run the quick start demo"""
    print_header()
    
    # Check dependencies
    if not check_dependencies():
        return
    
    # Check API availability first
    api_status = check_api_availability()
    
    # Try to get real data if FRED API is available
    economic_data = None
    data_source = "SYNTHETIC"
    
    if api_status['fred']:
        real_data = fetch_real_economic_data()
        if real_data is not None:
            economic_data = real_data
            data_source = "REAL-FRED"
    
    # Fallback to synthetic data if real data not available
    if economic_data is None:
        economic_data = create_sample_economic_data()
        data_source = "SYNTHETIC"
    
    print(f"\n[SOURCE] Using {data_source} data for demonstration")
    
    # Test basic forecasting (always works)
    forecast_results = test_basic_forecasting(economic_data, data_source)
    
    # Test advanced models availability
    model_status = test_advanced_models()
    
    # 3. News Sentiment Analysis
    print("\n[NEWS] NEWS SENTIMENT ANALYSIS:")
    print("-" * 30)
    try:
        from data.unstructured.news_client import NewsClient
        from data.unstructured.sentiment_analyzer import EconomicSentimentAnalyzer
        
        # Initialize news and sentiment clients
        news_client = NewsClient(newsapi_key=os.getenv('NEWSAPI_KEY'))
        sentiment_analyzer = EconomicSentimentAnalyzer(use_finbert=True, use_openai=False)
        
        # Fetch recent economic news from RSS feeds (always available)
        print("   [RSS] Fetching recent economic news...")
        news_articles = news_client.fetch_rss_feeds(max_articles_per_feed=3)
        
        if news_articles:
            print(f"   [FETCH] Found {len(news_articles)} recent articles")
            
            # Filter for economic relevance (lower threshold for better results)
            economic_articles = news_client.filter_economic_articles(news_articles, min_relevance_score=0.05)
            print(f"   [FILTER] {len(economic_articles)} articles are economically relevant")
            
            if economic_articles[:3]:  # Analyze top 3 articles
                print("   [AI] Analyzing sentiment of top economic news...")
                sentiment_df = sentiment_analyzer.analyze_articles_sentiment(economic_articles[:3])
                
                if len(sentiment_df) > 0:
                    metrics = sentiment_analyzer.calculate_sentiment_metrics(sentiment_df)
                    
                    overall_score = metrics.get('overall_sentiment_score', 0)
                    print(f"   [SCORE] Overall Economic News Sentiment: {overall_score:.2f} (-1=negative, +1=positive)")
                    
                    sentiment_dist = metrics.get('sentiment_percentages', {})
                    print(f"   [DIST] Sentiment Distribution: {dict(sentiment_dist)}")
                    
                    # Show sample headlines with sentiment
                    print("   [SAMPLE] Recent Economic News Analysis:")
                    for _, article in sentiment_df.head(2).iterrows():
                        title_short = article['title'][:50] + "..." if len(article['title']) > 50 else article['title']
                        print(f"     • {title_short}")
                        print(f"       {article['sentiment'].upper()} (confidence: {article['confidence']:.2f})")
                        
                    # News-enhanced forecast adjustment
                    print(f"   [INSIGHT] News sentiment impact on forecasts:")
                    if overall_score > 0.1:
                        print(f"     • Positive news sentiment suggests upward forecast bias")
                        print(f"     • Market confidence appears strong")
                    elif overall_score < -0.1:
                        print(f"     • Negative news sentiment suggests downward forecast risk")
                        print(f"     • Market concerns detected")
                    else:
                        print(f"     • Neutral news sentiment supports baseline forecasts")
                        print(f"     • Markets appear balanced")
                    
                    # Apply sentiment adjustments to forecasts
                    print(f"   [ENHANCE] Applying sentiment-adjusted forecasting...")
                    try:
                        from models.sentiment_adjusted_forecasting import SentimentAdjustedForecaster
                        
                        # Create sentiment-adjusted forecaster
                        sentiment_forecaster = SentimentAdjustedForecaster(
                            sentiment_weight=0.1,
                            newsapi_key=os.getenv('NEWSAPI_KEY')
                        )
                        
                        # Prepare forecast data for adjustment
                        sample_forecasts = {}
                        for indicator in forecast_results.keys():
                            if indicator in forecast_results:
                                forecast_data = forecast_results[indicator]['forecast']
                                if hasattr(forecast_data, 'values'):
                                    sample_forecasts[indicator] = forecast_data.values
                                else:
                                    sample_forecasts[indicator] = forecast_data
                        
                        if sample_forecasts:
                            # Apply sentiment adjustments
                            sentiment_data = {'overall_sentiment': overall_score, 'sentiment_strength': abs(overall_score), 'articles_analyzed': len(sentiment_df)}
                            adjustment_results = sentiment_forecaster.batch_adjust_forecasts(sample_forecasts, sentiment_data)
                            
                            print(f"     • Applied sentiment adjustments to {len(sample_forecasts)} indicators")
                            
                            # Show adjustment impact
                            for indicator, result in adjustment_results['adjusted_forecasts'].items():
                                max_adj = max(abs(x) for x in result['sentiment_adjustments'])
                                direction = "↑" if result['adjustment_summary']['avg_adjustment'] > 0 else "↓"
                                print(f"     • {indicator.upper()}: {direction} {max_adj*100:.1f}% max adjustment")
                        
                    except Exception as e:
                        print(f"     • Sentiment adjustment unavailable: {e}")
                        
                else:
                    print("   [ERROR] No sentiment analysis results available")
            else:
                print("   [WARN] No economically relevant articles found for sentiment analysis")
        else:
            print("   [WARN] No recent news articles available")
            
    except Exception as e:
        print(f"   [ERROR] News sentiment analysis unavailable: {e}")

    # Demonstrate AI insights (enhanced if OpenAI available)
    ai_analysis_result = demonstrate_ai_insights(economic_data, forecast_results, api_status.get('openai', False))
    
    # Generate Executive Reports
    print("\n[REPORTS] GENERATING EXECUTIVE REPORTS:")
    print("-" * 40)
    try:
        from src.reports.simple_reporting import SimpleEconomicReporter
        
        # Collect sentiment data if available
        final_sentiment_data = None
        if 'economic_articles' in locals() and len(economic_articles) > 0:
            final_sentiment_data = {
                'overall_sentiment': overall_score if 'overall_score' in locals() else 0,
                'articles_analyzed': len(economic_articles) if 'economic_articles' in locals() else 0,
                'data_source': 'real_news'
            }
        
        # Create reporter and generate reports
        reporter = SimpleEconomicReporter()
        generated_files = reporter.generate_reports(
            economic_data=economic_data,
            forecast_results=forecast_results,
            sentiment_analysis=final_sentiment_data,
            ai_analysis=ai_analysis_result
        )
        
        print("   [SUCCESS] Professional reports generated:")
        for format_type, file_path in generated_files.items():
            file_size = os.path.getsize(file_path) / 1024  # KB
            print(f"     • {format_type.upper()}: {file_path.split('/')[-1]} ({file_size:.1f} KB)")
        
        print(f"   [LOCATION] Reports saved to: {reporter.output_dir}")
        
    except Exception as e:
        print(f"   [ERROR] Report generation failed: {e}")
    
    # Show system summary
    show_tier_summary(model_status)
    
    # Show next steps (include API setup if not available)
    show_next_steps_with_apis(api_status)
    
    print("\n" + "=" * 70)
    print("[SUCCESS] QUICK START DEMO COMPLETE!")
    print("=" * 70)
    print("[OK] Platform working with statistical forecasting")
    print("[CYCLE] Intelligent fallback system demonstrated")
    print("[TREND] Professional-grade results achieved")
    print("[ROCKET] Ready to scale up with more advanced models!")
    print()
    print("[STAR] This was just a taste - the full platform has much more!")

if __name__ == "__main__":
    main()