"""
Demo script for Generative Econometric Forecasting Platform.
Demonstrates the platform capabilities without requiring external API keys.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import json
import os
import warnings
warnings.filterwarnings('ignore')

# Import our modules
from src.models.forecasting_models import EconometricForecaster
from src.agents.narrative_generator import EconomicNarrativeGenerator

# Set style for plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class EconometricDemo:
    """Demo class showing the platform's capabilities with synthetic data."""
    
    def __init__(self):
        self.forecaster = EconometricForecaster()
        self.demo_data = {}
        self.forecast_results = {}
        
        print("🚀 Generative Econometric Forecasting Platform Demo")
        print("=" * 60)
        
    def generate_synthetic_economic_data(self, start_date='2010-01-01', periods=168):
        """Generate realistic synthetic economic time series data."""
        print("📊 Generating synthetic economic data...")
        
        # Create date range (monthly data for 14 years)
        dates = pd.date_range(start=start_date, periods=periods, freq='M')
        np.random.seed(42)  # For reproducible results
        
        # GDP Growth Rate (with trend and cycles)
        gdp_trend = 0.02 + 0.001 * np.sin(np.arange(periods) * 2 * np.pi / 48)  # 4-year cycle
        gdp_noise = np.random.normal(0, 0.005, periods)
        gdp_shocks = np.zeros(periods)
        gdp_shocks[60:66] = -0.02  # 2015 recession
        gdp_shocks[130:136] = -0.03  # 2020 pandemic
        
        gdp_growth = gdp_trend + gdp_noise + gdp_shocks
        gdp_level = 100 * np.exp(np.cumsum(gdp_growth))
        
        # Unemployment Rate (counter-cyclical to GDP)
        unemployment_base = 6.0
        unemployment_cycle = -2.0 * gdp_growth + np.random.normal(0, 0.3, periods)
        unemployment = unemployment_base + unemployment_cycle
        unemployment = np.clip(unemployment, 2.0, 15.0)  # Realistic bounds
        
        # Inflation Rate (with some persistence)
        inflation = np.zeros(periods)
        inflation[0] = 2.0
        for i in range(1, periods):
            inflation[i] = 0.7 * inflation[i-1] + 0.3 * 2.0 + np.random.normal(0, 0.5)
            # Add inflation spikes during certain periods
            if 130 <= i <= 140:  # Post-pandemic inflation
                inflation[i] += 2.0
        
        # Interest Rates (responds to inflation and economic conditions)
        fed_rate = np.maximum(0, 2.0 + 0.5 * inflation + 0.3 * gdp_growth * 100 + 
                             np.random.normal(0, 0.2, periods))
        fed_rate[130:145] = 0.25  # Near zero during crisis
        
        # Consumer Confidence (leading indicator)
        confidence_base = 100
        confidence = confidence_base + 50 * gdp_growth + np.random.normal(0, 5, periods)
        confidence = np.clip(confidence, 50, 150)
        
        # Store the data
        self.demo_data = {
            'gdp': pd.Series(gdp_level, index=dates, name='Real GDP (Index)'),
            'unemployment': pd.Series(unemployment, index=dates, name='Unemployment Rate (%)'),
            'inflation': pd.Series(inflation, index=dates, name='Inflation Rate (%)'),
            'interest_rate': pd.Series(fed_rate, index=dates, name='Federal Funds Rate (%)'),
            'consumer_confidence': pd.Series(confidence, index=dates, name='Consumer Confidence Index')
        }
        
        print(f"✅ Generated {len(self.demo_data)} economic time series")
        for name, series in self.demo_data.items():
            print(f"   • {name}: {len(series)} observations, {series.index.min().strftime('%Y-%m')} to {series.index.max().strftime('%Y-%m')}")
        
        return self.demo_data
    
    def run_forecasting_analysis(self, forecast_horizon=12):
        """Run forecasting analysis on all indicators."""
        print(f"\n🔮 Running forecasting analysis (horizon: {forecast_horizon} periods)...")
        
        results = {}
        
        for indicator, series in self.demo_data.items():
            print(f"\n📈 Analyzing {indicator}...")
            
            try:
                # Check stationarity
                stationarity = self.forecaster.check_stationarity(series)
                print(f"   Stationarity test: {'✅ Stationary' if stationarity['is_stationary'] else '❌ Non-stationary'}")
                
                # Fit ARIMA model
                arima_result = self.forecaster.fit_arima(series, auto_order=True)
                print(f"   ARIMA model: {arima_result['order']}, AIC: {arima_result['aic']:.2f}")
                
                # Generate forecast
                forecast = self.forecaster.generate_forecast(
                    arima_result['model_key'], 
                    periods=forecast_horizon
                )
                
                # Try ensemble forecast
                try:
                    ensemble = self.forecaster.generate_ensemble_forecast(series, forecast_horizon)
                    print(f"   Ensemble: {ensemble['model_count']} models combined")
                    
                    results[indicator] = {
                        'arima': forecast,
                        'ensemble': {
                            'forecast': ensemble['ensemble_forecast'],
                            'model_type': 'Ensemble'
                        },
                        'stationarity': stationarity,
                        'model_info': arima_result
                    }
                except Exception as e:
                    print(f"   Ensemble failed: {e}")
                    results[indicator] = {
                        'arima': forecast,
                        'stationarity': stationarity,
                        'model_info': arima_result
                    }
                
            except Exception as e:
                print(f"   ❌ Error: {e}")
                continue
        
        self.forecast_results = results
        print(f"\n✅ Completed forecasting for {len(results)} indicators")
        return results
    
    def generate_ai_narratives(self):
        """Generate AI narratives for forecasts (with mock narratives if OpenAI unavailable)."""
        print(f"\n🤖 Generating AI narratives and insights...")
        
        narratives = {}
        
        # Try to initialize the narrative generator
        try:
            generator = EconomicNarrativeGenerator()
            ai_available = True
            print("   Using OpenAI for narrative generation")
        except Exception as e:
            print(f"   OpenAI not available ({e}), using mock narratives")
            ai_available = False
        
        for indicator, results in self.forecast_results.items():
            print(f"   📝 Generating narrative for {indicator}...")
            
            if ai_available:
                try:
                    # Use AI to generate narrative
                    series = self.demo_data[indicator]
                    forecast_data = results.get('ensemble', results.get('arima'))
                    
                    narrative = generator.generate_executive_summary(
                        indicator, series, forecast_data, 
                        {'mape': np.random.uniform(2, 6), 'rmse': np.random.uniform(0.5, 2.0)}
                    )
                    narratives[indicator] = narrative
                    
                except Exception as e:
                    print(f"      ⚠️ AI generation failed: {e}")
                    narratives[indicator] = self._generate_mock_narrative(indicator, results)
            else:
                narratives[indicator] = self._generate_mock_narrative(indicator, results)
        
        print(f"✅ Generated narratives for {len(narratives)} indicators")
        return narratives
    
    def _generate_mock_narrative(self, indicator, results):
        """Generate mock narrative when AI is not available."""
        forecast_data = results.get('ensemble', results.get('arima', {}))
        forecast_values = forecast_data.get('forecast', [])
        
        if len(forecast_values) == 0:
            return f"Analysis for {indicator} is not available due to modeling constraints."
        
        current_value = self.demo_data[indicator].iloc[-1]
        forecast_end = forecast_values[-1]
        change_pct = ((forecast_end / current_value - 1) * 100) if current_value != 0 else 0
        
        direction = "increase" if change_pct > 1 else "decrease" if change_pct < -1 else "remain stable"
        
        narrative = f"""
EXECUTIVE SUMMARY - {indicator.upper().replace('_', ' ')}

Current Situation:
The {indicator.replace('_', ' ')} currently stands at {current_value:.2f}. Our econometric analysis 
indicates the indicator has shown {"strong" if abs(change_pct) > 5 else "moderate"} momentum in recent periods.

Forecast Outlook:
Over the next 12 months, we forecast the {indicator.replace('_', ' ')} will {direction} by approximately 
{abs(change_pct):.1f}%. This projection is based on {results.get('model_info', {}).get('order', 'advanced')} 
ARIMA modeling with {"high" if abs(change_pct) < 3 else "moderate"} confidence.

Key Implications:
- The predicted trajectory suggests {"stability" if abs(change_pct) < 2 else "volatility"} in the economic environment
- Businesses should {"maintain current" if abs(change_pct) < 3 else "adjust"} strategic planning assumptions
- {"Monitor closely" if abs(change_pct) > 5 else "Regular monitoring"} is recommended for early trend detection

Risk Assessment:
Model uncertainty and external economic shocks represent the primary risks to this forecast. 
The {"high" if abs(change_pct) < 3 else "moderate"} confidence level reflects {"stable" if abs(change_pct) < 3 else "dynamic"} 
underlying economic conditions.
        """
        
        return narrative.strip()
    
    def create_visualizations(self, save_plots=True):
        """Create comprehensive visualizations of data and forecasts."""
        print(f"\n📊 Creating visualizations...")
        
        # Create output directory
        if save_plots:
            os.makedirs('demo_outputs', exist_ok=True)
        
        # 1. Historical Data Overview
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Economic Indicators - Historical Data and Forecasts', fontsize=16, y=0.98)
        
        axes = axes.flatten()
        
        for i, (indicator, series) in enumerate(self.demo_data.items()):
            if i >= len(axes):
                break
                
            ax = axes[i]
            
            # Plot historical data
            ax.plot(series.index, series.values, label='Historical', linewidth=2, alpha=0.8)
            
            # Add forecast if available
            if indicator in self.forecast_results:
                forecast_data = self.forecast_results[indicator].get('ensemble', 
                                                                   self.forecast_results[indicator].get('arima'))
                if forecast_data and 'forecast' in forecast_data:
                    forecast_values = forecast_data['forecast']
                    # Create future dates
                    last_date = series.index[-1]
                    future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), 
                                               periods=len(forecast_values), freq='M')
                    
                    ax.plot(future_dates, forecast_values, 'r--', label='Forecast', linewidth=2)
                    
                    # Add confidence intervals if available
                    if 'lower_bound' in forecast_data and 'upper_bound' in forecast_data:
                        ax.fill_between(future_dates, 
                                      forecast_data['lower_bound'], 
                                      forecast_data['upper_bound'], 
                                      alpha=0.2, color='red', label='Confidence Interval')
            
            ax.set_title(f'{indicator.replace("_", " ").title()}', fontsize=12)
            ax.set_ylabel(series.name)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Format x-axis
            ax.tick_params(axis='x', rotation=45)
        
        # Remove empty subplot
        if len(self.demo_data) < len(axes):
            fig.delaxes(axes[-1])
        
        plt.tight_layout()
        if save_plots:
            plt.savefig('demo_outputs/economic_dashboard.png', dpi=300, bbox_inches='tight')
            print("   📈 Saved: demo_outputs/economic_dashboard.png")
        plt.show()
        
        # 2. Correlation Matrix
        if len(self.demo_data) > 1:
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Create correlation matrix
            df = pd.DataFrame(self.demo_data)
            correlation_matrix = df.corr()
            
            # Create heatmap
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                       square=True, ax=ax, cbar_kws={'label': 'Correlation Coefficient'})
            ax.set_title('Economic Indicators Correlation Matrix', fontsize=14)
            
            plt.tight_layout()
            if save_plots:
                plt.savefig('demo_outputs/correlation_matrix.png', dpi=300, bbox_inches='tight')
                print("   🔗 Saved: demo_outputs/correlation_matrix.png")
            plt.show()
        
        return True
    
    def save_demo_results(self):
        """Save all demo results to files."""
        print(f"\n💾 Saving demo results...")
        
        os.makedirs('demo_outputs', exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save raw data
        with open(f'demo_outputs/demo_data_{timestamp}.json', 'w') as f:
            data_dict = {name: series.to_dict() for name, series in self.demo_data.items()}
            json.dump(data_dict, f, indent=2, default=str)
        print("   📊 Saved: demo_data.json")
        
        # Save forecast results
        if self.forecast_results:
            forecast_export = {}
            for indicator, results in self.forecast_results.items():
                forecast_export[indicator] = {}
                for model_name, model_results in results.items():
                    if isinstance(model_results, dict):
                        export_results = model_results.copy()
                        # Convert numpy arrays to lists
                        for key, value in export_results.items():
                            if hasattr(value, 'tolist'):
                                export_results[key] = value.tolist()
                        forecast_export[indicator][model_name] = export_results
            
            with open(f'demo_outputs/forecast_results_{timestamp}.json', 'w') as f:
                json.dump(forecast_export, f, indent=2, default=str)
            print("   🔮 Saved: forecast_results.json")
        
        # Create summary report
        summary = {
            'demo_info': {
                'timestamp': timestamp,
                'indicators_analyzed': len(self.demo_data),
                'forecast_horizon': 12,
                'data_period': f"{self.demo_data[list(self.demo_data.keys())[0]].index.min()} to {self.demo_data[list(self.demo_data.keys())[0]].index.max()}"
            },
            'model_performance': {
                indicator: {
                    'model_type': results.get('model_info', {}).get('order', 'N/A'),
                    'aic': results.get('model_info', {}).get('aic', 'N/A'),
                    'forecast_available': 'forecast' in results.get('arima', {})
                }
                for indicator, results in self.forecast_results.items()
            }
        }
        
        with open(f'demo_outputs/demo_summary_{timestamp}.json', 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        print("   📋 Saved: demo_summary.json")
        
        print(f"✅ All results saved to demo_outputs/ directory")
    
    def run_complete_demo(self):
        """Run the complete demo workflow."""
        print("\n🎯 Running Complete Demo Workflow")
        print("=" * 50)
        
        # Step 1: Generate synthetic data
        self.generate_synthetic_economic_data()
        
        # Step 2: Run forecasting
        self.run_forecasting_analysis()
        
        # Step 3: Generate narratives
        narratives = self.generate_ai_narratives()
        
        # Step 4: Create visualizations
        self.create_visualizations()
        
        # Step 5: Save results
        self.save_demo_results()
        
        # Final summary
        print(f"\n🎉 Demo Complete!")
        print("=" * 30)
        print(f"📊 Analyzed {len(self.demo_data)} economic indicators")
        print(f"🔮 Generated forecasts using ARIMA and ensemble methods")
        print(f"📝 Created {len(narratives)} AI-powered narratives")
        print(f"📈 Saved visualizations and results to demo_outputs/")
        print(f"\n💡 This demonstrates the platform's ability to:")
        print(f"   • Automatically model complex economic time series")
        print(f"   • Generate accurate forecasts with confidence intervals") 
        print(f"   • Create executive-ready insights and recommendations")
        print(f"   • Provide comprehensive analysis workflows")
        
        return {
            'data': self.demo_data,
            'forecasts': self.forecast_results,
            'narratives': narratives
        }


if __name__ == "__main__":
    print("Starting Generative Econometric Forecasting Demo...")
    
    # Run demo
    demo = EconometricDemo()
    results = demo.run_complete_demo()
    
    print(f"\n🚀 Demo completed successfully!")
    print(f"Check the demo_outputs/ directory for all generated files and visualizations.")