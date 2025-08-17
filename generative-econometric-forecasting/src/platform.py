"""
Core Platform for Generative Econometric Forecasting.
Orchestrates data, models, and AI agents.
"""

import pandas as pd
import numpy as np
import logging
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

# Local imports
from data.fred_client import FredDataClient, validate_data_quality
from models.forecasting_models import EconometricForecaster, analyze_forecast_accuracy
from src.agents.narrative_generator import EconomicNarrativeGenerator, format_forecast_report
from src.agents.demand_planner import GenAIDemandPlanner, generate_comprehensive_demand_analysis

# Configure logging
logger = logging.getLogger(__name__)


class EconometricForecastingPlatform:
    """Main platform orchestrating data, models, and AI agents."""
    
    def __init__(self, fred_api_key: Optional[str] = None):
        """Initialize the forecasting platform."""
        self.fred_client = FredDataClient(fred_api_key)
        self.forecaster = EconometricForecaster()
        self.narrative_generator = EconomicNarrativeGenerator()
        self.demand_planner = GenAIDemandPlanner()
        
        # Storage for results
        self.data = {}
        self.forecasts = {}
        self.narratives = {}
        self.reports = {}
        self.demand_analysis = {}
        
        logger.info("Econometric Forecasting Platform initialized")
    
    def load_economic_data(self, indicators: List[str] = None, 
                          start_date: str = '2010-01-01',
                          end_date: Optional[str] = None) -> Dict[str, pd.Series]:
        """
        Load economic data for specified indicators.
        
        Args:
            indicators: List of economic indicators to load
            start_date: Start date for data
            end_date: End date for data
        
        Returns:
            Dictionary of loaded time series data
        """
        if indicators is None:
            # Default economic dashboard indicators
            indicators = ['gdp', 'unemployment', 'inflation', 'interest_rate', 'consumer_confidence']
        
        logger.info(f"Loading data for indicators: {indicators}")
        
        try:
            # Load individual series
            for indicator in indicators:
                series = self.fred_client.fetch_indicator(indicator, start_date, end_date)
                self.data[indicator] = series
                
                # Validate data quality
                quality = validate_data_quality(pd.DataFrame({indicator: series}))
                logger.info(f"{indicator} - Observations: {quality['total_observations']}, "
                           f"Missing: {quality['missing_percentage'][indicator]:.1f}%")
            
            logger.info(f"Successfully loaded {len(self.data)} indicators")
            return self.data
            
        except Exception as e:
            logger.error(f"Error loading economic data: {e}")
            raise
    
    def generate_forecasts(self, indicators: List[str] = None,
                          forecast_horizon: int = 12,
                          models: List[str] = None) -> Dict[str, Dict[str, Any]]:
        """
        Generate forecasts for specified indicators.
        
        Args:
            indicators: List of indicators to forecast (uses all loaded if None)
            forecast_horizon: Number of periods to forecast
            models: List of models to use ['arima', 'prophet', 'ensemble']
        
        Returns:
            Dictionary of forecast results
        """
        if indicators is None:
            indicators = list(self.data.keys())
        
        if models is None:
            models = ['arima', 'ensemble']
        
        logger.info(f"Generating forecasts for {len(indicators)} indicators using {models}")
        
        for indicator in indicators:
            if indicator not in self.data:
                logger.warning(f"Data for {indicator} not loaded. Skipping forecast.")
                continue
            
            series = self.data[indicator]
            indicator_forecasts = {}
            
            for model in models:
                try:
                    if model == 'arima':
                        result = self.forecaster.fit_arima(series)
                        forecast = self.forecaster.generate_forecast(
                            result['model_key'], periods=forecast_horizon
                        )
                        indicator_forecasts[model] = {
                            'forecast': forecast['forecast'],
                            'confidence_intervals': forecast['confidence_intervals'],
                            'metrics': result['metrics']
                        }
                    
                    elif model == 'prophet':
                        result = self.forecaster.fit_prophet(series)
                        forecast = self.forecaster.generate_forecast(
                            result['model_key'], periods=forecast_horizon
                        )
                        indicator_forecasts[model] = {
                            'forecast': forecast['forecast'],
                            'confidence_intervals': forecast['confidence_intervals'],
                            'metrics': result['metrics']
                        }
                    
                    elif model == 'ensemble':
                        ensemble_result = self.forecaster.create_ensemble_forecast(
                            series, forecast_horizon
                        )
                        indicator_forecasts[model] = ensemble_result
                
                except Exception as e:
                    logger.error(f"Error forecasting {indicator} with {model}: {e}")
                    continue
            
            self.forecasts[indicator] = indicator_forecasts
        
        logger.info(f"Generated forecasts for {len(self.forecasts)} indicators")
        return self.forecasts
    
    def generate_narratives(self) -> Dict[str, str]:
        """Generate AI narratives for all forecasts."""
        logger.info("Generating AI narratives for forecasts")
        
        for indicator, forecasts in self.forecasts.items():
            try:
                # Use ensemble forecast if available, otherwise first available
                if 'ensemble' in forecasts:
                    forecast_data = forecasts['ensemble']
                else:
                    forecast_data = list(forecasts.values())[0]
                
                # Generate narrative
                narrative = self.narrative_generator.generate_executive_summary(
                    indicator, self.data[indicator], forecast_data, {}
                )
                self.narratives[indicator] = narrative
                
            except Exception as e:
                logger.error(f"Error generating narrative for {indicator}: {e}")
        
        logger.info(f"Generated narratives for {len(self.narratives)} indicators")
        return self.narratives
    
    def generate_demand_analysis(self, industry: str = "retail") -> Dict[str, Any]:
        """Generate comprehensive demand planning analysis."""
        logger.info(f"Generating demand analysis for {industry} industry")
        
        try:
            # Prepare forecast data for demand analysis
            forecast_data = {}
            for indicator, forecasts in self.forecasts.items():
                if 'ensemble' in forecasts:
                    forecast_data[indicator] = forecasts['ensemble']['forecast']
                elif forecasts:
                    forecast_data[indicator] = list(forecasts.values())[0]['forecast']
            
            # Generate demand analysis
            self.demand_analysis = generate_comprehensive_demand_analysis(
                economic_forecasts=forecast_data,
                industry=industry,
                customer_context=f"{industry} customers"
            )
            
            logger.info("Demand analysis generated successfully")
            return self.demand_analysis
            
        except Exception as e:
            logger.error(f"Error generating demand analysis: {e}")
            return {}
    
    def run_full_analysis(self, indicators: List[str] = None,
                         forecast_horizon: int = 12,
                         start_date: str = '2010-01-01',
                         save_outputs: bool = True,
                         include_demand_planning: bool = True,
                         industry: str = "retail") -> Dict[str, Any]:
        """
        Run complete econometric analysis pipeline.
        
        Returns:
            Comprehensive analysis results
        """
        logger.info("Starting full econometric analysis pipeline")
        
        # 1. Load economic data
        data = self.load_economic_data(indicators, start_date)
        
        # 2. Generate forecasts
        forecasts = self.generate_forecasts(indicators, forecast_horizon)
        
        # 3. Generate AI narratives
        narratives = self.generate_narratives()
        
        # 4. Generate demand planning analysis
        demand_analysis = {}
        if include_demand_planning:
            demand_analysis = self.generate_demand_analysis(industry)
        
        # 5. Compile results
        results = {
            'data_summary': {k: {'observations': len(v), 'latest_value': v.iloc[-1]} 
                           for k, v in data.items()},
            'forecast_summary': {k: list(v.keys()) for k, v in forecasts.items()},
            'narratives': narratives,
            'demand_analysis': demand_analysis,
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'indicators': list(data.keys()),
                'forecast_horizon': forecast_horizon,
                'start_date': start_date,
                'industry': industry
            }
        }
        
        # 6. Save outputs
        if save_outputs:
            self._save_results(results)
        
        logger.info("Full analysis pipeline completed successfully")
        return results
    
    def _save_results(self, results: Dict[str, Any]):
        """Save analysis results to files."""
        output_dir = "outputs"
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save comprehensive results
        with open(f"{output_dir}/analysis_results_{timestamp}.json", 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save individual narratives
        for indicator, narrative in results['narratives'].items():
            with open(f"{output_dir}/{indicator}_narrative_{timestamp}.txt", 'w') as f:
                f.write(narrative)
        
        logger.info(f"Results saved to {output_dir}/")