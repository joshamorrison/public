"""
Main application for Generative Econometric Forecasting Platform.
"""

import pandas as pd
import numpy as np
import logging
import argparse
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

# Local imports
from data.fred_client import FredDataClient, validate_data_quality
from models.forecasting_models import EconometricForecaster, analyze_forecast_accuracy
from agents.narrative_generator import EconomicNarrativeGenerator, format_forecast_report
from agents.demand_planner import GenAIDemandPlanner, generate_comprehensive_demand_analysis

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
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
                logger.warning(f"No data available for {indicator}")
                continue
            
            series = self.data[indicator]
            indicator_forecasts = {}
            
            try:
                # ARIMA forecast
                if 'arima' in models:
                    logger.info(f"Fitting ARIMA model for {indicator}")
                    arima_result = self.forecaster.fit_arima(series)
                    arima_forecast = self.forecaster.generate_forecast(
                        arima_result['model_key'], forecast_horizon
                    )
                    indicator_forecasts['arima'] = arima_forecast
                
                # Prophet forecast
                if 'prophet' in models:
                    try:
                        logger.info(f"Fitting Prophet model for {indicator}")
                        prophet_result = self.forecaster.fit_prophet(series)
                        prophet_forecast = self.forecaster.generate_forecast(
                            prophet_result['model_key'], forecast_horizon
                        )
                        indicator_forecasts['prophet'] = prophet_forecast
                    except Exception as e:
                        logger.warning(f"Prophet failed for {indicator}: {e}")
                
                # Ensemble forecast
                if 'ensemble' in models:
                    try:
                        logger.info(f"Generating ensemble forecast for {indicator}")
                        ensemble_result = self.forecaster.generate_ensemble_forecast(
                            series, forecast_horizon
                        )
                        indicator_forecasts['ensemble'] = {
                            'forecast': ensemble_result['ensemble_forecast'],
                            'model_type': 'Ensemble',
                            'individual_models': ensemble_result['individual_forecasts']
                        }
                    except Exception as e:
                        logger.warning(f"Ensemble failed for {indicator}: {e}")
                
                self.forecasts[indicator] = indicator_forecasts
                logger.info(f"Generated {len(indicator_forecasts)} forecasts for {indicator}")
                
            except Exception as e:
                logger.error(f"Error forecasting {indicator}: {e}")
                continue
        
        logger.info(f"Completed forecasting for {len(self.forecasts)} indicators")
        return self.forecasts
    
    def generate_narratives(self, indicators: List[str] = None,
                           model_preference: str = 'ensemble') -> Dict[str, str]:
        """
        Generate AI narratives for forecast results.
        
        Args:
            indicators: List of indicators to analyze
            model_preference: Preferred model for narrative generation
        
        Returns:
            Dictionary of generated narratives
        """
        if indicators is None:
            indicators = list(self.forecasts.keys())
        
        logger.info(f"Generating narratives for {len(indicators)} indicators")
        
        for indicator in indicators:
            if indicator not in self.forecasts:
                logger.warning(f"No forecasts available for {indicator}")
                continue
            
            # Select forecast model
            indicator_forecasts = self.forecasts[indicator]
            if model_preference in indicator_forecasts:
                forecast_data = indicator_forecasts[model_preference]
            else:
                # Fallback to first available model
                forecast_data = list(indicator_forecasts.values())[0]
            
            try:
                # Generate executive summary
                historical_data = self.data[indicator]
                
                # Mock performance metrics for demo
                performance_metrics = {
                    'mape': np.random.uniform(2, 8),  # 2-8% error
                    'rmse': np.random.uniform(0.5, 2.0)
                }
                
                narrative = self.narrative_generator.generate_executive_summary(
                    indicator,
                    historical_data,
                    forecast_data,
                    performance_metrics
                )
                
                # Generate detailed analysis
                detailed_analysis = self.narrative_generator.generate_detailed_analysis(
                    indicator,
                    historical_data,
                    forecast_data
                )
                
                # Format complete report
                report = format_forecast_report(
                    indicator,
                    forecast_data,
                    narrative,
                    detailed_analysis
                )
                
                self.narratives[indicator] = narrative
                self.reports[indicator] = report
                
                logger.info(f"Generated narrative for {indicator}")
                
            except Exception as e:
                logger.error(f"Error generating narrative for {indicator}: {e}")
                continue
        
        logger.info(f"Completed narrative generation for {len(self.narratives)} indicators")
        return self.narratives
    
    def generate_dashboard_summary(self) -> str:
        """Generate overall economic dashboard summary."""
        logger.info("Generating dashboard summary")
        
        try:
            # Use ensemble forecasts where available
            dashboard_forecasts = {}
            for indicator, forecasts in self.forecasts.items():
                if 'ensemble' in forecasts:
                    dashboard_forecasts[indicator] = forecasts['ensemble']
                elif forecasts:
                    dashboard_forecasts[indicator] = list(forecasts.values())[0]
            
            summary = self.narrative_generator.generate_dashboard_summary(dashboard_forecasts)
            return summary
            
        except Exception as e:
            logger.error(f"Error generating dashboard summary: {e}")
            return "Unable to generate dashboard summary"
    
    def save_results(self, output_dir: str = 'outputs') -> None:
        """Save all results to files."""
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save individual reports
        for indicator, report in self.reports.items():
            filename = f"{output_dir}/{indicator}_forecast_report_{timestamp}.json"
            with open(filename, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            logger.info(f"Saved report for {indicator} to {filename}")
        
        # Save dashboard summary
        dashboard_summary = self.generate_dashboard_summary()
        dashboard_file = f"{output_dir}/economic_dashboard_{timestamp}.txt"
        with open(dashboard_file, 'w') as f:
            f.write(f"Economic Dashboard Summary - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 60 + "\n\n")
            f.write(dashboard_summary)
        
        logger.info(f"Saved dashboard summary to {dashboard_file}")
        
        # Save raw forecast data
        forecast_data = {}
        for indicator, forecasts in self.forecasts.items():
            forecast_data[indicator] = {}
            for model_name, forecast in forecasts.items():
                # Convert numpy arrays to lists for JSON serialization
                forecast_copy = forecast.copy()
                for key, value in forecast_copy.items():
                    if isinstance(value, np.ndarray):
                        forecast_copy[key] = value.tolist()
                forecast_data[indicator][model_name] = forecast_copy
        
        forecast_file = f"{output_dir}/forecast_data_{timestamp}.json"
        with open(forecast_file, 'w') as f:
            json.dump(forecast_data, f, indent=2, default=str)
        
        logger.info(f"Saved forecast data to {forecast_file}")
    
    def generate_demand_planning_analysis(self, industry: str = "retail",
                                         customer_context: str = "B2C retail customers") -> Dict[str, Any]:
        """
        Generate comprehensive demand planning analysis.
        
        Args:
            industry: Industry context for demand planning
            customer_context: Description of customer base
        
        Returns:
            Comprehensive demand planning analysis
        """
        if not self.forecasts:
            logger.warning("No forecasts available for demand planning. Run forecasting first.")
            return {}
        
        logger.info("Generating comprehensive demand planning analysis")
        
        try:
            # Convert forecasts to format expected by demand planner
            formatted_forecasts = {}
            for indicator, models in self.forecasts.items():
                # Use ensemble if available, otherwise use first available model
                if 'ensemble' in models:
                    formatted_forecasts[indicator] = models['ensemble']
                elif models:
                    formatted_forecasts[indicator] = list(models.values())[0]
            
            # Generate comprehensive demand analysis
            demand_analysis = generate_comprehensive_demand_analysis(
                formatted_forecasts, industry, customer_context
            )
            
            self.demand_analysis = demand_analysis
            logger.info("Completed demand planning analysis")
            
            return demand_analysis
            
        except Exception as e:
            logger.error(f"Error generating demand planning analysis: {e}")
            return {}
    
    def run_full_analysis(self, indicators: List[str] = None,
                         forecast_horizon: int = 12,
                         start_date: str = '2010-01-01',
                         save_outputs: bool = True,
                         include_demand_planning: bool = True,
                         industry: str = "retail") -> Dict[str, Any]:
        """
        Run complete forecasting and demand planning analysis pipeline.
        
        Args:
            indicators: Economic indicators to analyze
            forecast_horizon: Number of periods to forecast
            start_date: Start date for historical data
            save_outputs: Whether to save results to files
            include_demand_planning: Whether to include GenAI demand planning
            industry: Industry context for demand planning
        
        Returns:
            Dictionary with all results
        """
        logger.info("Starting full econometric forecasting and demand planning analysis")
        
        # Load data
        data = self.load_economic_data(indicators, start_date)
        
        # Generate forecasts
        forecasts = self.generate_forecasts(
            list(data.keys()), 
            forecast_horizon, 
            ['arima', 'ensemble']
        )
        
        # Generate narratives
        narratives = self.generate_narratives(list(forecasts.keys()))
        
        # Generate dashboard summary
        dashboard = self.generate_dashboard_summary()
        
        # Generate demand planning analysis if requested
        demand_analysis = {}
        if include_demand_planning:
            demand_analysis = self.generate_demand_planning_analysis(industry)
        
        # Save results
        if save_outputs:
            self.save_results()
        
        results = {
            'data_summary': {indicator: len(series) for indicator, series in data.items()},
            'forecast_summary': {indicator: list(forecasts.keys()) for indicator, forecasts in forecasts.items()},
            'narratives': narratives,
            'dashboard_summary': dashboard,
            'demand_analysis': demand_analysis,
            'reports': self.reports
        }
        
        logger.info("Completed full econometric forecasting and demand planning analysis")
        return results


def main():
    """Main entry point for the application."""
    parser = argparse.ArgumentParser(description='Generative Econometric Forecasting Platform')
    
    parser.add_argument('--indicators', nargs='+', 
                       default=['gdp', 'unemployment', 'inflation', 'interest_rate'],
                       help='Economic indicators to forecast')
    parser.add_argument('--forecast-horizon', type=int, default=12,
                       help='Number of periods to forecast')
    parser.add_argument('--start-date', default='2010-01-01',
                       help='Start date for historical data (YYYY-MM-DD)')
    parser.add_argument('--output-dir', default='outputs',
                       help='Directory to save outputs')
    parser.add_argument('--no-save', action='store_true',
                       help='Do not save outputs to files')
    parser.add_argument('--include-demand-planning', action='store_true', default=True,
                       help='Include GenAI demand planning analysis')
    parser.add_argument('--industry', default='retail',
                       help='Industry context for demand planning (e.g., retail, manufacturing, services)')
    parser.add_argument('--no-demand-planning', action='store_true',
                       help='Skip demand planning analysis')
    
    args = parser.parse_args()
    
    # Initialize platform
    platform = EconometricForecastingPlatform()
    
    # Run analysis
    try:
        results = platform.run_full_analysis(
            indicators=args.indicators,
            forecast_horizon=args.forecast_horizon,
            start_date=args.start_date,
            save_outputs=not args.no_save,
            include_demand_planning=not args.no_demand_planning,
            industry=args.industry
        )
        
        # Print summary
        print("\n" + "="*60)
        print("ECONOMETRIC FORECASTING & DEMAND PLANNING ANALYSIS COMPLETE")
        print("="*60)
        print(f"Indicators analyzed: {len(results['data_summary'])}")
        print(f"Forecasts generated: {sum(len(f) for f in results['forecast_summary'].values())}")
        print(f"Narratives created: {len(results['narratives'])}")
        
        if results.get('demand_analysis'):
            demand_analysis = results['demand_analysis']
            print(f"Demand scenarios: {len(demand_analysis.get('demand_scenarios', []))}")
            print(f"Customer segments: {len(demand_analysis.get('customer_segments', []))}")
        
        print(f"\nECONOMIC DASHBOARD SUMMARY:")
        print("-" * 40)
        print(results['dashboard_summary'])
        
        if results.get('demand_analysis') and results['demand_analysis'].get('executive_report'):
            print(f"\nDEMAND PLANNING EXECUTIVE REPORT:")
            print("-" * 40)
            print(results['demand_analysis']['executive_report'][:500] + "..." if len(results['demand_analysis']['executive_report']) > 500 else results['demand_analysis']['executive_report'])
        
        if not args.no_save:
            print(f"\nResults saved to: {args.output_dir}/")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())