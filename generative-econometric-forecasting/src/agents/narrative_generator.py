"""
LangChain agents for generating executive narratives and insights from econometric forecasts.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime
import json

from langchain.agents import AgentType, initialize_agent
from langchain.tools import Tool
from langchain.schema import SystemMessage, HumanMessage
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field

logger = logging.getLogger(__name__)


class ForecastInsight(BaseModel):
    """Structure for forecast insights."""
    metric: str = Field(description="Economic metric being analyzed")
    current_trend: str = Field(description="Current trend direction")
    forecast_direction: str = Field(description="Predicted direction")
    confidence_level: str = Field(description="Confidence in prediction")
    key_drivers: List[str] = Field(description="Main factors influencing the forecast")
    business_implications: List[str] = Field(description="Business implications")
    risk_factors: List[str] = Field(description="Potential risks")


class EconomicNarrativeGenerator:
    """Generate executive-ready narratives from econometric forecasts."""
    
    def __init__(self, model_name: str = "gpt-3.5-turbo", temperature: float = 0.3):
        """Initialize the narrative generator."""
        self.llm = ChatOpenAI(model=model_name, temperature=temperature)
        self.parser = JsonOutputParser(pydantic_object=ForecastInsight)
        
        # Template for executive summary
        self.executive_template = PromptTemplate(
            input_variables=["metric", "historical_data", "forecast_data", "model_performance"],
            template="""
            You are a senior economic analyst preparing an executive briefing. 
            
            ECONOMIC METRIC: {metric}
            
            HISTORICAL CONTEXT:
            {historical_data}
            
            FORECAST RESULTS:
            {forecast_data}
            
            MODEL PERFORMANCE:
            {model_performance}
            
            Generate a concise executive summary that includes:
            1. Current economic situation
            2. Key forecast insights
            3. Business implications
            4. Risk assessment
            5. Recommended actions
            
            Keep the language professional but accessible to non-economists.
            Focus on actionable insights rather than technical details.
            """
        )
        
        # Template for detailed analysis
        self.analysis_template = PromptTemplate(
            input_variables=["metric", "trend_analysis", "forecast_details", "uncertainty"],
            template="""
            Provide a detailed economic analysis for {metric}.
            
            TREND ANALYSIS:
            {trend_analysis}
            
            FORECAST DETAILS:
            {forecast_details}
            
            UNCERTAINTY FACTORS:
            {uncertainty}
            
            Structure your response as a JSON object with the following fields:
            - metric: The economic indicator name
            - current_trend: Current trend direction (increasing/decreasing/stable)
            - forecast_direction: Predicted direction over forecast horizon
            - confidence_level: High/Medium/Low based on model performance
            - key_drivers: List of main factors influencing the forecast
            - business_implications: List of business implications
            - risk_factors: List of potential risks or uncertainties
            
            {format_instructions}
            """
        )
        
        self.analysis_template = self.analysis_template.partial(
            format_instructions=self.parser.get_format_instructions()
        )
    
    def analyze_trend(self, data: pd.Series, periods: int = 12) -> Dict[str, Any]:
        """
        Analyze trend characteristics of time series data.
        
        Args:
            data: Historical time series data
            periods: Number of recent periods to analyze
        
        Returns:
            Dictionary with trend analysis
        """
        recent_data = data.tail(periods)
        
        # Calculate trend metrics
        trend_direction = "increasing" if recent_data.iloc[-1] > recent_data.iloc[0] else "decreasing"
        volatility = recent_data.std()
        avg_change = recent_data.pct_change().mean() * 100
        
        # Identify turning points
        rolling_mean = data.rolling(window=6).mean()
        recent_mean = rolling_mean.tail(6)
        trend_consistency = len(recent_mean) - len(recent_mean[recent_mean.diff().apply(lambda x: x * recent_mean.diff().iloc[0] < 0)])
        
        return {
            'trend_direction': trend_direction,
            'volatility': float(volatility),
            'average_change_pct': float(avg_change),
            'trend_consistency': trend_consistency / len(recent_mean),
            'current_level': float(recent_data.iloc[-1]),
            'recent_range': (float(recent_data.min()), float(recent_data.max()))
        }
    
    def generate_executive_summary(self, metric: str, historical_data: pd.Series,
                                 forecast_results: Dict[str, Any],
                                 model_performance: Dict[str, float]) -> str:
        """
        Generate executive summary for forecast results.
        
        Args:
            metric: Name of economic metric
            historical_data: Historical time series data
            forecast_results: Forecast results from model
            model_performance: Model performance metrics
        
        Returns:
            Executive summary text
        """
        # Prepare historical context
        trend_analysis = self.analyze_trend(historical_data)
        historical_summary = f"""
        Current Level: {trend_analysis['current_level']:.2f}
        Recent Trend: {trend_analysis['trend_direction']} 
        Average Change: {trend_analysis['average_change_pct']:.1f}% per period
        Volatility: {trend_analysis['volatility']:.2f}
        """
        
        # Prepare forecast summary
        forecast_values = forecast_results.get('forecast', [])
        if len(forecast_values) > 0:
            forecast_summary = f"""
            Forecast Horizon: {len(forecast_values)} periods
            Predicted Direction: {'Increasing' if forecast_values[-1] > forecast_values[0] else 'Decreasing'}
            Forecast Range: {min(forecast_values):.2f} to {max(forecast_values):.2f}
            Expected Level in {len(forecast_values)} periods: {forecast_values[-1]:.2f}
            """
        else:
            forecast_summary = "Forecast data not available"
        
        # Prepare performance summary
        performance_summary = f"""
        Model Accuracy (MAPE): {model_performance.get('mape', 'N/A')}%
        Root Mean Square Error: {model_performance.get('rmse', 'N/A')}
        Model Type: {forecast_results.get('model_type', 'Unknown')}
        """
        
        # Generate executive summary
        prompt = self.executive_template.format(
            metric=metric,
            historical_data=historical_summary,
            forecast_data=forecast_summary,
            model_performance=performance_summary
        )
        
        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            return response.content
        except Exception as e:
            logger.error(f"Error generating executive summary: {e}")
            return f"Error generating summary for {metric}"
    
    def generate_detailed_analysis(self, metric: str, historical_data: pd.Series,
                                 forecast_results: Dict[str, Any],
                                 uncertainty_factors: List[str] = None) -> ForecastInsight:
        """
        Generate detailed structured analysis of forecast.
        
        Args:
            metric: Name of economic metric
            historical_data: Historical time series data
            forecast_results: Forecast results from model
            uncertainty_factors: List of uncertainty factors
        
        Returns:
            Structured forecast insight
        """
        # Analyze trends
        trend_analysis = self.analyze_trend(historical_data)
        trend_summary = f"""
        The {metric} has shown a {trend_analysis['trend_direction']} trend recently.
        Current level: {trend_analysis['current_level']:.2f}
        Average change rate: {trend_analysis['average_change_pct']:.1f}% per period
        Volatility level: {'High' if trend_analysis['volatility'] > historical_data.std() else 'Normal'}
        """
        
        # Forecast details
        forecast_values = forecast_results.get('forecast', [])
        forecast_summary = f"""
        Model Type: {forecast_results.get('model_type', 'Unknown')}
        Forecast Horizon: {len(forecast_values)} periods
        Predicted Change: {((forecast_values[-1] / forecast_values[0] - 1) * 100):.1f}% over forecast period
        Confidence Intervals: {'Available' if 'lower_bound' in forecast_results else 'Not available'}
        """
        
        # Uncertainty factors
        uncertainty_list = uncertainty_factors or [
            "Model estimation uncertainty",
            "Unexpected economic shocks",
            "Policy changes",
            "External market conditions"
        ]
        
        # Generate structured analysis
        prompt = self.analysis_template.format(
            metric=metric,
            trend_analysis=trend_summary,
            forecast_details=forecast_summary,
            uncertainty="; ".join(uncertainty_list)
        )
        
        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            parsed_result = self.parser.parse(response.content)
            return parsed_result
        except Exception as e:
            logger.error(f"Error generating detailed analysis: {e}")
            # Return default structure
            return ForecastInsight(
                metric=metric,
                current_trend="Unable to determine",
                forecast_direction="Unable to determine",
                confidence_level="Low",
                key_drivers=["Analysis error"],
                business_implications=["Unable to analyze"],
                risk_factors=["Analysis uncertainty"]
            )
    
    def generate_scenario_analysis(self, base_forecast: Dict[str, Any],
                                 scenarios: Dict[str, str]) -> str:
        """
        Generate scenario analysis with different economic conditions.
        
        Args:
            base_forecast: Base case forecast results
            scenarios: Dictionary of scenario names and descriptions
        
        Returns:
            Scenario analysis narrative
        """
        scenario_template = PromptTemplate(
            input_variables=["base_forecast", "scenarios"],
            template="""
            Based on the base forecast scenario, analyze the following alternative scenarios:
            
            BASE FORECAST:
            {base_forecast}
            
            ALTERNATIVE SCENARIOS:
            {scenarios}
            
            For each scenario, provide:
            1. How it would modify the base forecast
            2. Probability assessment
            3. Impact on business planning
            4. Early warning indicators to monitor
            
            Structure as a clear executive briefing on scenario planning.
            """
        )
        
        base_summary = f"""
        Model: {base_forecast.get('model_type', 'Unknown')}
        Forecast Values: {base_forecast.get('forecast', [])}
        Confidence: {base_forecast.get('confidence_level', 'Unknown')}
        """
        
        scenario_list = "\n".join([f"{name}: {desc}" for name, desc in scenarios.items()])
        
        prompt = scenario_template.format(
            base_forecast=base_summary,
            scenarios=scenario_list
        )
        
        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            return response.content
        except Exception as e:
            logger.error(f"Error generating scenario analysis: {e}")
            return "Unable to generate scenario analysis"
    
    def generate_dashboard_summary(self, multiple_forecasts: Dict[str, Dict[str, Any]]) -> str:
        """
        Generate summary for multiple economic indicators.
        
        Args:
            multiple_forecasts: Dictionary of metric forecasts
        
        Returns:
            Dashboard summary text
        """
        dashboard_template = PromptTemplate(
            input_variables=["forecasts_summary"],
            template="""
            Generate an executive dashboard summary for multiple economic indicators:
            
            {forecasts_summary}
            
            Provide:
            1. Overall economic outlook
            2. Key risks and opportunities
            3. Interconnections between indicators
            4. Strategic recommendations
            5. Monitoring priorities
            
            Keep it concise but comprehensive for C-level executives.
            """
        )
        
        # Summarize all forecasts
        forecasts_text = []
        for metric, forecast in multiple_forecasts.items():
            forecast_values = forecast.get('forecast', [])
            if len(forecast_values) > 0:
                direction = "↑" if forecast_values[-1] > forecast_values[0] else "↓"
                change_pct = ((forecast_values[-1] / forecast_values[0] - 1) * 100)
                forecasts_text.append(f"{metric}: {direction} {change_pct:.1f}% change predicted")
            else:
                forecasts_text.append(f"{metric}: Forecast unavailable")
        
        prompt = dashboard_template.format(
            forecasts_summary="\n".join(forecasts_text)
        )
        
        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            return response.content
        except Exception as e:
            logger.error(f"Error generating dashboard summary: {e}")
            return "Unable to generate dashboard summary"


def format_forecast_report(metric: str, forecast_results: Dict[str, Any],
                          narrative: str, detailed_analysis: ForecastInsight) -> Dict[str, Any]:
    """
    Format complete forecast report with narrative and data.
    
    Args:
        metric: Economic metric name
        forecast_results: Raw forecast results
        narrative: Executive narrative
        detailed_analysis: Structured analysis
    
    Returns:
        Complete formatted report
    """
    return {
        'metric': metric,
        'timestamp': datetime.now().isoformat(),
        'executive_summary': narrative,
        'detailed_analysis': detailed_analysis.dict(),
        'forecast_data': {
            'values': forecast_results.get('forecast', []).tolist() if hasattr(forecast_results.get('forecast', []), 'tolist') else forecast_results.get('forecast', []),
            'model_type': forecast_results.get('model_type', 'Unknown'),
            'confidence_intervals': {
                'lower': forecast_results.get('lower_bound', []).tolist() if hasattr(forecast_results.get('lower_bound', []), 'tolist') else forecast_results.get('lower_bound', []),
                'upper': forecast_results.get('upper_bound', []).tolist() if hasattr(forecast_results.get('upper_bound', []), 'tolist') else forecast_results.get('upper_bound', [])
            }
        }
    }


if __name__ == "__main__":
    # Example usage
    generator = EconomicNarrativeGenerator()
    
    # Generate sample data
    dates = pd.date_range('2020-01-01', '2023-12-01', freq='M')
    sample_data = pd.Series(
        np.random.randn(len(dates)).cumsum() + 100,
        index=dates,
        name='Sample Economic Indicator'
    )
    
    # Sample forecast results
    sample_forecast = {
        'forecast': np.array([102, 103, 104, 105, 106, 107]),
        'model_type': 'ARIMA',
        'lower_bound': np.array([100, 101, 102, 103, 104, 105]),
        'upper_bound': np.array([104, 105, 106, 107, 108, 109])
    }
    
    # Generate narratives
    summary = generator.generate_executive_summary(
        "GDP Growth", sample_data, sample_forecast, {'mape': 2.5, 'rmse': 1.2}
    )
    print("Executive Summary:")
    print(summary)
    
    detailed = generator.generate_detailed_analysis(
        "GDP Growth", sample_data, sample_forecast
    )
    print("\nDetailed Analysis:")
    print(detailed)