"""
Response models for the Econometric Forecasting API.

Pydantic models for API response validation and serialization.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from enum import Enum

class ForecastStatus(str, Enum):
    """Status of forecast operations."""
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    FAILED = "failed"
    IN_PROGRESS = "in_progress"

class ModelInfo(BaseModel):
    """Information about the model used for forecasting."""
    
    name: str = Field(description="Model name")
    tier: str = Field(description="Model tier (tier1, tier2, tier3)")
    method: str = Field(description="Forecasting method")
    version: str = Field(description="Model version")
    performance_score: Optional[float] = Field(None, description="Model performance score")

class ForecastPoint(BaseModel):
    """Individual forecast data point."""
    
    date: str = Field(description="Forecast date (YYYY-MM-DD)")
    value: float = Field(description="Forecasted value")
    lower_bound: Optional[float] = Field(None, description="Lower confidence bound")
    upper_bound: Optional[float] = Field(None, description="Upper confidence bound")
    confidence: float = Field(description="Confidence level")

class IndicatorForecast(BaseModel):
    """Forecast results for a single indicator."""
    
    indicator: str = Field(description="Economic indicator name")
    current_value: Optional[float] = Field(None, description="Most recent actual value")
    forecast_points: List[ForecastPoint] = Field(description="Forecast data points")
    model_info: ModelInfo = Field(description="Model information")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

class SentimentData(BaseModel):
    """Sentiment analysis data."""
    
    overall_sentiment: float = Field(description="Overall sentiment score (-1 to 1)")
    sentiment_sources: List[str] = Field(description="Data sources for sentiment")
    confidence: float = Field(description="Sentiment confidence score")
    impact_assessment: str = Field(description="Narrative impact assessment")

class ExecutiveSummary(BaseModel):
    """Executive summary of forecast results."""
    
    key_insights: List[str] = Field(description="Key insights from analysis")
    risk_assessment: str = Field(description="Risk assessment narrative")
    recommendations: List[str] = Field(description="Action recommendations")
    confidence_level: str = Field(description="Overall confidence assessment")
    generated_at: str = Field(description="Summary generation timestamp")

class ForecastResponse(BaseModel):
    """Response model for forecast requests."""
    
    success: bool = Field(description="Whether forecast succeeded")
    status: ForecastStatus = Field(description="Detailed status")
    request_id: str = Field(description="Unique request identifier")
    
    forecasts: List[IndicatorForecast] = Field(
        default_factory=list,
        description="Forecast results by indicator"
    )
    
    sentiment_analysis: Optional[SentimentData] = Field(
        None,
        description="Sentiment analysis data (if requested)"
    )
    
    executive_summary: Optional[ExecutiveSummary] = Field(
        None,
        description="Executive summary (if requested)"
    )
    
    processing_time: float = Field(description="Processing time in seconds")
    model_performance: Dict[str, float] = Field(
        default_factory=dict,
        description="Model performance metrics"
    )
    
    error: Optional[str] = Field(None, description="Error message if failed")

class ScenarioResult(BaseModel):
    """Results for a single scenario."""
    
    scenario_name: str = Field(description="Scenario identifier")
    scenario_description: str = Field(description="Scenario description")
    probability: float = Field(description="Scenario probability")
    
    forecasts: List[IndicatorForecast] = Field(description="Forecast results for scenario")
    impact_summary: str = Field(description="Narrative impact summary")
    risk_factors: List[str] = Field(description="Key risk factors")

class ScenarioAnalysisResponse(BaseModel):
    """Response model for scenario analysis."""
    
    success: bool = Field(description="Whether analysis succeeded")
    request_id: str = Field(description="Unique request identifier")
    
    scenarios: List[ScenarioResult] = Field(description="Results by scenario")
    comparative_analysis: str = Field(description="Cross-scenario comparison")
    recommended_scenario: str = Field(description="Most likely scenario")
    
    monte_carlo_summary: Dict[str, Any] = Field(
        default_factory=dict,
        description="Monte Carlo simulation summary"
    )
    
    processing_time: float = Field(description="Processing time in seconds")
    error: Optional[str] = Field(None, description="Error message if failed")

class CausalEffect(BaseModel):
    """Causal effect measurement."""
    
    outcome_indicator: str = Field(description="Outcome variable")
    treatment_effect: float = Field(description="Estimated treatment effect")
    confidence_interval: List[float] = Field(description="95% confidence interval")
    p_value: float = Field(description="Statistical significance")
    effect_size: str = Field(description="Effect size interpretation")

class CausalInferenceResponse(BaseModel):
    """Response model for causal inference analysis."""
    
    success: bool = Field(description="Whether analysis succeeded")
    request_id: str = Field(description="Unique request identifier")
    
    treatment_indicator: str = Field(description="Treatment variable")
    treatment_date: str = Field(description="Treatment date")
    method_used: str = Field(description="Causal inference method")
    
    causal_effects: List[CausalEffect] = Field(description="Estimated causal effects")
    
    robustness_tests: Dict[str, float] = Field(
        default_factory=dict,
        description="Robustness test results"
    )
    
    interpretation: str = Field(description="Natural language interpretation")
    policy_implications: List[str] = Field(description="Policy recommendations")
    
    processing_time: float = Field(description="Processing time in seconds")
    error: Optional[str] = Field(None, description="Error message if failed")

class SensitivityResult(BaseModel):
    """Sensitivity test result for a parameter."""
    
    parameter: str = Field(description="Parameter tested")
    base_value: Union[float, str] = Field(description="Base parameter value")
    test_values: List[Union[float, str]] = Field(description="Test parameter values")
    sensitivity_score: float = Field(description="Sensitivity impact score")
    impact_description: str = Field(description="Impact description")

class SensitivityTestResponse(BaseModel):
    """Response model for sensitivity testing."""
    
    success: bool = Field(description="Whether testing succeeded")
    request_id: str = Field(description="Unique request identifier")
    
    base_forecast_id: str = Field(description="Original forecast ID")
    sensitivity_results: List[SensitivityResult] = Field(description="Sensitivity test results")
    
    overall_robustness: str = Field(description="Overall robustness assessment")
    most_sensitive_parameter: str = Field(description="Most sensitive parameter")
    
    llm_analysis: Optional[str] = Field(
        None,
        description="LLM-generated sensitivity analysis"
    )
    
    recommendations: List[str] = Field(description="Recommendations for model improvement")
    
    processing_time: float = Field(description="Processing time in seconds")
    error: Optional[str] = Field(None, description="Error message if failed")

class BatchForecastResponse(BaseModel):
    """Response model for batch forecasting."""
    
    success: bool = Field(description="Whether batch processing succeeded")
    request_id: str = Field(description="Unique request identifier")
    
    completed_forecasts: List[ForecastResponse] = Field(description="Completed forecasts")
    failed_requests: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Failed request details"
    )
    
    combined_summary: Optional[ExecutiveSummary] = Field(
        None,
        description="Combined executive summary (if requested)"
    )
    
    processing_summary: Dict[str, Any] = Field(
        default_factory=dict,
        description="Batch processing statistics"
    )
    
    total_processing_time: float = Field(description="Total processing time")

class HealthResponse(BaseModel):
    """Health check response model."""
    
    status: str = Field(description="Service status")
    timestamp: str = Field(description="Health check timestamp")
    version: str = Field(description="API version")
    
    services: Dict[str, str] = Field(
        default_factory=dict,
        description="Status of dependent services"
    )
    
    system_info: Dict[str, Any] = Field(
        default_factory=dict,
        description="System information"
    )

class APIErrorResponse(BaseModel):
    """Standard error response model."""
    
    success: bool = Field(False, description="Always false for errors")
    error: Dict[str, Any] = Field(description="Error details")
    data: None = Field(None, description="Always null for errors")
    request_id: Optional[str] = Field(None, description="Request identifier")