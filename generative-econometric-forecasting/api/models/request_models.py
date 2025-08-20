"""
Request models for the Econometric Forecasting API.

Pydantic models for validating incoming API requests.
"""

from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from enum import Enum

class IndicatorType(str, Enum):
    """Economic indicators supported by the platform."""
    GDP = "gdp"
    UNEMPLOYMENT = "unemployment" 
    INFLATION = "inflation"
    INTEREST_RATE = "interest_rate"
    CONSUMER_SPENDING = "consumer_spending"
    HOUSING_STARTS = "housing_starts"
    MANUFACTURING = "manufacturing"
    RETAIL_SALES = "retail_sales"

class ModelTier(str, Enum):
    """Foundation model tiers."""
    TIER1 = "tier1"  # TimeGPT (premium)
    TIER2 = "tier2"  # Nixtla OSS (professional)
    TIER3 = "tier3"  # HuggingFace (good)
    AUTO = "auto"    # Automatic selection

class ForecastMethod(str, Enum):
    """Available forecasting methods."""
    FOUNDATION = "foundation"      # Foundation models
    NEURAL = "neural"             # Neural networks
    STATISTICAL = "statistical"   # ARIMA, VAR, etc.
    ENSEMBLE = "ensemble"         # Combined methods
    SENTIMENT_ADJUSTED = "sentiment_adjusted"

class ScenarioType(str, Enum):
    """Economic scenario types."""
    BASELINE = "baseline"
    OPTIMISTIC = "optimistic"
    PESSIMISTIC = "pessimistic"
    RECESSION = "recession"
    RECOVERY = "recovery"
    CUSTOM = "custom"

class ForecastRequest(BaseModel):
    """Request model for economic forecasting."""
    
    indicators: List[IndicatorType] = Field(
        ..., 
        description="List of economic indicators to forecast",
        example=["gdp", "unemployment"]
    )
    
    horizon: int = Field(
        12, 
        ge=1, 
        le=24,
        description="Forecast horizon in months (1-24)",
        example=12
    )
    
    method: ForecastMethod = Field(
        ForecastMethod.FOUNDATION,
        description="Forecasting method to use",
        example="foundation"
    )
    
    model_tier: ModelTier = Field(
        ModelTier.AUTO,
        description="Foundation model tier (if using foundation method)",
        example="auto"
    )
    
    start_date: Optional[str] = Field(
        None,
        description="Start date for historical data (YYYY-MM-DD format)",
        example="2020-01-01"
    )
    
    confidence_interval: float = Field(
        0.95,
        ge=0.8,
        le=0.99,
        description="Confidence interval for predictions (0.8-0.99)",
        example=0.95
    )
    
    include_sentiment: bool = Field(
        False,
        description="Include sentiment analysis in forecast",
        example=False
    )
    
    generate_report: bool = Field(
        True,
        description="Generate executive summary report",
        example=True
    )
    
    @validator('start_date')
    def validate_start_date(cls, v):
        """Validate start date format."""
        if v is not None:
            try:
                datetime.strptime(v, '%Y-%m-%d')
            except ValueError:
                raise ValueError('start_date must be in YYYY-MM-DD format')
        return v

class ScenarioAnalysisRequest(BaseModel):
    """Request model for scenario analysis."""
    
    indicators: List[IndicatorType] = Field(
        ...,
        description="Economic indicators for scenario analysis"
    )
    
    scenarios: List[ScenarioType] = Field(
        [ScenarioType.BASELINE, ScenarioType.OPTIMISTIC, ScenarioType.PESSIMISTIC],
        description="Scenarios to analyze"
    )
    
    horizon: int = Field(
        12,
        ge=1,
        le=24,
        description="Analysis horizon in months"
    )
    
    custom_parameters: Optional[Dict[str, float]] = Field(
        None,
        description="Custom scenario parameters"
    )
    
    monte_carlo_runs: int = Field(
        1000,
        ge=100,
        le=10000,
        description="Number of Monte Carlo simulations"
    )

class CausalInferenceRequest(BaseModel):
    """Request model for causal inference analysis."""
    
    treatment_indicator: IndicatorType = Field(
        ...,
        description="Treatment variable (policy intervention)"
    )
    
    outcome_indicators: List[IndicatorType] = Field(
        ...,
        description="Outcome variables to analyze"
    )
    
    treatment_date: str = Field(
        ...,
        description="Date of treatment/intervention (YYYY-MM-DD)",
        example="2020-03-01"
    )
    
    control_variables: Optional[List[IndicatorType]] = Field(
        None,
        description="Control variables for analysis"
    )
    
    method: str = Field(
        "difference_in_differences",
        description="Causal inference method",
        example="difference_in_differences"
    )
    
    @validator('treatment_date')
    def validate_treatment_date(cls, v):
        """Validate treatment date format."""
        try:
            datetime.strptime(v, '%Y-%m-%d')
        except ValueError:
            raise ValueError('treatment_date must be in YYYY-MM-DD format')
        return v

class SensitivityTestRequest(BaseModel):
    """Request model for sensitivity testing."""
    
    base_forecast_id: str = Field(
        ...,
        description="ID of base forecast for sensitivity testing"
    )
    
    parameters_to_test: List[str] = Field(
        ...,
        description="Parameters to test for sensitivity",
        example=["confidence_interval", "horizon", "method"]
    )
    
    variation_range: float = Field(
        0.2,
        ge=0.05,
        le=0.5,
        description="Parameter variation range (±percentage)"
    )
    
    llm_analysis: bool = Field(
        True,
        description="Include LLM-based analysis of sensitivity results"
    )

class BatchForecastRequest(BaseModel):
    """Request model for batch forecasting multiple indicators."""
    
    requests: List[ForecastRequest] = Field(
        ...,
        description="List of forecast requests to process",
        max_items=10
    )
    
    parallel_processing: bool = Field(
        True,
        description="Process requests in parallel when possible"
    )
    
    combine_report: bool = Field(
        False,
        description="Combine results into single executive report"
    )