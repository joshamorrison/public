"""
Request models for the Media Mix Modeling API.

Pydantic models for validating incoming API requests.
"""

from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any, Union
from datetime import datetime, date
from enum import Enum

class ChannelType(str, Enum):
    """Marketing channel types."""
    SEARCH = "search"
    SOCIAL = "social"
    DISPLAY = "display"
    VIDEO = "video"
    EMAIL = "email"
    PRINT = "print"
    RADIO = "radio"
    TV = "tv"
    OUTDOOR = "outdoor"
    DIRECT_MAIL = "direct_mail"
    AFFILIATE = "affiliate"
    REFERRAL = "referral"

class AttributionModel(str, Enum):
    """Attribution model types."""
    FIRST_TOUCH = "first_touch"
    LAST_TOUCH = "last_touch"
    LINEAR = "linear"
    TIME_DECAY = "time_decay"
    POSITION_BASED = "position_based"
    DATA_DRIVEN = "data_driven"
    SHAPLEY = "shapley"
    MARKOV_CHAIN = "markov_chain"

class OptimizationObjective(str, Enum):
    """Budget optimization objectives."""
    MAXIMIZE_REVENUE = "maximize_revenue"
    MAXIMIZE_CONVERSIONS = "maximize_conversions"
    MAXIMIZE_ROAS = "maximize_roas"
    MINIMIZE_CPA = "minimize_cpa"
    MAXIMIZE_REACH = "maximize_reach"
    MAXIMIZE_BRAND_AWARENESS = "maximize_brand_awareness"

class TimeGranularity(str, Enum):
    """Time granularity for analysis."""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"

class AttributionRequest(BaseModel):
    """Request model for attribution analysis."""
    
    start_date: date = Field(
        ...,
        description="Start date for attribution analysis",
        example="2024-01-01"
    )
    
    end_date: date = Field(
        ...,
        description="End date for attribution analysis", 
        example="2024-12-31"
    )
    
    channels: List[ChannelType] = Field(
        ...,
        description="Marketing channels to include in analysis",
        example=["search", "social", "display"]
    )
    
    attribution_model: AttributionModel = Field(
        AttributionModel.DATA_DRIVEN,
        description="Attribution model to use",
        example="data_driven"
    )
    
    conversion_window_days: int = Field(
        30,
        ge=1,
        le=365,
        description="Conversion attribution window in days",
        example=30
    )
    
    include_view_through: bool = Field(
        True,
        description="Include view-through conversions",
        example=True
    )
    
    granularity: TimeGranularity = Field(
        TimeGranularity.DAILY,
        description="Time granularity for results",
        example="daily"
    )
    
    confidence_level: float = Field(
        0.95,
        ge=0.8,
        le=0.99,
        description="Confidence level for statistical measures",
        example=0.95
    )
    
    @validator('end_date')
    def validate_date_range(cls, v, values):
        """Validate that end_date is after start_date."""
        if 'start_date' in values and v <= values['start_date']:
            raise ValueError('end_date must be after start_date')
        return v

class BudgetOptimizationRequest(BaseModel):
    """Request model for budget optimization."""
    
    current_budget: Dict[ChannelType, float] = Field(
        ...,
        description="Current budget allocation by channel",
        example={"search": 10000, "social": 5000, "display": 3000}
    )
    
    total_budget: float = Field(
        ...,
        gt=0,
        description="Total available budget",
        example=20000
    )
    
    optimization_objective: OptimizationObjective = Field(
        OptimizationObjective.MAXIMIZE_ROAS,
        description="Optimization objective",
        example="maximize_roas"
    )
    
    constraints: Optional[Dict[str, Any]] = Field(
        None,
        description="Budget constraints and limits",
        example={
            "min_budget_per_channel": 1000,
            "max_budget_increase": 0.5,
            "required_channels": ["search", "social"]
        }
    )
    
    time_horizon_days: int = Field(
        30,
        ge=1,
        le=365,
        description="Optimization time horizon in days",
        example=30
    )
    
    historical_data_days: int = Field(
        90,
        ge=30,
        le=730,
        description="Historical data period for modeling",
        example=90
    )
    
    include_seasonality: bool = Field(
        True,
        description="Include seasonality in optimization",
        example=True
    )
    
    monte_carlo_simulations: int = Field(
        1000,
        ge=100,
        le=10000,
        description="Number of Monte Carlo simulations",
        example=1000
    )

class PerformanceAnalysisRequest(BaseModel):
    """Request model for campaign performance analysis."""
    
    start_date: date = Field(
        ...,
        description="Analysis start date",
        example="2024-01-01"
    )
    
    end_date: date = Field(
        ...,
        description="Analysis end date",
        example="2024-12-31"
    )
    
    channels: Optional[List[ChannelType]] = Field(
        None,
        description="Specific channels to analyze (all if not specified)",
        example=["search", "social"]
    )
    
    campaigns: Optional[List[str]] = Field(
        None,
        description="Specific campaign IDs to analyze",
        example=["campaign_001", "campaign_002"]
    )
    
    metrics: List[str] = Field(
        ["impressions", "clicks", "conversions", "revenue", "roas"],
        description="Metrics to include in analysis",
        example=["impressions", "clicks", "conversions", "revenue"]
    )
    
    granularity: TimeGranularity = Field(
        TimeGranularity.DAILY,
        description="Time granularity for analysis",
        example="daily"
    )
    
    include_trends: bool = Field(
        True,
        description="Include trend analysis",
        example=True
    )
    
    benchmark_comparison: bool = Field(
        False,
        description="Include industry benchmark comparison",
        example=False
    )
    
    anomaly_detection: bool = Field(
        True,
        description="Include anomaly detection",
        example=True
    )

class IncrementalityTestRequest(BaseModel):
    """Request model for incrementality testing."""
    
    test_channels: List[ChannelType] = Field(
        ...,
        description="Channels to test for incrementality",
        example=["search", "social"]
    )
    
    control_groups: Dict[str, List[str]] = Field(
        ...,
        description="Control group definitions by geography/segment",
        example={
            "geo_control": ["US-CA", "US-NY"],
            "geo_test": ["US-TX", "US-FL"]
        }
    )
    
    test_start_date: date = Field(
        ...,
        description="Test start date",
        example="2024-06-01"
    )
    
    test_end_date: date = Field(
        ...,
        description="Test end date",
        example="2024-06-30"
    )
    
    pre_test_period_days: int = Field(
        30,
        ge=14,
        le=90,
        description="Pre-test period for baseline calculation",
        example=30
    )
    
    minimum_detectable_effect: float = Field(
        0.05,
        ge=0.01,
        le=0.5,
        description="Minimum detectable effect size",
        example=0.05
    )
    
    confidence_level: float = Field(
        0.95,
        ge=0.8,
        le=0.99,
        description="Statistical confidence level",
        example=0.95
    )

class SaturationAnalysisRequest(BaseModel):
    """Request model for media saturation analysis."""
    
    channels: List[ChannelType] = Field(
        ...,
        description="Channels to analyze for saturation",
        example=["search", "display", "social"]
    )
    
    analysis_period_days: int = Field(
        90,
        ge=30,
        le=365,
        description="Analysis period in days",
        example=90
    )
    
    saturation_model: str = Field(
        "adstock",
        description="Saturation model type",
        example="adstock"
    )
    
    adstock_rate: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Adstock decay rate (if using adstock model)",
        example=0.5
    )
    
    budget_scenarios: List[float] = Field(
        [0.5, 0.75, 1.0, 1.25, 1.5, 2.0],
        description="Budget multiplier scenarios to test",
        example=[0.5, 1.0, 1.5, 2.0]
    )

class CrossChannelSynergyRequest(BaseModel):
    """Request model for cross-channel synergy analysis."""
    
    primary_channel: ChannelType = Field(
        ...,
        description="Primary channel for synergy analysis",
        example="search"
    )
    
    synergy_channels: List[ChannelType] = Field(
        ...,
        description="Channels to test for synergy effects",
        example=["social", "display", "video"]
    )
    
    analysis_start_date: date = Field(
        ...,
        description="Analysis start date",
        example="2024-01-01"
    )
    
    analysis_end_date: date = Field(
        ...,
        description="Analysis end date",
        example="2024-12-31"
    )
    
    interaction_window_hours: int = Field(
        24,
        ge=1,
        le=168,
        description="Time window for interaction effects (hours)",
        example=24
    )
    
    minimum_correlation: float = Field(
        0.1,
        ge=0.05,
        le=0.5,
        description="Minimum correlation threshold for synergy",
        example=0.1
    )