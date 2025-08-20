"""
Response models for the Media Mix Modeling API.

Pydantic models for API response validation and serialization.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from datetime import date as DateType
from enum import Enum

class AnalysisStatus(str, Enum):
    """Status of analysis operations."""
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    FAILED = "failed"
    IN_PROGRESS = "in_progress"

class AttributionResult(BaseModel):
    """Attribution analysis result for a single touchpoint."""
    
    channel: str = Field(description="Marketing channel")
    attribution_value: float = Field(description="Attributed conversion value")
    attribution_percentage: float = Field(description="Attribution percentage")
    confidence_interval: List[float] = Field(description="95% confidence interval")
    touch_count: int = Field(description="Number of touchpoints")
    conversion_rate: float = Field(description="Channel conversion rate")

class AttributionResponse(BaseModel):
    """Response model for attribution analysis."""
    
    success: bool = Field(description="Whether analysis succeeded")
    status: AnalysisStatus = Field(description="Detailed status")
    request_id: str = Field(description="Unique request identifier")
    
    attribution_results: List[AttributionResult] = Field(
        default_factory=list,
        description="Attribution results by channel"
    )
    
    total_conversions: int = Field(description="Total conversions analyzed")
    total_revenue: float = Field(description="Total revenue attributed")
    
    model_performance: Dict[str, float] = Field(
        default_factory=dict,
        description="Attribution model performance metrics"
    )
    
    time_series_data: Optional[List[Dict[str, Any]]] = Field(
        None,
        description="Time series attribution data"
    )
    
    processing_time: float = Field(description="Processing time in seconds")
    error: Optional[str] = Field(None, description="Error message if failed")

class BudgetRecommendation(BaseModel):
    """Budget recommendation for a single channel."""
    
    channel: str = Field(description="Marketing channel")
    current_budget: float = Field(description="Current budget allocation")
    recommended_budget: float = Field(description="Recommended budget allocation")
    budget_change: float = Field(description="Absolute budget change")
    budget_change_percentage: float = Field(description="Percentage budget change")
    expected_roi: float = Field(description="Expected ROI at recommended budget")
    confidence_score: float = Field(description="Confidence in recommendation")

class OptimizationScenario(BaseModel):
    """Budget optimization scenario results."""
    
    scenario_name: str = Field(description="Scenario identifier")
    total_budget: float = Field(description="Total budget for scenario")
    budget_allocation: Dict[str, float] = Field(description="Budget by channel")
    expected_outcomes: Dict[str, float] = Field(description="Expected performance metrics")
    risk_score: float = Field(description="Risk assessment score")

class BudgetOptimizationResponse(BaseModel):
    """Response model for budget optimization."""
    
    success: bool = Field(description="Whether optimization succeeded")
    status: AnalysisStatus = Field(description="Detailed status")
    request_id: str = Field(description="Unique request identifier")
    
    recommendations: List[BudgetRecommendation] = Field(
        description="Budget recommendations by channel"
    )
    
    optimization_scenarios: List[OptimizationScenario] = Field(
        description="Alternative optimization scenarios"
    )
    
    current_performance: Dict[str, float] = Field(
        description="Current performance metrics"
    )
    
    projected_performance: Dict[str, float] = Field(
        description="Projected performance with recommendations"
    )
    
    optimization_summary: str = Field(description="Summary of optimization results")
    constraints_applied: List[str] = Field(description="Constraints that were applied")
    
    processing_time: float = Field(description="Processing time in seconds")
    error: Optional[str] = Field(None, description="Error message if failed")

class ChannelPerformance(BaseModel):
    """Performance metrics for a single channel."""
    
    channel: str = Field(description="Marketing channel")
    impressions: int = Field(description="Total impressions")
    clicks: int = Field(description="Total clicks")
    conversions: int = Field(description="Total conversions")
    revenue: float = Field(description="Total revenue")
    spend: float = Field(description="Total spend")
    cpc: float = Field(description="Cost per click")
    cpa: float = Field(description="Cost per acquisition")
    roas: float = Field(description="Return on ad spend")
    ctr: float = Field(description="Click-through rate")
    conversion_rate: float = Field(description="Conversion rate")

class TrendAnalysis(BaseModel):
    """Trend analysis results."""
    
    metric: str = Field(description="Metric name")
    trend_direction: str = Field(description="Trend direction (up/down/stable)")
    trend_strength: float = Field(description="Trend strength (0-1)")
    period_over_period_change: float = Field(description="Period over period change")
    seasonal_component: Optional[float] = Field(None, description="Seasonal component")

class AnomalyDetection(BaseModel):
    """Anomaly detection results."""
    
    date: DateType = Field(..., description="Date of anomaly")
    metric: str = Field(..., description="Metric with anomaly")
    expected_value: float = Field(..., description="Expected value")
    actual_value: float = Field(..., description="Actual value")
    anomaly_score: float = Field(..., description="Anomaly severity score")
    potential_causes: List[str] = Field(default_factory=list, description="Potential causes")

class PerformanceAnalysisResponse(BaseModel):
    """Response model for performance analysis."""
    
    success: bool = Field(description="Whether analysis succeeded")
    status: AnalysisStatus = Field(description="Detailed status")
    request_id: str = Field(description="Unique request identifier")
    
    channel_performance: List[ChannelPerformance] = Field(
        description="Performance metrics by channel"
    )
    
    overall_metrics: Dict[str, float] = Field(
        description="Overall performance metrics"
    )
    
    trend_analysis: List[TrendAnalysis] = Field(
        default_factory=list,
        description="Trend analysis results"
    )
    
    anomalies: List[AnomalyDetection] = Field(
        default_factory=list,
        description="Detected anomalies"
    )
    
    benchmark_comparison: Optional[Dict[str, float]] = Field(
        None,
        description="Industry benchmark comparison"
    )
    
    insights: List[str] = Field(
        default_factory=list,
        description="Key insights and recommendations"
    )
    
    processing_time: float = Field(description="Processing time in seconds")
    error: Optional[str] = Field(None, description="Error message if failed")

class IncrementalityResult(BaseModel):
    """Incrementality test results for a channel."""
    
    channel: str = Field(description="Tested channel")
    incrementality_percentage: float = Field(description="Incrementality percentage")
    confidence_interval: List[float] = Field(description="95% confidence interval")
    p_value: float = Field(description="Statistical significance p-value")
    statistical_significance: bool = Field(description="Whether result is significant")
    lift_in_conversions: int = Field(description="Incremental conversions")
    lift_in_revenue: float = Field(description="Incremental revenue")

class IncrementalityTestResponse(BaseModel):
    """Response model for incrementality testing."""
    
    success: bool = Field(description="Whether test analysis succeeded")
    status: AnalysisStatus = Field(description="Detailed status")
    request_id: str = Field(description="Unique request identifier")
    
    test_results: List[IncrementalityResult] = Field(
        description="Incrementality results by channel"
    )
    
    test_summary: str = Field(description="Summary of test results")
    methodology: str = Field(description="Test methodology used")
    statistical_power: float = Field(description="Statistical power of the test")
    
    control_group_performance: Dict[str, float] = Field(
        description="Control group performance metrics"
    )
    
    test_group_performance: Dict[str, float] = Field(
        description="Test group performance metrics"
    )
    
    recommendations: List[str] = Field(
        description="Recommendations based on test results"
    )
    
    processing_time: float = Field(description="Processing time in seconds")
    error: Optional[str] = Field(None, description="Error message if failed")

class SaturationCurve(BaseModel):
    """Media saturation curve data."""
    
    channel: str = Field(description="Marketing channel")
    spend_levels: List[float] = Field(description="Spend levels tested")
    response_levels: List[float] = Field(description="Response at each spend level")
    saturation_point: float = Field(description="Saturation point spend level")
    optimal_spend: float = Field(description="Optimal spend level")
    diminishing_returns_threshold: float = Field(description="Diminishing returns threshold")

class SaturationAnalysisResponse(BaseModel):
    """Response model for saturation analysis."""
    
    success: bool = Field(description="Whether analysis succeeded")
    status: AnalysisStatus = Field(description="Detailed status")
    request_id: str = Field(description="Unique request identifier")
    
    saturation_curves: List[SaturationCurve] = Field(
        description="Saturation curves by channel"
    )
    
    summary_insights: List[str] = Field(
        description="Key insights about saturation"
    )
    
    optimization_opportunities: Dict[str, str] = Field(
        description="Optimization opportunities by channel"
    )
    
    processing_time: float = Field(description="Processing time in seconds")
    error: Optional[str] = Field(None, description="Error message if failed")

class SynergyEffect(BaseModel):
    """Cross-channel synergy effect measurement."""
    
    primary_channel: str = Field(description="Primary channel")
    synergy_channel: str = Field(description="Synergy channel")
    synergy_strength: float = Field(description="Synergy effect strength")
    interaction_coefficient: float = Field(description="Statistical interaction coefficient")
    combined_performance_lift: float = Field(description="Performance lift when combined")
    statistical_significance: bool = Field(description="Whether synergy is significant")

class CrossChannelSynergyResponse(BaseModel):
    """Response model for cross-channel synergy analysis."""
    
    success: bool = Field(description="Whether analysis succeeded")
    status: AnalysisStatus = Field(description="Detailed status")
    request_id: str = Field(description="Unique request identifier")
    
    synergy_effects: List[SynergyEffect] = Field(
        description="Synergy effects between channels"
    )
    
    optimal_channel_combinations: List[Dict[str, Any]] = Field(
        description="Optimal channel combinations"
    )
    
    synergy_recommendations: List[str] = Field(
        description="Recommendations for leveraging synergies"
    )
    
    processing_time: float = Field(description="Processing time in seconds")
    error: Optional[str] = Field(None, description="Error message if failed")

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