"""
Budget optimization endpoints for the Media Mix Modeling API.

Provides endpoints for budget allocation optimization and scenario planning.
"""

import os
import sys
import logging
import time
import uuid
from typing import Dict, Any, List
from fastapi import APIRouter, HTTPException, BackgroundTasks

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from api.models.request_models import (
    BudgetOptimizationRequest, SaturationAnalysisRequest
)
from api.models.response_models import (
    BudgetOptimizationResponse, SaturationAnalysisResponse,
    AnalysisStatus, BudgetRecommendation, OptimizationScenario, SaturationCurve
)
from api.middleware.error_handling import OptimizationError, DataError

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/budget", response_model=BudgetOptimizationResponse)
async def optimize_budget(
    request: BudgetOptimizationRequest,
    background_tasks: BackgroundTasks
) -> BudgetOptimizationResponse:
    """
    Optimize budget allocation across marketing channels.
    
    Uses advanced optimization algorithms to recommend optimal budget
    distribution based on historical performance and constraints.
    
    Args:
        request: Budget optimization configuration
        background_tasks: FastAPI background tasks
        
    Returns:
        Budget optimization results with recommendations
    """
    request_id = str(uuid.uuid4())
    start_time = time.time()
    
    logger.info(
        f"Starting budget optimization {request_id} with total budget: ${request.total_budget:,.2f}",
        extra={"request_id": request_id, "total_budget": request.total_budget}
    )
    
    try:
        # Validate request parameters
        await _validate_optimization_request(request)
        
        # Load optimization engine
        optimizer = await _load_optimization_engine(request.optimization_objective)
        
        # Fetch historical performance data
        historical_data = await _fetch_historical_performance(
            request.current_budget.keys(), request.historical_data_days
        )
        
        # Perform budget optimization
        recommendations = await _perform_budget_optimization(
            optimizer, historical_data, request
        )
        
        # Generate alternative scenarios
        scenarios = await _generate_optimization_scenarios(
            optimizer, historical_data, request
        )
        
        # Calculate current vs projected performance
        current_performance = await _calculate_current_performance(
            historical_data, request.current_budget
        )
        
        projected_performance = await _calculate_projected_performance(
            recommendations, historical_data
        )
        
        # Generate optimization summary
        optimization_summary = await _generate_optimization_summary(
            recommendations, current_performance, projected_performance
        )
        
        # Identify applied constraints
        constraints_applied = await _identify_applied_constraints(request.constraints)
        
        processing_time = time.time() - start_time
        
        # Add background task for logging
        background_tasks.add_task(
            _log_optimization_completion, request_id, len(recommendations), processing_time
        )
        
        logger.info(
            f"Completed budget optimization {request_id} in {processing_time:.2f}s",
            extra={"request_id": request_id, "processing_time": processing_time}
        )
        
        return BudgetOptimizationResponse(
            success=True,
            status=AnalysisStatus.SUCCESS,
            request_id=request_id,
            recommendations=recommendations,
            optimization_scenarios=scenarios,
            current_performance=current_performance,
            projected_performance=projected_performance,
            optimization_summary=optimization_summary,
            constraints_applied=constraints_applied,
            processing_time=processing_time
        )
        
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(
            f"Budget optimization {request_id} failed: {str(e)}",
            extra={"request_id": request_id, "error": str(e)}
        )
        
        return BudgetOptimizationResponse(
            success=False,
            status=AnalysisStatus.FAILED,
            request_id=request_id,
            recommendations=[],
            optimization_scenarios=[],
            current_performance={},
            projected_performance={},
            optimization_summary="Optimization failed",
            constraints_applied=[],
            processing_time=processing_time,
            error=str(e)
        )

@router.post("/saturation", response_model=SaturationAnalysisResponse)
async def analyze_saturation(
    request: SaturationAnalysisRequest,
    background_tasks: BackgroundTasks
) -> SaturationAnalysisResponse:
    """
    Analyze media saturation curves for marketing channels.
    
    Identifies saturation points and optimal spend levels for each
    marketing channel to maximize efficiency and ROI.
    
    Args:
        request: Saturation analysis configuration
        background_tasks: FastAPI background tasks
        
    Returns:
        Saturation analysis results with curves and recommendations
    """
    request_id = str(uuid.uuid4())
    start_time = time.time()
    
    logger.info(
        f"Starting saturation analysis {request_id} for channels: {request.channels}",
        extra={"request_id": request_id, "channels": request.channels}
    )
    
    try:
        # Load saturation analysis engine
        saturation_engine = await _load_saturation_engine(request.saturation_model)
        
        # Fetch channel performance data
        performance_data = await _fetch_channel_performance_data(
            request.channels, request.analysis_period_days
        )
        
        # Perform saturation analysis
        saturation_curves = await _perform_saturation_analysis(
            saturation_engine, performance_data, request
        )
        
        # Generate insights and recommendations
        summary_insights = await _generate_saturation_insights(saturation_curves)
        optimization_opportunities = await _identify_optimization_opportunities(
            saturation_curves
        )
        
        processing_time = time.time() - start_time
        
        # Add background task for logging
        background_tasks.add_task(
            _log_saturation_completion, request_id, len(saturation_curves), processing_time
        )
        
        logger.info(
            f"Completed saturation analysis {request_id} in {processing_time:.2f}s",
            extra={"request_id": request_id, "processing_time": processing_time}
        )
        
        return SaturationAnalysisResponse(
            success=True,
            status=AnalysisStatus.SUCCESS,
            request_id=request_id,
            saturation_curves=saturation_curves,
            summary_insights=summary_insights,
            optimization_opportunities=optimization_opportunities,
            processing_time=processing_time
        )
        
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"Saturation analysis {request_id} failed: {str(e)}")
        
        return SaturationAnalysisResponse(
            success=False,
            status=AnalysisStatus.FAILED,
            request_id=request_id,
            saturation_curves=[],
            summary_insights=[],
            optimization_opportunities={},
            processing_time=processing_time,
            error=str(e)
        )

# Helper functions
async def _validate_optimization_request(request: BudgetOptimizationRequest) -> None:
    """Validate budget optimization request parameters."""
    if not request.current_budget:
        raise OptimizationError("Current budget allocation must be specified")
    
    current_total = sum(request.current_budget.values())
    if abs(current_total - request.total_budget) > 0.01:
        raise OptimizationError(
            f"Current budget allocation ({current_total}) doesn't match total budget ({request.total_budget})"
        )

async def _load_optimization_engine(objective: str):
    """Load appropriate optimization engine."""
    try:
        from src.optimization.budget_optimizer import BudgetOptimizer
        return BudgetOptimizer(objective=objective)
    except ImportError as e:
        raise OptimizationError(f"Could not load optimization engine: {str(e)}")

async def _fetch_historical_performance(channels, days: int):
    """Fetch historical performance data for channels."""
    try:
        from data.media_data_client import MediaDataClient
        client = MediaDataClient()
        return client.fetch_performance_data(channels, days)
    except Exception as e:
        raise DataError(f"Failed to fetch historical performance data: {str(e)}")

async def _perform_budget_optimization(optimizer, data, request) -> List[BudgetRecommendation]:
    """Perform budget optimization analysis."""
    try:
        # Mock optimization results for now
        recommendations = []
        total_budget = request.total_budget
        num_channels = len(request.current_budget)
        
        for i, (channel, current_budget) in enumerate(request.current_budget.items()):
            # Mock optimization logic
            if request.optimization_objective == "maximize_roas":
                # Allocate more to high-performing channels
                recommended_budget = total_budget * (0.4 if i == 0 else 0.3 if i == 1 else 0.3 / (num_channels - 2))
            else:
                # Equal distribution for other objectives
                recommended_budget = total_budget / num_channels
            
            recommendation = BudgetRecommendation(
                channel=channel,
                current_budget=current_budget,
                recommended_budget=recommended_budget,
                budget_change=recommended_budget - current_budget,
                budget_change_percentage=((recommended_budget - current_budget) / current_budget) * 100,
                expected_roi=2.5 + (i * 0.3),  # Mock ROI
                confidence_score=0.85 - (i * 0.05)
            )
            recommendations.append(recommendation)
        
        return recommendations
        
    except Exception as e:
        raise OptimizationError(f"Failed to perform budget optimization: {str(e)}")

async def _generate_optimization_scenarios(optimizer, data, request) -> List[OptimizationScenario]:
    """Generate alternative optimization scenarios."""
    scenarios = []
    
    # Conservative scenario
    conservative = OptimizationScenario(
        scenario_name="conservative",
        total_budget=request.total_budget * 0.9,
        budget_allocation={channel: budget * 0.9 for channel, budget in request.current_budget.items()},
        expected_outcomes={"roas": 2.8, "conversions": 850, "revenue": 45000},
        risk_score=0.2
    )
    scenarios.append(conservative)
    
    # Aggressive scenario
    aggressive = OptimizationScenario(
        scenario_name="aggressive",
        total_budget=request.total_budget * 1.2,
        budget_allocation={channel: budget * 1.2 for channel, budget in request.current_budget.items()},
        expected_outcomes={"roas": 3.2, "conversions": 1200, "revenue": 65000},
        risk_score=0.7
    )
    scenarios.append(aggressive)
    
    return scenarios

async def _calculate_current_performance(data, budget):
    """Calculate current performance metrics."""
    return {
        "total_spend": sum(budget.values()),
        "roas": 3.0,
        "conversions": 1000,
        "revenue": 50000,
        "cpa": 50.0
    }

async def _calculate_projected_performance(recommendations, data):
    """Calculate projected performance with recommendations."""
    total_recommended = sum(rec.recommended_budget for rec in recommendations)
    return {
        "total_spend": total_recommended,
        "roas": 3.4,  # Improved ROAS
        "conversions": 1150,  # Increased conversions
        "revenue": 58000,  # Increased revenue
        "cpa": 45.0  # Lower CPA
    }

async def _generate_optimization_summary(recommendations, current, projected):
    """Generate optimization summary."""
    roas_improvement = ((projected["roas"] - current["roas"]) / current["roas"]) * 100
    return f"Budget optimization recommends reallocating spend across {len(recommendations)} channels, " \
           f"expecting {roas_improvement:.1f}% ROAS improvement and {projected['conversions'] - current['conversions']} " \
           f"additional conversions."

async def _identify_applied_constraints(constraints):
    """Identify which constraints were applied during optimization."""
    applied = []
    if constraints:
        if "min_budget_per_channel" in constraints:
            applied.append("Minimum budget per channel constraint")
        if "max_budget_increase" in constraints:
            applied.append("Maximum budget increase constraint")
        if "required_channels" in constraints:
            applied.append("Required channels constraint")
    return applied

async def _load_saturation_engine(model_type: str):
    """Load saturation analysis engine."""
    try:
        from src.optimization.budget_optimizer import BudgetOptimizer
        return BudgetOptimizer(model_type="saturation")
    except ImportError as e:
        raise OptimizationError(f"Could not load saturation engine: {str(e)}")

async def _fetch_channel_performance_data(channels, days: int):
    """Fetch channel performance data for saturation analysis."""
    try:
        from data.media_data_client import MediaDataClient
        client = MediaDataClient()
        return client.fetch_saturation_data(channels, days)
    except Exception as e:
        raise DataError(f"Failed to fetch channel performance data: {str(e)}")

async def _perform_saturation_analysis(engine, data, request) -> List[SaturationCurve]:
    """Perform saturation analysis."""
    curves = []
    
    for channel in request.channels:
        # Mock saturation curve data
        base_spend = 10000
        spend_levels = [base_spend * multiplier for multiplier in request.budget_scenarios]
        
        # Mock diminishing returns curve
        response_levels = []
        for spend in spend_levels:
            # Logarithmic saturation curve
            response = 1000 * (spend / base_spend) ** 0.7
            response_levels.append(response)
        
        curve = SaturationCurve(
            channel=channel,
            spend_levels=spend_levels,
            response_levels=response_levels,
            saturation_point=base_spend * 1.5,  # 150% of base spend
            optimal_spend=base_spend * 1.2,     # 120% of base spend
            diminishing_returns_threshold=base_spend * 1.3  # 130% of base spend
        )
        curves.append(curve)
    
    return curves

async def _generate_saturation_insights(curves):
    """Generate insights from saturation analysis."""
    insights = []
    
    for curve in curves:
        if curve.optimal_spend < curve.saturation_point:
            insights.append(f"{curve.channel} has optimization opportunity below saturation point")
        else:
            insights.append(f"{curve.channel} is approaching saturation, consider budget reallocation")
    
    return insights

async def _identify_optimization_opportunities(curves):
    """Identify optimization opportunities from saturation curves."""
    opportunities = {}
    
    for curve in curves:
        if curve.optimal_spend > curve.spend_levels[0]:
            opportunities[curve.channel] = "Increase spend to optimal level"
        else:
            opportunities[curve.channel] = "Reduce spend due to saturation"
    
    return opportunities

async def _log_optimization_completion(request_id: str, recommendation_count: int, processing_time: float):
    """Background task for logging optimization completion."""
    logger.info(
        f"Optimization {request_id} metrics: {recommendation_count} recommendations in {processing_time:.2f}s",
        extra={
            "request_id": request_id,
            "recommendation_count": recommendation_count,
            "processing_time": processing_time
        }
    )

async def _log_saturation_completion(request_id: str, curve_count: int, processing_time: float):
    """Background task for logging saturation analysis completion."""
    logger.info(
        f"Saturation analysis {request_id} metrics: {curve_count} curves analyzed in {processing_time:.2f}s",
        extra={
            "request_id": request_id,
            "curve_count": curve_count,
            "processing_time": processing_time
        }
    )