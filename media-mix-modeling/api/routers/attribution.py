"""
Attribution analysis endpoints for the Media Mix Modeling API.

Provides endpoints for multi-touch attribution, model comparison, and attribution insights.
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
    AttributionRequest, IncrementalityTestRequest, CrossChannelSynergyRequest
)
from api.models.response_models import (
    AttributionResponse, IncrementalityTestResponse, CrossChannelSynergyResponse,
    AnalysisStatus, AttributionResult
)
from api.middleware.error_handling import AttributionError, DataError

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/analyze", response_model=AttributionResponse)
async def analyze_attribution(
    request: AttributionRequest,
    background_tasks: BackgroundTasks
) -> AttributionResponse:
    """
    Perform multi-touch attribution analysis.
    
    Analyzes customer journey touchpoints to determine the contribution
    of each marketing channel to conversions and revenue.
    
    Args:
        request: Attribution analysis configuration
        background_tasks: FastAPI background tasks
        
    Returns:
        Attribution analysis results with channel contributions
    """
    request_id = str(uuid.uuid4())
    start_time = time.time()
    
    logger.info(
        f"Starting attribution analysis {request_id} for channels: {request.channels}",
        extra={"request_id": request_id, "channels": request.channels}
    )
    
    try:
        # Initialize response
        response = AttributionResponse(
            success=False,
            status=AnalysisStatus.IN_PROGRESS,
            request_id=request_id,
            total_conversions=0,
            total_revenue=0.0,
            processing_time=0.0
        )
        
        # Validate request parameters
        await _validate_attribution_request(request)
        
        # Load attribution engine
        attribution_engine = await _load_attribution_engine(request.attribution_model)
        
        # Fetch customer journey data
        journey_data = await _fetch_journey_data(
            request.start_date, request.end_date, request.channels
        )
        
        # Perform attribution analysis
        attribution_results = await _perform_attribution_analysis(
            attribution_engine, journey_data, request
        )
        
        # Calculate summary metrics
        total_conversions = sum(result.touch_count for result in attribution_results)
        total_revenue = sum(result.attribution_value for result in attribution_results)
        
        # Generate time series data if needed
        time_series_data = None
        if request.granularity:
            time_series_data = await _generate_time_series_attribution(
                attribution_engine, journey_data, request
            )
        
        # Calculate model performance metrics
        model_performance = await _calculate_model_performance(
            attribution_engine, journey_data, attribution_results
        )
        
        processing_time = time.time() - start_time
        
        # Update response with results
        response.success = True
        response.status = AnalysisStatus.SUCCESS
        response.attribution_results = attribution_results
        response.total_conversions = total_conversions
        response.total_revenue = total_revenue
        response.model_performance = model_performance
        response.time_series_data = time_series_data
        response.processing_time = processing_time
        
        # Add background task for logging/monitoring
        background_tasks.add_task(
            _log_attribution_completion, request_id, len(attribution_results), processing_time
        )
        
        logger.info(
            f"Completed attribution analysis {request_id} in {processing_time:.2f}s",
            extra={"request_id": request_id, "processing_time": processing_time}
        )
        
        return response
        
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(
            f"Attribution analysis {request_id} failed: {str(e)}",
            extra={"request_id": request_id, "error": str(e)}
        )
        
        return AttributionResponse(
            success=False,
            status=AnalysisStatus.FAILED,
            request_id=request_id,
            total_conversions=0,
            total_revenue=0.0,
            processing_time=processing_time,
            error=str(e)
        )

@router.post("/incrementality", response_model=IncrementalityTestResponse)
async def test_incrementality(
    request: IncrementalityTestRequest,
    background_tasks: BackgroundTasks
) -> IncrementalityTestResponse:
    """
    Perform incrementality testing for marketing channels.
    
    Tests the true incremental impact of marketing channels using
    controlled experiments and statistical analysis.
    
    Args:
        request: Incrementality test configuration
        background_tasks: FastAPI background tasks
        
    Returns:
        Incrementality test results with statistical significance
    """
    request_id = str(uuid.uuid4())
    start_time = time.time()
    
    logger.info(
        f"Starting incrementality test {request_id} for channels: {request.test_channels}",
        extra={"request_id": request_id, "channels": request.test_channels}
    )
    
    try:
        # Load incrementality testing engine
        test_engine = await _load_incrementality_engine()
        
        # Fetch control and test group data
        control_data = await _fetch_control_group_data(request)
        test_data = await _fetch_test_group_data(request)
        
        # Perform incrementality analysis
        test_results = await _perform_incrementality_analysis(
            test_engine, control_data, test_data, request
        )
        
        # Calculate statistical power and significance
        statistical_power = await _calculate_statistical_power(
            control_data, test_data, request
        )
        
        # Generate recommendations
        recommendations = await _generate_incrementality_recommendations(test_results)
        
        processing_time = time.time() - start_time
        
        return IncrementalityTestResponse(
            success=True,
            status=AnalysisStatus.SUCCESS,
            request_id=request_id,
            test_results=test_results,
            test_summary=f"Incrementality test completed for {len(request.test_channels)} channels",
            methodology="Difference-in-Differences with Geographic Controls",
            statistical_power=statistical_power,
            control_group_performance={},  # Placeholder
            test_group_performance={},     # Placeholder
            recommendations=recommendations,
            processing_time=processing_time
        )
        
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"Incrementality test {request_id} failed: {str(e)}")
        
        return IncrementalityTestResponse(
            success=False,
            status=AnalysisStatus.FAILED,
            request_id=request_id,
            test_results=[],
            test_summary="",
            methodology="",
            statistical_power=0.0,
            control_group_performance={},
            test_group_performance={},
            recommendations=[],
            processing_time=processing_time,
            error=str(e)
        )

@router.post("/synergy", response_model=CrossChannelSynergyResponse)
async def analyze_cross_channel_synergy(
    request: CrossChannelSynergyRequest,
    background_tasks: BackgroundTasks
) -> CrossChannelSynergyResponse:
    """
    Analyze cross-channel synergy effects.
    
    Identifies interaction effects between marketing channels and
    optimal channel combinations for maximum performance.
    
    Args:
        request: Cross-channel synergy analysis configuration
        background_tasks: FastAPI background tasks
        
    Returns:
        Synergy analysis results with interaction effects
    """
    request_id = str(uuid.uuid4())
    start_time = time.time()
    
    logger.info(
        f"Starting synergy analysis {request_id} for primary channel: {request.primary_channel}",
        extra={"request_id": request_id, "primary_channel": request.primary_channel}
    )
    
    try:
        # Load synergy analysis engine
        synergy_engine = await _load_synergy_engine()
        
        # Fetch interaction data
        interaction_data = await _fetch_interaction_data(request)
        
        # Perform synergy analysis
        synergy_effects = await _perform_synergy_analysis(
            synergy_engine, interaction_data, request
        )
        
        # Find optimal combinations
        optimal_combinations = await _find_optimal_combinations(
            synergy_engine, synergy_effects, request
        )
        
        # Generate recommendations
        recommendations = await _generate_synergy_recommendations(
            synergy_effects, optimal_combinations
        )
        
        processing_time = time.time() - start_time
        
        return CrossChannelSynergyResponse(
            success=True,
            status=AnalysisStatus.SUCCESS,
            request_id=request_id,
            synergy_effects=synergy_effects,
            optimal_channel_combinations=optimal_combinations,
            synergy_recommendations=recommendations,
            processing_time=processing_time
        )
        
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"Synergy analysis {request_id} failed: {str(e)}")
        
        return CrossChannelSynergyResponse(
            success=False,
            status=AnalysisStatus.FAILED,
            request_id=request_id,
            synergy_effects=[],
            optimal_channel_combinations=[],
            synergy_recommendations=[],
            processing_time=processing_time,
            error=str(e)
        )

# Helper functions
async def _validate_attribution_request(request: AttributionRequest) -> None:
    """Validate attribution request parameters."""
    if not request.channels:
        raise AttributionError("At least one channel must be specified")
    
    if request.conversion_window_days < 1 or request.conversion_window_days > 365:
        raise AttributionError("Conversion window must be between 1 and 365 days")

async def _load_attribution_engine(model_type: str):
    """Load appropriate attribution engine."""
    try:
        from src.attribution.attribution_engine import AttributionEngine
        return AttributionEngine(model_type=model_type)
    except ImportError as e:
        raise AttributionError(f"Could not load attribution engine: {str(e)}")

async def _fetch_journey_data(start_date, end_date, channels):
    """Fetch customer journey data."""
    try:
        from data.media_data_client import MediaDataClient
        client = MediaDataClient()
        return client.fetch_journey_data(start_date, end_date, channels)
    except Exception as e:
        raise DataError(f"Failed to fetch journey data: {str(e)}")

async def _perform_attribution_analysis(engine, data, request) -> List[AttributionResult]:
    """Perform attribution analysis."""
    try:
        # Mock attribution results for now
        results = []
        for i, channel in enumerate(request.channels):
            result = AttributionResult(
                channel=channel,
                attribution_value=10000.0 * (i + 1),  # Mock values
                attribution_percentage=100.0 / len(request.channels),
                confidence_interval=[0.8, 1.2],
                touch_count=100 * (i + 1),
                conversion_rate=0.05 + (i * 0.01)
            )
            results.append(result)
        
        return results
        
    except Exception as e:
        raise AttributionError(f"Failed to perform attribution analysis: {str(e)}")

async def _generate_time_series_attribution(engine, data, request):
    """Generate time series attribution data."""
    # Placeholder implementation
    return []

async def _calculate_model_performance(engine, data, results):
    """Calculate attribution model performance metrics."""
    return {
        "accuracy": 0.85,
        "precision": 0.82,
        "recall": 0.88,
        "f1_score": 0.85
    }

async def _load_incrementality_engine():
    """Load incrementality testing engine."""
    try:
        from src.attribution.attribution_engine import AttributionEngine
        return AttributionEngine(model_type="incrementality")
    except ImportError as e:
        raise AttributionError(f"Could not load incrementality engine: {str(e)}")

async def _fetch_control_group_data(request):
    """Fetch control group data for incrementality testing."""
    # Placeholder implementation
    return {}

async def _fetch_test_group_data(request):
    """Fetch test group data for incrementality testing."""
    # Placeholder implementation
    return {}

async def _perform_incrementality_analysis(engine, control_data, test_data, request):
    """Perform incrementality analysis."""
    # Placeholder implementation
    return []

async def _calculate_statistical_power(control_data, test_data, request):
    """Calculate statistical power of incrementality test."""
    return 0.8

async def _generate_incrementality_recommendations(test_results):
    """Generate recommendations based on incrementality results."""
    return ["Continue testing with larger sample size", "Focus on statistically significant channels"]

async def _load_synergy_engine():
    """Load synergy analysis engine."""
    try:
        from src.attribution.attribution_engine import AttributionEngine
        return AttributionEngine(model_type="synergy")
    except ImportError as e:
        raise AttributionError(f"Could not load synergy engine: {str(e)}")

async def _fetch_interaction_data(request):
    """Fetch cross-channel interaction data."""
    # Placeholder implementation
    return {}

async def _perform_synergy_analysis(engine, data, request):
    """Perform synergy analysis."""
    # Placeholder implementation
    return []

async def _find_optimal_combinations(engine, effects, request):
    """Find optimal channel combinations."""
    # Placeholder implementation
    return []

async def _generate_synergy_recommendations(effects, combinations):
    """Generate synergy-based recommendations."""
    return ["Coordinate campaigns across channels", "Time channel activations for maximum synergy"]

async def _log_attribution_completion(request_id: str, result_count: int, processing_time: float):
    """Background task for logging attribution completion."""
    logger.info(
        f"Attribution {request_id} metrics: {result_count} channels analyzed in {processing_time:.2f}s",
        extra={
            "request_id": request_id,
            "result_count": result_count,
            "processing_time": processing_time,
            "avg_time_per_channel": processing_time / max(result_count, 1)
        }
    )