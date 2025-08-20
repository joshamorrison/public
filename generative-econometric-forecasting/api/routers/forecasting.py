"""
Forecasting endpoints for the Econometric Forecasting API.

Provides endpoints for economic forecasting using various methods and models.
"""

import os
import sys
import logging
import time
import uuid
from typing import Dict, Any, List
from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from api.models.request_models import (
    ForecastRequest, BatchForecastRequest, SensitivityTestRequest
)
from api.models.response_models import (
    ForecastResponse, BatchForecastResponse, SensitivityTestResponse,
    ForecastStatus, ModelInfo, IndicatorForecast, ForecastPoint
)
from api.middleware.error_handling import ForecastingError, DataError

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/single", response_model=ForecastResponse)
async def create_forecast(
    request: ForecastRequest,
    background_tasks: BackgroundTasks
) -> ForecastResponse:
    """
    Generate economic forecast for specified indicators.
    
    Creates forecasts using the specified method and model tier.
    Supports multiple economic indicators and various forecasting approaches.
    
    Args:
        request: Forecast configuration and parameters
        background_tasks: FastAPI background tasks
        
    Returns:
        Forecast results with predictions and analysis
    """
    request_id = str(uuid.uuid4())
    start_time = time.time()
    
    logger.info(
        f"Starting forecast request {request_id} for indicators: {request.indicators}",
        extra={"request_id": request_id, "indicators": request.indicators}
    )
    
    try:
        # Initialize response
        response = ForecastResponse(
            success=False,
            status=ForecastStatus.IN_PROGRESS,
            request_id=request_id,
            processing_time=0.0
        )
        
        # Validate request parameters
        await _validate_forecast_request(request)
        
        # Load appropriate forecasting models
        forecaster = await _load_forecaster(request.method, request.model_tier)
        
        # Process each indicator
        forecasts = []
        for indicator in request.indicators:
            try:
                logger.info(f"Processing indicator: {indicator}", extra={"request_id": request_id})
                
                # Get historical data
                historical_data = await _fetch_historical_data(indicator, request.start_date)
                
                # Generate forecast
                forecast_result = await _generate_forecast(
                    forecaster, indicator, historical_data, request
                )
                
                forecasts.append(forecast_result)
                
            except Exception as e:
                logger.error(
                    f"Failed to forecast {indicator}: {str(e)}", 
                    extra={"request_id": request_id, "indicator": indicator}
                )
                # Continue with other indicators
                continue
        
        if not forecasts:
            raise ForecastingError(
                "Failed to generate forecasts for all requested indicators",
                {"requested_indicators": request.indicators}
            )
        
        # Add sentiment analysis if requested
        sentiment_data = None
        if request.include_sentiment:
            sentiment_data = await _generate_sentiment_analysis(request.indicators)
        
        # Generate executive summary if requested
        executive_summary = None
        if request.generate_report:
            executive_summary = await _generate_executive_summary(
                forecasts, sentiment_data, request
            )
        
        # Update response with results
        processing_time = time.time() - start_time
        response.success = True
        response.status = ForecastStatus.SUCCESS
        response.forecasts = forecasts
        response.sentiment_analysis = sentiment_data
        response.executive_summary = executive_summary
        response.processing_time = processing_time
        
        # Add background task for logging/monitoring
        background_tasks.add_task(
            _log_forecast_completion, request_id, len(forecasts), processing_time
        )
        
        logger.info(
            f"Completed forecast request {request_id} in {processing_time:.2f}s",
            extra={"request_id": request_id, "processing_time": processing_time}
        )
        
        return response
        
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(
            f"Forecast request {request_id} failed: {str(e)}",
            extra={"request_id": request_id, "error": str(e)}
        )
        
        return ForecastResponse(
            success=False,
            status=ForecastStatus.FAILED,
            request_id=request_id,
            processing_time=processing_time,
            error=str(e)
        )

@router.post("/batch", response_model=BatchForecastResponse)
async def create_batch_forecast(
    request: BatchForecastRequest,
    background_tasks: BackgroundTasks
) -> BatchForecastResponse:
    """
    Process multiple forecast requests in batch.
    
    Efficiently processes multiple forecasting requests, optionally in parallel.
    Useful for comprehensive economic analysis across multiple indicators.
    
    Args:
        request: Batch of forecast requests
        background_tasks: FastAPI background tasks
        
    Returns:
        Batch processing results with individual forecast responses
    """
    request_id = str(uuid.uuid4())
    start_time = time.time()
    
    logger.info(
        f"Starting batch forecast {request_id} with {len(request.requests)} requests",
        extra={"request_id": request_id, "batch_size": len(request.requests)}
    )
    
    try:
        completed_forecasts = []
        failed_requests = []
        
        # Process requests
        if request.parallel_processing:
            # TODO: Implement parallel processing
            logger.info("Parallel processing not yet implemented, using sequential")
        
        # Sequential processing for now
        for i, forecast_request in enumerate(request.requests):
            try:
                # Create individual forecast
                forecast_response = await create_forecast(forecast_request, background_tasks)
                completed_forecasts.append(forecast_response)
                
            except Exception as e:
                logger.error(f"Batch item {i} failed: {str(e)}")
                failed_requests.append({
                    "index": i,
                    "request": forecast_request.dict(),
                    "error": str(e)
                })
        
        # Generate combined summary if requested
        combined_summary = None
        if request.combine_report and completed_forecasts:
            combined_summary = await _generate_combined_summary(completed_forecasts)
        
        processing_time = time.time() - start_time
        
        return BatchForecastResponse(
            success=len(completed_forecasts) > 0,
            request_id=request_id,
            completed_forecasts=completed_forecasts,
            failed_requests=failed_requests,
            combined_summary=combined_summary,
            processing_summary={
                "total_requests": len(request.requests),
                "completed": len(completed_forecasts),
                "failed": len(failed_requests),
                "success_rate": len(completed_forecasts) / len(request.requests)
            },
            total_processing_time=processing_time
        )
        
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"Batch forecast {request_id} failed: {str(e)}")
        
        return BatchForecastResponse(
            success=False,
            request_id=request_id,
            completed_forecasts=[],
            failed_requests=[{"error": str(e)}],
            total_processing_time=processing_time
        )

@router.post("/sensitivity", response_model=SensitivityTestResponse)
async def run_sensitivity_test(
    request: SensitivityTestRequest,
    background_tasks: BackgroundTasks
) -> SensitivityTestResponse:
    """
    Run sensitivity testing on forecast parameters.
    
    Analyzes how sensitive forecast results are to changes in key parameters.
    Includes optional LLM-based analysis of sensitivity patterns.
    
    Args:
        request: Sensitivity testing configuration
        background_tasks: FastAPI background tasks
        
    Returns:
        Sensitivity analysis results and recommendations
    """
    request_id = str(uuid.uuid4())
    start_time = time.time()
    
    logger.info(
        f"Starting sensitivity test {request_id} for forecast {request.base_forecast_id}",
        extra={"request_id": request_id, "base_forecast": request.base_forecast_id}
    )
    
    try:
        # TODO: Implement sensitivity testing logic
        # For now, return a placeholder response
        
        processing_time = time.time() - start_time
        
        return SensitivityTestResponse(
            success=True,
            request_id=request_id,
            base_forecast_id=request.base_forecast_id,
            sensitivity_results=[],  # TODO: Implement
            overall_robustness="moderate",
            most_sensitive_parameter="horizon",
            recommendations=[
                "Consider testing with different model tiers",
                "Validate results with shorter forecast horizons"
            ],
            processing_time=processing_time
        )
        
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"Sensitivity test {request_id} failed: {str(e)}")
        
        return SensitivityTestResponse(
            success=False,
            request_id=request_id,
            base_forecast_id=request.base_forecast_id,
            sensitivity_results=[],
            overall_robustness="unknown",
            most_sensitive_parameter="unknown",
            recommendations=[],
            processing_time=processing_time,
            error=str(e)
        )

# Helper functions
async def _validate_forecast_request(request: ForecastRequest) -> None:
    """Validate forecast request parameters."""
    if not request.indicators:
        raise ForecastingError("At least one indicator must be specified")
    
    if request.horizon < 1 or request.horizon > 24:
        raise ForecastingError("Forecast horizon must be between 1 and 24 months")

async def _load_forecaster(method: str, model_tier: str):
    """Load appropriate forecasting model."""
    try:
        if method == "foundation":
            from models.foundation_models.huggingface_forecaster import HybridFoundationEnsemble
            return HybridFoundationEnsemble()
        elif method == "neural":
            from models.neural_forecasting import NeuralModelEnsemble
            return NeuralModelEnsemble()
        elif method == "statistical":
            from models.forecasting_models import EconometricForecaster
            return EconometricForecaster()
        else:
            from models.forecasting_models import EconometricForecaster
            return EconometricForecaster()
            
    except ImportError as e:
        raise ForecastingError(f"Could not load forecasting model: {str(e)}")

async def _fetch_historical_data(indicator: str, start_date: str = None):
    """Fetch historical data for indicator."""
    try:
        from data.fred_client import FredDataClient
        
        # Use environment variable or placeholder
        fred_client = FredDataClient()
        return fred_client.fetch_indicator(indicator, start_date=start_date)
        
    except Exception as e:
        raise DataError(f"Failed to fetch data for {indicator}: {str(e)}")

async def _generate_forecast(forecaster, indicator: str, data, request: ForecastRequest) -> IndicatorForecast:
    """Generate forecast for a single indicator."""
    try:
        # This is a placeholder implementation
        # In reality, you'd call the actual forecasting methods
        
        # Mock forecast points
        forecast_points = []
        for i in range(request.horizon):
            forecast_points.append(ForecastPoint(
                date=f"2024-{(i+1):02d}-01",  # Placeholder dates
                value=100.0 + i * 0.5,  # Mock increasing trend
                lower_bound=95.0 + i * 0.5,
                upper_bound=105.0 + i * 0.5,
                confidence=request.confidence_interval
            ))
        
        return IndicatorForecast(
            indicator=indicator,
            current_value=100.0,  # Mock current value
            forecast_points=forecast_points,
            model_info=ModelInfo(
                name="HybridFoundationEnsemble",
                tier=request.model_tier,
                method=request.method,
                version="1.0.0",
                performance_score=0.85
            ),
            metadata={
                "data_points": len(data) if data else 0,
                "forecast_horizon": request.horizon
            }
        )
        
    except Exception as e:
        raise ForecastingError(f"Failed to generate forecast for {indicator}: {str(e)}")

async def _generate_sentiment_analysis(indicators: List[str]):
    """Generate sentiment analysis for indicators."""
    # Placeholder implementation
    return None

async def _generate_executive_summary(forecasts, sentiment_data, request):
    """Generate executive summary of forecast results."""
    # Placeholder implementation
    return None

async def _generate_combined_summary(completed_forecasts):
    """Generate combined summary for batch forecasts."""
    # Placeholder implementation
    return None

async def _log_forecast_completion(request_id: str, forecast_count: int, processing_time: float):
    """Background task for logging forecast completion."""
    logger.info(
        f"Forecast {request_id} metrics: {forecast_count} forecasts in {processing_time:.2f}s",
        extra={
            "request_id": request_id,
            "forecast_count": forecast_count,
            "processing_time": processing_time,
            "avg_time_per_forecast": processing_time / max(forecast_count, 1)
        }
    )