"""
Performance analysis endpoints for the Media Mix Modeling API.

Provides endpoints for campaign performance monitoring and insights.
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

from api.models.request_models import PerformanceAnalysisRequest
from api.models.response_models import (
    PerformanceAnalysisResponse, AnalysisStatus, ChannelPerformance,
    TrendAnalysis, AnomalyDetection
)
from api.middleware.error_handling import DataError

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/analyze", response_model=PerformanceAnalysisResponse)
async def analyze_performance(
    request: PerformanceAnalysisRequest,
    background_tasks: BackgroundTasks
) -> PerformanceAnalysisResponse:
    """
    Analyze campaign and channel performance.
    
    Provides comprehensive performance analysis including metrics,
    trends, anomalies, and actionable insights.
    
    Args:
        request: Performance analysis configuration
        background_tasks: FastAPI background tasks
        
    Returns:
        Performance analysis results with insights and recommendations
    """
    request_id = str(uuid.uuid4())
    start_time = time.time()
    
    logger.info(
        f"Starting performance analysis {request_id}",
        extra={"request_id": request_id, "channels": request.channels}
    )
    
    try:
        # Fetch performance data
        performance_data = await _fetch_performance_data(request)
        
        # Calculate channel performance metrics
        channel_performance = await _calculate_channel_performance(
            performance_data, request.metrics
        )
        
        # Calculate overall metrics
        overall_metrics = await _calculate_overall_metrics(channel_performance)
        
        # Perform trend analysis if requested
        trend_analysis = []
        if request.include_trends:
            trend_analysis = await _perform_trend_analysis(
                performance_data, request.granularity
            )
        
        # Detect anomalies if requested
        anomalies = []
        if request.anomaly_detection:
            anomalies = await _detect_anomalies(performance_data, request.metrics)
        
        # Get benchmark comparison if requested
        benchmark_comparison = None
        if request.benchmark_comparison:
            benchmark_comparison = await _get_benchmark_comparison(
                channel_performance, request.channels
            )
        
        # Generate insights and recommendations
        insights = await _generate_performance_insights(
            channel_performance, trend_analysis, anomalies
        )
        
        processing_time = time.time() - start_time
        
        # Add background task for logging
        background_tasks.add_task(
            _log_performance_completion, request_id, len(channel_performance), processing_time
        )
        
        logger.info(
            f"Completed performance analysis {request_id} in {processing_time:.2f}s",
            extra={"request_id": request_id, "processing_time": processing_time}
        )
        
        return PerformanceAnalysisResponse(
            success=True,
            status=AnalysisStatus.SUCCESS,
            request_id=request_id,
            channel_performance=channel_performance,
            overall_metrics=overall_metrics,
            trend_analysis=trend_analysis,
            anomalies=anomalies,
            benchmark_comparison=benchmark_comparison,
            insights=insights,
            processing_time=processing_time
        )
        
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(
            f"Performance analysis {request_id} failed: {str(e)}",
            extra={"request_id": request_id, "error": str(e)}
        )
        
        return PerformanceAnalysisResponse(
            success=False,
            status=AnalysisStatus.FAILED,
            request_id=request_id,
            channel_performance=[],
            overall_metrics={},
            processing_time=processing_time,
            error=str(e)
        )

@router.get("/dashboard")
async def get_performance_dashboard(
    channels: str = None,
    date_range: int = 30
):
    """
    Get performance dashboard data.
    
    Returns key performance metrics for dashboard visualization.
    
    Args:
        channels: Comma-separated list of channels (optional)
        date_range: Number of days to include (default: 30)
        
    Returns:
        Dashboard performance data
    """
    try:
        # Parse channels if provided
        channel_list = None
        if channels:
            channel_list = [ch.strip() for ch in channels.split(",")]
        
        # Fetch dashboard data
        dashboard_data = await _fetch_dashboard_data(channel_list, date_range)
        
        return {
            "success": True,
            "data": dashboard_data,
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Dashboard data fetch failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch dashboard data: {str(e)}"
        )

@router.get("/kpis")
async def get_key_performance_indicators():
    """
    Get key performance indicators across all channels.
    
    Returns:
        Current KPIs and performance status
    """
    try:
        # Fetch current KPIs
        kpis = await _fetch_current_kpis()
        
        return {
            "success": True,
            "kpis": kpis,
            "last_updated": time.time()
        }
        
    except Exception as e:
        logger.error(f"KPI fetch failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch KPIs: {str(e)}"
        )

# Helper functions
async def _fetch_performance_data(request: PerformanceAnalysisRequest):
    """Fetch performance data based on request parameters."""
    try:
        from data.media_data_client import MediaDataClient
        client = MediaDataClient()
        
        data = client.fetch_campaign_performance(
            start_date=request.start_date,
            end_date=request.end_date,
            channels=request.channels,
            campaigns=request.campaigns,
            granularity=request.granularity
        )
        
        return data
        
    except Exception as e:
        raise DataError(f"Failed to fetch performance data: {str(e)}")

async def _calculate_channel_performance(data, metrics: List[str]) -> List[ChannelPerformance]:
    """Calculate performance metrics by channel."""
    try:
        # Mock channel performance data
        channels = ["search", "social", "display", "video"]
        channel_performance = []
        
        for i, channel in enumerate(channels):
            performance = ChannelPerformance(
                channel=channel,
                impressions=100000 * (i + 1),
                clicks=5000 * (i + 1),
                conversions=250 * (i + 1),
                revenue=12500.0 * (i + 1),
                spend=5000.0 * (i + 1),
                cpc=1.0 + (i * 0.2),
                cpa=20.0 + (i * 5),
                roas=2.5 + (i * 0.3),
                ctr=0.05 + (i * 0.01),
                conversion_rate=0.05 + (i * 0.005)
            )
            channel_performance.append(performance)
        
        return channel_performance
        
    except Exception as e:
        raise DataError(f"Failed to calculate channel performance: {str(e)}")

async def _calculate_overall_metrics(channel_performance: List[ChannelPerformance]) -> Dict[str, float]:
    """Calculate overall performance metrics."""
    total_impressions = sum(ch.impressions for ch in channel_performance)
    total_clicks = sum(ch.clicks for ch in channel_performance)
    total_conversions = sum(ch.conversions for ch in channel_performance)
    total_revenue = sum(ch.revenue for ch in channel_performance)
    total_spend = sum(ch.spend for ch in channel_performance)
    
    return {
        "total_impressions": total_impressions,
        "total_clicks": total_clicks,
        "total_conversions": total_conversions,
        "total_revenue": total_revenue,
        "total_spend": total_spend,
        "overall_roas": total_revenue / total_spend if total_spend > 0 else 0,
        "overall_ctr": total_clicks / total_impressions if total_impressions > 0 else 0,
        "overall_conversion_rate": total_conversions / total_clicks if total_clicks > 0 else 0,
        "overall_cpa": total_spend / total_conversions if total_conversions > 0 else 0
    }

async def _perform_trend_analysis(data, granularity: str) -> List[TrendAnalysis]:
    """Perform trend analysis on performance data."""
    trends = []
    
    # Mock trend analysis
    metrics = ["impressions", "clicks", "conversions", "roas"]
    
    for metric in metrics:
        trend = TrendAnalysis(
            metric=metric,
            trend_direction="up" if metric in ["impressions", "conversions"] else "stable",
            trend_strength=0.7 if metric == "conversions" else 0.3,
            period_over_period_change=0.15 if metric == "conversions" else 0.05,
            seasonal_component=0.1 if metric == "impressions" else None
        )
        trends.append(trend)
    
    return trends

async def _detect_anomalies(data, metrics: List[str]) -> List[AnomalyDetection]:
    """Detect performance anomalies."""
    anomalies = []
    
    # Mock anomaly detection
    from datetime import date, timedelta
    
    anomaly = AnomalyDetection(
        date=date.today() - timedelta(days=2),
        metric="clicks",
        expected_value=5000.0,
        actual_value=3000.0,
        anomaly_score=0.8,
        potential_causes=["Campaign budget exhausted", "Ad disapproval", "Weekend effect"]
    )
    anomalies.append(anomaly)
    
    return anomalies

async def _get_benchmark_comparison(channel_performance: List[ChannelPerformance], channels) -> Dict[str, float]:
    """Get industry benchmark comparison."""
    # Mock benchmark data
    return {
        "industry_avg_ctr": 0.045,
        "industry_avg_conversion_rate": 0.035,
        "industry_avg_cpa": 35.0,
        "industry_avg_roas": 2.8,
        "percentile_ranking": 75.0  # Performance percentile
    }

async def _generate_performance_insights(
    channel_performance: List[ChannelPerformance],
    trends: List[TrendAnalysis],
    anomalies: List[AnomalyDetection]
) -> List[str]:
    """Generate performance insights and recommendations."""
    insights = []
    
    # Analyze best performing channel
    best_channel = max(channel_performance, key=lambda x: x.roas)
    insights.append(f"{best_channel.channel} is the top performing channel with {best_channel.roas:.2f} ROAS")
    
    # Analyze trends
    positive_trends = [t for t in trends if t.trend_direction == "up" and t.trend_strength > 0.5]
    if positive_trends:
        insights.append(f"Strong positive trends detected in {', '.join([t.metric for t in positive_trends])}")
    
    # Analyze anomalies
    if anomalies:
        insights.append(f"Performance anomalies detected: {len(anomalies)} metrics require attention")
    
    # Optimization recommendations
    low_performing = [ch for ch in channel_performance if ch.roas < 2.0]
    if low_performing:
        insights.append(f"Consider optimizing or reducing spend in {', '.join([ch.channel for ch in low_performing])}")
    
    return insights

async def _fetch_dashboard_data(channels, date_range: int):
    """Fetch dashboard data."""
    # Mock dashboard data
    return {
        "summary": {
            "total_spend": 50000,
            "total_revenue": 150000,
            "overall_roas": 3.0,
            "total_conversions": 2500
        },
        "top_channels": [
            {"channel": "search", "roas": 3.5, "spend": 20000},
            {"channel": "social", "roas": 2.8, "spend": 15000},
            {"channel": "display", "roas": 2.2, "spend": 15000}
        ],
        "recent_performance": [
            {"date": "2024-08-15", "conversions": 85, "revenue": 4250},
            {"date": "2024-08-16", "conversions": 92, "revenue": 4600},
            {"date": "2024-08-17", "conversions": 78, "revenue": 3900}
        ]
    }

async def _fetch_current_kpis():
    """Fetch current key performance indicators."""
    # Mock KPI data
    return {
        "overall_roas": {"value": 3.2, "change": 0.15, "status": "good"},
        "total_conversions": {"value": 2500, "change": 0.08, "status": "good"},
        "avg_cpa": {"value": 25.0, "change": -0.12, "status": "excellent"},
        "conversion_rate": {"value": 0.045, "change": 0.05, "status": "good"},
        "budget_utilization": {"value": 0.85, "change": 0.02, "status": "good"}
    }

async def _log_performance_completion(request_id: str, channel_count: int, processing_time: float):
    """Background task for logging performance analysis completion."""
    logger.info(
        f"Performance analysis {request_id} metrics: {channel_count} channels analyzed in {processing_time:.2f}s",
        extra={
            "request_id": request_id,
            "channel_count": channel_count,
            "processing_time": processing_time
        }
    )