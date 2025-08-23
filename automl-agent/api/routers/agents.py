from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Dict, Any
from ..models.request_models import AgentExecutionRequest
from ..models.response_models import AgentInfo
import sys
import os

# Add src to path so we can import from src
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

router = APIRouter()

# This would be imported from the main agent registry
# For now, we'll define a basic structure
AVAILABLE_AGENTS = {
    "eda": {
        "name": "EDA Agent",
        "description": "Exploratory Data Analysis agent",
        "capabilities": ["data_profiling", "visualization", "statistical_analysis"],
        "supported_task_types": ["analysis", "exploration"],
        "performance_metrics": ["processing_time", "insights_generated"],
        "quality_indicators": ["data_coverage", "insight_quality"]
    },
    "classification": {
        "name": "Classification Agent", 
        "description": "Machine learning classification agent",
        "capabilities": ["binary_classification", "multiclass_classification", "model_training"],
        "supported_task_types": ["classification", "prediction"],
        "performance_metrics": ["accuracy", "f1_score", "precision", "recall"],
        "quality_indicators": ["model_performance", "feature_importance"]
    },
    "hyperparameter_tuning": {
        "name": "Hyperparameter Tuning Agent",
        "description": "Model optimization and hyperparameter tuning agent",
        "capabilities": ["bayesian_optimization", "grid_search", "random_search"],
        "supported_task_types": ["optimization", "model_tuning"],
        "performance_metrics": ["optimization_score", "convergence_rate"],
        "quality_indicators": ["parameter_stability", "improvement_rate"]
    },
    "ensemble": {
        "name": "Ensemble Agent",
        "description": "Model ensemble and combination agent",
        "capabilities": ["voting_ensembles", "stacking", "blending"],
        "supported_task_types": ["ensemble_learning", "model_combination"],
        "performance_metrics": ["ensemble_accuracy", "diversity_score"],
        "quality_indicators": ["ensemble_stability", "individual_contribution"]
    },
    "time_series": {
        "name": "Time Series Agent",
        "description": "Time series analysis and forecasting agent", 
        "capabilities": ["forecasting", "trend_analysis", "seasonality_detection"],
        "supported_task_types": ["time_series_prediction", "forecasting"],
        "performance_metrics": ["mae", "mse", "mape"],
        "quality_indicators": ["forecast_accuracy", "trend_capture"]
    }
}

@router.get("/agents")
async def list_agents() -> Dict[str, AgentInfo]:
    """List all available agents with their capabilities"""
    agent_info = {}
    for agent_id, info in AVAILABLE_AGENTS.items():
        agent_info[agent_id] = AgentInfo(**info)
    return {"agents": agent_info}

@router.get("/agents/{agent_id}")
async def get_agent_info(agent_id: str) -> AgentInfo:
    """Get detailed information about a specific agent"""
    if agent_id not in AVAILABLE_AGENTS:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found")
    
    return AgentInfo(**AVAILABLE_AGENTS[agent_id])

@router.post("/agents/{agent_id}/execute")
async def execute_agent(
    agent_id: str,
    request: AgentExecutionRequest,
    background_tasks: BackgroundTasks
):
    """Execute a specific agent with given parameters"""
    if agent_id not in AVAILABLE_AGENTS:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found")
    
    # This would integrate with the actual agent execution system
    # For now, return a placeholder response
    return {
        "job_id": f"job_{agent_id}_placeholder",
        "agent_id": agent_id,
        "status": "queued",
        "message": f"Agent {agent_id} execution queued"
    }