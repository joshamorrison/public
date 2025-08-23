from fastapi import FastAPI, HTTPException, BackgroundTasks, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
import asyncio
import uuid
import json
import io
import pandas as pd
from datetime import datetime
import sys
import os

# Add src to path so we can import from src
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.agents.eda_agent import EDAAgent
from src.agents.classification_agent import ClassificationAgent
from src.agents.regression_agent import RegressionAgent
from src.agents.nlp_agent import NLPAgent
from src.agents.computer_vision_agent import ComputerVisionAgent
from src.agents.time_series_agent import TimeSeriesAgent
from src.agents.hyperparameter_tuning_agent import HyperparameterTuningAgent
from src.agents.ensemble_agent import EnsembleAgent
from src.agents.data_hygiene_agent import DataHygieneAgent
from src.agents.feature_engineering_agent import FeatureEngineeringAgent
from src.agents.qa_agent import QualityAssuranceAgent
from src.agents.base_agent import TaskContext
from src.pipelines.data_pipeline import DataProcessingPipeline
from src.automl_platform import AutoMLPlatform

app = FastAPI(
    title="AutoML Agent Platform API",
    description="Revolutionary multi-agent AutoML platform with specialized collaborative intelligence",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response Models
class TaskRequest(BaseModel):
    task_description: str = Field(..., description="Natural language description of the ML task")
    target_column: Optional[str] = Field(None, description="Target column for supervised learning")
    task_type: Optional[str] = Field(None, description="Override automatic task detection")
    quality_threshold: float = Field(0.8, description="Minimum quality threshold for completion")
    max_iterations: int = Field(5, description="Maximum refinement iterations")

class AgentExecutionRequest(BaseModel):
    agent_type: str = Field(..., description="Type of agent to execute")
    task_description: str = Field(..., description="Task description")
    parameters: Optional[Dict[str, Any]] = Field({}, description="Agent-specific parameters")

class JobStatus(BaseModel):
    job_id: str
    status: str
    progress: float
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: datetime
    updated_at: datetime

class WorkflowRequest(BaseModel):
    task_description: str
    agents: List[str] = Field(..., description="List of agent types to include in workflow")
    quality_threshold: float = Field(0.8)
    collaboration_mode: str = Field("sequential", description="sequential or parallel")

# In-memory job storage (replace with Redis in production)
jobs: Dict[str, JobStatus] = {}

# Agent instances - now using real implementations
agents = {
    "eda": EDAAgent(),
    "data_hygiene": DataHygieneAgent(),
    "feature_engineering": FeatureEngineeringAgent(),
    "classification": ClassificationAgent(),
    "regression": RegressionAgent(),
    "nlp": NLPAgent(),
    "computer_vision": ComputerVisionAgent(),
    "time_series": TimeSeriesAgent(),
    "hyperparameter_tuning": HyperparameterTuningAgent(),
    "ensemble": EnsembleAgent(),
    "qa": QualityAssuranceAgent()
}

@app.get("/")
async def root():
    return {
        "message": "AutoML Agent Platform API",
        "version": "1.0.0",
        "agents": list(agents.keys()),
        "endpoints": ["/docs", "/agents", "/execute", "/workflow", "/jobs"]
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now()}

@app.get("/agents")
async def list_agents():
    agent_info = {}
    for name, agent in agents.items():
        agent_info[name] = {
            "name": agent.__class__.__name__,
            "description": getattr(agent, 'description', 'No description available'),
            "capabilities": getattr(agent, 'capabilities', []),
            "task_types": getattr(agent, 'supported_task_types', [])
        }
    return {"agents": agent_info}

@app.post("/upload-data")
async def upload_data(file: UploadFile = File(...)):
    try:
        if file.filename.endswith('.csv'):
            contents = await file.read()
            df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        elif file.filename.endswith(('.xlsx', '.xls')):
            contents = await file.read()
            df = pd.read_excel(io.BytesIO(contents))
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format. Use CSV or Excel.")
        
        # Store data temporarily (implement proper storage in production)
        data_id = str(uuid.uuid4())
        # In production, store in database or file system
        
        return {
            "data_id": data_id,
            "filename": file.filename,
            "shape": df.shape,
            "columns": df.columns.tolist(),
            "dtypes": df.dtypes.to_dict(),
            "sample": df.head().to_dict('records')
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

@app.post("/execute/{agent_type}")
async def execute_agent(
    agent_type: str, 
    request: AgentExecutionRequest,
    background_tasks: BackgroundTasks
):
    if agent_type not in agents:
        raise HTTPException(status_code=404, detail=f"Agent type '{agent_type}' not found")
    
    job_id = str(uuid.uuid4())
    jobs[job_id] = JobStatus(
        job_id=job_id,
        status="queued",
        progress=0.0,
        created_at=datetime.now(),
        updated_at=datetime.now()
    )
    
    background_tasks.add_task(
        execute_agent_task, 
        job_id, 
        agent_type, 
        request.task_description, 
        request.parameters
    )
    
    return {"job_id": job_id, "status": "queued"}

async def execute_agent_task(job_id: str, agent_type: str, task_description: str, parameters: Dict[str, Any]):
    try:
        jobs[job_id].status = "running"
        jobs[job_id].progress = 0.1
        jobs[job_id].updated_at = datetime.now()
        
        agent = agents[agent_type]
        
        # Create TaskContext for the agent
        context = TaskContext(
            task_id=job_id,
            user_input=task_description,
            dataset_info=parameters.get("dataset_info"),
            constraints=parameters.get("constraints"),
            preferences=parameters.get("preferences")
        )
        
        jobs[job_id].progress = 0.3
        jobs[job_id].updated_at = datetime.now()
        
        # Execute agent
        result = await asyncio.to_thread(agent.execute_task, context)
        
        jobs[job_id].status = "completed"
        jobs[job_id].progress = 1.0
        jobs[job_id].result = {
            "success": result.success,
            "message": result.message,
            "data": result.data,
            "recommendations": result.recommendations
        }
        jobs[job_id].updated_at = datetime.now()
        
    except Exception as e:
        jobs[job_id].status = "failed"
        jobs[job_id].error = str(e)
        jobs[job_id].updated_at = datetime.now()

def generate_demo_data(agent_type: str) -> pd.DataFrame:
    """Generate appropriate demo data based on agent type"""
    if agent_type in ["classification", "regression"]:
        from sklearn.datasets import make_classification, make_regression
        if agent_type == "classification":
            X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
        else:
            X, y = make_regression(n_samples=1000, n_features=10, random_state=42)
        df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)])
        df['target'] = y
    elif agent_type == "nlp":
        # Generate sample text data
        texts = [
            "This product is amazing!",
            "Terrible service, very disappointed",
            "Good quality for the price",
            "Not worth the money",
            "Excellent customer support"
        ] * 200
        df = pd.DataFrame({'text': texts, 'sentiment': ['positive', 'negative', 'positive', 'negative', 'positive'] * 200})
    elif agent_type == "time_series":
        import numpy as np
        dates = pd.date_range('2023-01-01', periods=365, freq='D')
        values = np.sin(np.arange(365) * 2 * np.pi / 365) + np.random.normal(0, 0.1, 365)
        df = pd.DataFrame({'date': dates, 'value': values})
    else:
        # Default data
        from sklearn.datasets import make_classification
        X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
        df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)])
        df['target'] = y
    
    return df

@app.post("/workflow")
async def execute_workflow(request: WorkflowRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    jobs[job_id] = JobStatus(
        job_id=job_id,
        status="queued",
        progress=0.0,
        created_at=datetime.now(),
        updated_at=datetime.now()
    )
    
    background_tasks.add_task(execute_workflow_task, job_id, request)
    return {"job_id": job_id, "status": "queued"}

async def execute_workflow_task(job_id: str, request: WorkflowRequest):
    try:
        jobs[job_id].status = "running"
        jobs[job_id].updated_at = datetime.now()
        
        workflow_results = {}
        total_agents = len(request.agents)
        
        for i, agent_type in enumerate(request.agents):
            if agent_type not in agents:
                continue
                
            agent = agents[agent_type]
            
            # Create TaskContext for the agent
            context = TaskContext(
                task_id=f"{job_id}_{agent_type}",
                user_input=request.task_description
            )
            
            # Execute agent
            result = await asyncio.to_thread(agent.execute_task, context)
            workflow_results[agent_type] = {
                "success": result.success,
                "message": result.message,
                "data": result.data,
                "recommendations": result.recommendations
            }
            
            # Update progress
            jobs[job_id].progress = (i + 1) / total_agents
            jobs[job_id].updated_at = datetime.now()
        
        jobs[job_id].status = "completed"
        jobs[job_id].result = {
            "workflow_results": workflow_results,
            "summary": f"Executed {len(workflow_results)} agents successfully"
        }
        jobs[job_id].updated_at = datetime.now()
        
    except Exception as e:
        jobs[job_id].status = "failed"
        jobs[job_id].error = str(e)
        jobs[job_id].updated_at = datetime.now()

@app.get("/jobs/{job_id}")
async def get_job_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return jobs[job_id]

@app.get("/jobs")
async def list_jobs(limit: int = 50, status: Optional[str] = None):
    filtered_jobs = list(jobs.values())
    
    if status:
        filtered_jobs = [job for job in filtered_jobs if job.status == status]
    
    # Sort by creation time (newest first)
    filtered_jobs.sort(key=lambda x: x.created_at, reverse=True)
    
    return {"jobs": filtered_jobs[:limit], "total": len(filtered_jobs)}

@app.delete("/jobs/{job_id}")
async def cancel_job(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    if job.status in ["completed", "failed"]:
        raise HTTPException(status_code=400, detail="Cannot cancel completed or failed job")
    
    job.status = "cancelled"
    job.updated_at = datetime.now()
    
    return {"message": "Job cancelled successfully"}

@app.post("/automl")
async def run_automl_workflow(
    file: UploadFile = File(...),
    target_column: str = Form(..., description="Target column name"),
    task_type: str = Form("classification", description="ML task type")
):
    """
    Complete AutoML workflow: Upload data -> Process -> EDA -> Train Models
    Returns trained models and analysis results.
    """
    try:
        # 1. Load uploaded data
        contents = await file.read()
        if file.filename.endswith('.csv'):
            df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        else:
            raise HTTPException(status_code=400, detail="Only CSV files supported")
        
        if target_column not in df.columns:
            raise HTTPException(status_code=400, detail=f"Target column '{target_column}' not found")
        
        # 2. Process through pipeline
        pipeline = DataProcessingPipeline()
        pipeline_result = pipeline.process_dataset(df, target_column, task_type)
        
        if not pipeline_result.success:
            raise HTTPException(status_code=500, detail="Data processing failed")
        
        # 3. Run EDA Agent
        eda_agent = EDAAgent()
        eda_context = TaskContext(
            task_id=f"eda_{uuid.uuid4()}",
            user_input=f"Analyze {file.filename} for {task_type}"
        )
        eda_context.data = pipeline_result.data
        eda_context.target_column = target_column
        
        eda_result = eda_agent.execute_task(eda_context)
        
        # 4. Run Classification Agent (if classification task)
        model_result = None
        if task_type == "classification":
            class_agent = ClassificationAgent()
            class_context = TaskContext(
                task_id=f"classification_{uuid.uuid4()}",
                user_input=f"Train models on {file.filename}"
            )
            class_context.splits = pipeline_result.metadata['splits']
            
            model_result = class_agent.execute_task(class_context)
        
        # 5. Run QA Agent for quality validation
        qa_agent = QualityAssuranceAgent()
        qa_context = TaskContext(
            task_id=f"qa_{uuid.uuid4()}",
            user_input=f"Validate quality of AutoML workflow for {file.filename}"
        )
        qa_context.data = pipeline_result.data
        qa_context.target_column = target_column
        qa_context.eda_result = eda_result if eda_result.success else None
        qa_context.model_result = model_result if model_result and model_result.success else None
        
        qa_result = qa_agent.execute_task(qa_context)
        
        # 6. Return comprehensive results
        return {
            "success": True,
            "data_info": {
                "original_shape": pipeline_result.metadata["original_shape"],
                "processed_shape": pipeline_result.metadata["processed_shape"],
                "data_quality_score": eda_result.data["data_profile"]["quality_score"] if eda_result.success else 0,
                "splits": pipeline_result.metadata["data_splits"]
            },
            "eda_results": {
                "success": eda_result.success,
                "quality_score": eda_result.data["data_profile"]["quality_score"] if eda_result.success else 0,
                "visualizations_count": len(eda_result.data["visualizations"]) if eda_result.success else 0,
                "recommendations": eda_result.recommendations if eda_result.success else []
            },
            "model_results": {
                "success": model_result.success if model_result else False,
                "best_model": model_result.data["best_model"] if model_result and model_result.success else None,
                "best_score": model_result.data["best_score"] if model_result and model_result.success else 0,
                "models_trained": len(model_result.data["model_results"]) if model_result and model_result.success else 0
            } if model_result else None,
            "quality_assurance": {
                "success": qa_result.success,
                "overall_quality_score": qa_result.data["overall_quality_score"] if qa_result.success else 0,
                "quality_level": qa_result.data["quality_level"] if qa_result.success else "unknown",
                "dimension_scores": qa_result.data["dimension_scores"] if qa_result.success else {},
                "validation_summary": qa_result.data["validation_summary"] if qa_result.success else {},
                "recommendations": qa_result.recommendations if qa_result.success else []
            },
            "processing_time": pipeline_result.processing_time
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze")
async def quick_analyze(
    task_description: str,
    target_column: Optional[str] = None,
    background_tasks: BackgroundTasks = None
):
    """Quick analysis endpoint that automatically selects appropriate agents"""
    
    # Simple task type detection
    task_lower = task_description.lower()
    selected_agents = []
    
    if any(word in task_lower for word in ["classify", "classification", "predict category", "label"]):
        selected_agents = ["eda", "data_hygiene", "feature_engineering", "classification", "qa"]
    elif any(word in task_lower for word in ["predict", "regression", "forecast value", "estimate"]):
        selected_agents = ["eda", "data_hygiene", "feature_engineering", "regression", "qa"]
    elif any(word in task_lower for word in ["text", "nlp", "sentiment", "language"]):
        selected_agents = ["nlp"]
    elif any(word in task_lower for word in ["image", "vision", "picture", "photo"]):
        selected_agents = ["computer_vision"]
    elif any(word in task_lower for word in ["time series", "temporal", "trend", "seasonal"]):
        selected_agents = ["time_series"]
    elif any(word in task_lower for word in ["hyperparameter", "optimization", "tune"]):
        selected_agents = ["hyperparameter_tuning"]
    elif any(word in task_lower for word in ["ensemble", "combine models", "model combination"]):
        selected_agents = ["ensemble"]
    else:
        # Default to full pipeline with QA validation
        selected_agents = ["eda", "data_hygiene", "feature_engineering", "classification", "qa"]
    
    workflow_request = WorkflowRequest(
        task_description=task_description,
        agents=selected_agents,
        quality_threshold=0.8,
        collaboration_mode="sequential"
    )
    
    return await execute_workflow(workflow_request, background_tasks)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)