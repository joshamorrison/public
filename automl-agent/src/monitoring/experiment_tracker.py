"""
Experiment Tracker for AutoML Agent Platform

Comprehensive experiment tracking for ML workflows, model performance,
and AutoML pipeline results. Integrates with LangSmith and MLflow.
"""

import os
import json
import time
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field, asdict
from pathlib import Path
import logging

from .langsmith_client import get_tracker

logger = logging.getLogger(__name__)

@dataclass
class ExperimentConfig:
    """Configuration for an ML experiment."""
    experiment_id: str
    experiment_name: str
    task_type: str  # classification, regression, time_series, etc.
    dataset_info: Dict[str, Any]
    agents_used: List[str]
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    quality_threshold: float = 0.8
    max_iterations: int = 5
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

@dataclass 
class ModelResult:
    """Results from a trained model."""
    model_name: str
    algorithm: str
    hyperparameters: Dict[str, Any]
    training_score: float
    validation_score: float
    test_score: Optional[float] = None
    metrics: Dict[str, float] = field(default_factory=dict)
    feature_importance: Optional[Dict[str, float]] = None
    training_time: float = 0.0
    model_size_mb: Optional[float] = None
    artifacts: Dict[str, str] = field(default_factory=dict)  # paths to saved models/plots

@dataclass
class ExperimentResult:
    """Complete experiment results."""
    experiment_id: str
    config: ExperimentConfig
    start_time: datetime
    end_time: Optional[datetime] = None
    success: bool = False
    total_duration: float = 0.0
    
    # Agent results
    agent_results: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Model results
    models_trained: List[ModelResult] = field(default_factory=list)
    best_model: Optional[ModelResult] = None
    
    # Workflow info
    workflow_steps: List[str] = field(default_factory=list)
    iterations_completed: int = 0
    final_quality_score: float = 0.0
    
    # Outputs
    artifacts: Dict[str, str] = field(default_factory=dict)
    visualizations: List[str] = field(default_factory=list)
    reports: List[str] = field(default_factory=list)
    
    # Error info
    error_message: Optional[str] = None
    failed_step: Optional[str] = None

class ExperimentTracker:
    """
    Comprehensive experiment tracking for AutoML workflows.
    
    Features:
    - Experiment lifecycle management
    - Model performance tracking
    - Agent execution results
    - Artifact management
    - Integration with LangSmith
    - Export to multiple formats
    """
    
    def __init__(self, experiments_dir: str = "outputs/experiments"):
        """Initialize the experiment tracker."""
        self.experiments_dir = Path(experiments_dir)
        self.experiments_dir.mkdir(parents=True, exist_ok=True)
        
        self.active_experiments: Dict[str, ExperimentResult] = {}
        self.langsmith_tracker = get_tracker()
        
        logger.info(f"Experiment tracker initialized: {self.experiments_dir}")
    
    def start_experiment(
        self,
        experiment_name: str,
        task_type: str,
        dataset_info: Dict[str, Any],
        agents_used: List[str],
        hyperparameters: Optional[Dict[str, Any]] = None,
        quality_threshold: float = 0.8,
        max_iterations: int = 5,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Start a new experiment and return the experiment ID."""
        experiment_id = str(uuid.uuid4())
        
        config = ExperimentConfig(
            experiment_id=experiment_id,
            experiment_name=experiment_name,
            task_type=task_type,
            dataset_info=dataset_info,
            agents_used=agents_used,
            hyperparameters=hyperparameters or {},
            quality_threshold=quality_threshold,
            max_iterations=max_iterations,
            metadata=metadata or {}
        )
        
        result = ExperimentResult(
            experiment_id=experiment_id,
            config=config,
            start_time=datetime.now()
        )
        
        self.active_experiments[experiment_id] = result
        
        # Log to LangSmith
        if self.langsmith_tracker.is_enabled():
            self.langsmith_tracker.log_workflow_execution(
                workflow_id=experiment_id,
                agents=agents_used,
                start_time=result.start_time
            )
        
        logger.info(f"Started experiment: {experiment_name} ({experiment_id})")
        return experiment_id
    
    def log_agent_result(
        self,
        experiment_id: str,
        agent_name: str,
        success: bool,
        results: Dict[str, Any],
        execution_time: float,
        error: Optional[str] = None
    ):
        """Log results from an agent execution."""
        if experiment_id not in self.active_experiments:
            logger.warning(f"Experiment {experiment_id} not found")
            return
        
        experiment = self.active_experiments[experiment_id]
        
        agent_result = {
            "agent_name": agent_name,
            "success": success,
            "results": results,
            "execution_time": execution_time,
            "timestamp": datetime.now(),
            "error": error
        }
        
        experiment.agent_results[agent_name] = agent_result
        experiment.workflow_steps.append(f"{agent_name}_{'success' if success else 'failed'}")
        
        logger.debug(f"Logged {agent_name} result for experiment {experiment_id}")
    
    def log_model_result(
        self,
        experiment_id: str,
        model_name: str,
        algorithm: str,
        hyperparameters: Dict[str, Any],
        training_score: float,
        validation_score: float,
        test_score: Optional[float] = None,
        metrics: Optional[Dict[str, float]] = None,
        feature_importance: Optional[Dict[str, float]] = None,
        training_time: float = 0.0,
        model_size_mb: Optional[float] = None,
        artifacts: Optional[Dict[str, str]] = None
    ):
        """Log results from model training."""
        if experiment_id not in self.active_experiments:
            logger.warning(f"Experiment {experiment_id} not found")
            return
        
        model_result = ModelResult(
            model_name=model_name,
            algorithm=algorithm,
            hyperparameters=hyperparameters,
            training_score=training_score,
            validation_score=validation_score,
            test_score=test_score,
            metrics=metrics or {},
            feature_importance=feature_importance,
            training_time=training_time,
            model_size_mb=model_size_mb,
            artifacts=artifacts or {}
        )
        
        experiment = self.active_experiments[experiment_id]
        experiment.models_trained.append(model_result)
        
        # Update best model
        if (experiment.best_model is None or 
            validation_score > experiment.best_model.validation_score):
            experiment.best_model = model_result
            experiment.final_quality_score = validation_score
        
        logger.debug(f"Logged model result: {model_name} for experiment {experiment_id}")
    
    def add_artifact(
        self,
        experiment_id: str,
        artifact_name: str,
        artifact_path: str,
        artifact_type: str = "file"
    ):
        """Add an artifact to the experiment."""
        if experiment_id not in self.active_experiments:
            logger.warning(f"Experiment {experiment_id} not found")
            return
        
        experiment = self.active_experiments[experiment_id]
        experiment.artifacts[artifact_name] = artifact_path
        
        # Categorize artifacts
        if artifact_type == "visualization":
            experiment.visualizations.append(artifact_path)
        elif artifact_type == "report":
            experiment.reports.append(artifact_path)
        
        logger.debug(f"Added artifact {artifact_name} to experiment {experiment_id}")
    
    def complete_experiment(
        self,
        experiment_id: str,
        success: bool = True,
        error_message: Optional[str] = None,
        failed_step: Optional[str] = None
    ) -> ExperimentResult:
        """Complete an experiment and save results."""
        if experiment_id not in self.active_experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        experiment = self.active_experiments[experiment_id]
        experiment.end_time = datetime.now()
        experiment.success = success
        experiment.total_duration = (experiment.end_time - experiment.start_time).total_seconds()
        experiment.error_message = error_message
        experiment.failed_step = failed_step
        
        # Log completion to LangSmith
        if self.langsmith_tracker.is_enabled():
            self.langsmith_tracker.log_workflow_execution(
                workflow_id=experiment_id,
                agents=experiment.config.agents_used,
                start_time=experiment.start_time,
                end_time=experiment.end_time,
                success=success,
                results={
                    "final_quality_score": experiment.final_quality_score,
                    "models_trained": len(experiment.models_trained),
                    "best_model": experiment.best_model.model_name if experiment.best_model else None
                },
                error=error_message
            )
        
        # Save experiment results
        self._save_experiment(experiment)
        
        # Remove from active experiments
        del self.active_experiments[experiment_id]
        
        logger.info(f"Completed experiment {experiment_id}: {'success' if success else 'failed'}")
        return experiment
    
    def get_experiment_status(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Get current status of an experiment."""
        if experiment_id not in self.active_experiments:
            return None
        
        experiment = self.active_experiments[experiment_id]
        
        return {
            "experiment_id": experiment_id,
            "experiment_name": experiment.config.experiment_name,
            "status": "running",
            "start_time": experiment.start_time,
            "duration": (datetime.now() - experiment.start_time).total_seconds(),
            "agents_used": experiment.config.agents_used,
            "workflow_steps": experiment.workflow_steps,
            "models_trained": len(experiment.models_trained),
            "best_score": experiment.final_quality_score,
            "quality_threshold": experiment.config.quality_threshold
        }
    
    def list_experiments(
        self,
        limit: int = 50,
        task_type: Optional[str] = None,
        success_only: bool = False
    ) -> List[Dict[str, Any]]:
        """List previous experiments."""
        experiments = []
        
        # Get saved experiments
        for exp_file in self.experiments_dir.glob("experiment_*.json"):
            try:
                with open(exp_file, 'r') as f:
                    exp_data = json.load(f)
                
                # Apply filters
                if task_type and exp_data.get("config", {}).get("task_type") != task_type:
                    continue
                    
                if success_only and not exp_data.get("success", False):
                    continue
                
                # Create summary
                summary = {
                    "experiment_id": exp_data["experiment_id"],
                    "experiment_name": exp_data["config"]["experiment_name"],
                    "task_type": exp_data["config"]["task_type"],
                    "start_time": exp_data["start_time"],
                    "duration": exp_data["total_duration"],
                    "success": exp_data["success"],
                    "final_score": exp_data["final_quality_score"],
                    "models_trained": len(exp_data.get("models_trained", [])),
                    "best_model": exp_data.get("best_model", {}).get("algorithm") if exp_data.get("best_model") else None
                }
                
                experiments.append(summary)
                
            except Exception as e:
                logger.warning(f"Failed to load experiment from {exp_file}: {e}")
        
        # Sort by start time (newest first) and limit
        experiments.sort(key=lambda x: x["start_time"], reverse=True)
        return experiments[:limit]
    
    def get_experiment_details(self, experiment_id: str) -> Optional[ExperimentResult]:
        """Get full details of a completed experiment."""
        exp_file = self.experiments_dir / f"experiment_{experiment_id}.json"
        
        if not exp_file.exists():
            return None
        
        try:
            with open(exp_file, 'r') as f:
                data = json.load(f)
            
            # Reconstruct ExperimentResult
            # This would require custom deserialization logic
            # For now, return the raw data
            return data
            
        except Exception as e:
            logger.error(f"Failed to load experiment {experiment_id}: {e}")
            return None
    
    def _save_experiment(self, experiment: ExperimentResult):
        """Save experiment results to disk."""
        exp_file = self.experiments_dir / f"experiment_{experiment.experiment_id}.json"
        
        try:
            # Convert to JSON-serializable format
            data = asdict(experiment)
            
            # Handle datetime serialization
            if isinstance(data["start_time"], datetime):
                data["start_time"] = data["start_time"].isoformat()
            if data["end_time"] and isinstance(data["end_time"], datetime):
                data["end_time"] = data["end_time"].isoformat()
            
            # Handle nested datetime objects
            for agent_name, agent_result in data.get("agent_results", {}).items():
                if "timestamp" in agent_result and isinstance(agent_result["timestamp"], datetime):
                    agent_result["timestamp"] = agent_result["timestamp"].isoformat()
            
            with open(exp_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            
            logger.debug(f"Saved experiment results to {exp_file}")
            
        except Exception as e:
            logger.error(f"Failed to save experiment {experiment.experiment_id}: {e}")
    
    def export_experiments(
        self,
        output_file: str,
        format: str = "csv",
        task_type: Optional[str] = None
    ):
        """Export experiment results to CSV or JSON."""
        experiments = self.list_experiments(limit=1000, task_type=task_type)
        
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format.lower() == "csv":
            import pandas as pd
            df = pd.DataFrame(experiments)
            df.to_csv(output_path, index=False)
        elif format.lower() == "json":
            with open(output_path, 'w') as f:
                json.dump(experiments, f, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Exported {len(experiments)} experiments to {output_path}")

# Global tracker instance
tracker = ExperimentTracker()

def get_experiment_tracker() -> ExperimentTracker:
    """Get the global experiment tracker."""
    return tracker