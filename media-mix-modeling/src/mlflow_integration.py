"""
MLflow Integration for Media Mix Modeling
Provides experiment tracking, model versioning, and performance monitoring
"""

import os
import mlflow
import mlflow.sklearn
from datetime import datetime
import json
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
import warnings
warnings.filterwarnings('ignore')

class MMMMLflowTracker:
    """MLflow tracking integration for Media Mix Modeling experiments"""
    
    def __init__(self, experiment_name: str = "media-mix-modeling", tracking_uri: Optional[str] = None):
        """
        Initialize MLflow tracking
        
        Args:
            experiment_name: Name of the MLflow experiment
            tracking_uri: MLflow tracking server URI (optional)
        """
        self.experiment_name = experiment_name
        
        # Set tracking URI if provided
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        
        # Set or create experiment
        try:
            mlflow.set_experiment(experiment_name)
            print(f"[MLFLOW] Using experiment: {experiment_name}")
        except Exception as e:
            print(f"[MLFLOW] Warning: Could not set experiment {experiment_name}: {e}")
    
    def start_run(self, run_name: Optional[str] = None):
        """
        Start a new MLflow run
        
        Args:
            run_name: Optional name for the run
        """
        try:
            mlflow.start_run(run_name=run_name)
            if run_name:
                print(f"[MLFLOW] Started run: {run_name}")
            else:
                print(f"[MLFLOW] Started new run")
        except Exception as e:
            print(f"[MLFLOW] Error starting run: {e}")
    
    def log_mmm_parameters(self, mmm_model):
        """
        Log MMM model parameters
        
        Args:
            mmm_model: EconometricMMM model instance
        """
        try:
            params = {
                'adstock_rate': mmm_model.adstock_rate,
                'saturation_param': mmm_model.saturation_param,
                'regularization_alpha': mmm_model.regularization_alpha,
                'include_baseline': mmm_model.include_baseline,
                'model_type': 'econometric_mmm'
            }
            
            mlflow.log_params(params)
            print(f"[MLFLOW] Logged {len(params)} model parameters")
            
        except Exception as e:
            print(f"[MLFLOW] Error logging parameters: {e}")
    
    def log_performance_metrics(self, performance_metrics: Dict[str, float]):
        """
        Log model performance metrics
        
        Args:
            performance_metrics: Dictionary of performance metrics
        """
        try:
            # Log main performance metrics
            metrics_to_log = {}
            
            if 'r2_score' in performance_metrics:
                metrics_to_log['r2_score'] = performance_metrics['r2_score']
            
            if 'mape' in performance_metrics:
                metrics_to_log['mape'] = performance_metrics['mape']
            
            if 'rmse' in performance_metrics:
                metrics_to_log['rmse'] = performance_metrics['rmse']
            
            if 'mean_actual' in performance_metrics:
                metrics_to_log['mean_actual'] = performance_metrics['mean_actual']
            
            if 'mean_predicted' in performance_metrics:
                metrics_to_log['mean_predicted'] = performance_metrics['mean_predicted']
            
            mlflow.log_metrics(metrics_to_log)
            print(f"[MLFLOW] Logged {len(metrics_to_log)} performance metrics")
            
        except Exception as e:
            print(f"[MLFLOW] Error logging metrics: {e}")
    
    def log_data_info(self, data_info: Dict[str, Any]):
        """
        Log information about the training data
        
        Args:
            data_info: Dictionary containing data information
        """
        try:
            data_params = {}
            
            if 'source_type' in data_info:
                data_params['data_source'] = data_info['source_type']
            
            if 'records_count' in data_info:
                data_params['training_records'] = data_info['records_count']
            
            if 'date_range' in data_info:
                data_params['date_range'] = str(data_info['date_range'])
            
            if 'channels_count' in data_info:
                data_params['channels_count'] = data_info['channels_count']
            
            mlflow.log_params(data_params)
            print(f"[MLFLOW] Logged data information")
            
        except Exception as e:
            print(f"[MLFLOW] Error logging data info: {e}")
    
    def log_attribution_results(self, attribution_results: Dict[str, float]):
        """
        Log channel attribution results
        
        Args:
            attribution_results: Dictionary of channel attribution percentages
        """
        try:
            # Log attribution as metrics
            attribution_metrics = {}
            
            for channel, attribution in attribution_results.items():
                if isinstance(attribution, (int, float)):
                    attribution_metrics[f"attribution_{channel}"] = float(attribution)
            
            if attribution_metrics:
                mlflow.log_metrics(attribution_metrics)
                print(f"[MLFLOW] Logged attribution for {len(attribution_metrics)} channels")
            
        except Exception as e:
            print(f"[MLFLOW] Error logging attribution: {e}")
    
    def log_budget_optimization(self, optimization_results: Dict[str, Any]):
        """
        Log budget optimization results
        
        Args:
            optimization_results: Budget optimization results
        """
        try:
            optimization_metrics = {}
            
            if 'projected_roi' in optimization_results:
                optimization_metrics['projected_roi'] = optimization_results['projected_roi']
            
            if 'projected_revenue' in optimization_results:
                optimization_metrics['projected_revenue'] = optimization_results['projected_revenue']
            
            if 'total_budget' in optimization_results:
                optimization_metrics['total_budget'] = optimization_results['total_budget']
            
            # Log allocation as parameters (since they're categorical)
            if 'allocation' in optimization_results:
                allocation_params = {}
                for channel, amount in optimization_results['allocation'].items():
                    allocation_params[f"optimal_{channel}"] = amount
                mlflow.log_params(allocation_params)
            
            if optimization_metrics:
                mlflow.log_metrics(optimization_metrics)
                print(f"[MLFLOW] Logged budget optimization results")
            
        except Exception as e:
            print(f"[MLFLOW] Error logging optimization: {e}")
    
    def log_model_artifact(self, model, artifact_name: str = "mmm_model"):
        """
        Log the trained model as an artifact
        
        Args:
            model: Trained MMM model
            artifact_name: Name for the model artifact
        """
        try:
            # Use MLflow sklearn integration for scikit-learn based models
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path=artifact_name,
                registered_model_name=f"MMM_{self.experiment_name}"
            )
            print(f"[MLFLOW] Logged model artifact: {artifact_name}")
            
        except Exception as e:
            print(f"[MLFLOW] Error logging model: {e}")
    
    def log_dataframe_artifact(self, df: pd.DataFrame, filename: str):
        """
        Log pandas DataFrame as CSV artifact
        
        Args:
            df: DataFrame to log
            filename: Filename for the artifact
        """
        try:
            import tempfile
            import os
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
                df.to_csv(f.name, index=False)
                temp_path = f.name
            
            mlflow.log_artifact(temp_path, filename)
            os.unlink(temp_path)
            
            print(f"[MLFLOW] Logged DataFrame artifact: {filename}")
            
        except Exception as e:
            print(f"[MLFLOW] Error logging DataFrame {filename}: {e}")
    
    def log_json_artifact(self, data: Dict[str, Any], filename: str):
        """
        Log dictionary as JSON artifact
        
        Args:
            data: Dictionary to log
            filename: Filename for the artifact
        """
        try:
            import tempfile
            import os
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(data, f, indent=2, default=str)
                temp_path = f.name
            
            mlflow.log_artifact(temp_path, filename)
            os.unlink(temp_path)
            
            print(f"[MLFLOW] Logged JSON artifact: {filename}")
            
        except Exception as e:
            print(f"[MLFLOW] Error logging JSON {filename}: {e}")
    
    def end_run(self):
        """End the current MLflow run"""
        try:
            mlflow.end_run()
            print(f"[MLFLOW] Run completed successfully")
        except Exception as e:
            print(f"[MLFLOW] Error ending run: {e}")

def track_mmm_experiment(mmm_model, mmm_results, data_info, experiment_name: str = "media-mix-modeling"):
    """
    Convenience function to track a complete MMM experiment
    
    Args:
        mmm_model: Trained MMM model
        mmm_results: Results from model fitting
        data_info: Information about training data
        experiment_name: Name for the MLflow experiment
        
    Returns:
        MLflow run ID
    """
    tracker = MMMMLflowTracker(experiment_name=experiment_name)
    
    try:
        # Start run with timestamp
        run_name = f"mmm_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        tracker.start_run(run_name=run_name)
        
        # Log model parameters
        tracker.log_mmm_parameters(mmm_model)
        
        # Log performance metrics
        if 'performance' in mmm_results:
            tracker.log_performance_metrics(mmm_results['performance'])
        
        # Log data information
        tracker.log_data_info(data_info)
        
        # Log attribution if available
        if 'attribution' in mmm_results:
            tracker.log_attribution_results(mmm_results['attribution'])
        
        # Log model artifact
        tracker.log_model_artifact(mmm_model)
        
        # Get run ID before ending
        run_id = mlflow.active_run().info.run_id if mlflow.active_run() else None
        
        # End run
        tracker.end_run()
        
        print(f"[MLFLOW] Experiment tracking completed. Run ID: {run_id}")
        return run_id
        
    except Exception as e:
        print(f"[MLFLOW] Error in experiment tracking: {e}")
        try:
            tracker.end_run()
        except:
            pass
        return None

# Example usage for MMM integration
if __name__ == "__main__":
    print("MLflow Integration for Media Mix Modeling")
    print("Usage: from src.mlflow_integration import track_mmm_experiment")