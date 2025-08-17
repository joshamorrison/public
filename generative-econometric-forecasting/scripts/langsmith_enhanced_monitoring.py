"""
Enhanced LangSmith Integration for Econometric Forecasting Platform
Provides comprehensive monitoring, tracing, and analytics for LangChain operations.
"""

import os
import json
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
from contextlib import contextmanager
import pandas as pd
import numpy as np

# LangSmith and LangChain imports
from langsmith import Client
from langsmith.schemas import Run, Example
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import LLMResult, AgentAction, AgentFinish
from langchain_core.messages import BaseMessage
import uuid

logger = logging.getLogger(__name__)

class EconometricForecastingTracer(BaseCallbackHandler):
    """
    Custom LangSmith callback handler for econometric forecasting operations.
    Tracks model performance, API usage, and forecasting accuracy.
    """
    
    def __init__(self, project_name: str = "econometric-forecasting"):
        """Initialize the tracer with project configuration."""
        self.project_name = project_name
        self.session_id = str(uuid.uuid4())
        self.client = Client()
        
        # Performance tracking
        self.start_times = {}
        self.token_usage = {}
        self.error_counts = {}
        self.forecast_metrics = {}
        
        logger.info(f"Initialized EconometricForecastingTracer for project: {project_name}")
    
    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs) -> None:
        """Called when LLM starts processing."""
        run_id = kwargs.get('run_id', str(uuid.uuid4()))
        self.start_times[run_id] = time.time()
        
        # Log prompt details for forecasting context
        for i, prompt in enumerate(prompts):
            if 'forecast' in prompt.lower() or 'economic' in prompt.lower():
                logger.info(f"Economic forecasting LLM call started - Run ID: {run_id}")
                
                # Extract economic indicators from prompt
                indicators = self._extract_indicators(prompt)
                if indicators:
                    self.forecast_metrics[run_id] = {
                        'indicators': indicators,
                        'start_time': datetime.utcnow().isoformat(),
                        'prompt_length': len(prompt)
                    }
    
    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        """Called when LLM finishes processing."""
        run_id = kwargs.get('run_id')
        
        if run_id and run_id in self.start_times:
            duration = time.time() - self.start_times[run_id]
            
            # Track token usage
            if hasattr(response, 'llm_output') and response.llm_output:
                token_usage = response.llm_output.get('token_usage', {})
                self.token_usage[run_id] = {
                    'prompt_tokens': token_usage.get('prompt_tokens', 0),
                    'completion_tokens': token_usage.get('completion_tokens', 0),
                    'total_tokens': token_usage.get('total_tokens', 0),
                    'duration_seconds': duration
                }
            
            # Update forecast metrics
            if run_id in self.forecast_metrics:
                self.forecast_metrics[run_id].update({
                    'end_time': datetime.utcnow().isoformat(),
                    'duration_seconds': duration,
                    'response_length': len(str(response.generations[0][0].text)) if response.generations else 0
                })
            
            logger.info(f"LLM call completed - Duration: {duration:.2f}s, Run ID: {run_id}")
    
    def on_llm_error(self, error: Union[Exception, KeyboardInterrupt], **kwargs) -> None:
        """Called when LLM encounters an error."""
        run_id = kwargs.get('run_id')
        error_type = type(error).__name__
        
        # Track error metrics
        if error_type not in self.error_counts:
            self.error_counts[error_type] = 0
        self.error_counts[error_type] += 1
        
        logger.error(f"LLM error occurred: {error_type} - {str(error)[:200]}")
        
        # Log to LangSmith
        if run_id:
            self.client.create_run(
                name="llm_error",
                run_type="llm",
                inputs={"error_type": error_type, "error_message": str(error)},
                project_name=self.project_name,
                session_name=self.session_id
            )
    
    def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs) -> None:
        """Called when a chain starts executing."""
        run_id = kwargs.get('run_id')
        chain_name = serialized.get('name', 'unknown_chain')
        
        if 'forecast' in chain_name.lower() or 'economic' in chain_name.lower():
            logger.info(f"Economic forecasting chain started: {chain_name}")
            
            # Track forecasting-specific inputs
            if run_id:
                self.forecast_metrics[run_id] = {
                    'chain_name': chain_name,
                    'inputs': inputs,
                    'start_time': datetime.utcnow().isoformat()
                }
    
    def on_chain_end(self, outputs: Dict[str, Any], **kwargs) -> None:
        """Called when a chain finishes executing."""
        run_id = kwargs.get('run_id')
        
        if run_id and run_id in self.forecast_metrics:
            # Analyze forecasting outputs
            forecast_quality = self._analyze_forecast_output(outputs)
            
            self.forecast_metrics[run_id].update({
                'outputs': outputs,
                'end_time': datetime.utcnow().isoformat(),
                'forecast_quality': forecast_quality
            })
            
            logger.info(f"Forecasting chain completed - Quality score: {forecast_quality.get('overall_score', 0):.2f}")
    
    def on_agent_action(self, action: AgentAction, **kwargs) -> None:
        """Called when an agent takes an action."""
        logger.info(f"Agent action: {action.tool} - {action.tool_input}")
    
    def on_agent_finish(self, finish: AgentFinish, **kwargs) -> None:
        """Called when an agent finishes."""
        logger.info(f"Agent finished: {finish.return_values}")
    
    def _extract_indicators(self, prompt: str) -> List[str]:
        """Extract economic indicators from prompt text."""
        indicators = []
        common_indicators = [
            'GDP', 'UNEMPLOYMENT', 'INFLATION', 'INTEREST_RATE',
            'CONSUMER_CONFIDENCE', 'HOUSING_STARTS', 'RETAIL_SALES',
            'INDUSTRIAL_PRODUCTION', 'CPI', 'PPI'
        ]
        
        prompt_upper = prompt.upper()
        for indicator in common_indicators:
            if indicator in prompt_upper:
                indicators.append(indicator)
        
        return indicators
    
    def _analyze_forecast_output(self, outputs: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the quality of forecasting outputs."""
        quality_metrics = {
            'overall_score': 0.0,
            'has_numerical_forecast': False,
            'has_confidence_intervals': False,
            'has_narrative': False,
            'completeness_score': 0.0
        }
        
        output_text = str(outputs)
        
        # Check for numerical forecasts
        import re
        if re.search(r'\d+\.?\d*%?', output_text):
            quality_metrics['has_numerical_forecast'] = True
        
        # Check for confidence intervals
        if 'confidence' in output_text.lower() or 'interval' in output_text.lower():
            quality_metrics['has_confidence_intervals'] = True
        
        # Check for narrative explanation
        if len(output_text) > 100:  # Substantial text content
            quality_metrics['has_narrative'] = True
        
        # Calculate completeness score
        completeness_factors = [
            quality_metrics['has_numerical_forecast'],
            quality_metrics['has_confidence_intervals'],
            quality_metrics['has_narrative']
        ]
        quality_metrics['completeness_score'] = sum(completeness_factors) / len(completeness_factors)
        quality_metrics['overall_score'] = quality_metrics['completeness_score']
        
        return quality_metrics
    
    def get_session_metrics(self) -> Dict[str, Any]:
        """Get metrics for the current session."""
        total_calls = len(self.start_times)
        total_tokens = sum(usage.get('total_tokens', 0) for usage in self.token_usage.values())
        avg_duration = np.mean([
            metrics.get('duration_seconds', 0) 
            for metrics in self.forecast_metrics.values()
        ]) if self.forecast_metrics else 0
        
        return {
            'session_id': self.session_id,
            'total_llm_calls': total_calls,
            'total_tokens_used': total_tokens,
            'average_call_duration': avg_duration,
            'error_counts': self.error_counts,
            'forecasting_calls': len(self.forecast_metrics),
            'generated_at': datetime.utcnow().isoformat()
        }


class LangSmithEconometricMonitor:
    """
    Advanced monitoring and analytics for LangSmith in econometric forecasting context.
    """
    
    def __init__(self, project_name: str = "econometric-forecasting"):
        """Initialize the monitor."""
        self.project_name = project_name
        self.client = Client()
        self.tracer = EconometricForecastingTracer(project_name)
        
        # Ensure project exists
        try:
            self.client.create_project(project_name=project_name)
        except Exception:
            pass  # Project already exists
        
        logger.info(f"Initialized LangSmith monitor for project: {project_name}")
    
    @contextmanager
    def trace_forecasting_operation(self, operation_name: str, **metadata):
        """Context manager for tracing forecasting operations."""
        run_id = str(uuid.uuid4())
        start_time = time.time()
        
        try:
            # Create run in LangSmith
            run = self.client.create_run(
                name=operation_name,
                run_type="chain",
                inputs=metadata,
                project_name=self.project_name,
                start_time=datetime.utcnow(),
                session_name=self.tracer.session_id
            )
            
            logger.info(f"Started tracing operation: {operation_name}")
            yield run_id
            
            # Mark as successful
            self.client.update_run(
                run_id=run.id,
                end_time=datetime.utcnow(),
                outputs={"status": "success", "duration": time.time() - start_time}
            )
            
        except Exception as e:
            # Mark as failed
            self.client.update_run(
                run_id=run.id,
                end_time=datetime.utcnow(),
                error=str(e),
                outputs={"status": "error", "error_message": str(e)}
            )
            logger.error(f"Operation {operation_name} failed: {e}")
            raise
    
    def log_forecast_results(self, 
                           indicator: str,
                           forecast_values: List[float],
                           confidence_intervals: Optional[List[List[float]]] = None,
                           model_name: str = "unknown",
                           metadata: Optional[Dict[str, Any]] = None) -> None:
        """Log forecasting results to LangSmith."""
        
        inputs = {
            "indicator": indicator,
            "model_name": model_name,
            "forecast_horizon": len(forecast_values),
            "metadata": metadata or {}
        }
        
        outputs = {
            "forecast_values": forecast_values,
            "confidence_intervals": confidence_intervals,
            "forecast_mean": np.mean(forecast_values),
            "forecast_std": np.std(forecast_values),
            "generated_at": datetime.utcnow().isoformat()
        }
        
        # Create run for forecast result
        run = self.client.create_run(
            name=f"forecast_{indicator.lower()}",
            run_type="chain",
            inputs=inputs,
            outputs=outputs,
            project_name=self.project_name,
            session_name=self.tracer.session_id
        )
        
        logger.info(f"Logged forecast results for {indicator} - Run ID: {run.id}")
    
    def log_model_performance(self,
                            model_name: str,
                            indicator: str,
                            performance_metrics: Dict[str, float],
                            training_data_size: int,
                            validation_metrics: Optional[Dict[str, float]] = None) -> None:
        """Log model performance metrics."""
        
        inputs = {
            "model_name": model_name,
            "indicator": indicator,
            "training_data_size": training_data_size,
            "evaluation_timestamp": datetime.utcnow().isoformat()
        }
        
        outputs = {
            "performance_metrics": performance_metrics,
            "validation_metrics": validation_metrics or {},
            "model_quality_score": self._calculate_model_quality(performance_metrics)
        }
        
        run = self.client.create_run(
            name=f"model_evaluation_{model_name}",
            run_type="tool",
            inputs=inputs,
            outputs=outputs,
            project_name=self.project_name,
            session_name=self.tracer.session_id
        )
        
        logger.info(f"Logged model performance for {model_name} on {indicator}")
    
    def log_data_quality_check(self,
                             data_source: str,
                             indicators: List[str],
                             quality_metrics: Dict[str, Any]) -> None:
        """Log data quality assessment results."""
        
        inputs = {
            "data_source": data_source,
            "indicators": indicators,
            "check_timestamp": datetime.utcnow().isoformat()
        }
        
        outputs = {
            "quality_metrics": quality_metrics,
            "overall_quality_score": quality_metrics.get('overall_score', 0),
            "passed_checks": quality_metrics.get('passed_checks', []),
            "failed_checks": quality_metrics.get('failed_checks', [])
        }
        
        run = self.client.create_run(
            name=f"data_quality_check_{data_source}",
            run_type="tool",
            inputs=inputs,
            outputs=outputs,
            project_name=self.project_name,
            session_name=self.tracer.session_id
        )
        
        logger.info(f"Logged data quality check for {data_source}")
    
    def create_forecasting_dataset(self,
                                 dataset_name: str,
                                 examples: List[Dict[str, Any]]) -> None:
        """Create a dataset for forecasting evaluation."""
        
        # Create dataset
        dataset = self.client.create_dataset(
            dataset_name=dataset_name,
            description=f"Econometric forecasting examples for evaluation - {datetime.utcnow().strftime('%Y-%m-%d')}"
        )
        
        # Add examples to dataset
        for example in examples:
            self.client.create_example(
                inputs=example.get('inputs', {}),
                outputs=example.get('outputs', {}),
                dataset_id=dataset.id
            )
        
        logger.info(f"Created forecasting dataset: {dataset_name} with {len(examples)} examples")
        return dataset.id
    
    def get_project_analytics(self, days_back: int = 7) -> Dict[str, Any]:
        """Get comprehensive analytics for the forecasting project."""
        
        # Get runs from the last N days
        start_time = datetime.utcnow() - timedelta(days=days_back)
        
        try:
            runs = list(self.client.list_runs(
                project_name=self.project_name,
                start_time=start_time
            ))
        except Exception as e:
            logger.error(f"Failed to fetch project analytics: {e}")
            return {"error": str(e)}
        
        # Analyze runs
        analytics = {
            "project_name": self.project_name,
            "analysis_period": f"{days_back} days",
            "total_runs": len(runs),
            "successful_runs": 0,
            "failed_runs": 0,
            "forecasting_runs": 0,
            "model_evaluation_runs": 0,
            "data_quality_runs": 0,
            "average_duration": 0,
            "token_usage": {"total": 0, "average_per_run": 0},
            "top_indicators": {},
            "model_performance_summary": {},
            "generated_at": datetime.utcnow().isoformat()
        }
        
        total_duration = 0
        total_tokens = 0
        indicator_counts = {}
        
        for run in runs:
            # Status analysis
            if run.error:
                analytics["failed_runs"] += 1
            else:
                analytics["successful_runs"] += 1
            
            # Run type analysis
            if "forecast" in run.name.lower():
                analytics["forecasting_runs"] += 1
                
                # Extract indicator from inputs
                if run.inputs and "indicator" in run.inputs:
                    indicator = run.inputs["indicator"]
                    indicator_counts[indicator] = indicator_counts.get(indicator, 0) + 1
            
            elif "model_evaluation" in run.name.lower():
                analytics["model_evaluation_runs"] += 1
            
            elif "data_quality" in run.name.lower():
                analytics["data_quality_runs"] += 1
            
            # Duration analysis
            if run.start_time and run.end_time:
                duration = (run.end_time - run.start_time).total_seconds()
                total_duration += duration
            
            # Token usage (if available in outputs)
            if run.outputs and "token_usage" in run.outputs:
                tokens = run.outputs["token_usage"].get("total_tokens", 0)
                total_tokens += tokens
        
        # Calculate averages
        if analytics["total_runs"] > 0:
            analytics["average_duration"] = total_duration / analytics["total_runs"]
            analytics["token_usage"]["total"] = total_tokens
            analytics["token_usage"]["average_per_run"] = total_tokens / analytics["total_runs"]
        
        # Top indicators
        analytics["top_indicators"] = dict(
            sorted(indicator_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        )
        
        return analytics
    
    def _calculate_model_quality(self, performance_metrics: Dict[str, float]) -> float:
        """Calculate overall model quality score."""
        
        # Common performance metrics and their weights
        metric_weights = {
            'mae': 0.3,   # Lower is better
            'mse': 0.3,   # Lower is better
            'rmse': 0.2,  # Lower is better
            'mape': 0.2,  # Lower is better
            'r2': 0.4,    # Higher is better
            'accuracy': 0.3  # Higher is better
        }
        
        quality_score = 0.0
        total_weight = 0.0
        
        for metric, value in performance_metrics.items():
            if metric in metric_weights:
                weight = metric_weights[metric]
                
                # Normalize score (assuming error metrics should be minimized)
                if metric in ['mae', 'mse', 'rmse', 'mape']:
                    # Convert error to quality (lower error = higher quality)
                    normalized_score = max(0, 1 - (value / 100))  # Assuming max reasonable error is 100
                else:
                    # For metrics like R2, accuracy (higher is better)
                    normalized_score = min(1, value)
                
                quality_score += normalized_score * weight
                total_weight += weight
        
        return quality_score / total_weight if total_weight > 0 else 0.0
    
    def export_analytics_report(self, output_path: str, days_back: int = 30) -> None:
        """Export comprehensive analytics report."""
        
        analytics = self.get_project_analytics(days_back)
        
        # Create detailed report
        report = {
            "report_metadata": {
                "generated_at": datetime.utcnow().isoformat(),
                "project_name": self.project_name,
                "analysis_period_days": days_back,
                "langsmith_version": "latest"
            },
            "executive_summary": {
                "total_operations": analytics["total_runs"],
                "success_rate": analytics["successful_runs"] / analytics["total_runs"] if analytics["total_runs"] > 0 else 0,
                "average_operation_duration": analytics["average_duration"],
                "most_forecasted_indicator": max(analytics["top_indicators"], key=analytics["top_indicators"].get) if analytics["top_indicators"] else "N/A"
            },
            "detailed_analytics": analytics,
            "session_metrics": self.tracer.get_session_metrics()
        }
        
        # Save report
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Exported analytics report to: {output_path}")


# Global monitor instance
_monitor = None

def get_langsmith_monitor(project_name: str = "econometric-forecasting") -> LangSmithEconometricMonitor:
    """Get or create the global LangSmith monitor instance."""
    global _monitor
    if _monitor is None:
        _monitor = LangSmithEconometricMonitor(project_name)
    return _monitor

def setup_langsmith_tracing(project_name: str = "econometric-forecasting") -> EconometricForecastingTracer:
    """Set up LangSmith tracing for the forecasting platform."""
    
    # Set environment variables for LangSmith
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = project_name
    
    # Ensure API key is set
    if not os.getenv("LANGCHAIN_API_KEY"):
        logger.warning("LANGCHAIN_API_KEY not set. LangSmith tracing may not work properly.")
    
    # Create and return tracer
    tracer = EconometricForecastingTracer(project_name)
    logger.info(f"LangSmith tracing enabled for project: {project_name}")
    
    return tracer


# Example usage and testing
if __name__ == "__main__":
    # Set up enhanced monitoring
    monitor = get_langsmith_monitor("econometric-forecasting-demo")
    
    # Example: Log a forecast operation
    with monitor.trace_forecasting_operation("gdp_forecast", model="arima", horizon=6):
        # Simulate forecasting operation
        time.sleep(1)
        
        # Log forecast results
        monitor.log_forecast_results(
            indicator="GDP",
            forecast_values=[23000, 23100, 23200, 23300, 23400, 23500],
            confidence_intervals=[[22800, 23200], [22900, 23300], [23000, 23400], [23100, 23500], [23200, 23600], [23300, 23700]],
            model_name="ARIMA(1,1,1)",
            metadata={"data_source": "FRED", "last_updated": "2025-01-17"}
        )
        
        # Log model performance
        monitor.log_model_performance(
            model_name="ARIMA(1,1,1)",
            indicator="GDP",
            performance_metrics={"mae": 150.2, "mse": 45000.5, "r2": 0.85},
            training_data_size=120
        )
    
    # Get analytics
    analytics = monitor.get_project_analytics(days_back=1)
    print("Project Analytics:")
    print(json.dumps(analytics, indent=2, default=str))
    
    # Export report
    monitor.export_analytics_report("langsmith_analytics_report.json", days_back=1)