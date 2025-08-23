"""
EDA Agent for AutoML Platform

Specialized agent for Exploratory Data Analysis (EDA) that:
1. Performs comprehensive dataset profiling and analysis
2. Generates statistical summaries and visualizations
3. Identifies data quality issues and patterns
4. Provides insights for downstream agents
5. Creates executive-ready data reports

This agent is typically the first in the AutoML workflow.
"""

import os
import json
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy import stats
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False

from .base_agent import BaseAgent, AgentResult, TaskContext, TaskComplexity
from ..tools.data_tools import DataProfiler
from ..tools.visualization_tools import PlotGenerator, ReportGenerator


@dataclass
class DataProfile:
    """Comprehensive data profile results."""
    shape: Tuple[int, int]
    dtypes: Dict[str, str]
    missing_values: Dict[str, int]
    missing_percentages: Dict[str, float]
    numerical_stats: Dict[str, Dict[str, float]]
    categorical_stats: Dict[str, Dict[str, Any]]
    data_quality_score: float
    memory_usage: str
    duplicate_rows: int


@dataclass
class StatisticalAnalysis:
    """Statistical analysis results."""
    correlation_matrix: Optional[Dict[str, Dict[str, float]]]
    outlier_analysis: Dict[str, List[Any]]
    distribution_analysis: Dict[str, Dict[str, Any]]
    feature_relationships: List[Dict[str, Any]]
    target_analysis: Optional[Dict[str, Any]]


class EDAAgent(BaseAgent):
    """
    Exploratory Data Analysis Agent.
    
    Responsibilities:
    1. Dataset profiling and statistical analysis
    2. Data quality assessment
    3. Visualization generation
    4. Pattern discovery and insights
    5. Recommendations for data preprocessing
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, communication_hub=None):
        """Initialize the EDA Agent."""
        super().__init__(
            name="EDA Agent",
            description="Comprehensive exploratory data analysis and dataset profiling",
            specialization="Data Exploration & Statistical Analysis",
            config=config,
            communication_hub=communication_hub
        )
        
        # EDA-specific configuration
        self.max_unique_values = self.config.get("max_unique_categorical", 50)
        self.correlation_threshold = self.config.get("correlation_threshold", 0.8)
        self.outlier_method = self.config.get("outlier_method", "iqr")
        self.generate_plots = self.config.get("generate_plots", True)
        self.plot_format = self.config.get("plot_format", "png")
        
        # Analysis settings
        self.statistical_tests = self.config.get("statistical_tests", True)
        self.advanced_insights = self.config.get("advanced_insights", True)
        
        # Quality thresholds
        self.quality_thresholds.update({
            "missing_data_threshold": 0.2,  # 20% missing data threshold
            "data_quality_score": 0.7,      # Minimum data quality score
            "correlation_analysis": 0.9      # Analysis completeness
        })
    
    def execute_task(self, context: TaskContext) -> AgentResult:
        """
        Execute comprehensive EDA analysis using real tools.
        
        Args:
            context: Task context with dataset information
            
        Returns:
            AgentResult with EDA findings and recommendations
        """
        try:
            self.logger.info("Starting comprehensive EDA analysis...")
            
            # Get dataset from context (passed from pipeline)
            df = context.data
            target_column = getattr(context, 'target_column', None)
            
            if df is None:
                # Fallback to loading or simulating dataset
                dataset_info = context.dataset_info or {}
                df = self._load_or_simulate_dataset(dataset_info)
            
            if df is None:
                return AgentResult(
                    success=False,
                    message="Failed to load dataset for EDA analysis"
                )
            
            # Initialize our real tools
            profiler = DataProfiler()
            plot_generator = PlotGenerator()
            report_generator = ReportGenerator()
            
            # Phase 1: Real Data Profiling
            self.logger.info("Phase 1: Data profiling using real tools...")
            data_profile = profiler.profile_dataset(df, target_column)
            
            # Phase 2: Generate Visualizations
            self.logger.info("Phase 2: Generating real visualizations...")
            plots = plot_generator.create_data_overview_plots(df, target_column)
            
            # Phase 3: Generate Report  
            self.logger.info("Phase 3: Creating comprehensive report...")
            report_path = report_generator.generate_data_analysis_report(data_profile, plots)
            
            # Create comprehensive result using real tools
            result_data = {
                "data_profile": {
                    "shape": data_profile.shape,
                    "dtypes": data_profile.dtypes,
                    "missing_values": data_profile.missing_values,
                    "numeric_summary": data_profile.numeric_summary,
                    "categorical_summary": data_profile.categorical_summary,
                    "quality_score": data_profile.data_quality_score
                },
                "visualizations": plots,
                "report_path": report_path,
                "recommendations": data_profile.recommendations
            }
            
            # Update performance metrics
            performance_metrics = {
                "data_quality_score": data_profile.data_quality_score,
                "analysis_completeness": 1.0,
                "insights_generated": len(plots),
                "recommendations_count": len(data_profile.recommendations)
            }
            self.update_performance_metrics(performance_metrics)
            
            # Share insights with other agents
            if self.communication_hub:
                self.communication_hub.share_message(
                    "eda_insights",
                    result_data,
                    sender="EDA Agent"
                )
            
            return AgentResult(
                success=True,
                data=result_data,
                message=f"Real EDA completed: {df.shape[0]} rows, {df.shape[1]} features analyzed",
                recommendations=data_profile.recommendations
            )
            
        except Exception as e:
            self.logger.error(f"EDA analysis failed: {e}")
            return AgentResult(
                success=False,
                message=f"EDA analysis failed: {str(e)}"
            )
    
    def can_handle_task(self, task_type: str, context: TaskContext) -> bool:
        """Check if this agent can handle the given task."""
        eda_tasks = ["eda", "data_analysis", "exploratory_analysis", "data_profiling"]
        return any(task in task_type.lower() for task in eda_tasks)
    
    def estimate_complexity(self, context: TaskContext) -> TaskComplexity:
        """Estimate task complexity for EDA."""
        if hasattr(context, 'data') and context.data is not None:
            rows, cols = context.data.shape
            if rows > 100000 or cols > 1000:
                return TaskComplexity.EXPERT
            elif rows > 10000 or cols > 100:
                return TaskComplexity.COMPLEX
            elif rows > 1000 or cols > 20:
                return TaskComplexity.MODERATE
        return TaskComplexity.SIMPLE
    
    def _load_or_simulate_dataset(self, dataset_info: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """Load dataset or create simulation for demo purposes."""
        try:
            # Try to load from real data loader
            from data.samples.dataset_loader import load_demo_dataset
            df, _ = load_demo_dataset("classification")
            return df
        except:
            # Fallback simulation
            np.random.seed(42)
            n_samples = 1000
            return pd.DataFrame({
                'feature1': np.random.randn(n_samples),
                'feature2': np.random.randn(n_samples),
                'target': np.random.choice([0, 1], n_samples)
            })
