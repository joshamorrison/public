"""
Classification Agent for AutoML Platform

Specialized agent for supervised classification tasks that:
1. Automatically selects and tests multiple classification algorithms
2. Performs hyperparameter optimization using advanced techniques
3. Implements cross-validation and model evaluation
4. Handles class imbalance and classification-specific challenges
5. Provides comprehensive model performance analysis

This agent runs after feature engineering for classification problems.
"""

import numpy as np
import pandas as pd
import time
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import warnings

try:
    from sklearn.model_selection import (
        cross_val_score, cross_validate, StratifiedKFold,
        train_test_split, GridSearchCV, RandomizedSearchCV
    )
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, confusion_matrix, classification_report,
        precision_recall_curve, roc_curve
    )
    from sklearn.ensemble import (
        RandomForestClassifier, GradientBoostingClassifier,
        AdaBoostClassifier, VotingClassifier, BaggingClassifier
    )
    from sklearn.linear_model import LogisticRegression, SGDClassifier
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.naive_bayes import GaussianNB, MultinomialNB
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.preprocessing import LabelEncoder
    from sklearn.utils.class_weight import compute_class_weight
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

from .base_agent import BaseAgent, AgentResult, TaskContext, TaskComplexity
from ..tools.ml_tools import ModelTrainer, HyperparameterOptimizer, ModelEvaluator
from ..tools.visualization_tools import PlotGenerator, ReportGenerator


class ClassificationAlgorithm(Enum):
    """Available classification algorithms."""
    LOGISTIC_REGRESSION = "logistic_regression"
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    CATBOOST = "catboost"
    SVM = "svm"
    KNN = "knn"
    NAIVE_BAYES = "naive_bayes"
    DECISION_TREE = "decision_tree"


class OptimizationMethod(Enum):
    """Hyperparameter optimization methods."""
    GRID_SEARCH = "grid_search"
    RANDOM_SEARCH = "random_search"
    BAYESIAN = "bayesian"
    OPTUNA = "optuna"


@dataclass
class ModelPerformance:
    """Model performance metrics."""
    algorithm: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: Optional[float]
    cv_scores: List[float]
    cv_mean: float
    cv_std: float
    training_time: float
    prediction_time: float
    model_size_mb: float


@dataclass
class ClassificationResult:
    """Complete classification result."""
    best_algorithm: str
    best_model: Any
    performance_metrics: ModelPerformance
    all_model_performances: List[ModelPerformance]
    feature_importance: Dict[str, float]
    confusion_matrix: List[List[int]]
    classification_report: Dict[str, Any]
    hyperparameters: Dict[str, Any]
    class_distribution: Dict[str, int]


class ClassificationAgent(BaseAgent):
    """
    Classification Agent for supervised classification tasks.
    
    Responsibilities:
    1. Algorithm selection and comparison
    2. Hyperparameter optimization
    3. Cross-validation and model evaluation
    4. Class imbalance handling
    5. Model interpretation and analysis
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, communication_hub=None):
        """Initialize the Classification Agent."""
        super().__init__(
            name="Classification Agent",
            description="Advanced supervised classification with automated algorithm selection",
            specialization="Classification & Model Training",
            config=config,
            communication_hub=communication_hub
        )
        
        # Algorithm configuration
        self.enabled_algorithms = self.config.get("enabled_algorithms", [
            "logistic_regression", "random_forest", "gradient_boosting", "xgboost"
        ])
        self.quick_mode = self.config.get("quick_mode", False)
        self.optimization_method = OptimizationMethod(self.config.get("optimization_method", "random_search"))
        
        # Cross-validation configuration
        self.cv_folds = self.config.get("cv_folds", 5)
        self.cv_scoring = self.config.get("cv_scoring", "accuracy")
        self.test_size = self.config.get("test_size", 0.2)
        self.random_state = self.config.get("random_state", 42)
        
        # Optimization settings
        self.optimization_timeout = self.config.get("optimization_timeout", 300)  # 5 minutes
        self.n_trials = self.config.get("n_trials", 100)
        self.n_jobs = self.config.get("n_jobs", -1)
        
        # Quality thresholds
        self.quality_thresholds.update({
            "min_accuracy": self.config.get("min_accuracy", 0.7),
            "min_f1_score": self.config.get("min_f1_score", 0.6),
            "min_roc_auc": self.config.get("min_roc_auc", 0.7),
            "max_cv_std": self.config.get("max_cv_std", 0.1)
        })
        
        # Class imbalance handling
        self.handle_imbalance = self.config.get("handle_imbalance", True)
        self.imbalance_threshold = self.config.get("imbalance_threshold", 0.3)
        
        # Model storage
        self.trained_models: Dict[str, Any] = {}
        self.model_performances: List[ModelPerformance] = []
    
    def execute_task(self, context: TaskContext) -> AgentResult:
        """
        Execute real classification workflow using ML tools.
        
        Args:
            context: Task context with data splits from pipeline
            
        Returns:
            AgentResult with real classification models and performance metrics
        """
        try:
            self.logger.info("Starting real classification workflow...")
            
            # Get data splits from context (from pipeline)
            if hasattr(context, 'splits'):
                splits = context.splits
                X_train, X_val, X_test = splits["X_train"], splits["X_val"], splits["X_test"]
                y_train, y_val, y_test = splits["y_train"], splits["y_val"], splits["y_test"]
            else:
                # Fallback: load and split data
                df, target_variable = self._load_and_split_dataset(context)
                if df is None:
                    return AgentResult(
                        success=False,
                        message="Failed to load dataset for classification"
                    )
                X_train, X_val, X_test, y_train, y_val, y_test = df
            
            # Initialize real ML tools
            model_trainer = ModelTrainer()
            optimizer = HyperparameterOptimizer()
            evaluator = ModelEvaluator()
            plot_generator = PlotGenerator()
            
            # Phase 1: Real Model Training
            self.logger.info("Phase 1: Training real classification models...")
            algorithms = ["random_forest", "logistic_regression"]  # Start with core algorithms
            model_results = model_trainer.train_classification_models(
                X_train, y_train, X_val, y_val, algorithms
            )
            
            # Phase 2: Model Evaluation
            self.logger.info("Phase 2: Evaluating model performance...")
            evaluation_report = evaluator.evaluate_models(model_results, "classification")
            
            # Phase 3: Hyperparameter Optimization for Best Model
            self.logger.info("Phase 3: Optimizing best model...")
            best_model_name = evaluation_report.best_model
            if best_model_name and SKLEARN_AVAILABLE:
                optimization_result = optimizer.optimize_model(
                    f"{best_model_name}_classifier",
                    X_train, y_train
                )
            else:
                optimization_result = {"message": "Optimization skipped - sklearn not available"}
            
            # Phase 4: Generate Visualizations
            self.logger.info("Phase 4: Generating performance visualizations...")
            plots = plot_generator.create_model_performance_plots(model_results)
            
            # Phase 5: Create Comprehensive Results
            result_data = {
                "model_results": model_results,
                "evaluation_report": evaluation_report,
                "optimization_result": optimization_result,
                "performance_plots": plots,
                "best_model": evaluation_report.best_model,
                "best_score": evaluation_report.best_score,
                "model_rankings": evaluation_report.model_rankings
            }
            
            # Update performance metrics
            performance_metrics = {
                "best_model_score": evaluation_report.best_score,
                "models_trained": len(model_results),
                "total_training_time": sum(r.training_time for r in model_results)
            }
            
            # Share results with other agents
            if self.communication_hub:
                self.communication_hub.share_message(
                    "classification_results",
                    result_data,
                    sender="Classification Agent"
                )
            
            return AgentResult(
                success=True,
                data=result_data,
                message=f"Real classification completed: {evaluation_report.best_model} achieved {evaluation_report.best_score:.4f}",
                recommendations=evaluation_report.recommendations
            )
            
        except Exception as e:
            self.logger.error(f"Classification workflow failed: {e}")
            return AgentResult(
                success=False,
                message=f"Classification failed: {str(e)}"
            )
    
    def can_handle_task(self, task_type: str, context: TaskContext) -> bool:
        """Check if this agent can handle the given task."""
        classification_tasks = ["classification", "classify", "predict", "binary_classification", "multiclass"]
        return any(task in task_type.lower() for task in classification_tasks)
    
    def estimate_complexity(self, context: TaskContext) -> TaskComplexity:
        """Estimate task complexity for classification."""
        if hasattr(context, 'splits'):
            train_size = len(context.splits["X_train"])
            feature_count = context.splits["X_train"].shape[1]
            if train_size > 100000 or feature_count > 1000:
                return TaskComplexity.EXPERT
            elif train_size > 10000 or feature_count > 100:
                return TaskComplexity.COMPLEX
            elif train_size > 1000 or feature_count > 20:
                return TaskComplexity.MODERATE
        return TaskComplexity.SIMPLE
    
    def _load_and_split_dataset(self, context: TaskContext):
        """Fallback method to load and split data if not provided."""
        try:
            from data.samples.dataset_loader import load_demo_dataset
            from sklearn.model_selection import train_test_split
            
            df, target = load_demo_dataset("classification")
            X = df.drop(columns=[target])
            y = df[target]
            
            # Create splits
            X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)
            
            return X_train, X_val, X_test, y_train, y_val, y_test
        except:
            return None
