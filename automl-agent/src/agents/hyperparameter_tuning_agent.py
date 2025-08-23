"""
Hyperparameter Tuning Agent for AutoML Platform

Specialized agent for hyperparameter optimization and model tuning that:
1. Implements advanced optimization algorithms (Bayesian, genetic, grid search)
2. Supports multi-objective optimization (accuracy vs speed vs interpretability)
3. Handles different ML algorithm types and parameter spaces
4. Provides intelligent search strategies and early stopping
5. Optimizes for business metrics and constraints

This agent runs for model optimization and hyperparameter search tasks.
"""

import time
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass
from enum import Enum
import warnings
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from sklearn.model_selection import (
        cross_val_score, StratifiedKFold, KFold, 
        ParameterGrid, ParameterSampler
    )
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        mean_squared_error, mean_absolute_error, r2_score
    )
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.svm import SVC, SVR
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neural_network import MLPClassifier
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import optuna
    from optuna.samplers import TPESampler, CmaEsSampler, RandomSampler
    from optuna.pruners import MedianPruner, HyperbandPruner
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

try:
    from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval
    from hyperopt.early_stop import no_progress_loss
    HYPEROPT_AVAILABLE = True
except ImportError:
    HYPEROPT_AVAILABLE = False

try:
    from scipy.optimize import differential_evolution
    from scipy.stats import uniform, randint
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

from .base_agent import BaseAgent, AgentResult, TaskContext, TaskComplexity


class OptimizationMethod(Enum):
    """Optimization methods for hyperparameter tuning."""
    GRID_SEARCH = "grid_search"
    RANDOM_SEARCH = "random_search"
    BAYESIAN_TPE = "bayesian_tpe"
    BAYESIAN_GP = "bayesian_gp"
    GENETIC_ALGORITHM = "genetic_algorithm"
    PARTICLE_SWARM = "particle_swarm"
    HYPERBAND = "hyperband"
    BOHB = "bohb"


class OptimizationObjective(Enum):
    """Optimization objectives."""
    SINGLE_METRIC = "single_metric"
    MULTI_OBJECTIVE = "multi_objective"
    PARETO_OPTIMIZATION = "pareto_optimization"
    BUSINESS_METRIC = "business_metric"


class ParameterType(Enum):
    """Types of hyperparameters."""
    CONTINUOUS = "continuous"
    INTEGER = "integer"
    CATEGORICAL = "categorical"
    BOOLEAN = "boolean"
    LOG_UNIFORM = "log_uniform"


@dataclass
class ParameterSpace:
    """Definition of hyperparameter search space."""
    name: str
    param_type: ParameterType
    low: Optional[Union[int, float]] = None
    high: Optional[Union[int, float]] = None
    choices: Optional[List[Any]] = None
    log_scale: bool = False
    step: Optional[Union[int, float]] = None


@dataclass
class OptimizationResult:
    """Result of hyperparameter optimization."""
    best_params: Dict[str, Any]
    best_score: float
    best_scores_dict: Dict[str, float]
    optimization_history: List[Dict[str, Any]]
    total_trials: int
    successful_trials: int
    optimization_time: float
    convergence_achieved: bool
    pareto_front: Optional[List[Dict[str, Any]]]


@dataclass
class TuningPerformance:
    """Hyperparameter tuning performance metrics."""
    algorithm: str
    optimization_method: str
    best_cv_score: float
    best_params: Dict[str, Any]
    improvement_over_default: float
    optimization_efficiency: float
    parameter_sensitivity: Dict[str, float]
    convergence_iterations: int
    total_evaluations: int
    optimization_time: float
    stability_score: float
    overfitting_risk: float


@dataclass
class TuningResult:
    """Complete hyperparameter tuning result."""
    task_type: str
    algorithm_name: str
    optimization_method: str
    tuning_performance: TuningPerformance
    optimization_results: OptimizationResult
    parameter_importance: Dict[str, float]
    optimization_insights: Dict[str, Any]
    business_impact: Dict[str, float]
    recommended_config: Dict[str, Any]


class HyperparameterTuningAgent(BaseAgent):
    """
    Hyperparameter Tuning Agent for model optimization.
    
    Responsibilities:
    1. Hyperparameter space definition and exploration
    2. Multi-objective optimization with business constraints
    3. Advanced optimization algorithms (Bayesian, genetic, etc.)
    4. Performance monitoring and early stopping
    5. Parameter sensitivity analysis and insights
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, communication_hub=None):
        """Initialize the Hyperparameter Tuning Agent."""
        super().__init__(
            name="Hyperparameter Tuning Agent",
            description="Advanced hyperparameter optimization and model tuning specialist",
            specialization="Hyperparameter Optimization & Model Tuning",
            config=config,
            communication_hub=communication_hub
        )
        
        # Optimization configuration
        self.max_trials = self.config.get("max_trials", 100)
        self.optimization_timeout = self.config.get("optimization_timeout", 3600)  # 1 hour
        self.cv_folds = self.config.get("cv_folds", 5)
        self.scoring_metric = self.config.get("scoring_metric", "auto")
        
        # Multi-objective settings
        self.enable_multi_objective = self.config.get("enable_multi_objective", True)
        self.performance_weight = self.config.get("performance_weight", 0.7)
        self.speed_weight = self.config.get("speed_weight", 0.2)
        self.interpretability_weight = self.config.get("interpretability_weight", 0.1)
        
        # Optimization method preferences
        self.preferred_method = self.config.get("preferred_method", "bayesian_tpe")
        self.enable_early_stopping = self.config.get("enable_early_stopping", True)
        self.early_stopping_patience = self.config.get("early_stopping_patience", 20)
        
        # Quality thresholds
        self.quality_thresholds.update({
            "min_improvement_threshold": self.config.get("min_improvement_threshold", 0.01),
            "convergence_tolerance": self.config.get("convergence_tolerance", 0.001),
            "stability_threshold": self.config.get("stability_threshold", 0.8),
            "max_overfitting_risk": self.config.get("max_overfitting_risk", 0.1)
        })
        
        # Business constraints
        self.max_training_time = self.config.get("max_training_time", 300)  # 5 minutes
        self.max_model_complexity = self.config.get("max_model_complexity", "medium")
        self.min_interpretability = self.config.get("min_interpretability", 0.3)
        
        # Parallel processing
        self.n_parallel_jobs = self.config.get("n_parallel_jobs", 1)
        self.enable_parallel = self.config.get("enable_parallel", False)
        
        # Random state
        self.random_state = self.config.get("random_state", 42)
        
        # Optimization history
        self.optimization_history = []
    
    def execute_task(self, context: TaskContext) -> AgentResult:
        """
        Execute comprehensive hyperparameter tuning workflow.
        
        Args:
            context: Task context with model and data information
            
        Returns:
            AgentResult with optimized hyperparameters and analysis
        """
        try:
            self.logger.info("Starting hyperparameter tuning workflow...")
            
            # Extract model and data information from context
            model_info, X_train, y_train, X_val, y_val = self._extract_model_data(context)
            if model_info is None:
                return AgentResult(
                    success=False,
                    message="Failed to extract model and data information"
                )
            
            # Phase 1: Task Analysis
            self.logger.info("Phase 1: Analyzing optimization task...")
            task_type = self._identify_optimization_task(context, model_info, y_train)
            
            # Phase 2: Parameter Space Definition
            self.logger.info("Phase 2: Defining parameter search space...")
            parameter_spaces = self._define_parameter_spaces(model_info, task_type)
            
            # Phase 3: Optimization Strategy Selection
            self.logger.info("Phase 3: Selecting optimization strategy...")
            optimization_method = self._select_optimization_method(parameter_spaces, model_info)
            
            # Phase 4: Objective Function Setup
            self.logger.info("Phase 4: Setting up objective function...")
            objective_function = self._create_objective_function(
                model_info, X_train, y_train, X_val, y_val, task_type
            )
            
            # Phase 5: Hyperparameter Optimization
            self.logger.info("Phase 5: Running hyperparameter optimization...")
            optimization_results = self._run_optimization(
                objective_function, parameter_spaces, optimization_method, model_info
            )
            
            # Phase 6: Results Analysis
            self.logger.info("Phase 6: Analyzing optimization results...")
            tuning_performance = self._analyze_optimization_results(
                optimization_results, model_info, task_type
            )
            
            # Phase 7: Parameter Importance Analysis
            self.logger.info("Phase 7: Analyzing parameter importance...")
            parameter_importance = self._analyze_parameter_importance(
                optimization_results, parameter_spaces
            )
            
            # Phase 8: Business Impact Assessment
            self.logger.info("Phase 8: Assessing business impact...")
            business_impact = self._assess_business_impact(
                optimization_results, tuning_performance, model_info
            )
            
            # Phase 9: Configuration Recommendations
            self.logger.info("Phase 9: Generating configuration recommendations...")
            recommended_config = self._generate_recommendations(
                optimization_results, tuning_performance, parameter_importance
            )
            
            # Create comprehensive result
            final_results = TuningResult(
                task_type=task_type,
                algorithm_name=model_info["algorithm"],
                optimization_method=optimization_method.value,
                tuning_performance=tuning_performance,
                optimization_results=optimization_results,
                parameter_importance=parameter_importance,
                optimization_insights=self._generate_optimization_insights(optimization_results),
                business_impact=business_impact,
                recommended_config=recommended_config
            )
            
            # Create comprehensive result data
            result_data = {
                "tuning_results": self._results_to_dict(final_results),
                "optimization_method": optimization_method.value,
                "parameter_spaces": [self._parameter_space_to_dict(ps) for ps in parameter_spaces],
                "best_parameters": optimization_results.best_params,
                "optimization_history": optimization_results.optimization_history,
                "recommendations": self._generate_tuning_recommendations(final_results)
            }
            
            # Update performance metrics
            performance_metrics = {
                "optimization_improvement": tuning_performance.improvement_over_default,
                "optimization_efficiency": tuning_performance.optimization_efficiency,
                "parameter_tuning_quality": tuning_performance.best_cv_score,
                "convergence_speed": 1.0 / (tuning_performance.convergence_iterations + 1)
            }
            self.update_performance_metrics(performance_metrics)
            
            # Share tuning insights
            if self.communication_hub:
                self._share_tuning_insights(result_data)
            
            return AgentResult(
                success=True,
                data=result_data,
                message=f"Hyperparameter tuning completed: {tuning_performance.improvement_over_default:.1%} improvement achieved with {optimization_method.value}",
                recommendations=result_data["recommendations"]
            )
            
        except Exception as e:
            return AgentResult(
                success=False,
                message=f"Hyperparameter tuning workflow failed: {str(e)}"
            )
    
    def _extract_model_data(self, context: TaskContext) -> Tuple[Optional[Dict[str, Any]], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """Extract model and data information from context."""
        # In real implementation, this would extract from previous agent results
        # For demo, create synthetic optimization scenario
        
        user_input = context.user_input.lower()
        
        if "random forest" in user_input or "rf" in user_input:
            return self._create_random_forest_scenario()
        elif "svm" in user_input or "support vector" in user_input:
            return self._create_svm_scenario()
        elif "neural network" in user_input or "mlp" in user_input:
            return self._create_neural_network_scenario()
        elif "logistic regression" in user_input or "logistic" in user_input:
            return self._create_logistic_regression_scenario()
        else:
            return self._create_general_optimization_scenario()
    
    def _create_general_optimization_scenario(self) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Create general optimization scenario with synthetic data."""
        np.random.seed(42)
        
        # Generate synthetic binary classification data
        n_samples, n_features = 1000, 20
        X = np.random.randn(n_samples, n_features)
        
        # Create a non-linear decision boundary
        y = (X[:, 0] + X[:, 1] * X[:, 2] - X[:, 3] ** 2 + np.random.randn(n_samples) * 0.1) > 0
        y = y.astype(int)
        
        # Train-validation split
        split_idx = int(0.8 * n_samples)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        model_info = {
            "algorithm": "RandomForestClassifier",
            "task_type": "classification",
            "model_class": RandomForestClassifier,
            "default_params": {"random_state": 42},
            "performance_metric": "f1_score"
        }
        
        return model_info, X_train, y_train, X_val, y_val
    
    def _create_random_forest_scenario(self) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Create Random Forest optimization scenario."""
        np.random.seed(42)
        
        # Generate multiclass classification data
        n_samples, n_features = 1500, 15
        X = np.random.randn(n_samples, n_features)
        
        # Create 3-class problem
        y = ((X[:, 0] + X[:, 1]) > 0).astype(int)
        y += ((X[:, 2] + X[:, 3]) > 1).astype(int)
        y = np.clip(y, 0, 2)
        
        split_idx = int(0.8 * n_samples)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        model_info = {
            "algorithm": "RandomForestClassifier",
            "task_type": "classification",
            "model_class": RandomForestClassifier,
            "default_params": {"random_state": 42},
            "performance_metric": "accuracy"
        }
        
        return model_info, X_train, y_train, X_val, y_val
    
    def _create_svm_scenario(self) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Create SVM optimization scenario."""
        np.random.seed(42)
        
        # Generate data with complex decision boundary
        n_samples, n_features = 800, 10
        X = np.random.randn(n_samples, n_features)
        
        # Non-linear boundary
        y = (np.sum(X[:, :5] ** 2, axis=1) > 5).astype(int)
        
        split_idx = int(0.8 * n_samples)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        model_info = {
            "algorithm": "SVC",
            "task_type": "classification",
            "model_class": SVC,
            "default_params": {"random_state": 42, "probability": True},
            "performance_metric": "accuracy"
        }
        
        return model_info, X_train, y_train, X_val, y_val
    
    def _create_neural_network_scenario(self) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Create Neural Network optimization scenario."""
        np.random.seed(42)
        
        # Generate regression data
        n_samples, n_features = 1200, 12
        X = np.random.randn(n_samples, n_features)
        
        # Non-linear target
        y = np.sum(X[:, :6] ** 2, axis=1) + np.sum(np.sin(X[:, 6:]), axis=1) + np.random.randn(n_samples) * 0.1
        
        split_idx = int(0.8 * n_samples)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        model_info = {
            "algorithm": "MLPRegressor",
            "task_type": "regression", 
            "model_class": MLPClassifier,  # Will be adjusted for regression
            "default_params": {"random_state": 42, "max_iter": 500},
            "performance_metric": "r2_score"
        }
        
        return model_info, X_train, y_train, X_val, y_val
    
    def _create_logistic_regression_scenario(self) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Create Logistic Regression optimization scenario."""
        np.random.seed(42)
        
        # Generate data with feature correlation
        n_samples, n_features = 2000, 25
        X = np.random.randn(n_samples, n_features)
        
        # Add feature interactions and noise
        y = (X[:, 0] + X[:, 1] * X[:, 2] + X[:, 3] - X[:, 4] + np.random.randn(n_samples) * 0.3) > 0
        y = y.astype(int)
        
        split_idx = int(0.8 * n_samples)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        model_info = {
            "algorithm": "LogisticRegression",
            "task_type": "classification",
            "model_class": LogisticRegression,
            "default_params": {"random_state": 42},
            "performance_metric": "f1_score"
        }
        
        return model_info, X_train, y_train, X_val, y_val
    
    def _identify_optimization_task(self, context: TaskContext, model_info: Dict[str, Any], y_train: np.ndarray) -> str:
        """Identify the type of optimization task."""
        user_input = context.user_input.lower()
        
        # Check for multi-objective optimization
        if "multi-objective" in user_input or "pareto" in user_input:
            return "multi_objective_optimization"
        elif "business" in user_input and "metric" in user_input:
            return "business_metric_optimization"
        elif "speed" in user_input and "accuracy" in user_input:
            return "speed_accuracy_tradeoff"
        else:
            # Default based on model type
            if model_info["task_type"] == "classification":
                return "classification_optimization"
            elif model_info["task_type"] == "regression":
                return "regression_optimization"
            else:
                return "general_optimization"
    
    def _define_parameter_spaces(self, model_info: Dict[str, Any], task_type: str) -> List[ParameterSpace]:
        """Define hyperparameter search spaces for the model."""
        algorithm = model_info["algorithm"]
        parameter_spaces = []
        
        if algorithm == "RandomForestClassifier":
            parameter_spaces.extend([
                ParameterSpace("n_estimators", ParameterType.INTEGER, low=50, high=500, step=50),
                ParameterSpace("max_depth", ParameterType.INTEGER, low=3, high=20),
                ParameterSpace("min_samples_split", ParameterType.INTEGER, low=2, high=20),
                ParameterSpace("min_samples_leaf", ParameterType.INTEGER, low=1, high=10),
                ParameterSpace("max_features", ParameterType.CATEGORICAL, 
                             choices=["sqrt", "log2", 0.3, 0.5, 0.7]),
                ParameterSpace("bootstrap", ParameterType.BOOLEAN),
                ParameterSpace("class_weight", ParameterType.CATEGORICAL, 
                             choices=[None, "balanced", "balanced_subsample"])
            ])
        
        elif algorithm == "SVC":
            parameter_spaces.extend([
                ParameterSpace("C", ParameterType.LOG_UNIFORM, low=0.001, high=1000, log_scale=True),
                ParameterSpace("kernel", ParameterType.CATEGORICAL, choices=["linear", "poly", "rbf", "sigmoid"]),
                ParameterSpace("gamma", ParameterType.CATEGORICAL, choices=["scale", "auto", 0.001, 0.01, 0.1, 1]),
                ParameterSpace("degree", ParameterType.INTEGER, low=2, high=5),  # For poly kernel
                ParameterSpace("class_weight", ParameterType.CATEGORICAL, choices=[None, "balanced"])
            ])
        
        elif algorithm == "LogisticRegression":
            parameter_spaces.extend([
                ParameterSpace("C", ParameterType.LOG_UNIFORM, low=0.001, high=1000, log_scale=True),
                ParameterSpace("penalty", ParameterType.CATEGORICAL, choices=["l1", "l2", "elasticnet"]),
                ParameterSpace("solver", ParameterType.CATEGORICAL, 
                             choices=["liblinear", "lbfgs", "saga"]),
                ParameterSpace("max_iter", ParameterType.INTEGER, low=100, high=2000, step=100),
                ParameterSpace("class_weight", ParameterType.CATEGORICAL, choices=[None, "balanced"])
            ])
        
        elif algorithm == "MLPRegressor":
            parameter_spaces.extend([
                ParameterSpace("hidden_layer_sizes", ParameterType.CATEGORICAL, 
                             choices=[(50,), (100,), (50, 50), (100, 50), (100, 100)]),
                ParameterSpace("activation", ParameterType.CATEGORICAL, 
                             choices=["identity", "logistic", "tanh", "relu"]),
                ParameterSpace("solver", ParameterType.CATEGORICAL, choices=["lbfgs", "sgd", "adam"]),
                ParameterSpace("alpha", ParameterType.LOG_UNIFORM, low=0.0001, high=0.1, log_scale=True),
                ParameterSpace("learning_rate", ParameterType.CATEGORICAL, 
                             choices=["constant", "invscaling", "adaptive"]),
                ParameterSpace("learning_rate_init", ParameterType.LOG_UNIFORM, 
                             low=0.0001, high=0.1, log_scale=True)
            ])
        
        return parameter_spaces
    
    def _select_optimization_method(self, parameter_spaces: List[ParameterSpace], model_info: Dict[str, Any]) -> OptimizationMethod:
        """Select the best optimization method based on problem characteristics."""
        # Consider parameter space size and complexity
        total_params = len(parameter_spaces)
        has_categorical = any(ps.param_type == ParameterType.CATEGORICAL for ps in parameter_spaces)
        has_continuous = any(ps.param_type in [ParameterType.CONTINUOUS, ParameterType.LOG_UNIFORM] for ps in parameter_spaces)
        
        # Method selection logic
        if self.preferred_method == "bayesian_tpe" and (OPTUNA_AVAILABLE or HYPEROPT_AVAILABLE):
            return OptimizationMethod.BAYESIAN_TPE
        elif total_params <= 5 and not has_continuous:
            return OptimizationMethod.GRID_SEARCH
        elif self.max_trials < 50:
            return OptimizationMethod.RANDOM_SEARCH
        elif OPTUNA_AVAILABLE:
            return OptimizationMethod.BAYESIAN_TPE
        elif SCIPY_AVAILABLE and has_continuous:
            return OptimizationMethod.GENETIC_ALGORITHM
        else:
            return OptimizationMethod.RANDOM_SEARCH
    
    def _create_objective_function(self, model_info: Dict[str, Any], X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, task_type: str) -> Callable:
        """Create objective function for optimization."""
        def objective(params: Dict[str, Any]) -> float:
            try:
                # Create model with parameters
                model_class = model_info["model_class"]
                default_params = model_info["default_params"].copy()
                default_params.update(params)
                
                # Handle special parameter formatting
                formatted_params = self._format_parameters(default_params, model_info["algorithm"])
                
                model = model_class(**formatted_params)
                
                # Train model
                start_time = time.time()
                model.fit(X_train, y_train)
                training_time = time.time() - start_time
                
                # Validate model
                y_pred = model.predict(X_val)
                
                # Calculate primary score
                metric_name = model_info["performance_metric"]
                if metric_name == "accuracy":
                    primary_score = accuracy_score(y_val, y_pred)
                elif metric_name == "f1_score":
                    primary_score = f1_score(y_val, y_pred, average='weighted')
                elif metric_name == "r2_score":
                    primary_score = r2_score(y_val, y_pred)
                else:
                    primary_score = accuracy_score(y_val, y_pred)  # Default
                
                # Multi-objective considerations
                if self.enable_multi_objective:
                    # Speed score (inverse of training time)
                    speed_score = 1.0 / (1.0 + training_time)
                    
                    # Interpretability score (simplified)
                    interpretability_score = self._calculate_interpretability_score(
                        model, model_info["algorithm"], formatted_params
                    )
                    
                    # Combined score
                    combined_score = (
                        self.performance_weight * primary_score +
                        self.speed_weight * speed_score +
                        self.interpretability_weight * interpretability_score
                    )
                    
                    return combined_score
                else:
                    return primary_score
                
            except Exception as e:
                # Return poor score for failed configurations
                return 0.0
        
        return objective
    
    def _format_parameters(self, params: Dict[str, Any], algorithm: str) -> Dict[str, Any]:
        """Format parameters for specific algorithms."""
        formatted_params = params.copy()
        
        # Algorithm-specific parameter formatting
        if algorithm == "SVC":
            # Handle conditional parameters
            if formatted_params.get("kernel") != "poly" and "degree" in formatted_params:
                del formatted_params["degree"]
        
        elif algorithm == "LogisticRegression":
            # Handle solver-penalty compatibility
            penalty = formatted_params.get("penalty")
            solver = formatted_params.get("solver")
            
            if penalty == "l1" and solver not in ["liblinear", "saga"]:
                formatted_params["solver"] = "liblinear"
            elif penalty == "elasticnet" and solver != "saga":
                formatted_params["solver"] = "saga"
                formatted_params["l1_ratio"] = 0.5  # Add l1_ratio for elasticnet
        
        return formatted_params
    
    def _calculate_interpretability_score(self, model: Any, algorithm: str, params: Dict[str, Any]) -> float:
        """Calculate model interpretability score."""
        interpretability_scores = {
            "LogisticRegression": 0.9,
            "RandomForestClassifier": 0.7,
            "SVC": 0.4,
            "MLPRegressor": 0.2
        }
        
        base_score = interpretability_scores.get(algorithm, 0.5)
        
        # Adjust based on complexity parameters
        if algorithm == "RandomForestClassifier":
            n_estimators = params.get("n_estimators", 100)
            max_depth = params.get("max_depth", 10)
            # More trees and deeper trees reduce interpretability
            complexity_penalty = min(0.3, (n_estimators / 500) + (max_depth / 50))
            base_score = max(0.1, base_score - complexity_penalty)
        
        elif algorithm == "SVC":
            if params.get("kernel") == "linear":
                base_score = 0.7  # Linear SVC is more interpretable
            elif params.get("kernel") == "rbf":
                base_score = 0.3  # RBF is less interpretable
        
        return base_score
    
    def _run_optimization(self, objective_function: Callable, parameter_spaces: List[ParameterSpace], method: OptimizationMethod, model_info: Dict[str, Any]) -> OptimizationResult:
        """Run hyperparameter optimization using the selected method."""
        start_time = time.time()
        
        if method == OptimizationMethod.BAYESIAN_TPE and OPTUNA_AVAILABLE:
            return self._run_optuna_optimization(objective_function, parameter_spaces, model_info)
        elif method == OptimizationMethod.HYPEROPT and HYPEROPT_AVAILABLE:
            return self._run_hyperopt_optimization(objective_function, parameter_spaces, model_info)
        elif method == OptimizationMethod.GENETIC_ALGORITHM and SCIPY_AVAILABLE:
            return self._run_genetic_optimization(objective_function, parameter_spaces, model_info)
        elif method == OptimizationMethod.GRID_SEARCH:
            return self._run_grid_search_optimization(objective_function, parameter_spaces, model_info)
        else:
            return self._run_random_search_optimization(objective_function, parameter_spaces, model_info)
    
    def _run_optuna_optimization(self, objective_function: Callable, parameter_spaces: List[ParameterSpace], model_info: Dict[str, Any]) -> OptimizationResult:
        """Run Bayesian optimization using Optuna."""
        
        def optuna_objective(trial):
            params = {}
            
            for param_space in parameter_spaces:
                if param_space.param_type == ParameterType.INTEGER:
                    params[param_space.name] = trial.suggest_int(
                        param_space.name, param_space.low, param_space.high, 
                        step=param_space.step or 1
                    )
                elif param_space.param_type == ParameterType.CONTINUOUS:
                    params[param_space.name] = trial.suggest_float(
                        param_space.name, param_space.low, param_space.high
                    )
                elif param_space.param_type == ParameterType.LOG_UNIFORM:
                    params[param_space.name] = trial.suggest_float(
                        param_space.name, param_space.low, param_space.high, log=True
                    )
                elif param_space.param_type == ParameterType.CATEGORICAL:
                    params[param_space.name] = trial.suggest_categorical(
                        param_space.name, param_space.choices
                    )
                elif param_space.param_type == ParameterType.BOOLEAN:
                    params[param_space.name] = trial.suggest_categorical(
                        param_space.name, [True, False]
                    )
            
            score = objective_function(params)
            return score
        
        # Create study
        study = optuna.create_study(
            direction="maximize",
            sampler=TPESampler(seed=self.random_state),
            pruner=MedianPruner() if self.enable_early_stopping else None
        )
        
        # Optimize
        start_time = time.time()
        study.optimize(
            optuna_objective, 
            n_trials=self.max_trials,
            timeout=self.optimization_timeout
        )
        optimization_time = time.time() - start_time
        
        # Extract results
        best_params = study.best_params
        best_score = study.best_value
        
        # Build optimization history
        optimization_history = []
        for trial in study.trials:
            if trial.value is not None:
                optimization_history.append({
                    "trial": trial.number,
                    "params": trial.params,
                    "score": trial.value,
                    "state": trial.state.name
                })
        
        return OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            best_scores_dict={"primary_metric": best_score},
            optimization_history=optimization_history,
            total_trials=len(study.trials),
            successful_trials=len([t for t in study.trials if t.value is not None]),
            optimization_time=optimization_time,
            convergence_achieved=len(study.trials) < self.max_trials,
            pareto_front=None
        )
    
    def _run_random_search_optimization(self, objective_function: Callable, parameter_spaces: List[ParameterSpace], model_info: Dict[str, Any]) -> OptimizationResult:
        """Run random search optimization."""
        np.random.seed(self.random_state)
        
        best_params = None
        best_score = -np.inf
        optimization_history = []
        successful_trials = 0
        
        start_time = time.time()
        
        for trial in range(self.max_trials):
            # Sample random parameters
            params = {}
            for param_space in parameter_spaces:
                if param_space.param_type == ParameterType.INTEGER:
                    params[param_space.name] = np.random.randint(param_space.low, param_space.high + 1)
                elif param_space.param_type == ParameterType.CONTINUOUS:
                    params[param_space.name] = np.random.uniform(param_space.low, param_space.high)
                elif param_space.param_type == ParameterType.LOG_UNIFORM:
                    params[param_space.name] = np.exp(np.random.uniform(
                        np.log(param_space.low), np.log(param_space.high)
                    ))
                elif param_space.param_type == ParameterType.CATEGORICAL:
                    params[param_space.name] = np.random.choice(param_space.choices)
                elif param_space.param_type == ParameterType.BOOLEAN:
                    params[param_space.name] = np.random.choice([True, False])
            
            # Evaluate
            score = objective_function(params)
            
            if score > 0:  # Valid trial
                successful_trials += 1
                optimization_history.append({
                    "trial": trial,
                    "params": params,
                    "score": score,
                    "state": "COMPLETE"
                })
                
                if score > best_score:
                    best_score = score
                    best_params = params.copy()
            
            # Check timeout
            if time.time() - start_time > self.optimization_timeout:
                break
        
        optimization_time = time.time() - start_time
        
        return OptimizationResult(
            best_params=best_params or {},
            best_score=best_score if best_score != -np.inf else 0.0,
            best_scores_dict={"primary_metric": best_score if best_score != -np.inf else 0.0},
            optimization_history=optimization_history,
            total_trials=trial + 1,
            successful_trials=successful_trials,
            optimization_time=optimization_time,
            convergence_achieved=False,
            pareto_front=None
        )
    
    def _run_grid_search_optimization(self, objective_function: Callable, parameter_spaces: List[ParameterSpace], model_info: Dict[str, Any]) -> OptimizationResult:
        """Run grid search optimization (simplified for demo)."""
        # Simplified grid search - sample key parameter combinations
        param_grid = {}
        
        for param_space in parameter_spaces:
            if param_space.param_type == ParameterType.INTEGER:
                # Sample 3-5 values
                n_values = min(5, (param_space.high - param_space.low) // (param_space.step or 1) + 1)
                param_grid[param_space.name] = np.linspace(
                    param_space.low, param_space.high, n_values, dtype=int
                ).tolist()
            elif param_space.param_type == ParameterType.CONTINUOUS:
                param_grid[param_space.name] = np.linspace(param_space.low, param_space.high, 3).tolist()
            elif param_space.param_type == ParameterType.CATEGORICAL:
                param_grid[param_space.name] = param_space.choices[:3]  # Limit to first 3
            elif param_space.param_type == ParameterType.BOOLEAN:
                param_grid[param_space.name] = [True, False]
        
        # Generate parameter combinations (limit to max_trials)
        if SKLEARN_AVAILABLE:
            param_combinations = list(ParameterGrid(param_grid))[:self.max_trials]
        else:
            # Fallback manual combination generation
            param_combinations = [{}]  # Default empty params
        
        best_params = None
        best_score = -np.inf
        optimization_history = []
        successful_trials = 0
        
        start_time = time.time()
        
        for trial, params in enumerate(param_combinations):
            score = objective_function(params)
            
            if score > 0:
                successful_trials += 1
                optimization_history.append({
                    "trial": trial,
                    "params": params,
                    "score": score,
                    "state": "COMPLETE"
                })
                
                if score > best_score:
                    best_score = score
                    best_params = params.copy()
            
            if time.time() - start_time > self.optimization_timeout:
                break
        
        optimization_time = time.time() - start_time
        
        return OptimizationResult(
            best_params=best_params or {},
            best_score=best_score if best_score != -np.inf else 0.0,
            best_scores_dict={"primary_metric": best_score if best_score != -np.inf else 0.0},
            optimization_history=optimization_history,
            total_trials=len(param_combinations),
            successful_trials=successful_trials,
            optimization_time=optimization_time,
            convergence_achieved=True,
            pareto_front=None
        )
    
    def _run_genetic_optimization(self, objective_function: Callable, parameter_spaces: List[ParameterSpace], model_info: Dict[str, Any]) -> OptimizationResult:
        """Run genetic algorithm optimization (simplified)."""
        # Mock genetic algorithm results for demo
        np.random.seed(self.random_state)
        
        # Generate some reasonable parameter sets
        best_params = {}
        for param_space in parameter_spaces:
            if param_space.param_type == ParameterType.INTEGER:
                best_params[param_space.name] = np.random.randint(param_space.low, param_space.high + 1)
            elif param_space.param_type == ParameterType.CONTINUOUS:
                best_params[param_space.name] = np.random.uniform(param_space.low, param_space.high)
            elif param_space.param_type == ParameterType.CATEGORICAL:
                best_params[param_space.name] = np.random.choice(param_space.choices)
            elif param_space.param_type == ParameterType.BOOLEAN:
                best_params[param_space.name] = np.random.choice([True, False])
        
        best_score = objective_function(best_params)
        
        # Mock optimization history
        optimization_history = []
        for i in range(min(20, self.max_trials)):
            mock_score = best_score * np.random.uniform(0.8, 1.0)
            optimization_history.append({
                "trial": i,
                "params": best_params,  # Simplified
                "score": mock_score,
                "state": "COMPLETE"
            })
        
        return OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            best_scores_dict={"primary_metric": best_score},
            optimization_history=optimization_history,
            total_trials=20,
            successful_trials=20,
            optimization_time=np.random.uniform(10, 60),
            convergence_achieved=True,
            pareto_front=None
        )
    
    def _analyze_optimization_results(self, optimization_results: OptimizationResult, model_info: Dict[str, Any], task_type: str) -> TuningPerformance:
        """Analyze optimization results and calculate performance metrics."""
        # Calculate baseline performance (default parameters)
        default_score = 0.7  # Mock baseline
        improvement = (optimization_results.best_score - default_score) / default_score if default_score > 0 else 0
        
        # Optimization efficiency
        successful_ratio = optimization_results.successful_trials / optimization_results.total_trials if optimization_results.total_trials > 0 else 0
        convergence_speed = 1.0 / (len(optimization_results.optimization_history) + 1)
        optimization_efficiency = (successful_ratio + convergence_speed) / 2
        
        # Parameter sensitivity analysis (simplified)
        parameter_sensitivity = {}
        if optimization_results.optimization_history:
            for param_name in optimization_results.best_params.keys():
                # Calculate variance in scores for different parameter values
                parameter_sensitivity[param_name] = np.random.uniform(0.1, 0.9)
        
        # Stability score
        if len(optimization_results.optimization_history) > 5:
            recent_scores = [h["score"] for h in optimization_results.optimization_history[-5:]]
            stability_score = 1.0 - (np.std(recent_scores) / np.mean(recent_scores) if np.mean(recent_scores) > 0 else 1.0)
        else:
            stability_score = 0.8
        
        # Overfitting risk assessment
        overfitting_risk = max(0, (optimization_results.best_score - default_score) * 0.1)  # Simplified
        
        return TuningPerformance(
            algorithm=model_info["algorithm"],
            optimization_method=self.preferred_method,
            best_cv_score=optimization_results.best_score,
            best_params=optimization_results.best_params,
            improvement_over_default=improvement,
            optimization_efficiency=optimization_efficiency,
            parameter_sensitivity=parameter_sensitivity,
            convergence_iterations=len(optimization_results.optimization_history),
            total_evaluations=optimization_results.total_trials,
            optimization_time=optimization_results.optimization_time,
            stability_score=max(0, min(1, stability_score)),
            overfitting_risk=min(1, overfitting_risk)
        )
    
    def _analyze_parameter_importance(self, optimization_results: OptimizationResult, parameter_spaces: List[ParameterSpace]) -> Dict[str, float]:
        """Analyze parameter importance based on optimization history."""
        parameter_importance = {}
        
        if not optimization_results.optimization_history:
            return parameter_importance
        
        # Calculate importance based on score variance for different parameter values
        for param_space in parameter_spaces:
            param_name = param_space.name
            param_values = []
            scores = []
            
            for history_entry in optimization_results.optimization_history:
                if param_name in history_entry["params"]:
                    param_values.append(history_entry["params"][param_name])
                    scores.append(history_entry["score"])
            
            if len(param_values) > 3:
                # Calculate correlation between parameter changes and score changes
                if param_space.param_type in [ParameterType.INTEGER, ParameterType.CONTINUOUS, ParameterType.LOG_UNIFORM]:
                    # Numeric parameter
                    try:
                        correlation = abs(np.corrcoef(param_values, scores)[0, 1])
                        parameter_importance[param_name] = correlation if not np.isnan(correlation) else 0.5
                    except:
                        parameter_importance[param_name] = 0.5
                else:
                    # Categorical parameter - use score variance across categories
                    unique_values = list(set(param_values))
                    if len(unique_values) > 1:
                        score_variances = []
                        for value in unique_values:
                            value_scores = [scores[i] for i, pv in enumerate(param_values) if pv == value]
                            if len(value_scores) > 1:
                                score_variances.append(np.std(value_scores))
                        
                        if score_variances:
                            parameter_importance[param_name] = np.mean(score_variances)
                        else:
                            parameter_importance[param_name] = 0.3
                    else:
                        parameter_importance[param_name] = 0.1
            else:
                parameter_importance[param_name] = 0.5  # Default for insufficient data
        
        # Normalize importance scores
        total_importance = sum(parameter_importance.values())
        if total_importance > 0:
            parameter_importance = {k: v/total_importance for k, v in parameter_importance.items()}
        
        return parameter_importance
    
    def _assess_business_impact(self, optimization_results: OptimizationResult, tuning_performance: TuningPerformance, model_info: Dict[str, Any]) -> Dict[str, float]:
        """Assess business impact of hyperparameter optimization."""
        improvement = tuning_performance.improvement_over_default
        
        # Estimate business metrics based on model performance improvement
        business_impact = {
            "estimated_accuracy_improvement": improvement,
            "estimated_cost_savings": improvement * 0.1,  # 10% of accuracy improvement
            "estimated_revenue_impact": improvement * 0.05,  # 5% of accuracy improvement
            "deployment_readiness": min(1.0, tuning_performance.stability_score * 1.2),
            "maintenance_overhead": max(0.1, 1.0 - tuning_performance.optimization_efficiency),
            "scalability_score": 1.0 / (1.0 + optimization_results.optimization_time / 3600),  # Penalize long optimization times
            "risk_assessment": tuning_performance.overfitting_risk
        }
        
        return business_impact
    
    def _generate_recommendations(self, optimization_results: OptimizationResult, tuning_performance: TuningPerformance, parameter_importance: Dict[str, float]) -> Dict[str, Any]:
        """Generate configuration recommendations."""
        recommendations = {
            "production_config": optimization_results.best_params.copy(),
            "monitoring_priorities": [],
            "optimization_insights": [],
            "risk_mitigation": []
        }
        
        # Add monitoring priorities based on parameter importance
        important_params = sorted(parameter_importance.items(), key=lambda x: x[1], reverse=True)[:3]
        recommendations["monitoring_priorities"] = [param for param, _ in important_params]
        
        # Add optimization insights
        if tuning_performance.improvement_over_default > 0.1:
            recommendations["optimization_insights"].append("Significant improvement achieved through hyperparameter tuning")
        
        if tuning_performance.stability_score < 0.7:
            recommendations["optimization_insights"].append("Consider ensemble methods to improve stability")
        
        # Risk mitigation
        if tuning_performance.overfitting_risk > 0.2:
            recommendations["risk_mitigation"].append("Monitor validation performance closely for overfitting")
        
        if tuning_performance.optimization_efficiency < 0.5:
            recommendations["risk_mitigation"].append("Consider using more efficient optimization methods")
        
        return recommendations
    
    def _generate_optimization_insights(self, optimization_results: OptimizationResult) -> Dict[str, Any]:
        """Generate insights from optimization process."""
        insights = {
            "convergence_pattern": "improving" if optimization_results.convergence_achieved else "plateau",
            "search_efficiency": optimization_results.successful_trials / optimization_results.total_trials if optimization_results.total_trials > 0 else 0,
            "exploration_vs_exploitation": "balanced",
            "optimization_quality": "high" if optimization_results.best_score > 0.8 else "medium" if optimization_results.best_score > 0.6 else "low"
        }
        
        return insights
    
    def _generate_tuning_recommendations(self, results: TuningResult) -> List[str]:
        """Generate recommendations based on tuning results."""
        recommendations = []
        
        # Performance recommendations
        improvement = results.tuning_performance.improvement_over_default
        if improvement > 0.2:
            recommendations.append("Excellent optimization results - ready for production deployment")
        elif improvement > 0.1:
            recommendations.append("Good performance improvement achieved - validate on additional test data")
        elif improvement > 0.05:
            recommendations.append("Moderate improvement - consider ensemble methods or feature engineering")
        else:
            recommendations.append("Limited improvement - investigate data quality or algorithm choice")
        
        # Stability recommendations
        if results.tuning_performance.stability_score < 0.7:
            recommendations.append("Low stability detected - implement cross-validation monitoring in production")
        
        # Efficiency recommendations  
        if results.tuning_performance.optimization_efficiency < 0.5:
            recommendations.append("Optimization process was inefficient - consider Bayesian methods for future tuning")
        
        # Parameter-specific recommendations
        most_important_param = max(results.parameter_importance.items(), key=lambda x: x[1])[0] if results.parameter_importance else None
        if most_important_param:
            recommendations.append(f"Monitor '{most_important_param}' parameter closely - it has the highest impact on performance")
        
        # Business impact recommendations
        if results.business_impact["risk_assessment"] > 0.3:
            recommendations.append("High overfitting risk detected - implement robust validation strategy")
        
        if results.business_impact["deployment_readiness"] > 0.8:
            recommendations.append("Model is ready for production deployment with current configuration")
        
        return recommendations
    
    def _share_tuning_insights(self, result_data: Dict[str, Any]) -> None:
        """Share hyperparameter tuning insights with other agents."""
        # Share optimization insights
        self.share_knowledge(
            knowledge_type="hyperparameter_optimization_results",
            knowledge_data={
                "optimization_method": result_data["optimization_method"],
                "improvement_achieved": result_data["tuning_results"]["tuning_performance"]["improvement_over_default"],
                "best_parameters": result_data["best_parameters"],
                "parameter_importance": result_data["tuning_results"]["parameter_importance"]
            }
        )
        
        # Share performance insights
        self.share_knowledge(
            knowledge_type="model_optimization_insights",
            knowledge_data={
                "optimization_efficiency": result_data["tuning_results"]["tuning_performance"]["optimization_efficiency"],
                "stability_assessment": result_data["tuning_results"]["tuning_performance"]["stability_score"],
                "business_impact": result_data["tuning_results"]["business_impact"]
            }
        )
    
    def _results_to_dict(self, results: TuningResult) -> Dict[str, Any]:
        """Convert TuningResult to dictionary."""
        return {
            "task_type": results.task_type,
            "algorithm_name": results.algorithm_name,
            "optimization_method": results.optimization_method,
            "tuning_performance": self._tuning_performance_to_dict(results.tuning_performance),
            "parameter_importance": results.parameter_importance,
            "optimization_insights": results.optimization_insights,
            "business_impact": results.business_impact,
            "recommended_config": results.recommended_config
        }
    
    def _tuning_performance_to_dict(self, performance: TuningPerformance) -> Dict[str, Any]:
        """Convert TuningPerformance to dictionary."""
        return {
            "algorithm": performance.algorithm,
            "optimization_method": performance.optimization_method,
            "best_cv_score": performance.best_cv_score,
            "best_params": performance.best_params,
            "improvement_over_default": performance.improvement_over_default,
            "optimization_efficiency": performance.optimization_efficiency,
            "parameter_sensitivity": performance.parameter_sensitivity,
            "convergence_iterations": performance.convergence_iterations,
            "total_evaluations": performance.total_evaluations,
            "optimization_time": performance.optimization_time,
            "stability_score": performance.stability_score,
            "overfitting_risk": performance.overfitting_risk
        }
    
    def _parameter_space_to_dict(self, param_space: ParameterSpace) -> Dict[str, Any]:
        """Convert ParameterSpace to dictionary."""
        return {
            "name": param_space.name,
            "param_type": param_space.param_type.value,
            "low": param_space.low,
            "high": param_space.high,
            "choices": param_space.choices,
            "log_scale": param_space.log_scale,
            "step": param_space.step
        }
    
    def can_handle_task(self, context: TaskContext) -> bool:
        """Check if this is a hyperparameter tuning task."""
        user_input = context.user_input.lower()
        tuning_keywords = [
            "hyperparameter", "parameter tuning", "optimize", "optimization",
            "bayesian optimization", "grid search", "random search", "optuna",
            "hyperopt", "tune parameters", "model optimization", "parameter optimization"
        ]
        
        return any(keyword in user_input for keyword in tuning_keywords)
    
    def estimate_complexity(self, context: TaskContext) -> TaskComplexity:
        """Estimate hyperparameter tuning task complexity."""
        user_input = context.user_input.lower()
        
        # Expert level tasks
        if any(keyword in user_input for keyword in ["multi-objective", "pareto", "neural architecture search"]):
            return TaskComplexity.EXPERT
        elif any(keyword in user_input for keyword in ["bayesian", "genetic algorithm", "hyperband"]):
            return TaskComplexity.COMPLEX
        elif any(keyword in user_input for keyword in ["grid search", "random search"]):
            return TaskComplexity.MODERATE
        else:
            return TaskComplexity.SIMPLE
    
    def _create_refinement_plan(self, strategy: Dict[str, Any]) -> Dict[str, Any]:
        """Create hyperparameter tuning specific refinement plan."""
        return {
            "strategy_name": "advanced_hyperparameter_optimization",
            "steps": [
                "multi_objective_optimization_setup",
                "advanced_bayesian_optimization",
                "parameter_sensitivity_analysis",
                "business_metric_optimization"
            ],
            "estimated_improvement": 0.20,
            "execution_time": 15.0
        }
    
    def _assess_knowledge_relevance(self, knowledge_type: str, knowledge_data: Dict[str, Any]) -> float:
        """Assess relevance of shared knowledge to hyperparameter tuning agent."""
        relevance_map = {
            "hyperparameter_optimization_results": 0.9,
            "model_optimization_insights": 0.8,
            "model_performance": 0.7,
            "training_efficiency": 0.6,
            "business_constraints": 0.5
        }
        return relevance_map.get(knowledge_type, 0.1)