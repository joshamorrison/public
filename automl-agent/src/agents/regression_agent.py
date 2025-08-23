"""
Regression Agent for AutoML Platform

Specialized agent for supervised regression tasks that:
1. Automatically selects and tests multiple regression algorithms
2. Performs hyperparameter optimization and model validation
3. Handles regression-specific challenges (outliers, non-linearity)
4. Provides comprehensive performance analysis and model interpretation
5. Implements advanced evaluation metrics for continuous targets

This agent runs after feature engineering for regression problems.
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
        cross_val_score, cross_validate, KFold,
        train_test_split, GridSearchCV, RandomizedSearchCV
    )
    from sklearn.metrics import (
        mean_squared_error, mean_absolute_error, r2_score,
        mean_absolute_percentage_error, explained_variance_score
    )
    from sklearn.ensemble import (
        RandomForestRegressor, GradientBoostingRegressor,
        AdaBoostRegressor, VotingRegressor, BaggingRegressor
    )
    from sklearn.linear_model import (
        LinearRegression, Ridge, Lasso, ElasticNet,
        BayesianRidge, HuberRegressor, SGDRegressor
    )
    from sklearn.svm import SVR
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.pipeline import Pipeline
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
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

from .base_agent import BaseAgent, AgentResult, TaskContext, TaskComplexity


class RegressionAlgorithm(Enum):
    """Available regression algorithms."""
    LINEAR_REGRESSION = "linear_regression"
    RIDGE = "ridge"
    LASSO = "lasso"
    ELASTIC_NET = "elastic_net"
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    CATBOOST = "catboost"
    SVR = "svr"
    KNN = "knn"
    DECISION_TREE = "decision_tree"
    HUBER = "huber"
    BAYESIAN_RIDGE = "bayesian_ridge"


class OptimizationMethod(Enum):
    """Hyperparameter optimization methods."""
    GRID_SEARCH = "grid_search"
    RANDOM_SEARCH = "random_search"
    BAYESIAN = "bayesian"
    OPTUNA = "optuna"


@dataclass
class RegressionPerformance:
    """Regression model performance metrics."""
    algorithm: str
    rmse: float
    mae: float
    r2_score: float
    mape: Optional[float]
    explained_variance: float
    cv_scores: List[float]
    cv_mean: float
    cv_std: float
    training_time: float
    prediction_time: float
    model_size_mb: float
    residual_analysis: Dict[str, Any]


@dataclass
class RegressionResult:
    """Complete regression result."""
    best_algorithm: str
    best_model: Any
    performance_metrics: RegressionPerformance
    all_model_performances: List[RegressionPerformance]
    feature_importance: Dict[str, float]
    residual_analysis: Dict[str, Any]
    hyperparameters: Dict[str, Any]
    target_distribution: Dict[str, Any]
    prediction_intervals: Optional[Dict[str, Any]]


class RegressionAgent(BaseAgent):
    """
    Regression Agent for supervised regression tasks.
    
    Responsibilities:
    1. Algorithm selection and comparison for continuous targets
    2. Hyperparameter optimization for regression models
    3. Cross-validation and performance evaluation
    4. Residual analysis and model diagnostics
    5. Feature importance and model interpretation
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, communication_hub=None):
        """Initialize the Regression Agent."""
        super().__init__(
            name="Regression Agent",
            description="Advanced supervised regression with automated algorithm selection",
            specialization="Regression & Continuous Prediction",
            config=config,
            communication_hub=communication_hub
        )
        
        # Algorithm configuration
        self.enabled_algorithms = self.config.get("enabled_algorithms", [
            "linear_regression", "ridge", "random_forest", "gradient_boosting", "xgboost"
        ])
        self.quick_mode = self.config.get("quick_mode", False)
        self.optimization_method = OptimizationMethod(self.config.get("optimization_method", "random_search"))
        
        # Cross-validation configuration
        self.cv_folds = self.config.get("cv_folds", 5)
        self.cv_scoring = self.config.get("cv_scoring", "neg_mean_squared_error")
        self.test_size = self.config.get("test_size", 0.2)
        self.random_state = self.config.get("random_state", 42)
        
        # Optimization settings
        self.optimization_timeout = self.config.get("optimization_timeout", 300)  # 5 minutes
        self.n_trials = self.config.get("n_trials", 100)
        self.n_jobs = self.config.get("n_jobs", -1)
        
        # Quality thresholds
        self.quality_thresholds.update({
            "min_r2_score": self.config.get("min_r2_score", 0.6),
            "max_rmse_ratio": self.config.get("max_rmse_ratio", 0.3),  # RMSE relative to target std
            "max_cv_std": self.config.get("max_cv_std", 0.1),
            "min_explained_variance": self.config.get("min_explained_variance", 0.5)
        })
        
        # Regression-specific settings
        self.handle_outliers = self.config.get("handle_outliers", True)
        self.polynomial_features = self.config.get("polynomial_features", False)
        self.feature_selection = self.config.get("feature_selection", True)
        
        # Model storage
        self.trained_models: Dict[str, Any] = {}
        self.model_performances: List[RegressionPerformance] = []
    
    def execute_task(self, context: TaskContext) -> AgentResult:
        """
        Execute comprehensive regression workflow.
        
        Args:
            context: Task context with engineered features
            
        Returns:
            AgentResult with regression models and performance metrics
        """
        try:
            self.logger.info("Starting regression workflow...")
            
            # Load engineered dataset
            df, target_variable = self._load_engineered_dataset(context)
            if df is None or target_variable is None:
                return AgentResult(
                    success=False,
                    message="Failed to load dataset or identify target variable for regression"
                )
            
            # Get feature engineering insights
            feature_insights = self._get_feature_insights()
            
            # Phase 1: Dataset Preparation
            self.logger.info("Phase 1: Preparing dataset for regression...")
            X, y, target_info = self._prepare_dataset(df, target_variable)
            
            # Phase 2: Train-Test Split
            self.logger.info("Phase 2: Splitting dataset...")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.test_size, random_state=self.random_state
            )
            
            # Phase 3: Algorithm Selection and Training
            self.logger.info("Phase 3: Training and evaluating regression algorithms...")
            algorithm_performances = self._train_and_evaluate_algorithms(
                X_train, y_train, X_test, y_test, target_info
            )
            
            # Phase 4: Select Best Model
            self.logger.info("Phase 4: Selecting best performing model...")
            best_model_info = self._select_best_model(algorithm_performances)
            
            # Phase 5: Hyperparameter Optimization
            self.logger.info("Phase 5: Optimizing hyperparameters...")
            optimized_model, optimized_performance = self._optimize_hyperparameters(
                best_model_info, X_train, y_train, X_test, y_test, target_info
            )
            
            # Phase 6: Final Model Evaluation
            self.logger.info("Phase 6: Final model evaluation and diagnostics...")
            final_results = self._final_model_evaluation(
                optimized_model, X_train, y_train, X_test, y_test, target_info
            )
            
            # Phase 7: Residual Analysis
            self.logger.info("Phase 7: Performing residual analysis...")
            residual_analysis = self._perform_residual_analysis(
                optimized_model, X_test, y_test
            )
            
            # Phase 8: Model Interpretation
            self.logger.info("Phase 8: Model interpretation and feature analysis...")
            interpretation_results = self._interpret_model(
                optimized_model, X_train.columns, final_results
            )
            
            # Create comprehensive result
            result_data = {
                "regression_results": self._results_to_dict(final_results),
                "model_performance": self._performance_to_dict(optimized_performance),
                "algorithm_comparison": [self._performance_to_dict(perf) for perf in algorithm_performances],
                "feature_importance": interpretation_results["feature_importance"],
                "residual_analysis": residual_analysis,
                "model_interpretation": interpretation_results,
                "dataset_info": {
                    "total_samples": len(X),
                    "training_samples": len(X_train),
                    "test_samples": len(X_test),
                    "feature_count": X.shape[1],
                    "target_statistics": target_info
                },
                "recommendations": self._generate_recommendations(final_results, optimized_performance, residual_analysis)
            }
            
            # Update performance metrics
            performance_metrics = {
                "regression_r2_score": optimized_performance.r2_score,
                "rmse_normalized": optimized_performance.rmse / target_info["std"] if target_info["std"] > 0 else 1.0,
                "mae_normalized": optimized_performance.mae / target_info["std"] if target_info["std"] > 0 else 1.0,
                "cv_stability": 1.0 - optimized_performance.cv_std,
                "explained_variance": optimized_performance.explained_variance
            }
            self.update_performance_metrics(performance_metrics)
            
            # Check quality thresholds and trigger refinement if needed
            quality_violations = self._check_quality_thresholds(optimized_performance, target_info)
            if quality_violations and self.communication_hub:
                self._request_refinement(quality_violations, optimized_performance)
            
            # Share regression insights
            if self.communication_hub:
                self._share_regression_insights(result_data)
            
            return AgentResult(
                success=True,
                data=result_data,
                message=f"Regression completed: {optimized_performance.algorithm} achieved R² = {optimized_performance.r2_score:.3f}",
                recommendations=result_data["recommendations"]
            )
            
        except Exception as e:
            return AgentResult(
                success=False,
                message=f"Regression workflow failed: {str(e)}"
            )
    
    def _load_engineered_dataset(self, context: TaskContext) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        """Load the engineered dataset from previous agent or simulate for demo."""
        # In real implementation, this would load from feature engineering agent results
        # For demo, create synthetic engineered data for regression
        
        if context.dataset_info:
            shape = context.dataset_info.get("shape", [1000, 20])
            columns = context.dataset_info.get("columns", [f"feature_{i}" for i in range(shape[1])])
        else:
            shape = [1000, 20]
            columns = [f"feature_{i}" for i in range(20)]
        
        # Create synthetic engineered features for regression
        np.random.seed(42)
        data = {}
        target_variable = None
        
        # Generate base features with some correlation structure
        base_features = np.random.multivariate_normal(
            mean=np.zeros(5),
            cov=np.eye(5) + 0.3 * np.ones((5, 5)),
            size=shape[0]
        )
        
        feature_idx = 0
        for i, col in enumerate(columns):
            if "target" in col.lower() or "price" in col.lower() or "value" in col.lower() or "amount" in col.lower():
                # Continuous target variable with realistic relationship to features
                target_variable = col
                # Create target as linear combination of features plus noise
                data[col] = (
                    2.5 * base_features[:, 0] +
                    1.8 * base_features[:, 1] +
                    -1.2 * base_features[:, 2] +
                    0.8 * base_features[:, 3] +
                    np.random.normal(0, 1, shape[0])
                ) * 10 + 50  # Scale and offset
            else:
                # Engineered features
                if "poly_" in col or "interact_" in col:
                    # Polynomial/interaction features
                    if feature_idx < 5:
                        data[col] = base_features[:, feature_idx % 5] ** 2
                    else:
                        data[col] = base_features[:, (feature_idx-5) % 5] * base_features[:, (feature_idx-3) % 5]
                elif "log_" in col:
                    # Log transformed features
                    data[col] = np.log1p(np.abs(base_features[:, feature_idx % 5]) + 1)
                elif "sqrt_" in col:
                    # Square root features
                    data[col] = np.sqrt(np.abs(base_features[:, feature_idx % 5]))
                elif "ratio_" in col:
                    # Ratio features
                    denominator = base_features[:, (feature_idx+1) % 5]
                    denominator = np.where(np.abs(denominator) < 0.1, 0.1, denominator)
                    data[col] = base_features[:, feature_idx % 5] / denominator
                else:
                    # Standard scaled features
                    data[col] = base_features[:, feature_idx % 5] + np.random.normal(0, 0.1, shape[0])
                
                feature_idx += 1
        
        # If no target found, create one
        if not target_variable:
            target_variable = "target"
            data[target_variable] = (
                2.0 * base_features[:, 0] +
                1.5 * base_features[:, 1] +
                -1.0 * base_features[:, 2] +
                np.random.normal(0, 1, shape[0])
            ) * 10 + 50
        
        return pd.DataFrame(data), target_variable
    
    def _get_feature_insights(self) -> Dict[str, Any]:
        """Get feature engineering insights from communication hub."""
        insights = {}
        
        if self.communication_hub:
            messages = self.communication_hub.get_messages(self.name)
            for message in messages:
                if message.message_type.value == "knowledge_share":
                    knowledge_type = message.content.get("knowledge_type")
                    if knowledge_type in ["feature_importance", "modeling_readiness"]:
                        insights[knowledge_type] = message.content.get("data", {})
        
        return insights
    
    def _prepare_dataset(self, df: pd.DataFrame, target_variable: str) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
        """Prepare dataset for regression."""
        # Separate features and target
        X = df.drop(columns=[target_variable])
        y = df[target_variable]
        
        # Analyze target distribution
        target_info = {
            "mean": float(y.mean()),
            "std": float(y.std()),
            "min": float(y.min()),
            "max": float(y.max()),
            "median": float(y.median()),
            "skewness": float(y.skew()),
            "kurtosis": float(y.kurtosis()),
            "range": float(y.max() - y.min()),
            "iqr": float(y.quantile(0.75) - y.quantile(0.25))
        }
        
        # Check for target outliers
        Q1 = y.quantile(0.25)
        Q3 = y.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outlier_count = ((y < lower_bound) | (y > upper_bound)).sum()
        target_info["outlier_percentage"] = float(outlier_count / len(y) * 100)
        
        return X, y, target_info
    
    def _train_and_evaluate_algorithms(self, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series, target_info: Dict[str, Any]) -> List[RegressionPerformance]:
        """Train and evaluate multiple regression algorithms."""
        algorithm_performances = []
        
        # Define algorithms to test
        algorithms = self._get_algorithms(target_info)
        
        for algo_name, model in algorithms.items():
            try:
                self.logger.info(f"Training {algo_name}...")
                start_time = time.time()
                
                # Train model
                model.fit(X_train, y_train)
                training_time = time.time() - start_time
                
                # Make predictions
                start_pred_time = time.time()
                y_pred = model.predict(X_test)
                prediction_time = time.time() - start_pred_time
                
                # Calculate performance metrics
                performance = self._calculate_performance_metrics(
                    model, X_train, y_train, X_test, y_test, y_pred,
                    algo_name, training_time, prediction_time, target_info
                )
                
                algorithm_performances.append(performance)
                self.trained_models[algo_name] = model
                
                self.logger.info(f"{algo_name} - R²: {performance.r2_score:.3f}, RMSE: {performance.rmse:.3f}")
                
            except Exception as e:
                self.logger.warning(f"Failed to train {algo_name}: {str(e)}")
                continue
        
        return algorithm_performances
    
    def _get_algorithms(self, target_info: Dict[str, Any]) -> Dict[str, Any]:
        """Get available regression algorithms."""
        algorithms = {}
        
        # Basic algorithms (always available if sklearn is installed)
        if SKLEARN_AVAILABLE:
            if "linear_regression" in self.enabled_algorithms:
                algorithms["Linear Regression"] = LinearRegression()
            
            if "ridge" in self.enabled_algorithms:
                algorithms["Ridge Regression"] = Ridge(random_state=self.random_state)
            
            if "lasso" in self.enabled_algorithms:
                algorithms["Lasso Regression"] = Lasso(random_state=self.random_state)
            
            if "elastic_net" in self.enabled_algorithms:
                algorithms["Elastic Net"] = ElasticNet(random_state=self.random_state)
            
            if "random_forest" in self.enabled_algorithms:
                algorithms["Random Forest"] = RandomForestRegressor(
                    n_estimators=100, random_state=self.random_state, n_jobs=self.n_jobs
                )
            
            if "gradient_boosting" in self.enabled_algorithms:
                algorithms["Gradient Boosting"] = GradientBoostingRegressor(
                    random_state=self.random_state
                )
            
            if "svr" in self.enabled_algorithms and not self.quick_mode:
                algorithms["Support Vector Regression"] = SVR()
            
            if "knn" in self.enabled_algorithms:
                algorithms["K-Nearest Neighbors"] = KNeighborsRegressor()
            
            if "decision_tree" in self.enabled_algorithms:
                algorithms["Decision Tree"] = DecisionTreeRegressor(
                    random_state=self.random_state
                )
            
            if "huber" in self.enabled_algorithms and target_info["outlier_percentage"] > 5:
                # Use robust regression if outliers detected
                algorithms["Huber Regressor"] = HuberRegressor()
            
            if "bayesian_ridge" in self.enabled_algorithms:
                algorithms["Bayesian Ridge"] = BayesianRidge()
        
        # Advanced algorithms (if available)
        if XGBOOST_AVAILABLE and "xgboost" in self.enabled_algorithms:
            algorithms["XGBoost"] = xgb.XGBRegressor(
                random_state=self.random_state, eval_metric='rmse'
            )
        
        if LIGHTGBM_AVAILABLE and "lightgbm" in self.enabled_algorithms:
            algorithms["LightGBM"] = lgb.LGBMRegressor(
                random_state=self.random_state, verbose=-1
            )
        
        if CATBOOST_AVAILABLE and "catboost" in self.enabled_algorithms:
            algorithms["CatBoost"] = CatBoostRegressor(
                random_state=self.random_state, verbose=False
            )
        
        return algorithms
    
    def _calculate_performance_metrics(self, model: Any, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series, y_pred: np.ndarray, algorithm: str, training_time: float, prediction_time: float, target_info: Dict[str, Any]) -> RegressionPerformance:
        """Calculate comprehensive performance metrics for a regression model."""
        import time
        import sys
        
        # Basic metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        explained_var = explained_variance_score(y_test, y_pred)
        
        # MAPE (handle division by zero)
        mape = None
        try:
            # Only calculate MAPE if no zero values in y_test
            if (y_test != 0).all():
                mape = mean_absolute_percentage_error(y_test, y_pred)
            else:
                # Alternative MAPE calculation for data with zeros
                non_zero_mask = y_test != 0
                if non_zero_mask.sum() > 0:
                    mape = np.mean(np.abs((y_test[non_zero_mask] - y_pred[non_zero_mask]) / y_test[non_zero_mask])) * 100
        except Exception:
            pass
        
        # Cross-validation scores
        cv_scores = []
        try:
            cv = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
            cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=self.cv_scoring, n_jobs=self.n_jobs)
            # Convert negative MSE scores to positive RMSE scores
            if self.cv_scoring == "neg_mean_squared_error":
                cv_scores = np.sqrt(-cv_scores)
        except Exception as e:
            self.logger.warning(f"Cross-validation failed for {algorithm}: {str(e)}")
            cv_scores = [rmse]  # Fallback to test RMSE
        
        # Residual analysis
        residuals = y_test - y_pred
        residual_analysis = {
            "mean_residual": float(np.mean(residuals)),
            "std_residual": float(np.std(residuals)),
            "residual_skewness": float(pd.Series(residuals).skew()),
            "residual_kurtosis": float(pd.Series(residuals).kurtosis()),
            "heteroscedasticity_test": self._test_heteroscedasticity(y_pred, residuals)
        }
        
        # Model size estimation
        model_size_mb = sys.getsizeof(model) / (1024 * 1024)
        
        return RegressionPerformance(
            algorithm=algorithm,
            rmse=rmse,
            mae=mae,
            r2_score=r2,
            mape=mape,
            explained_variance=explained_var,
            cv_scores=list(cv_scores),
            cv_mean=np.mean(cv_scores),
            cv_std=np.std(cv_scores),
            training_time=training_time,
            prediction_time=prediction_time,
            model_size_mb=model_size_mb,
            residual_analysis=residual_analysis
        )
    
    def _test_heteroscedasticity(self, y_pred: np.ndarray, residuals: np.ndarray) -> Dict[str, Any]:
        """Simple test for heteroscedasticity (non-constant variance)."""
        try:
            # Breusch-Pagan test approximation
            # Split predictions into groups and compare residual variances
            sorted_indices = np.argsort(y_pred)
            n = len(residuals)
            group_size = n // 3
            
            low_group_var = np.var(residuals[sorted_indices[:group_size]])
            high_group_var = np.var(residuals[sorted_indices[-group_size:]])
            
            # Ratio of variances (should be close to 1 for homoscedasticity)
            variance_ratio = high_group_var / low_group_var if low_group_var > 0 else 1.0
            
            is_heteroscedastic = variance_ratio > 2.0 or variance_ratio < 0.5
            
            return {
                "variance_ratio": float(variance_ratio),
                "is_heteroscedastic": bool(is_heteroscedastic),
                "test_type": "variance_ratio"
            }
            
        except Exception:
            return {
                "variance_ratio": 1.0,
                "is_heteroscedastic": False,
                "test_type": "failed"
            }
    
    def _select_best_model(self, performances: List[RegressionPerformance]) -> Dict[str, Any]:
        """Select the best performing model based on multiple criteria."""
        if not performances:
            raise ValueError("No models were successfully trained")
        
        # Sort by composite score considering multiple factors
        def score_model(perf: RegressionPerformance) -> float:
            # Weighted score considering multiple factors
            r2_weight = 0.4
            rmse_weight = 0.3
            stability_weight = 0.2  # Lower CV std is better
            efficiency_weight = 0.1  # Faster training is better
            
            r2_score = max(0, perf.r2_score)  # Ensure non-negative
            rmse_score = max(0, 1.0 - (perf.rmse / (perf.rmse + 1)))  # Normalize RMSE
            stability_score = max(0, 1 - perf.cv_std)  # Convert std to score
            efficiency_score = min(1.0, 60.0 / (perf.training_time + 1))  # Normalize training time
            
            total_score = (
                r2_weight * r2_score +
                rmse_weight * rmse_score +
                stability_weight * stability_score +
                efficiency_weight * efficiency_score
            )
            
            return total_score
        
        # Find best model
        best_performance = max(performances, key=score_model)
        best_model = self.trained_models[best_performance.algorithm]
        
        return {
            "performance": best_performance,
            "model": best_model,
            "algorithm_name": best_performance.algorithm
        }
    
    def _optimize_hyperparameters(self, best_model_info: Dict[str, Any], X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series, target_info: Dict[str, Any]) -> Tuple[Any, RegressionPerformance]:
        """Optimize hyperparameters for the best model."""
        model = best_model_info["model"]
        algorithm_name = best_model_info["algorithm_name"]
        
        # Skip optimization in quick mode or if no optimization method available
        if self.quick_mode or not SKLEARN_AVAILABLE:
            return model, best_model_info["performance"]
        
        try:
            self.logger.info(f"Optimizing hyperparameters for {algorithm_name}...")
            
            # Get hyperparameter search space
            param_space = self._get_hyperparameter_space(algorithm_name, target_info)
            
            if not param_space:
                self.logger.info("No hyperparameters to optimize")
                return model, best_model_info["performance"]
            
            # Perform optimization
            if self.optimization_method == OptimizationMethod.OPTUNA and OPTUNA_AVAILABLE:
                optimized_model = self._optuna_optimization(model, param_space, X_train, y_train)
            elif self.optimization_method == OptimizationMethod.RANDOM_SEARCH:
                optimized_model = self._random_search_optimization(model, param_space, X_train, y_train)
            else:
                optimized_model = self._grid_search_optimization(model, param_space, X_train, y_train)
            
            # Evaluate optimized model
            start_time = time.time()
            optimized_model.fit(X_train, y_train)
            training_time = time.time() - start_time
            
            start_pred_time = time.time()
            y_pred = optimized_model.predict(X_test)
            prediction_time = time.time() - start_pred_time
            
            optimized_performance = self._calculate_performance_metrics(
                optimized_model, X_train, y_train, X_test, y_test, y_pred,
                f"{algorithm_name} (Optimized)", training_time, prediction_time, target_info
            )
            
            self.logger.info(f"Optimization complete - R²: {optimized_performance.r2_score:.3f}")
            
            return optimized_model, optimized_performance
            
        except Exception as e:
            self.logger.warning(f"Hyperparameter optimization failed: {str(e)}")
            return model, best_model_info["performance"]
    
    def _get_hyperparameter_space(self, algorithm_name: str, target_info: Dict[str, Any]) -> Dict[str, Any]:
        """Get hyperparameter search space for the algorithm."""
        param_spaces = {
            "Random Forest": {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['auto', 'sqrt', 'log2']
            },
            "Ridge Regression": {
                'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]
            },
            "Lasso Regression": {
                'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]
            },
            "Elastic Net": {
                'alpha': [0.01, 0.1, 1.0, 10.0],
                'l1_ratio': [0.1, 0.5, 0.7, 0.9]
            },
            "Gradient Boosting": {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 0.9, 1.0]
            },
            "XGBoost": {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            },
            "Support Vector Regression": {
                'C': [0.1, 1, 10, 100],
                'epsilon': [0.01, 0.1, 0.2],
                'kernel': ['linear', 'rbf']
            }
        }
        
        return param_spaces.get(algorithm_name, {})
    
    def _random_search_optimization(self, model: Any, param_space: Dict[str, Any], X_train: pd.DataFrame, y_train: pd.Series) -> Any:
        """Perform random search hyperparameter optimization."""
        try:
            cv = KFold(n_splits=3, shuffle=True, random_state=self.random_state)
            
            random_search = RandomizedSearchCV(
                model, param_space, n_iter=20, cv=cv,
                scoring=self.cv_scoring, random_state=self.random_state,
                n_jobs=self.n_jobs, verbose=0
            )
            
            random_search.fit(X_train, y_train)
            return random_search.best_estimator_
            
        except Exception as e:
            self.logger.warning(f"Random search failed: {str(e)}")
            return model
    
    def _grid_search_optimization(self, model: Any, param_space: Dict[str, Any], X_train: pd.DataFrame, y_train: pd.Series) -> Any:
        """Perform grid search hyperparameter optimization."""
        try:
            cv = KFold(n_splits=3, shuffle=True, random_state=self.random_state)
            
            # Limit grid search to prevent excessive computation
            limited_param_space = {}
            for param, values in param_space.items():
                limited_param_space[param] = values[:3] if len(values) > 3 else values
            
            grid_search = GridSearchCV(
                model, limited_param_space, cv=cv,
                scoring=self.cv_scoring, n_jobs=self.n_jobs, verbose=0
            )
            
            grid_search.fit(X_train, y_train)
            return grid_search.best_estimator_
            
        except Exception as e:
            self.logger.warning(f"Grid search failed: {str(e)}")
            return model
    
    def _optuna_optimization(self, model: Any, param_space: Dict[str, Any], X_train: pd.DataFrame, y_train: pd.Series) -> Any:
        """Perform Optuna-based hyperparameter optimization."""
        if not OPTUNA_AVAILABLE:
            return self._random_search_optimization(model, param_space, X_train, y_train)
        
        try:
            def objective(trial):
                # Sample hyperparameters
                params = {}
                for param, values in param_space.items():
                    if isinstance(values[0], int):
                        params[param] = trial.suggest_int(param, min(values), max(values))
                    elif isinstance(values[0], float):
                        params[param] = trial.suggest_float(param, min(values), max(values))
                    else:
                        params[param] = trial.suggest_categorical(param, values)
                
                # Create model with sampled parameters
                temp_model = model.__class__(**{**model.get_params(), **params})
                
                # Cross-validation score
                cv = KFold(n_splits=3, shuffle=True, random_state=self.random_state)
                scores = cross_val_score(temp_model, X_train, y_train, cv=cv, scoring=self.cv_scoring)
                return scores.mean()
            
            # Create study and optimize
            study = optuna.create_study(direction='maximize' if 'neg_' in self.cv_scoring else 'minimize')
            study.optimize(objective, n_trials=min(self.n_trials, 50), timeout=self.optimization_timeout)
            
            # Create best model
            best_params = study.best_params
            optimized_model = model.__class__(**{**model.get_params(), **best_params})
            
            return optimized_model
            
        except Exception as e:
            self.logger.warning(f"Optuna optimization failed: {str(e)}")
            return self._random_search_optimization(model, param_space, X_train, y_train)
    
    def _final_model_evaluation(self, model: Any, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series, target_info: Dict[str, Any]) -> RegressionResult:
        """Perform final comprehensive evaluation of the best model."""
        # Final predictions
        y_pred = model.predict(X_test)
        
        # Performance metrics
        performance = self._calculate_performance_metrics(
            model, X_train, y_train, X_test, y_test, y_pred,
            "Final Model", 0.0, 0.0, target_info  # Times already recorded
        )
        
        # Feature importance
        feature_importance = self._get_feature_importance(model, X_train.columns)
        
        # Residual analysis
        residuals = y_test - y_pred
        residual_analysis = self._comprehensive_residual_analysis(y_test, y_pred, residuals)
        
        # Get model hyperparameters
        hyperparameters = model.get_params()
        
        # Prediction intervals (simplified)
        prediction_intervals = self._calculate_prediction_intervals(model, X_test, y_pred)
        
        return RegressionResult(
            best_algorithm=performance.algorithm,
            best_model=model,
            performance_metrics=performance,
            all_model_performances=self.model_performances,
            feature_importance=feature_importance,
            residual_analysis=residual_analysis,
            hyperparameters=hyperparameters,
            target_distribution=target_info,
            prediction_intervals=prediction_intervals
        )
    
    def _perform_residual_analysis(self, model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """Perform comprehensive residual analysis."""
        y_pred = model.predict(X_test)
        residuals = y_test - y_pred
        
        return self._comprehensive_residual_analysis(y_test, y_pred, residuals)
    
    def _comprehensive_residual_analysis(self, y_true: pd.Series, y_pred: np.ndarray, residuals: np.ndarray) -> Dict[str, Any]:
        """Perform comprehensive residual analysis."""
        analysis = {
            # Basic residual statistics
            "mean_residual": float(np.mean(residuals)),
            "std_residual": float(np.std(residuals)),
            "min_residual": float(np.min(residuals)),
            "max_residual": float(np.max(residuals)),
            "median_residual": float(np.median(residuals)),
            
            # Distribution properties
            "residual_skewness": float(pd.Series(residuals).skew()),
            "residual_kurtosis": float(pd.Series(residuals).kurtosis()),
            
            # Normality assessment
            "residuals_approximately_normal": abs(pd.Series(residuals).skew()) < 1.0,
            
            # Heteroscedasticity test
            "heteroscedasticity": self._test_heteroscedasticity(y_pred, residuals),
            
            # Outlier analysis
            "outlier_analysis": self._analyze_residual_outliers(residuals),
            
            # Model assumptions
            "assumptions_check": self._check_regression_assumptions(y_true, y_pred, residuals)
        }
        
        return analysis
    
    def _analyze_residual_outliers(self, residuals: np.ndarray) -> Dict[str, Any]:
        """Analyze outliers in residuals."""
        # Calculate outlier thresholds
        Q1 = np.percentile(residuals, 25)
        Q3 = np.percentile(residuals, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Find outliers
        outlier_mask = (residuals < lower_bound) | (residuals > upper_bound)
        outlier_count = outlier_mask.sum()
        outlier_percentage = (outlier_count / len(residuals)) * 100
        
        # Extreme outliers (3 IQR)
        extreme_lower = Q1 - 3 * IQR
        extreme_upper = Q3 + 3 * IQR
        extreme_mask = (residuals < extreme_lower) | (residuals > extreme_upper)
        extreme_count = extreme_mask.sum()
        
        return {
            "outlier_count": int(outlier_count),
            "outlier_percentage": float(outlier_percentage),
            "extreme_outlier_count": int(extreme_count),
            "outlier_threshold_lower": float(lower_bound),
            "outlier_threshold_upper": float(upper_bound),
            "has_significant_outliers": outlier_percentage > 5.0
        }
    
    def _check_regression_assumptions(self, y_true: pd.Series, y_pred: np.ndarray, residuals: np.ndarray) -> Dict[str, Any]:
        """Check key regression assumptions."""
        assumptions = {}
        
        # 1. Linearity (simplified check)
        # Calculate correlation between residuals and predictions
        residual_pred_corr = np.corrcoef(residuals, y_pred)[0, 1]
        assumptions["linearity"] = {
            "correlation": float(residual_pred_corr),
            "assumption_met": abs(residual_pred_corr) < 0.1
        }
        
        # 2. Independence (simplified check - no autocorrelation test here)
        assumptions["independence"] = {
            "assumption_met": True,  # Assume independence for now
            "note": "Independence assumed - time series analysis needed for temporal data"
        }
        
        # 3. Homoscedasticity
        hetero_test = self._test_heteroscedasticity(y_pred, residuals)
        assumptions["homoscedasticity"] = {
            "assumption_met": not hetero_test["is_heteroscedastic"],
            "variance_ratio": hetero_test["variance_ratio"]
        }
        
        # 4. Normality of residuals
        skewness = abs(pd.Series(residuals).skew())
        kurtosis = abs(pd.Series(residuals).kurtosis())
        assumptions["normality"] = {
            "assumption_met": skewness < 1.0 and kurtosis < 3.0,
            "skewness": float(skewness),
            "kurtosis": float(kurtosis)
        }
        
        # Overall assessment
        assumptions_met = sum([
            assumptions["linearity"]["assumption_met"],
            assumptions["independence"]["assumption_met"],
            assumptions["homoscedasticity"]["assumption_met"],
            assumptions["normality"]["assumption_met"]
        ])
        
        assumptions["overall_assessment"] = {
            "assumptions_met_count": assumptions_met,
            "total_assumptions": 4,
            "model_validity": "good" if assumptions_met >= 3 else "fair" if assumptions_met >= 2 else "poor"
        }
        
        return assumptions
    
    def _calculate_prediction_intervals(self, model: Any, X_test: pd.DataFrame, y_pred: np.ndarray) -> Optional[Dict[str, Any]]:
        """Calculate prediction intervals (simplified approach)."""
        try:
            # For tree-based models, we can get prediction std from individual trees
            if hasattr(model, 'estimators_'):
                # Ensemble method - calculate prediction variance
                predictions = np.array([tree.predict(X_test) for tree in model.estimators_])
                pred_std = np.std(predictions, axis=0)
                
                # 95% prediction intervals
                lower_bound = y_pred - 1.96 * pred_std
                upper_bound = y_pred + 1.96 * pred_std
                
                return {
                    "method": "ensemble_variance",
                    "confidence_level": 0.95,
                    "lower_bound": lower_bound.tolist()[:10],  # First 10 for demo
                    "upper_bound": upper_bound.tolist()[:10],
                    "average_interval_width": float(np.mean(upper_bound - lower_bound))
                }
            else:
                # For other models, use residual-based intervals
                # This is a simplified approach
                residual_std = np.std(y_pred)  # Approximate
                interval_width = 1.96 * residual_std
                
                return {
                    "method": "residual_based",
                    "confidence_level": 0.95,
                    "interval_width": float(interval_width),
                    "note": "Simplified interval calculation"
                }
                
        except Exception as e:
            self.logger.warning(f"Failed to calculate prediction intervals: {str(e)}")
            return None
    
    def _get_feature_importance(self, model: Any, feature_names: List[str]) -> Dict[str, float]:
        """Extract feature importance from the model."""
        try:
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
            elif hasattr(model, 'coef_'):
                # For linear models, use absolute coefficient values
                importances = np.abs(model.coef_)
            else:
                # Uniform importance if no method available
                importances = np.ones(len(feature_names)) / len(feature_names)
            
            return dict(zip(feature_names, importances))
            
        except Exception as e:
            self.logger.warning(f"Failed to extract feature importance: {str(e)}")
            return {name: 1.0/len(feature_names) for name in feature_names}
    
    def _interpret_model(self, model: Any, feature_names: List[str], results: RegressionResult) -> Dict[str, Any]:
        """Provide comprehensive model interpretation."""
        interpretation = {
            "feature_importance": results.feature_importance,
            "top_features": sorted(
                results.feature_importance.items(), 
                key=lambda x: x[1], reverse=True
            )[:10],
            "model_complexity": self._assess_model_complexity(model),
            "prediction_quality": self._assess_prediction_quality(results),
            "model_insights": self._generate_model_insights(model, results)
        }
        
        return interpretation
    
    def _assess_model_complexity(self, model: Any) -> Dict[str, Any]:
        """Assess model complexity."""
        complexity = {"type": "unknown", "interpretability": "medium"}
        
        model_name = model.__class__.__name__.lower()
        
        if "linear" in model_name or "ridge" in model_name or "lasso" in model_name:
            complexity.update({"type": "linear", "interpretability": "high"})
        elif "tree" in model_name:
            complexity.update({"type": "tree", "interpretability": "high"})
        elif "forest" in model_name or "boosting" in model_name:
            complexity.update({"type": "ensemble", "interpretability": "medium"})
        elif "svr" in model_name or "svm" in model_name:
            complexity.update({"type": "kernel", "interpretability": "low"})
        elif "neural" in model_name or "mlp" in model_name:
            complexity.update({"type": "neural", "interpretability": "low"})
        
        return complexity
    
    def _assess_prediction_quality(self, results: RegressionResult) -> Dict[str, Any]:
        """Assess prediction quality metrics."""
        perf = results.performance_metrics
        
        # Quality score based on multiple factors
        r2_score = max(0, perf.r2_score)
        rmse_score = max(0, 1.0 - (perf.rmse / (perf.rmse + 1)))
        stability_score = max(0, 1.0 - perf.cv_std)
        
        quality_score = (
            0.5 * r2_score +
            0.3 * rmse_score +
            0.2 * stability_score
        )
        
        quality_level = "excellent" if quality_score > 0.8 else "good" if quality_score > 0.6 else "fair" if quality_score > 0.4 else "poor"
        
        return {
            "quality_score": quality_score,
            "quality_level": quality_level,
            "r2_score": perf.r2_score,
            "rmse": perf.rmse,
            "cv_stability": 1.0 - perf.cv_std,
            "explained_variance": perf.explained_variance,
            "residual_analysis_summary": {
                "assumptions_met": results.residual_analysis.get("assumptions_check", {}).get("overall_assessment", {}).get("model_validity", "unknown"),
                "outlier_percentage": results.residual_analysis.get("outlier_analysis", {}).get("outlier_percentage", 0)
            }
        }
    
    def _generate_model_insights(self, model: Any, results: RegressionResult) -> List[str]:
        """Generate insights about the model."""
        insights = []
        perf = results.performance_metrics
        
        # Performance insights
        if perf.r2_score > 0.8:
            insights.append("Excellent model performance - explains >80% of target variance")
        elif perf.r2_score > 0.6:
            insights.append("Good model performance suitable for most applications")
        elif perf.r2_score > 0.3:
            insights.append("Moderate performance - consider feature engineering or ensemble methods")
        else:
            insights.append("Low performance - model may not be capturing underlying patterns")
        
        # Feature insights
        top_features = sorted(results.feature_importance.items(), key=lambda x: x[1], reverse=True)[:3]
        insights.append(f"Most predictive features: {[f[0] for f in top_features]}")
        
        # Model specific insights
        model_name = model.__class__.__name__
        if "Forest" in model_name:
            insights.append("Random Forest provides robust predictions with feature importance")
        elif "XGB" in model_name or "LightGBM" in model_name:
            insights.append("Gradient boosting model effective for complex non-linear patterns")
        elif any(name in model_name for name in ["Linear", "Ridge", "Lasso"]):
            insights.append("Linear model provides interpretable and stable predictions")
        
        # Residual analysis insights
        residual_analysis = results.residual_analysis
        assumptions = residual_analysis.get("assumptions_check", {})
        if assumptions.get("overall_assessment", {}).get("model_validity") == "good":
            insights.append("Model assumptions are well satisfied")
        elif residual_analysis.get("heteroscedasticity", {}).get("is_heteroscedastic"):
            insights.append("Heteroscedasticity detected - consider robust regression methods")
        
        # Cross-validation insights
        if perf.cv_std < 0.05:
            insights.append("Consistent performance across different data splits")
        elif perf.cv_std > 0.15:
            insights.append("High performance variance - consider regularization or more data")
        
        return insights
    
    def _check_quality_thresholds(self, performance: RegressionPerformance, target_info: Dict[str, Any]) -> List[str]:
        """Check if performance meets quality thresholds."""
        violations = []
        
        if performance.r2_score < self.quality_thresholds["min_r2_score"]:
            violations.append(f"r2_score: {performance.r2_score:.3f} < {self.quality_thresholds['min_r2_score']}")
        
        # Check RMSE relative to target standard deviation
        rmse_ratio = performance.rmse / target_info["std"] if target_info["std"] > 0 else 1.0
        if rmse_ratio > self.quality_thresholds["max_rmse_ratio"]:
            violations.append(f"rmse_ratio: {rmse_ratio:.3f} > {self.quality_thresholds['max_rmse_ratio']}")
        
        if performance.cv_std > self.quality_thresholds["max_cv_std"]:
            violations.append(f"cv_std: {performance.cv_std:.3f} > {self.quality_thresholds['max_cv_std']}")
        
        if performance.explained_variance < self.quality_thresholds["min_explained_variance"]:
            violations.append(f"explained_variance: {performance.explained_variance:.3f} < {self.quality_thresholds['min_explained_variance']}")
        
        return violations
    
    def _request_refinement(self, violations: List[str], performance: RegressionPerformance) -> None:
        """Request refinement from other agents when quality thresholds are not met."""
        # Request feature engineering refinement
        self.send_quality_feedback(
            target_agent="Feature Engineering Agent",
            performance_metrics={
                "r2_score": performance.r2_score,
                "rmse": performance.rmse,
                "explained_variance": performance.explained_variance,
                "cv_stability": 1.0 - performance.cv_std
            },
            thresholds=[
                {"metric": "r2_score", "threshold": self.quality_thresholds["min_r2_score"]},
                {"metric": "explained_variance", "threshold": self.quality_thresholds["min_explained_variance"]}
            ]
        )
        
        # Request data hygiene refinement if needed
        if performance.cv_std > 0.15:  # High variance suggests data quality issues
            self.send_quality_feedback(
                target_agent="Data Hygiene Agent",
                performance_metrics={"cv_stability": 1.0 - performance.cv_std},
                thresholds=[{"metric": "cv_stability", "threshold": 0.9}]
            )
    
    def _generate_recommendations(self, results: RegressionResult, performance: RegressionPerformance, residual_analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on regression results."""
        recommendations = []
        
        # Performance recommendations
        if performance.r2_score > 0.8:
            recommendations.append("Excellent performance achieved - model ready for production")
        elif performance.r2_score > 0.6:
            recommendations.append("Good performance - consider ensemble methods for improvement")
        else:
            recommendations.append("Performance below target - consider feature engineering, more data, or complex models")
        
        # Model selection recommendations
        if performance.cv_std > 0.1:
            recommendations.append("High variance detected - consider regularization or ensemble methods")
        
        # Residual analysis recommendations
        assumptions = residual_analysis.get("assumptions_check", {})
        if not assumptions.get("linearity", {}).get("assumption_met", True):
            recommendations.append("Non-linearity detected - consider polynomial features or tree-based models")
        
        if residual_analysis.get("heteroscedasticity", {}).get("is_heteroscedastic"):
            recommendations.append("Heteroscedasticity detected - consider robust regression or weighted least squares")
        
        if residual_analysis.get("outlier_analysis", {}).get("has_significant_outliers"):
            recommendations.append("Significant outliers detected - consider robust regression methods")
        
        # Feature recommendations
        top_features = sorted(results.feature_importance.items(), key=lambda x: x[1], reverse=True)
        if len(top_features) > 0 and top_features[0][1] > 0.3:
            recommendations.append(f"Strong predictive feature identified: {top_features[0][0]}")
        
        # Next steps
        recommendations.append("Consider ensemble modeling and advanced feature interactions")
        
        return recommendations
    
    def _share_regression_insights(self, result_data: Dict[str, Any]) -> None:
        """Share regression insights with other agents."""
        # Share model performance
        self.share_knowledge(
            knowledge_type="model_performance",
            knowledge_data={
                "best_algorithm": result_data["regression_results"]["best_algorithm"],
                "r2_score": result_data["model_performance"]["r2_score"],
                "rmse": result_data["model_performance"]["rmse"],
                "feature_importance": result_data["feature_importance"],
                "explained_variance": result_data["model_performance"]["explained_variance"]
            }
        )
        
        # Share residual analysis insights
        self.share_knowledge(
            knowledge_type="residual_analysis",
            knowledge_data={
                "model_assumptions": result_data["residual_analysis"]["assumptions_check"],
                "outlier_analysis": result_data["residual_analysis"]["outlier_analysis"],
                "prediction_quality": result_data["model_interpretation"]["prediction_quality"]
            }
        )
    
    def _results_to_dict(self, results: RegressionResult) -> Dict[str, Any]:
        """Convert RegressionResult to dictionary."""
        return {
            "best_algorithm": results.best_algorithm,
            "feature_importance": results.feature_importance,
            "residual_analysis": results.residual_analysis,
            "hyperparameters": results.hyperparameters,
            "target_distribution": results.target_distribution,
            "prediction_intervals": results.prediction_intervals
        }
    
    def _performance_to_dict(self, performance: RegressionPerformance) -> Dict[str, Any]:
        """Convert RegressionPerformance to dictionary."""
        return {
            "algorithm": performance.algorithm,
            "rmse": performance.rmse,
            "mae": performance.mae,
            "r2_score": performance.r2_score,
            "mape": performance.mape,
            "explained_variance": performance.explained_variance,
            "cv_mean": performance.cv_mean,
            "cv_std": performance.cv_std,
            "training_time": performance.training_time,
            "prediction_time": performance.prediction_time,
            "model_size_mb": performance.model_size_mb,
            "residual_analysis": performance.residual_analysis
        }
    
    def can_handle_task(self, context: TaskContext) -> bool:
        """Check if this is a regression task."""
        user_input = context.user_input.lower()
        regression_keywords = [
            "predict", "regression", "forecast", "estimate", "continuous",
            "price", "value", "amount", "revenue", "sales", "temperature",
            "age", "income", "score", "rating", "cost"
        ]
        
        # Also check for absence of classification keywords
        classification_keywords = ["classify", "classification", "category", "class", "binary"]
        
        has_regression_keywords = any(keyword in user_input for keyword in regression_keywords)
        has_classification_keywords = any(keyword in user_input for keyword in classification_keywords)
        
        return has_regression_keywords and not has_classification_keywords
    
    def estimate_complexity(self, context: TaskContext) -> TaskComplexity:
        """Estimate regression task complexity."""
        if not context.dataset_info:
            return TaskComplexity.MODERATE
        
        shape = context.dataset_info.get("shape", [1000, 10])
        rows, cols = shape
        
        # Complexity based on dataset size and number of features
        if rows > 100000 or cols > 100:
            return TaskComplexity.COMPLEX
        elif rows > 10000 or cols > 50:
            return TaskComplexity.MODERATE
        else:
            return TaskComplexity.SIMPLE
    
    def _create_refinement_plan(self, strategy: Dict[str, Any]) -> Dict[str, Any]:
        """Create regression specific refinement plan."""
        return {
            "strategy_name": "advanced_regression_optimization",
            "steps": [
                "ensemble_model_creation",
                "residual_analysis_optimization",
                "feature_transformation_enhancement",
                "regularization_tuning"
            ],
            "estimated_improvement": 0.18,
            "execution_time": 12.0
        }
    
    def _assess_knowledge_relevance(self, knowledge_type: str, knowledge_data: Dict[str, Any]) -> float:
        """Assess relevance of shared knowledge to regression agent."""
        relevance_map = {
            "feature_importance": 0.9,
            "modeling_readiness": 0.8,
            "data_quality_issues": 0.6,
            "preprocessing_results": 0.7,
            "residual_analysis": 0.8,
            "algorithm_performance": 0.5
        }
        return relevance_map.get(knowledge_type, 0.1)