"""
Machine Learning Tools for AutoML Agents

Core ML functionality including model training, hyperparameter optimization,
and model evaluation tools used by specialized agents.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import warnings
import time

try:
    from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
        mean_squared_error, mean_absolute_error, r2_score,
        classification_report, confusion_matrix
    )
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.linear_model import LogisticRegression, LinearRegression
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

warnings.filterwarnings("ignore")


@dataclass
class ModelResult:
    """Model training and evaluation results."""
    model_name: str
    model_object: Any
    train_score: float
    validation_score: float
    test_score: float
    training_time: float
    hyperparameters: Dict[str, Any]
    feature_importance: Optional[Dict[str, float]] = None
    predictions: Optional[np.ndarray] = None
    model_size_mb: Optional[float] = None


@dataclass
class EvaluationReport:
    """Comprehensive model evaluation report."""
    task_type: str
    best_model: str
    best_score: float
    model_rankings: List[Dict[str, Any]]
    detailed_metrics: Dict[str, Dict[str, float]]
    cross_validation_scores: Dict[str, List[float]]
    recommendations: List[str]


class ModelTrainer:
    """
    Automated model training tool that handles different ML algorithms
    and provides consistent training interface for agents.
    """
    
    def __init__(self):
        """Initialize the model trainer."""
        self.trained_models = {}
        self.training_history = []
    
    def train_classification_models(self, X_train: pd.DataFrame, y_train: pd.Series,
                                  X_val: pd.DataFrame, y_val: pd.Series,
                                  algorithms: Optional[List[str]] = None) -> List[ModelResult]:
        """
        Train multiple classification models.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features  
            y_val: Validation target
            algorithms: List of algorithms to try
            
        Returns:
            List of ModelResult objects
        """
        if not SKLEARN_AVAILABLE:
            return self._create_mock_results("classification")
        
        if algorithms is None:
            algorithms = ["random_forest", "logistic_regression"]
        
        results = []
        
        for algo in algorithms:
            try:
                start_time = time.time()
                model, params = self._get_classification_model(algo)
                
                # Train model
                model.fit(X_train, y_train)
                training_time = time.time() - start_time
                
                # Evaluate
                train_score = model.score(X_train, y_train)
                val_score = model.score(X_val, y_val)
                
                # Feature importance (if available)
                feature_importance = None
                if hasattr(model, 'feature_importances_'):
                    feature_importance = dict(zip(X_train.columns, model.feature_importances_))
                
                result = ModelResult(
                    model_name=algo,
                    model_object=model,
                    train_score=train_score,
                    validation_score=val_score,
                    test_score=val_score,  # Using validation as test for now
                    training_time=training_time,
                    hyperparameters=params,
                    feature_importance=feature_importance
                )
                
                results.append(result)
                self.trained_models[algo] = model
                
            except Exception as e:
                print(f"Error training {algo}: {e}")
                continue
        
        self.training_history.extend(results)
        return results
    
    def train_regression_models(self, X_train: pd.DataFrame, y_train: pd.Series,
                               X_val: pd.DataFrame, y_val: pd.Series,
                               algorithms: Optional[List[str]] = None) -> List[ModelResult]:
        """
        Train multiple regression models.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
            algorithms: List of algorithms to try
            
        Returns:
            List of ModelResult objects
        """
        if not SKLEARN_AVAILABLE:
            return self._create_mock_results("regression")
        
        if algorithms is None:
            algorithms = ["random_forest", "linear_regression"]
        
        results = []
        
        for algo in algorithms:
            try:
                start_time = time.time()
                model, params = self._get_regression_model(algo)
                
                # Train model
                model.fit(X_train, y_train)
                training_time = time.time() - start_time
                
                # Evaluate
                train_score = model.score(X_train, y_train)
                val_score = model.score(X_val, y_val)
                
                # Feature importance (if available)
                feature_importance = None
                if hasattr(model, 'feature_importances_'):
                    feature_importance = dict(zip(X_train.columns, model.feature_importances_))
                
                result = ModelResult(
                    model_name=algo,
                    model_object=model,
                    train_score=train_score,
                    validation_score=val_score,
                    test_score=val_score,
                    training_time=training_time,
                    hyperparameters=params,
                    feature_importance=feature_importance
                )
                
                results.append(result)
                self.trained_models[algo] = model
                
            except Exception as e:
                print(f"Error training {algo}: {e}")
                continue
        
        self.training_history.extend(results)
        return results
    
    def _get_classification_model(self, algorithm: str) -> Tuple[Any, Dict]:
        """Get classification model and default parameters."""
        if algorithm == "random_forest":
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            params = {"n_estimators": 100, "random_state": 42}
        elif algorithm == "logistic_regression":
            model = LogisticRegression(random_state=42, max_iter=1000)
            params = {"random_state": 42, "max_iter": 1000}
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        return model, params
    
    def _get_regression_model(self, algorithm: str) -> Tuple[Any, Dict]:
        """Get regression model and default parameters."""
        if algorithm == "random_forest":
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            params = {"n_estimators": 100, "random_state": 42}
        elif algorithm == "linear_regression":
            model = LinearRegression()
            params = {}
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        return model, params
    
    def _create_mock_results(self, task_type: str) -> List[ModelResult]:
        """Create mock results when sklearn is not available."""
        mock_results = []
        
        algorithms = ["random_forest", "logistic_regression"] if task_type == "classification" else ["random_forest", "linear_regression"]
        
        for algo in algorithms:
            score = np.random.uniform(0.75, 0.95) if task_type == "classification" else np.random.uniform(0.7, 0.9)
            
            result = ModelResult(
                model_name=algo,
                model_object=None,
                train_score=score + 0.05,
                validation_score=score,
                test_score=score - 0.02,
                training_time=np.random.uniform(1, 10),
                hyperparameters={"mock": True}
            )
            mock_results.append(result)
        
        return mock_results


class HyperparameterOptimizer:
    """
    Automated hyperparameter optimization tool using various search strategies.
    """
    
    def __init__(self):
        """Initialize the hyperparameter optimizer."""
        self.optimization_history = []
    
    def optimize_model(self, model_type: str, X_train: pd.DataFrame, y_train: pd.Series,
                      search_type: str = "random", n_iterations: int = 20) -> Dict[str, Any]:
        """
        Optimize hyperparameters for a specific model.
        
        Args:
            model_type: Type of model to optimize
            X_train: Training features
            y_train: Training target  
            search_type: Type of search ("random" or "grid")
            n_iterations: Number of iterations for random search
            
        Returns:
            Dictionary with optimization results
        """
        if not SKLEARN_AVAILABLE:
            return self._create_mock_optimization_result(model_type)
        
        try:
            # Get model and parameter grid
            base_model, param_grid = self._get_model_and_params(model_type)
            
            # Perform optimization
            if search_type == "random":
                optimizer = RandomizedSearchCV(
                    base_model, param_grid, n_iter=n_iterations,
                    cv=3, scoring='accuracy', random_state=42, n_jobs=-1
                )
            else:
                optimizer = GridSearchCV(
                    base_model, param_grid, cv=3, scoring='accuracy', n_jobs=-1
                )
            
            start_time = time.time()
            optimizer.fit(X_train, y_train)
            optimization_time = time.time() - start_time
            
            result = {
                "model_type": model_type,
                "best_params": optimizer.best_params_,
                "best_score": optimizer.best_score_,
                "optimization_time": optimization_time,
                "n_trials": len(optimizer.cv_results_['params']),
                "best_model": optimizer.best_estimator_
            }
            
            self.optimization_history.append(result)
            return result
            
        except Exception as e:
            return {"error": str(e), "model_type": model_type}
    
    def _get_model_and_params(self, model_type: str) -> Tuple[Any, Dict]:
        """Get model and parameter grid for optimization."""
        if model_type == "random_forest_classifier":
            model = RandomForestClassifier(random_state=42)
            params = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 10, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        elif model_type == "random_forest_regressor":
            model = RandomForestRegressor(random_state=42)
            params = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 10, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        return model, params
    
    def _create_mock_optimization_result(self, model_type: str) -> Dict[str, Any]:
        """Create mock optimization result when sklearn is not available."""
        return {
            "model_type": model_type,
            "best_params": {"n_estimators": 100, "max_depth": 10},
            "best_score": np.random.uniform(0.8, 0.95),
            "optimization_time": np.random.uniform(10, 60),
            "n_trials": 20,
            "best_model": None,
            "mock": True
        }


class ModelEvaluator:
    """
    Comprehensive model evaluation tool for comparing and ranking models.
    """
    
    def __init__(self):
        """Initialize the model evaluator."""
        self.evaluation_history = []
    
    def evaluate_models(self, models: List[ModelResult], task_type: str) -> EvaluationReport:
        """
        Comprehensive evaluation of multiple models.
        
        Args:
            models: List of trained models
            task_type: Type of ML task
            
        Returns:
            EvaluationReport with comprehensive evaluation
        """
        # Rank models by validation score
        sorted_models = sorted(models, key=lambda x: x.validation_score, reverse=True)
        best_model = sorted_models[0] if sorted_models else None
        
        # Create model rankings
        rankings = []
        for i, model in enumerate(sorted_models):
            rankings.append({
                "rank": i + 1,
                "model_name": model.model_name,
                "validation_score": round(model.validation_score, 4),
                "training_time": round(model.training_time, 2),
                "overfit_score": round(abs(model.train_score - model.validation_score), 4)
            })
        
        # Detailed metrics
        detailed_metrics = {}
        for model in models:
            detailed_metrics[model.model_name] = {
                "train_score": round(model.train_score, 4),
                "validation_score": round(model.validation_score, 4),
                "test_score": round(model.test_score, 4),
                "training_time": round(model.training_time, 2)
            }
        
        # Generate recommendations
        recommendations = self._generate_recommendations(models, task_type)
        
        report = EvaluationReport(
            task_type=task_type,
            best_model=best_model.model_name if best_model else "None",
            best_score=best_model.validation_score if best_model else 0.0,
            model_rankings=rankings,
            detailed_metrics=detailed_metrics,
            cross_validation_scores={},  # Would need CV implementation
            recommendations=recommendations
        )
        
        self.evaluation_history.append(report)
        return report
    
    def _generate_recommendations(self, models: List[ModelResult], task_type: str) -> List[str]:
        """Generate recommendations based on model evaluation."""
        recommendations = []
        
        if not models:
            return ["No models to evaluate"]
        
        # Performance recommendations
        best_model = max(models, key=lambda x: x.validation_score)
        worst_model = min(models, key=lambda x: x.validation_score)
        
        performance_gap = best_model.validation_score - worst_model.validation_score
        if performance_gap > 0.1:
            recommendations.append(f"Significant performance difference: {best_model.model_name} outperforms {worst_model.model_name} by {performance_gap:.3f}")
        
        # Overfitting detection
        for model in models:
            overfit_score = abs(model.train_score - model.validation_score)
            if overfit_score > 0.1:
                recommendations.append(f"{model.model_name} shows signs of overfitting (gap: {overfit_score:.3f})")
        
        # Training time recommendations
        fast_models = [m for m in models if m.training_time < 5]
        if fast_models:
            fastest = min(fast_models, key=lambda x: x.training_time)
            recommendations.append(f"{fastest.model_name} is fastest to train ({fastest.training_time:.1f}s)")
        
        # Performance threshold recommendations
        target_score = 0.85 if task_type == "classification" else 0.8
        good_models = [m for m in models if m.validation_score >= target_score]
        if good_models:
            recommendations.append(f"{len(good_models)} models meet target performance (≥{target_score})")
        else:
            recommendations.append(f"No models meet target performance (≥{target_score}). Consider feature engineering or hyperparameter tuning.")
        
        return recommendations