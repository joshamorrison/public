"""
Agent Performance Models

Models for predicting agent performance, workload optimization, and resource management.
"""

import numpy as np
from typing import Dict, List, Any, Tuple, Optional
import logging
from datetime import datetime, timedelta
import pickle

logger = logging.getLogger(__name__)

class AgentPerformanceModel:
    """
    Predicts agent performance metrics for different task types.
    """
    
    def __init__(self):
        """Initialize the performance model."""
        self.model_weights: Dict[str, np.ndarray] = {}
        self.feature_scalers: Dict[str, Dict[str, float]] = {}
        self.training_history: List[Dict[str, Any]] = []
        self.is_trained = False
    
    def extract_features(self, agent_data: Dict[str, Any], task_data: Dict[str, Any]) -> np.ndarray:
        """
        Extract numerical features for performance prediction.
        
        Args:
            agent_data: Agent information and capabilities
            task_data: Task specifications and requirements
            
        Returns:
            Feature vector for prediction
        """
        features = []
        
        # Agent features
        agent_experience = agent_data.get("tasks_completed", 0)
        agent_success_rate = agent_data.get("success_rate", 0.5)
        agent_avg_confidence = agent_data.get("average_confidence", 0.5)
        current_workload = agent_data.get("current_workload", 0)
        
        # Task features
        task_complexity = self._estimate_task_complexity(task_data)
        task_priority = self._encode_priority(task_data.get("priority", "medium"))
        required_capabilities_count = len(task_data.get("required_capabilities", []))
        
        # Capability match
        capability_match = self._calculate_capability_overlap(
            agent_data.get("capabilities", []),
            task_data.get("required_capabilities", [])
        )
        
        # Historical performance for this task type
        task_type = task_data.get("type", "general")
        task_type_performance = agent_data.get("task_type_metrics", {}).get(task_type, {})
        type_success_rate = task_type_performance.get("success_rate", agent_success_rate)
        type_avg_time = task_type_performance.get("average_time", 60.0)
        type_experience = task_type_performance.get("task_count", 0)
        
        # Combine features
        features.extend([
            agent_experience / 100.0,  # Normalize
            agent_success_rate,
            agent_avg_confidence,
            current_workload / 10.0,  # Normalize
            task_complexity,
            task_priority,
            required_capabilities_count / 10.0,  # Normalize
            capability_match,
            type_success_rate,
            type_avg_time / 300.0,  # Normalize (5 min max)
            type_experience / 20.0  # Normalize
        ])
        
        return np.array(features)
    
    def _estimate_task_complexity(self, task_data: Dict[str, Any]) -> float:
        """Estimate task complexity from 0 to 1."""
        complexity_indicators = {
            "description_length": len(task_data.get("description", "")),
            "requirements_count": len(task_data.get("requirements", {})),
            "deadline_urgency": self._calculate_urgency(task_data.get("deadline")),
            "quality_level": self._encode_quality_level(task_data.get("quality_level", "standard"))
        }
        
        # Normalize and combine
        normalized_length = min(1.0, complexity_indicators["description_length"] / 1000)
        normalized_requirements = min(1.0, complexity_indicators["requirements_count"] / 10)
        
        complexity = (
            normalized_length * 0.3 +
            normalized_requirements * 0.3 +
            complexity_indicators["deadline_urgency"] * 0.2 +
            complexity_indicators["quality_level"] * 0.2
        )
        
        return complexity
    
    def _encode_priority(self, priority: str) -> float:
        """Encode priority as numerical value."""
        priority_map = {"low": 0.2, "medium": 0.5, "high": 0.8, "urgent": 1.0}
        return priority_map.get(priority.lower(), 0.5)
    
    def _encode_quality_level(self, quality_level: str) -> float:
        """Encode quality level as numerical value."""
        quality_map = {"basic": 0.2, "standard": 0.5, "high": 0.8, "premium": 1.0}
        return quality_map.get(quality_level.lower(), 0.5)
    
    def _calculate_urgency(self, deadline: Optional[str]) -> float:
        """Calculate urgency based on deadline."""
        if not deadline:
            return 0.5  # No deadline = medium urgency
        
        try:
            deadline_dt = datetime.fromisoformat(deadline.replace('Z', '+00:00'))
            time_left = (deadline_dt - datetime.now()).total_seconds()
            
            if time_left <= 0:
                return 1.0  # Past deadline
            elif time_left <= 3600:  # 1 hour
                return 0.9
            elif time_left <= 24 * 3600:  # 1 day
                return 0.7
            elif time_left <= 7 * 24 * 3600:  # 1 week
                return 0.5
            else:
                return 0.3
        except:
            return 0.5
    
    def _calculate_capability_overlap(self, agent_caps: List[str], 
                                    required_caps: List[str]) -> float:
        """Calculate capability overlap ratio."""
        if not required_caps:
            return 1.0
        
        overlap = len(set(agent_caps) & set(required_caps))
        return overlap / len(required_caps)
    
    def train(self, training_data: List[Dict[str, Any]]):
        """
        Train the performance model on historical data.
        
        Args:
            training_data: List of training examples with agent_data, task_data, and outcomes
        """
        if len(training_data) < 10:
            logger.warning("Insufficient training data. Need at least 10 examples.")
            return
        
        # Extract features and targets
        X = []
        y_time = []
        y_success = []
        y_confidence = []
        
        for example in training_data:
            features = self.extract_features(example["agent_data"], example["task_data"])
            X.append(features)
            
            outcome = example["outcome"]
            y_time.append(outcome["execution_time"])
            y_success.append(1.0 if outcome["success"] else 0.0)
            y_confidence.append(outcome["confidence"])
        
        X = np.array(X)
        y_time = np.array(y_time)
        y_success = np.array(y_success)
        y_confidence = np.array(y_confidence)
        
        # Simple linear regression for each target
        self.model_weights["time"] = self._fit_linear_model(X, y_time)
        self.model_weights["success"] = self._fit_linear_model(X, y_success)
        self.model_weights["confidence"] = self._fit_linear_model(X, y_confidence)
        
        # Store feature scaling parameters
        self.feature_scalers = {
            "mean": np.mean(X, axis=0),
            "std": np.std(X, axis=0) + 1e-8  # Avoid division by zero
        }
        
        self.is_trained = True
        self.training_history.append({
            "timestamp": datetime.now().isoformat(),
            "samples": len(training_data),
            "features": X.shape[1]
        })
        
        logger.info(f"Trained performance model on {len(training_data)} examples")
    
    def _fit_linear_model(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Fit simple linear regression model."""
        # Add bias term
        X_with_bias = np.column_stack([np.ones(X.shape[0]), X])
        
        # Normal equation: w = (X^T X)^-1 X^T y
        try:
            weights = np.linalg.solve(X_with_bias.T @ X_with_bias, X_with_bias.T @ y)
        except np.linalg.LinAlgError:
            # Fallback to pseudoinverse if singular
            weights = np.linalg.pinv(X_with_bias.T @ X_with_bias) @ X_with_bias.T @ y
        
        return weights
    
    def predict(self, agent_data: Dict[str, Any], task_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Predict agent performance for a task.
        
        Args:
            agent_data: Agent information
            task_data: Task specification
            
        Returns:
            Predicted performance metrics
        """
        if not self.is_trained:
            # Return default predictions
            return {
                "predicted_time": 60.0,
                "success_probability": 0.7,
                "confidence_estimate": 0.6,
                "model_confidence": 0.0
            }
        
        # Extract and normalize features
        features = self.extract_features(agent_data, task_data)
        normalized_features = (features - self.feature_scalers["mean"]) / self.feature_scalers["std"]
        
        # Add bias term
        features_with_bias = np.concatenate([[1.0], normalized_features])
        
        # Make predictions
        predictions = {}
        for target, weights in self.model_weights.items():
            prediction = np.dot(features_with_bias, weights)
            
            # Apply constraints
            if target == "time":
                predictions["predicted_time"] = max(5.0, prediction)  # At least 5 seconds
            elif target == "success":
                predictions["success_probability"] = np.clip(prediction, 0.0, 1.0)
            elif target == "confidence":
                predictions["confidence_estimate"] = np.clip(prediction, 0.0, 1.0)
        
        # Model confidence based on training data similarity
        predictions["model_confidence"] = self._calculate_prediction_confidence(normalized_features)
        
        return predictions
    
    def _calculate_prediction_confidence(self, features: np.ndarray) -> float:
        """Calculate how confident the model is in its prediction."""
        # Simple approach: confidence decreases with distance from training data
        if not hasattr(self, '_training_features'):
            return 0.5
        
        # Distance to nearest training example
        distances = np.linalg.norm(self._training_features - features, axis=1)
        min_distance = np.min(distances)
        
        # Convert distance to confidence (closer = more confident)
        confidence = np.exp(-min_distance)
        return confidence

class WorkloadPredictor:
    """
    Predicts optimal workload distribution across agents.
    """
    
    def __init__(self):
        """Initialize the workload predictor."""
        self.agent_capacities: Dict[str, float] = {}
        self.historical_workloads: Dict[str, List[Tuple[datetime, int]]] = {}
        self.performance_curves: Dict[str, List[Tuple[int, float]]] = {}
    
    def record_agent_capacity(self, agent_id: str, max_concurrent_tasks: int):
        """Record agent's maximum capacity."""
        self.agent_capacities[agent_id] = float(max_concurrent_tasks)
    
    def record_workload_performance(self, agent_id: str, current_load: int, 
                                  performance_score: float):
        """
        Record how performance varies with workload.
        
        Args:
            agent_id: Agent identifier
            current_load: Current number of concurrent tasks
            performance_score: Performance score (0-1)
        """
        if agent_id not in self.performance_curves:
            self.performance_curves[agent_id] = []
        
        self.performance_curves[agent_id].append((current_load, performance_score))
        
        # Keep only recent data points
        if len(self.performance_curves[agent_id]) > 100:
            self.performance_curves[agent_id] = self.performance_curves[agent_id][-100:]
    
    def predict_optimal_load(self, agent_id: str) -> int:
        """
        Predict optimal workload for an agent.
        
        Args:
            agent_id: Agent to predict for
            
        Returns:
            Optimal number of concurrent tasks
        """
        if agent_id not in self.performance_curves:
            # Default to 70% of capacity
            capacity = self.agent_capacities.get(agent_id, 3)
            return max(1, int(capacity * 0.7))
        
        curve_data = self.performance_curves[agent_id]
        if not curve_data:
            capacity = self.agent_capacities.get(agent_id, 3)
            return max(1, int(capacity * 0.7))
        
        # Find load level with best performance
        best_load = 1
        best_performance = 0.0
        
        # Group by load level and average performance
        load_performance = {}
        for load, performance in curve_data:
            if load not in load_performance:
                load_performance[load] = []
            load_performance[load].append(performance)
        
        for load, performances in load_performance.items():
            avg_performance = np.mean(performances)
            if avg_performance > best_performance:
                best_performance = avg_performance
                best_load = load
        
        return best_load
    
    def distribute_tasks(self, available_agents: List[str], 
                        num_tasks: int) -> Dict[str, int]:
        """
        Distribute tasks optimally across available agents.
        
        Args:
            available_agents: List of available agent IDs
            num_tasks: Number of tasks to distribute
            
        Returns:
            Dictionary mapping agent_id to number of tasks assigned
        """
        if not available_agents:
            return {}
        
        distribution = {agent_id: 0 for agent_id in available_agents}
        
        # Get optimal loads for each agent
        optimal_loads = {}
        total_capacity = 0
        
        for agent_id in available_agents:
            optimal_load = self.predict_optimal_load(agent_id)
            optimal_loads[agent_id] = optimal_load
            total_capacity += optimal_load
        
        if total_capacity == 0:
            # Fallback: equal distribution
            tasks_per_agent = num_tasks // len(available_agents)
            remainder = num_tasks % len(available_agents)
            
            for i, agent_id in enumerate(available_agents):
                distribution[agent_id] = tasks_per_agent + (1 if i < remainder else 0)
        else:
            # Distribute proportionally to optimal loads
            remaining_tasks = num_tasks
            
            for agent_id in available_agents:
                if remaining_tasks <= 0:
                    break
                
                proportion = optimal_loads[agent_id] / total_capacity
                assigned_tasks = min(remaining_tasks, max(1, int(num_tasks * proportion)))
                
                distribution[agent_id] = assigned_tasks
                remaining_tasks -= assigned_tasks
            
            # Distribute any remaining tasks
            while remaining_tasks > 0:
                for agent_id in available_agents:
                    if remaining_tasks <= 0:
                        break
                    if distribution[agent_id] < optimal_loads[agent_id]:
                        distribution[agent_id] += 1
                        remaining_tasks -= 1
        
        return distribution