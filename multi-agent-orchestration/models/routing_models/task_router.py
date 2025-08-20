"""
Task Router

Intelligent task routing and agent assignment system.
Routes tasks to optimal agents based on capabilities, performance, and workload.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime, timedelta
from collections import defaultdict
import json

logger = logging.getLogger(__name__)

class AgentPerformancePredictor:
    """
    Predicts agent performance for specific task types.
    Uses historical performance data to estimate execution time and success probability.
    """
    
    def __init__(self):
        """Initialize the performance predictor."""
        self.performance_history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.agent_capabilities: Dict[str, List[str]] = {}
        self.task_complexity_weights = {
            "research": 1.2,
            "analysis": 1.0, 
            "synthesis": 1.3,
            "review": 0.8,
            "coordination": 1.1
        }
    
    def record_performance(self, agent_id: str, task_type: str, 
                          execution_time: float, success: bool, 
                          confidence: float, metadata: Dict[str, Any] = None):
        """
        Record agent performance for a completed task.
        
        Args:
            agent_id: ID of the agent
            task_type: Type of task performed
            execution_time: Time taken to complete task (seconds)
            success: Whether task was successful
            confidence: Confidence score of the result
            metadata: Additional performance metadata
        """
        performance_record = {
            "timestamp": datetime.now().isoformat(),
            "task_type": task_type,
            "execution_time": execution_time,
            "success": success,
            "confidence": confidence,
            "metadata": metadata or {}
        }
        
        self.performance_history[agent_id].append(performance_record)
        
        # Keep only recent history (last 100 records per agent)
        if len(self.performance_history[agent_id]) > 100:
            self.performance_history[agent_id] = self.performance_history[agent_id][-100:]
    
    def predict_performance(self, agent_id: str, task_type: str) -> Dict[str, float]:
        """
        Predict agent performance for a specific task type.
        
        Args:
            agent_id: ID of the agent
            task_type: Type of task to predict performance for
            
        Returns:
            Dictionary with predicted metrics
        """
        if agent_id not in self.performance_history:
            # No history - return default predictions
            return {
                "estimated_time": 60.0,  # Default 1 minute
                "success_probability": 0.7,  # Conservative estimate
                "confidence_estimate": 0.6,
                "reliability_score": 0.5,
                "sample_size": 0
            }
        
        # Filter history for relevant task type
        relevant_history = [
            record for record in self.performance_history[agent_id]
            if record["task_type"] == task_type
        ]
        
        if not relevant_history:
            # No history for this task type - use overall agent performance
            relevant_history = self.performance_history[agent_id]
        
        if not relevant_history:
            # Still no history
            return {
                "estimated_time": 60.0,
                "success_probability": 0.7,
                "confidence_estimate": 0.6,
                "reliability_score": 0.5,
                "sample_size": 0
            }
        
        # Calculate metrics from history
        times = [record["execution_time"] for record in relevant_history]
        successes = [record["success"] for record in relevant_history]
        confidences = [record["confidence"] for record in relevant_history]
        
        # Apply task complexity weighting
        complexity_weight = self.task_complexity_weights.get(task_type, 1.0)
        
        predictions = {
            "estimated_time": np.mean(times) * complexity_weight,
            "success_probability": np.mean(successes),
            "confidence_estimate": np.mean(confidences),
            "reliability_score": self._calculate_reliability(relevant_history),
            "sample_size": len(relevant_history)
        }
        
        return predictions
    
    def _calculate_reliability(self, history: List[Dict[str, Any]]) -> float:
        """Calculate reliability score based on consistency and recency."""
        if not history:
            return 0.5
        
        # Recent performance weighs more heavily
        now = datetime.now()
        weighted_scores = []
        
        for record in history:
            timestamp = datetime.fromisoformat(record["timestamp"])
            age_days = (now - timestamp).days
            
            # Weight decreases with age
            weight = max(0.1, 1.0 - (age_days / 30))  # 30-day decay
            
            # Score combines success and confidence
            score = (record["success"] * 0.7) + (record["confidence"] * 0.3)
            weighted_scores.append(score * weight)
        
        return np.mean(weighted_scores) if weighted_scores else 0.5
    
    def get_agent_rankings(self, task_type: str, agent_ids: List[str]) -> List[Tuple[str, float]]:
        """
        Rank agents by predicted performance for a task type.
        
        Args:
            task_type: Type of task
            agent_ids: List of available agent IDs
            
        Returns:
            List of (agent_id, score) tuples sorted by score (descending)
        """
        rankings = []
        
        for agent_id in agent_ids:
            predictions = self.predict_performance(agent_id, task_type)
            
            # Composite score combining multiple factors
            score = (
                predictions["success_probability"] * 0.4 +
                predictions["confidence_estimate"] * 0.3 +
                predictions["reliability_score"] * 0.2 +
                min(1.0, predictions["sample_size"] / 10) * 0.1  # Experience bonus
            )
            
            # Penalize slow agents
            if predictions["estimated_time"] > 120:  # Over 2 minutes
                score *= 0.9
            
            rankings.append((agent_id, score))
        
        # Sort by score (descending)
        rankings.sort(key=lambda x: x[1], reverse=True)
        
        return rankings

class TaskRouter:
    """
    Main task routing system that assigns tasks to optimal agents.
    """
    
    def __init__(self):
        """Initialize the task router."""
        self.performance_predictor = AgentPerformancePredictor()
        self.agent_workload: Dict[str, int] = defaultdict(int)
        self.agent_capabilities: Dict[str, List[str]] = {}
        self.routing_rules: Dict[str, Dict[str, Any]] = self._load_default_rules()
        self.routing_history: List[Dict[str, Any]] = []
    
    def _load_default_rules(self) -> Dict[str, Dict[str, Any]]:
        """Load default routing rules."""
        return {
            "research": {
                "preferred_agents": ["researcher"],
                "min_capabilities": ["web_search", "data_analysis"],
                "max_workload": 3,
                "priority_weight": 1.0
            },
            "analysis": {
                "preferred_agents": ["analyst"],
                "min_capabilities": ["data_analysis", "statistical_analysis"],
                "max_workload": 2,
                "priority_weight": 1.1
            },
            "synthesis": {
                "preferred_agents": ["synthesizer"],
                "min_capabilities": ["content_generation", "summarization"],
                "max_workload": 2,
                "priority_weight": 1.2
            },
            "review": {
                "preferred_agents": ["critic"],
                "min_capabilities": ["quality_assessment", "feedback_generation"],
                "max_workload": 4,
                "priority_weight": 0.9
            },
            "coordination": {
                "preferred_agents": ["supervisor"],
                "min_capabilities": ["workflow_management", "delegation"],
                "max_workload": 1,
                "priority_weight": 1.3
            }
        }
    
    def register_agent(self, agent_id: str, agent_type: str, capabilities: List[str]):
        """
        Register an agent with its capabilities.
        
        Args:
            agent_id: Unique agent identifier
            agent_type: Type of agent
            capabilities: List of agent capabilities
        """
        self.agent_capabilities[agent_id] = capabilities
        if agent_id not in self.agent_workload:
            self.agent_workload[agent_id] = 0
        
        logger.info(f"Registered agent {agent_id} ({agent_type}) with capabilities: {capabilities}")
    
    def route_task(self, task: Dict[str, Any], available_agents: List[str] = None) -> Optional[str]:
        """
        Route a task to the optimal agent.
        
        Args:
            task: Task specification
            available_agents: List of available agent IDs (None for all)
            
        Returns:
            Selected agent ID or None if no suitable agent found
        """
        task_type = task.get("type", "general")
        task_priority = task.get("priority", "medium")
        
        # Use all registered agents if none specified
        if available_agents is None:
            available_agents = list(self.agent_capabilities.keys())
        
        # Filter agents by capabilities
        capable_agents = self._filter_by_capabilities(task_type, available_agents)
        
        if not capable_agents:
            logger.warning(f"No agents capable of handling task type: {task_type}")
            return None
        
        # Filter by workload
        available_agents = self._filter_by_workload(task_type, capable_agents)
        
        if not available_agents:
            logger.warning(f"All capable agents at maximum workload for task type: {task_type}")
            return None
        
        # Get performance rankings
        rankings = self.performance_predictor.get_agent_rankings(task_type, available_agents)
        
        # Apply routing rules and priority
        selected_agent = self._apply_routing_logic(task, rankings)
        
        if selected_agent:
            # Update workload and record routing decision
            self.agent_workload[selected_agent] += 1
            self._record_routing_decision(task, selected_agent, rankings)
        
        return selected_agent
    
    def _filter_by_capabilities(self, task_type: str, agent_ids: List[str]) -> List[str]:
        """Filter agents by required capabilities for task type."""
        rules = self.routing_rules.get(task_type, {})
        required_capabilities = rules.get("min_capabilities", [])
        
        capable_agents = []
        for agent_id in agent_ids:
            agent_caps = self.agent_capabilities.get(agent_id, [])
            
            # Check if agent has all required capabilities
            if all(cap in agent_caps for cap in required_capabilities):
                capable_agents.append(agent_id)
        
        return capable_agents
    
    def _filter_by_workload(self, task_type: str, agent_ids: List[str]) -> List[str]:
        """Filter agents by current workload limits."""
        rules = self.routing_rules.get(task_type, {})
        max_workload = rules.get("max_workload", 5)  # Default max
        
        available_agents = []
        for agent_id in agent_ids:
            current_workload = self.agent_workload.get(agent_id, 0)
            
            if current_workload < max_workload:
                available_agents.append(agent_id)
        
        return available_agents
    
    def _apply_routing_logic(self, task: Dict[str, Any], 
                           rankings: List[Tuple[str, float]]) -> Optional[str]:
        """Apply final routing logic to select agent."""
        if not rankings:
            return None
        
        task_type = task.get("type", "general")
        task_priority = task.get("priority", "medium")
        
        # Priority weights
        priority_weights = {
            "low": 0.8,
            "medium": 1.0,
            "high": 1.2,
            "urgent": 1.5
        }
        
        priority_weight = priority_weights.get(task_priority, 1.0)
        
        # Adjust scores by priority and routing rules
        adjusted_rankings = []
        rules = self.routing_rules.get(task_type, {})
        rule_weight = rules.get("priority_weight", 1.0)
        
        for agent_id, score in rankings:
            # Boost score for preferred agent types
            if any(pref in agent_id.lower() for pref in rules.get("preferred_agents", [])):
                score *= 1.1
            
            # Apply priority and rule weights
            final_score = score * priority_weight * rule_weight
            
            adjusted_rankings.append((agent_id, final_score))
        
        # Sort by adjusted score
        adjusted_rankings.sort(key=lambda x: x[1], reverse=True)
        
        # Return top agent
        return adjusted_rankings[0][0]
    
    def _record_routing_decision(self, task: Dict[str, Any], selected_agent: str, 
                               rankings: List[Tuple[str, float]]):
        """Record routing decision for analysis."""
        decision_record = {
            "timestamp": datetime.now().isoformat(),
            "task_type": task.get("type", "general"),
            "task_priority": task.get("priority", "medium"),
            "selected_agent": selected_agent,
            "available_agents": [agent for agent, _ in rankings],
            "ranking_scores": dict(rankings),
            "workload_at_decision": dict(self.agent_workload)
        }
        
        self.routing_history.append(decision_record)
        
        # Keep only recent history
        if len(self.routing_history) > 500:
            self.routing_history = self.routing_history[-500:]
    
    def complete_task(self, agent_id: str, task_type: str, execution_time: float, 
                     success: bool, confidence: float):
        """
        Mark task as completed and update metrics.
        
        Args:
            agent_id: ID of agent that completed the task
            task_type: Type of task that was completed
            execution_time: Time taken to complete
            success: Whether task was successful
            confidence: Confidence in the result
        """
        # Decrease workload
        if self.agent_workload[agent_id] > 0:
            self.agent_workload[agent_id] -= 1
        
        # Record performance for future predictions
        self.performance_predictor.record_performance(
            agent_id, task_type, execution_time, success, confidence
        )
        
        logger.info(f"Task completed by {agent_id}: {task_type} in {execution_time:.1f}s (success: {success})")
    
    def get_routing_statistics(self) -> Dict[str, Any]:
        """Get routing performance statistics."""
        if not self.routing_history:
            return {"message": "No routing history available"}
        
        # Analyze routing patterns
        task_type_counts = defaultdict(int)
        agent_assignment_counts = defaultdict(int)
        
        for record in self.routing_history:
            task_type_counts[record["task_type"]] += 1
            agent_assignment_counts[record["selected_agent"]] += 1
        
        return {
            "total_routings": len(self.routing_history),
            "task_type_distribution": dict(task_type_counts),
            "agent_assignment_distribution": dict(agent_assignment_counts),
            "current_workload": dict(self.agent_workload),
            "registered_agents": len(self.agent_capabilities)
        }