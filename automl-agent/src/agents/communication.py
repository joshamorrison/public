"""
Agent Communication System - Advanced Collaborative Intelligence

Enables sophisticated agent-to-agent communication for:
- Quality feedback loops
- Iterative refinement protocols  
- Performance-based workflow adaptation
- Collaborative problem solving
- Emergent intelligence through agent interaction
"""

import asyncio
import time
import json
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import logging

from .base_agent import BaseAgent, AgentResult, TaskContext


class MessageType(Enum):
    """Types of inter-agent messages."""
    QUALITY_FEEDBACK = "quality_feedback"
    REFINEMENT_REQUEST = "refinement_request"
    PERFORMANCE_UPDATE = "performance_update"
    COLLABORATION_OFFER = "collaboration_offer"
    KNOWLEDGE_SHARE = "knowledge_share"
    THRESHOLD_ALERT = "threshold_alert"
    OPTIMIZATION_SUGGESTION = "optimization_suggestion"
    VALIDATION_REQUEST = "validation_request"
    ERROR_NOTIFICATION = "error_notification"
    SUCCESS_CONFIRMATION = "success_confirmation"


class Priority(Enum):
    """Message priority levels."""
    CRITICAL = 1    # Immediate action required
    HIGH = 2        # Important, handle soon
    NORMAL = 3      # Standard priority
    LOW = 4         # Handle when convenient
    INFO = 5        # Informational only


@dataclass
class AgentMessage:
    """Inter-agent communication message."""
    from_agent: str
    to_agent: str
    message_type: MessageType
    priority: Priority
    content: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    message_id: str = field(default_factory=lambda: f"msg_{int(time.time()*1000)}")
    requires_response: bool = False
    correlation_id: Optional[str] = None  # For threading conversations
    expires_at: Optional[float] = None    # Message expiration


@dataclass
class QualityThreshold:
    """Quality threshold definition."""
    metric_name: str
    threshold_value: float
    comparison: str  # 'gt', 'lt', 'eq', 'gte', 'lte'
    action_on_fail: str  # 'refine', 'escalate', 'retry', 'abort'
    max_iterations: int = 3
    current_iterations: int = 0


@dataclass
class RefinementStrategy:
    """Strategy for iterative refinement."""
    strategy_name: str
    target_metric: str
    improvement_target: float
    max_iterations: int
    parameters: Dict[str, Any] = field(default_factory=dict)
    success_criteria: List[QualityThreshold] = field(default_factory=list)


class AgentCommunicationHub:
    """
    Central communication hub for agent-to-agent messaging and coordination.
    
    Handles:
    - Message routing and delivery
    - Quality threshold monitoring
    - Iterative refinement coordination
    - Performance feedback loops
    - Collaborative optimization protocols
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the communication hub."""
        self.config = config or {}
        
        # Message management
        self.message_queue: Dict[str, deque] = defaultdict(deque)  # agent_name -> messages
        self.message_history: List[AgentMessage] = []
        self.active_conversations: Dict[str, List[AgentMessage]] = {}  # correlation_id -> messages
        
        # Agent registry and status
        self.registered_agents: Dict[str, BaseAgent] = {}
        self.agent_status: Dict[str, Dict[str, Any]] = {}
        self.agent_capabilities: Dict[str, List[str]] = {}
        
        # Quality and refinement tracking
        self.quality_thresholds: Dict[str, List[QualityThreshold]] = defaultdict(list)
        self.active_refinements: Dict[str, RefinementStrategy] = {}
        self.performance_history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        
        # Collaboration patterns
        self.collaboration_patterns: Dict[str, Dict[str, Any]] = {}
        self.success_patterns: List[Dict[str, Any]] = []
        
        # Setup logging
        self.logger = logging.getLogger("agent_communication")
        self.logger.setLevel(logging.INFO)
        
    def register_agent(self, agent: BaseAgent, capabilities: List[str] = None) -> None:
        """Register an agent with the communication hub."""
        self.registered_agents[agent.name] = agent
        self.agent_capabilities[agent.name] = capabilities or []
        self.agent_status[agent.name] = {
            "status": "idle",
            "last_activity": time.time(),
            "performance_score": 0.0,
            "success_rate": 0.0
        }
        
        self.logger.info(f"Registered agent: {agent.name}")
    
    def send_message(self, message: AgentMessage) -> bool:
        """Send a message from one agent to another."""
        if message.to_agent not in self.registered_agents:
            self.logger.error(f"Cannot send message: {message.to_agent} not registered")
            return False
        
        # Add to recipient's queue
        self.message_queue[message.to_agent].append(message)
        
        # Add to history and conversation tracking
        self.message_history.append(message)
        if message.correlation_id:
            if message.correlation_id not in self.active_conversations:
                self.active_conversations[message.correlation_id] = []
            self.active_conversations[message.correlation_id].append(message)
        
        self.logger.info(
            f"Message sent: {message.from_agent} → {message.to_agent} "
            f"({message.message_type.value}, priority: {message.priority.value})"
        )
        return True
    
    def get_messages(self, agent_name: str, priority_filter: Optional[Priority] = None) -> List[AgentMessage]:
        """Get pending messages for an agent."""
        if agent_name not in self.message_queue:
            return []
        
        messages = []
        queue = self.message_queue[agent_name]
        
        # Process messages in priority order
        while queue:
            message = queue.popleft()
            
            # Check expiration
            if message.expires_at and time.time() > message.expires_at:
                continue
                
            # Apply priority filter
            if priority_filter and message.priority != priority_filter:
                queue.append(message)  # Put back for later
                continue
                
            messages.append(message)
        
        return sorted(messages, key=lambda m: m.priority.value)
    
    def initiate_quality_feedback_loop(
        self, 
        requesting_agent: str, 
        target_agent: str, 
        current_performance: Dict[str, float],
        thresholds: List[QualityThreshold]
    ) -> str:
        """Initiate a quality feedback loop between agents."""
        
        correlation_id = f"qfl_{int(time.time()*1000)}"
        
        # Store quality thresholds
        self.quality_thresholds[correlation_id] = thresholds
        
        # Send initial feedback request
        message = AgentMessage(
            from_agent=requesting_agent,
            to_agent=target_agent,
            message_type=MessageType.QUALITY_FEEDBACK,
            priority=Priority.HIGH,
            content={
                "current_performance": current_performance,
                "required_thresholds": [
                    {
                        "metric": t.metric_name,
                        "threshold": t.threshold_value,
                        "comparison": t.comparison,
                        "action": t.action_on_fail
                    }
                    for t in thresholds
                ],
                "feedback_type": "performance_improvement_needed"
            },
            correlation_id=correlation_id,
            requires_response=True
        )
        
        self.send_message(message)
        self.logger.info(f"Initiated quality feedback loop: {correlation_id}")
        return correlation_id
    
    def request_iterative_refinement(
        self,
        requesting_agent: str,
        target_agent: str, 
        strategy: RefinementStrategy
    ) -> str:
        """Request iterative refinement from another agent."""
        
        correlation_id = f"ref_{int(time.time()*1000)}"
        self.active_refinements[correlation_id] = strategy
        
        message = AgentMessage(
            from_agent=requesting_agent,
            to_agent=target_agent,
            message_type=MessageType.REFINEMENT_REQUEST,
            priority=Priority.HIGH,
            content={
                "strategy": {
                    "name": strategy.strategy_name,
                    "target_metric": strategy.target_metric,
                    "improvement_target": strategy.improvement_target,
                    "max_iterations": strategy.max_iterations,
                    "parameters": strategy.parameters
                },
                "success_criteria": [
                    {
                        "metric": sc.metric_name,
                        "threshold": sc.threshold_value,
                        "comparison": sc.comparison
                    }
                    for sc in strategy.success_criteria
                ]
            },
            correlation_id=correlation_id,
            requires_response=True
        )
        
        self.send_message(message)
        return correlation_id
    
    def share_knowledge(
        self,
        from_agent: str,
        knowledge_type: str,
        knowledge_data: Dict[str, Any],
        target_agents: Optional[List[str]] = None
    ) -> None:
        """Share knowledge/insights between agents."""
        
        # If no target agents specified, broadcast to relevant agents
        if target_agents is None:
            target_agents = self._find_relevant_agents(knowledge_type, knowledge_data)
        
        for target_agent in target_agents:
            if target_agent == from_agent:
                continue
                
            message = AgentMessage(
                from_agent=from_agent,
                to_agent=target_agent,
                message_type=MessageType.KNOWLEDGE_SHARE,
                priority=Priority.NORMAL,
                content={
                    "knowledge_type": knowledge_type,
                    "data": knowledge_data,
                    "confidence": knowledge_data.get("confidence", 1.0),
                    "context": knowledge_data.get("context", {})
                }
            )
            
            self.send_message(message)
    
    def _find_relevant_agents(self, knowledge_type: str, knowledge_data: Dict[str, Any]) -> List[str]:
        """Find agents that would benefit from specific knowledge."""
        relevant_agents = []
        
        # Knowledge routing logic based on type and content
        knowledge_routing = {
            "feature_importance": ["Classification Agent", "Regression Agent", "Feature Engineering Agent"],
            "data_quality_issues": ["Data Hygiene Agent", "EDA Agent"],
            "algorithm_performance": ["Classification Agent", "Regression Agent", "Hyperparameter Tuning Agent"],
            "preprocessing_effectiveness": ["Data Hygiene Agent", "Feature Engineering Agent"],
            "correlation_patterns": ["EDA Agent", "Feature Engineering Agent"],
            "outlier_patterns": ["Data Hygiene Agent", "EDA Agent"]
        }
        
        base_agents = knowledge_routing.get(knowledge_type, [])
        
        # Filter to only registered agents
        relevant_agents = [agent for agent in base_agents if agent in self.registered_agents]
        
        return relevant_agents
    
    def update_performance(self, agent_name: str, metrics: Dict[str, float]) -> None:
        """Update agent performance metrics and trigger feedback if needed."""
        
        # Store performance history
        self.performance_history[agent_name].append({
            "timestamp": time.time(),
            "metrics": metrics.copy()
        })
        
        # Update agent status
        if agent_name in self.agent_status:
            self.agent_status[agent_name]["performance_score"] = metrics.get("overall_score", 0.0)
            self.agent_status[agent_name]["last_activity"] = time.time()
        
        # Check if any active quality thresholds are violated
        self._check_quality_thresholds(agent_name, metrics)
        
        # Broadcast performance update to interested agents
        self._broadcast_performance_update(agent_name, metrics)
    
    def _check_quality_thresholds(self, agent_name: str, metrics: Dict[str, float]) -> None:
        """Check if current performance violates any quality thresholds."""
        
        for correlation_id, thresholds in self.quality_thresholds.items():
            for threshold in thresholds:
                metric_value = metrics.get(threshold.metric_name)
                if metric_value is None:
                    continue
                
                threshold_violated = False
                
                if threshold.comparison == "gt" and metric_value <= threshold.threshold_value:
                    threshold_violated = True
                elif threshold.comparison == "lt" and metric_value >= threshold.threshold_value:
                    threshold_violated = True
                elif threshold.comparison == "gte" and metric_value < threshold.threshold_value:
                    threshold_violated = True
                elif threshold.comparison == "lte" and metric_value > threshold.threshold_value:
                    threshold_violated = True
                
                if threshold_violated:
                    self._handle_threshold_violation(agent_name, threshold, metric_value, correlation_id)
    
    def _handle_threshold_violation(
        self, 
        agent_name: str, 
        threshold: QualityThreshold, 
        actual_value: float,
        correlation_id: str
    ) -> None:
        """Handle quality threshold violation."""
        
        threshold.current_iterations += 1
        
        if threshold.current_iterations >= threshold.max_iterations:
            # Max iterations reached, escalate
            self._escalate_quality_issue(agent_name, threshold, actual_value, correlation_id)
        else:
            # Request refinement based on action
            if threshold.action_on_fail == "refine":
                self._request_refinement(agent_name, threshold, actual_value, correlation_id)
            elif threshold.action_on_fail == "retry":
                self._request_retry(agent_name, threshold, correlation_id)
    
    def _request_refinement(
        self, 
        agent_name: str, 
        threshold: QualityThreshold, 
        actual_value: float,
        correlation_id: str
    ) -> None:
        """Request agent to refine their approach."""
        
        message = AgentMessage(
            from_agent="Communication Hub",
            to_agent=agent_name,
            message_type=MessageType.REFINEMENT_REQUEST,
            priority=Priority.HIGH,
            content={
                "threshold_violation": {
                    "metric": threshold.metric_name,
                    "required": threshold.threshold_value,
                    "actual": actual_value,
                    "gap": abs(threshold.threshold_value - actual_value)
                },
                "refinement_suggestion": self._generate_refinement_suggestion(
                    agent_name, threshold, actual_value
                ),
                "iteration": threshold.current_iterations,
                "max_iterations": threshold.max_iterations
            },
            correlation_id=correlation_id,
            requires_response=True
        )
        
        self.send_message(message)
    
    def _generate_refinement_suggestion(
        self, 
        agent_name: str, 
        threshold: QualityThreshold, 
        actual_value: float
    ) -> Dict[str, Any]:
        """Generate intelligent refinement suggestions based on agent type and performance gap."""
        
        suggestions = {
            "EDA Agent": {
                "low_quality_score": ["increase_visualization_detail", "deeper_statistical_analysis"],
                "missing_insights": ["explore_feature_interactions", "advanced_outlier_detection"]
            },
            "Data Hygiene Agent": {
                "low_data_quality": ["advanced_imputation_methods", "iterative_outlier_treatment"],
                "high_missing_values": ["domain_specific_imputation", "missing_pattern_analysis"]
            },
            "Classification Agent": {
                "low_accuracy": ["ensemble_methods", "feature_selection", "hyperparameter_tuning"],
                "poor_precision": ["class_rebalancing", "cost_sensitive_learning"],
                "low_recall": ["threshold_optimization", "synthetic_minority_oversampling"]
            }
        }
        
        agent_suggestions = suggestions.get(agent_name, {})
        metric_suggestions = agent_suggestions.get(threshold.metric_name, ["iterative_improvement"])
        
        return {
            "strategies": metric_suggestions,
            "priority": "high" if actual_value < threshold.threshold_value * 0.8 else "normal",
            "estimated_improvement": abs(threshold.threshold_value - actual_value) * 0.5
        }
    
    def _broadcast_performance_update(self, agent_name: str, metrics: Dict[str, float]) -> None:
        """Broadcast performance updates to relevant agents."""
        
        # Determine which agents might be interested in this performance data
        interested_agents = []
        
        if "accuracy" in metrics or "precision" in metrics:
            interested_agents.extend(["Feature Engineering Agent", "Hyperparameter Tuning Agent"])
        
        if "data_quality_score" in metrics:
            interested_agents.extend(["EDA Agent", "Classification Agent", "Regression Agent"])
        
        # Remove duplicates and the updating agent itself
        interested_agents = list(set(interested_agents))
        if agent_name in interested_agents:
            interested_agents.remove(agent_name)
        
        for target_agent in interested_agents:
            if target_agent in self.registered_agents:
                message = AgentMessage(
                    from_agent="Communication Hub",
                    to_agent=target_agent,
                    message_type=MessageType.PERFORMANCE_UPDATE,
                    priority=Priority.NORMAL,
                    content={
                        "updating_agent": agent_name,
                        "performance_metrics": metrics,
                        "timestamp": time.time()
                    }
                )
                
                self.send_message(message)
    
    def get_collaboration_opportunities(self, agent_name: str) -> List[Dict[str, Any]]:
        """Identify potential collaboration opportunities for an agent."""
        opportunities = []
        
        if agent_name not in self.agent_status:
            return opportunities
        
        current_performance = self.agent_status[agent_name]["performance_score"]
        
        # Look for agents with complementary capabilities
        for other_agent, capabilities in self.agent_capabilities.items():
            if other_agent == agent_name:
                continue
                
            # Example collaboration patterns
            if agent_name == "Feature Engineering Agent" and other_agent == "Classification Agent":
                opportunities.append({
                    "type": "iterative_feature_optimization",
                    "partner": other_agent,
                    "description": "Collaborate on feature selection based on model performance",
                    "estimated_benefit": 0.15
                })
            
            elif agent_name == "Data Hygiene Agent" and other_agent == "EDA Agent":
                opportunities.append({
                    "type": "data_quality_feedback_loop",
                    "partner": other_agent,
                    "description": "Iterative data cleaning based on EDA insights",
                    "estimated_benefit": 0.20
                })
        
        return opportunities
    
    def get_communication_summary(self) -> Dict[str, Any]:
        """Get summary of communication activity and performance."""
        
        recent_messages = [msg for msg in self.message_history 
                          if time.time() - msg.timestamp < 3600]  # Last hour
        
        message_counts = defaultdict(int)
        for msg in recent_messages:
            message_counts[msg.message_type.value] += 1
        
        return {
            "total_messages": len(self.message_history),
            "recent_messages": len(recent_messages),
            "message_types": dict(message_counts),
            "active_conversations": len(self.active_conversations),
            "active_refinements": len(self.active_refinements),
            "registered_agents": len(self.registered_agents),
            "quality_thresholds": sum(len(thresholds) for thresholds in self.quality_thresholds.values())
        }